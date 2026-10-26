"""Train shared MLP, GB and GRU models from raw m6Anet JSON and matched labels.

Uses supplied immutable splits by default; can generate gene-grouped splits and
optionally tune model settings on development validation data. Inference and shared
cleaning live in predict.py.
"""
import argparse
from itertools import product
import json
from pathlib import Path
import platform
import tempfile
import warnings

import joblib
import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import auc, precision_recall_curve, average_precision_score, roc_auc_score
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits

from predict import (ROOT, KEY, IDENTITY, SCORE_COLUMNS, CLEANING_VERSION, SCHEMA,
                     clean_dataset, read_labels, sha256, flat_features, PooledMLP, SiteGRU,
                     fit_preprocessing, transform, loader, load_models, score_frame,
                     transform_scores)

MLP_CONFIG = dict(hidden=[256,128], dropout=0.5, lr=0.001, weight_decay=0.0001, batch_size=256, epochs=16)
GRU_CONFIG = dict(hidden_size=32, dropout=0., lr=0.0003, weight_decay=0., batch_size=256,
                  epochs=32, class_weight_alpha=0.0, bidirectional=False,
                  all_position_outputs=False, head_size=32)
GB_CONFIG = dict(max_leaf_nodes=63, max_iter=600, learning_rate=0.03, l2_regularization=1., min_samples_leaf=10)


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')


def metrics(y, s):
    if len(np.unique(y)) < 2:
        return dict(pr_auc=None, average_precision=None, roc_auc=None)
    p, r, _ = precision_recall_curve(y, s)
    return dict(pr_auc=float(auc(r,p)), average_precision=float(average_precision_score(y,s)),
                roc_auc=float(roc_auc_score(y,s)))


def read_manifest(path):
    frame = pd.read_csv(path, dtype={'dataset_id':str,'transcript_id':str,'group_id':str,'gene_id':str})
    required = IDENTITY + ['original_split','fold','label','group_id']
    if set(required)-set(frame):
        raise ValueError(f'Split manifest missing: {set(required)-set(frame)}')
    if frame.duplicated(IDENTITY).any() or frame[IDENTITY].isna().any().any():
        raise ValueError('Duplicate/missing split identities')
    positions = pd.to_numeric(frame.transcript_position, errors='raise')
    if not ((positions >= 0) & (positions == np.floor(positions)) & (positions < 2**63)).all():
        raise ValueError('Invalid manifest positions')
    frame['transcript_position'] = positions.astype('int64')
    if not frame.original_split.isin(['train','validation','test']).all() or not frame.label.isin([0,1]).all():
        raise ValueError('Invalid split names or labels')
    fit = frame.original_split.ne('test')
    if not frame.loc[fit,'fold'].isin([1,2,3,4,5]).all() or frame.loc[~fit,'fold'].notna().any():
        raise ValueError('Training/validation require folds 1–5; test must have no fold')
    if frame.group_id.isna().any():
        raise ValueError('Missing biological groups')
    if {'gene_id','group_source'}.issubset(frame.columns):
        gene_rows = frame.group_source.eq('gene')
        if frame.loc[gene_rows,'gene_id'].isna().any():
            raise ValueError('Manifest marks rows as gene-grouped but gene_id is missing')
        expected = 'gene:' + frame.loc[gene_rows,'gene_id'].astype(str)
        if not frame.loc[gene_rows,'group_id'].astype(str).reset_index(drop=True).equals(expected.reset_index(drop=True)):
            raise ValueError('Manifest gene group IDs do not match gene_id values')
    # Link genes and transcripts globally; mixture groups are already explicitly assigned.
    parent = {}
    def root(x):
        parent.setdefault(x,x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    nodes = []
    for row in frame.itertuples():
        group = row.group_id
        if ':mixture_position:' not in group:
            tx = 'transcript:'+row.transcript_id
            parent[root(tx)] = root(group)
        nodes.append(group)
    group = pd.Series([root(n) for n in nodes])
    partition = pd.Series(np.where(fit, frame.fold.fillna(0).astype(int).astype(str), 'test'))
    if pd.DataFrame({'group':group,'partition':partition}).groupby('group').partition.nunique().gt(1).any():
        raise ValueError('Biological group crosses folds or train/test boundary')
    return frame


def generated_gene_manifest(config_path, output, seed=42):
    """Create a deterministic held-out test set and five grouped development folds.

    Generated splits require gene_id on every labelled row. A supplied manifest can
    still use explicitly documented project-specific group IDs for mixture data.
    """
    spec = json.loads(Path(config_path).resolve().read_text())
    base = Path(config_path).resolve().parent
    tables = []
    names = set()
    for item in spec['datasets']:
        dataset_id = str(item['dataset_id'])
        if dataset_id in names:
            raise ValueError(f'Duplicate dataset_id: {dataset_id}')
        names.add(dataset_id)
        label_path = (base / item['labels']).resolve()
        labels = read_labels(label_path)
        if 'gene_id' not in labels or labels.gene_id.isna().any():
            missing = int(labels.gene_id.isna().sum()) if 'gene_id' in labels else len(labels)
            raise ValueError(f'{dataset_id}: generated gene-level splits require gene_id for every labelled site; missing {missing}')
        labels.insert(0, 'dataset_id', dataset_id)
        tables.append(labels)
    if not tables:
        raise ValueError('No datasets configured')
    table = pd.concat(tables, ignore_index=True)
    if table.duplicated(IDENTITY).any():
        raise ValueError('Duplicate site identities across configured label files')
    # Ensure a transcript never points to different genes in different datasets.
    if (table.groupby('transcript_id').gene_id.nunique() > 1).any():
        raise ValueError('Conflicting transcript_id to gene_id annotations across datasets')
    table['group_id'] = 'gene:' + table.gene_id.astype(str)
    strata = table.dataset_id.astype(str) + ':' + table.label.astype(str)
    groups = table.group_id

    # One fifth of gene groups is held out as test; the rest receive five OOF folds.
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    development, test = next(outer.split(np.zeros(len(table)), strata, groups))
    table['original_split'] = 'train'
    table.loc[test, 'original_split'] = 'test'
    table['fold'] = pd.Series(pd.NA, index=table.index, dtype='Int64')
    inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed + 1)
    dev = table.iloc[development]
    dev_strata = strata.iloc[development]
    for fold, (_, heldout) in enumerate(inner.split(np.zeros(len(dev)), dev_strata, dev.group_id), 1):
        table.loc[dev.index[heldout], 'fold'] = fold
    if table.loc[table.original_split.ne('test'), 'fold'].isna().any():
        raise ValueError('Generated development folds are incomplete')
    table['group_source'] = 'gene'
    manifest = table[IDENTITY + ['gene_id', 'group_id', 'group_source', 'original_split', 'label', 'fold']]
    # Reuse the same validator as supplied manifests before saving any assignments.
    validated_path = output / 'splits.csv.gz'
    manifest.to_csv(validated_path, index=False, compression={'method': 'gzip', 'mtime': 0})
    checked = read_manifest(validated_path)
    if checked.group_id.nunique() < 10:
        raise ValueError('Too few gene groups for a five-fold grouped split')
    return validated_path


def resolved_training_config(config_path, output, generate_splits, seed):
    """Resolve input paths and, when requested or absent, save a gene-grouped manifest."""
    config_path = Path(config_path).resolve()
    spec = json.loads(config_path.read_text())
    create = generate_splits or not spec.get('splits')
    if create:
        splits = generated_gene_manifest(config_path, output, seed)
        spec['splits'] = str(splits.resolve())
    else:
        spec['splits'] = str((config_path.parent / spec['splits']).resolve())
    for item in spec['datasets']:
        item['data'] = str((config_path.parent / item['data']).resolve())
        item['labels'] = str((config_path.parent / item['labels']).resolve())
    resolved = output / 'training_inputs_resolved.json'
    write_json(resolved, spec)
    return resolved


def _candidate_configs(method, seed):
    if method == 'gb':
        grid = list(product((7, 15, 31, 63), (100, 200, 400, 600),
                            (.01, .03, .1, .2), (0., 1., 10., 30.), (10, 20, 50, 100)))
        rng = np.random.default_rng(seed)
        choices = [(31, 300, .03, 10., 20)] + [grid[i] for i in rng.choice(len(grid), 31, replace=False)]
        return [dict(max_leaf_nodes=leaves, max_iter=iterations, learning_rate=rate,
                     l2_regularization=l2, min_samples_leaf=min_leaf, class_weight='none')
                for leaves, iterations, rate, l2, min_leaf in choices]
    if method == 'mlp':
        configs = [dict(MLP_CONFIG, class_weight='none')]
        hidden = [[32, 16], [64, 32], [128, 64], [256, 128]]
        dropout = [0., .1, .3, .5]
        rates = [.0001, .0003, .001, .003]
        decay = [0., .0001, .001]
        batches = [128, 256, 512]
        for i in range(11):
            configs.append(dict(hidden=hidden[i % 4], dropout=dropout[i % 4],
                                lr=rates[(i // 3) % 4], weight_decay=decay[i % 3],
                                batch_size=batches[i % 3], epochs=4,
                                class_weight='balanced' if i % 3 == 0 else 'none'))
        return configs
    if method == 'gru':
        # One-factor candidates come from the phea-model experiment rounds. The
        # saved final config remains the starting point; architecture depth and
        # class-weight exponent stay fixed, and each candidate gets early stopping.
        base = dict(GRU_CONFIG, epochs=50, class_weight_alpha=0.0)
        candidates = [base]
        for key, values in [
            ('class_weight_alpha', (.25, .5, .75, 1.0)),
            ('hidden_size', (64,)),
            ('dropout', (.2, .4)),
            ('lr', (.001, .003)),
            ('weight_decay', (.0001, .001)),
            ('batch_size', (128, 512)),
            ('all_position_outputs', (True,)),
            ('bidirectional', (True,)),
            ('head_size', (64,)),
        ]:
            for value in values:
                candidate = dict(base)
                candidate[key] = value
                candidates.append(candidate)
        return candidates
    raise ValueError(f'Unknown tuning method: {method}')


def _class_weight(y, config):
    if 'class_weight_alpha' in config:
        positives = int(np.asarray(y).sum())
        negatives = len(y) - positives
        if positives == 0:
            raise ValueError('Cannot calculate positive class weight without positive sites')
        return (negatives / positives) ** float(config['class_weight_alpha'])
    if config.get('class_weight') != 'balanced':
        return 1.0
    positives = int(np.asarray(y).sum())
    negatives = len(y) - positives
    if positives == 0:
        raise ValueError('Cannot balance a training set without positive sites')
    return negatives / positives


def _validation_score(name, model, validation, mean=None, scale=None, prep=None):
    model.eval()
    with torch.no_grad():
        if name == 'mlp':
            x = flat_features(validation)
            z = (x - mean) / scale
            scores = torch.sigmoid(model(torch.from_numpy(z))).numpy()
        elif name == 'gru':
            seq, aux = transform(validation, prep)
            scores = np.concatenate([torch.sigmoid(model(s, a)).numpy()
                                     for s, a in loader(seq, aux, 1024)])
        else:
            scores = model.predict_proba(flat_features(validation))[:, 1]
    return metrics(validation.label.to_numpy(dtype='int64'), scores)


def tune_models(frame, output, seed, methods):
    """Run opt-in model searches using development rows only; never tune on test."""
    development = frame.loc[frame.original_split.ne('test')].reset_index(drop=True)
    train = development.loc[development.original_split.eq('train')].reset_index(drop=True)
    validation = development.loc[development.original_split.eq('validation')].reset_index(drop=True)
    groups_overlap = (not train.empty and not validation.empty and
                      not set(train.group_id).isdisjoint(set(validation.group_id)))
    if train.empty or validation.empty or groups_overlap:
        train = development.loc[development.fold.ne(5)].reset_index(drop=True)
        validation = development.loc[development.fold.eq(5)].reset_index(drop=True)
    if train.empty or validation.empty or validation.label.nunique() < 2:
        raise ValueError('Tuning requires both classes in a gene-grouped development validation set')
    if train.label.nunique() < 2:
        raise ValueError('Tuning training rows require both classes')

    selected = {}
    records = []
    with tempfile.TemporaryDirectory(prefix='.tuning-', dir=output) as temp_name:
        temporary = Path(temp_name)
        for method in dict.fromkeys(methods):
            candidates = _candidate_configs(method, seed)
            if method == 'mlp':
                stage_rows = []
                for number, config in enumerate(candidates, 1):
                    config = dict(config, epochs=4)
                    root = temporary / f'{method}-screen-{number}'
                    fit_model_with_validation(method, train, root, seed, config, validation)
                    model = load_models(root, method)
                    score = _validation_score(method, model[method][0], validation,
                                              mean=model[method][1]['mean'],
                                              scale=model[method][1]['scale'])
                    row = dict(method=method, stage='screen', candidate=number,
                               config=config, metrics=score)
                    records.append(row)
                    stage_rows.append((score['pr_auc'], number - 1, config))
                finalists = [0] + [idx for _, idx, _ in sorted(stage_rows, reverse=True) if idx != 0][:2]
                best_rank, best_config = None, None
                for number in finalists:
                    config = dict(candidates[number], epochs=30)
                    root = temporary / f'{method}-final-{number + 1}'
                    best_epoch = fit_model_with_validation(method, train, root, seed, config,
                                                           validation, patience=6, min_delta=.001)
                    model = load_models(root, method)
                    score = _validation_score(method, model[method][0], validation,
                                              mean=model[method][1]['mean'],
                                              scale=model[method][1]['scale'])
                    row = dict(method=method, stage='finalist', candidate=number + 1,
                               config=config, best_epoch=best_epoch, metrics=score)
                    records.append(row)
                    rank = (score['pr_auc'], score['roc_auc'] or -np.inf)
                    if best_rank is None or rank > best_rank:
                        best_rank = rank
                        best_config = dict(config, epochs=int(best_epoch))
                selected[method] = best_config
            elif method == 'gru':
                # The phea-model experiments compared GRU settings across the
                # five gene-grouped development folds. Keep test groups excluded.
                fold_scores = []
                for number, config in enumerate(candidates, 1):
                    per_fold = []
                    for fold in range(1, 6):
                        fold_train = development.loc[development.fold.ne(fold)].reset_index(drop=True)
                        fold_validation = development.loc[development.fold.eq(fold)].reset_index(drop=True)
                        if fold_train.empty or fold_validation.empty:
                            raise ValueError(f'GRU tuning fold {fold} is empty')
                        if fold_train.label.nunique() < 2 or fold_validation.label.nunique() < 2:
                            raise ValueError(f'GRU tuning fold {fold} requires both classes in train and validation')
                        root = temporary / f'{method}-{number}-fold-{fold}'
                        best_epoch = fit_model_with_validation(
                            method, fold_train, root, seed + fold, config, fold_validation,
                            patience=5, min_delta=.001)
                        model = load_models(root, method)
                        score = _validation_score(method, model[method][0], fold_validation,
                                                  prep=model[method][1])
                        per_fold.append(dict(fold=fold, best_epoch=int(best_epoch), metrics=score))
                    pr_values = [row['metrics']['pr_auc'] for row in per_fold]
                    roc_values = [row['metrics']['roc_auc'] for row in per_fold]
                    summary = dict(
                        pr_auc=float(np.mean(pr_values)),
                        average_precision=float(np.mean([row['metrics']['average_precision'] for row in per_fold])),
                        roc_auc=float(np.mean(roc_values)),
                    )
                    records.append(dict(method=method, stage='grouped_5_fold_search',
                                        candidate=number, config=config,
                                        folds=per_fold, metrics=summary))
                    fold_scores.append((summary['pr_auc'], summary['roc_auc'], number, config,
                                        [row['best_epoch'] for row in per_fold]))
                _, _, number, best_config, epochs = max(fold_scores, key=lambda row: (row[0], row[1]))
                best_config = dict(best_config, epochs=max(1, int(np.median(epochs))))
                selected[method] = best_config
            else:
                best_rank, best_config = None, None
                for number, config in enumerate(candidates, 1):
                    root = temporary / f'{method}-{number}'
                    best_epoch = fit_model_with_validation(method, train, root, seed, config,
                                                           validation)
                    model = load_models(root, method)
                    if method == 'gb':
                        score = _validation_score(method, model[method], validation)
                    row = dict(method=method, stage='search',
                               candidate=number, config=config, best_epoch=best_epoch, metrics=score)
                    records.append(row)
                    rank = (score['pr_auc'], score['roc_auc'] or -np.inf)
                    if best_rank is None or rank > best_rank:
                        best_rank = rank
                        best_config = dict(config)
                selected[method] = best_config
    write_json(output / 'tuning_history.json', records)
    write_json(output / 'selected_configurations.json', {
        'selected': selected,
        'selection_metric': 'validation trapezoidal PR-AUC, then ROC-AUC',
        'validation_source': 'original validation split when present; otherwise development fold 5',
        'test_used_for_tuning': False,
        'gru_scope': 'one-factor search over phea-model settings using five gene-grouped development folds; selected epochs are the median fold best epoch; held-out test is excluded',
        'cv_note': 'When fold 5 is the tuning validation, fold-5 ensemble validation is not an independent estimate; the held-out test remains untouched.',
    })
    return selected


def prepare(config_path, output):
    config_path = Path(config_path).resolve()
    spec = json.loads(config_path.read_text())
    resolve = lambda p: (config_path.parent/Path(p)).resolve()
    split_path = resolve(spec['splits'])
    before = sha256(split_path)
    manifest = read_manifest(split_path)
    entries = spec['datasets']
    names = [d['dataset_id'] for d in entries]
    if len(set(names)) != len(names) or set(names) != set(manifest.dataset_id):
        raise ValueError('Input datasets must match split manifest exactly')
    tables, audits = [], {}
    for item in entries:
        name = item['dataset_id']
        data, labels = resolve(item['data']), resolve(item['labels'])
        hashes = {'data_sha256':sha256(data), 'labels_sha256':sha256(labels)}
        print(f'Cleaning {name}: {data}', flush=True)
        frame, audit = clean_dataset(data, labels, name)
        reference = manifest.loc[manifest.dataset_id.eq(name)].copy()
        # Ignore unassigned sites explicitly; never silently lose an assigned site.
        audit['retained_unassigned_sites'] = len(frame)-len(frame.merge(reference[IDENTITY],on=IDENTITY))
        joined = reference.merge(frame, on=IDENTITY, how='left', suffixes=('_manifest',''), validate='one_to_one', indicator=True, sort=False)
        if not joined['_merge'].eq('both').all():
            raise ValueError(f'{name}: cleaning lost assigned sites; original split will NOT be changed')
        if joined.label.isna().any() or not joined.label.eq(joined.label_manifest).all():
            raise ValueError(f'{name}: raw labels missing or disagree with immutable split labels')
        if 'gene_id_manifest' in joined:
            conflict = joined.gene_id.notna() & joined.gene_id_manifest.notna() & joined.gene_id.ne(joined.gene_id_manifest)
            if conflict.any():
                raise ValueError(f'{name}: gene annotation conflicts with saved split')
            joined['gene_id'] = joined.gene_id.where(joined.gene_id.notna(), joined.gene_id_manifest)
            joined = joined.drop(columns='gene_id_manifest')
        joined = joined.drop(columns=['label_manifest','_merge'])
        joined['split'] = np.where(joined.original_split.eq('test'), 'test', 'train_validation')
        joined['label'] = joined.label.astype(int)
        if sha256(data) != hashes['data_sha256'] or sha256(labels) != hashes['labels_sha256']:
            raise ValueError('Raw inputs changed during cleaning')
        tables.append(joined)
        audits[name] = dict(data=str(data),labels=str(labels),**hashes,counts=audit)
    if sha256(split_path) != before:
        raise ValueError('Split manifest changed during preparation')
    combined = pd.concat(tables,ignore_index=True)
    # Preserve the saved manifest's order as well as membership and labels.
    order = manifest[IDENTITY].merge(combined,on=IDENTITY,how='left',validate='one_to_one')
    output.mkdir(parents=True,exist_ok=False)
    order.to_csv(output/'sites.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    report = dict(cleaning_version=CLEANING_VERSION,split_sha256=before,inputs=audits,
                  sites=len(order),train_validation_sites=int(order.original_split.ne('test').sum()),
                  test_sites=int(order.original_split.eq('test').sum()),
                  split_and_folds_unchanged=True,sites_sha256=sha256(output/'sites.csv.gz'))
    write_json(output/'audit.json',report)
    return order, report


def fit_model(name, frame, output, seed, config):
    return fit_model_with_validation(name, frame, output, seed, config)


def fit_model_with_validation(name, frame, output, seed, config, validation_frame=None,
                              patience=0, min_delta=0.0):
    y = frame.label.to_numpy(dtype='int64')
    if len(np.unique(y)) != 2:
        raise ValueError('Training requires both classes')
    positive_weight = _class_weight(y, config)
    directory = output/name
    directory.mkdir(parents=True,exist_ok=False)
    history = []
    if name == 'gb':
        parameters = {key: value for key, value in config.items()
                      if key in {'max_leaf_nodes','max_iter','learning_rate','l2_regularization','min_samples_leaf'}}
        model = HistGradientBoostingClassifier(**parameters,early_stopping=False,random_state=seed)
        sample_weight = np.where(y == 1, positive_weight, 1.0) if positive_weight != 1.0 else None
        model.fit(flat_features(frame),y,sample_weight=sample_weight)
        joblib.dump(dict(kind='gb',model=model,best_params=parameters,seed=seed),directory/'model.joblib')
        best_epoch = None
    else:
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)
        if name == 'mlp':
            x = flat_features(frame)
            mean, scale = x.astype('float64').mean(0),x.astype('float64').std(0)
            scale[scale < 1e-8] = 1.
            mean, scale = mean.astype('float32'),scale.astype('float32')
            x = (x-mean)/scale
            model = PooledMLP(config['hidden'],config['dropout'])
            rng = np.random.default_rng(seed)
        else:
            prep = fit_preprocessing(frame)
            seq, aux = transform(frame,prep)
            model = SiteGRU(config['hidden_size'],config['dropout'],
                            bidirectional=config.get('bidirectional',False),
                            all_position_outputs=config.get('all_position_outputs',False),
                            head_size=config.get('head_size',32))
            batches = loader(seq,aux,config['batch_size'],labels=y,shuffle=True,seed=seed)
        optimizer = torch.optim.AdamW(model.parameters(),lr=config['lr'],weight_decay=config['weight_decay'])
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(positive_weight,dtype=torch.float32))
        best_score, best_state, stale_epochs = -np.inf, None, 0
        best_epoch = int(config['epochs'])
        for epoch in range(1,config['epochs']+1):
            model.train()
            total = 0.
            if name == 'mlp':
                indices = rng.permutation(len(y))
                batches = ((torch.from_numpy(x[b]),torch.from_numpy(y[b].astype('float32')))
                           for b in (indices[i:i+config['batch_size']] for i in range(0,len(y),config['batch_size'])))
            for batch in batches:
                *features, targets = batch
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(*features),targets)
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite training loss')
                loss.backward()
                if name == 'gru':
                    torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                optimizer.step()
                total += loss.item()*len(targets)
            history.append(dict(epoch=epoch,train_loss=total/len(y)))
            print(f'{name} epoch {epoch}/{config["epochs"]}: loss={total/len(y):.6f}',flush=True)
            if validation_frame is not None:
                score = _validation_score(name, model, validation_frame,
                                          mean=mean if name == 'mlp' else None,
                                          scale=scale if name == 'mlp' else None,
                                          prep=prep if name == 'gru' else None)
                value = score['pr_auc']
                if value is not None and value > best_score + min_delta:
                    best_score, best_epoch = value, epoch
                    best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
                    stale_epochs = 0
                else:
                    stale_epochs += 1
                if patience and stale_epochs >= patience:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        if name == 'mlp':
            joblib.dump(dict(kind='mlp',network_config={k:config[k] for k in ['hidden','dropout']},
                        state={k:v.detach().numpy() for k,v in model.state_dict().items()},mean=mean,scale=scale,
                        seed=seed,best_epoch=best_epoch),directory/'model.joblib')
        else:
            torch.save(dict(config=config,state_dict=model.state_dict(),schema=SCHEMA,epoch=best_epoch),directory/'model.pt')
            write_json(directory/'preprocessing.json',prep)
        write_json(directory/'training_history.json',history)
    write_json(directory/'run_metadata.json',dict(config=config,seed=seed,training_sites=len(y),positive_sites=int(y.sum()),
               preprocessing_fit_on_training_rows_only=True,best_epoch=best_epoch,
               positive_class_weight=positive_weight))
    return best_epoch


def ensemble_fit(oof, out):
    """Fit the logistic ensemble on folds 1–4 and validate on held-out fold 5."""
    target = out/'models/ensemble/oof_holdout'
    target.mkdir(parents=True)
    tr = oof.loc[oof.fold.ne(5)]
    x, y = tr[SCORE_COLUMNS].to_numpy(),tr.label.to_numpy()
    folds = tr.fold.to_numpy()
    parts = [(np.flatnonzero(folds!=k),np.flatnonzero(folds==k)) for k in range(1,5)]
    def objective(rows):
        return (np.mean([r['pr_auc'] for r in rows]),np.mean([r['roc_auc'] for r in rows]))
    rank = lambda row:(row['mean_pr_auc'],row['mean_roc_auc'])
    candidates = []
    def build(c):
        return make_pipeline(StandardScaler(),LogisticRegression(C=c['C'],class_weight=c['class_weight'],
                         max_iter=2000,tol=1e-7,random_state=42))
    for representation in ['raw','logit','raw_logit']:
        z = transform_scores(x,representation)
        for c in np.logspace(-4,2,7):
            for weight in [None,'balanced']:
                config = dict(representation=representation,C=float(c),class_weight=weight)
                values = []
                with warnings.catch_warnings():
                    warnings.simplefilter('error',ConvergenceWarning)
                    for a,b in parts:
                        model = build(config).fit(z[a],y[a])
                        values.append(metrics(y[b],model.predict_proba(z[b])[:,1]))
                score = objective(values)
                candidates.append(dict(config=config,mean_pr_auc=float(score[0]),mean_roc_auc=float(score[1])))
    chosen = max(candidates,key=rank)
    config = chosen['config']
    model = build(config).fit(transform_scores(x,config['representation']),y)
    joblib.dump(dict(pipeline=model,config=config,feature_order=SCORE_COLUMNS,training_folds=[1,2,3,4]),target/'logistic_regression.joblib')
    write_json(out/'logistic_search.json',candidates)
    write_json(target/'selection.json',dict(logistic=chosen,training_folds=[1,2,3,4],validation_fold=5,
               caveat='Existing OOF meta-validation is not nested; base configurations were historically selected.'))
    val = oof.loc[oof.fold.eq(5)].copy()
    base = val[SCORE_COLUMNS].to_numpy()
    val['ensemble_score'] = model.predict_proba(transform_scores(base,config['representation']))[:,1]
    val.to_csv(out/'ensemble_validation.csv.gz',index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'training_inputs.json',help='JSON listing raw data and labels, plus an optional split manifest')
    parser.add_argument('--out-dir',type=Path,required=True,help='New output directory; existing results are never overwritten')
    parser.add_argument('--prepare-only',action='store_true',help='Only produce joined cleaned sites and verify all assignments')
    parser.add_argument('--methods',nargs='+',choices=['mlp','gb','gru'],default=['mlp','gb','gru'])
    parser.add_argument('--skip-ensemble',action='store_true')
    parser.add_argument('--generate-splits',action='store_true',help='Generate a new deterministic gene-grouped test split and five development folds, even if the config supplies splits')
    parser.add_argument('--tune',action='store_true',help='Opt in to GB/MLP searches and a GRU one-factor search; uses development validation only')
    parser.add_argument('--threads',type=int,default=1)
    parser.add_argument('--seed',type=int,default=42)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('--threads must be positive')
    if args.out_dir.exists():
        raise FileExistsError(args.out_dir)
    args.out_dir.mkdir(parents=True)
    config_path = resolved_training_config(args.config,args.out_dir,args.generate_splits,args.seed)
    frame,audit = prepare(config_path,args.out_dir/'cleaned')
    if args.prepare_only:
        print(f'Prepared {len(frame):,} sites with unchanged split/folds')
        return
    torch.set_num_threads(args.threads)
    configurations = tune_models(frame,args.out_dir,args.seed,args.methods) if args.tune else {
        'mlp':dict(MLP_CONFIG),'gb':dict(GB_CONFIG),'gru':dict(GRU_CONFIG)}
    fit = frame.loc[frame.original_split.ne('test')].reset_index(drop=True)
    test = frame.loc[frame.original_split.eq('test')].reset_index(drop=True)
    oof = fit[IDENTITY+['label','fold','group_id']].copy()
    scores_test = test[IDENTITY+['label']].copy()
    rows = []
    with threadpool_limits(limits=args.threads):
        for method in dict.fromkeys(args.methods):
            score = np.full(len(fit),np.nan)
            for fold in range(1,6):
                train_mask = fit.fold.ne(fold)
                heldout = fit.fold.eq(fold)
                checkpoint = args.out_dir/f'fold_models/fold_{fold}'
                seed = args.seed if method=='gb' else args.seed+fold
                fit_model(method,fit.loc[train_mask],checkpoint,seed,configurations[method])
                score[heldout] = score_frame(fit.loc[heldout],load_models(checkpoint,method),method)
            if not np.isfinite(score).all():
                raise ValueError('Incomplete OOF predictions')
            oof[method+'_score'] = score
            fit_model(method,fit,args.out_dir/'models',args.seed,configurations[method])
        oof.to_csv(args.out_dir/'oof_predictions.csv.gz',index=False)
        oof_metrics = []
        for method in dict.fromkeys(args.methods):
            for name, part in [('pooled',oof)] + list(oof.groupby('dataset_id')):
                oof_metrics.append(dict(dataset_id=name,method=method,n_sites=len(part),
                                        **metrics(part.label.to_numpy(),part[method+'_score'].to_numpy())))
        pd.DataFrame(oof_metrics).to_csv(args.out_dir/'oof_metrics.csv',index=False)
        methods = list(dict.fromkeys(args.methods))
        if set(methods)=={'mlp','gb','gru'} and not args.skip_ensemble:
            ensemble_fit(oof,args.out_dir)
            methods += ['ensemble']
        # Test scoring only after every model/ensemble selection and fit is frozen.
        for method in methods:
            values = score_frame(test,load_models(args.out_dir/'models',method),method)
            scores_test[method+'_score'] = values
            for name, indices in [('pooled',np.arange(len(test)))]+[(n,np.flatnonzero(test.dataset_id.eq(n))) for n in test.dataset_id.unique()]:
                rows.append(dict(dataset_id=name,method=method,n_sites=len(indices),**metrics(test.label.to_numpy()[indices],values[indices])))
                if name!='pooled':
                    directory = args.out_dir/'predictions'/method
                    directory.mkdir(parents=True,exist_ok=True)
                    test.iloc[indices][KEY].assign(score=values[indices]).to_csv(directory/f'{name}.csv',index=False)
    scores_test.to_csv(args.out_dir/'test_predictions.csv.gz',index=False)
    pd.DataFrame(rows).to_csv(args.out_dir/'evaluation_metrics.csv',index=False)
    resolved_spec = json.loads(Path(config_path).read_text())
    write_json(args.out_dir/'run_manifest.json',dict(status='COMPLETE',configurations=configurations,cleaning=audit,
               tuning_enabled=args.tune,split_manifest=resolved_spec['splits'],
               seed=args.seed,python=platform.python_version(),numpy=np.__version__,sklearn=sklearn.__version__,torch=torch.__version__,
               train_code_sha256=sha256(__file__),predict_code_sha256=sha256(ROOT/'predict.py')))


if __name__=='__main__':
    main()
