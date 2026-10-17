"""Train shared MLP, GB and GRU models from raw m6Anet JSON and matched labels.

Uses an immutable split manifest; never generates or changes split/fold assignments.
Shared parsing and model-specific preprocessing live in predict.py (the only module).
"""
import argparse
import json
from pathlib import Path
import platform
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
from threadpoolctl import threadpool_limits

from predict import (ROOT, KEY, IDENTITY, SCORE_COLUMNS, CLEANING_VERSION, SCHEMA,
                     clean_dataset, sha256, flat_features, PooledMLP, SiteGRU,
                     fit_preprocessing, transform, loader, load_models, score_frame,
                     transform_scores)

MLP_CONFIG = dict(hidden=[256,128], dropout=0.5, lr=0.001, weight_decay=0.0001, batch_size=256, epochs=16)
GRU_CONFIG = dict(hidden_size=32, dropout=0., lr=0.0003, weight_decay=0., batch_size=256, epochs=32)
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
    y = frame.label.to_numpy(dtype='int64')
    if len(np.unique(y)) != 2:
        raise ValueError('Training requires both classes')
    directory = output/name
    directory.mkdir(parents=True,exist_ok=False)
    history = []
    if name == 'gb':
        model = HistGradientBoostingClassifier(**config,early_stopping=False,random_state=seed)
        model.fit(flat_features(frame),y)
        joblib.dump(dict(kind='gb',model=model,best_params=config,seed=seed),directory/'model.joblib')
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
            model = SiteGRU(config['hidden_size'],config['dropout'])
            batches = loader(seq,aux,config['batch_size'],labels=y,shuffle=True,seed=seed)
        optimizer = torch.optim.AdamW(model.parameters(),lr=config['lr'],weight_decay=config['weight_decay'])
        criterion = torch.nn.BCEWithLogitsLoss()
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
        if name == 'mlp':
            joblib.dump(dict(kind='mlp',network_config={k:config[k] for k in ['hidden','dropout']},
                        state={k:v.detach().numpy() for k,v in model.state_dict().items()},mean=mean,scale=scale,
                        seed=seed,best_epoch=config['epochs']),directory/'model.joblib')
        else:
            torch.save(dict(config=config,state_dict=model.state_dict(),schema=SCHEMA,epoch=config['epochs']),directory/'model.pt')
            write_json(directory/'preprocessing.json',prep)
        write_json(directory/'training_history.json',history)
    write_json(directory/'run_metadata.json',dict(config=config,seed=seed,training_sites=len(y),positive_sites=int(y.sum()),
               preprocessing_fit_on_training_rows_only=True))


def ensemble_fit(oof, out):
    """Preserve original meta-training folds 1–4 and held-out fold 5 protocol."""
    target = out/'models/ensemble/oof_holdout'
    target.mkdir(parents=True)
    tr = oof.loc[oof.fold.ne(5)]
    x, y = tr[SCORE_COLUMNS].to_numpy(),tr.label.to_numpy()
    folds = tr.fold.to_numpy()
    parts = [(np.flatnonzero(folds!=k),np.flatnonzero(folds==k)) for k in range(1,5)]
    def objective(rows):
        return (np.mean([r['pr_auc'] for r in rows]),np.mean([r['roc_auc'] for r in rows]))
    weighted, seen = [],set()
    def assess(w):
        key = tuple(np.round(w,12))
        if key in seen:
            return
        seen.add(key)
        s = x@np.asarray(w)
        score = objective([metrics(y[b],s[b]) for _,b in parts])
        weighted.append(dict(weights=list(w),mean_pr_auc=float(score[0]),mean_roc_auc=float(score[1])))
    rank = lambda row:(row['mean_pr_auc'],row['mean_roc_auc'])
    assess([1/3]*3)
    for g in range(21):
        for m in range(21-g):
            assess([g/20,m/20,(20-g-m)/20])
    center = max(weighted,key=rank)['weights']
    for g in range(max(0,round(center[0]*100)-5),min(100,round(center[0]*100)+5)+1):
        for m in range(max(0,round(center[1]*100)-5),min(100,round(center[1]*100)+5)+1):
            b = 100-g-m
            if b>=0 and abs(b/100-center[2])<=.050001:
                assess([g/100,m/100,b/100])
    best = max(weighted,key=rank)
    write_json(target/'weighted_average.json',dict(feature_order=SCORE_COLUMNS,training_folds=[1,2,3,4],**best))
    pd.DataFrame(weighted).to_csv(out/'weighted_search.csv',index=False)
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
    write_json(target/'selection.json',dict(weighted=best,logistic=chosen,training_folds=[1,2,3,4],validation_fold=5,
               caveat='Existing OOF meta-validation is not nested; base configurations were historically selected.'))
    val = oof.loc[oof.fold.eq(5)].copy()
    base = val[SCORE_COLUMNS].to_numpy()
    val['weighted_score'] = base@np.asarray(best['weights'])
    val['ensemble_score'] = model.predict_proba(transform_scores(base,config['representation']))[:,1]
    val.to_csv(out/'ensemble_validation.csv.gz',index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'training_inputs.json',help='JSON listing raw data, labels and immutable split CSV')
    parser.add_argument('--out-dir',type=Path,required=True,help='New output directory; existing results are never overwritten')
    parser.add_argument('--prepare-only',action='store_true',help='Only produce joined cleaned sites and verify all assignments')
    parser.add_argument('--methods',nargs='+',choices=['mlp','gb','gru'],default=['mlp','gb','gru'])
    parser.add_argument('--skip-ensemble',action='store_true')
    parser.add_argument('--threads',type=int,default=1)
    parser.add_argument('--seed',type=int,default=42)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('--threads must be positive')
    if args.out_dir.exists():
        raise FileExistsError(args.out_dir)
    args.out_dir.mkdir(parents=True)
    frame,audit = prepare(args.config,args.out_dir/'cleaned')
    if args.prepare_only:
        print(f'Prepared {len(frame):,} sites with unchanged split/folds')
        return
    torch.set_num_threads(args.threads)
    configurations = {'mlp':MLP_CONFIG,'gb':GB_CONFIG,'gru':GRU_CONFIG}
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
            methods += ['weighted','ensemble']
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
    write_json(args.out_dir/'run_manifest.json',dict(status='COMPLETE',configurations=configurations,cleaning=audit,
               seed=args.seed,python=platform.python_version(),numpy=np.__version__,sklearn=sklearn.__version__,torch=torch.__version__,
               train_code_sha256=sha256(__file__),predict_code_sha256=sha256(ROOT/'predict.py')))


if __name__=='__main__':
    main()
