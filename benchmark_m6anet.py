"""Benchmark official pretrained m6Anet on the existing test split."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import joblib
import numpy as np
import pandas as pd
import torch
from clean_data import cleaned_input_paths, cleaned_sites, sha256, verified_cleaned_file
from features import KEY
from metrics import score_metrics

SOURCE_COMMIT = '590ec277cb48d61774f0872395099e466022e810'
MODEL_TAGS = {'HCT116_RNA002': 'm6anet_pretrained_rna002'}
DATASETS = ('dataset0', 'dataset1', 'dataset2')


def load_official(source, support, model_name):
    source = Path(source).resolve()
    actual = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != SOURCE_COMMIT:
        raise ValueError(f'Expected official source commit {SOURCE_COMMIT}, got {actual}')
    if subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain'], text=True).strip():
        raise ValueError('Official m6Anet checkout must be unmodified')
    sys.path[:0] = [str(source), str(Path(support).resolve())]
    import toml
    from m6anet.model.model import MILModel
    from m6anet.utils.constants import PRETRAINED_CONFIGS, DEFAULT_MODEL_CONFIG, KMER_TO_INT
    from m6anet.utils.inference_utils import _calculate_site_proba
    weights, threshold, norm_path = PRETRAINED_CONFIGS[model_name]
    config = toml.load(DEFAULT_MODEL_CONFIG)
    model = MILModel(copy.deepcopy(config))
    model.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True))
    model.eval()
    norm = joblib.load(norm_path)
    return model, norm, KMER_TO_INT, _calculate_site_proba, dict(
        source_commit=actual, pretrained_model=model_name, model_config=config,
        weights_path=str(weights), weights_sha256=sha256(weights),
        norm_path=str(norm_path), norm_sha256=sha256(norm_path),
        config_path=str(DEFAULT_MODEL_CONFIG), config_sha256=sha256(DEFAULT_MODEL_CONFIG),
        read_probability_threshold=threshold,
        normalization_note=f'Official {model_name} configuration uses the bundled {Path(norm_path).name} normalization. No normalization is fitted on evaluation data.')


def inputs_for(motif, reads, norm, vocabulary):
    kmers = [motif[i:i+5] for i in range(3)]
    mean = np.concatenate([norm[k][0] for k in kmers])
    std = np.concatenate([norm[k][1] for k in kmers])
    if not np.isfinite(std).all() or (std <= 0).any():
        raise ValueError('Invalid pretrained normalization')
    return {'X': torch.tensor((reads-mean)/std, dtype=torch.float32),
            'kmer': torch.tensor(np.tile([vocabulary[k] for k in kmers], (len(reads), 1)), dtype=torch.long)}


def check_adapter(model, norm, vocabulary, norm_path, key, motif, reads):
    """Check the 9-column adapter against the official indexed 10-column loader."""
    from m6anet.utils.data_utils import NanopolishDS
    with tempfile.TemporaryDirectory(prefix='m6anet-adapter-') as directory:
        root = Path(directory)
        # Synthetic IDs are bookkeeping only; never measured or predicted read labels.
        records = np.column_stack([reads, np.arange(len(reads))]).tolist()
        payload = json.dumps({key[0]: {key[1]: {motif: records}}}, ensure_ascii=True).encode()
        (root/'data.json').write_bytes(payload)
        pd.DataFrame([dict(transcript_id=key[0], transcript_position=key[1], n_reads=len(reads), start=0, end=len(payload))]).to_csv(root/'data.info', index=False)
        ds = NanopolishDS(str(root), min_reads=20, norm_path=norm_path, mode='Inference')
        official = ds[0]
        own = inputs_for(motif, reads, norm, vocabulary)
        torch.testing.assert_close(own['X'], official[0], rtol=0, atol=0)
        torch.testing.assert_close(own['kmer'], official[1], rtol=0, atol=0)
        with torch.no_grad():
            a = model.pooling_filter.probability_layer(model.get_read_representation(own))
            b = model.pooling_filter.probability_layer(model.get_read_representation({'X': official[0], 'kmer': official[1]}))
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', default='.tools/m6anet-source')
    p.add_argument('--support-dir', default='.tools/m6anet-deps')
    p.add_argument('--clean-root', default='data/cleaned')
    p.add_argument('--results-root', default='results')
    p.add_argument('--pretrained-model', choices=tuple(MODEL_TAGS), default='HCT116_RNA002')
    p.add_argument('--out-dir', help='Default: <results-root>/m6anet_rna002')
    p.add_argument('--iterations', type=int, default=1000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--check-only', action='store_true')
    args = p.parse_args()
    model_tag = MODEL_TAGS[args.pretrained_model]
    if args.out_dir is None:
        args.out_dir = str(Path(args.results_root)/model_tag.replace('_pretrained', ''))
    if min(args.iterations, args.threads) < 1: p.error('Iterations and threads must be positive')
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(args.seed)
    model, norm, vocabulary, site_probability, provenance = load_official(args.source, args.support_dir, args.pretrained_model)
    root = Path(args.results_root)
    manifest_path = root/'mlp/split_manifest.csv'
    manifest = pd.read_csv(manifest_path, dtype={k:str for k in KEY})
    columns = ['dataset']+KEY+['label','group_id','split']
    if manifest.duplicated(['dataset']+KEY).any() or manifest.groupby('group_id').split.nunique().max()!=1:
        raise ValueError('Invalid split manifest')
    for kind in ('gb','mil'):
        other = pd.read_csv(root/kind/'split_manifest.csv', dtype={k:str for k in KEY})
        pd.testing.assert_frame_equal(manifest[columns], other[columns])
    test = manifest.loc[manifest.split.eq('test')].copy()
    reference_report = json.loads((root/'mlp/metrics.json').read_text())
    paths = {}; input_hashes = {}
    for dataset in DATASETS:
        clean = Path(args.clean_root)/dataset
        data_path, labels_path = cleaned_input_paths(clean, require_labels=True)
        hashes = {name:sha256(path) for name,path in [('data.json.gz',data_path),('labels.csv',labels_path)]}
        recorded = reference_report['preprocessing'][dataset]['output_sha256']
        if any(hashes[name]!=recorded[name] for name in hashes):
            raise ValueError('Inputs changed since model training')
        paths[dataset] = data_path; input_hashes[dataset] = hashes
        metadata = pd.read_csv(verified_cleaned_file(clean,'site_manifest.csv'), dtype={k:str for k in KEY})
        selected = test.loc[test.dataset.eq(dataset), KEY].merge(metadata,on=KEY,how='left',validate='one_to_one')
        if selected.n_reads.isna().any() or not selected.n_reads.ge(20).all() or not selected.sequence.str.slice(1,6).str.fullmatch('[AGT][AG]AC[ACT]').all():
            raise ValueError('Some test sites are ineligible; require an explicit common-site comparison')
    first_key, first_motif, first_reads = next(cleaned_sites(paths['dataset0']))
    check_adapter(model,norm,vocabulary,provenance['norm_path'],first_key,first_motif,first_reads)
    print('PASS: input adapter exactly matches official loader and read probabilities',flush=True)
    if args.check_only: return
    out = Path(args.out_dir)
    if out.exists(): raise FileExistsError(f'Choose a new output directory: {out}')
    out.mkdir(parents=True); (out/'evaluation').mkdir()
    shutil.copy2(provenance['weights_path'],out/'pretrained_model.pt')
    shutil.copy2(provenance['norm_path'],out/'normalization.joblib')
    shutil.copy2(provenance['config_path'],out/'model_config.toml')
    test.to_csv(out/'test_manifest.csv',index=False)
    predictions = {}; prediction_hashes = {}
    for dataset in DATASETS:
        expected = test.loc[test.dataset.eq(dataset)].copy()
        keys = set(expected[KEY].itertuples(index=False,name=None))
        rows=[]
        with torch.no_grad():
            for key,motif,reads in cleaned_sites(paths[dataset]):
                if key not in keys: continue
                inputs = inputs_for(motif,reads,norm,vocabulary)
                probabilities = model.pooling_filter.probability_layer(model.get_read_representation(inputs)).flatten().numpy()
                digest = hashlib.sha256(json.dumps([args.seed,dataset,*key]).encode()).digest()
                np.random.seed(int.from_bytes(digest[:4],'little'))
                score = float(site_probability((probabilities,args.iterations,20)))
                if not np.isfinite(score) or not 0<=score<=1: raise ValueError('Invalid site probability')
                rows.append((*key,score))
                if len(rows)%5000==0: print(f'{dataset}: predicted {len(rows):,}/{len(keys):,} test sites',flush=True)
        predicted = pd.DataFrame(rows,columns=KEY+['score'])
        if len(predicted)!=len(keys) or predicted.duplicated(KEY).any(): raise ValueError('Incomplete test coverage')
        frame=expected.merge(predicted,on=KEY,how='left',validate='one_to_one')
        if frame.score.isna().any(): raise ValueError('Missing test scores')
        path=out/'evaluation'/f'{dataset}_heldout_predictions.csv'
        frame[KEY+['score']].to_csv(path,index=False)
        prediction_hashes[dataset]=sha256(path);predictions[dataset]=frame
        print(f'{dataset}: verified all {len(frame):,} held-out sites',flush=True)
    metrics=[]
    model_hashes={}
    for kind in ('gb','mlp','mil',model_tag):
        frames=[]
        for dataset in DATASETS:
            expected = test.loc[test.dataset.eq(dataset)].copy()
            if kind==model_tag: frame=predictions[dataset]
            else:
                model_path=root/kind/'model.joblib'
                fingerprint=sha256(model_path)
                evaluation_audit=json.loads((root/kind/'evaluation_audit.json').read_text())
                if fingerprint!=evaluation_audit['model_sha256']: raise ValueError('Comparator model changed after validation')
                model_hashes[kind]=fingerprint
                scores=pd.read_csv(root/kind/'evaluation'/f'{dataset}_heldout_predictions.csv',dtype={k:str for k in KEY})
                if len(scores)!=len(expected) or scores.duplicated(KEY).any(): raise ValueError('Comparator test coverage mismatch')
                frame=expected.merge(scores,on=KEY,how='left',validate='one_to_one')
                if frame.score.isna().any() or not np.isfinite(frame.score).all() or not frame.score.between(0,1).all(): raise ValueError('Invalid comparator predictions')
                submission_audit=json.loads((root/kind/'submission_audit.json').read_text())
                full_path=root/kind/'submission'/f'{dataset}_predictions.csv'
                if sha256(full_path)!=submission_audit['files'][full_path.name]['sha256'] or submission_audit['model_sha256']!=fingerprint:
                    raise ValueError('Comparator submission provenance mismatch')
                full=pd.read_csv(full_path,dtype={k:str for k in KEY})
                matched=frame[KEY+['score']].merge(full,on=KEY,validate='one_to_one')
                if len(matched)!=len(frame): raise ValueError('Comparator submission coverage mismatch')
                np.testing.assert_allclose(matched.score_x,matched.score_y,rtol=1e-5,atol=1e-6)

            metrics.append(dict(model=kind,dataset=dataset,evaluation='same_heldout_sites',**score_metrics(frame)))
            frames.append(frame)
        pooled=pd.concat(frames,ignore_index=True)
        metrics.append(dict(model=kind,dataset='pooled',evaluation='same_heldout_sites',**score_metrics(pooled)))
        if kind==model_tag:
            pooled[KEY+['score']].to_csv(out/'evaluation/pooled_heldout_predictions.csv',index=False)
            pooled[['dataset']+KEY].to_csv(out/'evaluation/pooled_provenance.csv',index=False)
    measured=pd.DataFrame(metrics)
    measured.to_csv(out/'benchmark_metrics.csv',index=False)
    measured.loc[measured.model.eq(model_tag)].to_csv(out/'evaluation_metrics.csv',index=False)
    audit=dict(settings=vars(args),**provenance,torch_version=str(torch.__version__),numpy_version=np.__version__,
        input_hashes=input_hashes,manifest_sha256=sha256(manifest_path),comparator_model_hashes=model_hashes,
        prediction_hashes=prediction_hashes,test_sites=len(test),training_performed=False,
        score=f'Official site probability: mean noisy-OR over {args.iterations} bootstrap samples of 20 reads; not mod_ratio.',
        sampling='Official _calculate_site_proba called serially with a reproducible per-dataset/site seed; independent of processing order.',
        checks=['Adapter matches official NanopolishDS tensors and read probabilities exactly.','All four methods evaluated on identical test coordinates and labels.','Every test site meets default m6Anet coverage/motif requirements.'],
        limitations=[f'{args.pretrained_model} was selected by user request; sample chemistry has not been verified from sequencing metadata.','Overlap with pretrained m6Anet training data is unknown; these sites are only guaranteed held out from the project models.','Official model code executed with project PyTorch 2.6/Python 3.11, outside upstream historical package pins; adapter compatibility checked.','This benchmarks pretrained weights, not a matched-training or equally tuned architecture comparison.'])
    (out/'benchmark_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(measured[['model','dataset','n_sites','pr_auc','roc_auc']].to_string(index=False))


if __name__=='__main__':
    main()
