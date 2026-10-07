"""Predict raw m6Anet JSON/JSON.gz with a saved model; write transcript_id,transcript_position,score."""
import argparse
from clean_data import model_inputs, cleaned_sites
import csv
import os
import tempfile
from pathlib import Path
import joblib
import numpy as np
from features import KEY, SCHEMA, MIL_SCHEMA, pooled_features, bag_features


def predict_one():
    p = argparse.ArgumentParser(description=__doc__)
    inputs = p.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--clean-dir', help='Directory produced by clean_data.py')
    inputs.add_argument('--data', help='Raw m6Anet JSON/JSON.gz; parsing and cleaning are automatic')
    p.add_argument('--model', default=str(DEFAULT_MODEL))
    p.add_argument('--output', required=True)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--threads', type=int, default=2)
    args = p.parse_args()
    try:
        with model_inputs(args.clean_dir, args.data, require_labels=False) as (data_path, _, _audit):
            predict_prepared(args, p, data_path)
    except (ValueError, OSError) as error:
        p.error(str(error))


def predict_prepared(args, p, data_path):
    source_path = Path(args.data).resolve() if args.data else data_path
    args.data = str(data_path)
    if min(args.batch_size, args.threads) < 1:
        p.error('Batch size and threads must be positive')
    target = Path(args.output)
    if target.resolve() in (Path(args.model).resolve(), source_path) or target.resolve().is_relative_to(data_path.parent):
        raise ValueError('Output must differ from input data and model')
    artifact = joblib.load(args.model)  # Load only models from a trusted source.
    kind = artifact.get('kind', 'gb')  # Backwards compatible with the original baseline.
    if artifact['schema'] != (MIL_SCHEMA if kind == 'mil' else SCHEMA):
        raise ValueError('Incompatible feature schema')
    if kind == 'gb':
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=args.threads)
        model = artifact['model']
    else:
        import torch
        from neural import restore, probabilities
        torch.set_num_threads(args.threads)
        model = restore(artifact)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', newline='', dir=target.parent,
                                         delete=False) as handle:
            temporary = handle.name
            writer = csv.writer(handle)
            writer.writerow(KEY + ['score'])
            keys, vectors, masks, sequences = [], [], [], []

            def flush():
                if not keys:
                    return
                X = np.stack(vectors)
                if kind == 'gb':
                    scores = model.predict_proba(X)[:, 1]
                else:
                    arrays = (X, np.stack(masks), np.stack(sequences)) if kind == 'mil' else (X,)
                    scores = probabilities(model, kind, arrays, np.arange(len(keys)),
                                           artifact['mean'], artifact['scale'], args.batch_size)
                if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
                    raise ValueError('Invalid predicted probabilities')
                writer.writerows([*key, float(s)] for key, s in zip(keys, scores))
                keys.clear()
                vectors.clear()
                masks.clear()
                sequences.clear()

            for key, motif, reads in cleaned_sites(args.data):
                keys.append(key)
                if kind == 'mil':
                    bag, mask, sequence = bag_features(key, motif, reads,
                                                       artifact['max_reads'], artifact['seed'])
                    vectors.append(bag)
                    masks.append(mask)
                    sequences.append(sequence)
                else:
                    vectors.append(pooled_features(motif, reads).astype(np.float32))
                if len(keys) >= args.batch_size:
                    flush()
            flush()
        os.replace(temporary, target)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)



import json
import sys
import subprocess
import math
from clean_data import cleaned_input_paths, sha256

DEFAULT_MODEL = Path(__file__).resolve().parent / 'results/mlp/model.joblib'

def predict_all():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--all-datasets',action='store_true',required=True)
    parser.add_argument('--clean-root',default='data/cleaned')
    parser.add_argument('--model',default=str(DEFAULT_MODEL))
    parser.add_argument('--out-dir',help='Default: submission/ beside the selected model')
    parser.add_argument('--batch-size',type=int,default=256)
    parser.add_argument('--threads',type=int,default=2)
    args=parser.parse_args()
    model=Path(args.model).resolve()
    artifact=joblib.load(model)
    if artifact.get('kind') not in ('gb','mlp','mil'): parser.error('Expected a GB, MLP, or MIL model')
    fingerprint=sha256(model)
    target=Path(args.out_dir).resolve() if args.out_dir else model.parent/'submission'
    if target.exists(): parser.error(f'Output already exists: {target}; choose a new directory')
    target.parent.mkdir(parents=True,exist_ok=True)
    audit=dict(model_path=str(model),model_sha256=fingerprint,files={},
               note='One unchanged model generates every file. Full files include training/validation/test rows; evaluate only test rows from split_manifest.csv.')
    with tempfile.TemporaryDirectory(prefix='.submission-',dir=target.parent) as temporary:
        stage=Path(temporary)/'submission';stage.mkdir()
        for dataset in ('dataset0','dataset1','dataset2'):
            clean=Path(args.clean_root)/dataset
            data,_=cleaned_input_paths(clean,require_labels=False)
            expected=[key for key,_,_ in cleaned_sites(data)]
            path=stage/f'{dataset}_predictions.csv'
            subprocess.run([sys.executable,str(Path(__file__).resolve().parent/'predict.py'),
                            '--clean-dir',str(clean),'--model',str(model),'--output',str(path),
                            '--batch-size',str(args.batch_size),'--threads',str(args.threads)],check=True)
            with path.open(newline='') as handle:
                reader=csv.DictReader(handle)
                if reader.fieldnames!=['transcript_id','transcript_position','score']:
                    raise ValueError('Invalid submission columns')
                seen=[]
                for row in reader:
                    score=float(row['score'])
                    if not math.isfinite(score) or not 0<=score<=1: raise ValueError('Invalid prediction score')
                    seen.append((row['transcript_id'],row['transcript_position']))
            if seen!=expected or len(set(seen))!=len(seen): raise ValueError('Missing, duplicate, or reordered coordinates')
            if sha256(model)!=fingerprint: raise ValueError('Model changed during prediction')
            audit['files'][path.name]=dict(n_sites=len(seen),sha256=sha256(path),input_sha256=sha256(data))
            print(f'Validated {path.name}: {len(seen):,} sites',flush=True)
        if len(list(stage.iterdir()))!=3: raise ValueError('Expected exactly three submission files')
        stage.rename(target)
    (target.parent/(target.name+'_audit.json')).write_text(json.dumps(audit,indent=2)+'\n')
    print('Saved exactly three predictions from one shared model:',target,flush=True)

def main():
    if '--all-datasets' in sys.argv[1:]:
        predict_all()
    else:
        predict_one()


if __name__ == '__main__':
    main()
