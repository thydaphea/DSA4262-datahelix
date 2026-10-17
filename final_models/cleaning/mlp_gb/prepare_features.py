#!/usr/bin/env python3
"""Reconstruct GB/MLP feature tables from equivalent standardized GRU summaries.

This submission helper preserves supplied labels/folds; it does not replace the
historical raw cleaner or claim to be the original preparation script.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from features import SCHEMA

ROOT=Path(__file__).resolve().parents[2]
SIGNALS=[f'{loc}_{measure}' for loc in ['minus1','center','plus1'] for measure in ['dwell','signal_sd','signal_mean']]

def convert(frame, training):
    columns=['transcript_id','transcript_position','label','n_reads','original_split'] + (['fold'] if training else [])
    result=frame[columns].copy()
    for src,dst in [('mean','mean'),('std','std'),('q25','q25'),('median','q50'),('q75','q75')]:
        for i,signal in enumerate(SIGNALS): result[f'signal_{dst}_{i}']=frame[f'{signal}_{src}'].to_numpy()
    result['log_read_count']=np.log1p(frame.n_reads_valid)
    for pos in range(7):
        for base in 'ACGT': result[f'seq_pos{pos}_{base}']=frame.sequence.str[pos].eq(base).astype(float)
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=ROOT/'dataset/cleaned/gru')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    metadata={'feature_schema':SCHEMA,'purpose':'Reconstruction from equivalent GRU summary features; original labels and folds retained','datasets':{}}
    for ds in ['dataset0','dataset1','dataset2']:
        (args.output/ds).mkdir()
        files={}
        for src,dst in [('train','train_validation'),('test','test')]:
            path=args.source/ds/f'{ds}_clean_{src}.csv.gz'
            frame=pd.read_csv(path)
            result=convert(frame,src=='train')
            target=args.output/ds/f'{dst}.csv.gz'
            result.to_csv(target,index=False)
            files[dst]={'path':str(target.relative_to(args.output)),'rows':len(result),'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        metadata['datasets'][ds]={'files':files}
    metadata['feature_columns']=[c for c in result if c.startswith(('signal_','seq_pos')) or c=='log_read_count']
    metadata['feature_count']=len(metadata['feature_columns'])
    (args.output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(args.output)
if __name__=='__main__': main()
