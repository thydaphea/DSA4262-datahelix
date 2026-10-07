"""Evaluate ONE shared model on unseen test sites in each dataset and their pooled test set."""
import argparse
import subprocess
import sys
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from clean_data import sha256, verified_cleaned_file
from features import KEY
from metrics import score_metrics


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-root',default='results/mlp')
    parser.add_argument('--clean-root',default='data/cleaned')
    parser.add_argument('--compare',action='store_true',help='Verify GB/MLP/MIL under results-root and build one comparison table')
    args=parser.parse_args()
    root=Path(args.results_root)
    if args.compare:
        tables=[];reference=None
        for kind in ('gb','mlp','mil'):
            folder=root/kind
            subprocess.run([sys.executable,str(Path(__file__).resolve()),'--results-root',str(folder),
                            '--clean-root',args.clean_root],check=True)
            manifest=pd.read_csv(folder/'split_manifest.csv',dtype={k:str for k in KEY})
            columns=['dataset']+KEY+['label','group_id','split']
            if reference is None: reference=manifest[columns]
            else: pd.testing.assert_frame_equal(reference,manifest[columns])
            table=pd.read_csv(folder/'evaluation_metrics.csv')
            if not table.model.eq('shared_'+kind).all(): raise ValueError('Wrong model in comparison')
            tables.append(table)
        combined=pd.concat(tables,ignore_index=True)
        combined.to_csv(root/'evaluation_metrics.csv',index=False)
        print('Shared models on identical held-out sites:')
        print(combined[['dataset','model','n_sites','pr_auc','roc_auc']].to_string(index=False))
        return
    report=json.loads((root/'metrics.json').read_text())
    submission=json.loads((root/'submission_audit.json').read_text())
    fingerprint=sha256(root/'model.joblib')
    if fingerprint!=submission['model_sha256']:
        raise ValueError('Predictions do not match the current model hash')
    artifact=joblib.load(root/'model.joblib')
    if artifact['report']!=report: raise ValueError('Saved model and training report disagree')
    manifest=pd.read_csv(root/'split_manifest.csv',dtype={k:str for k in KEY})
    if manifest.duplicated(['dataset']+KEY).any() or set(manifest.split)!={'train','validation','test'}:
        raise ValueError('Invalid development partitions')
    if manifest.groupby('group_id').split.nunique().max()!=1:
        raise ValueError('Related groups cross data partitions')
    human=manifest.loc[manifest.dataset!='dataset2']
    if human.groupby('transcript_id').split.nunique().max()!=1:
        raise ValueError('Shared transcripts cross data partitions')
    if manifest.loc[manifest.dataset=='dataset2'].groupby('transcript_position').split.nunique().max()!=1:
        raise ValueError('Dataset2 mixture positions cross partitions')
    for split,key in [('train','training_sites'),('validation','validation_sites'),('test','test_sites')]:
        if report.get(key)!=int(manifest.split.eq(split).sum()):
            raise ValueError('This evaluation requires a model fitted on training portions only; all-data refits are not eligible')
    if report['training_sites']>=len(manifest): raise ValueError('No unseen sites remain')
    history=json.loads((root/'validation_history.json').read_text())
    winner=max(history,key=lambda row:row['validation_pr_auc'])
    selection_key='candidate' if artifact['kind']=='gb' else 'epoch'
    artifact_key='best_candidate' if artifact['kind']=='gb' else 'best_epoch'
    if winner[selection_key]!=artifact[artifact_key] or not np.isclose(winner['validation_pr_auc'],report['model_selection']['best_score']):
        raise ValueError('Saved checkpoint does not match validation selection')
    if (root/'tuning_history.json').exists():
        trials=json.loads((root/'tuning_history.json').read_text())
        final_trials=[row for row in trials if row['stage']=='final']
        best=max(final_trials,key=lambda row:row['validation_pr_auc'])
        selected=report['model_selection']
        if best['candidate']!=selected['best_candidate'] or not np.isclose(best['validation_pr_auc'],selected['best_score']):
            raise ValueError('Model does not match the best final tuning candidate')
        if artifact.get('network_config')!=selected['best_params']['network_config']:
            raise ValueError('Saved network configuration differs from selected parameters')
    expected_files={f'{name}_predictions.csv' for name in ('dataset0','dataset1','dataset2')}
    if {p.name for p in (root/'submission').iterdir()}!=expected_files:
        raise ValueError('Submission directory must contain exactly three prediction CSVs')
    rows=[];pooled=[]
    for dataset in ('dataset0','dataset1','dataset2'):
        labels_path=verified_cleaned_file(Path(args.clean_root)/dataset,'labels.csv')
        if sha256(labels_path)!=report['preprocessing'][dataset]['output_sha256']['labels.csv']:
            raise ValueError('Evaluation labels differ from recorded inputs')
        labels=pd.read_csv(labels_path,dtype={k:str for k in KEY})
        prediction_path=root/'submission'/f'{dataset}_predictions.csv'
        scores=pd.read_csv(prediction_path,dtype={k:str for k in KEY})
        entry=submission['files'][prediction_path.name]
        if scores.columns.tolist()!=KEY+['score'] or scores.duplicated(KEY).any():
            raise ValueError('Invalid prediction CSV format or duplicates')
        if sha256(prediction_path)!=entry['sha256'] or len(scores)!=entry['n_sites']:
            raise ValueError('Prediction file changed after validation')
        if not np.isfinite(scores.score).all() or not scores.score.between(0,1).all():
            raise ValueError('Invalid prediction scores')
        all_rows=labels[KEY+['label']].merge(scores,on=KEY,how='outer',validate='one_to_one',indicator=True)
        if not all_rows['_merge'].eq('both').all(): raise ValueError('Incomplete full-dataset prediction coverage')
        test=manifest.loc[(manifest.dataset==dataset)&(manifest.split=='test'),KEY+['label']].reset_index(drop=True)
        frame=test.merge(scores,on=KEY,how='left',validate='one_to_one')
        if frame.score.isna().any() or not frame.label.isin([0,1]).all(): raise ValueError('Incomplete test predictions')
        # Confirm all-file inference agrees with the original held-out checkpoint scores.
        saved=pd.read_csv(root/'evaluation'/f'{dataset}_heldout_predictions.csv',dtype={k:str for k in KEY})
        pd.testing.assert_frame_equal(frame[KEY],saved[KEY])
        np.testing.assert_allclose(frame.score,saved.score,rtol=1e-5,atol=1e-6)
        rows.append(dict(dataset=dataset,model='shared_'+artifact['kind'],evaluation='heldout_test',**score_metrics(frame)))
        frame.insert(0,'dataset',dataset);pooled.append(frame)
    combined=pd.concat(pooled,ignore_index=True)
    if combined.duplicated(['dataset']+KEY).any(): raise ValueError('Duplicate sample-specific test examples')
    rows.append(dict(dataset='pooled',model='shared_'+artifact['kind'],evaluation='pooled_heldout_test',**score_metrics(combined)))
    metrics=pd.DataFrame(rows)
    metrics.to_csv(root/'evaluation_metrics.csv',index=False)
    combined[KEY+['score']].to_csv(root/'evaluation/pooled_heldout_predictions.csv',index=False)
    provenance=combined[['dataset']+KEY].copy();provenance.insert(0,'row_index',np.arange(len(provenance)))
    provenance.to_csv(root/'evaluation/pooled_provenance.csv',index=False)
    (root/'evaluation_audit.json').write_text(json.dumps(dict(
        model_sha256=fingerprint,evaluation='Test portions only. No training or validation rows contribute to reported metrics.',
        training_sites=report['training_sites'],validation_sites=report['validation_sites'],test_sites=len(combined),
        model='The exact same saved validation-selected checkpoint is used for every dataset and the pooled test set. No all-data refit.',
        checks=['Model artifact/report and validation epoch agree.', 'Shared gene/transcript and mixture groups never cross partitions.',
                'Original labels and prediction hashes verified.', 'Full-file test scores agree with checkpoint held-out scores.',
                'Pooled metrics concatenate test rows only, preserving sample identity.'],
        limitations=['These source datasets were previously explored, so results remain exploratory rather than fresh external validation.',
                     'Dataset2 position grouping assumes shared biological sites; cross-dataset gene/read provenance is incomplete.']),indent=2)+'\n')
    print(metrics.to_string(index=False))


if __name__=='__main__':
    main()
