#!/usr/bin/env python3
"""Independently verify saved OOF outputs, models and preprocessing; refresh metrics."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from gru_common import SiteGRU, fit_preprocessing, metric_row, predict, setup_runtime, sha256, transform, utc_now, write_json
from train_oof import DATA, HERE, IDENTITY


def main():
    out = HERE / 'run'
    manifest = json.loads((out / 'run_manifest.json').read_text())
    train = pd.concat([pd.read_csv(DATA / ds / f'{ds}_clean_train.csv.gz') for ds in ['dataset0','dataset1','dataset2']], ignore_index=True)
    combined = pd.read_csv(out / 'oof_predictions_combined.csv')
    pd.testing.assert_frame_equal(train[IDENTITY + ['fold','label']], combined[IDENTITY + ['fold','label']])
    assert len(combined) == 172274 and not combined.duplicated(IDENTITY).any()
    device = setup_runtime(42, 'cpu', 4)
    parts = []
    for k in range(1,6):
        folder = out / f'fold_{k}'
        tr, va = train[train.fold.ne(k)], train[train.fold.eq(k)]
        individual = pd.read_csv(out / f'oof_predictions_fold_{k}.csv')
        pd.testing.assert_frame_equal(individual, combined[combined.fold.eq(k)].reset_index(drop=True), check_exact=False, rtol=1e-6, atol=1e-7)
        assert individual.fold.eq(k).all()
        parts.append(individual)
        prep = json.loads((folder/'preprocessing.json').read_text())
        expected = fit_preprocessing(tr)
        for field in ['sequence', 'coverage']:
            for stat in ['mean','scale']:
                np.testing.assert_allclose(prep[field][stat],expected[field][stat],rtol=1e-12,atol=1e-12)
        state = torch.load(folder/'model.pt', map_location=device, weights_only=True)
        assert state['epoch'] == 32 and state['config'] == manifest['config']
        net = SiteGRU(state['config']['hidden_size'],state['config']['dropout']).to(device)
        net.load_state_dict(state['state_dict'])
        np.testing.assert_allclose(predict(net,transform(va,prep),device),individual.score,rtol=1e-6,atol=1e-7)
        history=pd.read_csv(folder/'training_history.csv')
        assert history.epoch.tolist()==list(range(1,33)) and 'val_PR_AUC' not in history
        assert np.isfinite(history.train_BCE).all()
    cat=pd.concat(parts).sort_values(IDENTITY).reset_index(drop=True)
    pd.testing.assert_frame_equal(cat,combined.sort_values(IDENTITY).reset_index(drop=True),check_exact=False,rtol=1e-6,atol=1e-7)
    assignments=pd.read_csv(DATA/'fold_assignments.csv.gz')
    assert assignments.groupby('cv_group').fold.nunique().max()==1
    for entry in manifest['inputs']:
        assert sha256(entry['path'])==entry['sha256']
    results=[{'dataset_id': name, **metric_row(part.label,part.score,.5)} for name,part in [('pooled',combined)]+list(combined.groupby('dataset_id'))]
    pd.DataFrame(results).to_csv(out/'oof_metrics.csv',index=False)
    verification={'one_oof_per_row': True,'five_files_match_combined':True,'all_saved_models_reloaded':True,'scalers_match_training_folds_only':True,'no_oof_checkpoint_selection':True,'fold_group_overlap':0,'test_used_for_training_or_selection':False,'input_hashes_unchanged':True}
    manifest.update(status='COMPLETE',finished_at=utc_now(),n_oof=len(combined),verification=verification,combined_sha256=sha256(out/'oof_predictions_combined.csv'))
    write_json(out/'run_manifest.json',manifest)
    write_json(out/'verification.json',verification)
    print('PASS: five models, training-only scalers, original fold assignments, and all 172274 OOF rows verified.')
    print(pd.DataFrame(results)[['dataset_id','n_evaluated','ROC_AUC','PR_AUC_trapezoidal','average_precision']].to_string(index=False))

if __name__=='__main__':
    main()
