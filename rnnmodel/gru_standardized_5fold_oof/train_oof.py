#!/usr/bin/env python3
"""Train the round-6 GRU on saved folds and export exactly one OOF score per row."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from gru_common import (IDENTITY, SiteGRU, audit_split, environment, fit, metric_row,
                        predict, read_partition, setup_runtime, sha256, transform, utc_now, write_json)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DATA = ROOT / 'outputs/data_cleaning/20261014T062257_386425Z_standardized_5folds'
SELECTION = ROOT / 'outputs/gru_weight_decay_round6/20261015T004132_998613Z/next_round_handoff.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=HERE / 'run')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    out = args.output
    device = setup_runtime(42, args.device, args.threads)
    selection = json.loads(SELECTION.read_text())
    config = selection['selected_config'].copy()
    epochs = int(selection['suggested_final_epochs'])
    config['epochs'] = epochs
    train_parts, test_parts, inputs = [], [], []
    for ds in ['dataset0', 'dataset1', 'dataset2']:
        for split, frames in [('train', train_parts), ('test', test_parts)]:
            path = DATA / ds / f'{ds}_clean_{split}.csv.gz'
            frames.append(read_partition(path, ds, split, metadata_only=split == 'test'))
            inputs.append({'path': str(path), 'sha256': sha256(path), 'use': 'fit and OOF' if split == 'train' else 'metadata-only separation audit'})
    train = pd.concat(train_parts, ignore_index=True)
    test = pd.concat(test_parts, ignore_index=True)
    groups, audit = audit_split(train, test, confirmed=True)
    assert set(train.fold.unique()) == {1, 2, 3, 4, 5}
    assert train.fold.notna().all() and not train.duplicated(IDENTITY).any()
    assignment = pd.read_csv(DATA / 'fold_assignments.csv.gz')
    pd.testing.assert_frame_equal(train[IDENTITY + ['fold']].reset_index(drop=True).astype({'dataset_id':str,'transcript_id':str}), assignment[IDENTITY + ['fold']].reset_index(drop=True))
    assert pd.DataFrame({'group': groups, 'fold': train.fold}).groupby('group').fold.nunique().max() == 1
    manifest = {'status': 'RUNNING', 'started_at': utc_now(), 'config': config, 'epochs': epochs,
                'configuration_source': str(SELECTION), 'configuration_source_sha256': sha256(SELECTION),
                'fold_manifest_sha256': sha256(DATA / 'fold_manifest.json'), 'inputs': inputs,
                'device': str(device), 'environment': environment(), 'seed_rule': '42 + held-out fold number',
                'split_audit': audit, 'fold_policy': 'Use existing fold column; fit fold != k, predict fold == k',
                'checkpoint_policy': 'Fixed 32 epochs from historical round-6 selection; no OOF early stopping',
                'caveat': 'Hyperparameters and duration were historically selected on this merged population; OOF predictions exclude rows from weight/scaler fitting but are not nested-CV generalization estimates.'}
    write_json(out / 'run_manifest.json', manifest)
    write_json(out / 'selected_config.json', config)
    train[IDENTITY + ['fold', 'label', 'original_split']].to_csv(out / 'fold_assignments.csv.gz', index=False)
    oof = np.full(len(train), np.nan)
    visits = np.zeros(len(train), dtype=int)
    fold_metrics, checks = [], []
    for fold in range(1, 6):
        ti, vi = np.flatnonzero(train.fold.ne(fold)), np.flatnonzero(train.fold.eq(fold))
        tr, va = train.iloc[ti], train.iloc[vi]
        assert set(groups[ti]).isdisjoint(groups[vi])
        folder = out / f'fold_{fold}'
        folder.mkdir()
        print(f'Start fold {fold}: {len(tr)} training rows, {len(va)} held-out rows', flush=True)
        model, prep, details = fit(tr, config, epochs, device, 42 + fold, folder)
        scores = predict(model, transform(va, prep), device)
        saved = torch.load(folder / 'model.pt', map_location=device, weights_only=True)
        restored = SiteGRU(config['hidden_size'], config['dropout']).to(device)
        restored.load_state_dict(saved['state_dict'])
        saved_prep = json.loads((folder / 'preprocessing.json').read_text())
        np.testing.assert_allclose(scores, predict(restored, transform(va, saved_prep), device), rtol=1e-6, atol=1e-7)
        assert details['epochs_run'] == epochs and np.isfinite(scores).all()
        oof[vi] = scores
        visits[vi] += 1
        prediction = va[IDENTITY + ['fold', 'label']].copy()
        prediction['score'] = scores
        prediction.to_csv(out / f'oof_predictions_fold_{fold}.csv', index=False)
        for name, part in [('pooled', prediction)] + list(prediction.groupby('dataset_id')):
            fold_metrics.append({'fold': fold, 'dataset_id': name, **metric_row(part.label, part.score, 0.5)})
        checks.append({'fold': fold, 'n_train': len(tr), 'n_oof': len(va), 'shared_groups': 0,
                       'saved_model_roundtrip': True, 'fit_summary': details,
                       'model_sha256': sha256(folder / 'model.pt'), 'preprocessing_sha256': sha256(folder / 'preprocessing.json')})
        pd.DataFrame(fold_metrics).to_csv(out / 'fold_metrics.csv', index=False)
        write_json(out / 'fold_checks.json', checks)
        print(f'Completed fold {fold}', flush=True)
    assert (visits == 1).all() and np.isfinite(oof).all()
    combined = train[IDENTITY + ['fold', 'label']].copy()
    combined['score'] = oof
    combined.to_csv(out / 'oof_predictions_combined.csv', index=False)
    individual = pd.concat([pd.read_csv(out / f'oof_predictions_fold_{k}.csv') for k in range(1, 6)])
    assert not individual.duplicated(IDENTITY).any() and len(individual) == len(train)
    expected = combined.sort_values(IDENTITY).reset_index(drop=True)
    actual = individual.sort_values(IDENTITY).reset_index(drop=True)
    pd.testing.assert_frame_equal(expected.astype({'dataset_id':str,'transcript_id':str}), actual, check_exact=False, check_dtype=False, rtol=1e-6, atol=1e-7)
    results = [{'dataset_id': name, **metric_row(part.label, part.score, .5)}
               for name, part in [('pooled', combined)] + list(combined.groupby('dataset_id'))]
    pd.DataFrame(results).to_csv(out / 'oof_metrics.csv', index=False)
    manifest.update(status='COMPLETE', finished_at=utc_now(), n_oof=len(combined),
                    verification={'one_oof_per_row': True, 'five_files_match_combined': True, 'all_saved_models_reloaded': True,
                                  'fold_group_overlap': 0, 'test_used_for_training_or_selection': False},
                    combined_sha256=sha256(out / 'oof_predictions_combined.csv'))
    write_json(out / 'run_manifest.json', manifest)
    print(pd.DataFrame(results)[['dataset_id','n_evaluated','ROC_AUC','PR_AUC_trapezoidal','average_precision']].to_string(index=False), flush=True)

if __name__ == '__main__':
    main()
