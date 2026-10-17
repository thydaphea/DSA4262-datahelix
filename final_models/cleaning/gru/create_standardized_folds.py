#!/usr/bin/env python3
"""Copy cleaned datasets and append shared, grouped five-fold training assignments."""
import csv
import gzip
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[2]
SOURCE = None
DEST = None
from gru_common import audit_split


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    global SOURCE, DEST
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='New cleaning run before adding folds')
    parser.add_argument('--output', type=Path, required=True, help='New standardized dataset folder')
    args = parser.parse_args()
    SOURCE, DEST = args.source.resolve(), args.output.resolve()
    if DEST.exists():
        raise FileExistsError(f'Refusing to overwrite {DEST}')
    train_parts, test_parts = [], []
    for i in range(3):
        for split, parts in [('train', train_parts), ('test', test_parts)]:
            p = SOURCE / f'dataset{i}/dataset{i}_clean_{split}.csv.gz'
            df = pd.read_csv(p)
            assert df.dataset_id.eq(f'dataset{i}').all() and df['split'].eq(split).all()
            assert df.label.isin([0, 1]).all() and df.group_id.notna().all()
            assert not df.duplicated(['dataset_id', 'transcript_id', 'transcript_position']).any()
            assert 'fold' not in df
            parts.append(df)
    train = pd.concat(train_parts, ignore_index=True)
    test = pd.concat(test_parts, ignore_index=True)
    assert set(train.original_split) == {'train', 'validation'}
    groups, audit = audit_split(train, test, confirmed=True)
    strata = train.dataset_id + ':' + train.label.astype(str)
    folds = np.zeros(len(train), dtype=int)
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (ti, vi) in enumerate(cv.split(train, strata, groups), 1):
        assert set(groups[ti]).isdisjoint(groups[vi])
        folds[vi] = fold
    assert set(folds) == set(range(1, 6))
    assignments = train[['dataset_id', 'transcript_id', 'transcript_position', 'original_split', 'label', 'group_id']].copy()
    assignments['cv_group'] = groups
    assignments['fold'] = folds
    assert assignments.groupby('cv_group').fold.nunique().max() == 1
    summary = assignments.groupby(['dataset_id', 'fold']).agg(rows=('label', 'size'), positives=('label', 'sum'), groups=('cv_group', 'nunique')).reset_index()
    summary['negatives'] = summary.rows - summary.positives
    summary['positive_fraction'] = summary.positives / summary.rows
    assert len(summary) == 15 and (summary.positives > 0).all() and (summary.negatives > 0).all()
    shutil.copytree(SOURCE, DEST)
    hashes = []
    for i in range(3):
        ds = f'dataset{i}'
        src = SOURCE / ds / f'{ds}_clean_train.csv.gz'
        dst = DEST / ds / src.name
        assigned = assignments.loc[assignments.dataset_id.eq(ds), 'fold'].tolist()
        # Preserve each original CSV field exactly; append only the new metadata column.
        with gzip.open(src, 'rt', newline='') as fi, gzip.open(dst, 'wt', newline='') as fo:
            reader, writer = csv.reader(fi), csv.writer(fo)
            header = next(reader)
            writer.writerow(header + ['fold'])
            count = 0
            for row, fold in zip(reader, assigned, strict=True):
                writer.writerow(row + [fold])
                count += 1
            assert count == len(assigned)
        with gzip.open(src, 'rt', newline='') as fi, gzip.open(dst, 'rt', newline='') as fo:
            old, new = csv.reader(fi), csv.reader(fo)
            assert next(new) == next(old) + ['fold']
            for row, appended, fold in zip(old, new, assigned, strict=True):
                assert appended[:-1] == row and appended[-1] == str(fold)
        test_src = SOURCE / ds / f'{ds}_clean_test.csv.gz'
        test_dst = DEST / ds / test_src.name
        assert sha(test_src) == sha(test_dst)
        hashes.append({'dataset': ds, 'source_train_sha256': sha(src), 'output_train_sha256': sha(dst), 'test_sha256_unchanged': sha(test_src)})
    # Redirect the copied summary to this derived dataset; detailed historical manifests remain provenance.
    summary_file = DEST / 'run_summary.csv'
    run_summary = pd.read_csv(summary_file)
    run_summary['output_file'] = run_summary.output_file.str.replace(str(SOURCE), str(DEST), regex=False)
    run_summary.to_csv(summary_file, index=False)
    assignments.to_csv(DEST / 'fold_assignments.csv.gz', index=False)
    summary.to_csv(DEST / 'fold_summary.csv', index=False)
    pooled = assignments.groupby('fold').agg(rows=('label','size'), positives=('label','sum'), groups=('cv_group','nunique')).reset_index()
    pooled['negatives'] = pooled.rows - pooled.positives
    pooled['positive_fraction'] = pooled.positives / pooled.rows
    pooled.to_csv(DEST / 'pooled_fold_summary.csv', index=False)
    metadata = {'status': 'COMPLETE', 'source': str(SOURCE), 'output': str(DEST), 'algorithm': 'StratifiedGroupKFold', 'n_splits': 5, 'shuffle': True, 'seed': 42, 'strata': 'dataset_id + label', 'population': 'All current merged train rows (original train + validation)', 'grouping': 'Existing GRU audit_split: connected transcript/gene/cleaner groups globally across datasets0/1; confirmed shared mixture-position groups for dataset2', 'grouping_implementation_sha256': sha(Path(__file__).parent/'gru_common.py'), 'sklearn_version': sklearn.__version__, 'split_audit': audit, 'checks': {'all_original_training_fields_preserved_exactly': True, 'test_files_byte_identical': True, 'each_group_in_one_fold': True, 'every_dataset_has_both_classes_in_each_fold': True, 'original_split_counts': train.original_split.value_counts().to_dict()}, 'file_hashes': hashes}
    (DEST / 'fold_manifest.json').write_text(json.dumps(metadata, indent=2) + '\n')
    (DEST / 'README_folds.md').write_text('''# Standardized five-fold merged training data

Each datasetN/datasetN_clean_train.csv.gz contains the same rows, order, columns and field values as the source, with one appended integer metadata column: fold (1–5). Both original train and original validation rows are included. Test CSVs and other cleaning artifacts are copied; original inputs are unchanged. run_summary.csv now points to this folder. Other copied cleaning manifests/audits retain historical source paths and describe the source cleaning run; fold_manifest.json records this derived run.

Use the same saved assignments for every model. In CV iteration k, train on fold != k and validate on fold == k. Fit all scalers, imputers, encoders and other learned preprocessing on the four training folds only. Neither fold nor original_split is a model input feature. Existing feature_columns.json files remain unchanged. Do not reshuffle or generate separate per-model folds.

Assignments use StratifiedGroupKFold with 5 folds and seed 42, stratifying by dataset and label. Transcript/gene/cleaner groups are linked globally across dataset0/1. Dataset2 uses the established confirmed shared mixture-position grouping. Group constraints make fold sizes and class proportions approximate. No group crosses folds; training and test sites/groups remain disjoint.

fold_assignments.csv.gz stores identities and group IDs for joining predictions or auditing. fold_summary.csv gives counts by dataset and fold; pooled_fold_summary.csv gives combined counts. fold_manifest.json records source hashes, settings and successful checks. Test files have no fold column.

Reproduce with scripts/create_standardized_folds.py (refuses to overwrite an existing output folder). Previously trained models were not retrained; they must use these assignments in future training to share these folds.
''')
    print(summary.to_string(index=False))
    print('\nPooled:\n' + pooled.to_string(index=False))
    print(f'Verified output: {DEST}')

if __name__ == '__main__':
    main()
