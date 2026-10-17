#!/usr/bin/env python3
"""Train the round-6 GRU; see README.md for inputs, modes, and output files."""
import argparse
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from gru_common import (
    CONFIG, IDENTITY, SCHEMA, audit_split, environment, export_results, fit,
    load_data, load_spec, metric_row, new_output, predict, setup_runtime,
    sha256, transform, utc_now, write_json, write_performance,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-config", required=True, help="JSON listing cleaned train/test CSV paths")
    parser.add_argument("--output-dir", required=True, help="New directory; existing directories are never overwritten")
    parser.add_argument("--mode", choices=["fixed", "cv-refit"], default="fixed",
                        help="fixed: all training data, 32 epochs; cv-refit: five-fold early stopping then full refit")
    parser.add_argument("--epochs", type=int, default=32, help="Fixed-mode refit duration (round-6 recommendation: 32)")
    parser.add_argument("--max-epochs", type=int, default=50, help="Maximum epochs per CV fold")
    parser.add_argument("--fold-assignments", help="Optional original CV assignments CSV, used only with cv-refit")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--threads", type=int, default=4)
    return parser.parse_args()


def get_folds(train, groups, seed, saved_path=None):
    if saved_path:
        saved = pd.read_csv(saved_path, dtype={"transcript_id": "string", "dataset_id": "string"})
        required = IDENTITY + ["label", "validation_fold"]
        if not set(required).issubset(saved) or saved.duplicated(IDENTITY).any() or len(saved) != len(train):
            raise ValueError("Invalid saved fold assignments")
        aligned = train[IDENTITY + ["label"]].merge(saved[required], on=IDENTITY, how="left",
                       validate="one_to_one", suffixes=("", "_saved"), sort=False)
        if not aligned.label.eq(aligned.label_saved).all():
            raise ValueError("Saved fold identities/labels do not match cleaned training data")
        numbers = aligned.validation_fold.to_numpy()
        if set(numbers) != set(range(1, 6)):
            raise ValueError("Saved folds must be numbered 1 through 5")
        folds = [(np.flatnonzero(numbers != k), np.flatnonzero(numbers == k)) for k in range(1, 6)]
    else:
        strata = train.dataset_id.astype(str) + ":" + train.label.astype(int).astype(str)
        folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(train, strata, groups))
    for number, (tr, va) in enumerate(folds, 1):
        if not len(tr) or not len(va) or set(groups[tr]) & set(groups[va]):
            raise ValueError(f"CV fold {number}: invalid or overlapping groups")
        if train.iloc[tr].label.nunique() != 2 or train.iloc[va].label.nunique() != 2:
            raise ValueError(f"CV fold {number}: both partitions need both label classes")
    return folds


def main():
    args = parse_args()
    if args.epochs < 1 or args.max_epochs < 1:
        raise ValueError("Epoch counts must be positive")
    if args.fold_assignments and args.mode != "cv-refit":
        raise ValueError("--fold-assignments requires --mode cv-refit")
    started = time.perf_counter()
    device = setup_runtime(args.seed, args.device, args.threads)
    spec = load_spec(args.data_config)
    # Test labels and signal features are never loaded by the training entry point.
    train, test_keys, inputs = load_data(spec, test_metadata_only=True)
    if set(train.original_split.unique()) != {"train", "validation"}:
        raise ValueError("Expected merged original train + validation rows")
    selection_path = Path(__file__).parent / "configuration_selection.json"
    import json
    selection = json.loads(selection_path.read_text())
    if any(CONFIG[k] != v for k, v in selection["selected_config"].items() if k != "epochs"):
        raise ValueError("Configuration differs from saved round-6 selection")
    groups, audit = audit_split(train, test_keys, spec.get("dataset2_position_groups_confirmed", False))
    if train.label.nunique() != 2:
        raise ValueError("Training requires positive and negative examples")
    folds = get_folds(train, groups, args.seed, args.fold_assignments) if args.mode == "cv-refit" else None
    out = new_output(args.output_dir)
    train[IDENTITY + ["original_split", "label", "fold"]].to_csv(out / "training_rows.csv.gz", index=False)
    audit["train_rows_by_dataset"] = train.groupby("dataset_id").size().to_dict()
    audit["original_split_counts"] = train.original_split.value_counts().to_dict()
    write_json(out / "split_audit.json", audit)
    write_json(out / "data_config_resolved.json", spec)
    config = dict(CONFIG, epochs=args.max_epochs)
    write_json(out / "selected_config.json", config)
    manifest = dict(schema=SCHEMA, status="RUNNING", started_at=utc_now(), arguments=vars(args),
                    config=config, device=str(device), seed=args.seed, inputs=inputs,
                    dataset2_position_groups_confirmed=spec.get("dataset2_position_groups_confirmed", False),
                    environment=environment(), test_evaluated=False, configuration_selection=selection, configuration_selection_sha256=sha256(selection_path), training_population="all original train + original validation; fold metadata excluded from model inputs")
    if args.fold_assignments:
        manifest["fold_assignments_sha256"] = sha256(args.fold_assignments)
    write_json(out / "run_manifest.json", manifest)
    source = out / "source"
    source.mkdir()
    for filename in ["train_gru.py", "evaluate_gru.py", "gru_common.py", "requirements.txt", "configuration_selection.json"]:
        shutil.copy2(Path(__file__).parent / filename, source / filename)
    rows, epochs = [], args.epochs
    if folds is not None:
        cvdir = out / "cv"
        cvdir.mkdir()
        oof = np.full(len(train), np.nan)
        assignments = train[IDENTITY + ["label"]].copy()
        assignments["cv_group"] = groups
        fold_rows = []
        for number, (tr, va) in enumerate(folds, 1):
            folder = cvdir / f"fold_{number}"
            folder.mkdir()
            assignments.loc[va, "validation_fold"] = number
            model, prep, details = fit(train.iloc[tr], config, args.max_epochs, device,
                                      args.seed + number, folder, validation=train.iloc[va])
            values = predict(model, transform(train.iloc[va], prep), device)
            oof[va] = values
            fold_rows.append(dict(fold=number, **details, **metric_row(train.iloc[va].label.to_numpy(), values, .5)))
        assignments["validation_fold"] = assignments.validation_fold.astype(int)
        assignments.to_csv(cvdir / "cv_fold_assignments.csv.gz", index=False)
        pd.DataFrame(fold_rows).to_csv(cvdir / "fold_metrics.csv", index=False)
        rows.extend(export_results(train, oof, cvdir, "cv_validation"))
        epochs = max(1, int(np.rint(np.median([r["selected_epoch"] for r in fold_rows]))))
        write_json(cvdir / "cv_summary.json", dict(
            mean_fold_PR_AUC=float(np.mean([r["PR_AUC_trapezoidal"] for r in fold_rows])),
            sd_fold_PR_AUC=float(np.std([r["PR_AUC_trapezoidal"] for r in fold_rows], ddof=1)),
            mean_fold_ROC_AUC=float(np.mean([r["ROC_AUC"] for r in fold_rows])),
            selected_refit_epochs=epochs,
            note="Validation folds selected checkpoints; these are selection estimates, not an independent test."))
    print(f"Fitting all {len(train):,} training rows for {epochs} epochs; device={device}", flush=True)
    model, prep, details = fit(train, config, epochs, device, args.seed, out)
    rows.extend(export_results(train, predict(model, transform(train, prep), device),
                               out / "train_outputs", "train_apparent"))
    notes = [
        "train_apparent measures fit on training data; it is not a generalization estimate.",
        "Test labels/features were not used. Run evaluate_gru.py for held-out test performance.",
        "CV validation, when present, was also used for early stopping; it is not an independent test.",
        "COMBINED pools rows; repeated biological sites in different datasets count once per dataset.",
        "PR_AUC_trapezoidal and average_precision are different metrics. Classification threshold = 0.5.",
        f"Full-data refit epochs: {epochs}; alpha=0 means positive class weight=1 (unweighted BCE).",
    ]
    write_performance(rows, out, notes)
    manifest.update(status="COMPLETE", finished_at=utc_now(), elapsed_seconds=time.perf_counter()-started,
                    final_fit=details, refit_epochs=epochs,
                    artifacts={name: sha256(out / name) for name in ["model.pt", "preprocessing.json"]})
    write_json(out / "run_manifest.json", manifest)
    print(f"Saved model and training outputs: {out}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
