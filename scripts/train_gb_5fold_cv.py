"""Train the selected shared GB configuration with the supplied five folds."""
import argparse
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import tempfile

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from threadpoolctl import threadpool_limits

from metrics import pr_auc_score


DATASETS = ("dataset0", "dataset1", "dataset2")
FEATURE_COLUMNS = (
    [f"signal_mean_{i}" for i in range(9)]
    + [f"signal_std_{i}" for i in range(9)]
    + [f"signal_q{q}_{i}" for q in (25, 50, 75) for i in range(9)]
    + ["log_read_count"]
    + [f"seq_pos{position}_{base}" for position in range(7) for base in "ACGT"]
)
PARAMETERS = {
    "max_leaf_nodes": 63,
    "max_iter": 600,
    "learning_rate": 0.03,
    "l2_regularization": 1.0,
    "min_samples_leaf": 10,
}
FOLDS = (1, 2, 3, 4, 5)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_training_rows(prepared_dir):
    frames = []
    required = {
        "transcript_id",
        "transcript_position",
        "label",
        "n_reads",
        "original_split",
        "fold",
        *FEATURE_COLUMNS,
    }
    for dataset in DATASETS:
        path = prepared_dir / dataset / "train_validation.csv.gz"
        frame = pd.read_csv(path, compression="gzip", dtype={
            "transcript_id": str,
            "transcript_position": str,
        })
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        if frame.empty:
            raise ValueError(f"{path} contains no training/validation rows")
        if not frame["original_split"].isin(["train", "validation"]).all():
            raise ValueError(f"{path} contains rows outside original train/validation")
        if not frame["fold"].isin(FOLDS).all():
            raise ValueError(f"{path} contains a missing or invalid fold")
        if not frame["label"].isin([0, 1]).all():
            raise ValueError(f"{path} contains non-binary labels")
        if frame[["transcript_id", "transcript_position"]].duplicated().any():
            raise ValueError(f"{path} contains duplicate site coordinates")
        frames.append(frame.assign(dataset=dataset))
    rows = pd.concat(frames, ignore_index=True)
    if set(rows["fold"].unique()) != set(FOLDS):
        raise ValueError("Combined training data must include all five folds")
    features = rows[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    if not np.isfinite(features).all():
        raise ValueError("Feature matrix contains non-finite values")
    labels = rows["label"].to_numpy(dtype=np.int64)
    folds = rows["fold"].to_numpy(dtype=np.int64)
    return rows, features, labels, folds


def write_predictions(path, rows, scores, indices):
    columns = [
        "dataset",
        "transcript_id",
        "transcript_position",
        "label",
        "score",
        "fold",
        "original_split",
        "n_reads",
    ]
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for row_index, score in zip(indices, scores):
            row = rows.iloc[row_index]
            writer.writerow([
                row["dataset"],
                row["transcript_id"],
                row["transcript_position"],
                int(row["label"]),
                repr(float(score)),
                int(row["fold"]),
                row["original_split"],
                int(row["n_reads"]),
            ])


def calculate_metrics(predictions):
    records = []
    for dataset in (*DATASETS, "pooled"):
        selected = predictions if dataset == "pooled" else predictions.loc[
            predictions["dataset"] == dataset
        ]
        y = selected["label"].to_numpy(dtype=np.int64)
        score = selected["score"].to_numpy(dtype=float)
        records.append({
            "dataset": dataset,
            "evaluation": "five_fold_oof",
            "n_sites": len(selected),
            "n_positive": int(y.sum()),
            "prevalence": float(y.mean()),
            "average_precision": float(average_precision_score(y, score)),
            "pr_auc": pr_auc_score(y, score),
            "roc_auc": float(roc_auc_score(y, score)),
            "brier": float(brier_score_loss(y, score)),
        })
    return pd.DataFrame(records)


def run(args):
    prepared_dir = Path(args.prepared_dir).resolve()
    output_dir = Path(args.out_dir).resolve()
    metadata_path = prepared_dir / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing prepared-data metadata: {metadata_path}")
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    if args.threads < 1:
        raise ValueError("--threads must be positive")

    prepared_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if prepared_metadata.get("feature_columns") != FEATURE_COLUMNS:
        raise ValueError("Prepared feature schema does not match this GB script")
    rows, features, labels, folds = load_training_rows(prepared_dir)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    all_oof = np.full(len(rows), np.nan, dtype=np.float64)
    fold_details = {}
    try:
        for fold in FOLDS:
            fit_indices = np.flatnonzero(folds != fold)
            heldout_indices = np.flatnonzero(folds == fold)
            if not len(fit_indices) or not len(heldout_indices):
                raise ValueError(f"Fold {fold} has an empty training or held-out partition")
            if np.unique(labels[fit_indices]).size != 2 or np.unique(labels[heldout_indices]).size != 2:
                raise ValueError(f"Fold {fold} must have both classes in train and held-out rows")
            model = HistGradientBoostingClassifier(
                **PARAMETERS,
                early_stopping=False,
                random_state=args.seed,
            )
            print(
                f"Training fold {fold}: fit={len(fit_indices):,}, "
                f"OOF={len(heldout_indices):,}",
                flush=True,
            )
            with threadpool_limits(limits=args.threads):
                model.fit(features[fit_indices], labels[fit_indices])
                scores = model.predict_proba(features[heldout_indices])[:, 1]
            if not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any():
                raise ValueError(f"Fold {fold} generated invalid probabilities")
            all_oof[heldout_indices] = scores
            fold_details[str(fold)] = {
                "fit_rows": len(fit_indices),
                "oof_rows": len(heldout_indices),
            }
            write_predictions(
                staging / f"fold_{fold}_oof_predictions.csv.gz",
                rows,
                scores,
                heldout_indices,
            )
            del model

        if not np.isfinite(all_oof).all():
            raise ValueError("OOF predictions do not cover every train/validation row")
        all_indices = np.arange(len(rows))
        write_predictions(
            staging / "oof_predictions.csv.gz", rows, all_oof, all_indices
        )
        prediction_frame = pd.DataFrame({
            "dataset": rows["dataset"].to_numpy(),
            "label": labels,
            "score": all_oof,
        })
        calculate_metrics(prediction_frame).to_csv(
            staging / "oof_metrics.csv", index=False
        )

        run_metadata = {
            "model": "shared_gb",
            "feature_schema": "nine-signals-summary-v1",
            "config": {
                **PARAMETERS,
                "loss": "binary cross-entropy",
                "class_weight": "none",
                "early_stopping": False,
                "random_state": args.seed,
                "threads": args.threads,
                "normalization": "none (matches selected GB configuration)",
            },
            "cv": {
                "fold_numbers": list(FOLDS),
                "training_rows": len(rows),
                "test_rows_excluded": True,
                "original_split_values": sorted(rows["original_split"].unique().tolist()),
                "per_fold": fold_details,
            },
            "inputs": {
                "prepared_metadata": {
                    "path": str(metadata_path),
                    "sha256": sha256(metadata_path),
                },
                "training_files": {
                    dataset: {
                        "path": str(prepared_dir / dataset / "train_validation.csv.gz"),
                        "sha256": sha256(prepared_dir / dataset / "train_validation.csv.gz"),
                    }
                    for dataset in DATASETS
                },
            },
            "environment": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scikit_learn": sklearn.__version__,
            },
            "outputs": {},
        }
        for path in sorted(staging.iterdir()):
            if path.is_file():
                run_metadata["outputs"][path.name] = {
                    "sha256": sha256(path),
                    "bytes": path.stat().st_size,
                }
        (staging / "run_metadata.json").write_text(
            json.dumps(run_metadata, indent=2) + "\n", encoding="utf-8"
        )
        if output_dir.exists():
            raise FileExistsError(f"Output directory appeared during training: {output_dir}")
        os.rename(staging, output_dir)
        print(f"Saved five-fold OOF predictions and metrics to {output_dir}", flush=True)
    except Exception:
        shutil.rmtree(staging)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", default="data/prepared_cv")
    parser.add_argument("--out-dir", default="outputs/gb_5fold_cv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=1)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
