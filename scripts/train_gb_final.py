"""Fit the selected GB configuration on original train and validation rows."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import tempfile

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from features import SCHEMA


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


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_training_data(prepared_dir):
    frames = []
    required = {
        "transcript_id",
        "transcript_position",
        "label",
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
        if frame.empty or not frame["original_split"].isin(["train", "validation"]).all():
            raise ValueError(f"{path} must contain only original train and validation rows")
        if not frame["label"].isin([0, 1]).all():
            raise ValueError(f"{path} contains non-binary labels")
        if frame[["transcript_id", "transcript_position"]].duplicated().any():
            raise ValueError(f"{path} contains duplicate site coordinates")
        frames.append((dataset, path, frame))

    X_parts, y_parts, counts = [], [], {}
    for dataset, path, frame in frames:
        X = frame[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        y = frame["label"].to_numpy(dtype=np.int64)
        if not np.isfinite(X).all():
            raise ValueError(f"{path} contains non-finite features")
        if np.unique(y).size != 2:
            raise ValueError(f"{path} must contain both label classes")
        X_parts.append(X)
        y_parts.append(y)
        counts[dataset] = {
            "rows": len(frame),
            "positives": int(y.sum()),
            "original_split_counts": {
                split: int(count)
                for split, count in frame["original_split"].value_counts().items()
            },
        }
    return np.concatenate(X_parts), np.concatenate(y_parts), counts, frames


def run(args):
    prepared_dir = Path(args.prepared_dir).resolve()
    output_dir = Path(args.out_dir).resolve()
    prepared_metadata_path = prepared_dir / "metadata.json"
    if not prepared_metadata_path.is_file():
        raise FileNotFoundError(f"Missing prepared-data metadata: {prepared_metadata_path}")
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    if args.threads < 1:
        raise ValueError("--threads must be positive")
    prepared_metadata = json.loads(prepared_metadata_path.read_text(encoding="utf-8"))
    if prepared_metadata.get("feature_columns") != FEATURE_COLUMNS:
        raise ValueError("Prepared feature schema does not match this GB trainer")

    X, y, dataset_counts, input_frames = load_training_data(prepared_dir)
    print(
        f"Fitting final GB model: rows={len(y):,}, features={X.shape[1]}, "
        f"positives={int(y.sum()):,}",
        flush=True,
    )
    model = HistGradientBoostingClassifier(
        **PARAMETERS,
        early_stopping=False,
        random_state=args.seed,
    )
    with threadpool_limits(limits=args.threads):
        model.fit(X, y)
    if model.n_iter_ != PARAMETERS["max_iter"]:
        raise RuntimeError(
            f"Expected all {PARAMETERS['max_iter']} boosting iterations; fitted {model.n_iter_}"
        )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    try:
        artifact = {
            "kind": "gb",
            "schema": SCHEMA,
            "model": model,
            "seed": args.seed,
            "best_params": PARAMETERS,
            "training_protocol": "final fit on original train and validation rows; original test excluded",
        }
        model_path = staging / "model.joblib"
        joblib.dump(artifact, model_path)
        metadata = {
            "model": "shared_gb_final",
            "schema": SCHEMA,
            "parameters": {
                **PARAMETERS,
                "early_stopping": False,
                "random_state": args.seed,
                "threads": args.threads,
                "normalization": "none",
            },
            "training": {
                "rows": len(y),
                "features": X.shape[1],
                "positives": int(y.sum()),
                "test_rows_used": 0,
                "datasets": dataset_counts,
                "iterations_fitted": model.n_iter_,
            },
            "inputs": {
                "prepared_metadata": {
                    "path": str(prepared_metadata_path),
                    "sha256": sha256(prepared_metadata_path),
                },
                "train_validation_files": {
                    dataset: {
                        "path": str(path),
                        "sha256": sha256(path),
                    }
                    for dataset, path, _ in input_frames
                },
            },
            "environment": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scikit_learn": sklearn.__version__,
            },
            "model_file": {
                "path": "model.joblib",
                "sha256": sha256(model_path),
                "bytes": model_path.stat().st_size,
            },
        }
        (staging / "run_metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        if output_dir.exists():
            raise FileExistsError(f"Output directory appeared during training: {output_dir}")
        os.rename(staging, output_dir)
        print(f"Saved final GB model to {output_dir / 'model.joblib'}", flush=True)
    except Exception:
        shutil.rmtree(staging)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", default="data/prepared_cv")
    parser.add_argument("--out-dir", default="outputs/gb_final_train_validation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=1)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
