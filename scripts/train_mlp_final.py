"""Fit the selected MLP configuration on original train and validation rows."""
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
import torch

from features import SCHEMA
from neural import build, fit_normalizer, tensors


DATASETS = ("dataset0", "dataset1", "dataset2")
FEATURE_COLUMNS = (
    [f"signal_mean_{i}" for i in range(9)]
    + [f"signal_std_{i}" for i in range(9)]
    + [f"signal_q{q}_{i}" for q in (25, 50, 75) for i in range(9)]
    + ["log_read_count"]
    + [f"seq_pos{position}_{base}" for position in range(7) for base in "ACGT"]
)
NETWORK_CONFIG = {"hidden": [256, 128], "dropout": 0.5}
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
BATCH_SIZE = 256
EPOCHS = 16


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
        features = frame[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        if not np.isfinite(features).all():
            raise ValueError(f"{path} contains non-finite features")
        labels = frame["label"].to_numpy(dtype=np.int64)
        if np.unique(labels).size != 2:
            raise ValueError(f"{path} must contain both label classes")
        frames.append((dataset, path, frame, features, labels))

    X = np.concatenate([item[3] for item in frames])
    y = np.concatenate([item[4] for item in frames])
    counts = {
        dataset: {
            "rows": len(frame),
            "positives": int(labels.sum()),
            "original_split_counts": {
                split: int(count)
                for split, count in frame["original_split"].value_counts().items()
            },
        }
        for dataset, _, frame, _, labels in frames
    }
    return X, y, counts, frames


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
        raise ValueError("Prepared feature schema does not match this MLP trainer")

    X, y, dataset_counts, input_frames = load_training_data(prepared_dir)
    indices = np.arange(len(y))
    arrays = (X,)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(args.threads)
    rng = np.random.default_rng(args.seed)
    mean, scale = fit_normalizer("mlp", arrays, indices)
    model = build("mlp", NETWORK_CONFIG)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    criterion = torch.nn.BCEWithLogitsLoss()
    history = []

    print(
        f"Fitting final MLP: rows={len(y):,}, features={X.shape[1]}, "
        f"positives={int(y.sum()):,}, epochs={EPOCHS}",
        flush=True,
    )
    for epoch in range(1, EPOCHS + 1):
        model.train()
        permutation = rng.permutation(indices)
        total_loss = 0.0
        for start in range(0, len(permutation), BATCH_SIZE):
            batch = permutation[start : start + BATCH_SIZE]
            optimizer.zero_grad()
            logits = model(*tensors("mlp", arrays, batch, mean, scale))
            targets = torch.from_numpy(y[batch].astype(np.float32))
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch)
        epoch_loss = total_loss / len(indices)
        history.append({"epoch": epoch, "train_loss": epoch_loss})
        print(f"epoch {epoch}/{EPOCHS}: loss={epoch_loss:.6f}", flush=True)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    try:
        artifact = {
            "kind": "mlp",
            "schema": SCHEMA,
            "seed": args.seed,
            "max_reads": 64,
            "network_config": NETWORK_CONFIG,
            "state": {
                key: value.detach().cpu().numpy()
                for key, value in model.state_dict().items()
            },
            "mean": mean,
            "scale": scale,
            "best_epoch": EPOCHS,
            "torch_version": str(torch.__version__),
            "best_params": {
                "network_config": NETWORK_CONFIG,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "batch_size": BATCH_SIZE,
                "class_weight": "none",
            },
            "training_protocol": "fresh final fit on original train and validation rows; original test excluded",
        }
        model_path = staging / "model.joblib"
        joblib.dump(artifact, model_path)
        (staging / "training_history.json").write_text(
            json.dumps(history, indent=2) + "\n", encoding="utf-8"
        )
        metadata = {
            "model": "shared_mlp_final",
            "schema": SCHEMA,
            "config": {
                "network_config": NETWORK_CONFIG,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "batch_size": BATCH_SIZE,
                "loss": "BCEWithLogitsLoss (unweighted)",
                "epochs": EPOCHS,
                "seed": args.seed,
                "threads": args.threads,
                "normalization": "74 feature means and population standard deviations fitted on all original train+validation rows",
            },
            "training": {
                "rows": len(y),
                "features": X.shape[1],
                "positives": int(y.sum()),
                "test_rows_used": 0,
                "datasets": dataset_counts,
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
                    for dataset, path, _, _, _ in input_frames
                },
            },
            "environment": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scikit_learn": sklearn.__version__,
                "torch": torch.__version__,
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
        print(f"Saved final MLP model to {output_dir / 'model.joblib'}", flush=True)
    except Exception:
        shutil.rmtree(staging)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", default="data/prepared_cv")
    parser.add_argument("--out-dir", default="outputs/mlp_final_train_validation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=1)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
