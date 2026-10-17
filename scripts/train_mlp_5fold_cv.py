"""Train the selected shared MLP configuration with the supplied five folds."""
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
import torch
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from metrics import pr_auc_score
from neural import build, fit_normalizer, tensors


DATASETS = ("dataset0", "dataset1", "dataset2")
FEATURE_COLUMNS = (
    [f"signal_mean_{i}" for i in range(9)]
    + [f"signal_std_{i}" for i in range(9)]
    + [f"signal_q{q}_{i}" for q in (25, 50, 75) for i in range(9)]
    + ["log_read_count"]
    + [f"seq_pos{position}_{base}" for position in range(7) for base in "ACGT"]
)
CONFIG = {"hidden": (256, 128), "dropout": 0.5}
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
BATCH_SIZE = 256
EPOCHS = 16
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


def train_fold(features, labels, fit_indices, fold, seed, threads):
    torch.manual_seed(seed + fold)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(threads)
    rng = np.random.default_rng(seed + fold)
    arrays = (features,)
    mean, scale = fit_normalizer("mlp", arrays, fit_indices)
    model = build("mlp", CONFIG)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    criterion = torch.nn.BCEWithLogitsLoss()
    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        permutation = rng.permutation(fit_indices)
        total_loss = 0.0
        for start in range(0, len(permutation), BATCH_SIZE):
            batch = permutation[start : start + BATCH_SIZE]
            optimizer.zero_grad()
            logits = model(*tensors("mlp", arrays, batch, mean, scale))
            target = torch.from_numpy(labels[batch].astype(np.float32))
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch)
        epoch_loss = total_loss / len(fit_indices)
        history.append({"epoch": epoch, "train_loss": epoch_loss})
        print(f"fold {fold} epoch {epoch}/{EPOCHS}: loss={epoch_loss:.6f}", flush=True)
    return model, mean, scale, history


def predict(model, features, indices, mean, scale):
    model.eval()
    scores = np.empty(len(indices), dtype=np.float32)
    arrays = (features,)
    with torch.no_grad():
        for start in range(0, len(indices), BATCH_SIZE):
            batch = indices[start : start + BATCH_SIZE]
            logits = model(*tensors("mlp", arrays, batch, mean, scale))
            scores[start : start + len(batch)] = torch.sigmoid(logits).cpu().numpy()
    return scores


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

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("feature_columns") != FEATURE_COLUMNS:
        raise ValueError("Prepared feature schema does not match this MLP script")

    rows, features, labels, folds = load_training_rows(prepared_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    histories = {}
    all_oof = np.full(len(rows), np.nan, dtype=np.float32)
    try:
        for fold in FOLDS:
            fit_indices = np.flatnonzero(folds != fold)
            heldout_indices = np.flatnonzero(folds == fold)
            if not len(fit_indices) or not len(heldout_indices):
                raise ValueError(f"Fold {fold} has an empty training or held-out partition")
            if np.unique(labels[fit_indices]).size != 2 or np.unique(labels[heldout_indices]).size != 2:
                raise ValueError(f"Fold {fold} must have both classes in train and held-out rows")
            print(
                f"Training fold {fold}: fit={len(fit_indices):,}, "
                f"OOF={len(heldout_indices):,}",
                flush=True,
            )
            model, mean, scale, history = train_fold(
                features, labels, fit_indices, fold, args.seed, args.threads
            )
            scores = predict(model, features, heldout_indices, mean, scale)
            if not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any():
                raise ValueError(f"Fold {fold} generated invalid probabilities")
            all_oof[heldout_indices] = scores
            histories[str(fold)] = {
                "fit_rows": len(fit_indices),
                "oof_rows": len(heldout_indices),
                "training_history": history,
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
        combined_path = staging / "oof_predictions.csv.gz"
        write_predictions(combined_path, rows, all_oof, all_indices)
        prediction_frame = pd.DataFrame({
            "dataset": rows["dataset"].to_numpy(),
            "label": labels,
            "score": all_oof,
        })
        metrics = calculate_metrics(prediction_frame)
        metrics.to_csv(staging / "oof_metrics.csv", index=False)

        run_metadata = {
            "model": "shared_mlp",
            "feature_schema": "nine-signals-summary-v1",
            "config": {
                "hidden": list(CONFIG["hidden"]),
                "dropout": CONFIG["dropout"],
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "batch_size": BATCH_SIZE,
                "loss": "BCEWithLogitsLoss (unweighted)",
                "epochs_per_fold": EPOCHS,
                "fold_seed_rule": "seed + fold",
                "base_seed": args.seed,
                "threads": args.threads,
                "normalization": "per-feature mean/std fit on that fold's training rows only; scales below 1e-8 set to 1",
            },
            "cv": {
                "fold_numbers": list(FOLDS),
                "training_rows": len(rows),
                "test_rows_excluded": True,
                "original_split_values": sorted(rows["original_split"].unique().tolist()),
                "per_fold": histories,
            },
            "environment": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scikit_learn": sklearn.__version__,
                "torch": torch.__version__,
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
    parser.add_argument("--out-dir", default="outputs/mlp_5fold_cv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
