#!/usr/bin/env python3
"""Reload a trained GRU and score cleaned train/validation CSVs without fitting anything."""
import argparse
import json
from pathlib import Path
import sys
import time

import torch

from gru_common import (
    SCHEMA, SiteGRU, audit_split, environment, export_results, load_data, load_spec,
    new_output, predict, setup_runtime, sha256, transform, utc_now, write_json, write_performance,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-config", required=True, help="JSON listing cleaned train/validation CSV paths")
    parser.add_argument("--model-dir", required=True, help="Completed train_gru.py output directory")
    parser.add_argument("--output-dir", required=True, help="New evaluation output directory")
    parser.add_argument("--threshold", type=float, default=0.5, help="Fixed beforehand; never optimized on validation labels")
    parser.add_argument("--batch-size", type=int, default=1024, help="Inference batch size; does not change the model")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1 or args.batch_size < 1:
        raise ValueError("Threshold must be in [0,1] and batch size must be positive")
    started = time.perf_counter()
    model_dir = Path(args.model_dir).expanduser().resolve()
    manifest = json.loads((model_dir / "run_manifest.json").read_text())
    if manifest.get("schema") != SCHEMA or manifest.get("status") != "COMPLETE":
        raise ValueError("Expected a completed compatible train_gru.py run")
    for name in ["model.pt", "preprocessing.json"]:
        if sha256(model_dir / name) != manifest["artifacts"][name]:
            raise ValueError(f"Saved artifact hash mismatch: {name}")
    device = setup_runtime(manifest["seed"], args.device, args.threads)
    spec = load_spec(args.data_config)
    train, validation, inputs = load_data(spec)
    # Require the actual training inputs, so overlap checks cannot be bypassed with unrelated data.
    if set(inputs) != set(manifest["inputs"]) or any(
        item["train_sha256"] != manifest["inputs"][name]["train_sha256"] for name, item in inputs.items()
    ):
        raise ValueError("Evaluation train files must exactly match the files used to fit this model")
    confirmed = spec.get("dataset2_position_groups_confirmed", False)
    if confirmed != manifest["dataset2_position_groups_confirmed"]:
        raise ValueError("Grouping policy differs from the training run")
    _, audit = audit_split(train, validation, confirmed)
    checkpoint = torch.load(model_dir / "model.pt", map_location="cpu", weights_only=True)
    if checkpoint["schema"] != SCHEMA or checkpoint["config"] != manifest["config"]:
        raise ValueError("Checkpoint metadata does not match training manifest")
    config = checkpoint["config"]
    model = SiteGRU(config["hidden_size"], config["dropout"]).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    prep = json.loads((model_dir / "preprocessing.json").read_text())
    out = new_output(args.output_dir)
    write_json(out / "split_audit.json", audit)
    rows = []
    for split, frame in [("train_apparent", train), ("validation", validation)]:
        values = predict(model, transform(frame, prep), device, args.batch_size)
        rows.extend(export_results(frame, values, out, split, args.threshold))
    notes = [
        "validation is held out from fitting; train_apparent is training fit, not a generalization estimate.",
        "All preprocessing statistics and model weights are loaded from the training run; nothing is refit.",
        f"Classification threshold is fixed at {args.threshold:g}; AUC metrics use continuous scores.",
        "PR_AUC_trapezoidal is trapezoidal precision–recall area; average_precision is reported separately.",
        "COMBINED pools observations, not dataset AUCs; repeated sites across datasets count per dataset.",
        "Missing validation labels are excluded from metrics, but all sites receive predictions.",
        "Single-class datasets have null AUC/AP; undefined threshold precision/recall/F1 use zero.",
    ]
    write_performance(rows, out, notes)
    write_json(out / "evaluation_manifest.json", dict(
        schema=SCHEMA, status="COMPLETE", finished_at=utc_now(), arguments=vars(args),
        model_dir=str(model_dir), model_sha256=sha256(model_dir / "model.pt"),
        preprocessing_sha256=sha256(model_dir / "preprocessing.json"),
        training_manifest_sha256=sha256(model_dir / "run_manifest.json"),
        inputs=inputs, threshold=args.threshold, device=str(device), environment=environment(),
        original_validation_files=all(v["validation_sha256"] == manifest["inputs"][k]["validation_sha256"] for k,v in inputs.items()),
        elapsed_seconds=time.perf_counter()-started))
    print(f"Saved evaluation and predictions: {out}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
