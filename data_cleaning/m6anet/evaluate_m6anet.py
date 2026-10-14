#!/usr/bin/env python3
"""Evaluate site probabilities against explicit binary ground truth.

Defaults: datasets 0 and 1. To add dataset2, pass its existing cleaned
binary-label CSV/CSV.GZ; no fraction-to-binary threshold is guessed.
Combined metrics pool matched rows, keeping dataset identity, rather than
averaging AUCs. Outputs include matched sites for a common-site comparison.
"""
import argparse
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (
    auc, average_precision_score, precision_recall_curve, roc_auc_score,
)

KEYS = ["transcript_id", "transcript_position"]
LABEL_FILES = {
    "dataset0": "data0/data.info.labelled",
    "dataset1": "data1/data.info",
}


def normalize_keys(frame, source):
    missing = [key for key in KEYS if key not in frame]
    if missing:
        raise ValueError("{}: missing columns {}".format(source, missing))
    frame = frame.copy()
    if frame["transcript_id"].isna().any():
        raise ValueError("{}: missing transcript IDs".format(source))
    frame["transcript_id"] = frame["transcript_id"].astype(str)
    if frame["transcript_id"].str.strip().eq("").any():
        raise ValueError("{}: empty transcript IDs".format(source))
    positions = pd.to_numeric(frame["transcript_position"], errors="raise")
    if (not np.isfinite(positions.to_numpy(dtype=float)).all()
            or (positions < 0).any() or (positions != np.floor(positions)).any()):
        raise ValueError("{}: positions must be nonnegative integers".format(source))
    frame["transcript_position"] = positions.astype("int64")
    return frame


def calculate_metrics(rows):
    y = rows["label"].to_numpy(dtype=int)
    scores = rows["score"].to_numpy(dtype=float)
    result = {
        "n_evaluated": len(y),
        "n_positive": int((y == 1).sum()),
        "n_negative": int((y == 0).sum()),
        "positive_rate": float(y.mean()) if len(y) else np.nan,
        "ROC_AUC": np.nan,
        "PR_AUC_TRAPEZOID": np.nan,
        "PR_AP": np.nan,
        "status": "OK",
        "details": "",
    }
    if not len(y):
        result.update(status="NO_EVALUABLE_ROWS", details="No matched nonmissing binary labels")
    elif len(np.unique(y)) != 2:
        result.update(status="ONE_CLASS", details="Binary ranking comparison requires both classes")
    else:
        precision, recall, _ = precision_recall_curve(y, scores)
        result.update(
            ROC_AUC=float(roc_auc_score(y, scores)),
            PR_AUC_TRAPEZOID=float(auc(recall, precision)),
            PR_AP=float(average_precision_score(y, scores)),
        )
    return result


def evaluate_one(name, prediction_path, label_path, label_column="label"):
    predictions = normalize_keys(
        pd.read_csv(prediction_path, dtype={"transcript_id": str}), str(prediction_path))
    if "probability_modified" not in predictions:
        raise ValueError("{}: missing probability_modified".format(prediction_path))
    if predictions.duplicated(KEYS).any():
        raise ValueError("{}: duplicate predicted sites".format(name))
    predictions = predictions[KEYS + ["probability_modified"]].rename(
        columns={"probability_modified": "score"})
    predictions["score"] = pd.to_numeric(predictions["score"], errors="raise")
    if not predictions["score"].between(0, 1).all():
        raise ValueError("{}: scores must be finite probabilities in [0,1]".format(name))

    labels = normalize_keys(
        pd.read_csv(label_path, dtype={"transcript_id": str}), str(label_path))
    n_label_rows = len(labels)
    if label_column not in labels:
        raise ValueError("{}: missing binary-label column {!r}".format(label_path, label_column))
    labels = labels[KEYS + [label_column]].rename(columns={label_column: "label"})
    labels["label"] = pd.to_numeric(labels["label"], errors="raise")
    present = labels["label"].notna()
    if not labels.loc[present, "label"].isin([0, 1]).all():
        examples = labels.loc[present & ~labels["label"].isin([0, 1]), "label"].unique()[:10]
        raise ValueError("{}: fractional/nonbinary labels {}. Use the same cleaned binary "
                         "label file as your model; no conversion is automatic.".format(name, examples))
    labels = labels.drop_duplicates()
    n_duplicates_collapsed = n_label_rows - len(labels)
    if labels.duplicated(KEYS).any():
        raise ValueError("{}: conflicting labels for the same site; resolve before evaluation".format(name))

    joined = predictions.merge(labels, on=KEYS, how="outer", validate="one_to_one", indicator=True)
    both = joined["_merge"].eq("both")
    evaluated = joined.loc[both & joined["label"].notna(), KEYS + ["label", "score"]].copy()
    evaluated["label"] = evaluated["label"].astype(int)
    evaluated.insert(0, "dataset_id", name)
    summary = {
        "dataset_id": name,
        "datasets_included": name,
        "n_predictions": len(predictions),
        "n_label_rows": n_label_rows,
        "n_unique_label_sites": len(labels),
        "n_label_duplicates_collapsed": n_duplicates_collapsed,
        "n_matched": int(both.sum()),
        "n_unmatched_predictions": int(joined["_merge"].eq("left_only").sum()),
        "n_labels_only": int(joined["_merge"].eq("right_only").sum()),
        "n_missing_labels": int((both & joined["label"].isna()).sum()),
        **calculate_metrics(evaluated),
    }
    return summary, evaluated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench-root", type=Path,
                        default=Path.home() / "project/m6anet_bench")
    parser.add_argument("--data-root", type=Path,
                        default=Path.home() / "project/datasets")
    parser.add_argument("--datasets", nargs="+", choices=["dataset0", "dataset1", "dataset2"],
                        default=["dataset0", "dataset1"])
    parser.add_argument("--dataset2-labels", type=Path,
                        help="Actual cleaned CSV/CSV.GZ with dataset2's binary labels")
    parser.add_argument("--dataset2-label-column", default="label")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    if len(args.datasets) != len(set(args.datasets)):
        parser.error("Do not repeat dataset IDs")
    if "dataset2" in args.datasets and args.dataset2_labels is None:
        parser.error("dataset2 requires --dataset2-labels pointing to your cleaned binary labels")
    out_dir = args.out_dir or args.bench_root / "evaluation"
    if out_dir.exists() and any(out_dir.iterdir()):
        parser.error("Output directory is not empty; choose a new --out-dir")
    summaries, matched, paths = [], [], {}
    # Validate and compute every requested dataset before writing outputs.
    for name in args.datasets:
        prediction_path = args.bench_root / "output" / name / "data.site_proba.csv"
        label_path = (args.dataset2_labels if name == "dataset2"
                      else args.data_root / LABEL_FILES[name])
        column = args.dataset2_label_column if name == "dataset2" else "label"
        summary, rows = evaluate_one(name, prediction_path, label_path, column)
        summaries.append(summary)
        matched.append(rows)
        paths[name] = {"predictions": str(prediction_path.resolve()),
                       "labels": str(label_path.resolve()), "label_column": column}
    pooled = pd.concat(matched, ignore_index=True)
    count_fields = ["n_predictions", "n_label_rows", "n_unique_label_sites",
                    "n_label_duplicates_collapsed", "n_matched", "n_unmatched_predictions",
                    "n_labels_only", "n_missing_labels"]
    combined = {"dataset_id": "combined", "datasets_included": "+".join(args.datasets),
                **{key: sum(row[key] for row in summaries) for key in count_fields},
                **calculate_metrics(pooled)}
    summaries.append(combined)
    table = pd.DataFrame(summaries)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in zip(args.datasets, matched):
        rows.to_csv(out_dir / ("matched_" + name + ".csv"), index=False)
    pooled.to_csv(out_dir / "matched_combined.csv", index=False)
    table.to_csv(out_dir / "m6anet_metrics.csv", index=False)
    metadata = {
        "python": platform.python_version(), "numpy": np.__version__,
        "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
        "datasets_included": args.datasets, "files": paths,
        "score": "m6Anet probability_modified; no probability threshold",
        "PR_AUC_TRAPEZOID": "sklearn.metrics.auc(recall, precision)",
        "PR_AP": "sklearn.metrics.average_precision_score(labels, scores)",
        "combined_method": "Pool evaluated rows, retaining dataset_id; do not average AUCs",
        "comparison_note": "Compare both models on the same dataset/site intersection. "
                           "Scores on a model's own training data are training performance.",
    }
    (out_dir / "evaluation_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    view_columns = ["dataset_id", "datasets_included", "n_evaluated", "n_positive", "positive_rate",
                    "ROC_AUC", "PR_AUC_TRAPEZOID", "PR_AP", "status"]
    print(table[view_columns].to_string(index=False, float_format=lambda value: "{:.6f}".format(value)))
    print("\nSaved results to: " + str(out_dir.resolve()))
    if table["status"].ne("OK").any():
        raise SystemExit("Some rows have undefined metrics; inspect status/details in m6anet_metrics.csv")


if __name__ == "__main__":
    main()
