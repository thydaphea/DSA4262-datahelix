#!/usr/bin/env python3
"""Evaluate site-level m6A prediction CSVs; no model training is performed.

Install: python -m pip install numpy pandas scikit-learn
Run:     python evaluate_m6a_scores.py
Notebook: %run evaluate_m6a_scores.py

Edit DATASETS and OUTPUT_CSV below. Relative paths are resolved from this
script's folder. Prediction files contain transcript_id, transcript_position,
and score (probability of label 1, between 0 and 1). Label files contain the
same two keys and a binary label column. Comma-separated .info files work too.
Set labels_path=None when the prediction CSV itself contains the true labels.
Set label_col to your existing binary column name, e.g. 'label_binary'.
Fractional modification measurements are not accepted as binary labels;
use the binary column produced by your cleaning workflow instead.

Each key must be unique within each file. Duplicate keys, invalid coordinates,
nonbinary labels, missing keys, and invalid scores produce an ERROR row.
Unmatched predictions and predictions with missing labels are excluded, with
counts in the table. Missing predictions are also reported as labels_only.
Single-class datasets have NaN metrics and a SINGLE_CLASS status.

ALL_COMBINED pools evaluated rows from every successfully loaded dataset,
including single-class datasets. It is NOT an average of the dataset AUCs.
Sites repeated in different datasets count once PER DATASET, without collapsing
them. Use comparable probabilities from the same model for this pooled result.
If an input fails, the combined row is marked PARTIAL_COMBINED.

PR_AUC_TRAPEZOID is auc(recall, precision); PR_AP is average precision.
Both use the continuous scores; no threshold of 0.5 is applied.
Training-data predictions measure training fit; out-of-fold or held-out
predictions are needed to assess performance on unseen data.

References:
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.auc.html
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc, average_precision_score, precision_recall_curve, roc_auc_score,
)


# ================= EDIT THIS SECTION =================
BASE_DIR = Path(__file__).resolve().parent
DATASETS = [
    dict(dataset_id="dataset0", predictions_path="D:/NUS/AY2627_Year_4/AY2627_Y4_Sem1/DSA4262_genomics/Assignments/Team_Project/outputs/mlp_output/mlp_20261007/predictions_dataset0.csv",
         labels_path="D:/NUS/AY2627_Year_4/AY2627_Y4_Sem1/DSA4262_genomics/Assignments/Team_Project/outputs/data_cleaning/dataset0/20261006T143139_734059Z/clean_labeled_sites.csv.gz", label_col="label"),
    dict(dataset_id="dataset1", predictions_path="D:/NUS/AY2627_Year_4/AY2627_Y4_Sem1/DSA4262_genomics/Assignments/Team_Project/outputs/mlp_output/mlp_20261007/predictions_dataset1.csv",
         labels_path="D:/NUS/AY2627_Year_4/AY2627_Y4_Sem1/DSA4262_genomics/Assignments/Team_Project/outputs/data_cleaning/dataset1/20261006T144017_412380Z/clean_labeled_sites.csv.gz", label_col="label"),
    dict(dataset_id="dataset2", predictions_path="D:/NUS/AY2627_Year_4/AY2627_Y4_Sem1/DSA4262_genomics/Assignments/Team_Project/outputs/mlp_output/mlp_20261007/predictions_dataset2.csv",
         labels_path="D:/NUS/AY2627_Year_4/AY2627_Y4_Sem1/DSA4262_genomics/Assignments/Team_Project/outputs/data_cleaning/dataset2/20261006T144639_349494Z/clean_labeled_sites.csv.gz", label_col="label"),
]
# The dataset2 path above is an example: use your actual cleaned file and
# actual binary-label column name. Remove any datasets you do not need.
# For labels inside the prediction CSV: labels_path=None, label_col="label".
OUTPUT_CSV = r"D:/NUS/AY2627_Year_4/AY2627_Y4_Sem1/DSA4262_genomics/Assignments/Team_Project/outputs/mlp_output/mlp_20261007/result_verification.csv"
# =====================================================

KEYS = ["transcript_id", "transcript_position"]
COUNT_COLUMNS = [
    "n_predictions", "n_label_rows", "n_matched", "n_unmatched_predictions",
    "n_labels_only", "n_missing_labels", "n_evaluated", "n_positive", "n_negative",
]
METRIC_COLUMNS = ["positive_rate", "ROC_AUC", "PR_AUC_TRAPEZOID", "PR_AP"]


def resolve_path(path, base_dir):
    path = Path(path).expanduser()
    return path if path.is_absolute() else Path(base_dir) / path


def load_columns(path, columns):
    """Read selected columns as strings, retaining transcript identifiers."""
    header = pd.read_csv(path, nrows=0, encoding="utf-8-sig").columns
    missing = set(columns) - set(header)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    frame = pd.read_csv(path, usecols=columns, dtype="string", encoding="utf-8-sig")
    for col in columns:
        frame[col] = frame[col].str.strip().replace("", pd.NA)
    return frame


def validate_keys(frame, source):
    if frame[KEYS].isna().any().any():
        raise ValueError(f"{source}: missing transcript ID or position")
    positions = pd.to_numeric(frame["transcript_position"], errors="coerce")
    values = positions.to_numpy(dtype=float, na_value=np.nan)
    if not (np.isfinite(values) & (values >= 0) & (values == np.floor(values))).all():
        raise ValueError(f"{source}: positions must be nonnegative integers")
    frame["transcript_position"] = positions.astype("int64")
    duplicates = frame.duplicated(KEYS, keep=False)
    if duplicates.any():
        raise ValueError(
            f"{source}: {int(duplicates.sum())} rows have duplicate transcript-position keys; "
            "resolve duplicates before evaluation"
        )


def validate_scores(frame, source):
    scores = pd.to_numeric(frame["score"], errors="coerce")
    values = scores.to_numpy(dtype=float, na_value=np.nan)
    if not (np.isfinite(values) & (values >= 0) & (values <= 1)).all():
        raise ValueError(f"{source}: every score must be a finite probability from 0 to 1")
    frame["score"] = scores.astype(float)


def validate_labels(frame, label_col, source):
    original = frame[label_col]
    numeric = pd.to_numeric(original, errors="coerce")
    invalid = original.notna() & (~numeric.isin([0, 1]))
    if invalid.any():
        raise ValueError(
            f"{source}: {int(invalid.sum())} nonbinary or invalid labels in '{label_col}'; "
            "use an existing 0/1 column, not the fraction column"
        )
    frame[label_col] = numeric.astype("Float64")


def calculate_metrics(evaluated):
    n = len(evaluated)
    positive = int(evaluated["label"].sum()) if n else 0
    result = dict(n_evaluated=n, n_positive=positive, n_negative=n-positive,
                  positive_rate=positive/n if n else np.nan,
                  ROC_AUC=np.nan, PR_AUC_TRAPEZOID=np.nan, PR_AP=np.nan)
    if not n:
        return result, "NO_EVALUABLE_ROWS"
    if positive == 0 or positive == n:
        return result, "SINGLE_CLASS"
    y = evaluated["label"].to_numpy(dtype=int)
    scores = evaluated["score"].to_numpy(dtype=float)
    precision, recall, _ = precision_recall_curve(y, scores)
    result.update(ROC_AUC=float(roc_auc_score(y, scores)),
                  PR_AUC_TRAPEZOID=float(auc(recall, precision)),
                  PR_AP=float(average_precision_score(y, scores)))
    return result, "OK"


def evaluate_dataset(config, base_dir):
    name = config["dataset_id"]
    label_col = config.get("label_col", "label")
    if label_col in KEYS + ["score"]:
        raise ValueError("label_col must be separate from the ID, position, and score columns")
    prediction_path = resolve_path(config["predictions_path"], base_dir)
    label_path = config.get("labels_path")
    embedded = label_path is None
    predictions = load_columns(prediction_path, KEYS + ["score"] + ([label_col] if embedded else []))
    validate_keys(predictions, prediction_path)
    validate_scores(predictions, prediction_path)
    if embedded:
        labels = predictions[KEYS + [label_col]].copy()
        label_source = prediction_path
        predictions = predictions[KEYS + ["score"]].copy()
    else:
        label_source = resolve_path(label_path, base_dir)
        labels = load_columns(label_source, KEYS + [label_col])
        validate_keys(labels, label_source)
    validate_labels(labels, label_col, label_source)
    labels = labels.rename(columns={label_col: "label"})
    merged = predictions.merge(labels, on=KEYS, how="left", validate="one_to_one", indicator=True)
    matched = merged["_merge"].eq("both")
    known = matched & merged["label"].notna()
    evaluated = merged.loc[known, ["label", "score"]].copy()
    evaluated["label"] = evaluated["label"].astype(int)
    metrics, status = calculate_metrics(evaluated)
    row = dict(dataset_id=name, n_predictions=len(predictions), n_label_rows=len(labels),
               n_matched=int(matched.sum()), n_unmatched_predictions=int((~matched).sum()),
               n_labels_only=len(labels)-int(matched.sum()),
               n_missing_labels=int((matched & merged["label"].isna()).sum()),
               **metrics, status=status, details="")
    return row, evaluated


def evaluate_all(datasets, output_csv=None, base_dir=BASE_DIR):
    """Return a DataFrame and optionally save it. Can also be imported in Jupyter."""
    if not datasets:
        raise ValueError("Add at least one dataset to DATASETS")
    names = [str(c["dataset_id"]) for c in datasets]
    if any(not n.strip() for n in names) or len(names) != len(set(names)):
        raise ValueError("dataset_id values must be nonempty and unique")
    if "ALL_COMBINED" in names:
        raise ValueError("ALL_COMBINED is reserved for the pooled result")
    rows, parts = [], []
    for config in datasets:
        try:
            row, evaluated = evaluate_dataset(config, base_dir)
            rows.append(row)
            parts.append(evaluated)
        except (OSError, ValueError, OverflowError, pd.errors.ParserError) as error:
            rows.append(dict(dataset_id=config["dataset_id"], status="ERROR", details=str(error)))
    pooled = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["label", "score"])
    metrics, metric_status = calculate_metrics(pooled)
    successful_rows = [row for row in rows if row["status"] != "ERROR"]
    failed_names = [str(row["dataset_id"]) for row in rows if row["status"] == "ERROR"]
    combined = dict(dataset_id="ALL_COMBINED", **metrics, status=metric_status, details="")
    for col in COUNT_COLUMNS[:6]:
        combined[col] = sum(row[col] for row in successful_rows)
    if failed_names:
        combined["status"] = "PARTIAL_COMBINED" if successful_rows else "NO_VALID_DATASETS"
        combined["details"] = f"Excluded inputs: {', '.join(failed_names)}; pooled metric status: {metric_status}"
    rows.append(combined)
    columns = ["dataset_id"] + COUNT_COLUMNS + METRIC_COLUMNS + ["status", "details"]
    results = pd.DataFrame(rows).reindex(columns=columns)
    for col in COUNT_COLUMNS:
        results[col] = results[col].astype("Int64")
    print(results.to_string(index=False, float_format=lambda x: f"{x:.6f}", na_rep="NaN"))
    if output_csv is not None:
        destination = resolve_path(output_csv, base_dir)
        destination.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(destination, index=False, na_rep="NaN")
        print(f"\nSaved table: {destination}")
    print("\nPR_AUC_TRAPEZOID = trapezoidal PR area; PR_AP = average precision.")
    print("ALL_COMBINED pools evaluated rows across datasets; repeated sites count per dataset.")
    return results


if __name__ == "__main__":
    results = evaluate_all(DATASETS, OUTPUT_CSV)
    # A nonzero exit code tells automated workflows if any requested input failed.
    if results["status"].eq("ERROR").any():
        sys.exit(1)
