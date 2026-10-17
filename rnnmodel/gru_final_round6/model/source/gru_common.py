"""Shared implementation for train_gru.py and evaluate_gru.py (rounds 1–5)."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import random
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, auc, average_precision_score, brier_score_loss,
    confusion_matrix, f1_score, log_loss, precision_recall_curve,
    precision_score, recall_score, roc_auc_score, roc_curve,
)

SCHEMA = "datahelix-site-gru-round6-final-v1"
KEY = ["transcript_id", "transcript_position"]
IDENTITY = ["dataset_id"] + KEY
POSITION_COLUMNS = [
    [f"{loc}_{measure}_{stat}" for stat in ["mean", "std", "median", "q25", "q75"]
     for measure in ["dwell", "signal_sd", "signal_mean"]]
    for loc in ["minus1", "center", "plus1"]
]
SUMMARY_COLUMNS = sum(POSITION_COLUMNS, [])
CONFIG = dict(hidden_size=32, dropout=0.0, lr=0.0003, weight_decay=0.0,
              batch_size=256, epochs=50, class_weighting=True, alpha=0.0,
              patience=5, min_delta=0.001)
METADATA = IDENTITY + ["sequence", "split", "group_id", "group_source", "gene_id"]


def write_json(path, obj):
    """Strict JSON: unavailable values are null, never nonstandard NaN."""
    def clean(value):
        if isinstance(value, dict):
            return {str(k): clean(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, np.ndarray)):
            return [clean(v) for v in value]
        if isinstance(value, np.generic):
            return clean(value.item())
        if isinstance(value, float) and not np.isfinite(value):
            return None
        if isinstance(value, Path):
            return str(value)
        return value
    Path(path).write_text(json.dumps(clean(obj), indent=2, allow_nan=False) + "\n")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def environment():
    return {"python": sys.version, "packages": {
        name: importlib.metadata.version(name)
        for name in ["numpy", "pandas", "torch", "scikit-learn", "matplotlib"]}}


def new_output(path):
    path = Path(path).expanduser().resolve()
    # A run is never silently overwritten or mixed with an earlier run.
    path.mkdir(parents=True, exist_ok=False)
    return path


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def setup_runtime(seed, device, threads):
    if threads < 1:
        raise ValueError("--threads must be positive")
    torch.set_num_threads(threads)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    return torch.device(device)


def load_spec(path):
    path = Path(path).expanduser().resolve()
    spec = json.loads(path.read_text())
    datasets = spec.get("datasets", [])
    if not datasets:
        raise ValueError("Data config requires a nonempty datasets list")
    names = [item["dataset_id"] for item in datasets]
    if len(names) != len(set(names)) or any(
        not re.fullmatch(r"[A-Za-z0-9_-]+", n) or n == "COMBINED" for n in names
    ):
        raise ValueError("Dataset IDs must be unique safe names; COMBINED is reserved")
    for item in datasets:
        for key in ["train_file", "test_file"]:
            p = Path(item[key]).expanduser()
            item[key] = str((p if p.is_absolute() else path.parent / p).resolve())
            if not Path(item[key]).is_file():
                raise FileNotFoundError(item[key])
    confirmed = spec.get("dataset2_position_groups_confirmed", False)
    if not isinstance(confirmed, bool):
        raise ValueError("dataset2_position_groups_confirmed must be a JSON boolean")
    return spec


def read_partition(path, name, split, metadata_only=False):
    header = pd.read_csv(path, nrows=0).columns
    required = METADATA if metadata_only else METADATA + ["n_reads_valid"] + SUMMARY_COLUMNS
    if split == "train" and not metadata_only:
        required = required + ["label"]
    missing = sorted(set(required) - set(header))
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    frame = pd.read_csv(path, usecols=METADATA if metadata_only else None,
                        dtype={c: "string" for c in ["transcript_id", "gene_id", "group_id",
                                                    "dataset_id", "group_source", "sequence"]})
    if frame.empty:
        raise ValueError(f"{path}: empty input")
    for col in ["transcript_id", "group_id"]:
        if frame[col].isna().any() or frame[col].str.strip().eq("").any():
            raise ValueError(f"{path}: missing {col}")
        if not frame[col].eq(frame[col].str.strip()).all():
            raise ValueError(f"{path}: whitespace in {col}")
    if not frame.dataset_id.eq(name).fillna(False).all() or not frame.split.eq(split).all():
        raise ValueError(f"{path}: dataset_id/split does not match {name}/{split}")
    positions = pd.to_numeric(frame.transcript_position, errors="raise").to_numpy(dtype=float)
    if not (np.isfinite(positions) & (positions >= 0) & (positions == np.floor(positions))
            & (positions < 2**63)).all():
        raise ValueError(f"{path}: positions must be nonnegative int64 integers")
    frame["transcript_position"] = positions.astype("int64")
    if frame.duplicated(KEY).any():
        raise ValueError(f"{path}: duplicate transcript/position keys")
    if not frame.sequence.str.fullmatch("[ACGT]{7}").fillna(False).all():
        raise ValueError(f"{path}: sequence must have seven A/C/G/T bases")
    if not frame.sequence.str.slice(1, 6).str.fullmatch("[AGT][AG]AC[ACT]").fillna(False).all():
        raise ValueError(f"{path}: central motif must be DRACH")
    if not metadata_only:
        values = frame[SUMMARY_COLUMNS + ["n_reads_valid"]].to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values[:, -1] < 1).any():
            raise ValueError(f"{path}: nonfinite features or coverage below 1")
        if "label" not in frame:
            frame["label"] = np.nan
        labels = pd.to_numeric(frame.label, errors="raise")
        if not labels.dropna().isin([0, 1]).all():
            raise ValueError(f"{path}: labels must be binary 0/1, not fractions")
        if split == "train" and labels.isna().any():
            raise ValueError(f"{path}: training labels cannot be missing")
        frame["label"] = labels
    return frame.reset_index(drop=True)


def load_data(spec, test_metadata_only=False):
    tables = {"train": [], "test": []}
    provenance = {}
    for item in spec["datasets"]:
        name = item["dataset_id"]
        provenance[name] = {}
        for split in tables:
            path = item[f"{split}_file"]
            tables[split].append(read_partition(path, name, split,
                metadata_only=split == "test" and test_metadata_only))
            provenance[name][f"{split}_file"] = path
            provenance[name][f"{split}_sha256"] = sha256(path)
    return (pd.concat(tables["train"], ignore_index=True),
            pd.concat(tables["test"], ignore_index=True), provenance)


def audit_split(train, test, confirmed=False):
    """Group globally across datasets by transcript, gene, and cleaner group ID."""
    both = pd.concat([train[METADATA], test[METADATA]], ignore_index=True)
    mixture = both.dataset_id.eq("dataset2") & confirmed
    if mixture.any():
        m = both.loc[mixture]
        expected = "dataset2:mixture_position:" + m.transcript_position.astype(str)
        if (not m.group_id.eq(expected).all()
                or not m.transcript_id.str.fullmatch(r"tx_id_[0-6]").all()
                or m.gene_id.notna().any()
                or m.groupby("group_id").sequence.nunique().gt(1).any()):
            raise ValueError("Dataset2 does not satisfy the confirmed mixture-position grouping")
    parent, row_nodes, genes = {}, [], {}

    def root(node):
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def link(left, right):
        parent[root(right)] = root(left)

    for is_mixture, row in zip(mixture, both.itertuples(index=False)):
        if is_mixture:
            node = "mixture:" + row.group_id
            root(node)
        else:
            node = "transcript:" + row.transcript_id
            root(node)
            gene = row.gene_id
            if row.group_id.startswith("gene:"):
                supplied_gene = row.group_id[5:]
                if pd.notna(gene) and gene != supplied_gene:
                    raise ValueError("gene_id conflicts with gene-based group_id")
                gene = supplied_gene
            if pd.notna(gene):
                if row.transcript_id in genes and genes[row.transcript_id] != gene:
                    raise ValueError("Conflicting transcript-to-gene annotations")
                genes[row.transcript_id] = gene
                link(node, "gene:" + gene)
            link(node, "supplied:" + row.group_id)
        row_nodes.append(node)
    groups = np.array([root(n) for n in row_nodes])
    tr_groups, te_groups = groups[:len(train)], groups[len(train):]
    shared_groups = set(tr_groups) & set(te_groups)
    shared_sites = train[KEY].drop_duplicates().merge(test[KEY].drop_duplicates(), on=KEY)
    audit = dict(train_rows=len(train), test_rows=len(test),
                 train_groups=len(set(tr_groups)), test_groups=len(set(te_groups)),
                 shared_sites=len(shared_sites), shared_biological_groups=len(shared_groups),
                 dataset2_position_groups_confirmed=confirmed,
                 cross_dataset_repeated_train_site_rows=int(train.duplicated(KEY, keep=False).sum()))
    if shared_groups or len(shared_sites):
        raise ValueError(f"Train/test overlap detected: {audit}")
    return tr_groups, audit


def raw_arrays(frame):
    sequence = np.stack([frame[c].to_numpy(dtype=np.float64) for c in POSITION_COLUMNS], axis=1)
    bases = {base: idx for idx, base in enumerate("ACGT")}
    indices = np.array([[bases[b] for b in seq] for seq in frame.sequence])
    one_hot = np.eye(4, dtype=np.float32)[indices].reshape(len(frame), 28)
    coverage = np.log1p(frame[["n_reads_valid"]].to_numpy(dtype=np.float64))
    return sequence.reshape(len(frame), 45), one_hot, coverage


def fit_preprocessing(frame):
    seq, _, cov = raw_arrays(frame)
    result = {"schema": SCHEMA, "position_columns": POSITION_COLUMNS}
    for name, values in [("sequence", seq), ("coverage", cov)]:
        scaler = StandardScaler().fit(values)
        result[name] = {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()}
    return result


def transform(frame, preprocessing):
    if (preprocessing["schema"] != SCHEMA or
            preprocessing["position_columns"] != POSITION_COLUMNS):
        raise ValueError("Saved feature schema does not match this implementation")
    seq, bases, cov = raw_arrays(frame)
    for name, values in [("sequence", seq), ("coverage", cov)]:
        stats = preprocessing[name]
        values -= np.asarray(stats["mean"])
        values /= np.asarray(stats["scale"])
    seq = seq.reshape(-1, 3, 15).astype("float32")
    aux = np.concatenate([bases, cov], axis=1).astype("float32")
    if not np.isfinite(seq).all() or not np.isfinite(aux).all():
        raise ValueError("Nonfinite transformed values")
    return seq, aux


class SiteGRU(nn.Module):
    def __init__(self, hidden_size=32, dropout=0.0):
        super().__init__()
        self.gru = nn.GRU(input_size=15, hidden_size=hidden_size, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden_size + 29, 32), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(32, 1))

    def forward(self, seq, aux):
        _, hidden = self.gru(seq)
        return self.head(torch.cat([hidden[-1], aux], dim=1)).squeeze(1)


def loader(seq, aux, batch_size, labels=None, shuffle=False, seed=42):
    tensors = [torch.from_numpy(seq), torch.from_numpy(aux)]
    if labels is not None:
        tensors.append(torch.from_numpy(np.asarray(labels, dtype="float32")))
    return DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, generator=torch.Generator().manual_seed(seed))


@torch.no_grad()
def predict(model, arrays, device, batch_size=1024):
    model.eval()
    return np.concatenate([
        torch.sigmoid(model(s.to(device), a.to(device))).cpu().numpy()
        for s, a in loader(*arrays, batch_size)
    ])


class EarlyStopping:
    """Same checkpoint and patience semantics as round 5."""
    def __init__(self, patience=5, min_delta=0.001):
        self.patience, self.min_delta = patience, min_delta
        self.best, self.reference = -np.inf, -np.inf
        self.best_epoch, self.wait = 0, 0

    def update(self, score, epoch):
        if not np.isfinite(score):
            raise ValueError("Validation PR AUC must be finite")
        improved = score > self.best
        if improved:
            self.best, self.best_epoch = float(score), epoch
        if score > self.reference + self.min_delta:
            self.reference, self.wait = float(score), 0
        else:
            self.wait += 1
        return improved, self.wait >= self.patience


def fit(frame, config, epochs, device, seed, output, validation=None):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if frame.label.nunique() != 2:
        raise ValueError("Training requires both label classes")
    prep = fit_preprocessing(frame)
    write_json(output / "preprocessing.json", prep)
    arrays = transform(frame, prep)
    val_arrays = transform(validation, prep) if validation is not None else None
    model = SiteGRU(config["hidden_size"], config["dropout"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    weight = float((frame.label == 0).sum() / (frame.label == 1).sum()) ** config["alpha"]
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weight, device=device))
    batches = loader(*arrays, config["batch_size"], labels=frame.label.to_numpy(), shuffle=True, seed=seed)
    stopper = EarlyStopping(config["patience"], config["min_delta"])
    history, best_state = [], None
    stop = False
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for s, a, y in batches:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(s.to(device), a.to(device)), y.to(device))
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            total += loss.item() * len(y)
        row = {"epoch": epoch, "train_BCE": total / len(frame), "pos_weight": weight}
        if validation is not None:
            scores = predict(model, val_arrays, device)
            metrics = metric_row(validation.label.to_numpy(), scores, 0.5)
            row.update(val_PR_AUC=metrics["PR_AUC_trapezoidal"],
                       val_ROC_AUC=metrics["ROC_AUC"], val_BCE=metrics["log_loss"])
            improved, stop = stopper.update(row["val_PR_AUC"], epoch)
            if improved:
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                torch.save({"schema": SCHEMA, "config": config, "state_dict": best_state,
                            "epoch": epoch}, output / "best_checkpoint.pt")
            row.update(best_epoch=stopper.best_epoch, patience_wait=stopper.wait)
        history.append(row)
        pd.DataFrame(history).to_csv(output / "training_history.csv", index=False)
        print(f"{output.name}: epoch {epoch}/{epochs}, loss={row['train_BCE']:.5f}" +
              (f", val PR-AUC={row['val_PR_AUC']:.5f}" if validation is not None else ""), flush=True)
        if stop:
            break
    # The final checkpoint contains optimizer state for inspection; no resume CLI is implied.
    torch.save({"epoch": epoch, "config": config, "state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict()}, output / "last_checkpoint.pt")
    if validation is not None:
        model.load_state_dict(best_state)
    details = dict(epochs_run=epoch, selected_epoch=stopper.best_epoch if validation is not None else epoch,
                   stop_reason="patience" if stop else "epoch_limit", pos_weight=weight)
    torch.save({"schema": SCHEMA, "config": config, "epoch": details["selected_epoch"],
                "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}}, output / "model.pt")
    write_json(output / "fit_summary.json", details)
    plot_history(pd.DataFrame(history), output / "learning_curves.png")
    return model, prep, details


def metric_row(labels, scores, threshold):
    labels, scores = np.asarray(labels, dtype=float), np.asarray(scores, dtype=float)
    if (len(labels) != len(scores) or not np.isfinite(scores).all()
            or ((scores < 0) | (scores > 1)).any()):
        raise ValueError("Scores must be aligned finite probabilities")
    known = ~np.isnan(labels)
    y, s = labels[known].astype(int), scores[known]
    if not np.isin(labels[known], [0, 1]).all():
        raise ValueError("Nonbinary labels")
    row = dict(n_predictions=len(scores), n_evaluated=len(y), n_missing_labels=int((~known).sum()),
               threshold=threshold, positive_fraction=float(y.mean()) if len(y) else None,
               ROC_AUC=None, PR_AUC_trapezoidal=None, average_precision=None)
    if not len(y):
        return dict(row, status="NO_LABELS")
    pred = s >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    row.update(n_positive=int(y.sum()), n_negative=int(len(y)-y.sum()),
               TN=int(tn), FP=int(fp), FN=int(fn), TP=int(tp),
               precision=float(precision_score(y, pred, zero_division=0)),
               recall=float(recall_score(y, pred, zero_division=0)),
               specificity=float(tn / (tn + fp)) if tn + fp else None,
               F1=float(f1_score(y, pred, zero_division=0)),
               accuracy=float(accuracy_score(y, pred)),
               log_loss=float(log_loss(y, np.clip(s, 1e-7, 1-1e-7), labels=[0, 1])),
               brier_score=float(brier_score_loss(y, s)), status="SINGLE_CLASS")
    if len(np.unique(y)) == 2:
        precision, recall, _ = precision_recall_curve(y, s)
        row.update(ROC_AUC=float(roc_auc_score(y, s)), PR_AUC_trapezoidal=float(auc(recall, precision)),
                   average_precision=float(average_precision_score(y, s)), status="OK")
    return row


def plot_history(history, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history.epoch, history.train_BCE, label="Training BCE")
    if "val_BCE" in history:
        axes[0].plot(history.epoch, history.val_BCE, label="Validation BCE")
        axes[1].plot(history.epoch, history.val_PR_AUC, label="Validation PR-AUC")
        axes[1].plot(history.epoch, history.val_ROC_AUC, label="Validation ROC-AUC")
        axes[1].legend()
    else:
        axes[1].text(.5, .5, "Full-training-data refit\nFixed duration selected before test evaluation",
                     ha="center", va="center", transform=axes[1].transAxes)
        axes[1].set_axis_off()
    axes[0].set(xlabel="Epoch", ylabel="Binary cross entropy")
    axes[0].legend()
    axes[1].set_xlabel("Epoch")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_evaluation(frame, scores, threshold, path, title):
    labels = frame.label.to_numpy(dtype=float)
    known = ~np.isnan(labels)
    y, s = labels[known].astype(int), scores[known]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    if len(np.unique(y)) == 2:
        fpr, tpr, _ = roc_curve(y, s)
        axes[0, 0].plot(fpr, tpr, label=f"AUC = {roc_auc_score(y,s):.4f}")
        axes[0, 0].plot([0, 1], [0, 1], "--", color="gray")
        precision, recall, _ = precision_recall_curve(y, s)
        axes[0, 1].plot(recall, precision, label=f"PR-AUC = {auc(recall,precision):.4f}\nAP = {average_precision_score(y,s):.4f}")
        axes[0, 1].axhline(y.mean(), linestyle="--", color="gray", label="Prevalence")
        axes[0, 0].legend()
        axes[0, 1].legend()
    else:
        for ax in axes[0]:
            ax.text(.5, .5, "AUC unavailable: fewer than two label classes", ha="center", fontsize=8)
    axes[0, 0].set(title="ROC", xlabel="False positive rate", ylabel="True positive rate", xlim=(0,1), ylim=(0,1.02))
    axes[0, 1].set(title="Precision–recall", xlabel="Recall", ylabel="Precision", xlim=(0,1), ylim=(0,1.02))
    if len(y):
        matrix = confusion_matrix(y, s >= threshold, labels=[0, 1])
        axes[1, 0].imshow(matrix, cmap="Blues")
        for (i, j), value in np.ndenumerate(matrix):
            axes[1, 0].text(j, i, f"{value:,}", ha="center", va="center",
                            color="white" if value > matrix.max()/2 else "black")
        for label in np.unique(y):
            axes[1, 1].hist(s[y == label], bins=np.linspace(0,1,31), alpha=.55,
                            label=f"Label {label}")
        axes[1, 1].legend()
    else:
        axes[1, 0].text(0, 0, "No labels; prediction only")
        axes[1, 1].hist(scores, bins=np.linspace(0,1,31))
    axes[1, 0].set(title=f"Confusion matrix (threshold {threshold:g})", xlabel="Predicted", ylabel="True",
                   xticks=[0,1], yticks=[0,1])
    axes[1, 1].set(title="Score distribution", xlabel="Predicted probability", ylabel="Site count")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def export_results(frame, scores, output, split, threshold=0.5):
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in frame.dataset_id.unique():
        mask = frame.dataset_id.eq(name).to_numpy(dtype=bool)
        part, values = frame.loc[mask], scores[mask]
        # Each dataset gets exactly the requested three columns; no ambiguous pooled CSV.
        part[KEY].assign(score=values).to_csv(output / f"predictions_{split}_{name}.csv", index=False)
        rows.append(dict(split=split, dataset_id=name, **metric_row(part.label.to_numpy(), values, threshold)))
        plot_evaluation(part, values, threshold, output / f"evaluation_{split}_{name}.png", f"{split}: {name}")
    rows.append(dict(split=split, dataset_id="COMBINED", **metric_row(frame.label.to_numpy(), scores, threshold)))
    plot_evaluation(frame, scores, threshold, output / f"evaluation_{split}_COMBINED.png", f"{split}: pooled datasets")
    detailed = frame[IDENTITY + ["label"]].copy()
    detailed["score"] = scores
    detailed["predicted_label"] = (scores >= threshold).astype(int)
    detailed.to_csv(output / f"detailed_predictions_{split}.csv.gz", index=False)
    return rows


def write_performance(rows, output, notes):
    results = pd.DataFrame(rows)
    results.to_csv(output / "performance_summary.csv", index=False)
    write_json(output / "performance_summary.json", {"notes": notes, "results": rows})
    lines = ["GRU performance summary", "", *notes, "", results.to_string(index=False), ""]
    (output / "performance_summary.txt").write_text("\n".join(lines))
    print(results[["split", "dataset_id", "n_evaluated", "PR_AUC_trapezoidal", "ROC_AUC", "status"]].to_string(index=False))
