"""Predict m6A probabilities from raw m6Anet JSONL; also owns shared cleaning.

No labels or training files are required for inference. train.py imports this module.
"""
import argparse
from collections import Counter
import csv
import gzip
import hashlib
import json
import re
import os
from pathlib import Path
import tempfile

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from scipy.special import expit

ROOT = Path(__file__).resolve().parent
KEY = ["transcript_id", "transcript_position"]
IDENTITY = ["dataset_id"] + KEY
SCHEMA = "datahelix-site-gru-round6-final-v1"
CLEANING_VERSION = "shared-site-table-v1"
LOCATIONS = ("minus1", "center", "plus1")
MEASURES = ("dwell", "signal_sd", "signal_mean")
STAT_NAMES = ("mean", "std", "median", "q25", "q75")
NONNEGATIVE_COLUMNS = (0, 1, 3, 4, 6, 7)
POSITION_COLUMNS = [[f"{loc}_{measure}_{stat}" for stat in STAT_NAMES
                     for measure in MEASURES] for loc in LOCATIONS]
SUMMARY_COLUMNS = sum(POSITION_COLUMNS, [])
FLAT_COLUMNS = [f"{loc}_{measure}_{stat}" for stat in ("mean", "std", "q25", "median", "q75")
                for loc in LOCATIONS for measure in MEASURES]
SCORE_COLUMNS = ["gru_score", "mlp_score", "gb_score"]

def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key!r}")
        result[key] = value
    return result

def _read_matrix(reads):
    if isinstance(reads, np.ndarray):
        if reads.ndim != 2:
            raise ValueError("Expected a two-dimensional array of reads")
        reads = reads.tolist()
    elif not isinstance(reads, list):
        raise ValueError("Expected a list or two-dimensional array of reads")
    valid = []
    for read in reads:
        if (not isinstance(read, list) or len(read) != 9
                or any(isinstance(value, bool) or not isinstance(value, (int, float))
                       for value in read)):
            continue
        values = np.asarray(read, dtype=float)
        if (not np.isfinite(values).all()
                or (values[list(NONNEGATIVE_COLUMNS)] < 0).any()):
            continue
        valid.append(values)
    return np.asarray(valid, dtype=float).reshape(-1, 9)

def clean_gru_site(transcript, position, sequence, reads):
    matrix = _read_matrix(reads)
    invalid_reads_removed = len(reads) - len(matrix)
    if not re.fullmatch(r"[ACGT]{7}", sequence):
        return None, "invalid_sequences", invalid_reads_removed
    if not re.fullmatch(r"[AGT][AG]AC[ACT]", sequence[1:6]):
        return None, "non_drach_sequences", invalid_reads_removed
    if not len(matrix):
        return None, "too_few_valid_reads", invalid_reads_removed

    summaries = (
        matrix.mean(axis=0),
        matrix.std(axis=0),
        np.median(matrix, axis=0),
        np.quantile(matrix, 0.25, axis=0),
        np.quantile(matrix, 0.75, axis=0),
    )
    if not np.isfinite(np.concatenate(summaries)).all():
        return None, "nonfinite_summary", invalid_reads_removed
    row = {
        "transcript_id": transcript,
        "transcript_position": int(position),
        "sequence": sequence,
        "n_reads_valid": len(matrix),
    }
    for stat_name, summary in zip(STAT_NAMES, summaries):
        for location_index, location in enumerate(LOCATIONS):
            for measure_index, measure in enumerate(MEASURES):
                column_index = location_index * 3 + measure_index
                row[f"{location}_{measure}_{stat_name}"] = float(
                    summary[column_index]
                )
    return row, None, invalid_reads_removed

def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

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
    def __init__(self, hidden_size=32, dropout=0.0, bidirectional=False,
                 all_position_outputs=False, head_size=32):
        super().__init__()
        self.bidirectional = bool(bidirectional)
        self.all_position_outputs = bool(all_position_outputs)
        self.gru = nn.GRU(input_size=15, hidden_size=hidden_size, batch_first=True,
                          bidirectional=self.bidirectional)
        direction_count = 2 if self.bidirectional else 1
        recurrent_width = hidden_size * direction_count * (3 if self.all_position_outputs else 1)
        self.head = nn.Sequential(nn.Linear(recurrent_width + 29, head_size), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(head_size, 1))

    def forward(self, seq, aux):
        outputs, hidden = self.gru(seq)
        if self.all_position_outputs:
            recurrent = outputs.reshape(outputs.shape[0], -1)
        elif self.bidirectional:
            recurrent = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            recurrent = hidden[-1]
        return self.head(torch.cat([recurrent, aux], dim=1)).squeeze(1)

def loader(seq, aux, batch_size, labels=None, shuffle=False, seed=42):
    tensors = [torch.from_numpy(seq), torch.from_numpy(aux)]
    if labels is not None:
        tensors.append(torch.from_numpy(np.asarray(labels, dtype="float32")))
    return DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, generator=torch.Generator().manual_seed(seed))

class PooledMLP(nn.Module):
    def __init__(self, hidden=(64,32), dropout=.2):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(74, hidden[0]), nn.ReLU(), nn.Dropout(dropout),
                                     nn.Linear(hidden[0], hidden[1]), nn.ReLU(), nn.Linear(hidden[1], 1))

    def forward(self, features):
        return self.network(features).squeeze(-1)

def read_sites(path, audit):
    """Stream JSONL (plain/gzip), rejecting ambiguous identities and structures."""
    seen = set()
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt', encoding='utf-8') as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line, object_pairs_hook=_unique_object)
                if not isinstance(record, dict) or not record:
                    raise ValueError('Expected a nonempty transcript object')
                for transcript, positions in record.items():
                    if not transcript or transcript != transcript.strip() or not isinstance(positions, dict) or not positions:
                        raise ValueError('Invalid transcript ID or positions')
                    for position, motifs in positions.items():
                        if not re.fullmatch(r'[0-9]+', position) or int(position) >= 2**63:
                            raise ValueError('Position must be a nonnegative int64')
                        key = (transcript, int(position))
                        if key in seen:
                            raise ValueError(f'Duplicate site: {key}')
                        seen.add(key)
                        if not isinstance(motifs, dict) or len(motifs) != 1:
                            raise ValueError(f'Expected one sequence at {key}')
                        motif, reads = next(iter(motifs.items()))
                        motif = motif.upper().replace('U', 'T')
                        row, reason, removed = clean_gru_site(*key, motif, reads)
                        audit['input_sites'] += 1
                        audit['input_reads'] += len(reads)
                        audit['invalid_reads_removed'] += removed
                        if reason:
                            audit[reason] += 1
                            continue
                        row.update(n_reads_raw=len(reads), n_reads=row['n_reads_valid'],
                                   n_reads_removed=removed, had_invalid_reads=bool(removed))
                        audit['retained_sites'] += 1
                        yield row
            except (ValueError, TypeError, OverflowError) as e:
                raise ValueError(f'{path}, line {line_no}: {e}') from e


def read_labels(path):
    """Read each sample's labels independently; fractions > 0 become label 1."""
    opener = gzip.open if str(path).endswith('.gz') else open
    rows = []
    seen = set()
    with opener(path, 'rt', encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        if len(fields) != len(set(fields)) or not set(KEY + ['label']).issubset(fields):
            raise ValueError('Labels need transcript_id,transcript_position,label (gene_id optional)')
        for line, row in enumerate(reader, 2):
            tx, pos = row['transcript_id'], row['transcript_position']
            if not tx or tx != tx.strip() or not re.fullmatch(r'[0-9]+', pos) or int(pos) >= 2**63:
                raise ValueError(f'{path}:{line}: invalid coordinate')
            key = (tx, int(pos))
            if key in seen:
                raise ValueError(f'{path}:{line}: duplicate label {key}')
            seen.add(key)
            value = float(row['label'])
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f'{path}:{line}: label must be within [0,1]')
            rows.append(dict(zip(KEY, key), label=int(value > 0),
                             modification_fraction=value, gene_id=row.get('gene_id') or None))
    if not rows:
        raise ValueError(f'{path}: empty labels')
    return pd.DataFrame(rows)


def clean_dataset(data, labels=None, dataset_id='sample'):
    audit = Counter()
    frame = pd.DataFrame(read_sites(data, audit))
    if frame.empty:
        raise ValueError(f'{data}: no sites passed cleaning')
    frame.insert(0, 'dataset_id', dataset_id)
    if labels:
        lookup = read_labels(labels)
        audit['labels_without_retained_reads'] = int(len(lookup) - len(lookup.merge(frame[KEY], on=KEY)))
        frame = frame.merge(lookup, on=KEY, how='left', validate='one_to_one', sort=False)
        audit['retained_sites_without_labels'] = int(frame.label.isna().sum())
    return frame, dict(audit)


def flat_features(frame):
    """45 signal summaries, log1p(valid coverage), 7x4 A/C/G/T indicators."""
    _, one_hot, coverage = raw_arrays(frame)
    return np.column_stack([frame[FLAT_COLUMNS].to_numpy(dtype=float), coverage, one_hot]).astype('float32')


def transform_scores(x, representation):
    if representation == 'raw':
        return x
    clipped = np.clip(x, 1e-6, 1-1e-6)
    logits = np.log(clipped / (1-clipped))
    if representation == 'logit':
        return logits
    if representation == 'raw_logit':
        return np.column_stack([x, logits])
    raise ValueError(f'Unknown ensemble representation: {representation}')


def load_models(model_dir, method):
    if method not in ('ensemble', 'mlp', 'gb', 'gru'):
        raise ValueError(f'Unknown prediction method: {method}')
    root = Path(model_dir)
    result = {}
    names = ['gru', 'mlp', 'gb'] if method == 'ensemble' else [method]
    for name in names:
        if name == 'gru':
            directory = root / 'gru'
            state = torch.load(directory / 'model.pt', map_location='cpu', weights_only=True)
            config = state['config']
            model = SiteGRU(config['hidden_size'], config['dropout'],
                            bidirectional=config.get('bidirectional', False),
                            all_position_outputs=config.get('all_position_outputs', False),
                            head_size=config.get('head_size', 32))
            model.load_state_dict(state['state_dict'])
            prep = json.loads((directory / 'preprocessing.json').read_text())
            result[name] = (model.eval(), prep)
        else:
            artifact = joblib.load(root / name / 'model.joblib')
            if name == 'mlp':
                model = PooledMLP(**artifact['network_config'])
                model.load_state_dict({k: torch.as_tensor(v) for k, v in artifact['state'].items()})
                result[name] = (model.eval(), artifact)
            else:
                result[name] = artifact['model']
    if method == 'ensemble':
        result['ensemble'] = joblib.load(root / 'ensemble/oof_holdout/logistic_regression.joblib')
        if result['ensemble']['feature_order'] != SCORE_COLUMNS:
            raise ValueError('Ensemble feature order mismatch')
    return result


@torch.no_grad()
def score_frame(frame, models, method):
    x = flat_features(frame)
    scores = {}
    for name in ('gru', 'mlp', 'gb'):
        if name not in models:
            continue
        if name == 'gb':
            scores[name] = models[name].predict_proba(x)[:, 1]
        elif name == 'mlp':
            model, artifact = models[name]
            z = (x - artifact['mean']) / artifact['scale']
            scores[name] = np.concatenate([torch.sigmoid(model(torch.from_numpy(z[i:i+256]))).numpy()
                                          for i in range(0, len(z), 256)])
        else:
            model, prep = models[name]
            arrays = transform(frame, prep)
            scores[name] = np.concatenate([torch.sigmoid(model(s, a)).numpy() for s, a in loader(*arrays, 1024)])
    if method == 'ensemble':
        base = np.column_stack([scores[n] for n in ('gru', 'mlp', 'gb')])
        bundle = models['ensemble']
        # Binary logistic formula avoids version-specific predict_proba attributes.
        pipeline = bundle['pipeline']
        scaler, logistic = pipeline.steps[0][1], pipeline.steps[-1][1]
        if not np.array_equal(logistic.classes_, [0, 1]) or logistic.coef_.shape[0] != 1:
            raise ValueError('Expected a binary logistic ensemble with classes [0,1]')
        z = transform_scores(base, bundle['config']['representation'])
        z = (z - scaler.mean_) / scaler.scale_
        output = expit(z @ logistic.coef_[0] + logistic.intercept_[0])
    else:
        output = scores[method]
    if not np.isfinite(output).all() or ((output < 0) | (output > 1)).any():
        raise ValueError('Model produced invalid probabilities')
    return output


def predict_file(data, model_dir, output, method='ensemble', threads=1):
    output = Path(output)
    if output.exists() or output.with_suffix('.audit.json').exists():
        raise FileExistsError(f'Refusing to overwrite {output} or its audit')
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(threads)
    audit = Counter()
    before = sha256(data)
    models = load_models(model_dir, method)
    fd, temporary = tempfile.mkstemp(prefix='.prediction-', dir=output.parent)
    os.close(fd)
    try:
        with open(temporary, 'w', newline='') as handle, threadpool_limits(limits=threads):
            writer = csv.writer(handle)
            writer.writerow(KEY + ['score'])
            batch = []
            def flush():
                frame = pd.DataFrame(batch)
                scores = score_frame(frame, models, method)
                writer.writerows((*key, float(score)) for key, score in zip(frame[KEY].itertuples(index=False, name=None), scores))
                batch.clear()
            for row in read_sites(data, audit):
                batch.append(row)
                if len(batch) == 4096:
                    flush()
            if batch:
                flush()
        if not audit['retained_sites']:
            raise ValueError('No sites passed cleaning')
        if sha256(data) != before:
            raise ValueError('Input changed while predicting')
        os.replace(temporary, output)
        report = dict(cleaning_version=CLEANING_VERSION, data_sha256=before, method=method,
                      counts=dict(audit), output_columns=KEY+['score'], labels_required=False,
                      prediction_sha256=sha256(output))
        output.with_suffix('.audit.json').write_text(json.dumps(report, indent=2)+'\n')
    finally:
        if Path(temporary).exists():
            Path(temporary).unlink()
    print(f'Saved {audit["retained_sites"]:,} predictions to {output}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, type=Path, help='Raw m6Anet JSONL or JSONL.gz, any filename')
    parser.add_argument('--weights-dir', '--model-dir', dest='model_dir', type=Path,
                        default=ROOT/'models',
                        help='Folder containing gru/, mlp/, gb/ and ensemble/ weights; defaults to this package\'s models/')
    parser.add_argument('--method', choices=['ensemble','mlp','gb','gru'], default='ensemble')
    parser.add_argument('--output', required=True, type=Path, help='CSV, e.g. datahelix_dataset0.csv')
    parser.add_argument('--threads', type=int, default=1)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('--threads must be positive')
    predict_file(args.data, args.model_dir, args.output, args.method, args.threads)


if __name__ == '__main__':
    main()
