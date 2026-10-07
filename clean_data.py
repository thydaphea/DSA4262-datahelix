"""Standardize assignment JSONL and labels with deterministic output and provenance."""
import argparse
from contextlib import contextmanager
import csv
import gzip
import hashlib
import io
import json
import math
import os
import platform
import re
import shutil
import tempfile
import zlib
from pathlib import Path
import numpy as np

VERSION = '2.2.0'
KEY = ['transcript_id', 'transcript_position']


def unique_json_object(pairs):
    """Reject duplicate JSON keys rather than silently retaining the last value."""
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'Duplicate JSON key: {key}')
        result[key] = value
    return result


def raw_sites(path):
    seen = set()
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt', encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line, object_pairs_hook=unique_json_object)
            if not isinstance(record, dict) or not record:
                raise ValueError(f'Line {line_number}: expected a nonempty transcript object')
            for transcript, positions in record.items():
                if not isinstance(positions, dict) or not positions:
                    raise ValueError(f'Line {line_number}: expected a nonempty position object')
                for position, motifs in positions.items():
                    key = (transcript, str(position))
                    if key in seen:
                        raise ValueError(f'Duplicate site: {key}')
                    seen.add(key)
                    if not isinstance(motifs, dict) or len(motifs) != 1:
                        raise ValueError(f'Expected one 7-mer at {key}')
                    motif, reads = next(iter(motifs.items()))
                    motif = motif.upper().replace('U', 'T')
                    if len(motif) != 7 or set(motif) - set('ACGT'):
                        raise ValueError(f'Invalid 7-mer at {key}: {motif}')
                    a = np.asarray(reads, dtype=float)
                    if a.ndim != 2 or a.shape[0] == 0 or a.shape[1] != 9:
                        raise ValueError(f'Line {line_number}: expected nonempty N x 9 reads at {key}')
                    if not np.isfinite(a).all():
                        raise ValueError(f'Nonfinite signal at {key}')
                    yield key, motif, a


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verified_cleaned_file(directory, filename):
    """Load only cleaner-produced artifacts; detect edits before model execution."""
    directory = Path(directory).resolve()
    audit_path = directory / 'audit.json'
    if not audit_path.is_file():
        raise ValueError(f'{directory}: missing audit.json; run clean_data.py first')
    audit = json.loads(audit_path.read_text(encoding='utf-8'))
    path = directory / filename
    expected = audit.get('output_sha256', {}).get(filename)
    if expected is None or not path.is_file():
        raise ValueError(f'{filename} is not a recorded cleaned artifact in {directory}')
    if sha256(path) != expected:
        raise ValueError(f'Cleaned file hash mismatch: {path}; regenerate with clean_data.py')
    return path


def cleaned_input_paths(clean_dir=None, data=None, labels=None, require_labels=False):
    """Resolve CLI inputs; --data compatibility accepts only audited cleaned files."""
    if clean_dir is not None:
        if data is not None or labels is not None:
            raise ValueError('Use --clean-dir by itself, not together with --data or --labels')
        directory = Path(clean_dir).resolve()
    else:
        if data is None:
            raise ValueError('Supply --clean-dir with output from clean_data.py')
        path = Path(data).resolve()
        if path.name != 'data.json.gz':
            raise ValueError('--data must be the standardized data.json.gz; run clean_data.py first')
        directory = path.parent
        if labels is not None and Path(labels).resolve() != directory / 'labels.csv':
            raise ValueError('Labels must be labels.csv from the same cleaned directory')
    data_path = verified_cleaned_file(directory, 'data.json.gz')
    label_path = verified_cleaned_file(directory, 'labels.csv') if require_labels else None
    return data_path, label_path



@contextmanager
def model_inputs(clean_dir=None, data=None, labels=None, require_labels=False):
    """Accept raw assignment inputs or audited cleaned inputs for the two entry points."""
    audited = data is not None and Path(data).name == 'data.json.gz' and (Path(data).parent/'audit.json').is_file()
    if clean_dir is not None or audited:
        signal_path, label_path = cleaned_input_paths(clean_dir, data, labels, require_labels)
        yield signal_path, label_path, json.loads((signal_path.parent/'audit.json').read_text())
        return
    if data is None:
        raise ValueError('Supply --data with raw JSON/JSON.gz, or --clean-dir')
    if require_labels and labels is None:
        raise ValueError('Training on raw JSON requires --labels')
    # Keep preprocessing internal; classmates need not run a third script.
    with tempfile.TemporaryDirectory(prefix='m6a-input-') as temporary:
        directory = Path(temporary)/'cleaned'
        run(argparse.Namespace(data=data, labels=labels, out_dir=str(directory),
                               label_mode='fractional-binary', min_reads=1, unlabeled='keep'))
        signal_path, label_path = cleaned_input_paths(directory, require_labels=require_labels)
        yield signal_path, label_path, json.loads((directory/'audit.json').read_text())


def cleaned_sites(path):
    """Read canonical cleaned JSON; normalization/cleaning happens only in run()."""
    path = Path(path).resolve()
    if path.name != 'data.json.gz':
        raise ValueError('Expected cleaner output data.json.gz')
    verified_cleaned_file(path.parent, path.name)
    with gzip.open(path, 'rt', encoding='utf-8') as handle:
        for line in handle:
            record = json.loads(line)
            transcript, positions = next(iter(record.items()))
            position, motifs = next(iter(positions.items()))
            motif, reads = next(iter(motifs.items()))
            yield (transcript, position), motif, np.asarray(reads, dtype=float)


def load_labels(path, mode):
    """Keep exact coordinate strings. Extra indexing fields are not propagated."""
    lookup, transcript_genes = {}, {}
    with open(path, encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        if len(columns) != len(set(columns)) or not set(KEY + ['label']).issubset(columns):
            raise ValueError('Labels need unique columns including transcript_id, transcript_position, label')
        has_gene = 'gene_id' in columns
        for number, row in enumerate(reader, 2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f'Label row {number}: inconsistent number of columns')
            key = (row['transcript_id'], row['transcript_position'])
            if not key[0] or key[0] != key[0].strip() or not re.fullmatch(r'[0-9]+', key[1]):
                raise ValueError(f'Label row {number}: invalid coordinate {key}; IDs are not silently stripped')
            if key in lookup:
                raise ValueError(f'Duplicate label coordinate: {key}')
            label = float(row['label'])
            if not math.isfinite(label) or not 0 <= label <= 1:
                raise ValueError(f'Label row {number}: label must be finite in [0,1]')
            if mode == 'binary' and label not in (0., 1.):
                raise ValueError(f'Label row {number}: fractional label; explicitly use --label-mode fractional to preserve it')
            gene = row.get('gene_id')
            if has_gene:
                if not gene or gene != gene.strip():
                    raise ValueError(f'Label row {number}: missing or whitespace-padded gene_id')
                if key[0] in transcript_genes and transcript_genes[key[0]] != gene:
                    raise ValueError(f'Transcript maps to different genes: {key[0]}')
                transcript_genes[key[0]] = gene
            n_reads = None
            if 'n_reads' in columns:
                if not re.fullmatch(r'[0-9]+', row['n_reads']):
                    raise ValueError(f'Label row {number}: n_reads must be a nonnegative integer')
                n_reads = int(row['n_reads'])
            lookup[key] = dict(label=label, gene_id=gene, n_reads=n_reads)
    if not lookup:
        raise ValueError('Label file contains no sites')
    return lookup, has_gene, columns


def run(args):
    source = Path(args.data).resolve()
    label_path = Path(args.labels).resolve() if args.labels else None
    target = Path(args.out_dir).resolve()
    if target.exists():
        raise FileExistsError(f'Output directory already exists: {target}; choose a new directory')
    if args.min_reads < 1:
        raise ValueError('--min-reads must be positive')
    if args.unlabeled != 'keep' and label_path is None:
        raise ValueError('--unlabeled drop/error requires --labels')
    lookup, has_gene, columns = load_labels(label_path, args.label_mode) if label_path else ({}, False, [])
    sources = {'data': dict(path=str(source), sha256=sha256(source))}
    if label_path:
        sources['labels'] = dict(path=str(label_path), sha256=sha256(label_path))
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.' + target.name + '-', dir=target.parent))
    counts = dict(input_sites=0, output_sites=0, input_reads=0, output_reads=0,
                  below_min_reads=0, unlabeled_sites=0, dropped_unlabeled_sites=0,
                  labels_without_signal=0, output_labeled_sites=0, output_fractional_labels=0, retained_fractional_source_labels=0)
    seen = set()
    output_label_counts = {}
    binary_label_counts = {}
    try:
        # Fixed gzip timestamp and empty embedded filename produce reproducible bytes.
        with open(staging / 'data.json.gz', 'wb') as raw, \
             gzip.GzipFile(fileobj=raw, mode='wb', filename='', mtime=0, compresslevel=1) as compressed, \
             io.TextIOWrapper(compressed, encoding='utf-8', newline='\n') as output, \
             open(staging / 'site_manifest.csv', 'w', encoding='utf-8', newline='') as manifest_handle, \
             open(staging / 'exclusions.csv', 'w', encoding='utf-8', newline='') as exclusions_handle:
            manifest = csv.writer(manifest_handle, lineterminator='\n')
            manifest.writerow(KEY + ['n_reads', 'sequence', 'has_label'])
            exclusions = csv.writer(exclusions_handle, lineterminator='\n')
            exclusions.writerow(KEY + ['reason'])
            label_handle = open(staging / 'labels.csv', 'w', encoding='utf-8', newline='') if label_path else None
            try:
                if label_handle:
                    writer = csv.writer(label_handle, lineterminator='\n')
                    writer.writerow((['gene_id'] if has_gene else []) + KEY + ['label', 'n_reads'] + (['modification_fraction'] if args.label_mode == 'fractional-binary' else []))
                for key, motif, reads in raw_sites(source):
                    if not key[0] or key[0] != key[0].strip() or not re.fullmatch(r'[0-9]+', key[1]):
                        raise ValueError(f'Invalid signal coordinate: {key}')
                    if (reads[:, [0, 1, 3, 4, 6, 7]] < 0).any():
                        raise ValueError(f'Negative dwell time or signal SD at {key}')
                    seen.add(key)
                    n_reads = len(reads)
                    label = lookup.get(key)
                    counts['input_sites'] += 1
                    counts['input_reads'] += n_reads
                    if label and label['n_reads'] is not None and label['n_reads'] != n_reads:
                        raise ValueError(f'n_reads mismatch at {key}: labels={label["n_reads"]}, JSON={n_reads}')
                    if label_path and label is None:
                        counts['unlabeled_sites'] += 1
                        if args.unlabeled == 'error':
                            raise ValueError(f'No matching label for {key}')
                    if n_reads < args.min_reads:
                        counts['below_min_reads'] += 1
                        exclusions.writerow([*key, 'below_min_reads'])
                        continue
                    if label_path and label is None and args.unlabeled == 'drop':
                        counts['dropped_unlabeled_sites'] += 1
                        exclusions.writerow([*key, 'missing_label'])
                        continue
                    output.write(json.dumps({key[0]: {key[1]: {motif: reads.tolist()}}},
                                            separators=(',', ':'), ensure_ascii=False, allow_nan=False) + '\n')
                    manifest.writerow([*key, n_reads, motif, int(label is not None)])
                    counts['output_sites'] += 1
                    counts['output_reads'] += n_reads
                    if label is not None:
                        value = label['label']
                        output_value = int(value > 0) if args.label_mode == 'fractional-binary' else value
                        text = str(int(output_value)) if output_value in (0., 1.) else repr(output_value)
                        writer.writerow(([label['gene_id']] if has_gene else []) + [*key, text, n_reads] + ([repr(value)] if args.label_mode == 'fractional-binary' else []))
                        if args.label_mode == 'fractional-binary':
                            binary_value = str(int(value > 0))
                            binary_label_counts[binary_value] = binary_label_counts.get(binary_value, 0) + 1
                        counts['output_labeled_sites'] += 1
                        counts['output_fractional_labels'] += int(output_value not in (0., 1.))
                        counts['retained_fractional_source_labels'] += int(value not in (0., 1.))
                        output_label_counts[text] = output_label_counts.get(text, 0) + 1
                    if counts['input_sites'] % 25000 == 0:
                        print(f'Processed {counts["input_sites"]:,} sites', flush=True)
                for key in sorted(set(lookup) - seen):
                    counts['labels_without_signal'] += 1
                    exclusions.writerow([*key, 'label_without_signal'])
            finally:
                if label_handle:
                    label_handle.close()
        if counts['output_sites'] == 0:
            raise ValueError('No sites retained; no cleaned output published')
        # Detect input changes during processing before publishing a successful run.
        for name, entry in sources.items():
            if sha256(entry['path']) != entry['sha256']:
                raise ValueError(f'Input changed during processing: {name}')
        report = dict(cleaner_version=VERSION, inputs=sources,
            options=dict(label_mode=args.label_mode, min_reads=args.min_reads, unlabeled=args.unlabeled),
            counts=counts, output_label_counts=output_label_counts,
            binary_label_counts=binary_label_counts,
            binary_label_rule="label = int(modification_fraction > 0)" if args.label_mode == "fractional-binary" and label_path else None,
            environment=dict(python=platform.python_version(), numpy=np.__version__, zlib=zlib.ZLIB_RUNTIME_VERSION),
            code_sha256={name: sha256(Path(__file__).resolve().parent / name) for name in ('clean_data.py',)},
            dropped_label_columns=[c for c in columns if c not in KEY + ['gene_id','label','n_reads']],
            output_sha256={p.name: sha256(p) for p in sorted(staging.iterdir())},
            rules=['Input order and coordinate strings are preserved; no coordinate conversion or ID suffix stripping.',
                   'Blank lines are skipped; sequence is uppercased and U converted to T.',
                   'Malformed records, duplicate sites, invalid labels, and nonfinite signals stop the run.',
                   'Dwell times and signal SDs must be nonnegative; no current-intensity cutoff is applied.',
                   'No imputation, outlier removal, signal scaling, or read subsampling is performed.',
                   'n_reads is recomputed from JSON and checked against supplied label counts.',
                   'Fractional mode preserves label; fractional-binary sets label = int(modification_fraction > 0) and retains the source fraction in modification_fraction.',
                   'start/end index fields are not copied; rewritten JSON invalidates original offsets.',
                   'No cross-dataset coordinate filtering is applied.'])
        (staging / 'audit.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n', encoding='utf-8')
        if target.exists():
            raise FileExistsError(f'Output directory appeared during processing: {target}')
        os.rename(staging, target)
        print(json.dumps(counts, indent=2))
        print(f'Saved standardized files to {target}')
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, help='Assignment JSON Lines, optionally gzip compressed')
    parser.add_argument('--labels', help='CSV labels, including files named .info')
    parser.add_argument('--out-dir', required=True, help='New directory; existing paths are never overwritten')
    parser.add_argument('--label-mode', choices=['binary','fractional','fractional-binary'], default='fractional-binary',
                        help='Default: convert label > 0 to 1 and preserve modification_fraction; fractional preserves labels, binary validates 0/1')
    parser.add_argument('--min-reads', type=int, default=1)
    parser.add_argument('--unlabeled', choices=['keep','drop','error'], default='keep')
    args = parser.parse_args()
    try:
        run(args)
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(2, f'Cleaning failed: {error}\n')


if __name__ == '__main__':
    main()
