# Reproducible parsing and cleaning

`clean_data.py` is the standalone preprocessing entry point. It contains the raw JSON parser and the cleaned-data loader used by training
and prediction. `features.py` contains feature construction only. It works without a
trained model and does not learn anything from the data or labels.

## Setup

Use Python 3.11 and the pinned project dependencies. In the existing workspace:

```bash
source .venv311/bin/activate
python clean_data.py --help
```

For a fresh checkout with Python 3.11 and venv support:

```bash
python3.11 -m venv .venv311
source .venv311/bin/activate
python -m pip install -r requirements.txt
```

Cleaning requires NumPy and Python's standard library; PyTorch is not required.
The standalone cleaner no longer imports `features.py`. Use the same source version,
Python, NumPy, and compression-library version when comparing output hashes.

## Input format

Signal input is the assignment's JSON Lines format, optionally gzip compressed.
Each transcript/position maps to one seven-base sequence and a nonempty N x 9
read matrix. The nine columns are dwell time, current standard deviation, and
mean current at -1, central, and +1 positions, in that order. This expects
already processed nanopore features, not raw instrument signal files.

The optional label file is comma-separated, regardless of its extension. It must
contain `transcript_id,transcript_position,label`. `gene_id` and `n_reads` are
optional. If supplied, they are validated. Extra fields, including `start` and
`end`, are listed as dropped columns in the audit. Original input files remain
available with their full metadata.

## Run on the supplied datasets

Choose a new output directory for each run; the script refuses to overwrite an
existing directory.

```bash
python clean_data.py --data data/dataset0.json.gz --labels data/data.info.labelled --out-dir data/cleaned/dataset0
python clean_data.py --data data/dataset1.json.gz --labels data/dataset1.info --out-dir data/cleaned/dataset1
python clean_data.py --data data/dataset2.json.gz --labels data/dataset2.info --out-dir data/cleaned/dataset2
```

By default, labels are converted using `label = int(original_label > 0)`:
zero stays 0, and any positive fraction becomes 1. The original numeric value
is preserved in `modification_fraction`. Already binary labels stay unchanged.
This is equivalent to explicitly setting `--label-mode fractional-binary`.

Use `--label-mode fractional` to retain fractional values in `label` without
conversion, or `--label-mode binary` to require that input labels are already 0/1.
Raw source files are never changed. With no label file, only signals are cleaned.
Dataset2 evaluation uses the binary `label` for all 1,323 sites (1,134 positive,
189 negative), and groups score summaries by `modification_fraction`.
Training on dataset2 would still require gene annotations; none are invented.

For inference data with no labels:

```bash
python clean_data.py --data path/to/unseen.json.gz --out-dir data/cleaned/unseen
```

## Cleaning rules

| Rule | Default behavior |
|---|---|
| Blank JSON lines | Skip |
| Sequence notation | Uppercase and convert U to T |
| Site coordinates | Preserve exact strings, including transcript version suffixes and leading zeros in positions |
| Coordinate validity | Nonempty transcript IDs without surrounding whitespace; nonnegative digit-only positions; no offset conversion |
| Signal matrix | Require at least one read, nine measurements per read, and finite numeric values |
| Dwell time and current SD | Reject negative values; zero is allowed |
| Mean current | No arbitrary threshold or outlier cutoff |
| Sequence | Require exactly seven A/C/G/T bases; no additional DRACH filter |
| Duplicates | Reject duplicate JSON keys and repeated site coordinates; reject duplicate label coordinates even if labels agree |
| Invalid records | Stop with an error; do not silently skip or repair |
| Missing labels | Keep valid signal sites by default; output labels only for exact matches |
| Labels missing signal | Log in `exclusions.csv`; omit from output labels |
| Read counts | Recompute from JSON and require agreement with `n_reads` if supplied |
| Original JSON offsets | Drop `start`/`end`; they are invalid after rewriting |
| Missing values and outliers | No imputation or outlier removal |
| Scaling or subsampling | None; neural standardization and MIL sampling remain model-specific operations |
| Cross-dataset shared coordinates | Retain; do not treat independent cell lines as duplicate samples |

No reads are dropped from a retained site, and signal values are preserved as
parsed floating-point values. Textual numeric formatting may change (for example,
`100` may be written as `100.0`). Data and labels remain in signal traversal order;
the script does not globally sort sites or reorder reads.

## Explicit optional filters

`--min-reads N` excludes whole sites with fewer than N reads. Default is 1, so
all nonempty valid sites pass. This is not MIL's maximum-read sampling limit.

`--unlabeled keep|drop|error` controls sites without an exact matching label:

- `keep` retains their signals and records `has_label=0` in the manifest.
- `drop` excludes them and records the reason.
- `error` stops the run if a signal site lacks a label.

Both `drop` and `error` require `--labels`. When minimum coverage and a missing
label both apply under `drop`, the exclusion reason is `below_min_reads`.
Missing labels are never converted to negatives. Under default conversion, a positive fraction becomes 1; only zero becomes 0.

## Outputs

| File | Contents |
|---|---|
| `data.json.gz` | Standardized signal JSONL; one site per line |
| `labels.csv` | Matched, retained labels with recomputed read counts; optional `gene_id`; by default `label` is binary and `modification_fraction` preserves the source value |
| `site_manifest.csv` | Every retained coordinate, read count, normalized sequence, and label-presence indicator |
| `exclusions.csv` | Excluded coordinates and reasons, including labels with no corresponding signal |
| `audit.json` | Input/output counts, label counts, options, file SHA-256 hashes, source-code hashes, software versions, and documented rules |

`labels.csv` is omitted for unlabeled inference input. `exclusions.csv` still
contains a header when there are no exclusions. The outputs are not an m6Anet
index: this script does not recreate byte-offset metadata for m6Anet inference.

The original files are never changed. Output is built in a temporary sibling
directory and published only after successful validation. On a handled failure,
no final output directory is created. A hard process termination can leave a
hidden staging directory; it is not a completed dataset.

## Reproducibility checks

The gzip timestamp is fixed at zero and no output filename is embedded. With
identical inputs, options, source code, and environment, output files have
identical bytes even when the output directory changes. Audit provenance records
absolute input paths, so moving inputs changes the audit even if their content
hashes match. Reordering source records can also change output ordering/hashes.
Input hashes are checked again before publishing to detect concurrent changes.

Run the regression checks:

```bash
python check_cleaning.py
```

Checks use temporary fixtures, deleted when finished. They verify repeat-run
byte identity, plain/gzip equivalence, coordinate and feature preservation,
fractional-label handling, explicit filtering, duplicate rejection, and failure
without overwriting existing output. No demo data or demo models are retained.

To repeat an actual run, choose a different output directory and compare the
`output_sha256` entries in both audit files.

## Use with training and prediction

```bash
python train.py --model-type gb --clean-dir data/cleaned/dataset0 --out-dir results/gb_from_cleaned
python predict.py --clean-dir data/cleaned/dataset1 --model results/gb/model.joblib --output results/dataset1/gb_cleaned_predictions.csv
```

Training still requires binary labels and `gene_id` for gene-separated splitting.
The dataset1/dataset2 label files do not supply gene IDs; this script does not
invent them. Their cleaned data can be used for prediction and sample-matched
label evaluation. No retraining is needed simply because files were standardized
with the default rules and their measurements/features were preserved.

## Verified run on the supplied files

| Dataset | Sites retained | Reads retained | Excluded sites |
|---|---:|---:|---:|
| dataset0 | 121,838 | 11,027,106 | 0 |
| dataset1 | 90,810 | 7,907,952 | 0 |
| dataset2 | 1,323 | 1,171,940 | 0 |

All sites matched their own labels. No sites or reads were excluded under the
default rules. Dataset2 retains all 945 fractional labels. No models were
retrained and no existing predictions were changed.

## Cleaned inputs are required by the model pipeline

Use `--clean-dir` with a completed cleaning directory. Training loads `labels.csv`
and `data.json.gz` from that directory; prediction needs only the cleaned signals
and audit. Both verify SHA-256 hashes and reject raw/modified files with a clear
message. All JSON reading is implemented in `clean_data.py`: `raw_sites()` for
cleaning and `cleaned_sites()` for reading its canonical output. The latter does
not repeat sequence normalization or filtering. JSON serialization still requires
reading JSON at model runtime; the parsing code is centralized in the cleaner.

Training retains model-specific target/gene checks. Evaluation now also reads
cleaned labels for both dataset1 and dataset2. Already generated cleaned files
remain compatible; their audit records correctly describe the producer version
used at creation rather than being relabeled as outputs of a newer version.
