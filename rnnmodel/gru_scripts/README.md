# GRU training and evaluation scripts

Two Python entry points, with a shared `gru_common.py` module. No notebook or prior
tuning output is required. Keep these three `.py` files together.

## Quick start

Run from this folder. Use the existing `m6a-gru` environment, or install the listed
dependencies into your preferred Python 3.10+ environment:

```bash
python -m pip install -r requirements.txt
python train_gru.py --data-config datasets.example.json --output-dir gru_round5_model
python evaluate_gru.py --data-config datasets.example.json --model-dir gru_round5_model --output-dir gru_round5_evaluation
```

Edit `datasets.example.json` for your file paths. Paths in that JSON are relative
to the JSON's own folder; absolute paths also work. The supplied example points
to the three cleaned train/test datasets used in rounds 1–5 in this workspace.
On Windows, use forward slashes in JSON paths, e.g. `D:/data/clean_train.csv.gz`.
Remove dataset entries if you only want to train on a subset. Training pools the
listed training datasets into one model. Both scripts accept the same config.
Output directories must be new to prevent accidental overwrites.

## Selected configuration and training duration

| Setting | Value |
|---|---:|
| GRU hidden size | 32 |
| GRU layers / direction | 1 / unidirectional |
| Head dropout | 0.0 |
| Learning rate | 0.0003 |
| Optimizer | AdamW |
| Weight decay | 0.0001 |
| Batch size | 256 |
| Class-weight alpha | 0.0 |
| Gradient norm clipping | 1.0 |
| Seed | 42 |
| CV maximum epochs | 50 |
| CV early-stopping patience / minimum delta | 5 / 0.001 |
| Default full-data refit epochs | 32 |

Positive weight is `(n_negative / n_positive) ** alpha`; alpha 0 means weight 1,
so this is ordinary binary cross-entropy despite the saved class-weighting flag.
The architecture has 6,721 trainable parameters, matching round 5.

The default `--mode fixed` fits **all** cleaned training rows for **32 epochs**,
the round-5 median best epoch. There is no early stopping in this mode because
there is no held-out validation partition. The 50-epoch setting is the CV limit,
not a command to train on the test set until its score improves. `--epochs N`
overrides the refit duration, and is recorded in the run manifest.

To recalculate the duration from five grouped validation folds, then refit on
all training data for the rounded median best epoch:

```bash
python train_gru.py --data-config datasets.example.json --output-dir gru_round5_cv_model --mode cv-refit
```

Each fold fits its own preprocessing and selects its checkpoint by trapezoidal
validation PR-AUC. Patience resets only when the score exceeds its reference by
more than 0.001; the best checkpoint is saved even for smaller improvements,
matching round 5. CV uses seed 42 with per-fold training seed `42 + fold`.
New folds are stratified by dataset and label and grouped by biological identity.
To reuse the original tuning folds, add:

```bash
--fold-assignments ../../../outputs/gru_class_weight_train_test/20261014T081820_206913Z/cv_fold_assignments.csv.gz
```

These assignments must cover exactly the supplied training identities/labels.
CV estimates also select the best checkpoint, so they are not an independent
performance estimate. Use the separate test evaluation for that purpose.
Results can vary across PyTorch versions/devices despite fixed random seeds.

## Input contract and split checks

Inputs are the cleaner's site-level `.csv` or `.csv.gz` outputs, not raw read JSON.
Required metadata: `dataset_id`, `transcript_id`, `transcript_position`, `sequence`,
`split`, `group_id`, `group_source`, `gene_id` (the gene values may be blank).
Training additionally requires binary `label`, `n_reads_valid`, and 45 summaries:

```
{minus1,center,plus1}_{dwell,signal_sd,signal_mean}_{mean,std,median,q25,q75}
```

Evaluation test data needs the same features; `label` may be absent or partly
missing. All test sites receive scores, but only known binary labels enter metrics.
Sequences must be seven A/C/G/T bases with a central DRACH motif. Features must
be finite, coverage at least one, and site keys unique within each dataset.
Fractional modification measurements are not binary labels and are rejected.

The exact feature order is stored with preprocessing. Three positions × 15
signal summaries enter the GRU. Its last hidden state is concatenated with 28
sequence one-hot features and standardized log(1 + read count), then passed to
a 32-unit ReLU head and one output logit. Standardization is fitted on training
rows only and saved as JSON means/scales; evaluation never refits it.

The train/test audit rejects shared sites and biological groups across all
datasets, using connected transcript, gene, and cleaner group identities. The
example preserves the **previously confirmed** dataset2 mixture-position policy:
`tx_id_0` through `tx_id_6` share position groups instead of being treated as seven
biological transcripts. The code validates the expected group IDs, absence of
gene annotations, and sequence consistency. Set this flag to false for unrelated
data unless that interpretation is established; do not use it to bypass overlap.

Training reads test identifiers/group metadata for separation checks, but does
not load test labels or signal features. Evaluation requires the exact training
files used to fit the saved model (checked by SHA-256) and checks separation again.
It may evaluate replacement test files with matching dataset names; this is
recorded in `evaluation_manifest.json`.

## Outputs

Training writes:

- `model.pt`: portable CPU state dictionary and model configuration.
- `preprocessing.json`: fitted scalers and feature order.
- `last_checkpoint.pt`: final epoch weights and optimizer state for inspection;
  no automatic resume facility is provided.
- `selected_config.json`, `fit_summary.json`, `training_history.csv`,
  `learning_curves.png`, `split_audit.json`, and resolved data config.
- `run_manifest.json`: completion state, input/model hashes, seed, arguments,
  actual refit duration, package versions, device, and timing.
- `source/`: a snapshot of the scripts and requirements used for the run.
- `train_outputs/`: training predictions, detailed labelled predictions and plots.
- `performance_summary.csv`, `.json`, and `.txt`.
- With `cv-refit`: `cv/` contains fold models/scalers/checkpoints/histories,
  fold assignments, fold metrics, mean/SD fold AUC, validation predictions and plots.

Evaluation writes:

- `predictions_test_<dataset_id>.csv` and
  `predictions_train_apparent_<dataset_id>.csv` with **exactly three columns**:
  `transcript_id,transcript_position,score`. Score is sigmoid probability of label 1.
- `detailed_predictions_<split>.csv.gz` with dataset ID, keys, true label, score,
  and predicted label for inspecting errors. Separate per-dataset three-column
  files avoid ambiguous identities when the same site occurs in multiple datasets.
- `performance_summary.csv`, `.json`, and `.txt`: per-dataset and pooled counts,
  ROC-AUC, trapezoidal PR-AUC, average precision, precision, recall, specificity,
  F1, accuracy, log loss, Brier score, and TN/FP/FN/TP.
- `evaluation_<split>_<dataset_id>.png` and pooled plots: ROC, precision–recall
  with prevalence baseline, confusion matrix, and score distributions.
- `evaluation_manifest.json` and `split_audit.json` for provenance.

The threshold defaults to 0.5 and can be supplied with `--threshold`; select it
using training/validation data before looking at test outcomes. AUC metrics use
continuous scores. Single-class inputs report unavailable AUC/AP as null/blank,
and fully unlabelled tests produce predictions with `NO_LABELS` status.
Pooled results count each dataset observation and are not an average of AUCs.
Training performance is explicitly labelled `train_apparent`.

These scripts report point estimates. They do not calculate confidence intervals
or automatically tune thresholds, calibrate probabilities, or change the selected
hyperparameters. Repeated evaluation should not be used to retune on the test set.

## Verification

```bash
python -m unittest discover -s tests -v
```

The tests exercise preprocessing, group leakage checks, early stopping, invalid
inputs, metric edge cases, and training/evaluation artifact round trips on small
synthetic data. A short smoke run validates the pipeline, not predictive quality.
