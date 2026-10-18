# DataHelix: m6A site prediction

Predict whether an RNA site contains m6A modification from m6Anet-processed direct RNA-sequencing reads. One shared set of GB, MLP and GRU models serves all datasets; the default output combines their probabilities with a logistic ensemble.

- **`train.py`**: clean and join reads/labels, preserve supplied splits, generate five-fold out-of-fold (OOF) predictions, fit final models and ensembles, and evaluate held-out sites.
- **`predict.py`**: clean unlabeled JSON and produce site probabilities using saved models. It also contains the shared model definitions and preprocessing.

## Table of contents

- [Install on AWS Ubuntu](#install-on-aws-ubuntu)
- [Run the included unlabeled example](#run-the-included-unlabeled-example)
- [Inputs, execution and outputs](#inputs-execution-and-outputs)
- [Cleaning and models](#cleaning-and-models)
- [All command-line arguments](#all-command-line-arguments)
- [Interpret evaluation results](#interpret-evaluation-results)
- [Software license](#software-license)
- [How to cite](#how-to-cite)

## Install on AWS Ubuntu

**Requirements:** 64-bit x86 Ubuntu (22.04/24.04), Python 3.11, CPU and internet access for installation. No GPU is required. For full-data training, a suggested starting allocation is 2 vCPUs, 8 GB RAM and 5 GB free disk; these are planning estimates, not measured minimum requirements.

1. Copy/extract the supplied `final_models` directory to `~/final/final_models`, including `models/` and `examples/`. This is a script-based package; no build step is needed.
2. Install prerequisites and create an isolated Python environment:

```bash
sudo apt-get update
sudo apt-get install -y curl ca-certificates
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

cd ~/final/final_models
uv python install 3.11
uv venv --python 3.11 .venv
source .venv/bin/activate
```

3. Install CPU PyTorch and the pinned dependencies:

```bash
uv pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
uv pip install -r requirements.txt
```

These steps follow the [uv installation](https://docs.astral.sh/uv/getting-started/installation/), [Python installation](https://docs.astral.sh/uv/guides/install-python/) and [PyTorch CPU installation](https://pytorch.org/get-started/previous-versions/) instructions. On later logins, change to the package directory and activate `.venv` again.

For the existing setup on this machine, installation is already complete. At the start of each new terminal session, run:

```bash
cd /home/ubuntu/final/final_models
source .venv/bin/activate
```

This activates the package's own environment (including NumPy 2.2.6). Its Python interpreter and installed dependencies are separate from `/home/ubuntu/project`. Activation selects the interpreter; it does not retrain or modify models. The independent environment reproduced all five included example predictions exactly. Full results can still vary slightly across library versions or hardware, so use the pinned requirements. The `/home/ubuntu/final` Git checkout is also independent of `/home/ubuntu/project`, while retaining the same GitHub remote and branch `final`.

Saved scikit-learn artifacts record a newer version than the pinned runtime and may issue version warnings. In the tested runtime, all 41,697 saved test predictions were reproduced within numerical precision; see [verification.json](verification.json). Use the pinned dependencies rather than assuming arbitrary versions are compatible.

## Run the included unlabeled example

[examples/data.json](examples/data.json) contains **five real sites, 53,062 bytes**, copied from the supplied dataset0 read data. It contains transcript coordinates, sequences and read measurements, **no ground-truth labels**. See [provenance](examples/provenance.json). It is suitable for a repository example under 1 MB; it is not an independent accuracy benchmark.

After installation, run:

```bash
python predict.py --data examples/data.json --method ensemble --output example_results/predictions.csv
head -n 6 example_results/predictions.csv
```

Expect a header and five prediction rows. [examples/expected_predictions.csv](examples/expected_predictions.csv) contains reference model outputs, not labels; small floating-point differences are possible. Choose a new output path for subsequent runs because existing predictions and their audits are not overwritten. The example files have been prepared locally; no GitHub upload is performed by these commands.

## Inputs, execution and outputs

### Prediction

Input is m6Anet JSONL, optionally gzip-compressed (`.json` or `.json.gz`). Each record maps:

```text
transcript_id → transcript_position → seven-base sequence → N reads × 9 measurements
```

The nine measurements are dwell time, signal SD and signal mean at each of three neighbouring positions. Prediction needs **no labels, separate n_reads column, gene annotations or split file**; valid coverage is calculated from the reads.

Run these ensemble prediction commands from `~/final/final_models` with your environment activated:

```bash
# Existing dataset: logistic ensemble
python predict.py --data raw_datasets/dataset0.json.gz --method ensemble --output predictions/logistic_dataset0.csv

# Existing dataset: weighted-average ensemble
python predict.py --data raw_datasets/dataset0.json.gz --method weighted --output predictions/weighted_dataset0.csv

# New dataset: logistic ensemble
python predict.py --data /path/to/new_data.json.gz --method ensemble --output predictions/new_logistic.csv

# New dataset: weighted-average ensemble
python predict.py --data /path/to/new_data.json.gz --method weighted --output predictions/new_weighted.csv
```

For dataset1 or dataset2, change both the input and output filenames. Both methods reuse the saved models in `models/` without retraining. To use a completed new training run, add `--model-dir results/new_training/models`.

Output is exactly three comma-separated columns:

```csv
transcript_id,transcript_position,score
```

Each retained site receives a probability between 0 and 1. Larger scores indicate stronger predicted evidence of m6A; scores are not modification fractions or per-read predictions. No hard classification threshold is applied. A neighbouring `.audit.json` records cleaning counts and input/output hashes.


### Training and labelled evaluation

Edit [training_inputs.json](training_inputs.json) to specify each dataset's raw JSON and matching label file, plus `splits.csv.gz`. Paths resolve relative to the configuration file. The supplied paths point to `raw_datasets/`, containing the three original compressed JSON datasets and their matching label files on this machine. These raw files are not included in the environment-update commit; teammates must obtain them separately and place them at those paths.

Labels require `transcript_id,transcript_position,label`; `gene_id` is optional. Source fractions in [0,1] become binary labels: **0 → 0; any value > 0 → 1**. Each sample is joined only to its own labels. The split manifest requires `dataset_id,transcript_id,transcript_position,original_split,label,group_id,fold`; development folds are 1–5 and test folds are blank.

```bash
python train.py --config training_inputs.json --out-dir results/new_training
```

This fits all three base models and both ensembles. The new directory contains the joined cleaned table, fold checkpoints, OOF predictions, final `models/`, test predictions, metrics and provenance. The small unlabeled example is for prediction only; training requires the full labelled development inputs and matching assignments.

## Cleaning and models

One standard cleaner normalizes sequences (uppercase, U→T), retains seven-base A/C/G/T sequences with a central DRACH motif, removes invalid reads, and requires at least one valid read. Invalid reads include malformed/nonnumeric/nonfinite measurements or negative dwell time/signal SD. Ambiguous JSON structures and duplicate sites raise errors.

The [cleaned site table](cleaned/sites.csv.gz) has **213,971 rows and 63 columns**: identifiers, sequence, 45 signal summaries, raw/valid/removed read counts, cleaning indicator, label/source fraction and grouping/split metadata. These stored columns are not all model inputs. During training and labelled evaluation, matching uses dataset ID plus both coordinates. Assigned sites must survive cleaning and match saved labels; otherwise processing stops. Unassigned extra sites are audited and excluded from development. Current data retained all assigned sites, with no extras.

### GB and MLP

Both pool all valid reads at each site into **74 features**:

| Features | Count |
|---|---:|
| Mean and population SD of each of nine measurements | 18 |
| 25th percentile, median and 75th percentile of each measurement | 27 |
| log(1 + valid read count) | 1 |
| Seven sequence positions × four A/C/G/T indicators | 28 |
| **Total** | **74** |

For example, 100 reads become one 74-value site vector. The pooled statistics include the mean and SD **across reads of the original signal-SD measurements**. Sequence encoding uses A=`[1,0,0,0]`, C=`[0,1,0,0]`, G=`[0,0,1,0]`, T=`[0,0,0,1]`. Identifiers, labels and split metadata are never classifier inputs.

| Model | Method and fixed configuration |
|---|---|
| GB | `HistGradientBoostingClassifier`: sequential decision trees; 600 iterations, learning rate 0.03, maximum 63 leaves, minimum 10 samples/leaf, L2 1. No standardization, weighting or early stopping. |
| MLP | Fully connected network: 74 → 256 → 128 → 1, ReLU hidden activations, dropout 0.5 after the first hidden layer. AdamW: learning rate 0.001, decay 0.0001, batch 256; unweighted BCE; exactly 16 epochs. Sigmoid converts the output logit to a probability. |

MLP standardizes all 74 features using fitting-row means and population SDs (scales below `1e-8` become 1). Saved scaling is reused for inference; dropout is disabled. GB uses the unscaled feature vector. Neither model directly predicts individual reads.

**GB tuning:** 32 candidate configurations varied learning rate, boosting iterations, maximum leaves, minimum leaf size and L2 regularization. Selection used pooled validation trapezoidal PR-AUC; candidate 9 won with **0.479539** and the settings above. See [GB search evidence](hyperparameter_tuning/gb/search_summary.csv).

**MLP tuning:** 12 configurations were screened for four epochs, varying hidden layers, dropout, learning rate, AdamW decay, batch size and class weighting. Three finalists were trained for up to 30 epochs with early-stopping patience 6. Candidate 9 at epoch 16 achieved the best pooled validation trapezoidal PR-AUC, **0.492715**. Settings varied together; later controlled experiments are documented separately and did not replace the fixed configuration above. See [MLP search evidence](hyperparameter_tuning/README.md#later-mlp-experiments) and [candidate summary](hyperparameter_tuning/mlp/search_summary.csv). These are historical validation scores, not final test metrics.

Saved base-model files are `models/gb/model.joblib`, `models/mlp/model.joblib` and `models/gru/model.pt`. New training runs use the same layout.

### GRU

The **gated recurrent unit (GRU)** models the ordered signal context around a site. It uses the same 74 values as GB/MLP, arranged into two inputs:

- **Signal input:** three steps (upstream, centre, downstream), each containing 15 values: five read-level summary statistics × three measurements (dwell time, signal SD and signal mean).
- **Auxiliary input:** 28 nucleotide indicators plus log(1 + valid read count), giving 29 values.

A single-layer, unidirectional GRU with hidden size 32 processes the three signal positions. Its final 32-value hidden state is concatenated with the 29 auxiliary values; a fully connected head maps **61 → 32 → 1**, with ReLU and dropout after the 32-unit layer. Sigmoid converts the final logit into one site probability. The recurrent steps are neighbouring signal positions, not individual reads or the seven nucleotide characters.

The rationale is to learn relationships between neighbouring signal positions while retaining sequence context and coverage in the classifier. Pooling reads first gives a fixed-size input for sites with different read counts, although it discards individual-read variation beyond the summary statistics. This architecture provides a different way to use positional information from the MLP's flattened input; it does not guarantee better performance.

| Setting | Selected value |
|---|---|
| GRU hidden size / layers | 32 / 1 |
| Classifier-head dropout | 0 |
| Optimizer | AdamW |
| Learning rate / weight decay | 0.0003 / 0 |
| Batch size / fixed training epochs | 256 / 32 |
| Loss | Unweighted `BCEWithLogitsLoss` (`pos_weight = 1`) |
| Gradient clipping | Maximum gradient norm 1 |

Each of the 45 signal summaries and log coverage is standardized using only fitting rows; nucleotide indicators remain unscaled. Saved means/scales are reused at prediction time. The saved model is `models/gru/model.pt`, with its scaling parameters in `preprocessing.json`.

**Tuning:** the preserved [configuration-selection record](hyperparameter_tuning/gru/configuration_selection.json) specifies selection by mean fold trapezoidal PR-AUC, followed by ROC-AUC and weight-decay tie-breaks. It records a 50-epoch tuning limit, patience 5 and minimum improvement 0.001; the final duration was set to the rounded median best epoch across folds, **32 epochs**. The [final fit summary](models/gru/fit_summary.json) confirms 32 epochs and positive-class weight 1. Although the historical configuration has `class_weighting: true`, its `alpha: 0` gave no effective positive-class upweighting.

The available GRU records do not include the full candidate grid or per-candidate scores, so the tested ranges and number of trials cannot be verified. The current `train.py` reuses the selected configuration for all five OOF fits and the final refit, with exactly 32 epochs and no early stopping; it does not repeat the historical search.

### Splits, tuning and ensembles

The supplied manifest preserves **172,274 original training + validation sites** and **41,697 test sites**, including all five grouped fold assignments. Grouping links genes/transcripts across dataset0/1 and matching mixture positions in dataset2. Each base model trains on four folds and predicts the fifth; normalizers use only fitting rows. Final base models are then fitted on all original training + validation sites.

All three base-model configurations are fixed in the current pipeline. GB/MLP search histories and GRU selection evidence are in [hyperparameter_tuning/](hyperparameter_tuning/README.md): GB tested 32 candidates; MLP screened 12 and trained three finalists longer; GRU selected settings by mean fold trapezoidal PR-AUC and used 32 final epochs. GRU's retained selection and fitting records are in [gru/](hyperparameter_tuning/gru/README.md). 

The ensembles use the three OOF probabilities. OOF folds 1–4 select/fit ensemble settings; fold 5 validates them. The saved weighted ensemble uses GRU/MLP/GB weights **0.44/0.40/0.16**. The default logistic ensemble uses raw probabilities and clipped logits, standardized before binary logistic regression. Both are applied to final-model test probabilities without an all-OOF refit. Ensemble selection uses mean trapezoidal PR-AUC, with ROC-AUC breaking ties. Historical tuning and OOF ensemble validation are not fully nested.

## All command-line arguments

Both scripts accept `-h` / `--help` to display usage.

### `predict.py`

| Argument | Meaning | Default |
|---|---|---|
| `--data PATH` | Raw JSONL or JSONL.gz | Required |
| `--output PATH` | New prediction CSV; parent directories created automatically | Required |
| `--model-dir PATH` | Directory containing saved model subfolders | Package `models/` |
| `--method NAME` | `ensemble` (logistic), `weighted`, `mlp`, `gb` or `gru` | `ensemble` |
| `--threads N` | Positive CPU thread count | `1` |

### `train.py`

| Argument | Meaning | Default |
|---|---|---|
| `--config PATH` | JSON listing raw data, labels and split manifest | Package `training_inputs.json` |
| `--out-dir PATH` | New run directory; must not already exist | Required |
| `--prepare-only` | Clean/join data and verify assignments without model fitting | Off |
| `--methods NAME ...` | One or more of `mlp`, `gb`, `gru` | All three |
| `--skip-ensemble` | Skip ensemble fitting; otherwise ensembles require all three base models | Off |
| `--threads N` | Positive CPU thread count | `1` |
| `--seed N` | Random seed; neural fold fits use seed + fold, final fits use seed | `42` |

Architecture and optimizer settings are fixed constants in `train.py`, not additional CLI arguments.

## Interpret evaluation results

[results/test/test_metrics.csv](results/test/test_metrics.csv) contains preserved per-dataset and pooled test results; a new run writes `evaluation_metrics.csv`. PR-AUC uses trapezoidal integration; average precision (AP) is separate. ROC-AUC and PR-AUC evaluate probability rankings across thresholds. Pooled metrics combine all test predictions, rather than averaging dataset scores. All test metrics are unweighted.

The saved logistic ensemble achieved pooled **PR-AUC 0.527054** and **ROC-AUC 0.896287**. `results/test/dataset*_heldout_predictions.csv` contains its test-only predictions. The included unlabeled example cannot measure accuracy, PR-AUC or ROC-AUC. Existing results are internal held-out evaluations on previously explored datasets, not independent external validation.

## Software license

**Licensing is currently unspecified.** No software license has been selected for this project. Contact the contributors about reuse; dependencies retain their own licenses.

## How to cite

> DataHelix contributors (2026). *DataHelix: m6A site prediction with GB, MLP and GRU ensembles* [Software]. DSA4262-datahelix. https://github.com/thydaphea/DSA4262-datahelix

See [CITATION.cff](CITATION.cff) for machine-readable citation details. Include the Git commit or package checksum used in your analysis.
