# m6A prediction with shared GB, MLP, and MIL models

Each model learns jointly from the **training portions of dataset0, dataset1, and dataset2**. The same saved checkpoint predicts all three datasets. Pooled validation PR-AUC selects the checkpoint; test sites are excluded from training and selection.

- **Models:** `results/gb/model.joblib`, `results/mlp/model.joblib`, and `results/mil/model.joblib`
- **Comparison metrics:** [results/evaluation_metrics.csv](results/evaluation_metrics.csv)
- **Full-dataset predictions:** [GB](results/gb/submission/), [MLP](results/mlp/submission/), and [MIL](results/mil/submission/)
- **Training, prediction, and evaluation commands:** [MLP.md](MLP.md)
- **Data preparation:** [DATA_PREPARATION.md](DATA_PREPARATION.md)


## Entry points and modules

| File | Purpose |
|---|---|
| `train.py` | Split all three datasets into compatible groups, train one shared GB, MLP, or MIL, and select its checkpoint using pooled validation PR-AUC. |
| `predict.py` | Use one saved model for new JSON inputs or the three assignment datasets. |
| `evaluate_unified.py` | Verify predictions and calculate each dataset's test metrics and pooled test metrics. |
| `clean_data.py` | Parse and clean JSON and labels reproducibly; binarize modification fractions greater than zero. |
| `features.py` | Build pooled signal and sequence features. |
| `neural.py` | Define neural networks, training, and inference routines. |
| `metrics.py` | Calculate trapezoidal PR-AUC. |
| `benchmark_m6anet.py` | Run official pretrained HCT116_RNA002 m6Anet on the existing test sites and compare all four methods. |

The eight Python files listed above support the current workflow and pretrained benchmark. Obsolete separate-model scripts, duplicate entry points, and optional check/benchmark scripts have been removed.

## What each model learns from

One training example is **one transcript site**, identified by `transcript_id` and `transcript_position`. A site can contain many RNA reads, but has one site-level label. Each read has nine measurements: dwell time, signal standard deviation, and mean signal at three neighbouring positions (−1, central, +1).

For dataset2, modification fractions are converted to binary labels: **0 becomes 0; any fraction greater than 0 becomes 1**. A positive site means modification is present in some molecules; it does not mean every read is modified. The models predict a binary site probability, not the fraction of modified molecules.

Identifiers and dataset names are used for joining labels and grouping splits, not as classifier inputs. The sequence is a seven-base motif, encoded with four A/C/G/T indicators per position: `7 × 4 = 28` values. For example, A is `[1,0,0,0]` and C is `[0,1,0,0]`.

## Gradient boosting (GB)

GB is a **decision-tree ensemble**, implemented with scikit-learn's `HistGradientBoostingClassifier`. It adds trees sequentially to improve the binary classification loss. Histogram binning groups numerical feature values into bins to make tree fitting efficient. The trees' combined output gives one modification probability per site.

Before training, all reads at a site are pooled into **one vector containing 74 features**:

| Site feature | Number of values |
|---|---:|
| Mean of each of the nine measurement columns across reads | 9 |
| Standard deviation of each column across reads | 9 |
| 25th, 50th, and 75th percentiles of each column across reads | 27 |
| `log(1 + read count)` | 1 |
| Seven-base sequence indicators | 28 |
| **Total** | **74** |

For example, a site with 100 reads starts as a `100 × 9` signal matrix and becomes one 74-value vector. This includes the **mean and standard deviation across reads of the original signal-standard-deviation measurements**. The original within-read signal SD and the pooled across-read SD describe different kinds of variation.

The selected GB uses 600 boosting iterations, at most 63 leaves per tree, learning rate 0.03, L2 regularization 1.0, and a minimum of 10 samples per leaf. GB uses the pooled features without neural-network standardization. Its selected run is unweighted. Parameters were selected using GB's own pooled validation PR-AUC.

GB can learn nonlinear relationships among the summary features, but fixed pooling discards individual read identities and some relationships among their measurements.

## Multilayer perceptron (MLP)

The MLP is a **fully connected feed-forward neural network**. It uses the same 74 pooled features as GB. Each feature is standardized using its training-set mean and standard deviation; those saved values are reused during prediction.

The selected network is:

```text
All reads at one site
    → 74 pooled signal/count/sequence features
    → standardization
    → Linear(74, 256) → ReLU → Dropout(0.5)
    → Linear(256, 128) → ReLU
    → Linear(128, 1) → sigmoid
    → one site probability
```

A linear layer learns weighted combinations of its inputs. ReLU adds nonlinearity. Dropout randomly zeros activations during training to regularize the network and is disabled during prediction. This implementation has no convolutional layers, so it is an MLP rather than a convolutional neural network (CNN).

Training uses AdamW and `BCEWithLogitsLoss`, which accepts the final raw output (the logit). Sigmoid converts that logit to a probability during inference. The selected configuration uses learning rate 0.001, weight decay 0.0001, batch size 256, and unweighted loss (`pos_weight=1`). Epoch 16 was selected by this configuration's pooled validation PR-AUC after comparing hyperparameter candidates.

Like GB, the MLP learns from fixed site summaries. Its neural layers learn combinations of those summaries, but do not directly model individual reads.

## Attention-based multiple-instance learning (MIL)

MIL treats each site as a **bag** and its reads as the **instances**. Only the bag has a known label; individual reads are not assigned modification labels. This implementation uses a neural network to learn how to combine the read information.

The selected network processes a site as follows:

```text
Up to 64 reads, each with 9 measurements
    → standardize measurements using training reads only
    → shared read encoder: Linear(9, 32) → ReLU → Linear(32, 32) → ReLU
    → one learned 32-value representation per read
    → attention: Linear(32, 8) → tanh → Linear(8, 1)
    → softmax weights across real reads
    → weighted sum: one 32-value site representation
    → append 28 sequence indicators: 60 values
    → Linear(60, 32) → ReLU → Dropout(0.3)
    → Linear(32, 1) → sigmoid
    → one site probability
```

The **32 values are learned features**, not 32 additional measured quantities. The same encoder weights are applied to every read. Attention assigns relative weights that sum to one across the site's real reads, allowing the model to emphasize read representations useful for the site prediction. Attention weights are not read-level modification probabilities or proof that a particular read is modified.

Sites with more than 64 reads use a reproducible sample of 64; sites with fewer reads are padded. A mask excludes padding from attention and normalization. Unlike GB and MLP, MIL learns its pooling from read-level inputs. It uses the sequence indicators alongside the pooled representation, but does not append the explicit read-count feature used by GB and MLP.

MIL uses AdamW and `BCEWithLogitsLoss`. Its selected configuration has learning rate 0.003, weight decay 0.0001, batch size 256, and unweighted loss. Epoch 23 was selected by MIL's own pooled validation PR-AUC. Both neural searches included class-weighted candidates using training-derived `pos_weight`; the selected MLP and MIL configurations were unweighted.

## How the models differ

| Model | Input representation | How reads are combined | Learned predictor |
|---|---|---|---|
| GB | 74 features per site | Fixed statistical summaries | Boosted decision trees |
| MLP | The same 74 features, standardized | Fixed statistical summaries | Fully connected neural network |
| MIL | Up to 64 × 9 measurements plus 28 sequence values | Learned attention-weighted pooling | Read encoder, attention, and neural classifier |

There are **three trained models total: one per algorithm**, each trained jointly on all three datasets' training portions. Within each algorithm, the exact same saved model predicts dataset0, dataset1, and dataset2.

## Evaluation

Each model’s metrics file has four rows: dataset0, dataset1, dataset2, and pooled. The combined `results/evaluation_metrics.csv` has 12 rows, covering all three models. PR-AUC here uses trapezoidal integration; average precision (AP) is recorded separately. Pooled metrics are calculated from all held-out test predictions together, not by averaging the three scores. Full-dataset submission files also contain training and validation sites; those rows are excluded from reported test metrics.

Known gene/transcript groups are kept in one partition across dataset0 and dataset1. Dataset2 matching positions across mixture conditions are grouped together, assuming they represent related biological sites. These are internal held-out results on previously explored datasets, not independent external validation.

## Predict new data with a saved model

Activate the project environment, then use any one of the saved models. Prediction requires no labels or retraining. Replace `path/to/new_data.json.gz` with your input file; both `.json` and `.json.gz` are supported.

```bash
cd /home/ubuntu/project
source .venv311/bin/activate

# Gradient boosting
python predict.py --data path/to/new_data.json.gz --model results/gb/model.joblib --output gb_predictions.csv

# Multilayer perceptron
python predict.py --data path/to/new_data.json.gz --model results/mlp/model.joblib --output mlp_predictions.csv

# Attention-based MIL
python predict.py --data path/to/new_data.json.gz --model results/mil/model.joblib --output mil_predictions.csv
```

Each output CSV contains exactly `transcript_id,transcript_position,score`, with one probability score per retained site. Parsing and cleaning run automatically through `clean_data.py`. Choose distinct output filenames to preserve earlier predictions.

To predict all three supplied datasets with one saved model, use:

```bash
python predict.py --all-datasets --model results/mlp/model.joblib --out-dir results/mlp/submission_new
```

Replace `mlp` in both paths with `gb` or `mil` to use another model. The output directory must not already exist. This creates exactly three full-dataset CSVs, one for each dataset. For submission, choose one model's three files and rename them using your registered team name.

## Shared-model comparison

Each algorithm fits one model jointly across all three datasets' training portions. All use the same group split and each model's own pooled validation PR-AUC. Test labels never choose parameters.

GB searches 32 reproducible configurations: the previous baseline plus a seeded sample from leaves {7,15,31,63}, iterations {100,200,400,600}, learning rate {0.01,0.03,0.1,0.2}, L2 {0,1,10,30}, and minimum leaf samples {10,20,50,100}. The baseline also includes 300 iterations.

Each neural model screens 12 configurations for four epochs, then trains the baseline and the two highest-scoring other candidates for up to 30 epochs (early-stopping patience six). MLP hidden layers span 32/16 through 256/128. MIL encoder widths span 16/32/64 and attention widths 8/16/32; each bag has up to 64 deterministic reads. Both searches vary dropout {0,0.1,0.2,0.3,0.5}, learning rates {0.0001,0.0003,0.001,0.003}, weight decay {0,0.0001,0.001}, batches {128,256,512}, and unweighted/balanced training. These are selected combinations, not the full Cartesian product. Early screening can miss slow-learning configurations. The full neural trial records are in `tuning_history.json`; `validation_history.json` records the selected candidate's epochs.

Neural search configurations determine learning rate, batch size, and weighting; the corresponding training command defaults describe the baseline only. The winning settings are saved in `metrics.json` under `model_selection.best_params`.

```bash
python train.py --model-type gb --clean-dirs data/cleaned/dataset0 data/cleaned/dataset1 data/cleaned/dataset2
python train.py --model-type mil --clean-dirs data/cleaned/dataset0 data/cleaned/dataset1 data/cleaned/dataset2 --threads 1
python predict.py --all-datasets --model results/gb/model.joblib --out-dir results/gb/submission
python predict.py --all-datasets --model results/mil/model.joblib --out-dir results/mil/submission
python evaluate_unified.py --compare --results-root results
```

Use new output directories to repeat training or prediction. The combined metrics file has 12 rows: four evaluations per algorithm. Each model's directory also has its own four-row metrics file. Submission folders contain all sites; evaluation folders contain held-out sites only. For the assignment, choose one model and submit only its three files, renamed with your team name.

## Selected configurations

These settings were selected using each model’s own pooled validation PR-AUC.

| Model | Validation PR-AUC | Selected settings |
|---|---:|---|
| GB | 0.479539 | `{"max_leaf_nodes":63,"max_iter":600,"learning_rate":0.03,"l2_regularization":1.0,"min_samples_leaf":10}` |
| MLP | 0.492715 | `{"network_config":{"hidden":[256,128],"dropout":0.5},"learning_rate":0.001,"weight_decay":0.0001,"batch_size":256,"class_weight":"none"}` |
| MIL | 0.413224 | `{"network_config":{"width":32,"attention_width":8,"dropout":0.3},"learning_rate":0.003,"weight_decay":0.0001,"batch_size":256,"class_weight":"none"}` |

## Benchmark against pretrained m6Anet (HCT116_RNA002)

`benchmark_m6anet.py` evaluates the official `HCT116_RNA002` option (the official name; there is no `HCT115_RNA002` option) on the **same 41,697 held-out sites** used for GB, MLP, and MIL. There is no training or fine-tuning. The pretrained model is user-selected; sample chemistry has not been independently checked against sequencing metadata. Any overlap between these samples and the published model's training data is unknown; the sites are guaranteed held out from our project models only. This is a pretrained-tool benchmark, not a comparison of architectures trained on identical examples or with equal tuning budgets.

The official implementation is pinned to commit `590ec277cb48d61774f0872395099e466022e810`. It declares historical Python/PyTorch dependencies; the adapter runs the unchanged source using this project's Python 3.11/PyTorch 2.6 environment, with an explicit compatibility check against the official input loader. Extra support packages are isolated under `.tools/m6anet-deps/`.

To prepare a fresh checkout (skip cloning if the directory already exists):

```bash
git clone https://github.com/GoekeLab/m6anet.git .tools/m6anet-source
git -C .tools/m6anet-source checkout 590ec277cb48d61774f0872395099e466022e810
.tools/uv pip install --python .venv311/bin/python --target .tools/m6anet-deps --no-deps toml==0.10.2 tqdm==4.67.1 ujson==5.10.0 setuptools==75.8.0
```

Run the compatibility check and benchmark:

```bash
source .venv311/bin/activate
python benchmark_m6anet.py --check-only
python benchmark_m6anet.py --pretrained-model HCT116_RNA002 --out-dir results/m6anet_rna002
```

Use a new output directory for repeat runs. If `uv` is installed elsewhere, replace `.tools/uv` with its executable path. The normal training/prediction environment is not modified by the isolated support-package installation.

The adapter streams the cleaned nine-column measurements, applies the official pretrained k-mer normalization and embedding, and calls the official read encoder and site-probability calculation. No measurement columns are dropped. It tests equivalence with the official indexed loader using temporary bookkeeping read IDs; those IDs are not model features or recovered biological read identities. All test sites pass the default 20-read minimum and supported motif checks. If future test data do not meet those requirements, the script stops rather than silently comparing different sites.

For this pinned source, **the official HCT116_RNA002 option uses `rna002_hct116.pt` weights and `rna002_hct116.joblib` normalization**. We preserve that pairing exactly; we do not estimate normalization from the test data. The source configuration can be inspected in [m6Anet constants](https://github.com/GoekeLab/m6anet/blob/590ec277cb48d61774f0872395099e466022e810/m6anet/utils/constants.py).

Each score averages the official noisy-OR probability over 1,000 bootstrap samples of 20 reads, sampled with replacement. A reproducible seed is derived separately for each dataset/site. This serial seeding scheme avoids dependence on multiprocessing scheduling; numerical scores may differ slightly from an unseeded/default CLI run. We compare **site modification probability**, not `mod_ratio` or per-read probabilities.

Outputs in `results/m6anet_rna002/`:

- `benchmark_metrics.csv`: 16 rows comparing all four methods, per dataset and pooled, on identical test sites; includes trapezoidal PR-AUC, AP, and ROC-AUC.
- `evaluation_metrics.csv`: the four pretrained m6Anet metric rows.
- `evaluation/`: three test-only prediction CSVs, pooled test predictions, and dataset provenance.
- `test_manifest.csv`: the exact evaluated sites and labels.
- `pretrained_model.pt`, `normalization.joblib`, `model_config.toml`: the official pretrained assets used.
- `benchmark_audit.json`: source commit, asset/input hashes, settings, compatibility checks, and limitations.

The existing `results/evaluation_metrics.csv` remains the three-model comparison. No existing model or prediction file is overwritten. These m6Anet outputs cover **test sites only** and are not full-dataset assignment submission files.

Observed pooled test metrics for the pretrained RNA002 benchmark (41,697 sites):

| Method | Trapezoidal PR-AUC | ROC-AUC |
|---|---:|---:|
| GB | 0.4937 | 0.8893 |
| MLP | 0.5071 | 0.8892 |
| Attention MIL | 0.4348 | 0.8739 |
| Pretrained m6Anet HCT116_RNA002 | 0.4278 | 0.8703 |

These results describe transfer of the specified pretrained weights to these samples. Chemistry/domain compatibility and original training-set overlap remain unverified; they do not establish an architecture-only performance ranking. See [the full benchmark table](results/m6anet_rna002/benchmark_metrics.csv) for per-dataset results and [the audit](results/m6anet_rna002/benchmark_audit.json) for provenance.


## Controlled MLP positive-weight comparison

To test moderate positive-class training-loss weights while keeping the selected MLP configuration fixed:

```bash
python train.py --model-type mlp --clean-dirs data/cleaned/dataset0 data/cleaned/dataset1 data/cleaned/dataset2 --mlp-positive-weights 1 2 3 5 --out-dir results/mlp_weight_comparison
```

This mode takes the architecture, optimizer settings, batch size, seed, thread count, maximum epochs, early-stopping patience, and exact partitions from the saved baseline in `results/mlp/`. It changes only the positive multiplier in `BCEWithLogitsLoss`; it does not run the broader hyperparameter search. Other training-setting flags do not override those baseline settings in this comparison mode. Use a new output directory to repeat the experiment.

Weight 1 reuses and verifies the existing unweighted checkpoint. Weights 2, 3, and 5 train new models with the same 256→128 hidden layers, dropout 0.5, learning rate 0.001, batch size 256, AdamW decay 0.0001, seed 42, maximum 30 epochs, and patience six. Training-only standardization is identical across variants. Each new checkpoint and the preferred weight are selected by **unweighted pooled validation PR-AUC**, before new test metrics are computed.

`results/mlp_weight_comparison/evaluation_metrics.csv` contains 16 rows: dataset0, dataset1, dataset2, and pooled for each weight. It includes PR-AUC, AP, ROC-AUC, Brier score, and precision/recall at 0.5. `selection.json` records the validation-selected weight, fixed settings, model hashes, and limitations. The common `split_manifest.csv` records every site's partition. Each `weight_*/evaluation/` directory contains held-out predictions; weights greater than 1 also retain their models and validation histories. The weight-1 model remains at `results/mlp/model.joblib`.

These are **training-loss weights, not test weights**. Validation/test metrics use every site's natural contribution. Weighting can change probability calibration and the behaviour of a 0.5 cutoff. This single-seed experiment measures conditional effects for the current configuration; it does not establish statistical significance. Existing models, submissions, and comparison tables are not replaced.

Completed moderate-weight comparison on 41,697 pooled test sites:

| Positive training weight | Validation PR-AUC | Test PR-AUC | Test ROC-AUC |
|---|---:|---:|---:|
| 1 (unweighted) | 0.4927 | 0.5071 | 0.8892 |
| 2 | 0.4893 | 0.5030 | 0.8885 |
| 3 | 0.4887 | 0.5038 | 0.8890 |
| 5 | 0.4877 | 0.5031 | 0.8881 |

The unweighted baseline remains selected by validation PR-AUC. Moderate weighting increased recall at a 0.5 threshold but reduced precision and did not improve PR-AUC in this run. See [all per-dataset and pooled weighting results](results/mlp_weight_comparison/evaluation_metrics.csv). The existing final MLP remains unchanged.
