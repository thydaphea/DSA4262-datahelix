# m6A prediction with shared GB, MLP, and MIL models

Each model learns jointly from the **training portions of dataset0, dataset1, and dataset2**. The same saved checkpoint predicts all three datasets. Pooled validation PR-AUC selects the checkpoint; test sites are excluded from training and selection.

- **Models:** `results/gb/model.joblib`, `results/mlp/model.joblib`, and `results/mil/model.joblib`
- **Comparison metrics:** [results/evaluation_metrics.csv](results/evaluation_metrics.csv)
- **Full-dataset predictions:** [GB](results/gb/submission/), [MLP](results/mlp/submission/), and [MIL](results/mil/submission/)
- **Training, prediction, and evaluation commands:** [MLP.md](MLP.md)
- **Data preparation:** [DATA_PREPARATION.md](DATA_PREPARATION.md)

Older separate-dataset models and their results have been removed. The shared MLP is retained, with shared GB and MIL added for comparison. Raw and cleaned data are unchanged.

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

Only the seven Python files listed above are retained. Obsolete separate-model scripts, duplicate entry points, and optional check/benchmark scripts have been removed.

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
