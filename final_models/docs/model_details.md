# Cleaning and models
One standard cleaner normalizes sequences (uppercase, U→T), retains seven-base A/C/G/T sequences with a central DRACH motif, removes invalid reads, and requires at least one valid read. Invalid reads include malformed/nonnumeric/nonfinite measurements or negative dwell time/signal SD. Ambiguous JSON structures and duplicate sites raise errors.

The [cleaned site table](../training_data/sites.csv.gz) has ****213,971 rows and 63 columns****: identifiers, sequence, 45 signal summaries, raw/valid/removed read counts, cleaning indicator, label/source fraction and grouping/split metadata. These stored columns are not all model inputs. During training and labelled evaluation, matching uses dataset ID plus both coordinates. Assigned sites must survive cleaning and match saved labels; otherwise processing stops. Unassigned extra sites are audited and excluded from development. Current data retained all assigned sites, with no extras.

### Features Used for Gradient Boosting (GB) and Multilayer Perceptron (MLP)

Both models aggregate all valid reads at each site into **74 features**:

| Features | Count |
|---|---:|
| Mean and population standard deviation of each of the nine measurements | 18 |
| 25th percentile, median, and 75th percentile of each measurement | 27 |
| log(1 + valid read count) | 1 |
| Seven sequence positions × four A/C/G/T indicators | 28 |
| **Total** | **74** |

For example, 100 reads become one 74-value site vector. The pooled statistics include the mean and SD ****across reads of the original signal-SD measurements****. Sequence encoding uses A=`[1,0,0,0]`, C=`[0,1,0,0]`, G=`[0,0,1,0]`, T=`[0,0,0,1]`. Identifiers, labels and split metadata are never classifier inputs.


### Gradient Boosting (GB) Configurations

The GB model uses `HistGradientBoostingClassifier`, which builds sequential decision trees.

| Setting | Selected value |
|---|---|
| Model | `HistGradientBoostingClassifier` |
| Number of iterations | 600 |
| Learning rate | 0.03 |
| Maximum leaf nodes | 63 |
| Minimum samples per leaf | 10 |
| L2 regularization | 1 |
| Feature standardization | None |
| Sample weighting | None |
| Early stopping | Disabled |

### Multilayer Perceptron (MLP) Configurations

The MLP is a fully connected neural network that uses 74 input features.

| Setting | Selected value |
|---|---|
| Network architecture | 74 → 256 → 128 → 1 |
| Hidden activation | ReLU |
| Dropout | 0.5 after the first hidden layer |
| Optimizer | AdamW |
| Learning rate | 0.001 |
| Weight decay | 0.0001 |
| Batch size | 256 |
| Fixed training epochs | 16 |
| Loss | Unweighted binary cross-entropy (BCE) |
| Output activation | Sigmoid (converts logits to probabilities) |

MLP standardizes all 74 features using fitting-row means and population SDs (scales below `1e-8` become 1). Saved scaling is reused for inference; dropout is disabled. GB uses the unscaled feature vector. Neither model directly predicts individual reads.

****GB tuning:**** 32 candidate configurations varied learning rate, boosting iterations, maximum leaves, minimum leaf size and L2 regularization. Selection used pooled validation trapezoidal PR-AUC; candidate 9 won with ****0.479539**** and the settings above. See [GB search evidence](../hyperparameter_tuning/gb/search_summary.csv).

****MLP tuning:**** 12 configurations were screened for four epochs, varying hidden layers, dropout, learning rate, AdamW decay, batch size and class weighting. Three finalists were trained for up to 30 epochs with early-stopping patience 6. Candidate 9 at epoch 16 achieved the best pooled validation trapezoidal PR-AUC, ****0.492715****. Settings varied together; later controlled experiments are documented separately and did not replace the fixed configuration above. See [MLP search evidence](../hyperparameter_tuning/README.md#later-mlp-experiments) and [candidate summary](../hyperparameter_tuning/mlp/search_summary.csv). These are historical validation scores, not final test metrics.

Saved base-model files are `models/gb/model.joblib`, `models/mlp/model.joblib` and `models/gru/model.pt`. New training runs use the same layout.

### Gated Recurrent Unit (GRU)
The ****gated recurrent unit (GRU)**** models the ordered signal context around a site. It uses the same 74 values as GB/MLP, arranged into two inputs:

\- ****Signal input:**** three steps (upstream, centre, downstream), each containing 15 values: five read-level summary statistics × three measurements (dwell time, signal SD and signal mean).

\- ****Auxiliary input:**** 28 nucleotide indicators plus log(1 + valid read count), giving 29 values.

A single-layer, unidirectional GRU with hidden size 32 processes the three signal positions. Its final 32-value hidden state is concatenated with the 29 auxiliary values; a fully connected head maps ****61 → 32 → 1****, with ReLU and dropout after the 32-unit layer. Sigmoid converts the final logit into one site probability. The recurrent steps are neighbouring signal positions, not individual reads or the seven nucleotide characters.

The rationale is to learn relationships between neighbouring signal positions while retaining sequence context and coverage in the classifier. Pooling reads first gives a fixed-size input for sites with different read counts, although it discards individual-read variation beyond the summary statistics. This architecture provides a different way to use positional information from the MLP's flattened input; it does not guarantee better performance.

### Gated Recurrent Unit (GRU) Configurations

| Setting | Selected value |
|---|---|
| GRU hidden size / layers | 32 / 1 |
| Classifier-head dropout | 0 |
| Optimizer | AdamW |
| Learning rate | 0.0003 |
| Weight decay | 0 |
| Batch size | 256 |
| Fixed training epochs | 32 |
| Loss | Unweighted `BCEWithLogitsLoss` (`pos_weight = 1`) |
| Gradient clipping | Maximum gradient norm of 1 |

Each of the 45 signal summaries and log coverage is standardized using only fitting rows; nucleotide indicators remain unscaled. Saved means/scales are reused at prediction time. The saved model is `models/gru/model.pt`, with its scaling parameters in `preprocessing.json`.

****Tuning:**** the preserved [configuration-selection record](../hyperparameter_tuning/gru/configuration_selection.json) specifies selection by mean fold trapezoidal PR-AUC, followed by ROC-AUC and weight-decay tie-breaks. It records a 50-epoch tuning limit, patience 5 and minimum improvement 0.001; the final duration was set to the rounded median best epoch across folds, ****32 epochs****. The [final fit summary](../models/gru/fit_summary.json) confirms 32 epochs and positive-class weight 1. Although the historical configuration has `class_weighting: true`, its `alpha: 0` gave no effective positive-class upweighting.

By default, `train.py` reuses the selected configuration for all five OOF fits and the final refit, with exactly 32 epochs and no early stopping. With `--tune`, it runs the 17-candidate search described above, including early stopping within each development fold; the selected final duration is the median best epoch across folds. This current search does not reconstruct the unavailable historical candidate records.

### Splits and Ensemble Development

The supplied split manifest preserves **172,274 development sites (training and validation)** and **41,697 held-out test sites**, including five grouped cross-validation folds. Grouping accounts for shared genes/transcripts across Datasets 0 and 1 and matching genomic positions across mixture conditions in Dataset 2, reducing potential data leakage.

**Five-fold out-of-fold (OOF) predictions** were generated by training each base model on four folds and predicting the remaining fold. Preprocessing parameters were fitted exclusively on the training folds to prevent data leakage.

The resulting OOF predictions were used to develop two ensemble approaches: **weighted averaging and logistic regression stacking**. OOF folds 1–4 were used to select ensemble weights or fit the stacking model, while fold 5 was reserved for ensemble validation.

Following ensemble development, each base model was **retrained from scratch on the entire development set** using its selected hyperparameters. The retrained base models and saved ensemble configurations were then applied to the **untouched test set** for final performance evaluation.

### Logistic Regression Ensemble
The saved logistic ensemble uses raw probabilities and clipped logits, standardized before binary logistic regression. It is applied to final-model test probabilities without an all-OOF refit. Ensemble selection uses mean trapezoidal PR-AUC, with ROC-AUC breaking ties. Historical tuning and OOF ensemble validation are not fully nested.

The trained logistic regression stacking ensemble is saved at `models/ensemble/logistic_regression.joblib`.