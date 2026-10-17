# Cleaning and feature preparation

Two historically separate preparation workflows are included. Their 74 model input values match on the packaged merged training data after column mapping. Both use the same sites, labels and standardized folds.

## GRU

`gru/clean_gru.ipynb` contains the original cleaning algorithm from `m6a_clean_multiple_datasets_train_test.ipynb`. Only input/output paths have been adapted and stored notebook outputs cleared. Run its cells in order from this submission folder or any subfolder. It reads `dataset/raw/` and `dataset/raw/split_manifest.csv`, and writes a timestamped directory under `dataset/rebuilt/gru/`. It retains original binary manifest labels; original train and validation are merged, and test remains separate. It computes 45 signal summaries, keeps sequence and coverage, and does not fit a model scaler.

Then assign folds to that new run:

```bash
python cleaning/gru/create_standardized_folds.py --source dataset/rebuilt/gru/NEW_RUN --output dataset/rebuilt/gru_standardized
```

`gru/gru_common.py` supplies the original biological grouping audit used by the fold generator. All models must use the same assignments. Existing assignments and group IDs are in `dataset/cleaned/gru/fold_assignments.csv.gz`; preserve these for comparing with the submitted OOF outputs.

## MLP and GB

`mlp_gb/clean_data.py` is the original standalone raw JSON/label cleaner, and `features.py` constructs the 74 pooled features. `check_cleaning.py` exercises these rules with temporary fixtures. `HISTORICAL_DATA_PREPARATION.md` is the original documentation; its old example locations are historical, not current submission paths.

Example (writes new intermediate outputs, not the submitted feature tables):

```bash
python cleaning/mlp_gb/clean_data.py --data dataset/raw/data0/dataset0_json/dataset0.json --labels dataset/raw/data0/data.info.labelled --out-dir dataset/rebuilt/mlp_gb_raw/dataset0
```

For a reproducible route to the supplied feature schema and shared folds, the added `prepare_features.py` maps the equivalent GRU summaries to GB/MLP columns, including sequence one-hot and log read count:

```bash
python cleaning/mlp_gb/prepare_features.py --output dataset/rebuilt/mlp_gb
```

This helper was added for submission packaging; it is not represented as the historical generator. Its reconstructed feature values, labels and folds are checked against the submitted tables. Existing model inputs are `dataset/cleaned/mlp_gb/`. Feature/label columns are distinct from metadata; exclude fold, IDs and original_split from model inputs. Model scaling is learned only on the relevant training rows, never in these cleaning steps.
