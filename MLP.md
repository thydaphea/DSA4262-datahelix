# Shared GB, MLP, and MIL workflow

Each algorithm trains **one model jointly on the training portions of dataset0, dataset1, and dataset2**. The same saved model is used for all three datasets. Each model uses its own pooled validation PR-AUC for selection; test sites never select parameters or epochs.

## Train

Use `train.py` with `--model-type gb`, `mlp`, or `mil`:

```bash
source .venv311/bin/activate
python train.py --model-type mlp --clean-dirs data/cleaned/dataset0 data/cleaned/dataset1 data/cleaned/dataset2 --out-dir results/mlp_new --threads 1
```

Or use raw inputs with their matching labels:

```bash
python train.py --model-type mlp --data-files data/dataset0.json.gz data/dataset1.json.gz data/dataset2.json.gz --label-files data/data.info.labelled data/dataset1.info data/dataset2.info --out-dir results/mlp_new --threads 1
```

Choose a new output directory when repeating a run. The default is `results/<model-type>`. The broader search ranges and screening protocol are documented in [README.md](README.md#shared-model-comparison). This is a bounded search, not an exhaustive optimum. Neural architecture, optimizer settings, and weighting selected by validation are recorded under `model_selection.best_params` in each model's `metrics.json`.

## Splits

The joint group split is stratified by dataset and binary label, using seed 42. Known genes and transcript fallbacks stay in one partition across dataset0 and dataset1. Matching dataset2 positions across mixture conditions also stay together. Dataset2 uses label 1 for any modification fraction greater than zero.

All algorithms use identical partitions: 138,819 training sites, 33,455 validation sites, and 41,697 test sites. Normalization is learned from training rows only. There is no all-data refit after model selection. Dataset2 grouping assumes matching positions identify related sites; unknown biological/read provenance limits independence claims. These are internal held-out results from previously explored datasets.

## Predict

One unseen raw input requires no labels:

```bash
python predict.py --data unseen.json.gz --model results/mlp/model.joblib --output unseen_predictions.csv
```

Generate three full-dataset prediction files from one selected model:

```bash
python predict.py --all-datasets --model results/mlp/model.joblib --out-dir results/mlp/submission_new
```

Use `results/gb/model.joblib` or `results/mil/model.joblib` to choose another algorithm. The output has exactly `transcript_id,transcript_position,score`. Scores are between zero and one. The all-datasets command checks coverage, duplicates, probability ranges, and the common model hash before publishing exactly three CSVs.

## Evaluate and compare

```bash
python evaluate_unified.py --results-root results/mlp
python evaluate_unified.py --compare --results-root results
```

The evaluator uses each model folder's canonical `submission/` directory and its audit. It filters to test rows and verifies agreement with training-time checkpoint predictions. `results/evaluation_metrics.csv` contains the 12-row comparison: dataset0, dataset1, dataset2, and pooled for each algorithm. Per-model metrics remain in their own folders.

Pooled metrics concatenate all 41,697 test predictions, rather than averaging dataset metrics. Full submission files include training and validation sites as required by the assignment, but reported evaluation metrics exclude them.

## Submit

Choose **one** model's three full-dataset files from its `submission/` directory. Rename copies to `TEAMNAME_dataset0.csv`, `TEAMNAME_dataset1.csv`, and `TEAMNAME_dataset2.csv`. Do not submit held-out-only files or all nine files from all algorithms. No files are uploaded automatically.
