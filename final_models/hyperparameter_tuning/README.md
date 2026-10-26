# GB, MLP and GRU hyperparameter-tuning evidence

This directory contains GB/MLP source snapshots and saved search results copied from `/home/ubuntu/project`, plus GRU selection and fitting records preserved from the supplied final package. It is supporting evidence; the submission entry points remain `../train.py` and `../predict.py`. Neither entry point imports this directory. No search was rerun and no submitted models or predictions were replaced.

## Contents

- `original_code/train.py`: original shared trainer containing GB candidate search, MLP screening/finalist search, and controlled MLP comparison code.
- `original_code/clean_data.py`, `features.py`, `neural.py`, `metrics.py`: its original dependencies, copied without edits so imports remain valid. The shared source also includes MIL branches; this evidence collection covers GB and MLP.
- `original_code/requirements.txt`: original project environment specification, separate from the final package's inference environment.
- `gb/`: original 32-candidate validation history, original metrics/settings, selected configuration and a readable candidate-summary CSV.
- `mlp/`: original per-epoch tuning history, selected-run validation history, metrics/settings, selected configuration and candidate/stage summary CSV.
- `mlp/additional_experiments/`: selection records and validation histories for subsequent dropout, positive-loss-weight and epoch comparisons.
- `gru/`: historical configuration selection, selected settings, final-fit summary, subsequent OOF run manifest and source hashes; see [GRU evidence and limitations](gru/README.md). The full candidate search history is unavailable.
- `original_split_manifest.csv`: original train/validation/test assignments shared by GB and MLP. This is historical evidence, not a replacement for `../training_data/splits.csv.gz` and its preserved five OOF folds.
- `provenance.json`: original file paths and SHA-256 hashes of the exact copied files.

The source files are snapshots of the project files available when this evidence was assembled. The project code evolved after earlier experiments; saved histories are the record of the completed runs. This does not claim every current source byte is identical to the code used at the original run date.

## What was selected

| Search | Procedure | Winner | Pooled validation PR-AUC (trapezoidal) |
|---|---|---|---:|
| GB | 32 candidates; same held-out validation partition | Candidate 9: 63 leaves, 600 iterations, learning rate 0.03, minimum leaf size 10, L2 1 | 0.4795393435 |
| MLP | 12 configurations screened for 4 epochs; candidates 1, 9 and 12 trained up to 30 epochs with patience 6 | Candidate 9: hidden 256/128, dropout 0.5, learning rate 0.001, AdamW decay 0.0001, batch 256, unweighted loss; selected epoch 16 | 0.4927150286 |

Both winners match the fixed configurations in the final training script. The later final models were fitted on original training + validation data; they are different fitted artifacts from the original search checkpoints. These numbers are historical validation scores, not current final test scores. Refer to `../results/test/test_metrics.csv` for final test performance.

The original GB/MLP searches selected using pooled validation PR-AUC; it did not select by mean five-fold CV. The final pipeline subsequently uses the selected settings to produce five-fold OOF predictions. MLP parameters were varied together, so the broad search does not isolate their individual effects.

## GRU selection

The retained record specifies mean fold trapezoidal PR-AUC selection, followed by ROC-AUC and weight-decay tie-breaks. Selected settings are hidden size 32, dropout 0, learning rate 0.0003, decay 0, batch 256 and effectively unweighted loss. The final 32 epochs came from the rounded median best epoch. See [the GRU records](gru/README.md) for provenance and the distinction between historical tuning and subsequent fixed-setting OOF training. Candidate ranges, trial count and winning search score are not available.

## Later MLP experiments

- Dropout comparison: selected **0.2**, validation PR-AUC **0.4982226121**. This later configuration was **not adopted** by the submitted fixed MLP, which retains dropout **0.5**. Do not describe dropout 0.5 as the winner of every later experiment.
- Positive-loss weights 1, 2, 3 and 5: retained the unweighted baseline (weight 1).
- Fixed epoch 28 versus baseline epoch 16: retained epoch 16 by validation PR-AUC.

These records are preserved for transparency. The original selection JSON files reference historical model paths; the old checkpoints are not duplicated here. The current submitted models remain in `../models/`.

## Inspecting or reproducing the original code

Inspect the search loops in `original_code/train.py` and the saved CSV/JSON evidence. To see its command-line options without training:

```bash
python hyperparameter_tuning/original_code/train.py --help
```

Reproducing historical searches requires the original raw/cleaned data, the original environment and explicit new output directories. Controlled comparisons additionally require the original baseline checkpoint in `project/results/mlp`, not the final refitted checkpoint. Do not use this archived cleaner or splitter in place of the current standardised pipeline: the root `train.py` and `predict.py` implement the current cleaning and preserved assignments.
