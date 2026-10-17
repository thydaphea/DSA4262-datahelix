# GRU trained on original training rows only

This independent copy of `../gru_scripts` trains a fresh model using only
`original_split == "train"` from the three merged cleaned training CSVs.
`original_split == "validation"` is held out and scored after training.
The original scripts, model, and source CSVs are unchanged. Test CSVs are not used.

## Reproduce

Run from this folder with the same Python environment used for the existing model:

```bash
python -m pip install -r requirements.txt
python train_gru.py --data-config datasets.json --output-dir model
python evaluate_gru.py --data-config datasets.json --model-dir model --output-dir validation_predictions
```

The completed run is saved in `model/` and `validation_predictions/`. Use new output
folder names when rerunning; existing outputs are never overwritten. Source paths
in `datasets.json` are relative to that JSON file and point to the same three
cleaned training CSVs as the previous model.

## Configuration and separation

The architecture, feature order, preprocessing, optimizer and fixed duration match
the previous saved run: hidden size 32, one unidirectional GRU layer, dropout 0,
AdamW learning rate 0.0003, weight decay 0.0001, batch size 256, alpha 0 (unweighted
BCE), gradient clipping 1, seed 42, and **32 epochs**. Weights are initialized
fresh. Scalers are fitted exclusively on original training rows.

Fixed mode uses no early stopping or internal validation. Optional training-only
five-fold grouped CV can select the refit duration:

```bash
python train_gru.py --data-config datasets.json --output-dir model_cv --mode cv-refit
python evaluate_gru.py --data-config datasets.json --model-dir model_cv --output-dir validation_predictions_cv
```

Each internal fold contains only original training rows, fits its own scaler,
and uses the existing patience 5/minimum delta 0.001/maximum 50 epochs settings.
The rounded median best epoch determines a fresh refit on all original training
rows. Do not supply the old merged-data fold assignments: they do not match this
training subset and will be rejected.

Both entry points reject missing/unknown original split labels and audit site and
biological group separation across datasets. The established dataset2 mixture
position grouping policy is preserved. The saved model has a distinct schema to
prevent accidental interchange with the previous merged-data model. Evaluation
verifies the source and saved artifact hashes and never refits preprocessing.

These are the previously chosen hyperparameters and duration, as requested.
Because their historical selection used the merged train/validation population,
retraining removes validation rows from weight/scaler fitting but cannot undo that
historical selection. Validation scores should be interpreted with that limitation.

## Saved artifacts

- `model/model.pt`: new model weights and configuration.
- `model/preprocessing.json`: training-only means and scales.
- `model/training_rows.csv.gz`, `model/heldout_validation_rows.csv.gz`: exact row identities.
- `model/split_audit.json`: separation checks and per-dataset row counts.
- `model/run_manifest.json`: input/output hashes, arguments, environment and completion state.
- `model/training_history.csv`, checkpoints, plots and apparent training metrics.
- `validation_predictions/predictions_validation_dataset0.csv` (also dataset1 and dataset2):
  `transcript_id,transcript_position,score`, with sigmoid probability of label 1.
- `validation_predictions/detailed_predictions_validation.csv.gz`: pooled predictions
  including dataset identity and labels.
- `validation_predictions/performance_summary.csv` and `.json`: validation and apparent
  training metrics; threshold fixed at 0.5.
- `validation_predictions/evaluation_manifest.json`: saved model provenance.

## Verification

```bash
python -m unittest discover -s tests -v
```

Tests cover split filtering, invalid split labels, exclusion of held-out features
from preprocessing, biological overlap, model configuration, optional training-only
CV, saved-model prediction round trips, artifact integrity and overwrite prevention.
