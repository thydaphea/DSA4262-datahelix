# GRU with standardized five-fold OOF predictions

Run `python train_oof.py` from this folder. Default output: `run/`; use `--output NEW_DIRECTORY` to rerun. Existing run directories are never overwritten.

Uses all merged training rows in `outputs/data_cleaning/20261014T062257_386425Z_standardized_5folds`, preserving its fold column. One pooled GRU per fold fits the other four folds across all three datasets. Fold and identifiers are metadata, not model inputs. Each fold gets a fresh model and scalers fitted exclusively on its training rows. Test files are read only for metadata separation checks.

The latest completed tuning selection is round 6: hidden size 32, dropout 0, AdamW learning rate 0.0003, weight decay 0, batch size 256, alpha 0 (unweighted BCE). Each model trains for the historically recommended 32 fixed epochs, seed 42 + fold. There is no checkpoint selection using OOF labels. The original 50-epoch maximum and early-stopping settings apply to historical tuning, not this fixed-duration run.

`run/oof_predictions_fold_1.csv` through `oof_predictions_fold_5.csv` each contain one held-out fold. `run/oof_predictions_combined.csv` contains all rows once, in source order. Columns: dataset_id, transcript_id, transcript_position, fold, label, score. Use dataset_id plus transcript_id and transcript_position as join keys; sites repeat across datasets. Score is the probability of label 1. These files can be aligned with other models using the same saved folds for stacking.

Each `run/fold_k/` saves weights, preprocessing, training history, learning curves and fit summary. `oof_metrics.csv` reports combined OOF ROC AUC, trapezoidal PR AUC and AP by dataset and pooled; `fold_metrics.csv` provides per-fold results. Manifests record hashes, configuration provenance, fold counts, overlap audits and model reload checks.

Historical hyperparameter/duration tuning used this merged population. These predictions are out of fold for weight and scaler fitting, but their metrics are not an independent or nested-CV final performance estimate. Keep the separate test set for final evaluation.

Independent verification: `python verify_oof.py`. The completed run passed model reload, training-only scaler, fold coverage and input hash checks; see `run/verification.json`. The initial export comparison used overly strict CSV precision tolerance; this was corrected and the existing trained models and outputs were independently verified without retraining.
