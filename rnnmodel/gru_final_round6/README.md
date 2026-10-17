# Final GRU: full merged training fit, round-6 configuration

One fresh GRU trained on all 172,274 original train + validation rows. The standardized fold column is metadata and is not used as a feature or to hold out rows during this final fit. Test labels/features are excluded; only test metadata is loaded to audit separation. Original OOF models and predictions are unchanged.

Configuration: hidden size 32, dropout 0, AdamW learning rate 0.0003, weight decay 0, batch size 256, alpha 0 (unweighted BCE), seed 42, gradient clipping 1. Exactly 32 fixed epochs, selected from historical round-6 tuning; no new tuning or early stopping. configuration_selection.json preserves the historical selection, including its 50-epoch tuning limit; the actual final fit uses 32 epochs.

Reproduce from this directory (use a new output directory to avoid overwriting):

```bash
python train_gru.py --data-config datasets.json --output-dir model --mode fixed --epochs 32 --max-epochs 32 --seed 42
```

Use model/model.pt with model/preprocessing.json and this folder's gru_common.py for inference. model/source contains a training-source snapshot. model/training_rows.csv.gz records exact fitted rows. run_manifest.json and split_audit.json record completion, hashes, settings, row counts and test separation. Training metrics under model/ are apparent fit metrics, not held-out performance. No test evaluation is performed during this task.

Run `python verify_final.py` to independently check the saved model, full training-row coverage, preprocessing, source hashes and exported predictions. Verification results are written to `model/verification.json`.
