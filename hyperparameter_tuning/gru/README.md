# GRU hyperparameter-selection evidence

These files document historical selection and subsequent use of the selected settings. They are not a complete candidate-search archive.

| File | What it establishes |
|---|---|
| [configuration_selection.json](configuration_selection.json) | Selection by mean fold trapezoidal PR-AUC, then ROC-AUC and weight-decay tie-breaks; 50-epoch tuning limit, patience 5 and minimum improvement 0.001; final duration of 32 epochs from the rounded median best epoch |
| [oof_run_manifest.json](oof_run_manifest.json) | Subsequent five-fold OOF run used the historical round-6 configuration for 32 fixed epochs without OOF early stopping; includes the original configuration-source path and hash |
| [Saved model configuration](../../models/gru/selected_config.json) | Hidden size 32, dropout 0, learning rate 0.0003, weight decay 0, batch size 256 and 32 epochs |
| [Final fit summary](../../models/gru/fit_summary.json) | Final fit completed 32 epochs with positive-class weight 1 |

The last two records are linked rather than duplicated here. Historical `class_weighting: true` with `alpha: 0` gave an effective positive-class weight of 1, equivalent to unweighted loss.

The current `train.py` uses fixed defaults unless `--tune` is requested. Its current candidate search is separate from the incomplete historical evidence in this folder.
