# GRU hyperparameter-selection evidence

These records document the historical selected configuration and its use in subsequent fitting. They are not a complete candidate-search archive.

| File | What it establishes |
|---|---|
| [configuration_selection.json](configuration_selection.json) | Selection rule: mean fold trapezoidal PR-AUC, then ROC-AUC and weight-decay tie-breaks; tuning limit 50 epochs, patience 5, minimum improvement 0.001; final duration from rounded median best epoch, 32. |
| [selected_config.json](selected_config.json) | Final hidden size 32, dropout 0, learning rate 0.0003, weight decay 0, batch size 256, 32 epochs. |
| [final_fit_summary.json](final_fit_summary.json) | Final fit completed 32 epochs with positive-class weight 1 (effectively unweighted). |
| [oof_run_manifest.json](oof_run_manifest.json) | Subsequent five-fold OOF run reused the historical round-6 configuration for 32 fixed epochs, without OOF early stopping; includes the original configuration-source path and hash. This is a fitting audit, not search results. |
| [provenance.json](provenance.json) | Source paths and SHA-256 hashes for these exact copied records. |

The historical configuration has `class_weighting: true` with `alpha: 0`; the recorded effective positive-class weight is 1. The current trainer implements the equivalent unweighted loss.

The referenced original search directory is unavailable on this machine. Candidate ranges, trial count, individual fold scores and the winning tuning score cannot be verified from these records. No search scores have been reconstructed or invented. The current root `train.py` uses fixed settings and does not rerun GRU tuning. Historical tuning used the merged development population, so the subsequent OOF evaluation is not fully nested.
