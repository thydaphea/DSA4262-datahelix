# GB, MLP and GRU hyperparameter-tuning evidence

## Contents

- `gb/`: original 32-candidate validation history, original metrics/settings, selected configuration and a readable candidate-summary CSV.
- `mlp/`: original per-epoch tuning history, selected-run validation history, metrics/settings, selected configuration and candidate/stage summary CSV.
- `mlp/additional_experiments/`: selection records and validation histories for subsequent dropout, positive-loss-weight and epoch comparisons.
- `gru/`: historical configuration selection, selected settings, final-fit summary, subsequent OOF run manifest and source hashes; see [GRU evidence and limitations](gru/README.md). The full candidate search history is unavailable.


## GB and MLP Selections

| Search | Procedure | Winner | Pooled validation PR-AUC (trapezoidal) |
|---|---|---|---:|
| GB | 32 candidates; same held-out validation partition | Candidate 9: 63 leaves, 600 iterations, learning rate 0.03, minimum leaf size 10, L2 1 | 0.4795393435 |
| MLP | 12 configurations screened for 4 epochs; candidates 1, 9 and 12 trained up to 30 epochs with patience 6 | Candidate 9: hidden 256/128, dropout 0.5, learning rate 0.001, AdamW decay 0.0001, batch 256, unweighted loss; selected epoch 16 | 0.4927150286 |

Both winners match the fixed configurations in the final training script. The later final models were fitted on original training + validation data; they are different fitted artifacts from the original search checkpoints. These numbers are historical validation scores, not current final test scores. Refer to `../results/test/test_metrics.csv` for final test performance.

The original GB/MLP searches selected using pooled validation PR-AUC; it did not select by mean five-fold CV. The final pipeline subsequently uses the selected settings to produce five-fold OOF predictions. MLP parameters were varied together, so the broad search does not isolate their individual effects.

## GRU selection

The retained record specifies mean fold trapezoidal PR-AUC selection, followed by ROC-AUC and weight-decay tie-breaks. Selected settings are hidden size 32, dropout 0, learning rate 0.0003, decay 0, batch 256 and effectively unweighted loss. The final 32 epochs came from the rounded median best epoch. See [the GRU records](gru/README.md) for provenance and the distinction between historical tuning and subsequent fixed-setting OOF training. Candidate ranges, trial count and winning search score are not available.
