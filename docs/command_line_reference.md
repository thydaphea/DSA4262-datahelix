## All command-line arguments
Both scripts accept `-h` / `--help` to display usage.

### `predict.py`
| Argument | Meaning | Default |
|---|---|---|
| `--data PATH` | Raw JSONL or JSONL.gz | Required |
| `--output PATH` | New prediction CSV; parent directories created automatically | Required |
| `--model-dir PATH` | Directory containing saved model subfolders | Package `models/` |
| `--method NAME` | `ensemble` (logistic regression), `mlp`, `gb` or `gru` | `ensemble` |
| `--threads N` | Positive CPU thread count | `1` |

### `train.py`
| Argument | Meaning | Default |
|---|---|---|
| `--config PATH` | JSON listing raw data, labels and an optional split manifest | Package `training_data/training_inputs.json` |
| `--out-dir PATH` | New run directory; must not already exist | Required |
| `--generate-splits` | Generate a new gene-grouped test split and five development folds, even if a manifest is configured | Off; generated automatically if no manifest is supplied |
| `--tune` | Run GB, MLP and GRU hyperparameter searches on development data | Off |
| `--prepare-only` | Clean/join data and verify assignments without model fitting | Off |
| `--methods NAME ...` | One or more of `mlp`, `gb`, `gru` | All three |
| `--skip-ensemble` | Skip logistic ensemble fitting; otherwise all three base models are required | Off |
| `--threads N` | Positive CPU thread count | `1` |
| `--seed N` | Random seed; neural fold fits use seed + fold, final fits use seed | `42` |

Default architecture and optimizer settings are constants in `train.py`; `--tune` searches candidate settings and saves the selected configurations. These settings are not separate CLI arguments.

