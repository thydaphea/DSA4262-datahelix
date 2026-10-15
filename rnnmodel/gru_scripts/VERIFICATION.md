# Verification performed on 2026-10-15

- Seven automated tests passed using the existing `m6a-gru` Python environment.
- The CLI integration test exercised fixed-duration training, saved-model reload,
  all five early-stopping CV folds and full refit, prediction column names,
  train/evaluation score agreement, missing test labels, output overwrite
  prevention, changed training inputs, and artifact hash validation.
- Unit checks covered the 6,721-parameter architecture, standardization, feature
  ordering, train/test biological overlap, the dataset2 mixture grouping policy,
  checkpoint/patience semantics, malformed inputs, and metric edge cases.
- All three full cleaned datasets in `datasets.example.json` loaded successfully:
  172,274 training rows and 41,697 test rows, with 4,037 training groups and 1,023
  test groups. No shared train/test sites or biological groups were detected.
- The original round-1 five-fold assignments aligned successfully with these
  training rows and passed group separation checks.
- A two-epoch smoke run on small real-data samples (192 training rows and 96 test
  rows across the three datasets) completed training and separate evaluation.
  Per-dataset and pooled outputs were generated and a rendered plot inspected.
- Python syntax compilation passed.

The smoke runs test functionality only. They do not reproduce the full tuned
model's performance, and no full-dataset training run was launched for this
script-delivery task. Temporary smoke artifacts are separate from the scripts.
