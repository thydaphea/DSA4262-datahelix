# Shared model results

`gb/`, `mlp/`, and `mil/` each contain one model fitted jointly on training portions from dataset0, dataset1, and dataset2. They use identical group partitions.

Use [evaluation_metrics.csv](evaluation_metrics.csv) for the 12-row comparison: each dataset and pooled held-out evaluation for each algorithm. Regenerate it with `python evaluate_unified.py --compare --results-root results`.

Each model folder contains its artifact, validation history, split manifest, metrics and audits. `evaluation/` contains test-only predictions; `submission/` contains three complete dataset predictions. Choose one model's three files for submission and rename them using the registered team name.
