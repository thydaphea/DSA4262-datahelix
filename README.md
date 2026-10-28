# Ensemble Machine Learning for the Prediction of m6A RNA Modifications from Direct RNA Sequencing Data

This project predicts RNA N6-methyladenosine (m6A) modification sites using m6Anet-processed nanopore direct RNA sequencing data. The pipeline incorporates three base models—Gradient Boosting (GB), Multilayer Perceptron (MLP), and Gated Recurrent Unit (GRU)—and a logistic regression stacking ensemble, which serves as the final prediction model.

- **`train.py`**: Performs data cleaning, label matching, preserve supplied splits or generate gene-grouped splits, optionally tune hyperparameters, generate five-fold out-of-fold (OOF) predictions, fit final models and the logistic ensemble, and evaluate held-out sites.
- **`predict.py`**: Generates site-level m6A probabilities from unlabelled signal data using pretrained or newly trained models.

## Table of Contents

1. [Quick Start: Predict on the Sample Test Dataset](#1-quick-start-predict-on-the-sample-test-dataset)
   - [Step 1: Connect to an AWS Ubuntu Instance via Ronin](#step-1-connect-to-an-aws-ubuntu-instance-via-ronin)
   - [Step 2: Clone the Repository and Set Up the Environment](#step-2-clone-the-repository-and-set-up-the-environment)
   - [Step 3: Run Predictions on the Sample Dataset](#step-3-run-predictions-on-the-sample-dataset)
2. [Predicting Other Datasets](#2-predicting-other-datasets)
   - [2.1 Input Data Format](#21-input-data-format)
   - [2.2 Run Predictions](#22-run-predictions)
   - [2.3 Prediction Outputs](#23-prediction-outputs)
3. [Training New Models](#3-training-new-models)
   - [3.1 Prepare Training Data](#31-prepare-training-data)
   - [3.2 Run Model Training](#32-run-model-training)
   - [3.3 Use Newly Trained Models](#33-use-newly-trained-models)
4. [Model and Preprocessing Details](#4-model-and-preprocessing-details)
5. [Command-Line Reference](#5-command-line-reference)
6. [Troubleshooting](#6-troubleshooting)
7. [License and Citation](#7-license-and-citation)


## 1. Quick Start: Predict on the Sample Test Dataset

This section guides users through setting up the environment and generating predictions using the provided sample test dataset and pretrained models. No model training is required.

### Step 1: Connect to an AWS Ubuntu Instance via Ronin

Launch an Ubuntu instance on AWS Ronin and connect via SSH.

**System requirements:**

- **Operating system:** 64-bit x86 Linux (tested on Ubuntu Server 26.04 LTS)
- **Python:** 3.11, installed using `uv` and configured in a virtual environment
- **Hardware:** CPU only; no GPU required
- **Internet access:** Required for installing Python and project dependencies

**Recommended resources for full-data training:**

- CPU: 2 vCPUs
- Memory: 8 GB RAM
- Disk space: 5 GB available storage (estimated)

These are suggested starting allocations rather than verified minimum requirements.

**Tested environment:** This project was developed on Ubuntu Server 26.04 LTS with a 64 GB root volume. The root volume size reflects the development environment and is not a minimum software requirement.

### Step 2: Clone the Repository and Set Up the Environment

Install Python 3.11 using `uv`, without changing Ubuntu’s system Python.

```bash
# Install prerequisites
sudo apt-get update
sudo apt-get install -y git curl ca-certificates

# Install uv
curl -LsSf https://astral.sh/uv/install.sh -o /tmp/uv-install.sh
sh /tmp/uv-install.sh
export PATH="$HOME/.local/bin:$PATH"

# Clone the repository
git clone --branch main https://github.com/thydaphea/DSA4262-datahelix.git
cd DSA4262-datahelix

# Install Python and create the environment
uv python install 3.11
uv venv --python 3.11 --seed .venv
source .venv/bin/activate

# Install the pinned dependencies
python -m pip install --upgrade pip
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt

# Verify the installation
python --version
python -m pip check
```

The Python version should be `3.11.x`, and `pip check` should report no broken requirements.

In each new terminal session, enter the package directory and activate the environment before running training or prediction:

```bash
source .venv/bin/activate
```
### Step 3: Run Predictions on the Sample Dataset

The [sample test dataset](data/test_sample.json) contains 300 transcript positions sampled from Datasets 0–2, including transcript coordinates, RNA sequences and read-level signal measurements.

**Ground-truth labels are excluded.** Details of the sample's origin and composition are provided in the [provenance file](data/test_sample_provenance.json).

```bash
python predict.py \
  --data data/test_sample.json \
  --method ensemble \
  --output example_results/test_sample_predictions.csv
```

This generates site-level m6A modification probabilities using the pretrained logistic regression stacking ensemble.

The outputs are saved as:

- `example_results/test_sample_predictions.csv`: Predicted m6A probabilities
- `example_results/test_sample_predictions.audit.json`: Data cleaning and processing audit

Use `--model-dir PATH` to select another training run's weights.

## 2. Predicting Other Datasets

The same prediction pipeline can be applied to the original datasets or new m6Anet-processed datasets without retraining.

### 2.1 Input Data Format

Input files must follow the m6Anet JSONL structure, optionally gzip-compressed (`.json` or `.json.gz`).

Each record contains:

```text
transcript_id
  └── transcript_position
       └── seven-base RNA sequence
            └── N reads × 9 signal measurements
```
The nine measurements comprise dwell time, signal standard deviation, and signal mean at three positions: the central position (0) and its two flanking positions (−1 and +1).

Prediction does not require ground-truth labels, gene annotations or a split manifest. The pipeline automatically cleans and validates the input data.

### 2.2 Run Predictions

**Example: Predict on a new dataset**

```bash
python predict.py \
  --data /path/to/new_data.json.gz \
  --method ensemble \
  --output predictions/new_logistic.csv
```
This command uses the pretrained logistic regression stacking ensemble saved at `models/ensemble/logistic_regression.joblib`.

To use models from a separate training run, specify the model directory using `--model-dir`:

```bash
python predict.py \
  --data /path/to/new_data.json.gz \
  --model-dir results/my_run/models \
  --method ensemble \
  --output predictions/my_run_predictions.csv
```

### 2.3 Prediction Outputs

The prediction CSV contains one row per transcript position, with three columns:

| Column | Description |
| --- | --- |
| `transcript_id` | Transcript identifier |
| `transcript_position` | Position within the transcript |
| `score` | Predicted probability of m6A modification (0–1) |

**Interpreting Prediction Scores**

The score represents the model's estimated likelihood that a transcript position contains an m6A modification, based on its RNA sequence and nanopore signal measurements.
Higher scores indicate stronger predicted evidence of m6A modification.

**Important Considerations**

- Site-level predictions: Each score corresponds to a transcript position, not an individual sequencing read.

- Not modification fractions: A score of 0.92 does not mean that 92% of RNA molecules are modified at that position.

- No predefined classification threshold: The software outputs continuous scores rather than assigning modified/unmodified labels.

- Relative prioritisation: Users can rank transcript positions by score to identify candidate m6A sites for further investigation or experimental validation.

- Model uncertainty: Scores are model estimates, not experimental confirmation of m6A modification. Their numerical values should not be interpreted as calibrated probabilities unless calibration has been established.


## 3. Training New Models

This section is intended for users who wish to retrain the models, perform hyperparameter tuning or evaluate performance using labelled datasets.

**The unlabelled sample dataset in Section 1 cannot be used for training.**

### 3.1 Prepare Training Data

Create a JSON configuration file specifying the signal data and corresponding label files.

Example `new_training_inputs.json`:

```json
{
  "datasets": [
    {
      "dataset_id": "my_dataset",
      "data": "raw/my_dataset.json.gz",
      "labels": "labels/my_dataset.csv"
    }
  ]
}
```
Label files must contain `transcript_id`, `transcript_position` and `label`. A `gene_id` column is required when generating new gene-grouped splits.

Fractional modification labels are converted to binary labels: 0 remains 0, while any value greater than 0 becomes 1.

### 3.2 Run Model Training

The training pipeline produces cleaned data, model checkpoints, out-of-fold predictions, final model weights, test predictions, evaluation metrics and provenance records.

The training pipeline supports the following split configurations:

| Configuration | Split behaviour |
|---|---|
| No `"splits"` entry | Automatically generates gene-grouped train, validation, and test splits |
| Includes `"splits": "path/to/new_splits.csv.gz"` | Reuses the assignments in the supplied split manifest |
| Pass `--generate-splits` | Generates new grouped splits, even if a split manifest is specified in the configuration |

When generating new splits, use `--seed 42` to ensure reproducible assignments. If an existing split manifest is supplied without `--generate-splits`, its assignments are preserved.

#### Case 1: Train with Default Hyperparameters and Automatically Generated Splits

Use this option to train the models with the saved hyperparameter configurations. If no split manifest is provided, the pipeline automatically generates gene-grouped splits.

```bash
python train.py \
  --config new_training_inputs.json \
  --out-dir results/my_run \
  --seed 42
```

#### Case 2: Train with a Supplied Split Manifest

Use this option to train the models with predefined training, validation, and test splits instead of generating new ones. Refer to [`training_data/splits.csv.gz`](training_data/splits.csv.gz) for an example of the required split manifest format.

Specify the split manifest in your configuration file:

```json
{
  "splits": "path/to/new_splits.csv.gz",
  "datasets": [
    {
      "dataset_id": "my_dataset",
      "data": "raw_datasets/my_dataset.json.gz",
      "labels": "raw_datasets/my_dataset_labels.csv"
    }
  ]
}
```

Then run:

```bash
python train.py \
  --config new_training_inputs.json \
  --out-dir results/my_run
```

The pipeline preserves the supplied split assignments.

#### Case 3: Train with Newly Generated Splits and Hyperparameter Tuning

Use this option to generate new gene-grouped splits and perform hyperparameter tuning instead of relying on the saved hyperparameter configurations.

```bash
python train.py \
  --config new_training_inputs.json \
  --out-dir results/my_tuned_run \
  --generate-splits \
  --tune \
  --seed 42
```
The `--generate-splits` flag forces the creation of new splits, even if a split manifest is specified in the configuration. The `--tune` flag enables hyperparameter tuning.

Each output directory must be new and must not already exist.

Newly generated split assignments are saved as `splits.csv.gz` in the output directory.

### 3.3 Use Newly Trained Models

After training, follow the prediction instructions in **Section 2.2**, specifying the new model directory using `--model-dir`.

## 4. Model and Preprocessing Details

The prediction and training scripts share a standardized preprocessing pipeline. This includes sequence normalization, DRACH motif validation, removal of invalid reads and model-specific feature preparation.

The three base models use different input representations, while the logistic regression ensemble combines their predicted probabilities.

For full details of feature engineering, GB/MLP/GRU architectures, hyperparameter tuning, grouped cross-validation and logistic ensemble construction, see **[Model and Preprocessing Details](docs/model_details.md)**.

## 5. Command-Line Reference

See **[Command-Line Reference](docs/command_line_reference.md)** for all supported options, defaults and their meanings.

For a complete list of supported arguments and available prediction methods, run:

```bash
python predict.py --help
python train.py --help
```

## 6. Troubleshooting

| Issue | Suggested solution |
| --- | --- |
| `python: command not found` | Activate the Python 3.11 virtual environment. |
| `ModuleNotFoundError` | Install dependencies using `requirements.txt`. |
| File not found | Check the working directory and input file path. |
| scikit-learn version warning | Verify that the installed version matches the saved model requirements. |
| Invalid input data | Check the JSON structure, sequence format and signal measurements. |

## 7. License and Citation

Include the project's software license and citation information here.


### Software license

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

### How to Cite

If you use our models or ideas in your research or projects, please cite:

**APA-style citation:**

Sokun, T., Tan, E. S., Tan, J. W., & Tan, X. Y. (2026). *Ensemble Machine Learning for the Prediction of m6A RNA Modifications from Direct RNA Sequencing Data* [Computer software]. https://github.com/thydaphea/DSA4262-datahelix

**BibTeX:**

```bibtex
@software{datahelix2026,
  author = {Sokun, Thydaphea and Tan, Eng Shuen and Tan, Jia Wei and Tan, Xin Yu},
  title = {Ensemble Machine Learning for the Prediction of m6A RNA Modifications from Direct RNA Sequencing Data},
  year = {2026},
  url = {https://github.com/thydaphea/DSA4262-datahelix}
}
```
