# DSA4262 DataHelix

This repository contains data-cleaning code and notebooks for comparing
m6A prediction approaches, including recurrent neural networks, logistic
regression, and the pretrained [m6Anet](https://github.com/GoekeLab/m6anet)
inference workflow.

## Project files and data

The workspace used by the commands below has these directories next to each
other:

```text
workspace/
├── DSA4262-datahelix/     # this repository
├── datasets/              # supplied datasets and labels
└── m6anet_bench/          # m6Anet preparation/evaluation scripts and results
```

The `datasets/` and `m6anet_bench/` directories are separate from this Git
repository. Obtain the project datasets and benchmark scripts from the course
distribution if they are not already present. The commands assume the directory
names and file layout below; set the paths to match your workspace if needed.

| Dataset | Signal JSON used for m6Anet | Ground-truth labels |
|---|---|---|
| `dataset0` | `datasets/data0/dataset0_json/dataset0.json` | `datasets/data0/data.info.labelled` |
| `dataset1` | `datasets/data1/dataset1_json/dataset1.json` | `datasets/data1/data.info` |
| `dataset2` | `datasets/data2/dataset2_json/dataset2.json` | `datasets/data2/data.info` |

These JSON files already contain processed signal features. This workflow does
not start from raw FAST5/POD5 signal files and does not require running
`nanopolish eventalign` or `m6anet dataprep` on these inputs. The benchmark
preparation script converts the supplied JSON records into the `data.json` and
`data.info` files expected by m6Anet.

## 1. Install and start Conda

If Conda is not installed on Linux x86-64, install Miniforge (recommended) in
your home directory:

```bash
cd "$HOME"
curl -L -o Miniforge3.sh \
  https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3.sh -b -p "$HOME/miniforge3"
"$HOME/miniforge3/bin/conda" init bash
source "$HOME/.bashrc"
```

If you already use Conda or Miniforge, skip the installer and open a new
terminal (or source your existing Conda initialization). Start Conda and check
that it is available:

```bash
conda --version
```

For macOS or Windows, download the matching Miniforge installer from the
[Miniforge releases page](https://github.com/conda-forge/miniforge/releases).

## 2. Create an m6Anet environment and install m6Anet

m6Anet 2.1.0 requires Python 3.7 or 3.8; use Python 3.8 in a dedicated
environment. The commands below install the Bioconda package and its
dependencies:

```bash
conda create -n m6anet -c conda-forge -c bioconda \
  python=3.8 m6anet=2.1.0 -y
conda activate m6anet
m6anet --help
m6anet inference --help
```

Alternatively, install the upstream source release inside the same Python 3.8
environment:

```bash
git clone --branch v2.1.0 --depth 1 https://github.com/GoekeLab/m6anet.git
python -m pip install ./m6anet
```

Do not install m6Anet in the Python 3.11 environment used for the cleaning
scripts: the current m6Anet release declares Python `<3.9`.

## 3. Set workspace paths

Run the remaining commands from the common parent directory containing
`DSA4262-datahelix/`, `datasets/`, and `m6anet_bench/`. Change the first `cd`
to your workspace location:

```bash
cd /path/to/workspace
export WORKSPACE="$PWD"
export DATA_DIR="$WORKSPACE/datasets"
export BENCH_DIR="$WORKSPACE/m6anet_bench"
export PREPARED_DIR="$BENCH_DIR/input"
export DATASET2_LABELS="$BENCH_DIR/dataset2_binary_labels.csv"
```

Keep using this shell session for the commands below, or repeat these exports
after opening a new terminal.

Confirm that the inputs and benchmark scripts are in place:

```bash
ls "$DATA_DIR/data0/data.info.labelled"
ls "$DATA_DIR/data1/data.info" "$DATA_DIR/data2/data.info"
ls "$BENCH_DIR/prepare_m6anet_inputs.py" "$BENCH_DIR/evaluate_m6anet.py"
```

## 4. Prepare the data for m6Anet

The prepared benchmark inputs are already under `m6anet_bench/input/` in the
supplied workspace. If they exist and contain `data.json` and `data.info`, skip
to step 5. To prepare a fresh set, use a new output root; the script intentionally
refuses to overwrite prepared datasets or label files:

```bash
export PREPARED_DIR="$BENCH_DIR/input_rebuilt"
python "$BENCH_DIR/prepare_m6anet_inputs.py" \
  --data-root "$DATA_DIR" \
  --out-root "$PREPARED_DIR" \
  --datasets dataset0 dataset1 dataset2 \
  --dataset2-label-output "$BENCH_DIR/dataset2_binary_labels_rebuilt.csv"
```

The script validates each site and writes m6Anet-compatible files at
`$PREPARED_DIR/dataset0/`, `$PREPARED_DIR/dataset1/`, and
`$PREPARED_DIR/dataset2/`. It requires the DRACH motif and at least 20 reads
for a site to be eligible for inference. The extra read identifier is a
synthetic row number within that site; it does not link the same physical
nanopore read across sites.

After preparing with the command above, use its new dataset2 label file for
evaluation:

```bash
export DATASET2_LABELS="$BENCH_DIR/dataset2_binary_labels_rebuilt.csv"
```

Dataset2 labels are modification fractions, not binary classes. The default
conversion makes `label=1` when the fraction is greater than zero and `label=0`
otherwise, preserving the original fraction in the output CSV. If your
scientific protocol defines a different cutoff, choose it explicitly, for
example:

```bash
python "$BENCH_DIR/prepare_m6anet_inputs.py" \
  --data-root "$DATA_DIR" \
  --out-root "$PREPARED_DIR" \
  --datasets dataset2 \
  --labels-only \
  --dataset2-rule threshold \
  --dataset2-threshold 0.5 \
  --dataset2-label-output "$BENCH_DIR/dataset2_binary_labels_ge_0.5.csv"
export DATASET2_LABELS="$BENCH_DIR/dataset2_binary_labels_ge_0.5.csv"
```

Use the same label rule when evaluating results. Ground-truth labels are only
used after inference; they are not passed to m6Anet.

## 5. Smoke-test m6Anet inference

First activate the environment and test the smallest supplied dataset.
Dataset2 has 1,323 sites and is a convenient initial run. A small number of
iterations checks that the installation and prepared input work; its scores
are only a quick smoke test and will be less stable than a full run.

```bash
conda activate m6anet
export RUN_ROOT="$BENCH_DIR/runs/smoke-test-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$RUN_ROOT/output/dataset2"

m6anet inference \
  --input_dir "$PREPARED_DIR/dataset2" \
  --out_dir "$RUN_ROOT/output/dataset2" \
  --n_processes 1 \
  --num_iterations 5
```

For a rebuilt input, set `PREPARED_DIR="$BENCH_DIR/input_rebuilt"` instead.
Inference writes `data.site_proba.csv` (site-level probabilities) and
`data.indiv_proba.csv` (per-read probabilities) under the run output directory.
Each run uses a separate output directory so existing benchmark results are
not overwritten.

## 6. Run inference on all datasets

Once the smoke test succeeds, run the full benchmark. The upstream recommended
setting of 1,000 sampling iterations is more stable but takes longer; lower
`--n_processes` if memory is limited.

```bash
export RUN_ROOT="$BENCH_DIR/runs/full-1000-iterations-$(date +%Y%m%d-%H%M%S)"

for dataset in dataset0 dataset1 dataset2; do
  mkdir -p "$RUN_ROOT/output/$dataset"
  m6anet inference \
    --input_dir "$PREPARED_DIR/$dataset" \
    --out_dir "$RUN_ROOT/output/$dataset" \
    --n_processes 4 \
    --num_iterations 1000
done
```

The default pretrained model is the human RNA002/HCT116 model. Select a
different upstream model only when it matches the sequencing chemistry and
organism, using m6Anet's `--pretrained_model` option (for example
`arabidopsis_RNA002` or `HEK293T_RNA004`).

## 7. Evaluate predictions against the labels

Evaluate dataset0 and dataset1:

```bash
python "$BENCH_DIR/evaluate_m6anet.py" \
  --bench-root "$RUN_ROOT" \
  --data-root "$DATA_DIR" \
  --datasets dataset0 dataset1 \
  --out-dir "$RUN_ROOT/evaluation_dataset01"
```

Evaluate all three datasets, providing the explicit binary labels for dataset2.
Use the labels created during preparation; update the path if you selected a
different binary rule:

```bash
python "$BENCH_DIR/evaluate_m6anet.py" \
  --bench-root "$RUN_ROOT" \
  --data-root "$DATA_DIR" \
  --datasets dataset0 dataset1 dataset2 \
  --dataset2-labels "$DATASET2_LABELS" \
  --out-dir "$RUN_ROOT/evaluation_all"
```

The evaluator matches predictions and labels by transcript ID and transcript
position. It writes per-dataset matched sites, pooled matched sites,
`m6anet_metrics.csv`, and `evaluation_metadata.json`. Metrics include ROC AUC,
trapezoidal PR AUC, and average precision (PR AP). The combined score pools
matched rows; it is not an average of the per-dataset AUC values. For a fair
comparison with another model, compare predictions on the same dataset and
matched sites. Predictions from a dataset used to train a model are training
performance, not an independent test result.

## Optional: start from raw nanopore reads instead

This is not needed for the supplied JSON datasets. For new raw-read data, the
upstream m6Anet workflow first requires Nanopolish `eventalign.txt`, generated
from basecalled reads, a sorted BAM, and a transcript reference FASTA:

```bash
nanopolish eventalign \
  --reads reads.fastq \
  --bam reads.sorted.bam \
  --genome transcript.fa \
  --scale-events \
  --signal-index \
  --summary summary.txt \
  --threads 4 > eventalign.txt

m6anet dataprep \
  --eventalign eventalign.txt \
  --out_dir m6anet_raw_input \
  --n_processes 4

m6anet inference \
  --input_dir m6anet_raw_input \
  --out_dir m6anet_raw_output \
  --n_processes 4 \
  --num_iterations 1000
```

Install and configure Nanopolish separately before running this optional
workflow. See the [m6Anet quick start](https://m6anet.readthedocs.io/en/latest/quickstart.html)
and [installation guide](https://m6anet.readthedocs.io/en/latest/installation.html)
for current upstream requirements and options.

## Python environment for cleaning and notebooks

The standalone data-cleaning code uses Python 3.11 and pinned dependencies.
Create a separate environment when running it (do not replace the m6Anet
environment):

```bash
conda create -n datahelix-clean python=3.11 pip -y
conda activate datahelix-clean
python -m pip install -r "$WORKSPACE/DSA4262-datahelix/data_cleaning/requirements.txt"
python "$WORKSPACE/DSA4262-datahelix/data_cleaning/clean_data.py" --help
python "$WORKSPACE/DSA4262-datahelix/data_cleaning/check_cleaning.py"
```

See [`data_cleaning/DATA_PREPARATION.md`](data_cleaning/DATA_PREPARATION.md)
for the cleaning input format, commands for the supplied datasets, and output
files. The notebooks are under `logreg/`, `lstmmodel/`, and `rnnmodel/`;
their framework-specific requirements are separate from the m6Anet inference
setup above.