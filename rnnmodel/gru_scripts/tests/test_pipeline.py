"""Focused correctness and CLI integration checks; uses only tiny synthetic data."""
from pathlib import Path
import json
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gru_common import (  # noqa: E402
    CONFIG, EarlyStopping, POSITION_COLUMNS, SUMMARY_COLUMNS, SiteGRU,
    audit_split, fit_preprocessing, metric_row, raw_arrays, read_partition,
    transform, write_json,
)


def frame(split, n=50):
    rng = np.random.default_rng(42 if split == "train" else 99)
    rows = []
    for i in range(n):
        label = i % 2
        row = dict(dataset_id="demo", transcript_id=f"{split}_tx{i}",
                   transcript_position=100+i, sequence="AAGACTT", split=split,
                   group_id=f"gene:{split}_gene{i}", group_source="gene_id",
                   gene_id=f"{split}_gene{i}", label=label, n_reads_valid=20+i)
        row.update(zip(SUMMARY_COLUMNS, rng.normal(label, .4, 45)))
        rows.append(row)
    return pd.DataFrame(rows)


class CoreTests(unittest.TestCase):
    def test_exact_model_and_preprocessing(self):
        self.assertEqual(sum(p.numel() for p in SiteGRU().parameters()), 6721)
        train = frame("train")
        prep = fit_preprocessing(train)
        seq, _, cov = raw_arrays(train)
        actual, auxiliary = transform(train, prep)
        np.testing.assert_allclose(actual.reshape(-1, 45), StandardScaler().fit_transform(seq), rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(auxiliary[:, -1:], StandardScaler().fit_transform(cov), rtol=1e-6, atol=1e-6)
        # Reading held-out features must not mutate the train-fitted scaler.
        saved = json.dumps(prep)
        shifted = frame("test")
        shifted[SUMMARY_COLUMNS] += 100
        transform(shifted, prep)
        self.assertEqual(saved, json.dumps(prep))
        self.assertEqual(actual.shape, (50, 3, 15))
        self.assertEqual(auxiliary.shape, (50, 29))
        self.assertEqual(POSITION_COLUMNS[0][:3], ["minus1_dwell_mean", "minus1_signal_sd_mean", "minus1_signal_mean_mean"])

    def test_group_overlap_and_cross_dataset_links(self):
        train, test = frame("train"), frame("test")
        _, audit = audit_split(train, test)
        self.assertEqual(audit["shared_biological_groups"], 0)
        test.loc[0, "dataset_id"] = "another_dataset"
        test.loc[0, "gene_id"] = train.loc[0, "gene_id"]
        test.loc[0, "group_id"] = train.loc[0, "group_id"]
        with self.assertRaisesRegex(ValueError, "overlap"):
            audit_split(train, test)

    def test_mixture_exception_requires_matching_metadata(self):
        train, test = frame("train", 2), frame("test", 2)
        for part, offset in [(train, 0), (test, 1000)]:
            part["dataset_id"] = "dataset2"
            part["transcript_id"] = ["tx_id_0", "tx_id_1"]
            part["gene_id"] = pd.NA
            part["transcript_position"] += offset
            part["group_id"] = "dataset2:mixture_position:" + part.transcript_position.astype(str)
        with self.assertRaisesRegex(ValueError, "overlap"):
            audit_split(train, test, False)
        audit_split(train, test, True)
        test.loc[0, "group_id"] = "bad_group"
        with self.assertRaisesRegex(ValueError, "mixture-position"):
            audit_split(train, test, True)

    def test_early_stopping_saves_small_improvements(self):
        tracker = EarlyStopping(patience=2, min_delta=.001)
        self.assertEqual(tracker.update(.5, 1), (True, False))
        self.assertEqual(tracker.update(.5004, 2), (True, False))
        self.assertEqual(tracker.update(.5006, 3), (True, True))
        self.assertEqual(tracker.best_epoch, 3)

    def test_metrics_edge_cases(self):
        result = metric_row([0, 1, np.nan], [.1, .9, .5], .5)
        self.assertEqual(result["ROC_AUC"], 1)
        self.assertEqual(result["PR_AUC_trapezoidal"], 1)
        self.assertEqual(result["n_missing_labels"], 1)
        self.assertEqual(result["TP"], 1)
        self.assertEqual(metric_row([1, 1], [.2, .8], .5)["status"], "SINGLE_CLASS")
        self.assertEqual(metric_row([np.nan], [.2], .5)["status"], "NO_LABELS")
        with self.assertRaises(ValueError):
            metric_row([.5], [.2], .5)

    def test_invalid_input_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "train.csv"
            data = frame("train", 4)
            pd.concat([data, data.iloc[[0]]]).to_csv(p, index=False)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                read_partition(p, "demo", "train")
            data["label"] = data.label.astype(float)
            data.loc[0, "label"] = .2
            data.to_csv(p, index=False)
            with self.assertRaisesRegex(ValueError, "binary"):
                read_partition(p, "demo", "train")


class CliTests(unittest.TestCase):
    def call(self, *args, ok=True):
        result = subprocess.run([sys.executable, *map(str, args)], cwd=ROOT,
                                capture_output=True, text=True, timeout=180)
        if ok:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def test_training_reload_predictions_cv_and_tamper_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train, test = frame("train", 100), frame("test", 12)
            train.to_csv(root / "train.csv", index=False)
            test.to_csv(root / "test.csv.gz", index=False)
            spec = {"datasets": [{"dataset_id": "demo", "train_file": "train.csv", "test_file": "test.csv.gz"}]}
            write_json(root / "data.json", spec)
            cfg = root / "data.json"
            model = root / "model"
            self.call("train_gru.py", "--data-config", cfg, "--output-dir", model,
                      "--epochs", 2, "--device", "cpu", "--threads", 1)
            manifest = json.loads((model / "run_manifest.json").read_text())
            self.assertEqual(manifest["status"], "COMPLETE")
            self.assertEqual(manifest["refit_epochs"], 2)
            self.assertFalse(manifest["test_evaluated"])
            ev = root / "evaluation"
            self.call("evaluate_gru.py", "--data-config", cfg, "--model-dir", model,
                      "--output-dir", ev, "--device", "cpu", "--threads", 1)
            predictions = pd.read_csv(ev / "predictions_test_demo.csv")
            self.assertEqual(list(predictions), ["transcript_id", "transcript_position", "score"])
            self.assertEqual(len(predictions), len(test))
            self.assertTrue(predictions.score.between(0, 1).all())
            fit_scores = pd.read_csv(model / "train_outputs/predictions_train_apparent_demo.csv")
            reload_scores = pd.read_csv(ev / "predictions_train_apparent_demo.csv")
            np.testing.assert_array_equal(fit_scores.score, reload_scores.score)
            self.assertGreater((ev / "evaluation_test_demo.png").stat().st_size, 1000)
            summary = pd.read_csv(ev / "performance_summary.csv")
            self.assertEqual(len(summary), 4)
            # Existing outputs must not be overwritten.
            self.call("evaluate_gru.py", "--data-config", cfg, "--model-dir", model,
                      "--output-dir", ev, ok=False)
            # Entirely unlabeled tests still produce all requested predictions.
            test.drop(columns="label").to_csv(root / "test.csv.gz", index=False)
            self.call("evaluate_gru.py", "--data-config", cfg, "--model-dir", model,
                      "--output-dir", root / "unlabeled", "--threads", 1)
            results = pd.read_csv(root / "unlabeled/performance_summary.csv")
            self.assertTrue(results.loc[results.split.eq("test"), "status"].eq("NO_LABELS").all())
            # A replacement training file must be rejected before evaluation.
            original = (root / "train.csv").read_bytes()
            train.loc[0, "n_reads_valid"] += 1
            train.to_csv(root / "train.csv", index=False)
            result = self.call("evaluate_gru.py", "--data-config", cfg, "--model-dir", model,
                               "--output-dir", root / "bad_eval", ok=False)
            self.assertIn("exactly match", result.stderr)
            (root / "train.csv").write_bytes(original)
            # Exercise every CV fold, checkpoint restoration, and final refit.
            cv = root / "cv"
            self.call("train_gru.py", "--data-config", cfg, "--output-dir", cv,
                      "--mode", "cv-refit", "--max-epochs", 1, "--threads", 1, "--device", "cpu")
            metrics = pd.read_csv(cv / "cv/fold_metrics.csv")
            self.assertEqual(len(metrics), 5)
            self.assertTrue(metrics.selected_epoch.eq(1).all())
            self.assertEqual(json.loads((cv / "run_manifest.json").read_text())["refit_epochs"], 1)
            # Corrupt scaler hash is detected before loading predictions.
            with (model / "preprocessing.json").open("a") as handle:
                handle.write(" ")
            result = self.call("evaluate_gru.py", "--data-config", cfg, "--model-dir", model,
                               "--output-dir", root / "corrupt", ok=False)
            self.assertIn("hash mismatch", result.stderr)


if __name__ == "__main__":
    unittest.main()
