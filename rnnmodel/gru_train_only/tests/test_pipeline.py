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
                   transcript_position=100+i, sequence="AAGACTT", split="train", original_split="train" if split == "train" else "validation",
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


class SplitTests(unittest.TestCase):
    def test_original_split_filter_and_unknown_values(self):
        from gru_common import load_data, load_spec
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tr, va = frame("train", 20), frame("test", 10)
            # Held-out values must not affect training preprocessing or labels.
            va[SUMMARY_COLUMNS] = 1e8
            va["label"] = float("nan")
            combined = pd.concat([tr, va], ignore_index=True)
            combined.to_csv(root / "source.csv", index=False)
            write_json(root / "data.json", {"datasets": [{"dataset_id": "demo", "source_file": "source.csv"}]})
            train, validation, _ = load_data(load_spec(root / "data.json"), validation_metadata_only=True)
            self.assertEqual(len(train), 20)
            self.assertEqual(len(validation), 10)
            self.assertNotIn("label", validation)
            self.assertNotIn(SUMMARY_COLUMNS[0], validation)
            actual, expected = fit_preprocessing(train), fit_preprocessing(tr)
            for feature in ["sequence", "coverage"]:
                for stat in ["mean", "scale"]:
                    np.testing.assert_allclose(actual[feature][stat], expected[feature][stat], rtol=1e-12, atol=1e-12)
            self.assertTrue(train.original_split.eq("train").all())
            self.assertTrue(validation.original_split.eq("validation").all())
            combined.loc[0, "original_split"] = "typo"
            combined.to_csv(root / "source.csv", index=False)
            with self.assertRaisesRegex(ValueError, "original_split"):
                load_data(load_spec(root / "data.json"))

    def test_cli_train_reload_and_internal_cv(self):
        from gru_common import load_data, load_spec
        from train_gru import get_folds
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tr, va = frame("train", 100), frame("test", 12)
            pd.concat([tr, va]).to_csv(root / "source.csv", index=False)
            write_json(root / "data.json", {"datasets": [{"dataset_id": "demo", "source_file": "source.csv"}]})
            cfg = root / "data.json"
            train, validation, _ = load_data(load_spec(cfg))
            groups, _ = audit_split(train, validation)
            for ti, vi in get_folds(train, groups, 42):
                self.assertTrue(train.iloc[ti].original_split.eq("train").all())
                self.assertTrue(train.iloc[vi].original_split.eq("train").all())
            def run(*args, ok=True):
                p = subprocess.run([sys.executable, *map(str,args)], cwd=ROOT, capture_output=True, text=True, timeout=180)
                self.assertEqual(p.returncode == 0, ok, p.stdout+p.stderr)
            model = root / "model"
            run("train_gru.py", "--data-config", cfg, "--output-dir", model, "--epochs", 2, "--threads", 1)
            ev = root / "evaluation"
            run("evaluate_gru.py", "--data-config", cfg, "--model-dir", model, "--output-dir", ev, "--threads", 1)
            pred = pd.read_csv(ev / "predictions_validation_demo.csv")
            self.assertEqual(list(pred), ["transcript_id", "transcript_position", "score"])
            self.assertEqual(pred.transcript_id.tolist(), va.transcript_id.tolist())
            self.assertTrue(pred.score.between(0,1).all())
            fitted = pd.read_csv(model / "train_outputs/predictions_train_apparent_demo.csv")
            reloaded = pd.read_csv(ev / "predictions_train_apparent_demo.csv")
            np.testing.assert_array_equal(fitted.score, reloaded.score)
            run("train_gru.py", "--data-config", cfg, "--output-dir", root / "cv_model", "--mode", "cv-refit", "--max-epochs", 1, "--threads", 1)
            assignments = pd.read_csv(root / "cv_model/cv/cv_fold_assignments.csv.gz")
            self.assertEqual(set(assignments.transcript_id), set(tr.transcript_id))
            self.assertFalse(set(assignments.transcript_id) & set(va.transcript_id))
            run("evaluate_gru.py", "--data-config", cfg, "--model-dir", model, "--output-dir", ev, "--threads", 1, ok=False)
            with open(model / "preprocessing.json", "a") as f:
                f.write(" ")
            run("evaluate_gru.py", "--data-config", cfg, "--model-dir", model, "--output-dir", root / "tampered", "--threads", 1, ok=False)

if __name__ == "__main__":
    unittest.main()
