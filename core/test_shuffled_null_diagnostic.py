"""Unit tests for standalone shuffled-null diagnostic script."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd

from core.test_concept_embedding_decoders import _synthetic_embedding_payload
from scripts.run_shuffled_null_diagnostic import (
    compute_diagnostics,
    create_shuffled_null_payload,
    load_job_artifacts,
    main,
    plot_diagnostics,
    resolve_target_parcels,
)


def _create_mock_batch_dir(root: Path) -> Path:
    batch_dir = root / "20261019_mock_selected_parcel_subsets"
    batch_dir.mkdir(parents=True)

    # Synthetic payload with 5 concepts
    base = _synthetic_embedding_payload()
    classes = [f"concept_{i}" for i in range(5)]
    y = np.repeat(classes, 12)
    embeddings = np.repeat(np.eye(5, dtype=np.float32), 12, axis=0)

    # 2 ROIs: roi_A (informative) and roi_B (noise)
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, size=(60, 4)).astype(np.float32)
    # Give roi_A strong signal
    for idx, c in enumerate(classes):
        X[idx * 12 : (idx + 1) * 12, 0] += idx * 3.0

    trial_df = pd.DataFrame(
        {
            "trial_index": np.arange(60),
            "concept_label": y,
            "canonical_stimulus": y,
            "embedding_key": y,
        }
    )
    feature_meta_df = pd.DataFrame(
        {
            "feature_index": np.arange(4),
            "feature": ["f1", "f2", "f3", "f4"],
            "roi": ["roi_A", "roi_A", "roi_B", "roi_B"],
        }
    )
    payload = {
        "X": X,
        "X_tensor": X,
        "y": y,
        "embeddings": embeddings,
        "trial_df": trial_df,
        "feature_meta_df": feature_meta_df,
        "feature_names": ["f1", "f2", "f3", "f4"],
        "roi_names": ["roi_A", "roi_B"],
        "window_start": 0.15,
        "window_end": 0.45,
    }
    joblib.dump(payload, batch_dir / "source_features.joblib", compress=1)

    options = {
        "decoder_type": "embedding_contrastive",
        "split_mode": "zero_shot",
        "zero_shot_unseen_fraction": 0.4,  # 3 train concepts, 2 unseen test concepts
        "zero_shot_random_state": 42,
        "epochs": 2,
        "batch_size": 16,
        "selected_parcellations": ["roi_A", "roi_B"],
        "feature_space": "source",
        "avg_trials": 1,
    }
    (batch_dir / "settings_used.json").write_text(json.dumps(options, indent=2))

    manifest = {
        "best_subset_id": 1,
        "best_parcels": ["roi_A"],
        "best_top1_accuracy": 0.85,
        "best_mpr": 0.88,
        "candidate_pool": ["roi_A", "roi_B"],
    }
    (batch_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    comparison = pd.DataFrame(
        [
            {
                "subset_id": 1,
                "parcels": "roi_A",
                "n_parcels": 1,
                "status": "complete",
                "top1_accuracy": 0.85,
                "mpr": 0.88,
                "top5_accuracy": 1.0,
            },
            {
                "subset_id": 2,
                "parcels": "roi_B",
                "n_parcels": 1,
                "status": "complete",
                "top1_accuracy": 0.50,
                "mpr": 0.52,
                "top5_accuracy": 1.0,
            },
        ]
    )
    comparison.to_csv(batch_dir / "subset_comparison.csv", index=False)
    return batch_dir


class ShuffledNullDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.batch_dir = _create_mock_batch_dir(self.root)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_load_job_artifacts(self):
        artifacts = load_job_artifacts(self.batch_dir)
        self.assertEqual(artifacts["batch_dir"], self.batch_dir)
        self.assertIn("payload", artifacts)
        self.assertIn("options", artifacts)
        self.assertEqual(artifacts["manifest"]["best_subset_id"], 1)
        self.assertEqual(len(artifacts["comparison_df"]), 2)

    def test_resolve_target_parcels(self):
        artifacts = load_job_artifacts(self.batch_dir)

        # "best" resolves to roi_A
        parcels_best, info_best = resolve_target_parcels(artifacts, "best")
        self.assertEqual(parcels_best, ["roi_A"])
        self.assertEqual(info_best["subset_id"], 1)

        # "all" resolves to roi_A and roi_B
        parcels_all, info_all = resolve_target_parcels(artifacts, "all")
        self.assertEqual(set(parcels_all), {"roi_A", "roi_B"})

        # Specific parcel
        parcels_custom, _ = resolve_target_parcels(artifacts, "roi_B")
        self.assertEqual(parcels_custom, ["roi_B"])

    def test_create_shuffled_null_payload_zero_shot(self):
        artifacts = load_job_artifacts(self.batch_dir)
        payload = artifacts["payload"]
        rng = np.random.default_rng(12345)

        null_payload = create_shuffled_null_payload(
            payload,
            rng=rng,
            mode="concept_mapping",
            zero_shot=True,
            zero_shot_random_state=42,
            zero_shot_unseen_fraction=0.4,
        )

        self.assertEqual(len(null_payload["y"]), len(payload["y"]))
        # Test trials should remain untouched
        from core.concept_decoding_signal import zero_shot_concept_train_test_split
        train_idx, test_idx, _ = zero_shot_concept_train_test_split(payload["y"], unseen_fraction=0.4, random_state=42)
        np.testing.assert_array_equal(null_payload["y"][test_idx], payload["y"][test_idx])

    def test_compute_diagnostics_calculation(self):
        real_metrics = {
            "mpr": 0.88,
            "top1_accuracy": 0.85,
            "top5_accuracy": 1.0,
            "balanced_accuracy": 0.85,
        }
        null_records = [
            {"permutation": i, "mpr": 0.50 + np.random.normal(0, 0.01), "top1_accuracy": 0.50, "top5_accuracy": 1.0, "balanced_accuracy": 0.50}
            for i in range(20)
        ]

        summary = compute_diagnostics(real_metrics, null_records, n_classes=2)
        self.assertEqual(summary["verdict"], "PASS")
        self.assertTrue(summary["checks"]["null_centered_at_chance"])
        self.assertTrue(summary["checks"]["statistically_significant"])
        self.assertLess(summary["metrics"]["mpr"]["empirical_p_value"], 0.1)

    def test_plot_diagnostics(self):
        real_metrics = {"mpr": 0.85, "top1_accuracy": 0.80}
        null_records = [
            {"permutation": i, "mpr": 0.50 + 0.01 * (i % 3 - 1), "top1_accuracy": 0.50 + 0.01 * (i % 2)}
            for i in range(10)
        ]
        summary = compute_diagnostics(real_metrics, null_records, n_classes=2)
        df_null = pd.DataFrame(null_records)

        plot_path = self.root / "diagnostic_test.png"
        plot_diagnostics(summary, df_null, plot_path)
        self.assertTrue(plot_path.is_file())
        self.assertGreater(plot_path.stat().st_size, 1000)

    def test_main_cli_execution(self):
        out_dir = self.root / "test_output"
        ret = main(
            [
                "--job-dir",
                str(self.batch_dir),
                "--subset",
                "best",
                "--n-shuffles",
                "3",
                "--output-dir",
                str(out_dir),
            ]
        )
        self.assertEqual(ret, 0)
        self.assertTrue((out_dir / "shuffled_null_summary.json").is_file())
        self.assertTrue((out_dir / "shuffled_null_iterations.csv").is_file())
        self.assertTrue((out_dir / "shuffled_null_diagnostic.png").is_file())

        summary = json.loads((out_dir / "shuffled_null_summary.json").read_text())
        self.assertIn("verdict", summary)
        self.assertEqual(summary["n_permutations_completed"], 3)


if __name__ == "__main__":
    unittest.main()
