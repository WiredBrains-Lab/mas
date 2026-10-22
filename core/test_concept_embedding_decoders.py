import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd

from core.concept_decoding_transformer import (
    _compute_log_bandpower_tensor,
    _default_band_defs,
    _make_pseudo_epochs,
    compute_transformer_decoding_from_payload,
    is_semantic_embedding_decoder,
    is_source_space_semantic_decoder,
    is_zero_shot_decoder,
    semantic_embedding_decoder_kind,
)
from core.concept_decoding_signal import zero_shot_concept_train_test_split
from core.concept_decoding_source import compute_source_space_concept_decoding


def _synthetic_embedding_payload(seed=4):
    rng = np.random.default_rng(seed)
    class_names = ["airplane", "dog", "guitar"]
    labels = np.asarray([label for label in class_names for _ in range(12)], dtype=str)
    basis = np.eye(len(class_names), dtype=np.float32)
    embeddings = np.vstack(
        [np.repeat(basis[idx : idx + 1], 12, axis=0) for idx in range(len(class_names))]
    )
    centers = np.asarray(
        [
            [[[3.0, 0.0]], [[0.0, 0.0]]],
            [[[0.0, 3.0]], [[0.0, 0.0]]],
            [[[0.0, 0.0]], [[3.0, 0.0]]],
        ],
        dtype=np.float32,
    )
    X_tensor = np.stack(
        [
            centers[class_idx] + rng.normal(0.0, 0.15, centers[class_idx].shape)
            for class_idx in range(len(class_names))
            for _ in range(12)
        ]
    ).astype(np.float32)
    trial_df = pd.DataFrame(
        {
            "trial_index": np.arange(labels.size, dtype=int),
            "concept_label": labels,
            "canonical_stimulus": labels,
            "embedding_key": labels,
        }
    )
    feature_meta_df = pd.DataFrame(
        {
            "feature_index": np.arange(4, dtype=int),
            "feature": [f"feature_{idx}" for idx in range(4)],
        }
    )
    return {
        "X_tensor": X_tensor,
        "embeddings": embeddings,
        "y": labels,
        "trial_df": trial_df,
        "trial_qc_df": pd.DataFrame(),
        "feature_meta_df": feature_meta_df,
        "channel_names": ["MEG001", "MEG002"],
        "band_defs": [("synthetic", 1.0, 4.0)],
        "n_time_points": 2,
        "feature_binning_enabled": True,
        "window_start": 0.0,
        "window_end": 0.4,
        "n_runs": 1,
        "embedding_dict_path": "synthetic.npz",
    }


def _synthetic_source_feature_payload(seed=4):
    signal_payload = _synthetic_embedding_payload(seed)
    X = signal_payload["X_tensor"].reshape(
        signal_payload["X_tensor"].shape[0], -1
    )
    feature_meta_df = pd.DataFrame(
        {
            "feature_index": np.arange(X.shape[1], dtype=int),
            "feature": [f"source_feature_{idx}" for idx in range(X.shape[1])],
            "roi": ["left_roi", "left_roi", "right_roi", "right_roi"],
            "relative_time_s": [0.1, 0.3, 0.1, 0.3],
            "stat": ["mean"] * X.shape[1],
        }
    )
    return {
        "X": X,
        "y": signal_payload["y"],
        "trial_df": signal_payload["trial_df"].drop(
            columns=["embedding_key"], errors="ignore"
        ),
        "trial_qc_df": pd.DataFrame(),
        "feature_names": feature_meta_df["feature"].tolist(),
        "feature_meta_df": feature_meta_df,
        "channel_names": [],
        "roi_names": ["left_roi", "right_roi"],
        "bin_edges": [(0.0, 0.2), (0.2, 0.4)],
        "stat_names": ["mean"],
        "n_time_points": 2,
        "feature_binning_enabled": True,
        "window_start": 0.0,
        "window_end": 0.4,
        "bandpass_low_hz": 1.0,
        "bandpass_high_hz": 40.0,
        "group_mode": "canonical",
        "channel_mode": "source",
        "feature_space": "source",
        "baseline_correct": True,
        "source_parcel_baseline_zscore": True,
        "parcel_baseline_start": -0.1,
        "parcel_baseline_end": 0.0,
        "parcel_baseline_samples": 20,
        "method": "MNE",
        "lambda2": 1.0 / 9.0,
        "source_space_kind": "surface",
        "source_parcellation": "aparc",
        "source_common_space": "native",
        "source_subject_count": 1,
        "native_subjects": ["synthetic_subject"],
        "n_runs": 1,
        "dropped_trials": 0,
        "skipped_paths": [],
    }


class SemanticEmbeddingDecoderTests(unittest.TestCase):
    def test_balanced_pseudo_epochs_equalize_classes_and_rotate_extra_trials(self):
        labels = np.asarray(["animal"] * 8 + ["food"] * 4 + ["vehicle"] * 4)
        X = np.arange(16, dtype=np.float32).reshape(16, 1)
        embeddings = np.arange(32, dtype=np.float32).reshape(16, 2)
        trials = pd.DataFrame({"trial_index": np.arange(16)})

        balanced_a = _make_pseudo_epochs(
            X, embeddings, labels, trials, np.arange(16),
            trials_per_average=2, seed=420, subset="train", balance_classes=True,
        )
        balanced_b = _make_pseudo_epochs(
            X, embeddings, labels, trials, np.arange(16),
            trials_per_average=2, seed=421, subset="train", balance_classes=True,
        )
        unbalanced = _make_pseudo_epochs(
            X, embeddings, labels, trials, np.arange(16),
            trials_per_average=2, seed=420, subset="train", balance_classes=False,
        )

        self.assertEqual(pd.Series(balanced_a["y"]).value_counts().to_dict(), {
            "animal": 2, "food": 2, "vehicle": 2,
        })
        self.assertEqual(pd.Series(unbalanced["y"]).value_counts().to_dict(), {
            "animal": 4, "food": 2, "vehicle": 2,
        })
        self.assertEqual(balanced_a["source_trials_per_class_limit"], 4)
        animal_a = set(
            ",".join(
                balanced_a["df"].loc[
                    balanced_a["df"]["pseudo_epoch_label"] == "animal", "source_trial_indices"
                ]
            ).split(",")
        )
        animal_b = set(
            ",".join(
                balanced_b["df"].loc[
                    balanced_b["df"]["pseudo_epoch_label"] == "animal", "source_trial_indices"
                ]
            ).split(",")
        )
        self.assertNotEqual(animal_a, animal_b)

    def test_task_stratified_pseudo_epochs_n5_balances_all_5_tasks(self):
        tasks = ["ANIMAL", "FOOD", "VEHICLE", "MUSIC", "BODY"]
        # Concept A and B each have 48 animal + 24 of each other 4 tasks = 144 trials each
        task_list_per_concept = ["ANIMAL"] * 48 + ["FOOD"] * 24 + ["VEHICLE"] * 24 + ["MUSIC"] * 24 + ["BODY"] * 24
        labels = np.asarray(["dog"] * 144 + ["cat"] * 144)
        decisions = task_list_per_concept * 2
        X = np.arange(288, dtype=np.float32).reshape(288, 1)
        embeddings = np.arange(288 * 2, dtype=np.float32).reshape(288, 2)
        trials = pd.DataFrame({"semantic_decision": decisions})

        res = _make_pseudo_epochs(
            X, embeddings, labels, trials, np.arange(288),
            trials_per_average=5, seed=420, subset="train", task_stratified=True,
        )
        self.assertTrue(res["task_stratified"])
        self.assertEqual(len(res["y"]), 48)  # 24 per concept * 2 concepts = 48
        self.assertEqual(pd.Series(res["y"]).value_counts().to_dict(), {"dog": 24, "cat": 24})

        # Check every pseudo-epoch contains exactly 1 trial from all 5 tasks
        for _, row in res["df"].iterrows():
            self.assertEqual(row["n_source_trials"], 5)
            row_decisions = sorted(row["source_semantic_decisions"].split(","))
            self.assertEqual(row_decisions, sorted(tasks))

    def test_task_stratified_pseudo_epochs_n6_proportional_balances_tasks(self):
        tasks = ["ANIMAL", "FOOD", "VEHICLE", "MUSIC", "BODY"]
        task_list_per_concept = ["ANIMAL"] * 48 + ["FOOD"] * 24 + ["VEHICLE"] * 24 + ["MUSIC"] * 24 + ["BODY"] * 24
        labels = np.asarray(["dog"] * 144 + ["cat"] * 144)
        decisions = task_list_per_concept * 2
        X = np.arange(288, dtype=np.float32).reshape(288, 1)
        embeddings = np.arange(288 * 2, dtype=np.float32).reshape(288, 2)
        trials = pd.DataFrame({"semantic_decision": decisions})

        res = _make_pseudo_epochs(
            X, embeddings, labels, trials, np.arange(288),
            trials_per_average=6, seed=420, subset="train", task_stratified=True,
        )
        self.assertTrue(res["task_stratified"])
        self.assertEqual(len(res["y"]), 48)  # 24 per concept * 2 concepts = 48
        self.assertEqual(pd.Series(res["y"]).value_counts().to_dict(), {"dog": 24, "cat": 24})

        # Check every pseudo-epoch contains 2 ANIMAL and 1 of each of the other 4 tasks
        for _, row in res["df"].iterrows():
            self.assertEqual(row["n_source_trials"], 6)
            row_decisions = row["source_semantic_decisions"].split(",")
            counts = pd.Series(row_decisions).value_counts().to_dict()
            self.assertEqual(counts["ANIMAL"], 2)
            self.assertEqual(counts["FOOD"], 1)
            self.assertEqual(counts["VEHICLE"], 1)
            self.assertEqual(counts["MUSIC"], 1)
            self.assertEqual(counts["BODY"], 1)

    def test_task_stratified_disabled_falls_back_to_random(self):
        task_list_per_concept = ["ANIMAL"] * 48 + ["FOOD"] * 24 + ["VEHICLE"] * 24 + ["MUSIC"] * 24 + ["BODY"] * 24
        labels = np.asarray(["dog"] * 144)
        trials = pd.DataFrame({"semantic_decision": task_list_per_concept})
        X = np.arange(144, dtype=np.float32).reshape(144, 1)
        embeddings = np.arange(144 * 2, dtype=np.float32).reshape(144, 2)

        res = _make_pseudo_epochs(
            X, embeddings, labels, trials, np.arange(144),
            trials_per_average=5, seed=420, subset="train", task_stratified=False,
        )
        self.assertFalse(res["task_stratified"])

    def test_per_concept_train_test_split_stratified(self):
        from core.concept_decoding_signal import per_concept_train_test_split
        task_list_per_concept = ["ANIMAL"] * 48 + ["FOOD"] * 24 + ["VEHICLE"] * 24 + ["MUSIC"] * 24 + ["BODY"] * 24
        labels = np.asarray(["dog"] * 144 + ["cat"] * 144)
        strat_groups = np.asarray(task_list_per_concept * 2)

        train_idx, test_idx, split_df = per_concept_train_test_split(
            labels, test_fraction=0.2, random_state=42, stratify_groups=strat_groups,
        )
        # Verify both concepts have trials in train and test
        self.assertEqual(len(set(labels[train_idx])), 2)
        self.assertEqual(len(set(labels[test_idx])), 2)
        # Verify all 5 tasks are represented in test split for each concept
        for lbl in ["dog", "cat"]:
            concept_mask = labels[test_idx] == lbl
            test_tasks = set(strat_groups[test_idx[concept_mask]])
            self.assertEqual(test_tasks, {"ANIMAL", "FOOD", "VEHICLE", "MUSIC", "BODY"})

    def test_semantic_bands_must_survive_shared_passband_and_nyquist(self):
        self.assertEqual(
            [row[0] for row in _default_band_defs(200.0, 1.0, 40.0)],
            ["delta", "theta", "alpha", "beta"],
        )
        self.assertEqual(
            [row[0] for row in _default_band_defs(400.0, 1.0, 150.0)],
            ["delta", "theta", "alpha", "beta", "low_gamma", "high_gamma"],
        )

    def test_hilbert_power_bins_are_db_relative_to_prestimulus(self):
        sfreq = 200.0
        times = np.arange(-0.7, 0.9 + 1.0 / sfreq, 1.0 / sfreq)
        amplitude = np.where(times < 0.0, 1.0, 2.0)
        signal = amplitude * np.sin(2.0 * np.pi * 10.0 * times)
        data = signal[None, None, :].astype(np.float32)

        power = _compute_log_bandpower_tensor(
            data,
            times,
            [(0.2, 0.3), (0.3, 0.4)],
            sfreq,
            [("alpha", 8.0, 13.0)],
            baseline=(-0.2, 0.0),
        )

        self.assertEqual(power.shape, (1, 1, 1, 2))
        np.testing.assert_allclose(
            np.median(power), 10.0 * np.log10(4.0), atol=1.25
        )

    def test_new_decoder_aliases_are_semantic_embedding_models(self):
        self.assertTrue(is_semantic_embedding_decoder("embedding_ridge"))
        self.assertTrue(is_semantic_embedding_decoder("contrastive_linear"))
        self.assertTrue(is_source_space_semantic_decoder("embedding_ridge"))
        self.assertTrue(is_source_space_semantic_decoder("contrastive_linear"))
        self.assertFalse(is_source_space_semantic_decoder("transformer"))
        self.assertFalse(is_source_space_semantic_decoder("embedding_mlp"))
        self.assertTrue(is_semantic_embedding_decoder("word2vec_mlp"))
        self.assertTrue(is_semantic_embedding_decoder("zero_shot"))
        self.assertTrue(is_semantic_embedding_decoder("zero_shot_ridge"))
        self.assertTrue(is_semantic_embedding_decoder("zero_shot_contrastive"))
        self.assertTrue(is_semantic_embedding_decoder("zero_shot_mlp"))
        self.assertTrue(is_source_space_semantic_decoder("zero_shot"))
        self.assertTrue(is_zero_shot_decoder("zero_shot"))
        self.assertTrue(is_zero_shot_decoder("zero_shot_ridge"))
        self.assertEqual(semantic_embedding_decoder_kind("zero_shot"), "zero_shot")
        self.assertEqual(
            semantic_embedding_decoder_kind("zero_shot_contrastive"),
            "zero_shot_contrastive",
        )
        self.assertEqual(
            semantic_embedding_decoder_kind("zero_shot_mlp"),
            "zero_shot_mlp",
        )
        self.assertEqual(semantic_embedding_decoder_kind("linear_embedding"), "embedding_ridge")
        self.assertEqual(
            semantic_embedding_decoder_kind("class_aware_contrastive"),
            "embedding_contrastive",
        )
        self.assertEqual(semantic_embedding_decoder_kind("neural_embedding"), "embedding_mlp")

    def test_ridge_embedding_decoder_uses_shared_cosine_rank_scoring(self):
        result, error = compute_transformer_decoding_from_payload(
            None,
            {
                "decoder_type": "embedding_ridge",
                "test_fraction": 0.25,
                "n_permutations": 0,
                "transformer_trials_per_average": 1,
                "embedding_ridge_alpha": 1.0,
            },
            _synthetic_embedding_payload(),
        )

        self.assertIsNotNone(result, error)
        self.assertEqual(result["decoder_type"], "embedding_ridge")
        self.assertEqual(result["primary_metric"], "MPR")
        self.assertAlmostEqual(result["top1_chance_accuracy"], 1.0 / 3.0)
        self.assertGreater(result["top1_accuracy"], 0.9)
        self.assertGreater(result["mean_reciprocal_rank"], 0.9)
        self.assertIn("predicted_target_cosine", result["predictions_df"].columns)
        self.assertIn("reciprocal_rank", result["predictions_df"].columns)

    def test_source_features_route_to_semantic_ridge_decoder(self):
        source_payload = _synthetic_source_feature_payload()
        embedding_map = {
            "airplane": [1.0, 0.0, 0.0],
            "dog": [0.0, 1.0, 0.0],
            "guitar": [0.0, 0.0, 1.0],
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            embedding_path = Path(tmp_dir) / "embeddings.json"
            embedding_path.write_text(json.dumps(embedding_map), encoding="utf-8")
            options = {
                "decoder_type": "embedding_ridge",
                "feature_space": "source",
                "embedding_dict_path": str(embedding_path),
                "test_fraction": 0.25,
                "n_permutations": 0,
                "transformer_trials_per_average": 1,
                "embedding_ridge_alpha": 1.0,
            }
            with mock.patch(
                "core.concept_decoding_source.build_source_space_concept_features",
                return_value=(source_payload, None),
            ):
                result, error = compute_source_space_concept_decoding(
                    SimpleNamespace(b_fif_loaded=True), options
                )

        self.assertIsNotNone(result, error)
        self.assertEqual(result["decoder_type"], "embedding_ridge")
        self.assertEqual(result["feature_space"], "source")
        self.assertEqual(result["n_rois"], 2)
        self.assertEqual(result["n_bands"], 0)
        self.assertGreater(result["top1_accuracy"], 0.9)
        self.assertTrue(result["decoder"].startswith("source-space"))
        self.assertIn("embedding_key", result["predictions_df"].columns)
        architecture = result["model_architecture_df"]
        self.assertIn(
            "Source-parcel feature matrix", architecture["component"].tolist()
        )

    def test_contrastive_linear_decoder_accepts_source_feature_matrix(self):
        try:
            import torch  # noqa: F401
        except Exception:
            self.skipTest("PyTorch is unavailable")
        payload = _synthetic_source_feature_payload()
        embedding_by_label = {
            "airplane": np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
            "dog": np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
            "guitar": np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
        }
        payload["X_tensor"] = payload["X"]
        payload["embeddings"] = np.vstack(
            [embedding_by_label[label] for label in payload["y"]]
        )
        result, error = compute_transformer_decoding_from_payload(
            None,
            {
                "decoder_type": "embedding_contrastive",
                "test_fraction": 0.25,
                "n_permutations": 0,
                "transformer_trials_per_average": 1,
                "transformer_epochs": 60,
                "mlp_validation_fraction": 0.2,
                "embedding_contrastive_learning_rate": 0.01,
            },
            payload,
        )

        self.assertIsNotNone(result, error)
        self.assertEqual(result["feature_space"], "source")
        self.assertEqual(result["decoder_type"], "embedding_contrastive")
        self.assertGreater(result["top1_accuracy"], 0.9)

    def test_mlp_embedding_decoder_emits_live_validation_retrieval_metrics(self):
        try:
            import torch  # noqa: F401
        except Exception:
            self.skipTest("PyTorch is unavailable")
        metrics = []
        epochs = 5
        result, error = compute_transformer_decoding_from_payload(
            None,
            {
                "decoder_type": "embedding_mlp",
                "test_fraction": 0.25,
                "n_permutations": 0,
                "transformer_trials_per_average": 1,
                "transformer_epochs": epochs,
                "mlp_validation_fraction": 0.2,
                "mlp_hidden_1": 16,
                "mlp_hidden_2": 0,
                "mlp_dropout": 0.0,
                "training_metric_callback": metrics.append,
            },
            _synthetic_embedding_payload(),
        )

        self.assertIsNotNone(result, error)
        self.assertEqual(result["decoder_type"], "embedding_mlp")
        self.assertEqual(len(metrics), epochs)
        self.assertEqual(len(result["training_history_df"]), epochs)
        self.assertTrue(all(row["model"] == "embedding_mlp" for row in metrics))
        self.assertTrue(all(row["n_validation"] > 0 for row in metrics))
        self.assertTrue(all(np.isfinite(row["validation_accuracy"]) for row in metrics))
        self.assertGreater(result["n_source_validation_trials"], 0)
        self.assertEqual(
            result["n_source_train_trials"]
            + result["n_source_validation_trials"]
            + result["n_source_test_trials"],
            result["n_source_trials"],
        )

    def test_class_aware_contrastive_linear_decoder_ranks_unique_concepts(self):
        try:
            import torch  # noqa: F401
        except Exception:
            self.skipTest("PyTorch is unavailable")
        metrics = []
        epochs = 60
        result, error = compute_transformer_decoding_from_payload(
            None,
            {
                "decoder_type": "embedding_contrastive",
                "test_fraction": 0.25,
                "n_permutations": 0,
                "transformer_trials_per_average": 1,
                "transformer_epochs": epochs,
                "mlp_validation_fraction": 0.2,
                "embedding_contrastive_temperature": 0.07,
                "embedding_contrastive_learning_rate": 0.01,
                "embedding_contrastive_weight_decay": 1e-3,
                "training_metric_callback": metrics.append,
            },
            _synthetic_embedding_payload(),
        )

        self.assertIsNotNone(result, error)
        self.assertEqual(result["decoder_type"], "embedding_contrastive")
        self.assertEqual(len(metrics), epochs)
        self.assertTrue(
            all(row["model"] == "embedding_contrastive" for row in metrics)
        )
        self.assertTrue(all(row["n_candidates"] == 3 for row in metrics))
        self.assertTrue(all(row["n_validation"] > 0 for row in metrics))
        self.assertTrue(
            all(
                np.isclose(row["train_loss"], row["train_contrastive_loss"])
                for row in metrics
            )
        )
        self.assertGreater(result["top1_accuracy"], 0.9)
        self.assertFalse(result["prediction_collapse_detected"])
        self.assertAlmostEqual(result["embedding_contrastive_temperature"], 0.07)
        architecture = result["model_architecture_df"]
        self.assertIn(
            "Linear class-aware contrastive embedding",
            architecture["component"].tolist(),
        )

    def test_compact_contrastive_decoder_reduces_dimensions_and_stops_early(self):
        try:
            import torch  # noqa: F401
        except Exception:
            self.skipTest("PyTorch is unavailable")
        result, error = compute_transformer_decoding_from_payload(
            None,
            {
                "decoder_type": "embedding_contrastive",
                "test_fraction": 0.25,
                "n_permutations": 0,
                "transformer_trials_per_average": 1,
                "transformer_epochs": 20,
                "mlp_validation_fraction": 0.2,
                "embedding_contrastive_learning_rate": 0.01,
                "embedding_input_pca_components": 2,
                "embedding_target_svd_components": 2,
                "embedding_early_stopping_patience": 2,
                "embedding_early_stopping_min_delta": 100.0,
                "embedding_dynamic_pseudo_epochs": True,
            },
            _synthetic_embedding_payload(),
        )

        self.assertIsNotNone(result, error)
        self.assertTrue(result["embedding_input_pca_applied"])
        self.assertEqual(result["embedding_input_pca_components"], 2)
        self.assertEqual(result["embedding_original_dim"], 3)
        self.assertEqual(result["embedding_target_rank"], 2)
        self.assertEqual(result["embedding_target_svd_components"], 2)
        self.assertEqual(result["embedding_dim"], 2)
        self.assertEqual(result["embedding_model_parameter_count"], 6)
        self.assertTrue(result["embedding_dynamic_pseudo_epochs"])
        self.assertTrue(result["embedding_early_stopped"])
        self.assertLess(result["transformer_epochs"], 20)
        self.assertEqual(result["transformer_epochs_requested"], 20)
        selected_epoch = result["embedding_selected_epoch"]
        selected_loss = result["training_history_df"].iloc[selected_epoch - 1][
            "train_loss"
        ]
        self.assertAlmostEqual(result["transformer_final_train_loss"], selected_loss)
        self.assertTrue(np.all(np.isfinite(result["feature_df"]["importance"])))

    def test_mlp_contrastive_training_avoids_centroid_collapse_on_learnable_data(self):
        try:
            import torch  # noqa: F401
        except Exception:
            self.skipTest("PyTorch is unavailable")
        result, error = compute_transformer_decoding_from_payload(
            None,
            {
                "decoder_type": "embedding_mlp",
                "test_fraction": 0.25,
                "n_permutations": 0,
                "transformer_trials_per_average": 1,
                "transformer_epochs": 50,
                "mlp_validation_fraction": 0.2,
                "mlp_hidden_1": 16,
                "mlp_hidden_2": 0,
                "mlp_dropout": 0.0,
            },
            _synthetic_embedding_payload(),
        )

        self.assertIsNotNone(result, error)
        self.assertGreater(result["top1_accuracy"], 0.9)
        self.assertFalse(result["prediction_collapse_detected"])
        self.assertTrue(result["embedding_target_centered"])

    def test_constant_features_are_reported_as_prediction_collapse(self):
        payload = _synthetic_embedding_payload()
        payload["X_tensor"] = np.zeros_like(payload["X_tensor"])
        result, error = compute_transformer_decoding_from_payload(
            None,
            {
                "decoder_type": "embedding_ridge",
                "test_fraction": 0.25,
                "n_permutations": 0,
                "transformer_trials_per_average": 1,
            },
            payload,
        )

        self.assertIsNotNone(result, error)
        self.assertTrue(result["prediction_collapse_detected"])
        self.assertEqual(result["predicted_unique_concepts"], 1)


    def test_zero_shot_concept_train_test_split_partitions_unseen_concepts(self):
        labels = np.asarray(["cat", "cat", "dog", "dog", "bird", "bird", "fish", "fish"])
        train_idx, test_idx, split_df = zero_shot_concept_train_test_split(
            labels,
            unseen_fraction=0.5,
            random_state=42,
        )
        train_labels = set(labels[train_idx])
        test_labels = set(labels[test_idx])
        self.assertTrue(train_labels.isdisjoint(test_labels))
        self.assertEqual(len(train_labels), 2)
        self.assertEqual(len(test_labels), 2)
        self.assertEqual(train_idx.size + test_idx.size, labels.size)
        self.assertEqual(split_df.shape[0], 4)
        self.assertEqual(split_df["is_unseen"].sum(), 2)

        # Explicit unseen concepts
        train_idx2, test_idx2, split_df2 = zero_shot_concept_train_test_split(
            labels,
            unseen_concepts=["fish", "bird"],
        )
        self.assertEqual(set(labels[test_idx2]), {"fish", "bird"})
        self.assertEqual(set(labels[train_idx2]), {"cat", "dog"})

        # Rejects fewer than required concepts
        with self.assertRaises(ValueError):
            zero_shot_concept_train_test_split(
                np.asarray(["cat", "cat", "dog", "dog"]),
                min_train_concepts=2,
                min_test_concepts=1,
            )

    def test_zero_shot_split_balances_unseen_concepts_across_categories(self):
        concepts = [f"concept_{category}_{index}" for category in range(10) for index in range(5)]
        labels = np.repeat(concepts, 2)
        categories = np.repeat(
            [f"category_{category}" for category in range(10) for _ in range(5)],
            2,
        )
        train_idx, test_idx, split_df = zero_shot_concept_train_test_split(
            labels,
            unseen_fraction=0.2,
            random_state=42,
            concept_categories=categories,
            balance_categories=True,
        )
        self.assertTrue(set(labels[train_idx]).isdisjoint(set(labels[test_idx])))
        unseen = split_df.loc[split_df["is_unseen"]]
        self.assertEqual(len(unseen), 10)
        self.assertEqual(unseen.groupby("stimulus_category").size().to_dict(), {
            f"category_{category}": 1 for category in range(10)
        })
        with self.assertRaisesRegex(ValueError, "cannot be divided equally"):
            zero_shot_concept_train_test_split(
                labels,
                unseen_fraction=0.3,
                concept_categories=categories,
                balance_categories=True,
            )

    def test_zero_shot_decoding_from_payload_evaluates_held_out_unseen_concepts(self):
        rng = np.random.default_rng(4)
        class_names = ["airplane", "dog", "guitar", "piano"]
        labels = np.asarray([label for label in class_names for _ in range(12)], dtype=str)
        basis = np.eye(4, dtype=np.float32)
        embeddings = np.vstack([np.repeat(basis[idx : idx + 1], 12, axis=0) for idx in range(4)])
        centers = np.zeros((4, 2, 1, 2), dtype=np.float32)
        centers[0, 0, 0, 0] = 3.0
        centers[1, 0, 0, 1] = 3.0
        centers[2, 1, 0, 0] = 3.0
        centers[3, 1, 0, 1] = 3.0
        X_tensor = np.stack(
            [
                centers[class_idx] + rng.normal(0.0, 0.15, centers[class_idx].shape)
                for class_idx in range(4)
                for _ in range(12)
            ]
        ).astype(np.float32)
        trial_df = pd.DataFrame(
            {
                "trial_index": np.arange(labels.size, dtype=int),
                "concept_label": labels,
                "canonical_stimulus": labels,
                "embedding_key": labels,
            }
        )
        feature_meta_df = pd.DataFrame(
            {
                "feature_index": np.arange(4, dtype=int),
                "feature": [f"feature_{idx}" for idx in range(4)],
            }
        )
        payload = {
            "X_tensor": X_tensor,
            "embeddings": embeddings,
            "y": labels,
            "trial_df": trial_df,
            "trial_qc_df": pd.DataFrame(),
            "feature_meta_df": feature_meta_df,
            "channel_names": ["MEG001", "MEG002"],
            "band_defs": [("synthetic", 1.0, 4.0)],
            "n_time_points": 2,
            "feature_binning_enabled": True,
            "window_start": 0.0,
            "window_end": 0.4,
            "n_runs": 1,
            "embedding_dict_path": "synthetic.npz",
        }

        result, error = compute_transformer_decoding_from_payload(
            None,
            {
                "decoder_type": "embedding_ridge",
                "split_mode": "zero_shot",
                "zero_shot_unseen_concepts": ["guitar", "piano"],
                "n_permutations": 0,
                "transformer_trials_per_average": 1,
                "embedding_ridge_alpha": 1.0,
            },
            payload,
        )

        self.assertIsNotNone(result, error)
        self.assertTrue(result["is_zero_shot"])
        self.assertEqual(result["primary_metric"], "2AFC")
        self.assertEqual(sorted(result["unseen_concepts"]), ["guitar", "piano"])
        self.assertEqual(sorted(result["train_concepts"]), ["airplane", "dog"])
        self.assertAlmostEqual(result["pairwise_2afc_chance_accuracy"], 0.5)
        self.assertGreater(result["pairwise_2afc_accuracy"], 0.8)
        self.assertGreater(result["top1_accuracy"], 0.8)
        self.assertIn("pairwise_2afc_accuracy", result["predictions_df"].columns)
        self.assertIn("pairwise_2afc_accuracy", result["per_class_accuracy_df"].columns)
        self.assertIn("unseen_test", result["split_df"]["split"].values)

    def test_zero_shot_source_features_route_to_zero_shot_decoder(self):
        source_payload = _synthetic_source_feature_payload()
        embedding_map = {
            "airplane": [1.0, 0.0, 0.0],
            "dog": [0.0, 1.0, 0.0],
            "guitar": [0.0, 0.0, 1.0],
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            embedding_path = Path(tmp_dir) / "embeddings.json"
            embedding_path.write_text(json.dumps(embedding_map), encoding="utf-8")
            options = {
                "decoder_type": "embedding_ridge",
                "split_mode": "zero_shot",
                "feature_space": "source",
                "embedding_dict_path": str(embedding_path),
                "zero_shot_unseen_concepts": ["guitar"],
                "n_permutations": 0,
                "transformer_trials_per_average": 1,
            }
            with mock.patch(
                "core.concept_decoding_source.build_source_space_concept_features",
                return_value=(source_payload, None),
            ):
                result, error = compute_source_space_concept_decoding(
                    SimpleNamespace(b_fif_loaded=True), options
                )

        self.assertIsNotNone(result, error)
        self.assertTrue(result["is_zero_shot"])
        self.assertEqual(result["feature_space"], "source")
        self.assertEqual(result["unseen_concepts"], ["guitar"])
        self.assertTrue(result["decoder"].startswith("source-space zero-shot"))


    def test_discrete_decoder_rejects_zero_shot_split(self):
        from core.concept_decoding_signal import compute_signal_space_concept_decoding
        context = SimpleNamespace(b_fif_loaded=True)
        result, error = compute_signal_space_concept_decoding(
            context,
            {"decoder_type": "logistic", "split_mode": "zero_shot"},
        )
        self.assertIsNone(result)
        self.assertIn("requires a semantic embedding decoder", error)


if __name__ == "__main__":
    unittest.main()
