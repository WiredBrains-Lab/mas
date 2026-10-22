import unittest
from unittest import mock
from types import SimpleNamespace

import numpy as np
import pandas as pd

from core.concept_decoding_signal import (
    concept_bandpass_from_options,
    concept_decoder_spec,
    concept_epoch_reject_from_options,
    concept_resample_from_options,
    feature_edges_for_concept_decoding,
    fit_concept_decoder,
    parcel_trial_frequency_band_settings,
)
from core.concept_decoding_source import (
    baseline_db_hilbert_parcel_power,
    baseline_zscore_parcel_time_courses,
    build_source_parcel_trial_time_courses,
    clear_parcel_timecourse_cache,
    parcel_trial_candidate_pool_size,
    parcel_trial_timecourse_cache_key,
    source_bin_masks,
    source_trial_feature_values,
)
from core.source_parcellation import (
    anatomical_parcel_row_order,
    parcel_trial_contrast_map,
    parcel_trial_summary_maps,
    rank_parcel_contrast_features,
    temporal_bin_parcel_time_courses,
    is_visual_parcel,
    is_semantic_parcel,
)


class SourceFeatureTests(unittest.TestCase):
    def test_parcel_trial_frequency_band_resolves_shared_and_named_filters(self):
        self.assertEqual(
            parcel_trial_frequency_band_settings("broadband", 2.0, 35.0),
            (2.0, 35.0, "Broadband (2–35 Hz)"),
        )
        self.assertEqual(
            parcel_trial_frequency_band_settings("alpha", 2.0, 35.0),
            (8.0, 13.0, "Alpha (8–13 Hz)"),
        )
        self.assertEqual(
            parcel_trial_frequency_band_settings(
                "high_gamma",
                1.0,
                150.0,
                resample_enabled=True,
                resample_sfreq=400.0,
            ),
            (70.0, 150.0, "High gamma (70–150 Hz)"),
        )
        with self.assertRaisesRegex(ValueError, "unavailable after the shared 1–40 Hz"):
            parcel_trial_frequency_band_settings("low_gamma", 1.0, 40.0)
        with self.assertRaisesRegex(ValueError, "100 Hz Nyquist"):
            parcel_trial_frequency_band_settings(
                "high_gamma",
                1.0,
                150.0,
                resample_enabled=True,
                resample_sfreq=200.0,
            )
        with self.assertRaisesRegex(ValueError, "Unknown Parcel trials frequency band"):
            parcel_trial_frequency_band_settings("ultraviolet", 1.0, 40.0)

    def test_parcel_timecourse_cache_key_tracks_preprocessing_and_trials(self):
        context = SimpleNamespace(
            _fif_filename="run1.fif",
            _fif_trial_response_table=pd.DataFrame(
                {
                    "source_fif_path": ["run1.fif"],
                    "onset_sample": [100],
                    "canonical_stimulus": ["airplane"],
                }
            ),
            _fif_stimulus_summary_table=pd.DataFrame(),
        )
        base = {
            "source_fif_paths": ["run1.fif"],
            "selected_concept_label": "airplane",
            "bandpass_low_hz": 1.0,
            "bandpass_high_hz": 40.0,
        }
        key = parcel_trial_timecourse_cache_key(context, base)
        changed_filter = parcel_trial_timecourse_cache_key(
            context, {**base, "bandpass_high_hz": 30.0}
        )
        changed_band_identity = parcel_trial_timecourse_cache_key(
            context, {**base, "parcel_frequency_band": "alpha"}
        )
        changed_representation = parcel_trial_timecourse_cache_key(
            context, {**base, "parcel_representation": "band_power_db"}
        )
        context._fif_trial_response_table.loc[0, "onset_sample"] = 101
        changed_trial = parcel_trial_timecourse_cache_key(context, base)

        self.assertNotEqual(key, changed_filter)
        self.assertNotEqual(key, changed_band_identity)
        self.assertNotEqual(key, changed_representation)
        self.assertNotEqual(key, changed_trial)

    @mock.patch("core.concept_decoding_source.build_source_space_concept_features")
    def test_parcel_timecourse_cache_reuses_completed_source_result(self, build_features):
        feature_meta = pd.DataFrame(
            [
                {"roi": "parcel-a", "relative_time_s": sample_time}
                for sample_time in (0.0, 0.1)
            ]
        )
        build_features.return_value = (
            {
                "X": np.asarray([[1, 2], [3, 4]], dtype=np.float32),
                "roi_names": ["parcel-a"],
                "feature_meta_df": feature_meta,
                "trial_df": pd.DataFrame(
                    {
                        "trial_index": [1, 2],
                        "source_fif_path": ["run1.fif", "run1.fif"],
                    }
                ),
                "source_parcellation": "aparc",
                "n_runs": 1,
                "dropped_trials": 0,
            },
            None,
        )
        context = SimpleNamespace(
            _fif_filename="run1.fif",
            _fif_trial_response_table=pd.DataFrame(
                {
                    "source_fif_path": ["run1.fif", "run1.fif"],
                    "onset_sample": [100, 200],
                    "canonical_stimulus": ["airplane", "airplane"],
                }
            ),
            _fif_stimulus_summary_table=pd.DataFrame(),
        )
        options = {
            "source_fif_paths": ["run1.fif"],
            "selected_concept_label": "airplane",
            "max_presentations": 2,
        }

        first, _first_message = build_source_parcel_trial_time_courses(context, options)
        second, second_message = build_source_parcel_trial_time_courses(context, options)

        self.assertEqual(build_features.call_count, 1)
        self.assertFalse(first["parcel_cache_hit"])
        self.assertTrue(second["parcel_cache_hit"])
        self.assertIn("Reused cached", second_message)
        np.testing.assert_array_equal(first["data"], second["data"])
        clear_parcel_timecourse_cache(context)
        self.assertEqual(len(context._parcel_timecourse_cache), 0)

    def test_parcel_trial_candidate_pool_is_bounded_with_qc_reserve(self):
        self.assertEqual(parcel_trial_candidate_pool_size(0), 0)
        self.assertEqual(parcel_trial_candidate_pool_size(2), 4)
        self.assertEqual(parcel_trial_candidate_pool_size(12), 15)
        self.assertEqual(parcel_trial_candidate_pool_size(36), 45)

    def test_mlp_emits_structured_training_and_validation_metrics(self):
        try:
            import torch  # noqa: F401
        except Exception:
            self.skipTest("PyTorch is unavailable")
        rng = np.random.default_rng(42)
        X = rng.normal(size=(20, 6))
        y = np.asarray(["airplane"] * 10 + ["guitar"] * 10)
        X[y == "guitar", 0] += 1.5
        metrics = []
        options = {
            "mlp_epochs": 4,
            "mlp_validation_fraction": 0.2,
            "mlp_hidden_1": 16,
            "mlp_hidden_2": 0,
            "mlp_batch_size": 8,
            "training_metric_callback": metrics.append,
            "emit_mlp_epoch_progress": False,
        }
        clf, _label, decoder_key = concept_decoder_spec(
            "mlp", options=options, n_features=X.shape[1], n_classes=2
        )

        fit_concept_decoder(clf, X, y, options=options)

        self.assertEqual(decoder_key, "mlp")
        self.assertEqual([row["epoch"] for row in metrics], [1, 2, 3, 4])
        self.assertTrue(all(row["n_validation"] == 4 for row in metrics))
        self.assertTrue(all(np.isfinite(row["train_accuracy"]) for row in metrics))
        self.assertTrue(all(np.isfinite(row["validation_accuracy"]) for row in metrics))
        history = clf.named_steps["mlpclassifier"].training_history_
        self.assertEqual(len(history), 4)

    def test_shared_preprocessing_defaults_are_filter_resample_and_reject(self):
        self.assertEqual(concept_bandpass_from_options({}), (0.1, 150.0))
        self.assertEqual(concept_resample_from_options({}), (True, 400.0))
        reject = concept_epoch_reject_from_options({})
        self.assertAlmostEqual(reject["mag"], 4000.0e-15)
        self.assertAlmostEqual(reject["grad"], 4000.0e-13)

    def test_mean_and_raw_features_preserve_polarity(self):
        tc = np.asarray([[-3.0, -1.0, 1.0, 3.0], [2.0, 4.0, -2.0, -4.0]])
        masks = [np.asarray([True, True, False, False]), np.asarray([False, False, True, True])]

        mean_values = source_trial_feature_values(tc, ["mean"], masks)
        raw_values = source_trial_feature_values(tc, ["raw"], masks)

        np.testing.assert_allclose(mean_values, [-2.0, 3.0, 2.0, -3.0])
        np.testing.assert_allclose(raw_values, [-3.0, -1.0, 2.0, 4.0, 1.0, 3.0, -2.0, -4.0])

    def test_rms_features_remain_polarity_invariant(self):
        tc = np.asarray([[-3.0, -1.0], [2.0, 4.0]])
        flipped = -tc
        masks = [np.asarray([True, True])]

        values = source_trial_feature_values(tc, ["rms"], masks)
        flipped_values = source_trial_feature_values(flipped, ["rms"], masks)

        np.testing.assert_allclose(values, flipped_values)
        np.testing.assert_allclose(values, [np.sqrt(5.0), np.sqrt(10.0)])

    @mock.patch("core.concept_decoding_source.build_source_space_concept_features")
    def test_parcel_trial_wrapper_restores_trial_parcel_time_shape(self, build_features):
        sample_times = [0.0, 0.1]
        feature_meta = pd.DataFrame(
            [
                {"roi": roi, "relative_time_s": sample_time}
                for roi in ("parcel-a", "parcel-b")
                for sample_time in sample_times
            ]
        )
        build_features.return_value = (
            {
                "X": np.asarray(
                    [
                        [3, 5, 3, 5],
                        [3, 5, 3, 5],
                    ],
                    dtype=np.float32,
                ),
                "roi_names": ["parcel-a", "parcel-b"],
                "feature_meta_df": feature_meta,
                "trial_df": pd.DataFrame({"trial_index": [4, 9]}),
                "group_mode": "category",
                "source_space_kind": "surface",
                "source_parcellation": "aparc",
                "source_common_space": "native",
                "method": "MNE",
                "source_parcel_baseline_zscore": True,
                "parcel_baseline_samples": 2,
                "invalid_parcel_baselines": 0,
                "bandpass_low_hz": 8.0,
                "bandpass_high_hz": 13.0,
                "shared_bandpass_low_hz": 1.0,
                "shared_bandpass_high_hz": 40.0,
                "resample_enabled": True,
                "resample_sfreq": 200.0,
                "mne_epoch_reject_enabled": True,
                "mne_reject_mag_ft": 4000.0,
                "mne_reject_grad_ft_cm": 4000.0,
                "exclude_trial_outliers": True,
                "trial_outlier_z": 6.0,
                "n_runs": 1,
                "dropped_trials": 0,
            },
            None,
        )

        result, message = build_source_parcel_trial_time_courses(
            object(),
            {
                "selected_concept_label": "airplane",
                "window_start": 0.0,
                "window_end": 0.1,
                "bandpass_low_hz": 1.0,
                "bandpass_high_hz": 40.0,
                "parcel_frequency_band": "alpha",
                "parcel_frequency_band_label": "Alpha (8–13 Hz)",
            },
        )

        self.assertEqual(result["data"].shape, (2, 2, 2))
        np.testing.assert_allclose(result["data"][0], [[3, 5], [3, 5]])
        np.testing.assert_allclose(result["times"], [0.0, 0.1])
        self.assertEqual(result["stimulus"], "airplane")
        self.assertEqual(result["normalization"], "parcel_baseline_zscore")
        self.assertEqual(result["parcel_baseline_samples"], 2)
        self.assertEqual(result["frequency_band"], "alpha")
        self.assertEqual(result["frequency_band_label"], "Alpha (8–13 Hz)")
        self.assertEqual(result["shared_bandpass_low_hz"], 1.0)
        self.assertEqual(result["shared_bandpass_high_hz"], 40.0)
        self.assertIn("Frequency band: Alpha (8–13 Hz)", message)
        self.assertIn("2 presentation(s)", message)
        called_options = build_features.call_args.args[1]
        self.assertEqual(called_options["feature_mode"], "raw")
        self.assertFalse(called_options["feature_binning_enabled"])
        self.assertTrue(called_options["_allow_single_class"])
        self.assertEqual(called_options["bandpass_low_hz"], 1.0)
        self.assertEqual(called_options["bandpass_high_hz"], 40.0)
        self.assertTrue(called_options["mne_epoch_reject_enabled"])
        self.assertTrue(called_options["exclude_trial_outliers"])
        self.assertFalse(called_options["baseline_correct"])
        self.assertTrue(called_options["source_parcel_baseline_zscore"])
        self.assertEqual(called_options["window_start"], 0.0)
        self.assertTrue(called_options["_limit_presentations_after_qc"])

    @mock.patch("core.concept_decoding_source.build_source_space_concept_features")
    def test_parcel_trial_wrapper_applies_presentation_limit_after_qc(self, build_features):
        feature_meta = pd.DataFrame(
            [
                {"roi": "parcel-a", "relative_time_s": sample_time}
                for sample_time in (0.0, 0.1)
            ]
        )
        build_features.return_value = (
            {
                "X": np.asarray([[1, 2], [3, 4], [5, 6]], dtype=np.float32),
                "roi_names": ["parcel-a"],
                "feature_meta_df": feature_meta,
                "trial_df": pd.DataFrame(
                    {
                        "trial_index": [1, 2, 3],
                        "source_fif_path": ["run1.fif", "run1.fif", "run2.fif"],
                    }
                ),
                "source_parcellation": "aparc",
                "n_runs": 2,
                "dropped_trials": 1,
            },
            None,
        )

        result, message = build_source_parcel_trial_time_courses(
            object(), {"max_presentations": 2}
        )

        self.assertEqual(result["data"].shape, (2, 1, 2))
        self.assertEqual(result["trial_df"]["trial_index"].tolist(), [1, 2])
        self.assertEqual(result["available_presentations"], 3)
        self.assertEqual(result["n_runs"], 1)
        self.assertIn("Showing 2 of 3 usable", message)
        called_options = build_features.call_args.args[1]
        self.assertEqual(called_options["_presentation_candidate_limit"], 4)

    @mock.patch("core.concept_decoding_source.build_source_space_concept_features")
    def test_parcel_trial_wrapper_rejects_inconsistent_feature_shape(self, build_features):
        build_features.return_value = (
            {
                "X": np.asarray([[1, 2, 3]], dtype=np.float32),
                "roi_names": ["parcel-a", "parcel-b"],
                "feature_meta_df": pd.DataFrame(
                    [
                        {"roi": "parcel-a", "relative_time_s": 0.0},
                        {"roi": "parcel-a", "relative_time_s": 0.1},
                        {"roi": "parcel-b", "relative_time_s": 0.0},
                        {"roi": "parcel-b", "relative_time_s": 0.1},
                    ]
                ),
            },
            None,
        )

        result, message = build_source_parcel_trial_time_courses(object(), {})

        self.assertIsNone(result)
        self.assertIn("inconsistent", message.lower())

    def test_temporal_display_binning_preserves_raw_data_and_averages_wider_bins(self):
        times = np.linspace(0.0, 0.4, 81)
        data = np.arange(81, dtype=float).reshape(1, 1, 81)

        raw, raw_times, raw_extent, raw_bin_ms = temporal_bin_parcel_time_courses(
            data, times, 5.0
        )
        binned, centers, extent, bin_ms = temporal_bin_parcel_time_courses(
            data, times, 10.0
        )

        self.assertEqual(raw.shape, (1, 1, 81))
        np.testing.assert_allclose(raw_times, times)
        np.testing.assert_allclose(raw_extent, [0.0, 400.0])
        self.assertAlmostEqual(raw_bin_ms, 5.0)
        self.assertEqual(binned.shape, (1, 1, 40))
        np.testing.assert_allclose(binned[0, 0, :3], [0.5, 2.5, 4.5])
        np.testing.assert_allclose(centers[[0, -1]], [0.005, 0.395])
        np.testing.assert_allclose(extent, [0.0, 400.0])
        self.assertEqual(bin_ms, 10.0)

    def test_anatomical_parcel_order_groups_hemisphere_then_region(self):
        names = [
            "ctx-rh-superiorfrontal",
            "ctx-lh-bankssts",
            "ctx-rh-cuneus",
            "ctx-lh-precentral",
            "ctx-lh-cuneus",
        ]

        order, groups = anatomical_parcel_row_order(names, "aparc")
        ordered_names = [names[int(index)] for index in order]

        self.assertEqual(
            ordered_names,
            [
                "ctx-lh-cuneus",
                "ctx-lh-bankssts",
                "ctx-lh-precentral",
                "ctx-rh-cuneus",
                "ctx-rh-superiorfrontal",
            ],
        )
        self.assertEqual(
            [group["label"] for group in groups],
            [
                "LH › Occipital / visual (1)",
                "LH › Temporal / auditory (1)",
                "LH › Sensorimotor (1)",
                "RH › Occipital / visual (1)",
                "RH › Frontal / insular (1)",
            ],
        )

    def test_anatomical_parcel_order_supports_hcp_combined_groups(self):
        names = [
            "Dorsal Stream Visual Cortex-rh",
            "DorsoLateral Prefrontal Cortex-lh",
            "Early Auditory Cortex-lh",
            "Primary Visual Cortex (V1)-lh",
        ]

        order, groups = anatomical_parcel_row_order(names, "HCPMMP1_combined")

        self.assertEqual(
            [names[int(index)] for index in order],
            [
                "Primary Visual Cortex (V1)-lh",
                "Early Auditory Cortex-lh",
                "DorsoLateral Prefrontal Cortex-lh",
                "Dorsal Stream Visual Cortex-rh",
            ],
        )
        self.assertEqual(
            [(group["hemisphere"], group["region"]) for group in groups],
            [
                ("Left", "Occipital / visual"),
                ("Left", "Temporal / auditory"),
                ("Left", "Frontal / insular"),
                ("Right", "Occipital / visual"),
            ],
        )

    def test_visual_and_semantic_parcel_filters(self):
        visual_samples = [
            "ctx-lh-cuneus",
            "ctx-rh-lateraloccipital",
            "ctx-lh-pericalcarine",
            "ctx-rh-lingual",
            "ctx-lh-fusiform",
            "ctx-rh-inferiortemporal",
        ]
        for name in visual_samples:
            self.assertTrue(is_visual_parcel(name, "aparc"), f"{name} should be visual")

        semantic_samples = [
            "ctx-lh-temporalpole",
            "ctx-lh-middletemporal",
            "ctx-lh-inferiorparietal",
            "ctx-lh-parstriangularis",
            "ctx-lh-parsorbitalis",
            "ctx-lh-bankssts",
            "ctx-rh-supramarginal",
            "ctx-rh-precuneus",
            "ctx-lh-rostralmiddlefrontal",
        ]
        for name in semantic_samples:
            self.assertTrue(is_semantic_parcel(name, "aparc"), f"{name} should be semantic")

        non_visual_non_semantic = [
            "ctx-lh-precentral",
            "ctx-rh-postcentral",
            "ctx-lh-paracentral",
        ]
        for name in non_visual_non_semantic:
            self.assertFalse(is_visual_parcel(name, "aparc"), f"{name} should not be visual")
            self.assertFalse(is_semantic_parcel(name, "aparc"), f"{name} should not be semantic")

        hcp_visual = ["Primary Visual Cortex (V1)-lh", "Early Visual Cortex-lh", "V1_ROI-rh", "FFC_ROI-lh"]
        for name in hcp_visual:
            self.assertTrue(is_visual_parcel(name, "HCPMMP1_combined"), f"{name} should be visual in HCP")

        # Volume source space (aparc+aseg)
        for name in visual_samples:
            self.assertTrue(is_visual_parcel(name, "aparc+aseg"), f"{name} should be visual in volume")

        volume_semantic = semantic_samples + [
            "Left-Hippocampus",
            "Right-Hippocampus",
            "Left-Amygdala",
            "Right-Amygdala",
        ]
        for name in volume_semantic:
            self.assertTrue(is_semantic_parcel(name, "aparc+aseg"), f"{name} should be semantic in volume")

        volume_non_semantic = non_visual_non_semantic + [
            "ctx-lh-superiorfrontal",
            "ctx-lh-superiorparietal",
            "ctx-lh-transversetemporal",
            "Left-Thalamus-Proper",
            "Left-Cerebellum-Cortex",
            "Left-Cerebral-White-Matter",
        ]
        for name in volume_non_semantic:
            self.assertFalse(is_semantic_parcel(name, "aparc+aseg"), f"{name} should not be semantic in volume")
            self.assertFalse(is_visual_parcel(name, "aparc+aseg"), f"{name} should not be visual in volume")

    def test_parcel_baseline_zscore_is_trial_and_parcel_specific(self):
        times = np.asarray([-0.1, -0.05, 0.0, 0.05])
        data = np.asarray(
            [
                [[1.0, 3.0, 5.0, 7.0], [10.0, 14.0, 18.0, 22.0]],
                [[2.0, 6.0, 10.0, 14.0], [20.0, 28.0, 36.0, 44.0]],
            ]
        )

        z_data, n_baseline, n_invalid = baseline_zscore_parcel_time_courses(
            data, times, -0.1, 0.0
        )

        np.testing.assert_allclose(z_data[:, :, :2].mean(axis=2), 0.0, atol=1e-12)
        np.testing.assert_allclose(z_data[:, :, :2].std(axis=2), 1.0, atol=1e-12)
        np.testing.assert_allclose(z_data[0, :, 2:], [[3.0, 5.0], [3.0, 5.0]])
        np.testing.assert_allclose(z_data[1, :, 2:], [[3.0, 5.0], [3.0, 5.0]])
        self.assertEqual(n_baseline, 2)
        self.assertEqual(n_invalid, 0)

    def test_hilbert_power_is_db_relative_to_each_parcel_baseline(self):
        sfreq = 200.0
        times = np.arange(-1.0, 1.0 + 1.0 / sfreq, 1.0 / sfreq)
        amplitude = np.where(times < 0.0, 1.0, 2.0)
        signal = amplitude * np.sin(2.0 * np.pi * 10.0 * times)
        data = np.stack([signal, signal * 3.0], axis=0)[None, :, :]

        power_db, display_times, n_baseline, n_invalid = (
            baseline_db_hilbert_parcel_power(
                data,
                times,
                baseline_start=-0.8,
                baseline_end=-0.4,
                display_start=0.2,
                display_end=0.6,
            )
        )

        self.assertEqual(power_db.shape, (1, 2, 81))
        np.testing.assert_allclose(display_times[[0, -1]], [0.2, 0.6], atol=1e-12)
        np.testing.assert_allclose(np.median(power_db, axis=2), 10.0 * np.log10(4.0), atol=0.15)
        self.assertEqual(n_baseline, 80)
        self.assertEqual(n_invalid, 0)

    @mock.patch("core.concept_decoding_source.build_source_space_concept_features")
    def test_parcel_trial_power_uses_padded_unnormalized_source_epoch(self, build_features):
        sfreq = 200.0
        sample_times = np.arange(-0.45, 0.65 + 1.0 / sfreq, 1.0 / sfreq)
        amplitude = np.where(sample_times < 0.0, 1.0, 2.0)
        signal = amplitude * np.sin(2.0 * np.pi * 10.0 * sample_times)
        feature_meta = pd.DataFrame(
            [
                {"roi": "parcel-a", "relative_time_s": sample_time}
                for sample_time in sample_times
            ]
        )
        build_features.return_value = (
            {
                "X": signal[None, :].astype(np.float32),
                "roi_names": ["parcel-a"],
                "feature_meta_df": feature_meta,
                "trial_df": pd.DataFrame({"trial_index": [1]}),
                "source_parcellation": "aparc",
                "bandpass_low_hz": 8.0,
                "bandpass_high_hz": 13.0,
                "shared_bandpass_low_hz": 1.0,
                "shared_bandpass_high_hz": 40.0,
                "resample_enabled": True,
                "resample_sfreq": sfreq,
                "n_runs": 1,
                "dropped_trials": 0,
            },
            None,
        )

        result, message = build_source_parcel_trial_time_courses(
            object(),
            {
                "selected_concept_label": "airplane",
                "window_start": 0.0,
                "window_end": 0.1,
                "parcel_baseline_start": -0.2,
                "parcel_baseline_end": 0.0,
                "bandpass_low_hz": 1.0,
                "bandpass_high_hz": 40.0,
                "parcel_frequency_band": "alpha",
                "parcel_representation": "band_power_db",
                "enable_parcel_timecourse_cache": False,
            },
        )

        self.assertEqual(result["normalization"], "parcel_power_db")
        self.assertEqual(result["parcel_representation"], "band_power_db")
        self.assertEqual(result["data"].shape, (1, 1, 21))
        np.testing.assert_allclose(result["times"][[0, -1]], [0.0, 0.1], atol=1e-12)
        self.assertIn("Hilbert power expressed in dB", message)
        called_options = build_features.call_args.args[1]
        self.assertFalse(called_options["source_parcel_baseline_zscore"])
        self.assertAlmostEqual(called_options["window_start"], -0.45)
        self.assertAlmostEqual(called_options["window_end"], 0.35)
        self.assertGreaterEqual(called_options["max_source_features"], 500000)

    def test_parcel_baseline_zscore_precedes_temporal_feature_binning(self):
        times = np.asarray([-0.1, -0.05, 0.0, 0.05, 0.1, 0.15])
        data = np.asarray([[[1.0, 3.0, 5.0, 7.0, 9.0, 11.0]]])

        normalized, _, _ = baseline_zscore_parcel_time_courses(
            data, times, -0.1, 0.0
        )
        bin_masks = [
            (times >= 0.0) & (times < 0.1),
            (times >= 0.1) & (times <= 0.15),
        ]
        values = source_trial_feature_values(
            normalized[0], ["mean"], bin_masks
        )

        # Baseline mean=2 and std=1, then signed means are computed per bin.
        np.testing.assert_allclose(values, [4.0, 8.0])

    def test_five_ms_bins_align_to_two_hundred_hz_sample_grid(self):
        times = np.arange(-20, 81, dtype=float) / 200.0
        bin_edges = feature_edges_for_concept_decoding(0.0, 0.4, 5.0)

        masks = source_bin_masks(
            times, bin_edges, include_last_endpoint=False
        )
        counts = [int(np.sum(mask)) for mask in masks]
        assignments = np.sum(np.vstack(masks), axis=0)
        poststimulus = (times >= 0.0) & (times < 0.4)

        self.assertEqual(counts, [1] * 80)
        np.testing.assert_array_equal(assignments[poststimulus], 1)
        np.testing.assert_array_equal(assignments[~poststimulus], 0)

    def test_parcel_trial_summary_maps_report_mean_and_dominant_sign_fraction(self):
        data = np.asarray(
            [
                [[2.0, 1.0, 0.0], [-2.0, np.nan, 1.0]],
                [[4.0, -1.0, 0.0], [-4.0, 3.0, 2.0]],
                [[6.0, 2.0, 0.0], [2.0, -3.0, -3.0]],
                [[8.0, 3.0, 0.0], [-6.0, 5.0, 4.0]],
            ]
        )

        signed_mean, consistency = parcel_trial_summary_maps(data)

        np.testing.assert_allclose(signed_mean[0], [5.0, 1.25, 0.0])
        np.testing.assert_allclose(signed_mean[1], [-2.5, 5.0 / 3.0, 1.0])
        np.testing.assert_allclose(consistency[0, :2], [1.0, 0.75])
        np.testing.assert_allclose(consistency[1], [0.75, 2.0 / 3.0, 0.75])
        self.assertTrue(np.isnan(consistency[0, 2]))

    def test_parcel_trial_contrast_map_subtracts_group_means(self):
        primary = np.asarray(
            [
                [[2.0, 4.0], [10.0, 14.0]],
                [[4.0, 8.0], [14.0, 18.0]],
            ]
        )
        comparison = np.asarray(
            [
                [[1.0, 3.0], [4.0, 8.0]],
                [[3.0, 5.0], [8.0, 10.0]],
                [[5.0, 7.0], [12.0, 12.0]],
            ]
        )

        contrast = parcel_trial_contrast_map(primary, comparison)

        np.testing.assert_allclose(contrast, [[0.0, 1.0], [4.0, 6.0]])

    def test_parcel_trial_contrast_map_requires_matching_grids(self):
        with self.assertRaisesRegex(ValueError, "matching parcel and time"):
            parcel_trial_contrast_map(np.ones((2, 3, 4)), np.ones((2, 3, 5)))

    def test_parcel_ranking_reports_peak_agreement_and_nested_loo_score(self):
        primary = np.asarray(
            [
                [[0.0, 4.0], [0.0, 0.2]],
                [[0.0, 5.0], [0.0, -0.1]],
                [[0.0, 6.0], [0.0, 0.1]],
            ]
        )
        comparison = np.asarray(
            [
                [[0.0, -4.0], [0.0, 0.1]],
                [[0.0, -5.0], [0.0, -0.2]],
                [[0.0, -6.0], [0.0, 0.0]],
            ]
        )

        rows = rank_parcel_contrast_features(
            primary,
            comparison,
            ["parcel-a", "parcel-b"],
            [0.01, 0.03],
            20.0,
            extent_ms=(0.0, 40.0),
            primary_label="tractor",
            comparison_label="watch",
        )

        self.assertEqual(rows[0]["rank"], 1)
        self.assertEqual(rows[0]["parcel"], "parcel-a")
        self.assertEqual(rows[0]["direction"], "tractor")
        self.assertEqual(rows[0]["peak_start_ms"], 20.0)
        self.assertEqual(rows[0]["peak_end_ms"], 40.0)
        self.assertEqual(rows[0]["primary_sign_count"], 3)
        self.assertEqual(rows[0]["comparison_sign_count"], 3)
        self.assertAlmostEqual(rows[0]["nested_loo_balanced_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
