"""Scientific invariants and source-pipeline coverage for band-power decoding."""
import os
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import mne
import numpy as np
import pandas as pd

from core.concept_decoding_spectral import (
    resolve_source_power_bands, source_bandpower_trial, source_power_feature_metadata,
)
from core.concept_decoding_source import build_source_space_concept_features, source_bin_masks
from core.concept_decoding_subsets import subset_payload


class SourceSpectralTests(unittest.TestCase):
    def setUp(self):
        self.sfreq = 200.0
        self.times = np.arange(-3, 3, 1 / self.sfreq)
        self.masks = source_bin_masks(self.times, [(0.3, 0.7), (0.7, 1.1)])
        self.bands = [("alpha", 8., 13.), ("beta", 13., 30.)]

    def stc(self, data, volume=False):
        vector = data.ndim == 3
        cls = (mne.VolVectorSourceEstimate if vector else mne.VolSourceEstimate) if volume else (
            mne.VectorSourceEstimate if vector else mne.SourceEstimate)
        vertices = [np.arange(len(data))] if volume else [np.arange(len(data)), np.array([], dtype=int)]
        return cls(data, vertices, tmin=self.times[0], tstep=1 / self.sfreq, subject="sample")

    def compute(self, data, volume=False, baseline=None, aggregate=None, **kwargs):
        return source_bandpower_trial(self.stc(data, volume), self.bands, self.masks, baseline,
                                      aggregate or (lambda stc: stc.data), **kwargs)

    def test_frequency_recovery_surface_and_volume_without_rectification(self):
        # Opposite source signs must not cancel; a magnitude-first transform
        # would create a spurious 20-Hz peak from this 10-Hz signal.
        sine = np.sin(2 * np.pi * 10 * self.times)
        vector = np.zeros((2, 3, len(sine)))
        vector[0, 0] = sine
        vector[1, 0] = -sine
        for volume in (False, True):
            with self.subTest(volume=volume):
                output = self.compute(vector, volume, aggregate=lambda stc: stc.data.mean(axis=0, keepdims=True))
                self.assertEqual(output.shape, (1, 2, 2))
                self.assertTrue(np.all(output[0, 0] - output[0, 1] > 2))
                # Unit-amplitude sine has unit squared analytic-envelope power.
                np.testing.assert_allclose(output[0, 0], 0.0, atol=0.05)

    def test_vector_rotation_and_vertex_batching_preserve_power(self):
        rng = np.random.default_rng(5)
        data = rng.normal(size=(5, 3, len(self.times)))
        rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        rotated = np.einsum("ij,vjt->vit", rotation, data)
        original = self.compute(data, vertex_batch_size=1)
        transformed = self.compute(rotated, vertex_batch_size=3)
        np.testing.assert_allclose(original, transformed, rtol=1e-5, atol=1e-5)

    def test_fixed_signed_matches_single_vector_component(self):
        data = np.sin(2 * np.pi * 10 * self.times)[None, :]
        vector = np.zeros((1, 3, len(self.times)))
        vector[:, 0] = data
        np.testing.assert_allclose(self.compute(data), self.compute(vector), atol=1e-6)

    def test_db_normalizes_trial_gain_and_recovers_power_ratio(self):
        signal = np.sin(2 * np.pi * 10 * self.times)
        signal[self.times >= 0] *= 2
        baseline = (self.times >= -1.1) & (self.times < -0.3)
        first = self.compute(signal[None, :], baseline=baseline)
        second = self.compute(7 * signal[None, :], baseline=baseline)
        np.testing.assert_allclose(first, second, atol=1e-5)
        np.testing.assert_allclose(first[0, 0], 10 * np.log10(4), atol=0.35)

    def test_roi_power_is_averaged_before_db_normalization(self):
        sine = np.sin(2 * np.pi * 10 * self.times)
        data = np.stack([sine.copy(), 2 * sine])
        data[0, self.times >= 0] *= 3
        baseline = (self.times >= -1.1) & (self.times < -0.3)
        output = self.compute(data, baseline=baseline,
                              aggregate=lambda stc: stc.data.mean(axis=0, keepdims=True))
        np.testing.assert_allclose(output[0, 0], 10 * np.log10(13 / 5), atol=0.3)

    def test_band_validation(self):
        self.assertEqual([b[0] for b in resolve_source_power_bands("all", 100, 4, 40)], ["theta", "alpha", "beta"])
        for selection in ("high_gamma", "wrong", "all,alpha", ""):
            with self.subTest(selection=selection), self.assertRaises(ValueError):
                resolve_source_power_bands(selection, 100, 4, 40)
        with self.assertRaises(ValueError):
            resolve_source_power_bands("all", 10, 8, 40)

    def test_invalid_baseline_and_empty_roi_fail(self):
        data = np.sin(2 * np.pi * 10 * self.times)[None, :]
        with self.assertRaisesRegex(ValueError, "baseline requires"):
            self.compute(data, baseline=np.zeros(len(self.times), dtype=bool))
        with self.assertRaisesRegex(ValueError, "empty or has zero"):
            self.compute(data, aggregate=lambda stc: np.zeros_like(stc.data))

    def test_metadata_matches_roi_band_time_order_and_subsetting(self):
        metadata = pd.DataFrame(source_power_feature_metadata(["A", "B"], self.bands, [(0., .1), (.1, .2)], True))
        values = np.arange(8).reshape(1, 8)
        payload = dict(X=values, feature_meta_df=metadata, roi_names=["A", "B"], band_defs=self.bands, n_bands=2)
        selected = subset_payload(payload, ["B"])
        np.testing.assert_array_equal(selected["X"], [[4, 5, 6, 7]])
        self.assertEqual(selected["feature_meta_df"]["band"].tolist(), ["alpha", "alpha", "beta", "beta"])
        self.assertEqual(selected["band_defs"], self.bands)

    def test_source_pipeline_surface_volume_and_common_space(self):
        """Real MNE epochs/filters/STCs with inverse and anatomical IO replaced."""
        import core.concept_decoding_source as source
        path = os.path.abspath("spectral-test.fif")
        raw = mne.io.RawArray(np.random.default_rng(4).normal(0, 1e-13, (1, 6000)),
                              mne.create_info(["MEG001"], 200., "mag"), verbose=False)
        labels = [mne.Label(np.array([0, 1]), hemi="lh", name="roi")]
        src = mne.SourceSpaces([
            dict(type="surf", vertno=np.array([0, 1]), nuse=2, id=101),
            dict(type="surf", vertno=np.array([], dtype=int), nuse=0, id=102),
        ])
        trials = pd.DataFrame(dict(source_fif_path=[path] * 4, onset_sample=[1000, 2000, 3000, 4000],
                                   canonical_stimulus=["a", "a", "b", "b"]))
        context = SimpleNamespace(_fif_trial_response_table=trials, _fif_filename=path,
                                  _annotate_trial_df_with_stimulus_groups=lambda df, _: df)
        for geometry, common, fixed, binned, normalize in [
            ("surface", False, False, True, True), ("volume", False, False, True, True),
            ("surface", True, False, False, True), ("surface", False, True, True, True),
            ("volume", False, False, False, False),
        ]:
            with self.subTest(geometry=geometry, common=common, fixed=fixed), ExitStack() as stack:
                def mock_inverse(epochs, inv, **kwargs):
                    self.assertEqual(kwargs["pick_ori"], None if fixed else "vector")
                    sine = np.sin(2 * np.pi * 10 * epochs.times)
                    if fixed:
                        data = np.stack([sine, -sine])
                        cls = mne.SourceEstimate
                    else:
                        data = np.zeros((2, 3, len(sine)))
                        data[0, 0], data[1, 0] = sine, -sine
                        cls = mne.VolVectorSourceEstimate if geometry == "volume" else mne.VectorSourceEstimate
                    verts = [np.array([0, 1])] if geometry == "volume" else [np.array([0, 1]), np.array([], dtype=int)]
                    return [cls(data.copy(), verts, tmin=epochs.tmin, tstep=1/200., subject="sample") for _ in epochs]
                replacements = dict(
                    _meg_3d_viewer=lambda ctx: object(),
                    concept_raw_for_path=lambda *args: raw,
                    concept_label_from_trial_row=lambda ctx, row, **kw: row["canonical_stimulus"],
                    configure_source_subject_for_fif=lambda *a, **k: (("sample", "/tmp/subjects"), None),
                    concept_source_labels_for_subject=lambda *a, **k: (labels, ["roi"], ["roi"], "aparc"),
                    concept_volume_labels_for_subject=lambda *a, **k: ("atlas.mgz", ["roi"], [0], ["roi"], ["roi"]),
                    _find_or_create_source_inverse=lambda *a: ("inverse.fif", {"src": src, "source_ori": 1 if fixed else 2}, None),
                    restore_meg_sensor_fif=lambda *a: None,
                    ensure_fsaverage_in_subjects_dir=lambda *a: "/tmp/subjects",
                    fsaverage_source_space=lambda *a: (src, "/tmp/subjects"),
                )
                for name, replacement in replacements.items():
                    stack.enter_context(patch.object(source, name, replacement))
                inverse = stack.enter_context(patch.object(mne.minimum_norm, "apply_inverse_epochs", side_effect=mock_inverse))
                morph = stack.enter_context(patch.object(source, "morph_stcs_to_common_space", side_effect=lambda stcs, *args: stcs))
                if geometry == "volume":
                    def volume_extract(stc, atlas, **kwargs):
                        self.assertEqual(atlas, ("atlas.mgz", ["roi"]))
                        self.assertEqual(kwargs["mode"], "mean")
                        self.assertTrue(np.all(stc.data >= 0))
                        return stc.data.mean(axis=0, keepdims=True)
                    stack.enter_context(patch.object(mne.VolSourceEstimate, "extract_label_time_course", volume_extract))
                options = dict(feature_mode="bandpower", source_power_bands="alpha,beta", source_power_baseline=normalize,
                               source_space_kind=geometry, source_common_space="fsaverage" if common else "native",
                               window_start=.1, window_end=.3, bin_size_ms=100, feature_binning_enabled=binned,
                               bandpass_low_hz=1., bandpass_high_hz=40., mne_resample_enabled=False,
                               mne_epoch_reject_enabled=False, exclude_trial_outliers=False,
                               preprocess_continuous_recording=False, source_parcel_baseline_zscore=True)
                payload, error = build_source_space_concept_features(context, options)
                self.assertIsNone(error, error)
                self.assertEqual(payload["X"].shape, (4, 4 if binned else 2))
                self.assertEqual(len(payload["feature_meta_df"]), payload["X"].shape[1])
                self.assertEqual(payload["y"].tolist(), ["a", "a", "b", "b"])
                self.assertFalse(payload["source_parcel_baseline_zscore"])
                self.assertFalse(payload["baseline_correct"])
                self.assertEqual(payload["n_bands"], 2)
                self.assertEqual(payload["source_power_baseline"], normalize)
                if not normalize:
                    self.assertEqual(payload["feature_meta_df"]["stat"].unique().tolist(), ["log10_band_power"])
                self.assertTrue(np.all(np.isfinite(payload["X"])))
                self.assertTrue(inverse.called)
                self.assertEqual(morph.called, common)


if __name__ == "__main__":
    unittest.main()
