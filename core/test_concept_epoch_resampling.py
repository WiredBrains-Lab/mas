"""Regression checks using real MNE epochs and resampling."""

import unittest
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import mne
import numpy as np

from core.concept_decoding_signal import (
    concept_cached_epochs_for_file,
    _prune_concept_epoch_disk_cache,
    concept_preprocessed_recording,
    _events_for_preprocessed_recording,
)


class EpochResamplingTests(unittest.TestCase):
    def setUp(self):
        self.sfreq = 1000.0
        self.times = np.arange(6000) / self.sfreq
        data = (np.sin(2 * np.pi * 7 * self.times) + 0.2 * self.times) * 1e-13
        self.raw = mne.io.RawArray(
            data[None], mne.create_info(["MEG001"], self.sfreq, "mag"),
            first_samp=500, verbose="error",
        )
        self.events = np.array([[2500, 0, 1], [4500, 0, 1]])

    def extract(self, **kwargs):
        params = dict(
            context=SimpleNamespace(), options={"enable_signal_epoch_cache": False, "preprocess_continuous_recording": False},
            raw=self.raw, path_abs="synthetic.fif", events_arr=self.events,
            channel_names=self.raw.ch_names, bandpass_low_hz=None,
            bandpass_high_hz=None, epoch_tmin=-0.1, epoch_tmax=0.4,
            epoch_baseline=None, resample_sfreq=200.0,
        )
        params.update(kwargs)
        payload, error, _ = concept_cached_epochs_for_file(**params)
        self.assertIsNone(error)
        self.assertIsNotNone(payload)
        return payload

    def test_real_context_reduces_edge_error_and_preserves_timing(self):
        payload = self.extract()
        expected_times = np.arange(-20, 81) / 200.0
        np.testing.assert_allclose(payload["times"], expected_times, atol=1e-12)
        np.testing.assert_array_equal(payload["selected_idx"], [0, 1])
        unpadded = mne.Epochs(
            self.raw, self.events, {"trial": 1}, tmin=-0.1, tmax=0.4,
            baseline=None, preload=True, verbose="error",
        ).resample(200, verbose="error")
        # Compare with the known smooth continuous signal at each output time.
        t = payload["times"] + 2.0
        truth = (np.sin(2 * np.pi * 7 * t) + 0.2 * t) * 1e-13
        edge = np.r_[0:10, -10:0]
        padded_error = np.mean((payload["epoch_data"][0, 0, edge] - truth[edge]) ** 2)
        t_old = unpadded.times + 2.0
        truth_old = (np.sin(2 * np.pi * 7 * t_old) + 0.2 * t_old) * 1e-13
        old_error = np.mean((unpadded.get_data()[0, 0, edge] - truth_old[edge]) ** 2)
        self.assertLess(padded_error, old_error)

    def test_boundary_and_annotation_drops_keep_original_trial_indices(self):
        self.raw.set_annotations(mne.Annotations([3.45], [0.1], ["BAD_context"]))
        events = np.array([[700, 0, 1], [2500, 0, 1], [4500, 0, 1], [6100, 0, 1]])
        payload = self.extract(events_arr=events)
        np.testing.assert_array_equal(payload["selected_idx"], [1])
        self.assertEqual(len(payload["drop_log"]), 4)
        self.assertTrue(payload["drop_log"][0])
        self.assertEqual(payload["drop_log"][1], ())
        self.assertIn("BAD_context", payload["drop_log"][2])
        self.assertTrue(payload["drop_log"][3])

    def test_rejection_uses_requested_window_not_padding(self):
        self.raw._data[0, 1500] = 1e-9  # Context only for first event at 2 seconds.
        payload = self.extract(epoch_reject={"mag": 1e-11})
        np.testing.assert_array_equal(payload["selected_idx"], [0, 1])
        self.raw._data[0, 4000] = 1e-9  # Inside the second trial's analysis window.
        payload = self.extract(epoch_reject={"mag": 1e-11})
        np.testing.assert_array_equal(payload["selected_idx"], [0])
        self.assertIn("MEG001", payload["drop_log"][1])

    def test_noninteger_ratio_filters_and_returns_requested_sampling_rate(self):
        raw = mne.io.RawArray(
            self.raw.get_data(), mne.create_info(["MEG001"], 600.615, "mag"),
            first_samp=500, verbose="error",
        )
        payload = self.extract(raw=raw, bandpass_low_hz=1.0, bandpass_high_hz=40.0)
        self.assertEqual(payload["sfreq"], 200.0)
        np.testing.assert_allclose(np.diff(payload["times"]), 1 / 200.0, atol=1e-12)
        self.assertLessEqual(abs(payload["times"][0] + 0.1), 0.5 / 200.0)
        self.assertLessEqual(abs(payload["times"][-1] - 0.4), 0.5 / 200.0)
        np.testing.assert_array_equal(payload["selected_idx"], [0, 1])

    def test_open_baseline_does_not_expand_into_padding(self):
        implicit = self.extract(epoch_baseline=(None, 0.0))
        explicit = self.extract(epoch_baseline=(-0.1, 0.0))
        np.testing.assert_allclose(implicit["epoch_data"], explicit["epoch_data"], atol=0)

    def test_no_downsampling_does_not_require_extra_context(self):
        events = np.array([[700, 0, 1]])
        for rate in (None, 1000.0):
            with self.subTest(rate=rate):
                payload = self.extract(events_arr=events, resample_sfreq=rate)
                np.testing.assert_array_equal(payload["selected_idx"], [0])
                self.assertEqual(payload["epoch_data"].shape[-1], 501)

    def test_all_trials_dropped_preserves_qc_reasons(self):
        payload = self.extract(events_arr=np.array([[700, 0, 1]]))
        self.assertEqual(payload["epoch_data"].shape, (0, 1, 101))
        self.assertEqual(len(payload["selected_idx"]), 0)
        self.assertTrue(payload["drop_log"][0])
        self.assertEqual(payload["sfreq"], 200.0)

    def test_safe_integer_reduction_skips_fft_and_matches_filtered_samples(self):
        with mock.patch.object(mne.Epochs, "resample", side_effect=AssertionError("FFT called")):
            payload = self.extract(bandpass_low_hz=1.0, bandpass_high_hz=40.0)
        filtered = self.raw.copy().filter(1.0, 40.0, picks="meg", verbose="error")
        expected = mne.Epochs(
            filtered, self.events, {"trial": 1}, tmin=-0.1, tmax=0.4,
            baseline=None, preload=True, verbose="error",
        ).decimate(5, verbose="error")
        np.testing.assert_allclose(payload["times"], expected.times, atol=1e-12)
        np.testing.assert_allclose(payload["epoch_data"], expected.get_data(), rtol=1e-6, atol=1e-20)
        np.testing.assert_array_equal(payload["selected_idx"], [0, 1])

    def test_unsafe_or_noninteger_reductions_do_not_decimate(self):
        original_decimate = mne.Epochs.decimate

        def allow_constructor_noop(epochs, decim, *args, **kwargs):
            self.assertEqual(decim, 1)  # MNE calls decimate(1) during initialization.
            return original_decimate(epochs, decim, *args, **kwargs)

        for high, target in ((None, 200.0), (80.0, 200.0), (40.0, 210.0), (1.0, 4.0)):
            with self.subTest(high=high, target=target):
                with mock.patch.object(mne.Epochs, "decimate", new=allow_constructor_noop):
                    self.extract(bandpass_high_hz=high, resample_sfreq=target)

    def test_cache_returns_cropped_epochs_and_padding_invalidates_cache(self):
        context = SimpleNamespace()
        options = {"enable_signal_epoch_cache": True, "preprocess_continuous_recording": False}
        first = self.extract(context=context, options=options)
        second = self.extract(context=context, options=options)
        self.assertIs(first, second)
        with mock.patch("core.concept_decoding_signal.CONCEPT_RESAMPLE_PADDING_S", 0.75):
            changed = self.extract(context=context, options=options)
        self.assertIsNot(first, changed)
        np.testing.assert_allclose(first["times"], changed["times"], atol=1e-12)

    def test_disk_cache_survives_new_context_and_raw_instance(self):
        with tempfile.TemporaryDirectory() as directory:
            source = str(Path(directory) / "source_raw.fif")
            self.raw.save(source, overwrite=True, verbose="error")
            options = {"signal_epoch_cache_dir": str(Path(directory) / "cache")}
            raw = mne.io.read_raw_fif(source, verbose="error")
            first = self.extract(raw=raw, path_abs=source, options=options)
            reopened = mne.io.read_raw_fif(source, verbose="error")
            with mock.patch("core.concept_decoding_signal.apply_concept_bandpass_to_raw", side_effect=AssertionError("recomputed")):
                reused = self.extract(raw=reopened, path_abs=source, options=options)
            np.testing.assert_array_equal(first["epoch_data"], reused["epoch_data"])
            np.testing.assert_array_equal(first["times"], reused["times"])
            self.assertEqual(first["drop_log"], reused["drop_log"])
            self.assertEqual(len(list(Path(options["signal_epoch_cache_dir"]).glob("*.npz"))), 1)

    def test_disk_cache_invalidates_for_config_annotations_and_source_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source_raw.fif"
            self.raw.save(source, overwrite=True, verbose="error")
            raw = mne.io.read_raw_fif(source, verbose="error")
            options = {"signal_epoch_cache_dir": str(Path(directory) / "cache")}
            self.extract(raw=raw, path_abs=source, options=options)
            self.extract(raw=raw, path_abs=source, options=options, resample_sfreq=250.0)
            raw.set_annotations(mne.Annotations([0.0], [0.01], ["note"]))
            self.extract(raw=raw, path_abs=source, options=options)
            stat = source.stat()
            os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1000000))
            self.extract(raw=raw, path_abs=source, options=options)
            self.assertEqual(len(list(Path(options["signal_epoch_cache_dir"]).glob("*.npz"))), 4)

    def test_corrupt_disk_cache_rebuilds_and_opt_out_does_not_write(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source_raw.fif"
            self.raw.save(source, overwrite=True, verbose="error")
            raw = mne.io.read_raw_fif(source, verbose="error")
            cache_dir = Path(directory) / "cache"
            options = {"signal_epoch_cache_dir": str(cache_dir)}
            first = self.extract(raw=raw, path_abs=source, options=options)
            cached_file = next(cache_dir.glob("*.npz"))
            cached_file.write_bytes(b"incomplete cache")
            rebuilt = self.extract(raw=raw, path_abs=source, options=options)
            np.testing.assert_array_equal(first["epoch_data"], rebuilt["epoch_data"])
            with np.load(cached_file, allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved["epoch_data"], first["epoch_data"])
            disabled_dir = Path(directory) / "disabled"
            self.extract(raw=raw, path_abs=source, options={
                "signal_epoch_cache_dir": str(disabled_dir), "enable_signal_epoch_disk_cache": False,
            })
            self.assertFalse(disabled_dir.exists())
            self.assertFalse(list(cache_dir.glob(".concept_epochs_*")))

    def test_disk_budget_prunes_only_owned_old_entries(self):
        with tempfile.TemporaryDirectory() as directory:
            old = Path(directory) / "concept_epochs_old.npz"
            new = Path(directory) / "concept_epochs_new.npz"
            unrelated = Path(directory) / "important.npz"
            for path in (old, new, unrelated):
                path.write_bytes(b"12345")
            os.utime(old, (1, 1))
            _prune_concept_epoch_disk_cache(directory, max_bytes=5)
            self.assertFalse(old.exists())
            self.assertTrue(new.exists())
            self.assertTrue(unrelated.exists())

    def test_recording_cache_reused_across_events_windows_and_baselines(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source_raw.fif"
            self.raw.save(source, overwrite=True, verbose="error")
            raw = mne.io.read_raw_fif(source, verbose="error")
            options = {"signal_epoch_cache_dir": str(Path(directory) / "cache")}
            self.extract(raw=raw, path_abs=source, options=options, bandpass_low_hz=1., bandpass_high_hz=40.)
            # A new epoch configuration cannot use the existing epoch entry.
            # It must load the continuous recording without filtering/resampling.
            with mock.patch("core.concept_decoding_signal.apply_concept_bandpass_to_raw", side_effect=AssertionError("recomputed recording")):
                changed = self.extract(
                    raw=raw, path_abs=source, options=options,
                    bandpass_low_hz=1., bandpass_high_hz=40.,
                    events_arr=self.events[:1], epoch_tmin=-0.2, epoch_tmax=0.6,
                    epoch_baseline=(-0.2, 0.0),
                )
            self.assertEqual(changed["epoch_data"].shape, (1, 1, 161))
            self.assertEqual(len(list((Path(directory) / "cache").glob("recording_*"))), 1)

    def test_recording_events_match_mne_joint_resampling(self):
        for native, first_samp in ((1000., 503), (600.615, 317)):
            with self.subTest(native=native, first_samp=first_samp):
                raw = mne.io.RawArray(
                    self.raw.get_data(), mne.create_info(["MEG001"], native, "mag"),
                    first_samp=first_samp, verbose="error",
                )
                events = np.array([[first_samp + 1701, 0, 1], [first_samp + 2703, 0, 1]])
                expected_raw, expected_events = raw.copy().resample(200., events=events, verbose="error")
                prepared = concept_preprocessed_recording(raw, "unsaved.fif", {}, None, None, 200.)
                mapped = _events_for_preprocessed_recording(events, raw, prepared)
                np.testing.assert_array_equal(mapped, expected_events)
                np.testing.assert_allclose(prepared.get_data(), expected_raw.get_data(), atol=1e-25)
                self.assertEqual(prepared.first_samp, expected_raw.first_samp)
                self.assertEqual(raw.info["sfreq"], native)
                self.assertEqual(events[0, 0], first_samp + 1701)

    def test_recording_cache_matches_disabled_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source_raw.fif"
            self.raw.save(source, overwrite=True, verbose="error")
            raw = mne.io.read_raw_fif(source, verbose="error")
            enabled = self.extract(raw=raw, path_abs=source, options={"signal_epoch_cache_dir": str(Path(directory) / "cache")})
            disabled = self.extract(raw=raw, path_abs=source, options={"enable_signal_epoch_cache": False})
            np.testing.assert_allclose(enabled["epoch_data"], disabled["epoch_data"], atol=1e-20)
            np.testing.assert_array_equal(enabled["selected_idx"], disabled["selected_idx"])

    def test_corrupt_recording_cache_rebuilds(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source_raw.fif"
            self.raw.save(source, overwrite=True, verbose="error")
            raw = mne.io.read_raw_fif(source, verbose="error")
            cache = Path(directory) / "cache"
            options = {"signal_epoch_cache_dir": str(cache)}
            first = concept_preprocessed_recording(raw, str(source), options, 1., 40., 200.)
            expected = first.get_data()
            first.close()
            next(cache.glob("recording_*/prepared_raw.fif")).write_bytes(b"broken")
            rebuilt = concept_preprocessed_recording(raw, str(source), options, 1., 40., 200.)
            np.testing.assert_array_equal(rebuilt.get_data(), expected)
            self.assertFalse(list(cache.glob(".recording_*")))

    def test_recording_cache_pruning_handles_split_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            old = Path(directory) / "recording_old"
            keep = Path(directory) / "recording_current"
            for entry in (old, keep):
                entry.mkdir()
                (entry / "prepared_raw.fif").write_bytes(b"12345")
                (entry / "prepared_raw-1.fif").write_bytes(b"12345")
            os.utime(old, (1, 1))
            _prune_concept_epoch_disk_cache(directory, max_bytes=10, keep=keep)
            self.assertFalse(old.exists())
            self.assertTrue((keep / "prepared_raw-1.fif").exists())


if __name__ == "__main__":
    unittest.main()
