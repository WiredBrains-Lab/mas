"""Real spawned workers preserve serial features, QC, and restart semantics."""
from contextlib import ExitStack
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import mne
import numpy as np
import pandas as pd

from core.concept_decoding_headless_source import HeadlessSourceContext
from core.concept_decoding_source import _extract_source_space_concept_features
from core.concept_decoding_source_parallel import _source_run_worker
from scripts.run_concept_decoding_cli import HeadlessConceptDecodingContext, parse_args, build_decoder_options


def _anatomy_patches(geometry):
    stack = ExitStack()
    entries = [dict(type='surf', vertno=np.array([0, 1]), nuse=2, id=101, subject_his_id='sample'),
               dict(type='surf', vertno=np.array([], dtype=int), nuse=0, id=102, subject_his_id='sample')]
    if geometry == 'volume':
        entries = [dict(type='vol', vertno=np.array([0, 1]), nuse=2, subject_his_id='sample')]
    for entry in entries:
        entry['nn'] = np.array([[0., 0., 1.], [0., 0., -1.]])
    src = mne.SourceSpaces(entries)
    stack.enter_context(patch('mne.minimum_norm.read_inverse_operator',
                              return_value=dict(src=src, source_ori=2)))
    stack.enter_context(patch('mne.read_labels_from_annot', return_value=[
        mne.Label(np.array([0, 1]), hemi='lh', name='test-lh')]))
    if geometry == 'volume':
        stack.enter_context(patch('core.concept_decoding_source.concept_volume_labels_for_subject',
            return_value=('atlas.mgz', ['Left-Test'], [0], ['Left-Test'], ['Left-Test'])))
        stack.enter_context(patch.object(mne.VolSourceEstimate, 'extract_label_time_course',
            lambda stc, *args, **kwargs: stc.data.mean(axis=0, keepdims=True)))

    def reconstruct(epochs, inv, **kwargs):
        outputs = []
        for values in epochs.get_data():
            data = np.stack([values[0], -values[0]])
            if kwargs['pick_ori'] == 'vector':
                data = np.stack([data, data * .5, data * .2], axis=1)
                cls = mne.VolVectorSourceEstimate if geometry == 'volume' else mne.VectorSourceEstimate
            else:
                cls = mne.VolSourceEstimate if geometry == 'volume' else mne.SourceEstimate
            outputs.append(cls(data, [entry['vertno'] for entry in entries],
                               tmin=epochs.tmin, tstep=1 / epochs.info['sfreq'], subject='sample'))
        return outputs
    stack.enter_context(patch('mne.minimum_norm.apply_inverse_epochs', side_effect=reconstruct))
    return stack


def _synthetic_worker(run, validated, file_df, options, settings):
    # Patches must be installed inside the spawned process, not just the parent.
    if 'first' in run['fif']:
        time.sleep(.25)
    with _anatomy_patches(settings['source_space_kind']):
        return _source_run_worker(run, validated, file_df, options, settings)


class ParallelSourceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.runs, tables = [], []
        for index, name in enumerate(['first', 'second']):
            fif = self.root / f'{name}_raw.fif'
            inv = self.root / f'{name}-inv.fif'
            inv.touch()
            raw = mne.io.RawArray(np.random.default_rng(index).normal(0, 1e-13, (1, 2400)),
                                 mne.create_info(['MEG001'], 100, 'mag'), verbose='error')
            # Rejection exercises trial/QC alignment and global post-extraction counts.
            raw.set_annotations(mne.Annotations([6.0], [.2], ['BAD_test']))
            raw.save(fif, verbose='error')
            self.runs.append(dict(fif=str(fif), inverse=str(inv), subject='sample', subjects_dir=str(self.root)))
            tables.append(pd.DataFrame(dict(source_fif_path=[str(fif)] * 7,
                onset_sample=np.arange(200, 1600, 200), stimulus=[['apple', 'chair'][index]] * 7,
                canonical_stimulus=[['apple', 'chair'][index]] * 7,
                trial_index_in_file=np.arange(7))))
        context = HeadlessConceptDecodingContext(pd.concat(tables, ignore_index=True), first_fif_path=self.runs[0]['fif'])
        viewer = HeadlessSourceContext(self.runs, 'surface')
        viewer._validated = {(run['inverse'], run['fif']) for run in self.runs}
        viewer.attach(context)
        self.context = context
        self.options = dict(source_feature_cache=False, feature_mode='bandpower',
            source_space_kind='surface', source_common_space='native', source_power_bands=['alpha', 'beta'],
            source_power_baseline=True, source_batch_size=2, source_workers=2, source_worker_threads=1,
            bandpass_low_hz=1., bandpass_high_hz=40., mne_resample_enabled=False,
            baseline_correct=False, mne_epoch_reject_enabled=False, exclude_trial_outliers=False,
            min_trials_per_class=2, enable_signal_epoch_cache=False,
            window_start=.1, window_end=.3, bin_size_ms=100., feature_binning_enabled=True)

    def test_serial_and_spawned_outputs_match(self):
        for geometry, mode in [('surface', 'bandpower'), ('volume', 'bandpower'), ('surface', 'raw')]:
            with self.subTest(geometry=geometry, mode=mode), _anatomy_patches(geometry):
                self.context.meg_3d._meg_source_space_kind = geometry
                options = dict(self.options, source_space_kind=geometry, feature_mode=mode)
                serial, error = _extract_source_space_concept_features(self.context, dict(options, source_workers=1))
                self.assertIsNone(error)
                with patch('core.concept_decoding_source_parallel._source_run_worker', _synthetic_worker):
                    parallel, error = _extract_source_space_concept_features(self.context, options)
                self.assertIsNone(error)
                np.testing.assert_array_equal(serial['X'], parallel['X'])
                np.testing.assert_array_equal(serial['y'], parallel['y'])
                for key in ['trial_df', 'trial_qc_df', 'feature_meta_df']:
                    pd.testing.assert_frame_equal(serial[key], parallel[key])
                for key in ['dropped_trials', 'n_runs', 'feature_names', 'parcel_baseline_samples']:
                    self.assertEqual(serial[key], parallel[key])
                self.assertEqual(parallel['n_runs'], 2)
                self.assertEqual(set(parallel['y']), {'apple', 'chair'})
                self.assertEqual(parallel['trial_df'].source_fif_path.iloc[0], self.runs[0]['fif'])

    def test_checkpoints_reuse_completed_runs_and_rebuild_bypasses(self):
        directory = self.root / 'checkpoints'
        options = dict(self.options, _source_run_checkpoint_dir=str(directory))
        with _anatomy_patches('surface'), patch('core.concept_decoding_source_parallel._source_run_worker', _synthetic_worker):
            first, error = _extract_source_space_concept_features(self.context, options)
        self.assertIsNone(error)
        checkpoints = list(directory.glob('*.joblib'))
        self.assertEqual(len(checkpoints), 2)
        with _anatomy_patches('surface'), patch('core.concept_decoding_source_parallel.ProcessPoolExecutor',
                side_effect=AssertionError('Spawned workers on checkpoint hit')):
            second, error = _extract_source_space_concept_features(self.context, options)
        self.assertIsNone(error)
        np.testing.assert_array_equal(first['X'], second['X'])
        # An incomplete prior analysis only needs the missing run recomputed.
        checkpoints[1].unlink()
        with _anatomy_patches('surface'), patch('core.concept_decoding_source_parallel._source_run_worker', _synthetic_worker):
            third, error = _extract_source_space_concept_features(self.context, options)
        self.assertIsNone(error)
        np.testing.assert_array_equal(first['X'], third['X'])
        self.assertEqual(len(list(directory.glob('*.joblib'))), 2)
        with _anatomy_patches('surface'), patch('core.concept_decoding_source_parallel._source_run_worker', _synthetic_worker):
            rebuilt, error = _extract_source_space_concept_features(self.context, dict(options, rebuild_source_features=True))
        self.assertIsNone(error)
        np.testing.assert_array_equal(first['X'], rebuilt['X'])

    def test_cli_worker_controls(self):
        argv = ['--no-config', '--fif', self.runs[0]['fif'], '--feature-space', 'source', '--source-inverse', 'inverse.fif',
                '--source-subject', 'sample', '--subjects-dir', '.', '--decoder', 'logistic',
                '--source-workers', '12', '--source-worker-threads', '4']
        options = build_decoder_options(parse_args(argv))
        self.assertEqual(options['source_workers'], 12)
        self.assertEqual(options['source_worker_threads'], 4)
        with self.assertRaises(SystemExit):
            parse_args(argv + ['--source-workers', '0'])
