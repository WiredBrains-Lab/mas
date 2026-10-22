"""Persistent all-ROI cache identity, selection, and failure recovery."""
import ast
import inspect
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import joblib
import mne
import numpy as np
import pandas as pd

from core.concept_decoding_feature_cache import (
    FEATURE_OPTIONS, all_roi_extraction_options, cached_source_features, feature_cache_identity,
    source_feature_cache_directory,
)
from core.concept_decoding_headless_source import HeadlessSourceContext
from scripts.run_concept_decoding_cli import HeadlessConceptDecodingContext


class SourceFeatureCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.fif = self.root / 'run_raw.fif'
        raw = mne.io.RawArray(np.ones((1, 100)), mne.create_info(['MEG001'], 100, 'mag'), verbose='error')
        raw.save(self.fif, verbose='error')
        self.inverse = self.root / 'run-inv.fif'
        self.inverse.write_bytes(b'inverse identity')
        self.anatomy = self.root / 'sample' / 'mri' / 'aparc+aseg.mgz'
        self.anatomy.parent.mkdir(parents=True)
        self.anatomy.write_bytes(b'atlas identity')
        table = pd.DataFrame(dict(source_fif_path=[str(self.fif)] * 2,
                                  onset_sample=[10, 20], concept=['a', 'b']))
        self.context = HeadlessConceptDecodingContext(table, first_fif_path=str(self.fif))
        HeadlessSourceContext([dict(fif=str(self.fif), inverse=str(self.inverse),
                                   subject='sample', subjects_dir=str(self.root))], 'surface').attach(self.context)
        self.options = dict(source_feature_cache_dir=str(self.root / 'cache'),
                            selected_parcellations=['left'], feature_mode='bandpower',
                            source_power_bands=['alpha'], source_space_kind='surface')
        self.payload = dict(X=np.array([[1., 2., 3., 4.], [5., 6., 7., 8.]]),
                            trial_df=table, y=np.array(['a', 'b']), roi_names=['left', 'right'],
                            feature_meta_df=pd.DataFrame({'roi': ['left', 'right'] * 2,
                                                          'feature': ['l0', 'r0', 'l1', 'r1']}))
        self.extract = Mock(return_value=(self.payload, None))
        self.prepare = patch('core.concept_decoding_feature_cache.all_roi_extraction_options',
                             side_effect=lambda context, opts: (dict(opts, selected_parcellations=['left', 'right'], max_rois=0), ['empty']))

    def test_disk_cache_reuses_other_roi_and_training_settings(self):
        with self.prepare:
            first, _ = cached_source_features(self.context, self.options, self.extract)
            second, _ = cached_source_features(self.context, dict(self.options,
                selected_parcellations=['right'], decoder_type='embedding_contrastive', epochs=99,
                split_mode='zero_shot', embedding_dict_path='different.json'), self.extract)
        self.extract.assert_called_once()
        self.assertEqual(self.extract.call_args.args[1]['selected_parcellations'], ['left', 'right'])
        np.testing.assert_array_equal(first['X'], self.payload['X'][:, [0, 2]])
        np.testing.assert_array_equal(second['X'], self.payload['X'][:, [1, 3]])
        self.assertTrue(second['source_feature_cache_hit'])
        saved = joblib.load(second['source_feature_cache_path'])
        self.assertEqual(saved['payload']['roi_names'], ['left', 'right'])
        self.assertEqual(saved['payload']['source_feature_cache_excluded_rois'], ['empty'])

    def test_automatic_data_local_directory_on_laptop_and_hpc(self):
        for base in ('/Users/me/data', '/scratch/g/bgross/CIDMEG_SM/data'):
            subject = Path(base) / 'CIDMEG005'
            runs = [dict(fif=str(subject / 'ICA' / f'run{i}_raw.fif'),
                         subject='Anatomy', subjects_dir=str(subject)) for i in range(2)]
            for mode in ('bandpower', 'raw', 'mean_rms'):
                opts = dict(feature_mode=mode, _source_feature_cache_runs=runs)
                self.assertEqual(source_feature_cache_directory(self.context, opts),
                                 subject / 'source_feature_cache' / mode)
        # Shared FreeSurfer subjects_dir must not move the cache out of the data subject.
        runs = [dict(fif='/data/CIDMEG005/ICA/run_raw.fif', subject='CIDMEG005', subjects_dir='/data')]
        self.assertEqual(source_feature_cache_directory(self.context,
            dict(feature_mode='bandpower', _source_feature_cache_runs=runs)),
            Path('/data/CIDMEG005/source_feature_cache/bandpower'))
        self.assertEqual(source_feature_cache_directory(self.context, self.options), self.root / 'cache')

    def test_gui_context_caches_and_unsaved_raw_edits_invalidate(self):
        self.context.meg_3d = SimpleNamespace(_get_brain_subject_info=lambda: ('sample', str(self.root)))
        self.context._fif_raw = mne.io.read_raw_fif(self.fif, preload=True, verbose='error')
        opts = dict(self.options, source_feature_cache_dir=None)
        with patch('core.concept_decoding_source.configure_source_subject_for_fif',
                   return_value=(('sample', str(self.root)), None)), \
             patch('core.concept_decoding_source._find_or_create_source_inverse',
                   return_value=(str(self.inverse), {}, None)), self.prepare:
            first, _ = cached_source_features(self.context, opts, self.extract)
            second, _ = cached_source_features(self.context, opts, self.extract)
            self.assertTrue(second['source_feature_cache_hit'])
            self.assertEqual(Path(first['source_feature_cache_path']).parent,
                             self.root / 'source_feature_cache' / 'bandpower')
            self.context._fif_raw._data[0, 0] += 1
            changed, _ = cached_source_features(self.context, opts, self.extract)
            self.assertNotEqual(first['source_feature_cache_path'], changed['source_feature_cache_path'])
            before = feature_cache_identity(self.context, opts)[0]
            self.context._fif_raw.info['bads'] = ['MEG001']
            self.assertNotEqual(before, feature_cache_identity(self.context, opts)[0])
            before = feature_cache_identity(self.context, opts)[0]
            self.context._fif_raw.set_annotations(mne.Annotations([.1], [.1], ['BAD_edit']))
            self.assertNotEqual(before, feature_cache_identity(self.context, opts)[0])
        self.assertEqual(self.extract.call_count, 2)
        self.assertFalse(self.extract.call_args.args[1]['enable_signal_epoch_cache'])

    def test_default_wrapper_cache_and_explicit_disable(self):
        from core.concept_decoding_source import build_source_space_concept_features
        opts = dict(self.options, source_feature_cache_dir=None)
        with self.prepare, patch('core.concept_decoding_source._extract_source_space_concept_features', self.extract):
            build_source_space_concept_features(self.context, opts)
            result, _ = build_source_space_concept_features(self.context, opts)
            self.assertTrue(result['source_feature_cache_hit'])
            build_source_space_concept_features(self.context, dict(opts, source_feature_cache=False))
        self.assertEqual(self.extract.call_count, 2)

    def test_settings_inputs_and_atlas_invalidate_but_selection_does_not(self):
        key = feature_cache_identity(self.context, self.options)[0]
        for field, value in dict(source_power_bands=['beta'], bandpass_low_hz=2,
                                 resample_sfreq=150, window_start=.1, bin_size_ms=20,
                                 source_power_baseline=False, lambda2=.5, trial_outlier_z=8).items():
            with self.subTest(field=field):
                self.assertNotEqual(key, feature_cache_identity(self.context, dict(self.options, **{field: value}))[0])
        self.assertEqual(key, feature_cache_identity(self.context, dict(self.options,
            selected_parcellations=['right'], max_rois=1, subset_workers=4))[0])
        for path in (self.inverse, self.anatomy):
            previous = feature_cache_identity(self.context, self.options)[0]
            path.write_bytes(path.read_bytes() + b' changed')
            self.assertNotEqual(previous, feature_cache_identity(self.context, self.options)[0])
        previous = feature_cache_identity(self.context, self.options)[0]
        self.context._fif_trial_response_table.loc[0, 'onset_sample'] = 15
        self.assertNotEqual(previous, feature_cache_identity(self.context, self.options)[0])

    def test_corrupt_cache_rebuild_and_force_refresh(self):
        with self.prepare:
            first, _ = cached_source_features(self.context, self.options, self.extract)
            Path(first['source_feature_cache_path']).write_bytes(b'incomplete')
            cached_source_features(self.context, self.options, self.extract)
            cached_source_features(self.context, dict(self.options, rebuild_source_features=True), self.extract)
        self.assertEqual(self.extract.call_count, 3)
        self.assertFalse(list((self.root / 'cache').glob('*.tmp')))

    def test_failed_or_partial_extraction_is_not_cached(self):
        for result in ((None, 'failed'), (dict(self.payload, skipped_paths=[('run', 'error')]), None)):
            with self.subTest(result=result[1]), self.prepare:
                self.extract.return_value = result
                cached_source_features(self.context, self.options, self.extract)
                self.assertFalse(list((self.root / 'cache').glob('*.joblib')))

    def test_unsupported_requested_roi_errors_without_refitting(self):
        with self.prepare:
            cached_source_features(self.context, self.options, self.extract)
            result, error = cached_source_features(self.context,
                dict(self.options, selected_parcellations=['empty']), self.extract)
        self.assertIsNone(result)
        self.assertIn('empty', error)
        self.extract.assert_called_once()

    def test_empty_surface_roi_is_excluded_by_source_support(self):
        src = mne.SourceSpaces([
            dict(type='surf', vertno=np.array([0, 1]), nuse=2, id=101, subject_his_id='sample'),
            dict(type='surf', vertno=np.array([], dtype=int), nuse=0, id=102, subject_his_id='sample')])
        labels = [mne.Label(np.array([0]), hemi='lh', name='left'),
                  mne.Label(np.array([8]), hemi='lh', name='empty')]
        with patch('mne.minimum_norm.read_inverse_operator', return_value={'src': src}), \
             patch('core.concept_decoding_source.concept_source_labels_for_subject',
                   return_value=(labels, ['left', 'empty'], ['left', 'empty'], 'aparc')):
            options, excluded = all_roi_extraction_options(self.context, self.options)
        self.assertEqual(options['selected_parcellations'], ['left'])
        self.assertEqual(excluded, ['empty'])

    def test_direct_extraction_options_are_accounted_for(self):
        from core.concept_decoding_source import _extract_source_space_concept_features
        tree = ast.parse(inspect.getsource(_extract_source_space_concept_features))
        keys = {node.args[0].value for node in ast.walk(tree)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name) and node.func.value.id == 'options'
                and node.func.attr == 'get' and node.args and isinstance(node.args[0], ast.Constant)}
        self.assertFalse(keys - set(FEATURE_OPTIONS) - {'selected_parcellations', 'max_rois'})

    def test_copied_gui_cache_loads_in_relocated_cli_and_remaps_paths(self):
        # GUI preload alone must not make otherwise identical saved data distinct.
        disk_key = feature_cache_identity(self.context, self.options)[0]
        self.context._fif_raw = mne.io.read_raw_fif(self.fif, preload=True, verbose='error')
        self.assertEqual(disk_key, feature_cache_identity(self.context, self.options)[0])
        self.payload.update(
            inverse_paths={str(self.fif): str(self.inverse)},
            source_subject_by_path={str(self.fif): dict(subject='sample', subjects_dir=str(self.root))},
            subject='sample', subjects_dir=str(self.root), volume_aseg_path=str(self.anatomy),
            trial_qc_df=self.payload['trial_df'].copy())
        with self.prepare:
            original, _ = cached_source_features(self.context, self.options, self.extract)
        with tempfile.TemporaryDirectory() as destination:
            dest = Path(destination).resolve()
            fif, inverse = dest / 'recordings' / 'run_raw.fif', dest / 'inverses' / 'run-inv.fif'
            subjects = dest / 'freesurfer'
            anatomy = subjects / 'sample' / 'mri' / 'aparc+aseg.mgz'
            for source, target in ((self.fif, fif), (self.inverse, inverse), (self.anatomy, anatomy)):
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
                os.utime(target, (1234567890, 1234567890))
            cache = dest / 'copied_cache'
            shutil.copytree(self.root / 'cache', cache)
            table = self.context._fif_trial_response_table.copy()
            table['source_fif_path'] = str(fif)
            context = HeadlessConceptDecodingContext(table, first_fif_path=str(fif))
            HeadlessSourceContext([dict(fif=str(fif), inverse=str(inverse), subject='sample',
                                       subjects_dir=str(subjects))], 'surface').attach(context)
            opts = dict(self.options, source_feature_cache_dir=str(cache), selected_parcellations=['right'])
            no_extract = Mock(side_effect=AssertionError('Copied features were recomputed'))
            with patch('core.concept_decoding_feature_cache.platform.python_version', return_value='3.11.99'), \
                 patch('core.concept_decoding_feature_cache.scipy.__version__', 'different-producer-version'):
                self.assertEqual(disk_key, feature_cache_identity(context, opts)[0])
                result, error = cached_source_features(context, opts, no_extract)
            self.assertIsNone(error)
            self.assertTrue(result['source_feature_cache_hit'])
            self.assertEqual(Path(original['source_feature_cache_path']).name,
                             Path(result['source_feature_cache_path']).name)
            self.assertEqual(result['trial_df']['source_fif_path'].tolist(), [str(fif)] * 2)
            self.assertEqual(result['trial_qc_df']['source_fif_path'].tolist(), [str(fif)] * 2)
            self.assertEqual(result['inverse_paths'], {str(fif): str(inverse)})
            self.assertEqual(result['subjects_dir'], str(subjects))
            self.assertEqual(result['volume_aseg_path'], str(anatomy))
            self.assertEqual(result['source_subject_by_path'][str(fif)]['subjects_dir'], str(subjects))
            np.testing.assert_array_equal(result['X'], self.payload['X'][:, [1, 3]])
            no_extract.assert_not_called()
            # Same filenames and file sizes cannot disguise changed input bytes.
            original_stat = inverse.stat()
            inverse.write_bytes(b'Inverse identity')
            os.utime(inverse, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
            self.assertNotEqual(disk_key, feature_cache_identity(context, opts)[0])

    def test_timestamp_changes_keep_identity_but_content_changes_do_not(self):
        original = feature_cache_identity(self.context, self.options)[0]
        (self.anatomy.parent / '.DS_Store').write_bytes(b'OS metadata')
        self.assertEqual(original, feature_cache_identity(self.context, self.options)[0])
        for path in (self.fif, self.inverse, self.anatomy):
            os.utime(path, (1234567890, 1234567890))
        self.assertEqual(original, feature_cache_identity(self.context, self.options)[0])
        self.anatomy.write_bytes(b'Atlas identity')  # Same length as the original.
        self.assertNotEqual(original, feature_cache_identity(self.context, self.options)[0])

    def test_gui_and_cli_default_spellings_match(self):
        original = feature_cache_identity(self.context, self.options)[0]
        equivalent = dict(self.options, source_power_bands='alpha', max_source_features=75000,
                          source_batch_size=4, source_parcel_baseline_zscore=False,
                          baseline_correct=False, min_trials_for_outlier_qc=8,
                          source_common_space='native', bin_size_ms=50,
                          parcel_baseline_start=-.2, parcel_baseline_end=0)
        self.assertEqual(original, feature_cache_identity(self.context, equivalent)[0])


if __name__ == '__main__':
    unittest.main()
