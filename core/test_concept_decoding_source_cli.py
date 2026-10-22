"""CLI source routing, configuration, validation, and synthetic decoding."""
import contextlib
import io
import json
import os
import shutil
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import mne
import numpy as np
import pandas as pd

from core.concept_decoding_headless_source import HeadlessSourceContext
from scripts.run_concept_decoding_cli import build_decoder_options, main, parse_args, source_run_mappings, resolve_source_parcels


class SourceCLITests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.subjects = self.root / 'subjects'
        (self.subjects / 'sample').mkdir(parents=True)
        self.fif = self.root / 'run_raw.fif'
        self.inverse = self.root / 'run-inv.fif'
        self.inverse.touch()
        self.raw = mne.io.RawArray(
            np.random.default_rng(1).normal(0, 1e-13, (1, 12000)),
            mne.create_info(['MEG001'], 200, 'mag'), verbose='error')
        self.raw.save(self.fif, verbose='error')
        self.trials = pd.DataFrame(dict(source_fif_path=[str(self.fif)] * 20,
                                       onset_sample=np.arange(1000, 11000, 500),
                                       canonical_stimulus=['apple', 'chair'] * 10,
                                       stimulus=['apple', 'chair'] * 10))
        self.table = self.root / 'trials.csv'
        self.trials.to_csv(self.table, index=False)

    def argv(self, geometry='surface'):
        return ['--no-config', '--trial-table', str(self.table), '--output-dir', str(self.root / geometry),
                '--decoder', 'logistic', '--group-mode', 'canonical', '--feature-space', 'source',
                '--source-space-kind', geometry, '--source-inverse', str(self.inverse),
                '--source-subject', 'sample', '--subjects-dir', str(self.subjects),
                '--feature-mode', 'bandpower', '--source-power-bands', 'alpha', 'beta',
                '--feature-binning', '--bin-size-ms', '100', '--window-start', '0.1', '--window-end', '0.3',
                '--bandpass-low-hz', '1', '--bandpass-high-hz', '40', '--no-source-power-baseline',
                '--no-mne-epoch-reject', '--include-trial-outliers', '--n-permutations', '0', '--no-plots']

    def test_parcel_presets_and_hemisphere(self):
        from core.source_parcellation import filter_parcels, is_semantic_parcel, is_visual_parcel
        names = ['ctx-lh-middletemporal', 'ctx-rh-fusiform', 'ctx-lh-cuneus',
                 'Left-Hippocampus', 'Right-Amygdala', 'Brain-Stem', 'L_V1_ROI', 'R_TE1a_ROI']
        for atlas in ['aparc', 'aparc+aseg', 'HCPMMP1', 'HCPMMP1_combined']:
            self.assertEqual(filter_parcels(names, 'semantic', atlas),
                             [n for n in names if is_semantic_parcel(n, atlas)])
            self.assertEqual(filter_parcels(names, 'visual_semantic', atlas),
                             [n for n in names if is_semantic_parcel(n, atlas) or is_visual_parcel(n, atlas)])
        self.assertEqual(filter_parcels(names, 'semantic', 'aparc+aseg', 'left'),
                         ['ctx-lh-middletemporal', 'Left-Hippocampus'])
        self.assertEqual(filter_parcels(names, 'clear'), [])

    def test_resolve_parcel_filter_for_screening(self):
        from types import SimpleNamespace
        args = parse_args(self.argv('volume') + ['--decoder', 'embedding_ridge',
                          '--subset-search', 'exhaustive', '--source-parcel-filter', 'semantic',
                          '--source-hemisphere', 'left'])
        viewer = SimpleNamespace(runs={'run': {'subject': 'sample', 'subjects_dir': str(self.subjects)}})
        names = ['ctx-lh-middletemporal', 'ctx-rh-middletemporal', 'Left-Hippocampus', 'ctx-lh-cuneus']
        with patch('core.concept_decoding_source.concept_volume_labels_for_subject',
                   return_value=(None, None, None, None, names)):
            resolve_source_parcels(args, SimpleNamespace(meg_3d=viewer))
            self.assertEqual(args.source_parcels, ['ctx-lh-middletemporal', 'Left-Hippocampus'])
            self.assertEqual(build_decoder_options(args)['selected_parcellations'], args.source_parcels)
            args.source_parcels = ['ctx-lh-cuneus']
            with self.assertRaisesRegex(ValueError, 'matched no ROIs'):
                resolve_source_parcels(args, SimpleNamespace(meg_3d=viewer))
            args.source_parcels = ['missing']
            with self.assertRaisesRegex(ValueError, 'absent'):
                resolve_source_parcels(args, SimpleNamespace(meg_3d=viewer))

    def inverse_object(self, geometry='surface'):
        entries = [dict(type='surf', vertno=np.array([0, 1]), nuse=2, id=101, subject_his_id='sample'),
                   dict(type='surf', vertno=np.array([], dtype=int), nuse=0, id=102, subject_his_id='sample')]
        if geometry == 'volume':
            entries = [dict(type='vol', vertno=np.array([0, 1]), nuse=2, subject_his_id='sample')]
        return dict(src=mne.SourceSpaces(entries), info=self.raw.info.copy(),
                    noise_cov={'names': ['MEG001']}, source_ori=2)

    def test_parse_options_and_config_overrides(self):
        args = parse_args(self.argv())
        options = build_decoder_options(args)
        self.assertEqual(options['feature_space'], 'source')
        self.assertEqual(options['source_power_bands'], ['alpha', 'beta'])
        self.assertFalse(options['source_power_baseline'])
        self.assertTrue(options['feature_binning_enabled'])
        config = self.root / 'source.toml'
        config.write_text('''trial_table = "trials.csv"
feature_space = "source"
source_inverse = "run-inv.fif"
source_subject = "sample"
subjects_dir = "subjects"
decoder = "logistic"
feature_mode = "bandpower"
source_power_bands = ["alpha", "beta"]
source_power_baseline = true
source_parcel_filter = "semantic"
source_hemisphere = "both"
''')
        args = parse_args(['--config', str(config), '--source-power-bands', 'theta', '--no-source-power-baseline', '--source-hemisphere', 'left'])
        self.assertEqual(args.source_parcel_filter, 'semantic')
        self.assertEqual(args.source_hemisphere, 'left')
        self.assertEqual(args.source_inverse, str(self.inverse))
        self.assertEqual(args.subjects_dir, str(self.subjects))
        self.assertEqual(args.source_power_bands, ['theta'])
        self.assertFalse(args.source_power_baseline)

    def test_source_configuration_cannot_silently_use_signal_or_transformer(self):
        for extra in (['--feature-space', 'signal'], ['--decoder', 'transformer'],
                      ['--source-power-baseline', '--parcel-baseline-end', '0.1']):
            with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(self.argv() + extra)

    def test_manifest_resolves_paths_and_requires_every_analysis_run(self):
        manifest = self.root / 'sources.json'
        manifest.write_text(json.dumps({'runs': [dict(fif='run_raw.fif', inverse='run-inv.fif',
                                                    subject='sample', subjects_dir='subjects')]}))
        argv = self.argv()
        index = argv.index('--source-inverse')
        argv[index:index+2] = ['--source-manifest', str(manifest)]
        args = parse_args(argv)
        rows = source_run_mappings(args, self.trials)
        self.assertEqual(rows[0]['inverse'], str(self.inverse))
        self.assertEqual(rows[0]['subjects_dir'], str(self.subjects))
        changed = self.trials.copy()
        changed.loc[0, 'source_fif_path'] = str(self.root / 'different.fif')
        with self.assertRaisesRegex(ValueError, 'no entry'):
            source_run_mappings(args, changed)
        manifest.write_text(json.dumps({'runs': [rows[0], rows[0]]}))
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            source_run_mappings(args, self.trials)

    def test_single_inverse_rejects_multiple_analysis_runs(self):
        args = parse_args(self.argv())
        changed = self.trials.copy()
        changed.loc[0, 'source_fif_path'] = str(self.root / 'different.fif')
        with self.assertRaisesRegex(ValueError, 'source-manifest'):
            source_run_mappings(args, changed)

    def test_preflight_rejects_wrong_geometry_subject_and_head_transform(self):
        args = parse_args(self.argv())
        rows = source_run_mappings(args, self.trials)
        for problem in ('geometry', 'subject', 'head', 'channels'):
            inv = self.inverse_object('volume' if problem == 'geometry' else 'surface')
            if problem == 'subject':
                inv['src'][0]['subject_his_id'] = 'wrong'
            if problem == 'head':
                inv['info']['dev_head_t'] = mne.transforms.Transform('meg', 'head')
                inv['info']['dev_head_t']['trans'][0, 3] += .02
            if problem == 'channels':
                inv['noise_cov']['names'] = ['MISSING']
            with self.subTest(problem=problem), patch('mne.minimum_norm.read_inverse_operator', return_value=inv):
                with self.assertRaises(ValueError):
                    HeadlessSourceContext(rows, 'surface').validate(build_decoder_options(args))

    def test_surface_and_volume_cli_extract_train_and_export(self):
        # Real FIF loading, epochs, spectral transforms, atlas mean (surface),
        # classifier fit, held-out scoring, and output writing. Only inverse
        # reconstruction and anatomical disk IO are synthetic.
        for geometry in ('surface', 'volume'):
            with self.subTest(geometry=geometry), contextlib.ExitStack() as stack:
                inv = self.inverse_object(geometry)
                stack.enter_context(patch('mne.minimum_norm.read_inverse_operator', return_value=inv))
                stack.enter_context(patch('mne.read_labels_from_annot', return_value=[
                    mne.Label(np.array([0, 1]), hemi='lh', name='test-lh')]))
                if geometry == 'volume':
                    stack.enter_context(patch('core.concept_decoding_source.concept_volume_labels_for_subject',
                        return_value=('atlas.mgz', ['Left-Test'], [0], ['Left-Test'], ['Left-Test'])))
                    stack.enter_context(patch.object(mne.VolSourceEstimate, 'extract_label_time_course',
                        lambda stc, *args, **kwargs: stc.data.mean(axis=0, keepdims=True)))
                def reconstruct(epochs, *args, **kwargs):
                    self.assertEqual(kwargs['pick_ori'], 'vector')
                    outputs = []
                    for event in epochs.events:
                        frequency = 10 if ((event[0] - 1000) // 500) % 2 == 0 else 20
                        signal = np.sin(2 * np.pi * frequency * epochs.times)
                        data = np.zeros((2, 3, len(signal)))
                        data[0, 0], data[1, 0] = signal, -signal
                        cls = mne.VolVectorSourceEstimate if geometry == 'volume' else mne.VectorSourceEstimate
                        vertices = [np.array([0, 1])] if geometry == 'volume' else [np.array([0, 1]), np.array([], dtype=int)]
                        outputs.append(cls(data, vertices, tmin=epochs.tmin, tstep=1/200., subject='sample'))
                    return outputs
                reconstruction = stack.enter_context(patch('mne.minimum_norm.apply_inverse_epochs', side_effect=reconstruct))
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                self.assertEqual(main(self.argv(geometry)), 0)
                self.assertTrue(list((self.root / 'source_feature_cache' / 'bandpower').glob('*.joblib')))
                self.assertTrue(reconstruction.called)
                folder = self.root / geometry
                summary = json.loads((folder / 'summary.json').read_text())
                self.assertEqual(summary['feature_space'], 'source')
                self.assertEqual(summary['feature_mode'], 'bandpower')
                self.assertEqual(summary['source_space_kind'], geometry)
                self.assertEqual(summary['n_bands'], 2)
                self.assertGreater(summary['accuracy'], .9)
                metadata = pd.read_csv(folder / 'feature_importance.csv')
                self.assertEqual(set(metadata.band), {'alpha', 'beta'})
                self.assertTrue((folder / 'run_options.json').is_file())
                # A fresh process context and training-only change reuse the disk cache.
                reconstruction.reset_mock()
                reconstruction.side_effect = AssertionError('Reconstructed on a cache hit')
                self.assertEqual(main(self.argv(geometry) + [
                    '--output-dir', str(self.root / (geometry + '_cached')), '--epochs', '7']), 0)
                reconstruction.assert_not_called()
                # Transfer the saved FIF/inverse/anatomy/cache to a different
                # machine root and create its trial table with destination paths.
                with tempfile.TemporaryDirectory() as destination:
                    dest = Path(destination).resolve()
                    shutil.copyfile(self.fif, dest / self.fif.name)
                    shutil.copyfile(self.inverse, dest / self.inverse.name)
                    shutil.copytree(self.subjects, dest / 'subjects')
                    shutil.copytree(self.root / 'source_feature_cache', dest / 'source_feature_cache')
                    relocated_trials = self.trials.copy()
                    relocated_trials['source_fif_path'] = str(dest / self.fif.name)
                    relocated_trials.to_csv(dest / 'trials.csv', index=False)
                    relocated_argv = [arg.replace(str(self.root), str(dest)) for arg in self.argv(geometry)]
                    self.assertEqual(main(relocated_argv), 0)
                    reconstruction.assert_not_called()

    def test_fresh_subset_cli_routes_and_warns(self):
        argv = self.argv() + ['--decoder', 'embedding_ridge', '--embedding-dict', str(self.root / 'emb.json'),
                              '--subset-search', 'exhaustive', '--source-parcels', 'ctx-lh-test',
                              '--subset-workers', '2']
        batch_folder = self.root / 'batch'
        batch_folder.mkdir()
        (batch_folder / 'manifest.json').write_text('{"status": "complete"}')
        result = {'semantic_subset_output_dir': str(batch_folder)}
        output = io.StringIO()
        with patch('mne.minimum_norm.read_inverse_operator', return_value=self.inverse_object()), \
             patch('mne.read_labels_from_annot', return_value=[mne.Label(np.array([0, 1]), hemi='lh', name='test-lh')]), \
             patch('core.concept_decoding_subsets.run_semantic_subset_batch', return_value=(result, 'done')) as batch, \
             contextlib.redirect_stdout(output):
            self.assertEqual(main(argv), 0)
        options = batch.call_args.args[1]
        self.assertEqual(options['selected_parcellations'], ['ctx-lh-test'])
        self.assertEqual(options['subset_workers'], 2)
        self.assertEqual(options['subset_search_strategy'], 'exhaustive')
        self.assertIn('WARNING: 1 subset models', output.getvalue())

    def test_subset_cli_requires_explicit_pool_and_semantic_decoder(self):
        for extra in (['--subset-search', 'exhaustive'],
                      ['--subset-search', 'exhaustive', '--decoder', 'embedding_ridge'],
                      ['--subset-search', 'backward', '--decoder', 'embedding_ridge',
                       '--source-parcels', 'a', '--subset-workers', '2']):
            with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(self.argv() + extra)

    def test_embedding_ridge_alpha_option(self):
        args = parse_args(self.argv() + ['--decoder', 'embedding_ridge', '--embedding-ridge-alpha', '50.0'])
        self.assertEqual(args.decoder, 'embedding_ridge')
        self.assertEqual(args.embedding_ridge_alpha, 50.0)
        opts = build_decoder_options(args)
        self.assertEqual(opts['embedding_ridge_alpha'], 50.0)

    def test_dry_run_validates_without_reconstruction(self):
        with patch('mne.minimum_norm.read_inverse_operator', return_value=self.inverse_object()), \
             patch('mne.read_labels_from_annot', return_value=[mne.Label(np.array([0, 1]), hemi='lh', name='test-lh')]), \
             patch('mne.minimum_norm.apply_inverse_epochs', side_effect=AssertionError('Reconstructed during dry run')), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(self.argv() + ['--dry-run']), 0)
        self.assertTrue((self.root / 'surface' / 'run_options.json').is_file())
        self.assertFalse((self.root / 'surface' / 'summary.json').exists())


if __name__ == '__main__':
    unittest.main()
