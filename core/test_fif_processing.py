"""Real synthetic FIF regression tests for shared GUI/CLI preparation."""
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import mne
import numpy as np
import pandas as pd

from core.fif_processing import HeadlessFIFProcessor
from scripts.run_concept_decoding_cli import main, parse_args, select_fif_runs


class FIFProcessingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        (self.root / 'raw').mkdir()
        (self.root / 'events_log').mkdir()
        self.fif = self.root / 'raw' / 'subject_s1_run1_raw.fif'
        self.log = self.root / 'events_log' / 'events_s1_b1.csv'
        rng = np.random.default_rng(17)
        data = rng.normal(0, 1e-14, (3, 15000))
        data[2] = 0
        onsets = np.arange(1000, 13000, 500)
        for onset in onsets:
            data[2, onset:onset+2] = 2
            data[2, onset+200:onset+202] = 4
        raw = mne.io.RawArray(data, mne.create_info(['MEG001', 'MEG002', 'STI101'], 1000, ['mag', 'mag', 'stim']), first_samp=700, verbose='error')
        raw.save(self.fif, overwrite=True, verbose='error')
        pd.DataFrame({'trial': np.arange(len(onsets))+1, 'stimulus': ['apple_1.png', 'chair_1.png']*12,
                      'event_code': 2, 'onset_time': onsets/1000, 'session_id': 1, 'block_id': 1}).to_csv(self.log, index=False)
        self.processor = HeadlessFIFProcessor(lambda _: None, ica='off')

    def test_create_reuse_and_gui_share_cache(self):
        fresh, meta = self.processor.prepare(self.fif)
        self.assertEqual(fresh.onset_sample.iloc[0], 1000)  # first_samp removed
        self.assertTrue(self.processor.trial_table_path(self.fif).exists())
        with patch.object(self.processor, '_compute_trial_response_table_uncached', side_effect=AssertionError('Cache missed')):
            reused, _ = self.processor.prepare(self.fif)
        np.testing.assert_allclose(fresh.trial_score, reused.trial_score)
        self.assertEqual(fresh.channel_delta_pairs.iloc[0][0][0], reused.channel_delta_pairs.iloc[0][0][0])
        from core.orchestrator import Orchestrator
        gui = Orchestrator.__new__(Orchestrator)
        gui.c = self.processor.c
        with patch.object(gui, '_compute_trial_response_table_uncached', side_effect=AssertionError('GUI cache missed')):
            table, _, error = gui._compute_trial_response_table_for_fif(str(self.fif))
        self.assertIsNone(error)
        self.assertEqual(len(table), 24)

    def test_changed_event_log_rebuilds(self):
        self.processor.prepare(self.fif)
        df = pd.read_csv(self.log)
        df.loc[0, 'stimulus'] = 'banana_1.png'
        df.to_csv(self.log, index=False)
        table, _ = self.processor.prepare(self.fif)
        self.assertEqual(table.stimulus.iloc[0], 'banana_1.png')

    def test_new_ambiguous_log_rejected(self):
        self.processor.prepare(self.fif)
        shutil.copy2(self.log, self.log.with_name('repeat_s1_b1.csv'))
        with self.assertRaisesRegex(ValueError, 'Multiple event logs'):
            self.processor.prepare(self.fif)

    def test_corrupt_csv_rebuilt(self):
        self.processor.prepare(self.fif)
        self.processor.trial_table_path(self.fif).write_text('invalid')
        table, _ = self.processor.prepare(self.fif)
        self.assertEqual(len(table), 24)

    def test_group_map_changes_refresh_long_lived_processor(self):
        groups = self.root / 'stimulus_with_groups.csv'
        groups.write_text('stimulus,stimulus_group\napple,fruit\nchair,object\n')
        table, _ = self.processor.prepare(self.fif)
        self.assertEqual(table.stimulus_group.iloc[0], 'fruit')
        groups.write_text('stimulus,stimulus_group\napple,food\nchair,object\n')
        table, _ = self.processor.prepare(self.fif)
        self.assertEqual(table.stimulus_group.iloc[0], 'food')

    def test_fif_change_invalidates(self):
        self.processor.prepare(self.fif)
        import os
        stat = self.fif.stat()
        os.utime(self.fif, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1000000000))
        with patch.object(self.processor, '_compute_trial_response_table_uncached', wraps=self.processor._compute_trial_response_table_uncached) as build:
            self.processor.prepare(self.fif)
        build.assert_called_once()

    def test_cache_moves_and_rebases_paths(self):
        self.processor.prepare(self.fif)
        destination = self.root / 'moved'
        shutil.copytree(self.root / 'raw', destination / 'raw')
        shutil.copytree(self.root / 'events_log', destination / 'events_log')
        relocated = destination / 'raw' / self.fif.name
        with patch.object(self.processor, '_compute_trial_response_table_uncached', side_effect=AssertionError('Cache missed after move')):
            table, meta = self.processor.prepare(relocated)
        self.assertEqual(table.source_fif_path.iloc[0], str(relocated))

    def test_cli_prepare_and_real_logistic_training(self):
        common = ['--no-config', '--fif', str(self.fif), '--ica', 'off']
        self.assertEqual(main(common + ['--prepare-only']), 0)
        out = self.root / 'results'
        self.assertEqual(main(common + ['--decoder', 'logistic', '--group-mode', 'canonical',
                                       '--output-dir', str(out), '--n-permutations', '0', '--no-plots',
                                       '--bandpass-low-hz', '0', '--bandpass-high-hz', '0']), 0)
        self.assertTrue((out / 'summary.json').exists())
        self.assertTrue((out / 'input_trial_table.csv').exists())

    def test_config_defaults_cli_overrides_and_validation(self):
        config = self.root / 'config.toml'
        config.write_text('fif = "raw/subject_s1_run1_raw.fif"\ndecoder = "template"\ngroup_mode = "canonical"\n')
        args = parse_args(['--config', str(config)])
        self.assertEqual(args.fif, [str(self.fif)])
        self.assertEqual(args.decoder, 'template')
        args = parse_args(['--config', str(config), '--decoder', 'logistic', '--trial-table', str(self.root/'input.csv')])
        self.assertIsNone(args.fif)
        self.assertEqual(args.decoder, 'logistic')
        config.write_text('fif = "x.fif"\ndecoder = "typo"\n')
        with self.assertRaises(SystemExit):
            parse_args(['--config', str(config)])

    def test_auto_ica_creates_and_reuses_derivative(self):
        folder = self.root / 'tsss'
        folder.mkdir()
        source = folder / self.fif.name
        shutil.copy2(self.fif, source)
        processor = HeadlessFIFProcessor(lambda _: None)
        table, meta = processor.prepare(source)
        cleaned = self.root / 'ica' / self.fif.name
        self.assertTrue(cleaned.is_file())
        self.assertTrue(processor.trial_table_path(cleaned).is_file())
        self.assertEqual(table.source_fif_path.iloc[0], str(cleaned))
        with patch.object(processor, '_run_ica_cleaning_for_tsss_fif', side_effect=AssertionError('ICA recomputed')):
            processor.prepare(source)

    def test_cli_import_does_not_load_qt(self):
        import subprocess
        result = subprocess.run([sys.executable, '-c',
            'import scripts.run_concept_decoding_cli; import core.fif_processing; import sys; '
            'assert not any(x.startswith(("qtpy", "PyQt", "PySide", "pyvista")) for x in sys.modules)'],
            capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_config_enabled_flags_can_be_disabled(self):
        args = parse_args(['--fif', str(self.fif), '--no-downsample', '--feature-binning'])
        self.assertFalse(args.downsample)
        self.assertFalse(args.no_feature_binning)
        args = parse_args(['--fif', str(self.fif)])
        self.assertTrue(args.downsample)
        self.assertTrue(args.no_feature_binning)

    def test_multiple_fifs_dry_run(self):
        other = self.fif.with_name('second_s1_run1_raw.fif')
        shutil.copy2(self.fif, other)
        out = self.root / 'multi'
        self.assertEqual(main(['--fif', str(self.fif), str(other), '--ica', 'off',
                               '--dry-run', '--output-dir', str(out)]), 0)
        df = pd.read_csv(out / 'input_trial_table.csv')
        self.assertEqual(len(df), 48)
        self.assertEqual(df.source_fif_path.nunique(), 2)
        self.assertTrue(self.processor.trial_table_path(other).is_file())

    def test_missing_logs_fail_without_writing_cache(self):
        self.log.unlink()
        with self.assertRaisesRegex(ValueError, 'No events_log onset rows'):
            self.processor.prepare(self.fif)
        self.assertFalse(self.processor.trial_table_path(self.fif).exists())

    def _neighbor(self):
        other = self.fif.with_name('second_s1_run1_raw.fif')
        shutil.copy2(self.fif, other)
        return other

    def test_interactive_sibling_prompt_accept_and_decline(self):
        other = self._neighbor()
        args = parse_args(['--fif', str(self.fif)])
        with patch.dict('os.environ', {}, clear=True), patch('sys.stdin.isatty', return_value=True):
            with patch('builtins.input', return_value='yes') as prompt:
                self.assertEqual(select_fif_runs(args), [str(self.fif), str(other)])
                prompt.assert_called_once()
            with patch('builtins.input', return_value='n'):
                self.assertEqual(select_fif_runs(args), [str(self.fif)])
            with patch('builtins.input', side_effect=EOFError):
                self.assertEqual(select_fif_runs(args), [str(self.fif)])

    def test_noninteractive_and_explicit_list_do_not_prompt(self):
        other = self._neighbor()
        with patch('sys.stdin.isatty', return_value=False), patch('builtins.input', side_effect=AssertionError('Prompted')):
            self.assertEqual(select_fif_runs(parse_args(['--fif', str(self.fif)])), [str(self.fif)])
        with patch('sys.stdin.isatty', return_value=True), patch('builtins.input', side_effect=AssertionError('Prompted')):
            self.assertEqual(len(select_fif_runs(parse_args(['--fif', str(self.fif), str(other)]))), 2)
            self.assertEqual(select_fif_runs(parse_args(['--fif', str(self.fif), '--no-all-runs'])), [str(self.fif)])
        with patch.dict('os.environ', {'SLURM_JOB_ID': '123'}), patch('sys.stdin.isatty', return_value=True), patch('builtins.input', side_effect=AssertionError('Prompted')):
            self.assertEqual(select_fif_runs(parse_args(['--fif', str(self.fif)])), [str(self.fif)])

    def test_all_runs_preparation_combines_without_duplicate_appends(self):
        other = self._neighbor()
        # Invalid/auxiliary FIF files are not experimental runs.
        (self.fif.parent / 'model-ica.fif').write_text('not raw')
        command = ['--fif', str(self.fif), '--all-runs', '--ica', 'off', '--prepare-only']
        with patch('builtins.input', side_effect=AssertionError('Prompted')):
            self.assertEqual(main(command), 0)
            self.assertEqual(main(command), 0)
        combined = list(self.fif.parent.glob('*.combined-*.trial_table.csv'))
        self.assertEqual(len(combined), 1)
        df = pd.read_csv(combined[0])
        self.assertEqual(len(df), 48)
        self.assertEqual(set(df.source_fif_path), {str(self.fif), str(other)})
        self.assertTrue(df.combined_trial_index.is_unique)
        self.assertEqual(df.groupby('source_fif_path').size().tolist(), [24, 24])
        # Single-run caches are still single-run and reusable independently.
        self.assertEqual(len(pd.read_csv(self.processor.trial_table_path(self.fif))), 24)

    def test_split_parts_are_not_selected_as_additional_runs(self):
        from types import SimpleNamespace
        from unittest.mock import MagicMock
        part = self.fif.with_name('subject_s1_run1_raw-1.fif')
        part.touch()
        other = self._neighbor()
        def read(path, **kwargs):
            raw = MagicMock()
            raw.__enter__.return_value = SimpleNamespace(filenames=[str(self.fif), str(part)] if str(path) == str(self.fif) else [str(path)])
            return raw
        with patch('mne.io.read_raw_fif', side_effect=read):
            result = select_fif_runs(parse_args(['--fif', str(self.fif), str(part), '--all-runs']))
        self.assertEqual(result, [str(self.fif), str(other)])

    def test_all_runs_config_and_override(self):
        config = self.root / 'runs.toml'
        config.write_text('fif = "raw/subject_s1_run1_raw.fif"\nall_runs = true\n')
        self.assertTrue(parse_args(['--config', str(config)]).all_runs)
        self.assertFalse(parse_args(['--config', str(config), '--no-all-runs']).all_runs)

    def test_folder_processes_all_runs_without_prompt(self):
        other = self._neighbor()
        (self.fif.parent / 'solution-ica.fif').write_text('not raw')
        nested = self.fif.parent / 'nested'
        nested.mkdir()
        shutil.copy2(self.fif, nested / self.fif.name)
        out = self.root / 'folder_results'
        with patch('sys.stdin.isatty', return_value=True), patch('builtins.input', side_effect=AssertionError('Prompted')):
            self.assertEqual(main(['--fif_folder', str(self.fif.parent), '--ica', 'off',
                                   '--dry-run', '--output-dir', str(out)]), 0)
        df = pd.read_csv(out / 'input_trial_table.csv')
        self.assertEqual(len(df), 48)
        self.assertEqual(set(df.source_fif_path), {str(self.fif), str(other)})
        self.assertEqual(len(list(self.fif.parent.glob('*.combined-*.trial_table.csv'))), 1)

    def test_folder_config_alias_and_input_overrides(self):
        config = self.root / 'folder.toml'
        config.write_text('fif_folder = "raw"\n')
        args = parse_args(['--config', str(config)])
        self.assertEqual(args.fif_folder, str(self.fif.parent))
        self.assertEqual(parse_args(['--fif-folder', str(self.fif.parent)]).fif_folder, str(self.fif.parent))
        args = parse_args(['--config', str(config), '--fif', str(self.fif)])
        self.assertIsNone(args.fif_folder)
        config.write_text('fif = "raw/subject_s1_run1_raw.fif"\n')
        args = parse_args(['--config', str(config), '--fif_folder', str(self.fif.parent)])
        self.assertIsNone(args.fif)
        self.assertEqual(args.fif_folder, str(self.fif.parent))

    def test_empty_missing_and_nonraw_folders_fail_clearly(self):
        folder = self.root / 'empty'
        folder.mkdir()
        args = parse_args(['--fif_folder', str(folder)])
        with self.assertRaisesRegex(ValueError, 'No readable raw FIF'):
            select_fif_runs(args)
        (folder / 'model.fif').write_text('not raw')
        with self.assertRaisesRegex(ValueError, 'No readable raw FIF'):
            select_fif_runs(args)
        args.fif_folder = str(self.root / 'missing')
        with self.assertRaises(NotADirectoryError):
            select_fif_runs(args)

    def test_rebuild_switch(self):
        self.processor.prepare(self.fif)
        self.processor.rebuild_trial_table = True
        with patch.object(self.processor, '_compute_trial_response_table_uncached', wraps=self.processor._compute_trial_response_table_uncached) as build:
            self.processor.prepare(self.fif)
        build.assert_called_once()


if __name__ == '__main__':
    unittest.main()
