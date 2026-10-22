import unittest
from core.concept_decoding_run_summary import log_decoding_run_summary


class RunSummaryTests(unittest.TestCase):
    def test_spaces_and_modes(self):
        for space, geometry, mode in [('signal', 'surface', 'raw'),
                                      ('source', 'surface', 'mean'),
                                      ('source', 'volume', 'bandpower')]:
            messages = []
            log_decoding_run_summary(dict(feature_space=space, source_space_kind=geometry,
                feature_mode=mode, source_power_bands=['beta'], source_power_baseline=True,
                window_start=.15, window_end=.45, feature_binning_enabled=True, bin_size_ms=50,
                split_mode='zero_shot', selected_parcellations=['a', 'b']), messages.append, saved=True)
            text = '\n'.join(messages)
            self.assertIn('saved checkpoint settings', text)
            self.assertIn(f'{space} space', text)
            self.assertIn(f'feature mode={mode}', text)
            self.assertIn('split=zero_shot', text)
            self.assertIn('150 to 450 ms; 50 ms bins', text)
            if space == 'source':
                self.assertIn(f'({geometry})', text)
                self.assertIn('2 selected parcels', text)
            if mode == 'bandpower':
                self.assertIn('atlas=aparc+aseg', text)
                self.assertIn('requested bands=beta; normalization=baseline dB', text)
            else:
                self.assertNotIn('Source band power:', text)
