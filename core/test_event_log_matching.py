import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from core.orchestrator import Orchestrator


class _RawStub:
    info = {"sfreq": 1000.0}


def _write_log(path, session, block, onset_times):
    rows = []
    for trial, onset in enumerate(onset_times, start=1):
        rows.append(
            {
                "trial": trial,
                "stimulus": f"stim_{trial}.png",
                "event_code": 2,
                "onset_time": onset,
                "session_id": session,
                "block_id": block,
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


class EventLogMatchingTests(unittest.TestCase):
    def _paths(self, root, fif_name):
        subject_dir = Path(root) / "subject"
        fif_dir = subject_dir / "tsss"
        log_dir = subject_dir / "events_log"
        fif_dir.mkdir(parents=True)
        log_dir.mkdir()
        fif_path = fif_dir / fif_name
        fif_path.touch()
        return fif_path, log_dir

    def test_event_log_match_requires_same_session_and_run(self):
        with tempfile.TemporaryDirectory() as root:
            fif_path, log_dir = self._paths(root, "subject_s2_run3_raw.fif")
            _write_log(log_dir / "events_s1_b3.csv", 1, 3, [0.0, 1.0, 2.0])
            _write_log(log_dir / "events_s2_b3.csv", 2, 3, [0.0, 1.0, 2.0])

            orchestrator = Orchestrator.__new__(Orchestrator)
            rows, meta = orchestrator._load_event_log_onsets_for_fif(str(fif_path))

            self.assertEqual(len(rows), 3)
            self.assertEqual(int(rows["session_id"].iloc[0]), 2)
            self.assertEqual(os.path.basename(meta["event_log_csv"]), "events_s2_b3.csv")

    def test_duplicate_session_run_uses_fif_trigger_timing(self):
        with tempfile.TemporaryDirectory() as root:
            fif_path, log_dir = self._paths(root, "subject_s2_run3_repeat_raw.fif")
            _write_log(log_dir / "events_s2_b3_first.csv", 2, 3, [0.0, 1.4, 2.0, 3.5])
            _write_log(log_dir / "events_s2_b3_repeat.csv", 2, 3, [0.0, 1.0, 2.1, 3.0])
            onset_events = np.asarray([[0, 0, 2], [1000, 0, 2], [2100, 0, 2], [3000, 0, 2]])

            orchestrator = Orchestrator.__new__(Orchestrator)
            rows, meta = orchestrator._load_event_log_onsets_for_fif(
                str(fif_path),
                onset_events=onset_events,
                raw=_RawStub(),
            )

            self.assertEqual(len(rows), 4)
            self.assertEqual(os.path.basename(meta["event_log_csv"]), "events_s2_b3_repeat.csv")

    def test_ambiguous_duplicate_logs_are_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            fif_path, log_dir = self._paths(root, "subject_s2_run3_raw.fif")
            _write_log(log_dir / "events_s2_b3_a.csv", 2, 3, [0.0, 1.0, 2.0])
            _write_log(log_dir / "events_s2_b3_b.csv", 2, 3, [0.0, 1.0, 2.0])

            orchestrator = Orchestrator.__new__(Orchestrator)
            rows, meta = orchestrator._load_event_log_onsets_for_fif(str(fif_path))

            self.assertTrue(rows.empty)
            self.assertIn("Multiple event logs", meta["event_log_match_error"])

    def test_onset_count_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            fif_path, log_dir = self._paths(root, "subject_s2_run3_raw.fif")
            _write_log(log_dir / "events_s2_b3.csv", 2, 3, [0.0, 1.0, 2.0])
            onset_events = np.asarray([[0, 0, 2], [1000, 0, 2]])

            orchestrator = Orchestrator.__new__(Orchestrator)
            rows, meta = orchestrator._load_event_log_onsets_for_fif(
                str(fif_path),
                onset_events=onset_events,
                raw=_RawStub(),
            )

            self.assertTrue(rows.empty)
            self.assertIn("onset counts differ", meta["event_log_match_error"])


if __name__ == "__main__":
    unittest.main()
