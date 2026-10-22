import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.run_temporal_window_sweep import (
    collect_results,
    parse_args,
    temporal_windows,
    window_directory,
)


class TemporalWindowSweepTests(unittest.TestCase):
    def test_temporal_windows_include_only_complete_windows(self):
        windows = temporal_windows(-0.2, 0.6, 50, 20)
        self.assertEqual(len(windows), 38)
        self.assertEqual(windows[0], (-0.2, -0.15))
        self.assertEqual(windows[-1], (0.54, 0.59))

    def test_invalid_windows_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            temporal_windows(0, 1, 50, 0)
        with self.assertRaisesRegex(ValueError, "at least one"):
            temporal_windows(0, 0.02, 50, 10)

    def test_slurm_environment_supplies_task_index(self):
        from unittest.mock import patch

        with patch.dict("os.environ", {"SLURM_ARRAY_TASK_ID": "7"}):
            args, remainder = parse_args(["--sweep-output", "/tmp/sweep", "--", "--no-config", "--fif", "x.fif"])
        self.assertEqual(args.task_index, 7)
        self.assertEqual(remainder[-2:], ["--fif", "x.fif"])

    def test_collection_marks_complete_and_pending_windows(self):
        windows = temporal_windows(0, 0.1, 50, 50)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            complete = window_directory(root, 0, *windows[0])
            complete.mkdir(parents=True)
            (complete / "summary.json").write_text(json.dumps({"accuracy": 0.7, "n_trials": 12}))
            output = collect_results(root, windows)
            with output.open(newline="") as stream:
                rows = list(csv.DictReader(stream))
        self.assertEqual(rows[0]["status"], "complete")
        self.assertEqual(rows[0]["accuracy"], "0.7")
        self.assertEqual(rows[1]["status"], "pending")


if __name__ == "__main__":
    unittest.main()
