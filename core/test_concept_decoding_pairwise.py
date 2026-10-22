import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from core.concept_decoding_pairwise import compute_pairwise_decoding, supertrial_indices


class PairwiseTests(unittest.TestCase):
    def payload(self):
        rng = np.random.default_rng(10)
        y = np.repeat(["a", "b", "c"], 18)
        data = rng.normal(0, .1, (54, 2, 3))
        data[:, :, 1] += np.repeat([-5, 0, 5], 18)[:, None]
        meta = pd.DataFrame([dict(channel=ch, stat="raw", relative_time_s=t)
                             for ch in ["x", "y"] for t in [-.1, .1, .2]])
        return dict(X=data.reshape(54, -1), y=y, feature_meta_df=meta,
                    trial_df=pd.DataFrame({"concept_label": y, "trial_index": np.arange(len(y)) + 100}))

    def test_disjoint_groups(self):
        groups = supertrial_indices(np.arange(19), 6, 3, np.random.default_rng(4))
        self.assertEqual(groups.shape, (3, 6))
        self.assertEqual(len(np.unique(groups)), 18)

    def test_signal_time_symmetry_reproducibility_and_splits(self):
        options = dict(pairwise_repeats=2)
        result, message = compute_pairwise_decoding(self.payload(), options)
        self.assertIsNotNone(result, message)
        matrix = result["pairwise_accuracy_matrix"]
        np.testing.assert_equal(matrix, matrix.transpose(0, 2, 1))
        self.assertTrue(np.isnan(np.diagonal(matrix, axis1=1, axis2=2)).all())
        self.assertEqual(result["timecourse_df"].iloc[1].accuracy, 1.)
        self.assertLess(result["timecourse_df"].iloc[0].accuracy, .8)
        for _, group in result["split_df"].groupby(["concept_a", "concept_b", "repeat", "concept"]):
            ids = [int(i) for row in group.trial_indices for i in row.split(";")]
            self.assertEqual(len(ids), len(set(ids)))
        again, _ = compute_pairwise_decoding(self.payload(), options)
        np.testing.assert_equal(matrix, again["pairwise_accuracy_matrix"])

    def test_reject_unsupported_and_insufficient(self):
        for options in [dict(pairwise_supertrial_size=10), dict(pairwise_repeats=0), dict(split_mode="run")]:
            result, error = compute_pairwise_decoding(self.payload(), options)
            self.assertIsNone(result)
            self.assertTrue(error)
        payload = self.payload()
        payload["feature_meta_df"]["stat"] = "mean"
        self.assertIsNone(compute_pairwise_decoding(payload)[0])

    def test_dispatch_and_export(self):
        from core.concept_decoding_signal import compute_concept_decoding_from_payload
        from scripts.run_concept_decoding_cli import write_outputs, write_plots, parse_args, build_decoder_options
        args = parse_args(["--trial-table", "unused.csv", "--output-dir", "unused",
                           "--decoder", "pairwise_svm", "--pairwise-repeats", "1"])
        result, msg = compute_concept_decoding_from_payload(None, build_decoder_options(args), self.payload(), "signal")
        self.assertIsNotNone(result, msg)
        with tempfile.TemporaryDirectory() as tmp:
            outputs = write_outputs(result, Path(tmp))
            plots = write_plots(result, Path(tmp))
            self.assertTrue(Path(outputs["pairwise_accuracy_npz"]).exists())
            self.assertTrue(Path(plots["pairwise_timecourse_png"]).exists())
            self.assertEqual(np.load(outputs["pairwise_accuracy_npz"])["accuracy"].shape, (3, 3, 3))


if __name__ == "__main__":
    unittest.main()
