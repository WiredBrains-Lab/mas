import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from core.instruction_embeddings import read_instruction_csv, score_instructions


class InstructionEmbeddingTests(unittest.TestCase):
    def test_ridge_decoder_reports_instruction_matches_end_to_end(self):
        from core.test_concept_embedding_decoders import _synthetic_embedding_payload
        from core.concept_decoding_transformer import compute_transformer_decoding_from_payload

        payload = _synthetic_embedding_payload()
        payload["trial_df"]["block_id"] = np.repeat([1, 2, 3], 12)
        targets = {"status": "embedded", "names": ["VEHICLE", "ANIMAL", "MUSIC"],
                   "embeddings": np.eye(3).tolist(),
                   "block_mapping": {"1": "VEHICLE", "2": "ANIMAL", "3": "MUSIC"}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "embeddings.json"
            path.write_text(json.dumps({"metadata": {"instruction_targets": targets}}))
            payload["embedding_dict_path"] = str(path)
            result, error = compute_transformer_decoding_from_payload(None, {
                "decoder_type": "embedding_ridge", "test_fraction": 0.25,
                "n_permutations": 0, "transformer_trials_per_average": 2,
                "embedding_ridge_alpha": 1.0,
            }, payload)
        self.assertIsNotNone(result, error)
        self.assertEqual(result["instruction_evaluation_status"], "evaluated")
        self.assertGreater(result["instruction_top1_accuracy"], 0.9)
        self.assertEqual(result["instruction_scored_predictions"], len(result["predictions_df"]))
        self.assertIn("expected_instruction", result["predictions_df"])

    def test_reads_unique_text_and_joins_by_block_id(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "block-instructions.csv"
            path.write_text("block_id,line_2\n4,ANIMAL\n2,TOOL\n7,ANIMAL\n8,\n")
            targets = read_instruction_csv(path)
        self.assertEqual(targets["names"], ["ANIMAL", "TOOL"])
        self.assertEqual(targets["block_mapping"]["2"], "TOOL")

    def test_conflicting_blocks_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "block-instructions.csv"
            path.write_text("block_id,line_2\n4,A\n4,B\n")
            with self.assertRaisesRegex(ValueError, "Conflicting"):
                read_instruction_csv(path)

    def test_concept_label_uses_nearest_block_instruction_by_block_id(self):
        from core.concept_decoding_signal import (
            _semantic_decision_mapping_for_location,
            concept_label_from_trial_row,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "session" / "run"
            run_dir.mkdir(parents=True)
            fif_path = run_dir / "recording_raw.fif"
            fif_path.touch()
            (root / "block-instructions.csv").write_text(
                "block_id,line_2\n4,ANIMAL or INSECT\n2,TOOL\n"
            )
            _semantic_decision_mapping_for_location.cache_clear()
            row = pd.Series({"source_fif_path": str(fif_path), "block_id": 4.0})
            label = concept_label_from_trial_row(None, row, group_mode="semantic_decision")

        self.assertEqual(label, "ANIMAL or INSECT")

    def test_scores_only_unambiguous_expectations_using_supplied_transform(self):
        targets = {"names": ["ANIMAL", "TOOL"],
                   "embeddings": [[1, 0], [0, 1]],
                   "block_mapping": {"4": "ANIMAL", "2": "TOOL"}}
        trials = pd.DataFrame({"block_id": [4, 2, 99, 4]})
        predictions = pd.DataFrame({"source_trial_indices": ["0,3", "1", "0,1", "2"]})
        # A fitted projection swaps coordinates: instructions must follow it.
        metrics = score_instructions(predictions, np.array([[0, 1], [1, 0], [1, 0], [1, 0]]),
                                     targets, trials, lambda vectors: vectors[:, ::-1])
        self.assertEqual(metrics["instruction_scored_predictions"], 2)
        self.assertEqual(metrics["instruction_top1_accuracy"], 1.0)
        self.assertEqual(predictions.loc[2, "instruction_status"], "mixed_block_instructions")
        self.assertEqual(predictions.loc[3, "instruction_status"], "missing_block_instruction")
        self.assertTrue(pd.isna(predictions.loc[2, "instruction_match"]))


if __name__ == "__main__":
    unittest.main()
