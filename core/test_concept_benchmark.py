"""Leakage, reproducibility and end-to-end benchmark smoke checks."""
import argparse
import json
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from core.concept_benchmark_splits import make_splits, validate_indices
from core.concept_decoding_transformer import compute_transformer_decoding_from_payload
from core.test_concept_embedding_decoders import _synthetic_embedding_payload
from scripts.run_concept_benchmark import atomic_json, digest, identity, make_jobs, run_jobs


def frame():
    return pd.DataFrame([dict(trial_id=f"{c}-{i}-{s}", concept=f"c{c}", image=f"c{c}_image{i}", session=str(s))
                         for c in range(10) for i in range(10) for s in (1, 2)])


class BenchmarkSplitsTests(unittest.TestCase):
    def test_group_integrity_and_coverage(self):
        data = frame()
        splits = make_splits(data)
        self.assertEqual(len(splits), 12)
        for track in ("image", "zero_shot", "session"):
            tested = [i for split in splits if split["track"] == track for i in split["test"]]
            self.assertEqual(sorted(tested), list(range(len(data))))
        for split in splits:
            a, b, c = validate_indices(split, data.concept, zero_shot=split["track"] == "zero_shot")
            self.assertFalse(set(data.iloc[a].image) & set(data.iloc[b].image))
            if split["track"] == "image":
                self.assertFalse(set(data.iloc[c].image) & set(data.iloc[a].image))
                self.assertFalse(set(data.iloc[c].image) & set(data.iloc[b].image))
            if split["track"] == "session":
                self.assertFalse(set(data.iloc[c].session) & set(data.iloc[a].session))

    def test_reproducibility_and_independent_seed(self):
        self.assertEqual(make_splits(frame()), make_splits(frame()))
        self.assertNotEqual(make_splits(frame(), seed=1), make_splits(frame(), seed=2))

    def test_reject_overlap_float_indices_missing_and_negative(self):
        data = frame()
        split = make_splits(data)[0]
        for replacement in (split["train"] + [split["test"][0]], [-1], [1.2], []):
            with self.subTest(replacement=str(replacement)[:20]), self.assertRaises(ValueError):
                validate_indices(dict(split, train=replacement), data.concept)

    def test_missing_groups_fail(self):
        data = frame()
        data.loc[0, "image"] = None
        with self.assertRaises(ValueError):
            make_splits(data)

    def test_plan_counts(self):
        cfg = dict(training_seeds=[42, 43, 44], null_permutations=99)
        jobs = make_jobs(cfg, make_splits(frame()))
        self.assertEqual(sum(j["stage"] == "main" for j in jobs), 128)
        self.assertEqual(sum(j["stage"] == "null" for j in jobs), 990)
        self.assertEqual(len({j["id"] for j in jobs}), len(jobs))


class BenchmarkBackendTests(unittest.TestCase):
    def options_and_split(self):
        split = {"train": [], "validation": [], "test": []}
        for start in (0, 12, 24):
            split["train"] += list(range(start, start + 6))
            split["validation"] += list(range(start + 6, start + 9))
            split["test"] += list(range(start + 9, start + 12))
        options = dict(decoder_type="embedding_ridge", split_mode="image", explicit_split_indices=split,
                       transformer_trials_per_average=1, n_permutations=0, require_subset_validation=True,
                       embedding_validation_metric="mpr", training_seed=17)
        return options, split

    def test_backend_uses_explicit_indices(self):
        options, _ = self.options_and_split()
        result, error = compute_transformer_decoding_from_payload(None, options, _synthetic_embedding_payload())
        self.assertIsNotNone(result, error)
        self.assertEqual(result["n_source_train_trials"], 18)
        self.assertEqual(result["n_source_validation_trials"], 9)
        self.assertEqual(result["n_source_test_trials"], 9)
        self.assertEqual(result["split_mode"], "image")
        self.assertEqual(result["training_seed"], 17)
        self.assertGreater(result["mpr"], 0.9)
        self.assertTrue(np.isfinite(result["validation_mpr"]))

    def test_contrastive_explicit_split_and_mpr_selection(self):
        options, _ = self.options_and_split()
        options.update(decoder_type="embedding_contrastive", transformer_epochs=2, epochs=2)
        result, error = compute_transformer_decoding_from_payload(None, options, _synthetic_embedding_payload())
        self.assertIsNotNone(result, error)
        self.assertEqual(result["n_source_test_trials"], 9)
        self.assertTrue(np.isfinite(result["validation_mpr"]))

    def test_explicit_zero_shot_scores_validation_on_training_candidates(self):
        payload = _synthetic_embedding_payload()
        labels = np.repeat([f"class{i}" for i in range(5)], 12)
        payload["X_tensor"] = np.concatenate([payload["X_tensor"], payload["X_tensor"][:24]])
        payload["y"] = labels
        payload["embeddings"] = np.repeat(np.eye(5, dtype=np.float32), 12, axis=0)
        payload["trial_df"] = pd.DataFrame(dict(concept_label=labels, canonical_stimulus=labels,
                                                trial_index=np.arange(60), embedding_key=labels))
        options, _ = self.options_and_split()
        options.update(split_mode="zero_shot", explicit_split_indices=dict(
            train=[i for start in (0, 12, 24) for i in range(start, start + 6)],
            validation=[i for start in (0, 12, 24) for i in range(start + 6, start + 12)],
            test=list(range(36, 60))))
        result, message = compute_transformer_decoding_from_payload(None, options, payload)
        self.assertIsNotNone(result, message)
        self.assertEqual(result["n_unseen_concepts"], 2)
        self.assertEqual(result["n_source_test_trials"], 24)
        self.assertTrue(np.isfinite(result["validation_mpr"]))

    def test_run_collect_resume_and_provenance(self):
        options, split = self.options_and_split()
        payload = _synthetic_embedding_payload()
        n = len(payload["y"])
        payload["trial_df"]["source_fif_path"] = "synthetic.fif"
        payload["trial_df"]["onset_sample"] = np.arange(n)
        payload["trial_df"]["stimulus"] = [f"image{i}" for i in range(n)]
        payload["trial_df"]["session_id"] = "1"
        matched = pd.DataFrame(dict(trial_id=[f"synthetic.fif::{i}" for i in range(n)], concept=payload["y"],
                                    image=[f"image{i}" for i in range(n)], session=["1"] * n))
        job = dict(id="smoke", stage="main", track="image", fold="0", model="aparc34",
                   decoder="embedding_ridge", seed=17, permutation=0)
        provenance = {"synthetic": True}
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            (out / "features").mkdir()
            joblib.dump(payload, out / "features/aparc.joblib")
            matched.to_csv(out / "matched_trials.csv", index=False)
            atomic_json(out / "plan.json", dict(fingerprint=digest(provenance), splits=[dict(split, track="image", fold="0")],
                jobs=[job, dict(job, id="smoke-null", stage="null", permutation=1)], matched_digest=digest(matched.to_dict("records")), feature_files={"aparc": identity(out / "features/aparc.joblib")}))
            cfg = dict(output_dir=temp, training_seeds=[17], null_permutations=1, split_seed=3)
            options["selected_parcellations"] = []
            args = argparse.Namespace(stage="main", track=None, decoder=None, limit=1)
            run_jobs(cfg, options, provenance, args)
            marker = out / "runs/smoke/complete.json"
            before = marker.stat().st_mtime_ns
            run_jobs(cfg, options, provenance, args)
            self.assertEqual(before, marker.stat().st_mtime_ns)
            self.assertTrue((out / "report/per_concept_pooled.csv").exists())
            args.stage = "null"
            run_jobs(cfg, options, provenance, args)
            self.assertTrue((out / "runs/smoke-null/null_mapping.json").exists())
            null_settings = json.loads((out / "runs/smoke-null/settings_used.json").read_text())
            self.assertEqual(null_settings["explicit_split_indices"]["test"], split["test"])
            with self.assertRaises(ValueError):
                run_jobs(cfg, options, {"synthetic": False}, args)


if __name__ == "__main__":
    unittest.main()
