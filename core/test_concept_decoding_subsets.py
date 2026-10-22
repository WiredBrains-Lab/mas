import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

from core.concept_decoding_subsets import (
    SEMANTIC_APARC_POOL, run_semantic_subset_batch, semantic_subsets, subset_payload,
    exhaustive_subsets,
    backward_fit_count, choose_backward_candidate,
    _format_duration, _ScreeningBests, _log_screening_trial_counts,
)
from core.test_concept_embedding_decoders import _synthetic_source_feature_payload


def payload():
    result = _synthetic_source_feature_payload()
    # Time-major columns deliberately interleave parcels.
    X = np.tile(result["X"], (1, 3))
    meta = pd.DataFrame({"roi": list(SEMANTIC_APARC_POOL) * 2,
                         "feature": [f"f{i}" for i in range(12)]})
    result.update(X=X, feature_meta_df=meta, feature_names=meta.feature.tolist(),
                  roi_names=list(SEMANTIC_APARC_POOL))
    return result


class SemanticSubsetTests(unittest.TestCase):
    def test_metric_bests_independent_resume_ties_and_invalid_values(self):
        rows = [
            dict(status="complete", subset_id=7, n_parcels=3, top1_accuracy=.3, mpr=.55),
            dict(status="complete", subset_id=2, n_parcels=1, top1_accuracy=.2, mpr=.7),
            dict(status="failed", subset_id=1, n_parcels=4, top1_accuracy=1, mpr=1),
            dict(status="complete", subset_id=3, n_parcels=2, top1_accuracy=None, mpr=float("nan")),
        ]
        best = _ScreeningBests(rows)
        self.assertEqual(best.label(),
                         "best top-1 30.00% [subset 7, 3 parcels]; best MPR 0.7000 [subset 2, 1 parcels]")
        best.update(dict(status="complete", subset_id=4, n_parcels=2, top1_accuracy=.3, mpr=.6))
        self.assertEqual(best.leaders['top1_accuracy']['subset_id'], 4)
        self.assertEqual(best.leaders['mpr']['subset_id'], 2)
        self.assertEqual(_ScreeningBests().label(), "best top-1 unavailable; best MPR unavailable")

    def test_trial_summary_counts_retained_trials_not_qc_rows(self):
        data = dict(trial_df=pd.DataFrame({"source_fif_path": ["a", "a", "b"]}),
                    y=np.array(["apple", "chair", "apple"]),
                    trial_qc_df=pd.DataFrame({"qc_status": ["kept"] * 3 + ["dropped"] * 4}))
        messages = []
        _log_screening_trial_counts(data, messages.append)
        _log_screening_trial_counts(data, messages.append, saved=True)
        for message in messages:
            self.assertIn("3 individual trials retained after QC and eligibility filtering; 2 concepts, 2 runs", message)
            self.assertIn("before train/validation/test splitting and trial averaging", message)
        self.assertIn("loaded from saved checkpoint", messages[1])

    def test_eta_duration_format(self):
        self.assertEqual(_format_duration(0), "0s")
        self.assertEqual(_format_duration(65), "1m 05s")
        self.assertEqual(_format_duration(3661), "1h 01m 01s")
        self.assertEqual(_format_duration(90061), "1d 01h 01m")

    def test_all_combinations(self):
        subsets = semantic_subsets()
        self.assertEqual(len(set(subsets)), 63)
        self.assertEqual(subsets[0], SEMANTIC_APARC_POOL)
        self.assertEqual(Counter(map(len, subsets)), {1: 6, 2: 15, 3: 20, 4: 15, 5: 6, 6: 1})

    def test_interleaved_columns_and_missing_parcel(self):
        original = payload()
        selected = subset_payload(original, [SEMANTIC_APARC_POOL[1]])
        np.testing.assert_array_equal(selected["X_tensor"], original["X"][:, [1, 7]])
        self.assertIs(selected["trial_df"], original["trial_df"])
        self.assertEqual(selected["feature_meta_df"].feature_index.tolist(), [0, 1])
        self.assertEqual(original["X"].shape[1], 12)
        with self.assertRaisesRegex(ValueError, "Missing"):
            subset_payload(original, ["nonexistent"])

    def test_generic_pool_and_size_guard(self):
        self.assertEqual(list(exhaustive_subsets(["right-a", "left-b", "right-a"])),
                         [("right-a", "left-b"), ("right-a",), ("left-b",)])
        with self.assertRaisesRegex(ValueError, "at least one"):
            exhaustive_subsets([])
        self.assertEqual(exhaustive_subsets([f"p{i}" for i in range(34)]).total, 17179869183)
        self.assertEqual(backward_fit_count([f"p{i}" for i in range(34)]), 595)

    def test_exhaustive_eighteen_parcel_boundary(self):
        pool = tuple(f"p{i}" for i in range(18))
        subsets = exhaustive_subsets(pool)
        self.assertEqual(len(subsets), 262143)
        self.assertEqual(len(set(subsets)), 262143)
        self.assertEqual(subsets[0], pool)
        self.assertTrue(all(subsets))
        self.assertEqual(exhaustive_subsets([f"p{i}" for i in range(19)]).total, 524287)

    def test_lazy_large_pool_and_checkpoint_indexing(self):
        from itertools import islice
        pool = tuple(f"p{i}" for i in range(112))
        subsets = exhaustive_subsets(pool)
        self.assertEqual(subsets.total, 2**112 - 1)
        self.assertEqual(list(islice(subsets, 3)), [pool, ("p0",), ("p1",)])
        self.assertEqual(subsets[subsets.total - 1], pool[1:])
        small = exhaustive_subsets(pool[:8])
        for index, subset in enumerate(small):
            self.assertEqual(small[index], subset)

    def test_large_screen_starts_cancels_and_resumes_lazily(self):
        pool = [f'p{i}' for i in range(112)]
        data = payload()
        data.update(X=np.ones((len(data['trial_df']), len(pool))),
                    feature_meta_df=pd.DataFrame({'roi': pool, 'feature': pool}),
                    feature_names=pool, roi_names=pool)
        calls = []
        def fit(*_):
            calls.append(1)
            return {'top1_accuracy': 0.2, 'mpr': 0.5}, 'ok'
        with tempfile.TemporaryDirectory() as directory, \
             patch('core.concept_decoding_source.build_source_space_concept_features', return_value=(data, None)), \
             patch('core.concept_decoding_transformer.prepare_semantic_embedding_feature_payload', return_value=(data, None)), \
             patch('core.concept_decoding_transformer.compute_transformer_decoding_from_payload', side_effect=fit):
            result, _ = run_semantic_subset_batch(SimpleNamespace(), dict(
                decoder_type='embedding_ridge', selected_parcellations=pool,
                semantic_subset_output_dir=directory, cancel_callback=lambda: len(calls) >= 1))
            folder = Path(result['semantic_subset_output_dir'])
            manifest = json.loads((folder / 'manifest.json').read_text())
            self.assertEqual(manifest['total_subsets'], 2**112 - 1)
            self.assertEqual(manifest['attempted'], 1)
            run_semantic_subset_batch(SimpleNamespace(), dict(
                semantic_subset_resume_dir=str(folder), cancel_callback=lambda: len(calls) >= 2))
            rows = pd.read_csv(folder / 'subset_comparison.csv')
            self.assertEqual(rows.subset_id.tolist(), [1, 2])
            self.assertEqual(rows.iloc[1].parcels, 'p0')

    def test_backward_choice_ignores_test_and_breaks_ties(self):
        rows = [dict(status="complete", parcels="a", validation_top1_accuracy=0.3, validation_mpr=0.6, top1_accuracy=0.0),
                dict(status="complete", parcels="b", validation_top1_accuracy=0.2, validation_mpr=0.9, top1_accuracy=1.0)]
        self.assertEqual(choose_backward_candidate(rows)["parcels"], "a")
        rows[1].update(validation_top1_accuracy=0.3, validation_mpr=0.7)
        self.assertEqual(choose_backward_candidate(rows)["parcels"], "b")
        rows[1]["validation_mpr"] = 0.6
        self.assertEqual(choose_backward_candidate(rows)["parcels"], "a")
        rows[1]["status"] = "failed"
        with self.assertRaisesRegex(ValueError, "candidate fit failed"):
            choose_backward_candidate(rows)

    def test_real_backward_ridge_path(self):
        with tempfile.TemporaryDirectory() as directory:
            result, message = self.run_batch(directory, selected_parcellations=list(SEMANTIC_APARC_POOL[:3]),
                                            subset_search_strategy="backward", validation_fraction=0.2)
            folder = Path(result["semantic_subset_output_dir"])
            path = pd.read_csv(folder / "elimination_path.csv")
            self.assertEqual(path.n_parcels.tolist(), [3, 2, 1], message)
            self.assertEqual(len(list(folder.glob("*/result_summary.json"))), 6)
            self.assertTrue((folder / "elimination_path.png").exists())
            for i in range(1, len(path)):
                previous = set(path.iloc[i-1].parcels.split(";"))
                current = set(path.iloc[i].parcels.split(";"))
                self.assertTrue(current < previous)
                self.assertEqual(path.iloc[i].removed_parcel, next(iter(previous-current)))
            bundles=[joblib.load(f) for f in sorted(folder.glob("*/inference_bundle.joblib"))]
            for bundle in bundles:
                self.assertGreater(len(bundle["validation_indices"]),0)
                for key in ("test_indices", "validation_indices", "train_indices"):
                    np.testing.assert_array_equal(bundle[key],bundles[0][key])

    def test_backward_failure_keeps_path_and_stops(self):
        calls = []
        def fake(_context, _options, _payload):
            calls.append(1)
            if len(calls) == 2:
                return None, "failed candidate"
            return dict(mpr=0.5, top1_accuracy=0.2, validation_top1_accuracy=0.2, validation_mpr=0.5), "ok"
        with tempfile.TemporaryDirectory() as directory:
            with patch("core.concept_decoding_transformer.compute_transformer_decoding_from_payload", side_effect=fake):
                with self.assertRaisesRegex(RuntimeError, "candidate fit failed"):
                    self.run_batch(directory, selected_parcellations=list(SEMANTIC_APARC_POOL[:3]), subset_search_strategy="backward")
            folder = next(Path(directory).glob("*_backward_elimination"))
            self.assertEqual(len(calls), 4)
            self.assertEqual(len(pd.read_csv(folder / "elimination_path.csv")), 1)
            self.assertEqual(json.loads((folder / "manifest.json").read_text())["status"], "failed")

    def test_generic_two_parcels_preserve_atlas(self):
        with tempfile.TemporaryDirectory() as directory:
            result, message = self.run_batch(directory, selected_parcellations=list(SEMANTIC_APARC_POOL[:2]),
                                            source_parcellation="HCPMMP1", source_space_kind="surface")
            folder = Path(result["semantic_subset_output_dir"])
            self.assertEqual(len(list(folder.glob("*/result_summary.json"))), 3, message)
            settings = json.loads((folder / "settings_used.json").read_text())
            self.assertEqual(settings["source_parcellation"], "HCPMMP1")
            self.assertEqual(settings["selected_parcellations"], list(SEMANTIC_APARC_POOL[:2]))

    def test_zero_shot_subset_batch_reuses_unseen_concepts(self):
        with tempfile.TemporaryDirectory() as directory:
            result, message = self.run_batch(
                directory,
                selected_parcellations=list(SEMANTIC_APARC_POOL[:2]),
                split_mode="zero_shot",
                embedding_target_svd_components=0,
            )
            folder = Path(result["semantic_subset_output_dir"])
            manifest = json.loads((folder / "manifest.json").read_text())
            self.assertEqual(manifest["split_mode"], "zero_shot", message)
            bundles = [joblib.load(path) for path in sorted(folder.glob("*/inference_bundle.joblib"))]
            self.assertEqual(len(bundles), 3)
            unseen = [tuple(sorted(set(map(str, bundle["test_y"])))) for bundle in bundles]
            self.assertTrue(unseen[0])
            self.assertTrue(all(names == unseen[0] for names in unseen))

    def run_batch(self, folder, **extra):
        dictionary = Path(folder) / "embeddings.json"
        dictionary.write_text(json.dumps({"airplane": [1, 0, 0], "dog": [0, 1, 0], "guitar": [0, 0, 1]}))
        options = dict(decoder_type="embedding_ridge", split_mode="random",
                       selected_parcellations=list(SEMANTIC_APARC_POOL),
                       semantic_subset_output_dir=folder, embedding_dict_path=str(dictionary),
                       transformer_trials_per_average=1, n_permutations=0,
                       embedding_input_pca_components=2, embedding_target_svd_components=2)
        options.update(extra)
        with patch("core.concept_decoding_source.build_source_space_concept_features",
                   return_value=(payload(), None)) as build:
            result, message = run_semantic_subset_batch(SimpleNamespace(), options)
            build.assert_called_once()
            self.assertEqual(build.call_args.args[1]["max_rois"], 0)
        return result, message

    def test_real_ridge_all_63_saved_and_same_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            result, message = self.run_batch(directory)
            folder = Path(result["semantic_subset_output_dir"])
            manifest = json.loads((folder / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "complete", message)
            self.assertEqual(manifest["completed"], 63)
            comparison = pd.read_csv(folder / "subset_comparison.csv")
            self.assertEqual(len(comparison), 63)
            self.assertEqual(comparison.n_test.nunique(), 1)
            bundles = sorted(folder.glob("*/inference_bundle.joblib"))
            self.assertEqual(len(bundles), 63)
            first = joblib.load(bundles[0])
            last = joblib.load(bundles[-1])
            np.testing.assert_array_equal(first["test_indices"], last["test_indices"])
            np.testing.assert_array_equal(first["test_y"], last["test_y"])
            self.assertTrue((folder / "source_features.joblib").is_file())
            self.assertTrue((folder / "performance_vs_parcel_count.png").is_file())

    def test_cancel_keeps_first_result(self):
        checks = iter([False, True])
        with tempfile.TemporaryDirectory() as directory:
            result, message = self.run_batch(directory, cancel_callback=lambda: next(checks))
            folder = Path(result["semantic_subset_output_dir"])
            self.assertIn("cancelled", message)
            self.assertEqual(len(list(folder.glob("*/result_summary.json"))), 1)
            self.assertEqual(json.loads((folder / "manifest.json").read_text())["completed"], 1)

    def test_resume_exhaustive_reuses_checkpoint_and_saved_features(self):
        checks = iter([False, True])
        with tempfile.TemporaryDirectory() as directory:
            result, _ = self.run_batch(
                directory,
                selected_parcellations=list(SEMANTIC_APARC_POOL[:2]),
                cancel_callback=lambda: next(checks),
            )
            folder = Path(result["semantic_subset_output_dir"])
            with patch(
                "core.concept_decoding_source.build_source_space_concept_features",
                side_effect=AssertionError("resume must not re-extract source features"),
            ) as build:
                resumed, message = run_semantic_subset_batch(
                    SimpleNamespace(),
                    {"semantic_subset_resume_dir": str(folder)},
                )
            build.assert_not_called()
            self.assertIn("complete", message)
            self.assertEqual(len(pd.read_csv(folder / "subset_comparison.csv")), 3)
            manifest = json.loads((folder / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["resume_count"], 1)
            self.assertEqual(resumed["semantic_subset_output_dir"], str(folder))

    def test_cli_resume_relocated_checkpoint_and_read_only_dry_run(self):
        import shutil
        from scripts.run_concept_decoding_cli import main
        with tempfile.TemporaryDirectory() as directory:
            original = Path(directory) / 'original'
            original.mkdir()
            checks = iter([False, True])
            result, _ = self.run_batch(str(original), selected_parcellations=list(SEMANTIC_APARC_POOL[:2]),
                                       cancel_callback=lambda: next(checks))
            old_batch = Path(result['semantic_subset_output_dir'])
            moved = Path(directory) / 'rcc'
            shutil.move(str(original), moved)
            batch = moved / old_batch.name
            command = ['--resume', str(batch), '--resume-search-root', str(moved)]
            manifest_before = (batch / 'manifest.json').read_bytes()
            features_before = (batch / 'source_features.joblib').read_bytes()
            settings_before = (batch / 'settings_used.json').read_bytes()
            with patch('core.concept_decoding_source.build_source_space_concept_features', side_effect=AssertionError('Re-extracted')):
                self.assertEqual(main(command + ['--dry-run']), 0)
                self.assertEqual((batch / 'manifest.json').read_bytes(), manifest_before)
                self.assertEqual(main(command), 0)
            self.assertEqual(len(pd.read_csv(batch / 'subset_comparison.csv')), 3)
            self.assertEqual((batch / 'source_features.joblib').read_bytes(), features_before)
            self.assertEqual((batch / 'settings_used.json').read_bytes(), settings_before)
            rows = pd.read_csv(batch / 'subset_comparison.csv')
            self.assertTrue(all(str(batch) in p for p in rows.output_dir))

    def test_resume_path_ambiguity_missing_and_explicit_map(self):
        from core.decoding_resume_paths import ResumePathResolver
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            batch = root / 'batch'
            batch.mkdir()
            for name in ('a', 'b'):
                (root / name).mkdir()
                (root / name / 'dictionary.json').write_text('{}')
            old = '/absent/data/dictionary.json'
            resolver = ResumePathResolver(batch, '/old/batch', [root], log=lambda _: None)
            with self.assertRaisesRegex(ValueError, 'Ambiguous'):
                resolver.transform({'embedding_dict_path': old})
            with self.assertRaises(FileNotFoundError):
                resolver.transform({'embedding_dict_path': '/absent/missing.json'})
            mapped = ResumePathResolver(batch, '/old/batch', [root],
                                        [('/absent/data', str(root / 'a'))], log=lambda _: None)
            self.assertEqual(mapped.resolve(old, required=True), str((root / 'a/dictionary.json').resolve()))

    def test_resume_historical_paths_never_trigger_search(self):
        from core.decoding_resume_paths import ResumePathResolver
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            batch = root / 'batch'
            batch.mkdir()
            dictionary = root / 'dictionary.json'
            dictionary.write_text('{}')
            old = '/absent/data/dictionary.json'
            resolver = ResumePathResolver(batch, '/old/batch', [root], log=lambda _: None)
            with patch('core.decoding_resume_paths.os.walk', side_effect=AssertionError('Historical scan')):
                result = resolver.transform({'fif_path': old, 'anatomy': '/absent/anatomy',
                                             'output_dir': '/old/batch/001'})
                self.assertEqual(result['fif_path'], old)
                self.assertEqual(result['anatomy'], '/absent/anatomy')
                self.assertEqual(result['output_dir'], str(batch.resolve() / '001'))
            # An earlier optional reference must not suppress required recovery.
            self.assertEqual(resolver.transform({'embedding_dict_path': old})['embedding_dict_path'],
                             str(dictionary.resolve()))
            mapped = ResumePathResolver(batch, '/old/batch', [root],
                                        [('/absent/data', str(root))], log=lambda _: None)
            with patch('core.decoding_resume_paths.os.walk', side_effect=AssertionError('Mapped scan')):
                self.assertEqual(mapped.resolve(old), str(dictionary.resolve()))
                self.assertEqual(mapped.resolve(old, required=True), str(dictionary.resolve()))

    def test_parallel_resume_real_processes_matches_serial(self):
        import os
        with tempfile.TemporaryDirectory() as directory:
            checks = iter([False, True])
            result, _ = self.run_batch(directory, selected_parcellations=list(SEMANTIC_APARC_POOL[:3]),
                                       cancel_callback=lambda: next(checks))
            folder = Path(result['semantic_subset_output_dir'])
            first = pd.read_csv(folder / 'subset_comparison.csv').iloc[0].output_dir
            updates = []
            resumed, message = run_semantic_subset_batch(SimpleNamespace(), {
                'semantic_subset_resume_dir': str(folder), 'subset_workers': 2,
                'subset_worker_threads': 1, 'progress_callback': updates.append})
            rows = pd.read_csv(folder / 'subset_comparison.csv')
            self.assertEqual(rows.subset_id.tolist(), list(range(1, 8)), message)
            self.assertTrue((rows.status == 'complete').all(), rows.to_string())
            self.assertEqual(rows.iloc[0].output_dir, first)
            self.assertTrue(any('trial summary (loaded from saved checkpoint)' in m for m in updates))
            final_progress = [m for m in updates if 'fits recorded;' in m][-1]
            self.assertIn('fit ETA ~0s', final_progress)
            self.assertIn(f"best top-1 {100 * rows.top1_accuracy.max():.2f}%", final_progress)
            for metric, label in (("top1_accuracy", "top-1"), ("mpr", "MPR")):
                winner = rows.sort_values([metric, "subset_id"], ascending=[False, True]).iloc[0]
                score = f"{100 * winner[metric]:.2f}%" if metric == "top1_accuracy" else f"{winner[metric]:.4f}"
                self.assertIn(f"best {label} {score} [subset {int(winner.subset_id)}, "
                              f"{int(winner.n_parcels)} parcels]", final_progress)


            self.assertTrue(all(int(pid) != os.getpid() for pid in rows.worker_pid.dropna()))
            self.assertEqual(json.loads((folder / 'manifest.json').read_text())['subset_workers'], 2)
            # Recompute one subset serially from the same saved features.
            saved_row = rows.iloc[-1]
            comparison = __import__('core.concept_decoding_transformer', fromlist=['compute_transformer_decoding_from_payload'])
            saved_opts = json.loads((folder / 'settings_used.json').read_text())
            features = joblib.load(folder / 'source_features.joblib')
            serial, _ = comparison.compute_transformer_decoding_from_payload(SimpleNamespace(), saved_opts,
                               subset_payload(features, tuple(saved_row.parcels.split(';'))))
            self.assertAlmostEqual(saved_row.top1_accuracy, serial['top1_accuracy'])
            self.assertAlmostEqual(saved_row.mpr, serial['mpr'])

    def test_resume_sparse_checkpoint_runs_only_missing_subset(self):
        with tempfile.TemporaryDirectory() as directory:
            result, _ = self.run_batch(directory, selected_parcellations=list(SEMANTIC_APARC_POOL[:2]))
            folder = Path(result['semantic_subset_output_dir'])
            rows = pd.read_csv(folder / 'subset_comparison.csv')
            retained = rows.loc[rows.subset_id != 2]
            retained.to_csv(folder / 'subset_comparison.csv', index=False)
            manifest = json.loads((folder / 'manifest.json').read_text())
            manifest.update(attempted=2, completed=2, status='cancelled')
            (folder / 'manifest.json').write_text(json.dumps(manifest))
            run_semantic_subset_batch(SimpleNamespace(), {'semantic_subset_resume_dir': str(folder)})
            after = pd.read_csv(folder / 'subset_comparison.csv')
            self.assertEqual(after.subset_id.tolist(), [1, 2, 3])
            self.assertEqual(after.iloc[0].output_dir, rows.iloc[0].output_dir)
            self.assertEqual(after.iloc[2].output_dir, rows.iloc[2].output_dir)
            self.assertNotEqual(after.iloc[1].output_dir, rows.iloc[1].output_dir)

    def test_parallel_scheduler_out_of_order_and_cancel_drains_active(self):
        from concurrent.futures import ThreadPoolExecutor
        from threading import Event
        import time
        from core.concept_subset_parallel import parallel_rows
        second_started = Event()
        stopped = [False]
        submitted = []
        def fit(index, parcels, destination, note):
            submitted.append(index)
            if index == 1:
                self.assertTrue(second_started.wait(5))
                time.sleep(0.15)
            else:
                second_started.set()
            return {'subset_id': index, 'status': 'complete'}
        with tempfile.TemporaryDirectory() as folder:
            with patch('core.concept_subset_parallel.ProcessPoolExecutor',
                       side_effect=lambda **kw: ThreadPoolExecutor(max_workers=kw['max_workers'])), \
                 patch('core.concept_subset_parallel._fit', side_effect=fit):
                finished = []
                for row in parallel_rows([(1, ('a',)), (2, ('b',)), (3, ('c',))],
                                          folder, {}, {}, 2, 1, '', lambda: stopped[0], lambda _: None):
                    finished.append(row['subset_id'])
                    stopped[0] = True
        self.assertEqual(finished, [2, 1])
        self.assertEqual(sorted(submitted), [1, 2])

    def test_resume_lock_blocks_second_coordinator(self):
        import fcntl
        with tempfile.TemporaryDirectory() as directory:
            with (Path(directory) / '.subset_batch.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaisesRegex(RuntimeError, 'Another screening coordinator'):
                    run_semantic_subset_batch(SimpleNamespace(), {'semantic_subset_resume_dir': directory})

    def test_contrastive_inference_bundle_reloads(self):
        from core.concept_decoding_transformer import _predict_flat_embedding_model, _l2_normalize
        checks = iter([False, True])
        with tempfile.TemporaryDirectory() as directory:
            result, _ = self.run_batch(directory, decoder_type="embedding_contrastive",
                                       transformer_epochs=2, validation_fraction=0.2,
                                       subset_search_strategy="backward",
                                       cancel_callback=lambda: next(checks))
            folder = Path(result["semantic_subset_output_dir"])
            bundle = joblib.load(next(folder.glob("*/inference_bundle.joblib")))
            predicted = _predict_flat_embedding_model(bundle["model"], bundle["test_X"], bundle["train_meta"])
            scores = _l2_normalize(predicted) @ bundle["prototypes"].T
            np.testing.assert_allclose(scores, bundle["scores"], atol=1e-5)
            validation_pred = _predict_flat_embedding_model(bundle["model"], bundle["validation_X"], bundle["train_meta"])
            np.testing.assert_allclose(_l2_normalize(validation_pred) @ bundle["prototypes"].T,
                                       bundle["validation_scores"], atol=1e-5)
            self.assertEqual(len(pd.read_csv(folder / "elimination_path.csv")), 1)

    def test_backward_rejects_zero_validation_before_extraction(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "greater than zero"):
                self.run_batch(directory, subset_search_strategy="backward", validation_fraction=0)
            self.assertEqual(len(list(Path(directory).glob("*_backward_elimination"))), 0)

    def test_subset_failure_continues(self):
        calls = []
        def fail_first(_context, _options, _payload):
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError("synthetic fit failure")
            return {"mpr": 0.5, "top1_accuracy": 0.33}, "ok"
        with tempfile.TemporaryDirectory() as directory:
            with patch("core.concept_decoding_transformer.compute_transformer_decoding_from_payload",
                       side_effect=fail_first):
                result, message = self.run_batch(directory)
            manifest = json.loads((Path(result["semantic_subset_output_dir"]) / "manifest.json").read_text())
            self.assertEqual((manifest["completed"], manifest["failed"]), (62, 1), message)

    def test_exhaustive_logs_only_strict_top1_improvements(self):
        scores = iter([(0.20, 0.51, 0.40), (0.20, 0.60, 0.50), (0.30, 0.57, 0.70)])
        messages = []
        progress_messages = []

        def fake(_context, _options, _payload):
            top1, mpr, top5 = next(scores)
            return {"top1_accuracy": top1, "mpr": mpr, "top5_accuracy": top5}, "ok"

        with tempfile.TemporaryDirectory() as directory:
            with patch(
                "core.concept_decoding_transformer.compute_transformer_decoding_from_payload",
                side_effect=fake,
            ):
                result, _ = self.run_batch(
                    directory,
                    selected_parcellations=list(SEMANTIC_APARC_POOL[:2]),
                    log_callback=messages.append,
                    progress_callback=progress_messages.append,
                )
            self.assertTrue(any("trial summary (features ready)" in m for m in progress_messages))
            self.assertEqual(len(messages), 2)
            self.assertIn("20.00% (top-5 40.00%, MPR 0.5100)", messages[0])
            self.assertIn("30.00% (top-5 70.00%, MPR 0.5700)", messages[1])
            manifest = json.loads((Path(result["semantic_subset_output_dir"]) / "manifest.json").read_text())
            self.assertEqual(manifest["best_top1_accuracy"], 0.30)
            self.assertEqual(manifest["best_top5_accuracy"], 0.70)
            self.assertEqual(manifest["best_mpr"], 0.57)  # Legacy metric of top-1 winner.
            self.assertEqual(manifest["metric_bests"]["mpr"],
                             dict(value=.60, subset_id=2, n_parcels=1))
            final_progress = [m for m in progress_messages if "fits attempted" in m][-1]
            self.assertIn("best top-1 30.00% [subset 3, 1 parcels]", final_progress)
            self.assertIn("best MPR 0.6000 [subset 2, 1 parcels]", final_progress)

    def test_extraction_failure_records_output_path(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("core.concept_decoding_source.build_source_space_concept_features",
                       return_value=(None, "missing inverse")):
                with self.assertRaisesRegex(RuntimeError, "Saved progress"):
                    run_semantic_subset_batch(None, dict(decoder_type="embedding_ridge",
                                                         selected_parcellations=list(SEMANTIC_APARC_POOL),
                                                         semantic_subset_output_dir=directory))
            manifest = json.loads(next(Path(directory).glob("*/manifest.json")).read_text())
            self.assertEqual(manifest["status"], "failed")


if __name__ == "__main__":
    unittest.main()
