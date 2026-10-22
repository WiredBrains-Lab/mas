import json
import hashlib
import unittest
import tempfile
from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd
from core.concept_image_generation import PROFILE, LEGACY_PROFILE, validate_bundle, fit_vectors, average_trials, image_group_split, image_retrieval_metrics
from core.concept_image_generation import (generate_batch, batch_fingerprint, load_trial_metadata,
                                           trial_caption, block_concept_for_index)


class GenerationTests(unittest.TestCase):
    def batch_bundle(self, root):
        for name, color in [('one.png', 'blue'), ('two.png', 'green'), ('three.png', 'yellow')]:
            Image.new('RGB', (100, 80), color).save(root / name)
        return dict(generation_profile=LEGACY_PROFILE, predictions=np.ones((3, 768)),
                    prototypes=np.ones((2, 768)), test_labels=['a', 'a', 'b'],
                    class_names=['a', 'b'], test_members=[[2, 0], [1], [2]],
                    stimuli=['one.png', 'two.png', 'three.png'], image_root=str(root),
                    block_types={'2': 'ANIMAL', '3': 'VEHICLE', '4': 'INSTRUMENT'},
                    block_exemplars={'2': ['one.png'] * 3, '3': ['two.png'] * 3, '4': ['three.png'] * 3},
                    block_concept_vectors={str(n): np.full(768, n, dtype=float) for n in [2, 3, 4]},
                    block_concept_exemplar_sha256={}, block_concept_definition='test block vector',
                    trial_metadata=[dict(stimulus=name, trial_index_in_file=n, session_id=1,
                                         block_id=block, source_fif=f'run{block}.fif')
                                    for name, n, block in [('one.png', 51, 2), ('two.png', 9, 3), ('three.png', 18, 4)]])

    def test_batch_cancel_resume_corruption_and_input_identity(self):
        calls = []
        def render(v):
            calls.append(np.asarray(v).copy())
            return Image.new('RGB', (512, 512), 'red'), [False]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); bundle = self.batch_bundle(root); out = root / 'images_all_comparisons'
            state = generate_batch(bundle, out, render, cancelled=lambda: len(calls) >= 1, progress=lambda s: None)
            self.assertEqual(state['status'], 'cancelled')
            self.assertEqual(len(calls), 2)
            np.testing.assert_allclose(calls[0], np.ones(768))
            np.testing.assert_allclose(calls[1], np.full(768, -2.0))
            block_vector, metadata = block_concept_for_index(bundle, 0)
            np.testing.assert_allclose(block_vector, np.full(768, 3.0))
            self.assertEqual(metadata['block_ids'], ['4', '2'])
            rows = state['records']['meg_0']['input_trials']
            self.assertEqual([r['trial_index_in_file'] for r in rows], [18, 51])
            self.assertEqual([r['block_id'] for r in rows], [4, 2])
            with Image.open(out / 'test_00000.png') as panel:
                self.assertEqual(panel.width, 1596)
                self.assertEqual(panel.getpixel((798, panel.height - 274)), (255, 0, 0))
                colors = {rgb for count, rgb in panel.getcolors(panel.width * panel.height)}
                self.assertIn((255, 255, 0), colors)
                self.assertIn((0, 0, 255), colors)
            state = generate_batch(bundle, out, render, progress=lambda s: None)
            self.assertEqual(state['status'], 'complete')
            self.assertEqual(len(calls), 6)
            generate_batch(bundle, out, render, progress=lambda s: None)
            self.assertEqual(len(calls), 6)
            (out / 'test_00000.png').write_bytes(b'corrupt')
            generate_batch(bundle, out, render, progress=lambda s: None)
            self.assertEqual(len(calls), 8)
            self.assertEqual(len(list(out.glob('*.png'))), 3)
            bundle['trial_metadata'][0]['trial_index_in_file'] = 99
            with self.assertRaisesRegex(ValueError, 'different'):
                generate_batch(bundle, out, render)
            bundle['trial_metadata'][0]['trial_index_in_file'] = 51
            Image.new('RGB', (100, 80), 'black').save(root / 'one.png')
            with self.assertRaisesRegex(ValueError, 'different'):
                generate_batch(bundle, out, render)

    def test_reuse_legacy_predictions_without_rendering(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); bundle = self.batch_bundle(root)
            old = root / 'images_all_with_ideals'; old.mkdir()
            records = {}
            for i in range(3):
                path = old / f'test_{i:05d}.png'
                Image.new('RGB', (512, 512), 'red').save(path)
                records[f'meg_{i}'] = dict(file=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(), nsfw=[False])
            config = dict(fingerprint=batch_fingerprint(bundle), seed=42, steps=30, guidance=3., profile=bundle['generation_profile'], include_ideal=True)
            (old / 'batch_manifest.json').write_text(json.dumps(dict(config=config, records=records)))
            calls = []
            def render(v):
                calls.append(np.asarray(v))
                return Image.new('RGB', (512, 512), 'purple'), [False]
            state = generate_batch(bundle, root / 'images_all_comparisons', render, progress=lambda s: None)
            self.assertEqual(state['status'], 'complete')
            self.assertEqual(len(state['records']), 3)
            self.assertEqual(len(calls), 3)

    def test_metadata_sidecar_and_missing_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); bundle = self.batch_bundle(root)
            metadata = bundle.pop('trial_metadata')
            self.assertEqual(trial_caption({}), 'unavailable')
            pd.DataFrame(metadata).to_csv(root / 'trial_metadata.csv', index=False)
            load_trial_metadata(bundle, root / 'inference_bundle.joblib')
            self.assertEqual(bundle['trial_metadata'], metadata)
            bundle.pop('trial_metadata')
            metadata.reverse()
            pd.DataFrame(metadata).to_csv(root / 'trial_metadata.csv', index=False)
            with self.assertRaisesRegex(ValueError, 'stimulus sequence'):
                load_trial_metadata(bundle, root / 'inference_bundle.joblib')
            (root / 'one.png').unlink()
            with self.assertRaises(FileNotFoundError):
                generate_batch(bundle, root / 'output', lambda v: self.fail('Missing inputs must fail before generation'))

    def test_block_types_lookup_and_refresh_preserve_prediction(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); bundle = self.batch_bundle(root)
            out = root / 'images_all_comparisons'
            calls = []
            def render(v):
                calls.append(1)
                return Image.new('RGB', (512, 512), 'red'), [False]
            generate_batch(bundle, out, render, progress=lambda s: None)
            # Exercise old comparisons without explicit crop coordinates too.
            manifest = json.loads((out / 'batch_manifest.json').read_text())
            manifest['records']['meg_0'].pop('prediction_box')
            (out / 'batch_manifest.json').write_text(json.dumps(manifest))
            (root / 'block-instructions.csv').write_text(
                'block_id,line_2,image_1,image_2,image_3\n'
                '4,MUSICAL INSTRUMENT,three.png,three.png,three.png\n'
                '2,ANIMAL or INSECT,one.png,one.png,one.png\n'
                '3,VEHICLE,two.png,two.png,two.png\n')
            load_trial_metadata(bundle, root / 'inference_bundle.joblib')
            state = generate_batch(bundle, out, render, progress=lambda s: None)
            self.assertEqual(len(calls), 9)
            rows = state['records']['meg_0']['input_trials']
            self.assertEqual([r['block_type'] for r in rows], ['MUSICAL INSTRUMENT', 'ANIMAL or INSECT'])
            self.assertEqual(trial_caption(rows[0]), 'S1 / block 4 (MUSICAL INSTRUMENT): 18')
            for record in state['records'].values():
                with Image.open(out / record['file']) as image:
                    prediction = image.crop(record['prediction_box'])
                    self.assertEqual(prediction.size, (512, 512))
                    self.assertEqual(prediction.getextrema(), ((255, 255), (0, 0), (0, 0)))
                    residual = image.crop(record['block_subtracted_box'])
                    self.assertEqual(residual.size, (512, 512))
            generate_batch(bundle, out, render, progress=lambda s: None)
            self.assertEqual(len(calls), 9)
            (root / 'block-instructions.csv').write_text(
                'block_id,line_2,image_1,image_2,image_3\n'
                '4,NEW LABEL,three.png,three.png,three.png\n'
                '4,CONFLICT,three.png,three.png,three.png\n')
            with self.assertRaisesRegex(ValueError, 'Conflicting'):
                load_trial_metadata(bundle, root / 'inference_bundle.joblib')

    def test_incompatible_and_nonfinite_rejected(self):
        with self.assertRaisesRegex(ValueError,'Incompatible'):
            validate_bundle({})
        for arr in [np.zeros((2,40)),np.full((2,768),np.nan)]:
            with self.assertRaises(ValueError):
                validate_bundle(dict(generation_profile=PROFILE,predictions=arr,prototypes=arr))

    def test_averages_do_not_cross_partition(self):
        x=np.arange(12)[:,None];y=np.array(['a']*6+['b']*6)
        a,labels,members=average_trials(x,y,np.array([0,1,2,6,7]),3,42)
        self.assertEqual(sorted(sum(members,[])),[0,1,2,6,7])
        self.assertEqual(sorted(map(len,members)),[2,3])

    def test_train_only_targets_and_bundle(self):
        rng=np.random.default_rng(1);y=np.repeat(['a','b','c'],40)
        x=np.eye(3)[np.repeat(np.arange(3),40)]+rng.normal(0,.1,(120,3))
        names=[f'image_{i}.png' for i in range(120)];paths=['/images/'+s for s in names]
        vectors=rng.normal(size=(120,768))
        p=dict(X=x,y=y,trial_df=pd.DataFrame({'stimulus':names, 'trial_index_in_file':np.arange(1,121), 'block_id':1}))
        o=dict(generation_image_root='/images',validation_fraction=.2,transformer_trials_per_average=3)
        b,r=fit_vectors(p,o,vectors,paths)
        validate_bundle(b)
        self.assertEqual(b['trial_metadata'][119]['trial_index_in_file'], 120)
        self.assertEqual([r['stimulus'] for r in b['trial_metadata']], names)
        train=b['train_indices']
        for i,c in enumerate(b['class_names']):
            np.testing.assert_allclose(b['prototypes'][i],vectors[train[y[train]==c]].mean(0),atol=1e-6)
        self.assertFalse(set(train)&set(b['test_indices']))
        self.assertFalse(set(train)&set(b['validation_indices']))
        self.assertEqual(r['predictions_df'].shape[0],len(b['predictions']))
        # Changing test-only embeddings cannot change training, validation selection or predictions.
        changed=vectors.copy();changed[b['test_indices']]=9999
        b2,_=fit_vectors(p,o,changed,paths)
        np.testing.assert_allclose(b['predictions'],b2['predictions'])
        np.testing.assert_allclose(b['prototypes'],b2['prototypes'])
        self.assertEqual(b['validation_candidates'],b2['validation_candidates'])
        self.assertFalse(np.allclose(b['test_target_vectors'],b2['test_target_vectors']))

    def test_repeated_images_split_and_image_specific_training(self):
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import Ridge
        rng = np.random.default_rng(77)
        # 12 images per concept, 3 presentations each; one duplicate under a different name.
        labels = np.repeat(['a', 'b'], 36)
        ids = np.repeat(np.arange(24), 3)
        names = [f'image_{i}.png' for i in ids]
        names[1] = 'duplicate.png'
        paths = [f'/images/image_{i}.png' for i in range(24)] + ['/images/duplicate.png']
        vectors = rng.normal(size=(25, 768)); vectors[-1] = vectors[0]
        hashes = {path: str(i) for i, path in enumerate(paths)}; hashes[paths[-1]] = '0'
        x = rng.normal(size=(72, 7))
        payload = dict(X=x, y=labels, trial_df=pd.DataFrame({'stimulus': names}))
        options = dict(generation_image_root='/images', transformer_trials_per_average=2)
        b, r = fit_vectors(payload, options, vectors, paths, image_hashes=hashes)
        split_ids = [set(np.asarray(b['image_ids'])[b[k]]) for k in ['train_indices', 'validation_indices', 'test_indices']]
        self.assertFalse(split_ids[0] & split_ids[1] or split_ids[0] & split_ids[2] or split_ids[1] & split_ids[2])
        self.assertEqual(set.union(*split_ids), set(map(str, range(24))))
        for part in ['train_members', 'validation_members', 'test_members']:
            for members in b[part]:
                self.assertEqual(len({b['image_ids'][i] for i in members}), 1)
                self.assertLessEqual(len(members), 2)
        lookup = dict(zip(paths, vectors))
        target = lambda members: lookup['/images/' + names[members[0]]]
        train_x = np.stack([x[m].mean(0) for m in b['train_members']])
        train_y = np.stack([target(m) for m in b['train_members']])
        # Independent fit to individual image targets must match the implementation.
        expected = make_pipeline(StandardScaler(), Ridge(alpha=r['ridge_alpha'], solver='svd')).fit(train_x, train_y)
        test_x = np.stack([x[m].mean(0) for m in b['test_members']])
        np.testing.assert_allclose(expected.predict(test_x), b['predictions'])
        np.testing.assert_allclose(b['test_target_vectors'], np.stack([target(m) for m in b['test_members']]))
        np.testing.assert_allclose(b['training_target_mean'], train_y.mean(0))
        self.assertEqual(b['generation_profile'], PROFILE)
        self.assertIn('within_concept_image_rank', r['predictions_df'])
        self.assertEqual(r['n_test_images'], len(split_ids[2]))

    def test_image_split_rejects_insufficient_and_conflicting_images(self):
        with self.assertRaisesRegex(ValueError, '3 distinct images'):
            image_group_split(['a'] * 6, ['one'] * 3 + ['two'] * 3)
        with self.assertRaisesRegex(ValueError, 'conflicting'):
            image_group_split(['a', 'b'], ['same', 'same'])
        for fraction in [0, 1, -1]:
            with self.assertRaises(ValueError): image_group_split(['a']*3, ['1','2','3'], fraction)

    def test_image_retrieval_same_concept_and_ties(self):
        targets = np.eye(4)
        metrics, ranks, within, counts = image_retrieval_metrics(targets, targets, ['a1','a2','b1','b2'], ['a','a','b','b'])
        self.assertEqual(metrics['image_top1_accuracy'], 1)
        self.assertEqual(metrics['within_concept_image_top1_accuracy'], 1)
        self.assertEqual(metrics['within_concept_image_chance'], .5)
        self.assertEqual(counts, [2]*4)
        metrics, ranks, within, _ = image_retrieval_metrics(np.ones((4,4)), targets, ['a1','a2','b1','b2'], ['a','a','b','b'])
        self.assertEqual(metrics['image_top1_accuracy'], .25)
        self.assertEqual(metrics['within_concept_image_top1_accuracy'], .5)
        self.assertEqual(ranks, [2.5]*4)
        self.assertEqual(within, [1.5]*4)
        metrics, _, _, _ = image_retrieval_metrics(targets[:2], targets[:2], ['a1','b1'], ['a','b'])
        self.assertIsNone(metrics['within_concept_image_top1_accuracy'])

    def test_legacy_profile_and_fingerprint_remain_compatible(self):
        with tempfile.TemporaryDirectory() as tmp:
            b = self.batch_bundle(Path(tmp)); validate_bundle(b)
            identity = hashlib.sha256()
            for key in ['predictions', 'prototypes']:
                a = np.ascontiguousarray(b[key]);identity.update(str((a.shape,a.dtype)).encode());identity.update(a.tobytes())
            identity.update(json.dumps([LEGACY_PROFILE, list(map(str,b['test_labels'])),b['class_names'],b['test_members']],sort_keys=True).encode())
            self.assertEqual(batch_fingerprint(b), identity.hexdigest())

    def test_target_svd_dimensionality_reduction(self):
        rng = np.random.default_rng(99)
        labels = np.repeat(['a', 'b'], 36)
        ids = np.repeat(np.arange(24), 3)
        names = [f'img_{i}.png' for i in ids]
        paths = [f'/images/img_{i}.png' for i in range(24)]
        vectors = rng.normal(size=(24, 768))
        hashes = {path: str(i) for i, path in enumerate(paths)}
        x = rng.normal(size=(72, 8))
        payload = dict(X=x, y=labels, trial_df=pd.DataFrame({'stimulus': names}))
        options = dict(generation_image_root='/images', transformer_trials_per_average=2,
                       clip_target_svd_components=16)
        b, r = fit_vectors(payload, options, vectors, paths, image_hashes=hashes)
        validate_bundle(b)
        self.assertGreater(b['target_svd_components'], 0)
        self.assertLessEqual(b['target_svd_components'], 16)
        self.assertEqual(b['target_svd_projection'].shape, (768, b['target_svd_components']))
        self.assertEqual(b['predictions'].shape, (len(b['test_members']), 768))
        self.assertTrue(np.isfinite(b['predictions']).all())
        self.assertEqual(r['target_svd_components'], b['target_svd_components'])
        self.assertGreater(r['target_svd_explained_variance'], 0.0)
        self.assertLessEqual(r['target_svd_explained_variance'], 1.0)
        self.assertIn(f"SVD={b['target_svd_components']}", r['decoder'])

    def test_two_stage_hybrid_decoding(self):
        rng = np.random.default_rng(101)
        labels = np.repeat(['cat', 'dog'], 36)
        ids = np.repeat(np.arange(24), 3)
        names = [f'stim_{i}.png' for i in ids]
        paths = [f'/images/stim_{i}.png' for i in range(24)]
        vectors = rng.normal(size=(24, 768))
        hashes = {path: str(i) for i, path in enumerate(paths)}
        x = rng.normal(size=(72, 6))
        payload = dict(X=x, y=labels, trial_df=pd.DataFrame({'stimulus': names}))
        options = dict(generation_image_root='/images', transformer_trials_per_average=2,
                       decoder_type='clip_generation_hybrid')
        b, r = fit_vectors(payload, options, vectors, paths, image_hashes=hashes)
        validate_bundle(b)
        self.assertTrue(b['hybrid_decoding'])
        self.assertIsNotNone(b['stage1_model'])
        self.assertGreaterEqual(r['hybrid_residual_weight'], 0.0)
        self.assertLessEqual(r['hybrid_residual_weight'], 1.0)
        self.assertIn('stage1_validation_accuracy', r)
        self.assertIn('stage1_test_accuracy', r)
        self.assertEqual(b['predictions'].shape, (len(b['test_members']), 768))
        self.assertTrue(np.isfinite(b['predictions']).all())

    def test_target_svd_with_hybrid_decoding(self):
        rng = np.random.default_rng(202)
        labels = np.repeat(['car', 'bus'], 36)
        ids = np.repeat(np.arange(24), 3)
        names = [f'veh_{i}.png' for i in ids]
        paths = [f'/images/veh_{i}.png' for i in range(24)]
        vectors = rng.normal(size=(24, 768))
        hashes = {path: str(i) for i, path in enumerate(paths)}
        x = rng.normal(size=(72, 6))
        payload = dict(X=x, y=labels, trial_df=pd.DataFrame({'stimulus': names}))
        options = dict(generation_image_root='/images', transformer_trials_per_average=2,
                       clip_hybrid_decoding=True, clip_target_svd_components=8,
                       clip_hybrid_weight=0.5)
        b, r = fit_vectors(payload, options, vectors, paths, image_hashes=hashes)
        validate_bundle(b)
        self.assertTrue(b['hybrid_decoding'])
        self.assertEqual(b['target_svd_components'], 8)
        self.assertEqual(r['hybrid_residual_weight'], 0.5)
        self.assertIn('SVD=8', r['decoder'])
        self.assertEqual(b['predictions'].shape, (len(b['test_members']), 768))
        self.assertTrue(np.isfinite(b['predictions']).all())


if __name__=='__main__':unittest.main()
