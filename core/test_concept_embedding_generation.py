import json
import gzip
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from scripts.create_concept_embedding_dictionary import (
    _torch_device,
    create_clip_text_embeddings,
    create_image_embeddings,
    create_word2vec_embeddings,
    resolve_stimulus_image_paths,
    write_output,
)


class ConceptEmbeddingGenerationTests(unittest.TestCase):
    def setUp(self):
        self.vectors = {
            "polar_bear": np.asarray([9.0, 8.0, 7.0], dtype=np.float32),
            "red": np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
            "car": np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
            "known": np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
        }

    def test_exact_phrase_is_preferred_over_token_average(self):
        names, embeddings, metadata = create_word2vec_embeddings(
            ["Polar Bear"],
            model_path="fixture.kv",
            aggregation="exact_or_mean",
            keyed_vectors=self.vectors,
        )

        self.assertEqual(names, ["Polar Bear"])
        np.testing.assert_allclose(embeddings[0], [9.0, 8.0, 7.0])
        self.assertEqual(metadata["matched_keys"]["Polar Bear"], ["polar_bear"])

    def test_mean_uses_available_tokens_and_records_partial_coverage(self):
        names, embeddings, metadata = create_word2vec_embeddings(
            ["red car", "known mystery"],
            model_path="fixture.vec",
            aggregation="mean",
            keyed_vectors=self.vectors,
        )

        self.assertEqual(names, ["red car", "known mystery"])
        np.testing.assert_allclose(embeddings[0], [0.5, 0.5, 0.0])
        np.testing.assert_allclose(embeddings[1], [0.0, 0.0, 1.0])
        self.assertEqual(metadata["partial_labels"], {"known mystery": ["mystery"]})

    def test_oov_error_names_unavailable_labels(self):
        with self.assertRaisesRegex(ValueError, "missing stimulus"):
            create_word2vec_embeddings(
                ["red car", "missing stimulus"],
                model_path="fixture.bin",
                oov_policy="error",
                keyed_vectors=self.vectors,
            )

    def test_oov_skip_writes_decoder_compatible_json_and_metadata(self):
        requested = ["red car", "missing stimulus"]
        names, embeddings, metadata = create_word2vec_embeddings(
            requested,
            model_path="fixture.bin",
            oov_policy="skip",
            keyed_vectors=self.vectors,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "embeddings.json"
            args = SimpleNamespace(
                provider="word2vec",
                model=metadata["model"],
                input="stimulus_summary",
                column="stimulus",
            )
            write_output(output_path, names, embeddings, args, metadata=metadata)
            payload = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(payload["names"], ["red car"])
        self.assertEqual(len(payload["embeddings"]), 1)
        self.assertEqual(payload["metadata"]["provider"], "word2vec")
        self.assertEqual(payload["metadata"]["dimension"], 3)
        self.assertEqual(payload["metadata"]["requested_labels"], 2)
        self.assertEqual(payload["metadata"]["embedded_labels"], 1)
        self.assertEqual(payload["metadata"]["missing_labels"], ["missing stimulus"])

    def test_standard_word2vec_text_file_works_without_native_keyedvectors(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = Path(tmp_dir) / "tiny.vec"
            model_path.write_text(
                "3 3\nred 1 0 0\ncar 0 1 0\nunused 0 0 1\n",
                encoding="utf-8",
            )
            names, embeddings, metadata = create_word2vec_embeddings(
                ["red car"],
                model_path=str(model_path),
                aggregation="mean",
            )

        self.assertEqual(names, ["red car"])
        np.testing.assert_allclose(embeddings[0], [0.5, 0.5, 0.0])
        self.assertEqual(metadata["embedded_labels"], 1)

    def test_standard_word2vec_binary_file_is_supported(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = Path(tmp_dir) / "tiny.bin"
            payload = bytearray(b"2 3\n")
            payload.extend(b"red ")
            payload.extend(np.asarray([1.0, 0.0, 0.0], dtype="<f4").tobytes())
            payload.extend(b"\ncar ")
            payload.extend(np.asarray([0.0, 1.0, 0.0], dtype="<f4").tobytes())
            payload.extend(b"\n")
            model_path.write_bytes(bytes(payload))
            names, embeddings, _ = create_word2vec_embeddings(
                ["red car"],
                model_path=str(model_path),
                aggregation="mean",
            )

        self.assertEqual(names, ["red car"])
        np.testing.assert_allclose(embeddings[0], [0.5, 0.5, 0.0])

    def test_gensim_google_news_gz_filename_is_detected_as_binary(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = Path(tmp_dir) / "word2vec-google-news-300.gz"
            payload = bytearray(b"2 3\n")
            payload.extend(b"red ")
            payload.extend(np.asarray([1.0, 0.0, 0.0], dtype="<f4").tobytes())
            payload.extend(b"\ncar ")
            payload.extend(np.asarray([0.0, 1.0, 0.0], dtype="<f4").tobytes())
            payload.extend(b"\n")
            with gzip.open(model_path, "wb") as stream:
                stream.write(bytes(payload))
            names, embeddings, _ = create_word2vec_embeddings(
                ["red car"],
                model_path=str(model_path),
                aggregation="mean",
            )

        self.assertEqual(names, ["red car"])
        np.testing.assert_allclose(embeddings[0], [0.5, 0.5, 0.0])

    def test_image_resolution_rejects_ambiguous_basenames(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "set_a").mkdir()
            (root / "set_b").mkdir()
            (root / "set_a" / "cat.png").write_bytes(b"fixture")
            (root / "set_b" / "cat.png").write_bytes(b"fixture")
            (root / "set_a" / "dog.png").write_bytes(b"fixture")
            resolved, metadata = resolve_stimulus_image_paths(
                ["cat", "dog"],
                {"cat": ["cat.png"], "dog": ["dog.png"]},
                str(root),
            )

        self.assertNotIn("cat", resolved)
        self.assertEqual(len(resolved["dog"]), 1)
        self.assertEqual(metadata["missing_labels"], ["cat"])
        self.assertIn("cat", metadata["ambiguous_image_references"])

    def test_image_embeddings_average_normalized_variant_prototypes(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            for filename in ("bear_1.png", "bear_2.png", "car.png"):
                (root / filename).write_bytes(b"fixture")

            vectors_by_name = {
                "bear_1.png": [1.0, 0.0, 0.0],
                "bear_2.png": [0.0, 1.0, 0.0],
                "car.png": [0.0, 0.0, 2.0],
            }

            def extract(paths):
                return np.asarray(
                    [vectors_by_name[Path(path).name] for path in paths], dtype=np.float32
                )

            names, embeddings, metadata = create_image_embeddings(
                ["bear", "car"],
                image_references_by_label={
                    "bear": ["bear_1.png", "bear_2.png"],
                    "car": ["car.png"],
                },
                image_root=str(root),
                provider="clip_image",
                model_name="fixture-clip",
                feature_extractor=extract,
            )

        self.assertEqual(names, ["bear", "car"])
        np.testing.assert_allclose(
            embeddings[0], [2 ** -0.5, 2 ** -0.5, 0.0], atol=1e-6
        )
        np.testing.assert_allclose(embeddings[1], [0.0, 0.0, 1.0])
        self.assertEqual(metadata["resolved_image_count"], 3)
        self.assertEqual(
            metadata["prototype_aggregation"],
            "mean normalized image vectors, then L2 normalize",
        )

    def test_clip_text_applies_prompt_and_normalizes_features(self):
        captured = []

        def extract(prompts):
            captured.extend(prompts)
            return np.asarray([[3.0, 4.0], [0.0, 2.0]], dtype=np.float32)

        names, embeddings, metadata = create_clip_text_embeddings(
            ["polar bear", "car"],
            model_name="fixture-clip",
            prompt_template="a photo of a {label}",
            feature_extractor=extract,
        )

        self.assertEqual(names, ["polar bear", "car"])
        self.assertEqual(captured, ["a photo of a polar bear", "a photo of a car"])
        np.testing.assert_allclose(embeddings[0], [0.6, 0.8])
        np.testing.assert_allclose(embeddings[1], [0.0, 1.0])
        self.assertTrue(metadata["normalized"])

    def test_explicit_unavailable_torch_device_has_clear_error(self):
        unavailable = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: False),
            backends=SimpleNamespace(
                mps=SimpleNamespace(is_available=lambda: False)
            ),
        )

        self.assertEqual(_torch_device("auto", unavailable), "cpu")
        with self.assertRaisesRegex(ValueError, "CUDA was selected"):
            _torch_device("cuda", unavailable)
        with self.assertRaisesRegex(ValueError, "Metal.*not available"):
            _torch_device("mps", unavailable)


if __name__ == "__main__":
    unittest.main()
