import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from core.concept_embedding_residualization import (
    TargetEmbeddingResidualizer,
    residualize_embedding_dictionary,
    resolve_concept_confounds,
)


class TargetEmbeddingResidualizationTests(unittest.TestCase):
    def setUp(self):
        # 4 concepts in 2 categories:
        # Category 0 (Vehicles): airplane, helicopter
        # Category 1 (Animals): dog, cat
        self.category_signals = np.array([
            [5.0, 5.0, 0.0, 0.0],  # Vehicle centroid direction
            [0.0, 0.0, 5.0, 5.0],  # Animal centroid direction
        ], dtype=np.float32)

        self.raw_embeddings = np.array([
            # airplane: vehicle + [1.0, 0.0, 0.0, 0.0]
            [6.0, 5.0, 0.0, 0.0],
            # helicopter: vehicle + [-1.0, 0.0, 0.0, 0.0]
            [4.0, 5.0, 0.0, 0.0],
            # dog: animal + [0.0, 0.0, 1.0, 0.0]
            [0.0, 0.0, 6.0, 5.0],
            # cat: animal + [0.0, 0.0, -1.0, 0.0]
            [0.0, 0.0, 4.0, 5.0],
        ], dtype=np.float32)

        self.concept_names = ["airplane", "helicopter", "dog", "cat"]
        self.categories = ["vehicle", "vehicle", "animal", "animal"]

    def test_residualization_removes_category_centroid_and_preserves_differences(self):
        res = TargetEmbeddingResidualizer(include_intercept=True, l2_normalize_output=False)
        residuals = res.fit_transform(self.raw_embeddings, self.categories)

        # Vehicle centroid: [5.0, 5.0, 0.0, 0.0]
        # airplane residual: [1.0, 0.0, 0.0, 0.0]
        # helicopter residual: [-1.0, 0.0, 0.0, 0.0]
        np.testing.assert_allclose(residuals[0], [1.0, 0.0, 0.0, 0.0], atol=1e-5)
        np.testing.assert_allclose(residuals[1], [-1.0, 0.0, 0.0, 0.0], atol=1e-5)

        # Category sum of residuals must be zero (orthogonal to category indicator)
        np.testing.assert_allclose(residuals[0] + residuals[1], np.zeros(4), atol=1e-5)
        np.testing.assert_allclose(residuals[2] + residuals[3], np.zeros(4), atol=1e-5)

        self.assertGreater(res.variance_explained_, 0.8)

    def test_l2_normalization_returns_unit_length_vectors(self):
        res = TargetEmbeddingResidualizer(include_intercept=True, l2_normalize_output=True)
        residuals = res.fit_transform(self.raw_embeddings, self.categories)

        norms = np.linalg.norm(residuals, axis=1)
        np.testing.assert_allclose(norms, np.ones(4), atol=1e-5)

        # airplane and helicopter are opposite directions in the residual subspace
        cosine_sim = np.dot(residuals[0], residuals[1])
        self.assertAlmostEqual(cosine_sim, -1.0, places=4)

    def test_continuous_confound_projection(self):
        # Confound given as continuous 2D instruction vectors
        confound_vectors = np.array([
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ], dtype=np.float32)

        res = TargetEmbeddingResidualizer(include_intercept=False, l2_normalize_output=False)
        residuals = res.fit_transform(self.raw_embeddings, confound_vectors)

        # Residuals must be orthogonal to the confound columns
        self.assertAlmostEqual(np.dot(residuals[:, 0], confound_vectors[:, 0]), 0.0, places=4)
        self.assertAlmostEqual(np.dot(residuals[:, 1], confound_vectors[:, 0]), 0.0, places=4)

    def test_residualize_embedding_dictionary_file(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            in_path = Path(tmp_dir) / "test_embeddings.json"
            out_path = Path(tmp_dir) / "test_residualized.json"

            in_path.write_text(json.dumps({
                "names": self.concept_names,
                "embeddings": self.raw_embeddings.tolist(),
                "metadata": {"test": True},
            }))

            mapping = dict(zip(self.concept_names, self.categories))
            result = residualize_embedding_dictionary(in_path, mapping, out_path)

            self.assertTrue(out_path.exists())
            self.assertEqual(result["names"], self.concept_names)
            self.assertTrue(result["metadata"]["residualized"])
            self.assertGreater(result["metadata"]["residualization_variance_explained"], 0.8)

            saved = json.loads(out_path.read_text())
            self.assertEqual(len(saved["embeddings"]), 4)

    def test_resolve_concept_confounds_from_csv(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "mapping.csv"
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["concept", "category"])
                writer.writerow(["airplane", "vehicle"])
                writer.writerow(["helicopter", "vehicle"])
                writer.writerow(["dog", "animal"])
                writer.writerow(["cat", "animal"])

            mapping = resolve_concept_confounds(
                self.concept_names,
                confound_source="custom_csv",
                category_csv=csv_path,
            )
            self.assertEqual(mapping, dict(zip(self.concept_names, self.categories)))

    def test_resolve_concept_confounds_rejects_single_class(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "single_class.csv"
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["concept", "category"])
                for c in self.concept_names:
                    writer.writerow([c, "all_same"])

            with self.assertRaises(ValueError):
                resolve_concept_confounds(
                    self.concept_names,
                    confound_source="custom_csv",
                    category_csv=csv_path,
                )


if __name__ == "__main__":
    unittest.main()
