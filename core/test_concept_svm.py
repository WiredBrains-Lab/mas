import unittest
import numpy as np
from core.concept_decoding_signal import (
    concept_decoder_spec, fit_concept_decoder,
    concept_feature_importance_from_model, concept_model_architecture_table,
)


class SVMTests(unittest.TestCase):
    def test_multiclass_training_scaling_and_importance(self):
        rng = np.random.default_rng(42)
        y = np.repeat(["a", "b", "c"], 20)
        X = np.eye(3)[np.repeat(np.arange(3), 20)] * 5 + rng.normal(0, .1, (60, 3))
        clf, label, key = concept_decoder_spec("svm")
        self.assertEqual(key, "svm")
        fit_concept_decoder(clf, X, y)
        np.testing.assert_array_equal(clf.predict(X), y)
        np.testing.assert_allclose(clf.named_steps["standardscaler"].mean_, X.mean(axis=0))
        self.assertEqual(concept_feature_importance_from_model(clf, 3).shape, (3,))
        self.assertFalse(hasattr(clf, "predict_proba"))
        self.assertIn("LinearSVC", concept_model_architecture_table({"decoder_type": "svm"}).to_string())

    def test_binary(self):
        clf, _, _ = concept_decoder_spec("svm")
        fit_concept_decoder(clf, np.array([[-2.], [-1.], [1.], [2.]]), ["a", "a", "b", "b"])
        self.assertEqual(clf.predict([[3.]])[0], "b")


if __name__ == "__main__":
    unittest.main()
