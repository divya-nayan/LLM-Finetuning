"""Tests for evaluation metrics."""

import unittest
import numpy as np
from unittest.mock import Mock

from src.evaluation.metrics import (
    compute_perplexity,
    compute_rouge_scores,
    compute_bleu_scores,
    compute_classification_metrics,
    MetricsCalculator
)


class TestMetrics(unittest.TestCase):
    """Test metric computation functions."""

    def test_compute_perplexity(self):
        """Test perplexity computation."""
        predictions = np.array([2.0, 2.5, 3.0])
        labels = np.array([1, 2, 3])

        result = compute_perplexity(predictions, labels)

        self.assertIn("perplexity", result)
        self.assertIsInstance(result["perplexity"], float)
        self.assertGreater(result["perplexity"], 0)

    def test_compute_rouge_scores(self):
        """Test ROUGE score computation."""
        predictions = ["The cat sat on the mat"]
        references = ["The cat was sitting on the mat"]

        scores = compute_rouge_scores(predictions, references)

        self.assertIn("rouge1_f", scores)
        self.assertIn("rouge2_f", scores)
        self.assertIn("rougeL_f", scores)

        for key, value in scores.items():
            self.assertIsInstance(value, float)
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)

    def test_compute_bleu_scores(self):
        """Test BLEU score computation."""
        predictions = ["The cat sat on the mat"]
        references = ["The cat was sitting on the mat"]

        scores = compute_bleu_scores(predictions, references)

        self.assertIn("bleu", scores)
        self.assertIsInstance(scores["bleu"], float)
        self.assertGreaterEqual(scores["bleu"], 0.0)
        self.assertLessEqual(scores["bleu"], 1.0)

    def test_compute_classification_metrics(self):
        """Test classification metrics computation."""
        predictions = np.array([0, 1, 1, 0, 1])
        labels = np.array([0, 1, 0, 0, 1])

        metrics = compute_classification_metrics(predictions, labels)

        self.assertIn("accuracy", metrics)
        self.assertIn("precision", metrics)
        self.assertIn("recall", metrics)
        self.assertIn("f1", metrics)

        for value in metrics.values():
            self.assertIsInstance(value, float)
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)


class TestMetricsCalculator(unittest.TestCase):
    """Test MetricsCalculator class."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_tokenizer = Mock()
        self.mock_tokenizer.decode.return_value = "Test text"

    def test_metrics_calculator_initialization(self):
        """Test MetricsCalculator initialization."""
        config = {"perplexity": True, "rouge": False}
        calculator = MetricsCalculator(
            tokenizer=self.mock_tokenizer,
            metrics_config=config
        )

        self.assertEqual(calculator.metrics_config, config)
        self.assertEqual(calculator.tokenizer, self.mock_tokenizer)

    def test_compute_diversity(self):
        """Test diversity metric computation."""
        calculator = MetricsCalculator(tokenizer=self.mock_tokenizer)

        # Mock predictions
        predictions = np.array([[1, 2, 3, 4, 5]])
        labels = np.array([[1, 2, 3, 4, 5]])

        # Mock decode to return different texts
        self.mock_tokenizer.decode.side_effect = [
            "The cat sat on the mat"
        ]

        diversity = calculator.compute_diversity(predictions, labels)

        self.assertIn("distinct_1", diversity)
        self.assertIn("distinct_2", diversity)
        self.assertIn("distinct_3", diversity)


if __name__ == "__main__":
    unittest.main()