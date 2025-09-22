"""Metrics computation for model evaluation."""

import logging
from typing import Any, Dict, List, Optional

import evaluate
import numpy as np
import torch
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_score import rouge_scorer
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

from src.utils.common import decode_tokens

logger = logging.getLogger(__name__)


def compute_metrics(eval_pred, tokenizer=None) -> Dict[str, float]:
    """Compute metrics for evaluation.

    Args:
        eval_pred: EvalPrediction object from transformers
        tokenizer: Optional tokenizer for decoding

    Returns:
        Dictionary of computed metrics
    """
    predictions, labels = eval_pred

    if isinstance(predictions, tuple):
        predictions = predictions[0]

    metrics = {}
    metrics.update(compute_perplexity(predictions, labels))

    # Only compute generation metrics if tokenizer is available
    if tokenizer:
        decoded_preds = decode_tokens(predictions, tokenizer, clean_padding=True)
        decoded_labels = decode_tokens(labels, tokenizer, clean_padding=True)

        if decoded_preds and decoded_labels:
            # Ensure lists
            if isinstance(decoded_preds, str):
                decoded_preds = [decoded_preds]
            if isinstance(decoded_labels, str):
                decoded_labels = [decoded_labels]

            metrics.update(compute_rouge_scores(decoded_preds, decoded_labels))
            metrics.update(compute_bleu_scores(decoded_preds, decoded_labels))

    return metrics


def compute_perplexity(predictions: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Compute perplexity metric.

    Args:
        predictions: Model predictions
        labels: Ground truth labels

    Returns:
        Dictionary with perplexity score
    """
    try:
        loss = np.mean(predictions)
        perplexity = np.exp(loss)
        return {"perplexity": float(perplexity)}
    except Exception as e:
        logger.warning(f"Failed to compute perplexity: {e}")
        return {"perplexity": float("inf")}


def compute_rouge_scores(
    predictions: List[str],
    references: List[str],
) -> Dict[str, float]:
    """Compute ROUGE scores.

    Args:
        predictions: List of predicted texts
        references: List of reference texts

    Returns:
        Dictionary of ROUGE scores
    """
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=True,
    )

    scores = {
        "rouge1_f": [],
        "rouge1_p": [],
        "rouge1_r": [],
        "rouge2_f": [],
        "rouge2_p": [],
        "rouge2_r": [],
        "rougeL_f": [],
        "rougeL_p": [],
        "rougeL_r": [],
    }

    for pred, ref in zip(predictions, references):
        result = scorer.score(ref, pred)

        for key in ["rouge1", "rouge2", "rougeL"]:
            scores[f"{key}_f"].append(result[key].fmeasure)
            scores[f"{key}_p"].append(result[key].precision)
            scores[f"{key}_r"].append(result[key].recall)

    return {k: np.mean(v) for k, v in scores.items()}


def compute_bleu_scores(
    predictions: List[str],
    references: List[str],
) -> Dict[str, float]:
    """Compute BLEU scores.

    Args:
        predictions: List of predicted texts
        references: List of reference texts

    Returns:
        Dictionary of BLEU scores
    """
    smoothing = SmoothingFunction().method1

    bleu_scores = []
    for pred, ref in zip(predictions, references):
        pred_tokens = pred.split()
        ref_tokens = ref.split()

        if len(pred_tokens) > 0 and len(ref_tokens) > 0:
            score = sentence_bleu(
                [ref_tokens],
                pred_tokens,
                smoothing_function=smoothing,
            )
            bleu_scores.append(score)

    return {"bleu": np.mean(bleu_scores) if bleu_scores else 0.0}


def compute_classification_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, float]:
    """Compute classification metrics.

    Args:
        predictions: Model predictions
        labels: Ground truth labels

    Returns:
        Dictionary of classification metrics
    """
    if len(predictions.shape) > 1:
        predictions = np.argmax(predictions, axis=-1)

    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="weighted",
        zero_division=0,
    )

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


class MetricsCalculator:
    """Advanced metrics calculator with custom metrics support."""

    def __init__(self, tokenizer=None, metrics_config=None):
        """Initialize metrics calculator.

        Args:
            tokenizer: Tokenizer for decoding
            metrics_config: Configuration for metrics
        """
        self.tokenizer = tokenizer
        self.metrics_config = metrics_config or {}
        self.metrics_cache = {}

    def compute(self, eval_pred) -> Dict[str, float]:
        """Compute all configured metrics.

        Args:
            eval_pred: EvalPrediction object

        Returns:
            Dictionary of computed metrics
        """
        predictions, labels = eval_pred

        if isinstance(predictions, tuple):
            predictions = predictions[0]

        metrics = {}

        if self.metrics_config.get("perplexity", True):
            metrics.update(self._compute_perplexity(predictions, labels))

        if self.tokenizer and self.metrics_config.get("generation_metrics", True):
            decoded_preds = self._decode(predictions)
            decoded_labels = self._decode(labels)

            if self.metrics_config.get("rouge", True):
                metrics.update(compute_rouge_scores(decoded_preds, decoded_labels))

            if self.metrics_config.get("bleu", True):
                metrics.update(compute_bleu_scores(decoded_preds, decoded_labels))

        if self.metrics_config.get("classification_metrics", False):
            metrics.update(compute_classification_metrics(predictions, labels))

        for custom_metric in self.metrics_config.get("custom_metrics", []):
            if hasattr(self, f"compute_{custom_metric}"):
                metric_fn = getattr(self, f"compute_{custom_metric}")
                metrics.update(metric_fn(predictions, labels))

        return metrics

    def _compute_perplexity(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
    ) -> Dict[str, float]:
        """Compute perplexity with proper handling."""
        try:
            shift_logits = torch.from_numpy(predictions[..., :-1, :])
            shift_labels = torch.from_numpy(labels[..., 1:])

            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1).long(),
            )

            perplexity = torch.exp(loss).item()
            return {"perplexity": perplexity}
        except Exception as e:
            logger.warning(f"Perplexity computation failed: {e}")
            return {"perplexity": float("inf")}

    def _decode(self, token_ids: np.ndarray) -> List[str]:
        """Decode token IDs to text.

        Args:
            token_ids: Token IDs to decode

        Returns:
            List of decoded texts
        """
        if self.tokenizer is None:
            return []

        decoded_texts = []
        for ids in token_ids:
            ids = ids[ids != -100]
            text = self.tokenizer.decode(ids, skip_special_tokens=True)
            decoded_texts.append(text)

        return decoded_texts

    def compute_diversity(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
    ) -> Dict[str, float]:
        """Compute diversity metrics for generated text."""
        if self.tokenizer is None:
            return {}

        decoded = self._decode(predictions)

        unique_ngrams = {1: set(), 2: set(), 3: set()}
        total_ngrams = {1: 0, 2: 0, 3: 0}

        for text in decoded:
            tokens = text.split()
            for n in [1, 2, 3]:
                for i in range(len(tokens) - n + 1):
                    ngram = tuple(tokens[i : i + n])
                    unique_ngrams[n].add(ngram)
                    total_ngrams[n] += 1

        diversity_scores = {}
        for n in [1, 2, 3]:
            if total_ngrams[n] > 0:
                diversity_scores[f"distinct_{n}"] = (
                    len(unique_ngrams[n]) / total_ngrams[n]
                )
            else:
                diversity_scores[f"distinct_{n}"] = 0.0

        return diversity_scores
