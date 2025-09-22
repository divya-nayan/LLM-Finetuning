"""Evaluation script for fine-tuned models."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import hydra
from omegaconf import DictConfig
from tqdm import tqdm
import pandas as pd
from rich.console import Console
from rich.table import Table

from src.data.data_loader import DataLoader as CustomDataLoader
from src.evaluation.metrics import MetricsCalculator
from src.utils.logger import setup_logger

console = Console()
logger = setup_logger(__name__)


class ModelEvaluator:
    """Evaluator for fine-tuned language models."""

    def __init__(self, config: DictConfig):
        """Initialize evaluator.

        Args:
            config: Configuration object
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.metrics_calculator = None

    def load_model(self):
        """Load fine-tuned model and tokenizer."""
        logger.info(f"Loading model from {self.config.evaluation.model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.evaluation.model_path,
            trust_remote_code=self.config.model.trust_remote_code,
        )

        if self.config.evaluation.use_base_model:
            base_model = AutoModelForCausalLM.from_pretrained(
                self.config.model.name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=self.config.model.trust_remote_code,
            )

            self.model = PeftModel.from_pretrained(
                base_model,
                self.config.evaluation.model_path,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.evaluation.model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=self.config.model.trust_remote_code,
            )

        self.model.eval()
        logger.info("Model loaded successfully")

    def prepare_data(self):
        """Prepare evaluation dataset."""
        logger.info("Loading evaluation dataset")

        data_loader = CustomDataLoader(self.config, self.tokenizer)

        if self.config.evaluation.dataset_path:
            self.eval_dataset = data_loader._load_local_dataset(
                self.config.evaluation.dataset_path
            )
        else:
            self.eval_dataset = data_loader.load_eval_dataset()

        self.data_loader = DataLoader(
            self.eval_dataset,
            batch_size=self.config.evaluation.batch_size,
            shuffle=False,
            num_workers=self.config.evaluation.num_workers,
        )

        logger.info(f"Loaded {len(self.eval_dataset)} evaluation samples")

    def evaluate(self) -> Dict[str, Any]:
        """Run evaluation on the dataset.

        Returns:
            Dictionary of evaluation results
        """
        logger.info("Starting evaluation")

        all_predictions = []
        all_references = []
        all_losses = []

        with torch.no_grad():
            for batch in tqdm(self.data_loader, desc="Evaluating"):
                batch = {k: v.to(self.device) for k, v in batch.items()}

                outputs = self.model(**batch)
                loss = outputs.loss

                if loss is not None:
                    all_losses.append(loss.item())

                if self.config.evaluation.generate_predictions:
                    predictions = self.generate_predictions(batch)
                    references = self.decode_labels(batch["labels"])

                    all_predictions.extend(predictions)
                    all_references.extend(references)

        metrics = self.compute_metrics(
            all_predictions,
            all_references,
            all_losses,
        )

        self.save_results(metrics, all_predictions, all_references)
        self.display_results(metrics)

        return metrics

    def generate_predictions(self, batch: Dict[str, torch.Tensor]) -> List[str]:
        """Generate predictions for a batch.

        Args:
            batch: Input batch

        Returns:
            List of generated texts
        """
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        generated = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=self.config.evaluation.max_new_tokens,
            temperature=self.config.evaluation.temperature,
            top_p=self.config.evaluation.top_p,
            do_sample=self.config.evaluation.do_sample,
            num_beams=self.config.evaluation.num_beams,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        predictions = self.tokenizer.batch_decode(
            generated,
            skip_special_tokens=True,
        )

        return predictions

    def decode_labels(self, labels: torch.Tensor) -> List[str]:
        """Decode label tokens to text.

        Args:
            labels: Label tensor

        Returns:
            List of decoded texts
        """
        labels = labels.cpu().numpy()
        labels[labels == -100] = self.tokenizer.pad_token_id

        references = self.tokenizer.batch_decode(
            labels,
            skip_special_tokens=True,
        )

        return references

    def compute_metrics(
        self,
        predictions: List[str],
        references: List[str],
        losses: List[float],
    ) -> Dict[str, float]:
        """Compute evaluation metrics.

        Args:
            predictions: List of predictions
            references: List of references
            losses: List of losses

        Returns:
            Dictionary of metrics
        """
        metrics = {}

        if losses:
            avg_loss = sum(losses) / len(losses)
            perplexity = torch.exp(torch.tensor(avg_loss)).item()
            metrics["loss"] = avg_loss
            metrics["perplexity"] = perplexity

        if predictions and references:
            from src.evaluation.metrics import (
                compute_rouge_scores,
                compute_bleu_scores,
            )

            metrics.update(compute_rouge_scores(predictions, references))
            metrics.update(compute_bleu_scores(predictions, references))

        return metrics

    def save_results(
        self,
        metrics: Dict[str, float],
        predictions: List[str],
        references: List[str],
    ):
        """Save evaluation results.

        Args:
            metrics: Computed metrics
            predictions: Model predictions
            references: Reference texts
        """
        output_dir = Path(self.config.evaluation.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        if predictions and references:
            results_df = pd.DataFrame({
                "reference": references,
                "prediction": predictions,
            })
            results_df.to_csv(output_dir / "predictions.csv", index=False)

        logger.info(f"Results saved to {output_dir}")

    def display_results(self, metrics: Dict[str, float]):
        """Display evaluation results in a formatted table.

        Args:
            metrics: Dictionary of metrics
        """
        table = Table(title="Evaluation Results")
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="magenta")

        for metric, value in metrics.items():
            if isinstance(value, float):
                table.add_row(metric, f"{value:.4f}")
            else:
                table.add_row(metric, str(value))

        console.print(table)

    def run_inference(self, prompt: str) -> str:
        """Run inference on a single prompt.

        Args:
            prompt: Input prompt

        Returns:
            Generated text
        """
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.data.max_length,
        ).to(self.device)

        with torch.no_grad():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.config.evaluation.max_new_tokens,
                temperature=self.config.evaluation.temperature,
                top_p=self.config.evaluation.top_p,
                do_sample=self.config.evaluation.do_sample,
                num_beams=self.config.evaluation.num_beams,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        output = self.tokenizer.decode(
            generated[0],
            skip_special_tokens=True,
        )

        return output

    def benchmark(self) -> Dict[str, Any]:
        """Run comprehensive benchmarking.

        Returns:
            Benchmark results
        """
        logger.info("Running benchmark")

        benchmarks = {}

        if self.config.evaluation.run_speed_test:
            benchmarks["speed"] = self._benchmark_speed()

        if self.config.evaluation.run_memory_test:
            benchmarks["memory"] = self._benchmark_memory()

        return benchmarks

    def _benchmark_speed(self) -> Dict[str, float]:
        """Benchmark inference speed."""
        import time

        prompt = "The quick brown fox jumps over the lazy dog."
        num_runs = 10

        times = []
        for _ in range(num_runs):
            start = time.time()
            self.run_inference(prompt)
            times.append(time.time() - start)

        return {
            "avg_time": sum(times) / len(times),
            "min_time": min(times),
            "max_time": max(times),
        }

    def _benchmark_memory(self) -> Dict[str, float]:
        """Benchmark memory usage."""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()

        return {
            "rss_mb": memory_info.rss / 1024 / 1024,
            "vms_mb": memory_info.vms / 1024 / 1024,
        }


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    """Main evaluation entry point."""
    evaluator = ModelEvaluator(cfg)
    evaluator.load_model()

    if cfg.evaluation.mode == "dataset":
        evaluator.prepare_data()
        metrics = evaluator.evaluate()
    elif cfg.evaluation.mode == "interactive":
        while True:
            prompt = input("\nEnter prompt (or 'quit' to exit): ")
            if prompt.lower() == "quit":
                break
            response = evaluator.run_inference(prompt)
            console.print(f"[bold green]Response:[/bold green] {response}")
    elif cfg.evaluation.mode == "benchmark":
        benchmarks = evaluator.benchmark()
        console.print(benchmarks)


if __name__ == "__main__":
    main()