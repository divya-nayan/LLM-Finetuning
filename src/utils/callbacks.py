"""Custom callbacks for training."""

import json
import logging
from datetime import datetime
from pathlib import Path

from transformers import TrainerCallback, TrainerControl, TrainerState
from transformers.training_args import TrainingArguments

logger = logging.getLogger(__name__)


class CustomCallbacks(TrainerCallback):
    """Custom callbacks for training monitoring."""

    def __init__(self, config):
        """Initialize callbacks.

        Args:
            config: Configuration object
        """
        self.config = config
        self.start_time = None
        self.metrics_history = []

    def on_train_begin(self, args, state, control, **kwargs):
        """Called at the beginning of training."""
        self.start_time = datetime.now()
        logger.info("Training started")
        logger.info(f"Total training steps: {state.max_steps}")

        return control

    def on_train_end(self, args, state, control, **kwargs):
        """Called at the end of training."""
        duration = datetime.now() - self.start_time
        logger.info(f"Training completed in {duration}")

        self._save_metrics_history(args.output_dir)

        return control

    def on_step_end(self, args, state, control, **kwargs):
        """Called at the end of each training step."""
        if state.global_step % self.config.training.logging_steps == 0:
            if "loss" in state.log_history[-1]:
                loss = state.log_history[-1]["loss"]
                logger.info(f"Step {state.global_step}: loss = {loss:.4f}")

        return control

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        """Called after evaluation."""
        logger.info(f"Evaluation at step {state.global_step}")
        for key, value in metrics.items():
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.4f}")

        self.metrics_history.append(
            {
                "step": state.global_step,
                "timestamp": datetime.now().isoformat(),
                **metrics,
            }
        )

        return control

    def on_save(self, args, state, control, **kwargs):
        """Called when saving checkpoint."""
        logger.info(f"Checkpoint saved at step {state.global_step}")
        return control

    def on_log(self, args, state, control, logs, **kwargs):
        """Called when logging."""
        if self.config.tracking.use_tensorboard:
            self._log_to_tensorboard(logs, state.global_step)

        return control

    def _save_metrics_history(self, output_dir: str):
        """Save metrics history to file.

        Args:
            output_dir: Output directory
        """
        if self.metrics_history:
            metrics_file = Path(output_dir) / "metrics_history.json"
            with open(metrics_file, "w") as f:
                json.dump(self.metrics_history, f, indent=2)
            logger.info(f"Metrics history saved to {metrics_file}")

    def _log_to_tensorboard(self, logs: dict, step: int):
        """Log metrics to TensorBoard.

        Args:
            logs: Metrics to log
            step: Current step
        """
        try:
            from torch.utils.tensorboard import SummaryWriter

            writer = SummaryWriter(self.config.tracking.tensorboard_dir)

            for key, value in logs.items():
                if isinstance(value, (int, float)):
                    writer.add_scalar(key, value, step)

            writer.flush()
        except Exception as e:
            logger.warning(f"Failed to log to TensorBoard: {e}")


class EarlyStoppingCallback(TrainerCallback):
    """Early stopping callback."""

    def __init__(self, patience: int = 3, threshold: float = 0.001):
        """Initialize early stopping.

        Args:
            patience: Number of evaluations to wait
            threshold: Minimum improvement threshold
        """
        self.patience = patience
        self.threshold = threshold
        self.best_metric = None
        self.patience_counter = 0

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        """Check for early stopping condition."""
        metric_key = args.metric_for_best_model
        if metric_key not in metrics:
            return control

        current_metric = metrics[metric_key]

        if self.best_metric is None:
            self.best_metric = current_metric
        else:
            if args.greater_is_better:
                improved = current_metric > self.best_metric + self.threshold
            else:
                improved = current_metric < self.best_metric - self.threshold

            if improved:
                self.best_metric = current_metric
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            if self.patience_counter >= self.patience:
                logger.info(f"Early stopping triggered at step {state.global_step}")
                control.should_training_stop = True

        return control


class GradientAccumulationCallback(TrainerCallback):
    """Callback for gradient accumulation monitoring."""

    def __init__(self, accumulation_steps: int):
        """Initialize gradient accumulation callback.

        Args:
            accumulation_steps: Number of accumulation steps
        """
        self.accumulation_steps = accumulation_steps
        self.accumulated_loss = 0
        self.accumulation_counter = 0

    def on_step_end(self, args, state, control, **kwargs):
        """Monitor gradient accumulation."""
        self.accumulation_counter += 1

        if self.accumulation_counter >= self.accumulation_steps:
            if self.accumulated_loss > 0:
                avg_loss = self.accumulated_loss / self.accumulation_steps
                logger.debug(f"Accumulated loss: {avg_loss:.4f}")

            self.accumulated_loss = 0
            self.accumulation_counter = 0

        return control
