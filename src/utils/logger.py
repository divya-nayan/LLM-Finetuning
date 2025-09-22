"""Logging configuration and utilities."""

import logging
import sys
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler


def setup_logger(
    name: str,
    log_dir: str = "outputs/logs",
    log_level: str = "INFO",
    console_output: bool = True,
    file_output: bool = True,
) -> logging.Logger:
    """Setup logger with rich formatting.

    Args:
        name: Logger name
        log_dir: Directory for log files
        log_level: Logging level
        console_output: Enable console output
        file_output: Enable file output

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))

    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if console_output:
        console_handler = RichHandler(
            console=Console(stderr=True),
            show_time=False,
            show_path=False,
        )
        console_handler.setLevel(logging.INFO)
        logger.addHandler(console_handler)

    if file_output:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_handler = logging.FileHandler(log_path / f"{name}_{timestamp}.log")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class TrainingLogger:
    """Custom logger for training metrics."""

    def __init__(self, log_dir: str = "outputs/logs"):
        """Initialize training logger.

        Args:
            log_dir: Directory for log files
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.metrics_file = (
            self.log_dir / f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        )
        self.console = Console()

    def log_metrics(self, metrics: dict, step: int = None):
        """Log metrics to file.

        Args:
            metrics: Dictionary of metrics
            step: Training step
        """
        import json

        log_entry = {"timestamp": datetime.now().isoformat(), "step": step, **metrics}

        with open(self.metrics_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

    def log_hyperparameters(self, params: dict):
        """Log hyperparameters.

        Args:
            params: Dictionary of hyperparameters
        """
        import json

        params_file = self.log_dir / "hyperparameters.json"
        with open(params_file, "w") as f:
            json.dump(params, f, indent=2)

    def create_summary(self, results: dict):
        """Create training summary.

        Args:
            results: Training results
        """
        from rich.table import Table

        table = Table(title="Training Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")

        for key, value in results.items():
            if isinstance(value, float):
                table.add_row(key, f"{value:.4f}")
            else:
                table.add_row(key, str(value))

        self.console.print(table)
