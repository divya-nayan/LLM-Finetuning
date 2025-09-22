"""Main training script for LLM fine-tuning with PEFT."""

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import hydra
import torch
import transformers
import wandb
from accelerate import Accelerator
from omegaconf import DictConfig, OmegaConf
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from rich.console import Console
from rich.progress import track
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from src.data.data_loader import DataLoader
from src.evaluation.metrics import compute_metrics as base_compute_metrics
from src.models.model_loader import ModelLoader
from src.utils.callbacks import CustomCallbacks
from src.utils.logger import setup_logger

console = Console()
logger = setup_logger(__name__)


class LLMFineTuner:
    """Main class for LLM fine-tuning with PEFT."""

    def __init__(self, config: DictConfig):
        """Initialize the fine-tuner.

        Args:
            config: Hydra configuration object
        """
        self.config = config
        self.accelerator = Accelerator(
            mixed_precision=config.training.mixed_precision,
            gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        )

        self.device = self.accelerator.device
        self.model = None
        self.tokenizer = None
        self.trainer = None

    def setup_wandb(self):
        """Initialize Weights & Biases tracking."""
        if self.config.tracking.use_wandb:
            wandb.init(
                project=self.config.tracking.project_name,
                name=self.config.tracking.run_name,
                config=OmegaConf.to_container(self.config, resolve=True),
                tags=self.config.tracking.tags,
            )
            logger.info("WandB initialized successfully")

    def load_model_and_tokenizer(self):
        """Load the base model and tokenizer."""
        logger.info(f"Loading model: {self.config.model.name}")

        model_loader = ModelLoader(self.config)
        self.model, self.tokenizer = model_loader.load()

        if self.config.model.use_quantization:
            self.model = prepare_model_for_kbit_training(self.model)

        logger.info("Model and tokenizer loaded successfully")

    def setup_peft(self):
        """Configure and apply PEFT to the model."""
        logger.info("Setting up PEFT configuration")

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.peft.lora_r,
            lora_alpha=self.config.peft.lora_alpha,
            lora_dropout=self.config.peft.lora_dropout,
            target_modules=self.config.peft.target_modules,
            bias=self.config.peft.bias,
            inference_mode=False,
        )

        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()

        logger.info("PEFT setup completed")

    def prepare_data(self):
        """Load and prepare training data."""
        logger.info("Preparing datasets")

        data_loader = DataLoader(self.config, self.tokenizer)
        self.train_dataset = data_loader.load_train_dataset()
        self.eval_dataset = data_loader.load_eval_dataset()

        logger.info(f"Training samples: {len(self.train_dataset)}")
        logger.info(f"Evaluation samples: {len(self.eval_dataset)}")

    def setup_training(self):
        """Configure training arguments and trainer."""
        logger.info("Setting up training configuration")

        training_args = TrainingArguments(
            output_dir=self.config.training.output_dir,
            num_train_epochs=self.config.training.num_epochs,
            per_device_train_batch_size=self.config.training.batch_size,
            per_device_eval_batch_size=self.config.training.eval_batch_size,
            gradient_accumulation_steps=self.config.training.gradient_accumulation_steps,
            warmup_steps=self.config.training.warmup_steps,
            learning_rate=self.config.training.learning_rate,
            fp16=self.config.training.fp16,
            bf16=self.config.training.bf16,
            logging_steps=self.config.training.logging_steps,
            eval_strategy=self.config.training.eval_strategy,
            eval_steps=self.config.training.eval_steps,
            save_strategy=self.config.training.save_strategy,
            save_steps=self.config.training.save_steps,
            save_total_limit=self.config.training.save_total_limit,
            load_best_model_at_end=self.config.training.load_best_model_at_end,
            metric_for_best_model=self.config.training.metric_for_best_model,
            greater_is_better=self.config.training.greater_is_better,
            report_to=self.config.tracking.report_to,
            push_to_hub=self.config.training.push_to_hub,
            hub_model_id=self.config.training.hub_model_id,
            gradient_checkpointing=self.config.training.gradient_checkpointing,
            optim=self.config.training.optimizer,
            weight_decay=self.config.training.weight_decay,
            max_grad_norm=self.config.training.max_grad_norm,
            lr_scheduler_type=self.config.training.lr_scheduler_type,
            dataloader_num_workers=self.config.training.dataloader_num_workers,
            remove_unused_columns=False,
            label_names=["labels"],
        )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            pad_to_multiple_of=8,
        )

        # Create compute_metrics function with tokenizer
        def compute_metrics(eval_pred):
            return base_compute_metrics(eval_pred, tokenizer=self.tokenizer)

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=[CustomCallbacks(self.config)],
        )

        logger.info("Training configuration completed")

    def train(self):
        """Execute the training loop."""
        logger.info("Starting training")

        if self.config.training.resume_from_checkpoint:
            checkpoint = self.config.training.checkpoint_path
        else:
            checkpoint = None

        train_result = self.trainer.train(resume_from_checkpoint=checkpoint)

        metrics = train_result.metrics
        self.trainer.log_metrics("train", metrics)
        self.trainer.save_metrics("train", metrics)

        logger.info("Training completed successfully")
        return metrics

    def evaluate(self):
        """Run evaluation on the test set."""
        logger.info("Running evaluation")

        eval_metrics = self.trainer.evaluate()
        self.trainer.log_metrics("eval", eval_metrics)
        self.trainer.save_metrics("eval", eval_metrics)

        logger.info(f"Evaluation metrics: {eval_metrics}")
        return eval_metrics

    def save_model(self):
        """Save the fine-tuned model."""
        logger.info(f"Saving model to {self.config.training.final_model_path}")

        self.trainer.save_model(self.config.training.final_model_path)
        self.tokenizer.save_pretrained(self.config.training.final_model_path)

        if self.config.training.save_merged:
            logger.info("Saving merged model")
            merged_model = self.model.merge_and_unload()
            merged_model.save_pretrained(
                f"{self.config.training.final_model_path}_merged"
            )
            self.tokenizer.save_pretrained(
                f"{self.config.training.final_model_path}_merged"
            )

        logger.info("Model saved successfully")

    def run(self):
        """Execute the complete fine-tuning pipeline."""
        try:
            console.print("[bold blue]Starting LLM Fine-tuning Pipeline[/bold blue]")

            self.setup_wandb()
            self.load_model_and_tokenizer()
            self.setup_peft()
            self.prepare_data()
            self.setup_training()

            train_metrics = self.train()
            eval_metrics = self.evaluate()

            self.save_model()

            if self.config.tracking.use_wandb:
                wandb.finish()

            console.print(
                "[bold green]Fine-tuning completed successfully![/bold green]"
            )
            return {
                "train_metrics": train_metrics,
                "eval_metrics": eval_metrics,
            }

        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            console.print(f"[bold red]Training failed: {str(e)}[/bold red]")
            raise


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    """Main entry point for training."""
    console.print("[bold]Configuration:[/bold]")
    console.print(OmegaConf.to_yaml(cfg))

    fine_tuner = LLMFineTuner(cfg)
    results = fine_tuner.run()

    console.print("[bold]Final Results:[/bold]")
    console.print(results)


if __name__ == "__main__":
    main()
