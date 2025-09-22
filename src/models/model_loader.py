"""Model loader for various LLM architectures."""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)

logger = logging.getLogger(__name__)


class ModelLoader:
    """Loader for various LLM models with quantization support."""

    def __init__(self, config):
        """Initialize model loader.

        Args:
            config: Configuration object
        """
        self.config = config
        self.model_name = config.model.name
        self.cache_dir = config.model.cache_dir
        self.device_map = config.model.device_map
        self.use_quantization = config.model.use_quantization
        self.quantization_config = self._get_quantization_config()

    def _get_quantization_config(self) -> Optional[BitsAndBytesConfig]:
        """Get quantization configuration if enabled."""
        if not self.use_quantization:
            return None

        if self.config.model.quantization_bits == 4:
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=self.config.model.double_quant,
                bnb_4bit_quant_type=self.config.model.quant_type,
            )
        elif self.config.model.quantization_bits == 8:
            return BitsAndBytesConfig(
                load_in_8bit=True,
                int8_threshold=self.config.model.int8_threshold,
            )
        else:
            raise ValueError(
                f"Unsupported quantization bits: {self.config.model.quantization_bits}"
            )

    def load(self) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """Load model and tokenizer.

        Returns:
            Tuple of (model, tokenizer)
        """
        logger.info(f"Loading model: {self.model_name}")

        tokenizer = self._load_tokenizer()
        model = self._load_model()

        self._setup_tokenizer(tokenizer)
        self._validate_model(model)

        return model, tokenizer

    def _load_tokenizer(self) -> PreTrainedTokenizer:
        """Load tokenizer for the model."""
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            trust_remote_code=self.config.model.trust_remote_code,
            use_fast=self.config.model.use_fast_tokenizer,
        )

        return tokenizer

    def _load_model(self) -> PreTrainedModel:
        """Load the pre-trained model."""
        model_kwargs = {
            "cache_dir": self.cache_dir,
            "trust_remote_code": self.config.model.trust_remote_code,
            "device_map": self.device_map,
        }

        if self.use_quantization:
            model_kwargs["quantization_config"] = self.quantization_config
            model_kwargs["torch_dtype"] = torch.float16
        else:
            model_kwargs["torch_dtype"] = getattr(
                torch,
                self.config.model.torch_dtype,
            )

        if self.config.model.use_flash_attention:
            model_kwargs["attn_implementation"] = "flash_attention_2"

        if self.config.model.load_from_checkpoint:
            model_path = self.config.model.checkpoint_path
        else:
            model_path = self.model_name

        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)

        return model

    def _setup_tokenizer(self, tokenizer: PreTrainedTokenizer):
        """Setup tokenizer with proper padding token."""
        if tokenizer.pad_token is None:
            if tokenizer.eos_token:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        tokenizer.padding_side = self.config.model.padding_side

        if self.config.model.add_special_tokens:
            special_tokens = self.config.model.special_tokens
            if special_tokens:
                tokenizer.add_special_tokens(special_tokens)

    def _validate_model(self, model: PreTrainedModel):
        """Validate loaded model."""
        if hasattr(model, "config"):
            logger.info(f"Model architecture: {model.config.architectures}")
            logger.info(f"Model parameters: {model.num_parameters():,}")

            if self.use_quantization:
                logger.info("Model loaded with quantization")

    def prepare_for_training(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
    ) -> PreTrainedModel:
        """Prepare model for training.

        Args:
            model: Pre-trained model
            tokenizer: Tokenizer

        Returns:
            Model prepared for training
        """
        if model.config.pad_token_id is None:
            model.config.pad_token_id = tokenizer.pad_token_id

        if self.config.training.gradient_checkpointing:
            model.gradient_checkpointing_enable()
            model.config.use_cache = False

        model.enable_input_require_grads()

        return model

    def save_model(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        output_dir: str,
    ):
        """Save model and tokenizer.

        Args:
            model: Trained model
            tokenizer: Tokenizer
            output_dir: Output directory path
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving model to {output_dir}")

        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        logger.info("Model saved successfully")

    def load_adapter(
        self,
        base_model: PreTrainedModel,
        adapter_path: str,
    ) -> PreTrainedModel:
        """Load PEFT adapter into base model.

        Args:
            base_model: Base pre-trained model
            adapter_path: Path to adapter weights

        Returns:
            Model with loaded adapter
        """
        from peft import PeftModel

        logger.info(f"Loading adapter from {adapter_path}")

        model = PeftModel.from_pretrained(
            base_model,
            adapter_path,
            device_map=self.device_map,
        )

        return model
