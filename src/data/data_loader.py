"""Data loader for various dataset formats and sources."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import torch
from datasets import Dataset as HFDataset
from datasets import load_dataset
from rich.progress import track
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from src.utils.common import format_instruction_prompt, format_prompt, tokenize_text

logger = logging.getLogger(__name__)


class TextDataset(Dataset):
    """Custom dataset class for text data."""

    def __init__(
        self,
        data: List[Dict[str, str]],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        prompt_template: Optional[str] = None,
    ):
        """Initialize the dataset.

        Args:
            data: List of data samples
            tokenizer: Tokenizer to use
            max_length: Maximum sequence length
            prompt_template: Template for formatting prompts
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt_template = prompt_template

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        text = format_prompt(item, self.prompt_template)

        encoding = tokenize_text(
            text,
            self.tokenizer,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": encoding["input_ids"].squeeze(),
        }


class DataLoader:
    """Main data loader class supporting multiple formats."""

    def __init__(self, config, tokenizer: PreTrainedTokenizer):
        """Initialize the data loader.

        Args:
            config: Configuration object
            tokenizer: Tokenizer to use
        """
        self.config = config
        self.tokenizer = tokenizer
        self.max_length = config.data.max_length
        self.prompt_template = config.data.prompt_template

    def load_train_dataset(self) -> Union[Dataset, HFDataset]:
        """Load training dataset."""
        return self._load_dataset(
            self.config.data.train_path,
            self.config.data.dataset_name,
            "train",
        )

    def load_eval_dataset(self) -> Union[Dataset, HFDataset]:
        """Load evaluation dataset."""
        return self._load_dataset(
            self.config.data.eval_path,
            self.config.data.dataset_name,
            "validation",
        )

    def _load_dataset(
        self,
        data_path: Optional[str],
        dataset_name: Optional[str],
        split: str,
    ) -> Union[Dataset, HFDataset]:
        """Load dataset from various sources.

        Args:
            data_path: Path to local dataset file
            dataset_name: Name of HuggingFace dataset
            split: Dataset split to load

        Returns:
            Loaded dataset
        """
        if dataset_name:
            logger.info(f"Loading dataset from HuggingFace: {dataset_name}")
            return self._load_huggingface_dataset(dataset_name, split)
        elif data_path:
            logger.info(f"Loading dataset from file: {data_path}")
            return self._load_local_dataset(data_path)
        else:
            raise ValueError("Either dataset_name or data_path must be provided")

    def _load_huggingface_dataset(self, dataset_name: str, split: str) -> HFDataset:
        """Load dataset from HuggingFace Hub."""
        dataset = load_dataset(
            dataset_name,
            split=split,
            trust_remote_code=self.config.data.trust_remote_code,
        )

        if self.config.data.preprocessing_function:
            dataset = dataset.map(
                self._preprocess_function,
                batched=True,
                num_proc=self.config.data.num_proc,
                remove_columns=dataset.column_names,
            )
        else:
            dataset = dataset.map(
                self._tokenize_function,
                batched=True,
                num_proc=self.config.data.num_proc,
                remove_columns=dataset.column_names,
            )

        return dataset

    def _load_local_dataset(self, data_path: str) -> Dataset:
        """Load dataset from local file."""
        path = Path(data_path)

        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {data_path}")

        if path.suffix == ".json":
            data = self._load_json(path)
        elif path.suffix == ".jsonl":
            data = self._load_jsonl(path)
        elif path.suffix == ".csv":
            data = self._load_csv(path)
        elif path.suffix == ".txt":
            data = self._load_text(path)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

        return TextDataset(
            data,
            self.tokenizer,
            self.max_length,
            self.prompt_template,
        )

    def _load_json(self, path: Path) -> List[Dict]:
        """Load JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = [data]
        return data

    def _load_jsonl(self, path: Path) -> List[Dict]:
        """Load JSONL file."""
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line.strip()))
        return data

    def _load_csv(self, path: Path) -> List[Dict]:
        """Load CSV file."""
        df = pd.read_csv(path)
        return df.to_dict("records")

    def _load_text(self, path: Path) -> List[Dict]:
        """Load plain text file."""
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        return [{"text": line.strip()} for line in lines if line.strip()]

    def _tokenize_function(self, examples: Dict[str, List]) -> Dict:
        """Tokenization function for dataset mapping."""
        if self.prompt_template and "instruction" in examples:
            texts = []
            for i in range(len(examples["instruction"])):
                item = {k: v[i] for k, v in examples.items()}
                texts.append(format_prompt(item, self.prompt_template))
        else:
            texts = examples.get("text", examples.get("input", []))

        model_inputs = tokenize_text(
            texts,
            self.tokenizer,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors=None,
        )

        model_inputs["labels"] = model_inputs["input_ids"].copy()

        return model_inputs

    def _preprocess_function(self, examples: Dict[str, List]) -> Dict:
        """Custom preprocessing function."""
        if hasattr(self, self.config.data.preprocessing_function):
            preprocess_fn = getattr(self, self.config.data.preprocessing_function)
            return preprocess_fn(examples)
        else:
            return self._tokenize_function(examples)

    def prepare_instruction_dataset(self, examples: Dict[str, List]) -> Dict:
        """Prepare instruction-following dataset."""
        texts = []
        for i in range(len(examples["instruction"])):
            instruction = examples["instruction"][i]
            input_text = examples.get("input", [""] * len(examples["instruction"]))[i]
            output = examples["output"][i]

            prompt = format_instruction_prompt(
                instruction=instruction,
                input_text=input_text if input_text else None,
                output=output,
            )
            texts.append(prompt)

        model_inputs = tokenize_text(
            texts,
            self.tokenizer,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors=None,
        )

        model_inputs["labels"] = model_inputs["input_ids"].copy()

        return model_inputs

    def prepare_chat_dataset(self, examples: Dict[str, List]) -> Dict:
        """Prepare chat-style dataset."""
        texts = []
        for i in range(len(examples["conversations"])):
            conversation = examples["conversations"][i]
            formatted_conv = self.format_conversation(conversation)
            texts.append(formatted_conv)

        model_inputs = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
        )

        model_inputs["labels"] = model_inputs["input_ids"].copy()

        return model_inputs

    def format_conversation(self, conversation: List[Dict]) -> str:
        """Format conversation for training."""
        formatted = ""
        for turn in conversation:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            if role == "user":
                formatted += f"User: {content}\n"
            elif role == "assistant":
                formatted += f"Assistant: {content}\n"
        return formatted.strip()
