"""Tests for data loading functionality."""

import unittest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch

from src.data.data_loader import DataLoader, TextDataset
from src.utils.common import format_prompt, format_instruction_prompt


class TestDataLoader(unittest.TestCase):
    """Test data loader functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_tokenizer = Mock()
        self.mock_tokenizer.return_value = {
            "input_ids": [[1, 2, 3]],
            "attention_mask": [[1, 1, 1]]
        }
        self.mock_config = Mock()
        self.mock_config.data.max_length = 512
        self.mock_config.data.prompt_template = None
        self.mock_config.data.num_proc = 1
        self.mock_config.data.trust_remote_code = False

    def test_text_dataset_initialization(self):
        """Test TextDataset initialization."""
        data = [{"text": "Hello world"}]
        dataset = TextDataset(data, self.mock_tokenizer, max_length=256)

        self.assertEqual(len(dataset), 1)
        self.assertEqual(dataset.max_length, 256)

    def test_format_prompt_with_template(self):
        """Test prompt formatting with template."""
        item = {"instruction": "Test", "output": "Response"}
        template = "Instruction: {instruction}\nResponse: {output}"

        result = format_prompt(item, template)
        expected = "Instruction: Test\nResponse: Response"

        self.assertEqual(result, expected)

    def test_format_instruction_prompt(self):
        """Test instruction prompt formatting."""
        result = format_instruction_prompt(
            instruction="Explain AI",
            input_text="In simple terms",
            output="AI is..."
        )

        self.assertIn("### Instruction:", result)
        self.assertIn("Explain AI", result)
        self.assertIn("### Input:", result)
        self.assertIn("### Response:", result)

    def test_load_json_file(self):
        """Test loading JSON file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump([{"text": "test"}], f)
            temp_path = f.name

        try:
            loader = DataLoader(self.mock_config, self.mock_tokenizer)
            data = loader._load_json(Path(temp_path))

            self.assertEqual(len(data), 1)
            self.assertEqual(data[0]["text"], "test")
        finally:
            Path(temp_path).unlink()

    def test_load_jsonl_file(self):
        """Test loading JSONL file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write('{"text": "line1"}\n')
            f.write('{"text": "line2"}\n')
            temp_path = f.name

        try:
            loader = DataLoader(self.mock_config, self.mock_tokenizer)
            data = loader._load_jsonl(Path(temp_path))

            self.assertEqual(len(data), 2)
            self.assertEqual(data[0]["text"], "line1")
            self.assertEqual(data[1]["text"], "line2")
        finally:
            Path(temp_path).unlink()


class TestDataPreprocessing(unittest.TestCase):
    """Test data preprocessing functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_tokenizer = Mock()
        self.mock_tokenizer.return_value = {
            "input_ids": [[1, 2, 3]],
            "attention_mask": [[1, 1, 1]]
        }

    def test_prepare_instruction_dataset(self):
        """Test instruction dataset preparation."""
        mock_config = Mock()
        mock_config.data.max_length = 512
        mock_config.data.prompt_template = None

        loader = DataLoader(mock_config, self.mock_tokenizer)

        examples = {
            "instruction": ["Test instruction"],
            "input": ["Test input"],
            "output": ["Test output"]
        }

        # Mock tokenize_text to return proper structure
        with patch('src.data.data_loader.tokenize_text') as mock_tokenize:
            mock_tokenize.return_value = {
                "input_ids": [[1, 2, 3]],
                "attention_mask": [[1, 1, 1]]
            }

            result = loader.prepare_instruction_dataset(examples)

            self.assertIn("input_ids", result)
            self.assertIn("labels", result)
            self.assertEqual(result["input_ids"], result["labels"])


if __name__ == "__main__":
    unittest.main()