"""Tests for utility functions."""

import unittest
import torch
import numpy as np
from unittest.mock import Mock

from src.utils.common import (
    format_prompt,
    tokenize_text,
    decode_tokens,
    format_instruction_prompt,
    format_chat_prompt,
    compute_model_size
)


class TestCommonUtils(unittest.TestCase):
    """Test common utility functions."""

    def test_format_prompt_with_template(self):
        """Test prompt formatting with template."""
        item = {"name": "Alice", "age": "30"}
        template = "Name: {name}, Age: {age}"

        result = format_prompt(item, template)
        expected = "Name: Alice, Age: 30"

        self.assertEqual(result, expected)

    def test_format_prompt_without_template(self):
        """Test prompt formatting without template."""
        item = {"text": "Hello world"}

        result = format_prompt(item, None)

        self.assertEqual(result, "Hello world")

    def test_format_instruction_prompt(self):
        """Test instruction prompt formatting."""
        result = format_instruction_prompt(
            instruction="Write a poem",
            input_text="About nature",
            output="Roses are red..."
        )

        self.assertIn("### Instruction:", result)
        self.assertIn("Write a poem", result)
        self.assertIn("### Input:", result)
        self.assertIn("About nature", result)
        self.assertIn("### Response:", result)
        self.assertIn("Roses are red...", result)

    def test_format_instruction_prompt_without_input(self):
        """Test instruction prompt without input."""
        result = format_instruction_prompt(
            instruction="Tell a joke",
            output="Why did the chicken..."
        )

        self.assertIn("### Instruction:", result)
        self.assertNotIn("### Input:", result)
        self.assertIn("### Response:", result)

    def test_format_chat_prompt(self):
        """Test chat prompt formatting."""
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"}
        ]

        result = format_chat_prompt(messages, add_generation_prompt=False)

        self.assertIn("System: You are helpful", result)
        self.assertIn("User: Hello", result)
        self.assertIn("Assistant: Hi there", result)

    def test_format_chat_prompt_with_generation(self):
        """Test chat prompt with generation prompt."""
        messages = [
            {"role": "user", "content": "Hello"}
        ]

        result = format_chat_prompt(messages, add_generation_prompt=True)

        self.assertTrue(result.endswith("Assistant: "))

    def test_compute_model_size(self):
        """Test model size computation."""
        # Create a simple mock model
        model = Mock()
        param1 = Mock()
        param1.numel.return_value = 1000
        param1.requires_grad = True

        param2 = Mock()
        param2.numel.return_value = 500
        param2.requires_grad = False

        model.parameters.return_value = [param1, param2]

        result = compute_model_size(model)

        self.assertEqual(result["total_params"], 1500)
        self.assertEqual(result["trainable_params"], 1000)
        self.assertIn("total_size_mb", result)
        self.assertIn("trainable_size_mb", result)
        self.assertIn("trainable_percentage", result)


class TestTokenizationUtils(unittest.TestCase):
    """Test tokenization utility functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_tokenizer = Mock()
        self.mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]])
        }
        self.mock_tokenizer.pad_token_id = 0
        self.mock_tokenizer.decode.return_value = "Test text"

    def test_tokenize_text_single(self):
        """Test tokenizing single text."""
        result = tokenize_text(
            "Hello world",
            self.mock_tokenizer,
            max_length=10
        )

        self.mock_tokenizer.assert_called_once()
        self.assertIn("input_ids", result)
        self.assertIn("attention_mask", result)

    def test_decode_tokens_tensor(self):
        """Test decoding tensor tokens."""
        tokens = torch.tensor([1, 2, 3, 0, 0])  # With padding

        result = decode_tokens(
            tokens,
            self.mock_tokenizer,
            clean_padding=True
        )

        self.mock_tokenizer.decode.assert_called()
        self.assertEqual(result, "Test text")

    def test_decode_tokens_batch(self):
        """Test decoding batch of tokens."""
        tokens = np.array([[1, 2, 3, -100], [4, 5, -100, -100]])

        self.mock_tokenizer.decode.side_effect = ["Text 1", "Text 2"]

        result = decode_tokens(
            tokens,
            self.mock_tokenizer,
            clean_padding=True
        )

        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], "Text 1")
        self.assertEqual(result[1], "Text 2")


if __name__ == "__main__":
    unittest.main()