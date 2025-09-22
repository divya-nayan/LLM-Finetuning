"""Common utility functions to avoid code redundancy."""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from transformers import PreTrainedTokenizer


def format_prompt(
    item: Dict[str, str], template: Optional[str] = None, default_key: str = "text"
) -> str:
    """Format data item using prompt template.

    Args:
        item: Data dictionary
        template: Optional prompt template with placeholders
        default_key: Default key to use if no template

    Returns:
        Formatted text string
    """
    if template:
        try:
            return template.format(**item)
        except KeyError:
            pass

    return item.get(default_key, item.get("input", ""))


def tokenize_text(
    texts: Union[str, List[str]],
    tokenizer: PreTrainedTokenizer,
    max_length: int = 512,
    padding: str = "max_length",
    truncation: bool = True,
    return_tensors: Optional[str] = "pt",
) -> Dict[str, Any]:
    """Tokenize text with standard settings.

    Args:
        texts: Text or list of texts to tokenize
        tokenizer: Tokenizer to use
        max_length: Maximum sequence length
        padding: Padding strategy
        truncation: Whether to truncate
        return_tensors: Return tensor format

    Returns:
        Tokenized output dictionary
    """
    return tokenizer(
        texts,
        max_length=max_length,
        padding=padding,
        truncation=truncation,
        return_tensors=return_tensors,
    )


def decode_tokens(
    token_ids: Union[torch.Tensor, np.ndarray, List],
    tokenizer: PreTrainedTokenizer,
    skip_special_tokens: bool = True,
    clean_padding: bool = True,
) -> Union[str, List[str]]:
    """Decode token IDs to text.

    Args:
        token_ids: Token IDs to decode
        tokenizer: Tokenizer to use
        skip_special_tokens: Whether to skip special tokens
        clean_padding: Whether to remove padding tokens

    Returns:
        Decoded text or list of texts
    """
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.cpu().numpy()

    if isinstance(token_ids, np.ndarray):
        if clean_padding:
            # Remove padding tokens (-100 or pad_token_id)
            if len(token_ids.shape) == 2:
                # Batch of sequences
                decoded_texts = []
                for ids in token_ids:
                    ids = ids[ids != -100]
                    if tokenizer.pad_token_id is not None:
                        ids = ids[ids != tokenizer.pad_token_id]
                    text = tokenizer.decode(
                        ids, skip_special_tokens=skip_special_tokens
                    )
                    decoded_texts.append(text)
                return decoded_texts
            else:
                # Single sequence
                token_ids = token_ids[token_ids != -100]
                if tokenizer.pad_token_id is not None:
                    token_ids = token_ids[token_ids != tokenizer.pad_token_id]

    return tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)


def format_instruction_prompt(
    instruction: str,
    input_text: Optional[str] = None,
    output: Optional[str] = None,
    system_prompt: Optional[str] = None,
) -> str:
    """Format instruction-following prompt.

    Args:
        instruction: Main instruction
        input_text: Optional input context
        output: Expected output (for training)
        system_prompt: Optional system prompt

    Returns:
        Formatted prompt string
    """
    prompt_parts = []

    if system_prompt:
        prompt_parts.append(f"System: {system_prompt}")

    prompt_parts.append(f"### Instruction:\n{instruction}")

    if input_text:
        prompt_parts.append(f"### Input:\n{input_text}")

    prompt_parts.append("### Response:")

    if output:
        prompt_parts.append(output)

    return "\n\n".join(prompt_parts)


def format_chat_prompt(
    messages: List[Dict[str, str]], add_generation_prompt: bool = True
) -> str:
    """Format chat messages into prompt.

    Args:
        messages: List of message dictionaries with 'role' and 'content'
        add_generation_prompt: Whether to add prompt for generation

    Returns:
        Formatted chat prompt
    """
    formatted = []

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if role == "system":
            formatted.append(f"System: {content}")
        elif role == "user":
            formatted.append(f"User: {content}")
        elif role == "assistant":
            formatted.append(f"Assistant: {content}")

    prompt = "\n".join(formatted)

    if add_generation_prompt and (not messages or messages[-1]["role"] != "assistant"):
        prompt += "\nAssistant: "

    return prompt


def compute_model_size(model: torch.nn.Module) -> Dict[str, float]:
    """Compute model size statistics.

    Args:
        model: PyTorch model

    Returns:
        Dictionary with size statistics
    """
    param_count = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Calculate size in MB (assuming float16)
    param_size_mb = (param_count * 2) / (1024 * 1024)
    trainable_size_mb = (trainable_count * 2) / (1024 * 1024)

    return {
        "total_params": param_count,
        "trainable_params": trainable_count,
        "total_size_mb": param_size_mb,
        "trainable_size_mb": trainable_size_mb,
        "trainable_percentage": (
            (trainable_count / param_count * 100) if param_count > 0 else 0
        ),
    }
