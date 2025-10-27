"""Text-specific transforms for NLP tasks."""

from typing import Any, Mapping

import torch
import torch.nn as nn


class TokenizerTransform(nn.Module):
    """Apply a HuggingFace tokenizer to text in batch dict.

    :param tokenizer: HuggingFace tokenizer instance.
    :param key: Key in batch dict containing text to tokenize.
    :param tokenizer_kwargs: Additional arguments passed to tokenizer.
    """

    def __init__(
        self,
        tokenizer: Any,
        key: str = "text",
        **tokenizer_kwargs: Any,
    ) -> None:
        """Initialize the tokenizer transform.

        :param tokenizer: HuggingFace tokenizer to apply.
        :param key: Batch dict key containing text data.
        :param tokenizer_kwargs: Keyword arguments for tokenizer call.
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.key = key
        self.tokenizer_kwargs = tokenizer_kwargs

    def __call__(self, batch: Mapping[str, Any]) -> Mapping[str, Any]:
        """Tokenize text from batch and return tokenized outputs.

        :param batch: Input batch dict with text data.
        :return: Dict with tokenizer outputs (input_ids, attention_mask, etc.) as tensors.
        """
        texts = batch[self.key]
        # Tokenizer returns BatchEncoding with input_ids, attention_mask, token_type_ids
        tokenized = self.tokenizer(texts, **self.tokenizer_kwargs)
        # Convert to dict - tensors are already in the right shape (batch_size, seq_len)
        return {k: v for k, v in tokenized.items()}

