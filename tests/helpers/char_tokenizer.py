"""Simple character-level tokenizer for testing."""

from typing import Any, Dict, List, Union

import torch


class CharTokenizer:
    """A simple character-level tokenizer for testing.

    Maps each character to an integer ID. Compatible with HuggingFace tokenizer interface.
    """

    def __init__(self, vocab_size: int = 128, max_length: int = 128) -> None:
        """Initialize the character tokenizer.

        :param vocab_size: Size of vocabulary (number of unique characters).
        :param max_length: Maximum sequence length for padding/truncation.
        """
        self.vocab_size = vocab_size
        self.max_length = max_length

        # Special tokens
        self.pad_token_id = 0
        self.mask_token_id = 1
        self.unk_token_id = 2
        self.bos_token_id = 3
        self.eos_token_id = 4

        # Character mapping starts after special tokens
        self._char_offset = 5

    def __call__(
        self,
        text: Union[str, List[str]],
        padding: str = "max_length",
        truncation: bool = True,
        max_length: int | None = None,
        return_tensors: str | None = None,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        """Tokenize text into character IDs.

        :param text: Input text or list of texts.
        :param padding: Padding strategy ('max_length' or False).
        :param truncation: Whether to truncate sequences.
        :param max_length: Maximum sequence length (overrides self.max_length if provided).
        :param return_tensors: Return format ('pt' for PyTorch tensors).
        :return: Dictionary with 'input_ids' and 'attention_mask'.
        """
        if max_length is None:
            max_length = self.max_length

        # Handle single string or list of strings
        if isinstance(text, str):
            texts = [text]
        else:
            texts = text

        input_ids_list = []
        attention_mask_list = []

        for txt in texts:
            # Convert characters to IDs
            char_ids = [min(ord(c) + self._char_offset, self.vocab_size - 1) for c in txt]

            # Truncate if needed
            if truncation and len(char_ids) > max_length:
                char_ids = char_ids[:max_length]

            # Create attention mask (1 for real tokens, 0 for padding)
            attention_mask = [1] * len(char_ids)

            # Pad if needed
            if padding == "max_length":
                padding_length = max_length - len(char_ids)
                if padding_length > 0:
                    char_ids.extend([self.pad_token_id] * padding_length)
                    attention_mask.extend([0] * padding_length)

            input_ids_list.append(char_ids)
            attention_mask_list.append(attention_mask)

        result = {
            "input_ids": input_ids_list,
            "attention_mask": attention_mask_list,
        }

        # Convert to tensors if requested
        if return_tensors == "pt":
            result["input_ids"] = torch.tensor(result["input_ids"], dtype=torch.long)
            result["attention_mask"] = torch.tensor(result["attention_mask"], dtype=torch.long)

        return result

    @classmethod
    def from_pretrained(cls, *args: Any, **kwargs: Any) -> "CharTokenizer":
        """Compatibility method for from_pretrained interface.

        Ignores pretrained model arguments and returns a new instance.
        """
        # Extract relevant kwargs, ignore model name
        vocab_size = kwargs.get("vocab_size", 128)
        max_length = kwargs.get("max_length", 128)
        return cls(vocab_size=vocab_size, max_length=max_length)
