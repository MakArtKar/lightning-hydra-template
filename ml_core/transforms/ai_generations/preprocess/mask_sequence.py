"""Mask sequence tokens for language modeling."""

from torch import Tensor
from transformers import PreTrainedTokenizer


def mask_sequence(tensor: Tensor, mask: Tensor, tokenizer: PreTrainedTokenizer) -> Tensor:
    """Mask sequence by replacing masked positions with mask token.

    Args:
        tensor: Input sequence tensor with token IDs.
        mask: Binary mask tensor (bool, True for positions to mask).
        tokenizer: Tokenizer with mask_token_id attribute.

    Returns:
        Tensor where masked positions are replaced with tokenizer.mask_token_id.
        Formula: tensor * (1 - mask_float) + tokenizer.mask_token_id * mask_float
    """
    mask_float = mask.float()
    return tensor * (1 - mask_float) + tokenizer.mask_token_id * mask_float
