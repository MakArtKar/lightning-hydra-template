"""Mask target tensor for cross-entropy loss by replacing masked positions with ignore_index."""

import torch
from torch import LongTensor, Tensor


def mask_target_before_ce(target: LongTensor, mask: Tensor) -> LongTensor:
    """Mask target tensor by replacing positions where mask == 0 with -100 (ignore_index).

    Args:
        target: Target tensor with token IDs of type LongTensor.
        mask: Binary mask tensor (bool or int). Positions where mask == 0 will be
            replaced with -100 in the target.

    Returns:
        LongTensor where positions with mask == 0 are replaced with -100 (ignore_index
        for cross-entropy loss), preserving original values where mask != 0.
    """
    # Convert mask to boolean if needed
    if mask.dtype != torch.bool:
        mask = mask.bool()

    # Create ignore mask (True where we should ignore, i.e., where mask == 0)
    ignore_mask = ~mask

    # Replace ignored positions with -100
    masked_target = target.clone()
    masked_target[ignore_mask] = -100

    return masked_target
