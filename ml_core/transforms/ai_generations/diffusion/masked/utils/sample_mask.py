"""Sample binary masks for diffusion models."""

import torch
from torch import Tensor


def sample_mask(tensor: Tensor, rate: float = 0.5) -> Tensor:
    """Sample binary mask with the same shape as tensor.

    Args:
        tensor: Input tensor to match shape.
        rate: Probability of sampling True in the mask. Default is 0.5.

    Returns:
        Binary mask tensor of type bool with same shape as input tensor.
        Values are True sampled with `rate` probability, False otherwise.
    """
    mask = torch.rand_like(tensor, dtype=torch.float32) < rate
    return mask
