"""Unit tests for sample_mask function."""

import pytest
import torch

from ml_core.transforms.ai_generations.diffusion.masked.utils.sample_mask import (
    sample_mask,
)


def test_sample_mask_shape():
    """Test that mask has same shape as input tensor."""
    tensor = torch.randn(10, 20, 30)
    mask = sample_mask(tensor)
    assert mask.shape == tensor.shape


def test_sample_mask_dtype():
    """Test that mask is bool."""
    tensor = torch.randn(5, 5)
    mask = sample_mask(tensor)
    assert mask.dtype == torch.bool


def test_sample_mask_values():
    """Test that mask contains only True or False."""
    tensor = torch.randn(100, 100)
    mask = sample_mask(tensor)
    unique_values = torch.unique(mask)
    assert all(val in [False, True] for val in unique_values)


def test_sample_mask_rate_zero():
    """Test that rate=0.0 produces all False."""
    torch.manual_seed(42)
    tensor = torch.randn(50, 50)
    mask = sample_mask(tensor, rate=0.0)
    assert not torch.any(mask)


def test_sample_mask_rate_one():
    """Test that rate=1.0 produces all True."""
    torch.manual_seed(42)
    tensor = torch.randn(50, 50)
    mask = sample_mask(tensor, rate=1.0)
    assert torch.all(mask)


def test_sample_mask_rate_approximate():
    """Test that mask has approximately correct rate for large samples."""
    torch.manual_seed(42)
    tensor = torch.randn(1000, 1000)
    rate = 0.3
    mask = sample_mask(tensor, rate=rate)
    actual_rate = mask.float().mean().item()
    # With 1M samples, should be very close to target rate
    assert abs(actual_rate - rate) < 0.01


def test_sample_mask_empty_tensor():
    """Test with empty tensor."""
    tensor = torch.randn(0)
    mask = sample_mask(tensor)
    assert mask.shape == (0,)
    assert mask.dtype == torch.bool


def test_sample_mask_scalar_like():
    """Test with single element tensor."""
    tensor = torch.tensor([5.0])
    mask = sample_mask(tensor, rate=0.5)
    assert mask.shape == (1,)
    assert mask.item() in [False, True]


def test_sample_mask_different_input_dtypes():
    """Test that function works with different input dtypes."""
    for dtype in [torch.float32, torch.float64, torch.int32, torch.int64]:
        tensor = torch.ones(10, 10, dtype=dtype)
        mask = sample_mask(tensor)
        assert mask.dtype == torch.bool
        assert mask.shape == tensor.shape
