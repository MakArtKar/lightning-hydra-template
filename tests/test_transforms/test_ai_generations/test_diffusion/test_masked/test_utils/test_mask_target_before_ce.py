"""Unit tests for mask_target_before_ce function."""

import pytest
import torch

from ml_core.transforms.ai_generations.diffusion.masked.utils.mask_target_before_ce import (
    mask_target_before_ce,
)


def test_mask_target_before_ce_basic():
    """Test basic functionality with boolean mask."""
    target = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)
    mask = torch.tensor([True, True, False, True, False], dtype=torch.bool)
    result = mask_target_before_ce(target, mask)
    expected = torch.tensor([1, 2, -100, 4, -100], dtype=torch.long)
    assert torch.equal(result, expected)


def test_mask_target_before_ce_int_mask():
    """Test that function works with integer mask (0 and 1)."""
    target = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)
    mask = torch.tensor([1, 1, 0, 1, 0], dtype=torch.int32)
    result = mask_target_before_ce(target, mask)
    expected = torch.tensor([1, 2, -100, 4, -100], dtype=torch.long)
    assert torch.equal(result, expected)


def test_mask_target_before_ce_all_masked():
    """Test when all positions are masked (mask == 0 everywhere)."""
    target = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)
    mask = torch.tensor([0, 0, 0, 0, 0], dtype=torch.bool)
    result = mask_target_before_ce(target, mask)
    expected = torch.tensor([-100, -100, -100, -100, -100], dtype=torch.long)
    assert torch.equal(result, expected)


def test_mask_target_before_ce_none_masked():
    """Test when no positions are masked (mask == 1 everywhere)."""
    target = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)
    mask = torch.tensor([True, True, True, True, True], dtype=torch.bool)
    result = mask_target_before_ce(target, mask)
    assert torch.equal(result, target)


def test_mask_target_before_ce_shape_preservation():
    """Test that output shape matches input shape."""
    target = torch.randint(0, 100, (10, 20), dtype=torch.long)
    mask = torch.randint(0, 2, (10, 20), dtype=torch.bool)
    result = mask_target_before_ce(target, mask)
    assert result.shape == target.shape
    assert result.dtype == torch.long


def test_mask_target_before_ce_no_modification_to_original():
    """Test that original target tensor is not modified."""
    target = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)
    target_original = target.clone()
    mask = torch.tensor([True, True, False, True, False], dtype=torch.bool)
    _ = mask_target_before_ce(target, mask)
    assert torch.equal(target, target_original)


def test_mask_target_before_ce_2d_tensor():
    """Test with 2D tensors (batch_size, seq_len)."""
    target = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
    mask = torch.tensor([[True, False, True], [False, True, False]], dtype=torch.bool)
    result = mask_target_before_ce(target, mask)
    expected = torch.tensor([[1, -100, 3], [-100, 5, -100]], dtype=torch.long)
    assert torch.equal(result, expected)


def test_mask_target_before_ce_3d_tensor():
    """Test with 3D tensors."""
    target = torch.randint(0, 100, (2, 3, 4), dtype=torch.long)
    mask = torch.randint(0, 2, (2, 3, 4), dtype=torch.bool)
    result = mask_target_before_ce(target, mask)
    assert result.shape == target.shape
    assert result.dtype == torch.long
    # Verify that where mask is False, values are -100
    assert torch.all(result[~mask] == -100)
    # Verify that where mask is True, values match original
    assert torch.all(result[mask] == target[mask])


def test_mask_target_before_ce_empty_tensor():
    """Test with empty tensor."""
    target = torch.tensor([], dtype=torch.long)
    mask = torch.tensor([], dtype=torch.bool)
    result = mask_target_before_ce(target, mask)
    assert torch.equal(result, target)
    assert result.shape == (0,)


def test_mask_target_before_ce_single_element():
    """Test with single element tensor."""
    target = torch.tensor([42], dtype=torch.long)
    mask = torch.tensor([False], dtype=torch.bool)
    result = mask_target_before_ce(target, mask)
    assert result.item() == -100

    mask = torch.tensor([True], dtype=torch.bool)
    result = mask_target_before_ce(target, mask)
    assert result.item() == 42


def test_mask_target_before_ce_float_mask():
    """Test that float mask is correctly converted (0.0 -> False, non-zero -> True)."""
    target = torch.tensor([1, 2, 3], dtype=torch.long)
    mask = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32)
    result = mask_target_before_ce(target, mask)
    expected = torch.tensor([1, -100, 3], dtype=torch.long)
    assert torch.equal(result, expected)


def test_mask_target_before_ce_edge_case_zero_target():
    """Test with target containing zero values."""
    target = torch.tensor([0, 1, 2, 0], dtype=torch.long)
    mask = torch.tensor([True, False, True, False], dtype=torch.bool)
    result = mask_target_before_ce(target, mask)
    expected = torch.tensor([0, -100, 2, -100], dtype=torch.long)
    assert torch.equal(result, expected)


def test_mask_target_before_ce_large_values():
    """Test with large target values."""
    target = torch.tensor([1000, 2000, 3000], dtype=torch.long)
    mask = torch.tensor([True, False, True], dtype=torch.bool)
    result = mask_target_before_ce(target, mask)
    expected = torch.tensor([1000, -100, 3000], dtype=torch.long)
    assert torch.equal(result, expected)
