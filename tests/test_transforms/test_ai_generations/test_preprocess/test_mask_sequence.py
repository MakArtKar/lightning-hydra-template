"""Unit tests for mask_sequence function."""

from unittest.mock import Mock

import pytest
import torch

from ml_core.transforms.ai_generations.preprocess.mask_sequence import mask_sequence


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer with mask_token_id."""
    tokenizer = Mock()
    tokenizer.mask_token_id = 103  # Common BERT mask token ID
    return tokenizer


def test_mask_sequence_basic(mock_tokenizer):
    """Test basic mask application to sequence."""
    tensor = torch.tensor([1, 2, 3, 4, 5])
    mask = torch.tensor([False, True, False, True, False])
    result = mask_sequence(tensor, mask, mock_tokenizer)
    expected = torch.tensor([1, 103, 3, 103, 5])
    assert torch.equal(result, expected)


def test_mask_sequence_no_masking(mock_tokenizer):
    """Test with all False mask (no masking)."""
    tensor = torch.tensor([1, 2, 3, 4, 5])
    mask = torch.zeros_like(tensor, dtype=torch.bool)
    result = mask_sequence(tensor, mask, mock_tokenizer)
    assert torch.equal(result, tensor)


def test_mask_sequence_full_masking(mock_tokenizer):
    """Test with all True mask (full masking)."""
    tensor = torch.tensor([1, 2, 3, 4, 5])
    mask = torch.ones_like(tensor, dtype=torch.bool)
    result = mask_sequence(tensor, mask, mock_tokenizer)
    expected = torch.full_like(tensor, mock_tokenizer.mask_token_id)
    assert torch.equal(result, expected)


def test_mask_sequence_2d_tensor(mock_tokenizer):
    """Test with 2D tensor (batch of sequences)."""
    tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
    mask = torch.tensor([[True, False, True], [False, True, False]])
    result = mask_sequence(tensor, mask, mock_tokenizer)
    expected = torch.tensor([[103, 2, 103], [4, 103, 6]])
    assert torch.equal(result, expected)


def test_mask_sequence_bool_mask_coerced_to_float(mock_tokenizer):
    """Test that bool mask is correctly converted to float32 internally."""
    tensor = torch.tensor([10, 20, 30, 40])
    mask = torch.tensor([True, False, True, False])
    result = mask_sequence(tensor, mask, mock_tokenizer)
    # Formula: tensor * (1 - mask_float) + mask_token_id * mask_float
    # bool True becomes 1.0, False becomes 0.0
    expected = torch.tensor([103, 20, 103, 40])
    assert torch.equal(result, expected)


def test_mask_sequence_different_mask_token_id():
    """Test with different mask_token_id."""
    tokenizer = Mock()
    tokenizer.mask_token_id = 999
    tensor = torch.tensor([1, 2, 3])
    mask = torch.tensor([True, False, True])
    result = mask_sequence(tensor, mask, tokenizer)
    expected = torch.tensor([999, 2, 999])
    assert torch.equal(result, expected)


def test_mask_sequence_preserves_dtype(mock_tokenizer):
    """Test that output preserves input dtype when possible."""
    tensor = torch.tensor([1, 2, 3, 4], dtype=torch.long)
    mask = torch.tensor([False, True, False, True])
    result = mask_sequence(tensor, mask, mock_tokenizer)
    # Result dtype should be compatible with computation
    assert result.shape == tensor.shape


def test_mask_sequence_empty_tensor(mock_tokenizer):
    """Test with empty tensor."""
    tensor = torch.tensor([], dtype=torch.long)
    mask = torch.tensor([], dtype=torch.bool)
    result = mask_sequence(tensor, mask, mock_tokenizer)
    assert result.shape == (0,)


def test_mask_sequence_single_element(mock_tokenizer):
    """Test with single element."""
    tensor = torch.tensor([42])
    mask = torch.tensor([True])
    result = mask_sequence(tensor, mask, mock_tokenizer)
    expected = torch.tensor([103])
    assert torch.equal(result, expected)
