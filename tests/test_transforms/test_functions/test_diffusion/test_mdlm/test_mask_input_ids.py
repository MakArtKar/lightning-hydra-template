"""Unit tests for MaskInputIds function."""

import pytest
import torch

from ml_core.transforms.diffusion.mdlm import MaskInputIds


@pytest.fixture
def mask_input_ids():
    """Create a MaskInputIds instance with mask_token_id=103.
    
    :return: MaskInputIds instance.
    """
    return MaskInputIds(mask_token_id=103)


def test_mask_input_ids_shape(mask_input_ids):
    """Test that MaskInputIds returns correct shape.
    
    :param mask_input_ids: MaskInputIds fixture.
    """
    batch_size = 4
    seq_length = 10
    
    input_ids = torch.randint(0, 100, (batch_size, seq_length))
    mask = torch.rand(batch_size, seq_length) > 0.5
    
    masked_ids = mask_input_ids(input_ids, mask)
    
    # Check shape is preserved
    assert masked_ids.shape == input_ids.shape


def test_mask_input_ids_applies_mask(mask_input_ids):
    """Test that MaskInputIds correctly masks positions.
    
    :param mask_input_ids: MaskInputIds fixture.
    """
    batch_size = 2
    seq_length = 10
    
    input_ids = torch.randint(0, 100, (batch_size, seq_length))
    
    # Create a specific mask pattern
    mask = torch.zeros(batch_size, seq_length, dtype=torch.bool)
    mask[0, 2] = True
    mask[0, 5] = True
    mask[1, 3] = True
    
    masked_ids = mask_input_ids(input_ids, mask)
    
    # Check that masked positions have mask_token_id
    assert masked_ids[0, 2] == 103
    assert masked_ids[0, 5] == 103
    assert masked_ids[1, 3] == 103
    
    # Check that unmasked positions are unchanged
    assert masked_ids[0, 0] == input_ids[0, 0]
    assert masked_ids[0, 1] == input_ids[0, 1]
    assert masked_ids[1, 0] == input_ids[1, 0]


def test_mask_input_ids_no_mask():
    """Test that MaskInputIds preserves input when no mask is applied."""
    mask_fn = MaskInputIds(mask_token_id=103)
    
    batch_size = 2
    seq_length = 10
    
    input_ids = torch.randint(0, 100, (batch_size, seq_length))
    mask = torch.zeros(batch_size, seq_length, dtype=torch.bool)
    
    masked_ids = mask_fn(input_ids, mask)
    
    # Check that nothing is masked
    assert torch.all(masked_ids == input_ids)


def test_mask_input_ids_all_mask():
    """Test that MaskInputIds masks all positions when all mask is True."""
    mask_fn = MaskInputIds(mask_token_id=103)
    
    batch_size = 2
    seq_length = 10
    
    input_ids = torch.randint(0, 100, (batch_size, seq_length))
    mask = torch.ones(batch_size, seq_length, dtype=torch.bool)
    
    masked_ids = mask_fn(input_ids, mask)
    
    # Check that all positions are masked
    assert torch.all(masked_ids == 103)


def test_mask_input_ids_does_not_modify_original():
    """Test that MaskInputIds doesn't modify the original input_ids."""
    mask_fn = MaskInputIds(mask_token_id=103)
    
    batch_size = 2
    seq_length = 10
    
    input_ids = torch.randint(0, 100, (batch_size, seq_length))
    input_ids_copy = input_ids.clone()
    mask = torch.ones(batch_size, seq_length, dtype=torch.bool)
    
    masked_ids = mask_fn(input_ids, mask)
    
    # Check that original input_ids is unchanged
    assert torch.all(input_ids == input_ids_copy)
    
    # Check that masked_ids is different
    assert not torch.all(masked_ids == input_ids)


def test_mask_input_ids_with_tokenizer():
    """Test that MaskInputIds can extract mask_token_id from tokenizer."""
    # Create a mock tokenizer with mask_token_id attribute
    class MockTokenizer:
        mask_token_id = 42
    
    tokenizer = MockTokenizer()
    mask_fn = MaskInputIds(tokenizer=tokenizer)
    
    batch_size = 2
    seq_length = 10
    
    input_ids = torch.randint(0, 100, (batch_size, seq_length))
    mask = torch.ones(batch_size, seq_length, dtype=torch.bool)
    
    masked_ids = mask_fn(input_ids, mask)
    
    # Check that mask_token_id from tokenizer is used
    assert torch.all(masked_ids == 42)


def test_mask_input_ids_requires_param():
    """Test that MaskInputIds raises error when neither param is provided."""
    with pytest.raises(ValueError, match="Either mask_token_id or tokenizer must be provided"):
        MaskInputIds()


def test_mask_input_ids_device(mask_input_ids):
    """Test that MaskInputIds preserves device.
    
    :param mask_input_ids: MaskInputIds fixture.
    """
    batch_size = 2
    seq_length = 10
    
    # Test with CPU
    input_ids = torch.randint(0, 100, (batch_size, seq_length))
    mask = torch.rand(batch_size, seq_length) > 0.5
    masked_ids = mask_input_ids(input_ids, mask)
    assert masked_ids.device == input_ids.device
    
    # Test with CUDA if available
    if torch.cuda.is_available():
        input_ids_cuda = input_ids.cuda()
        mask_cuda = mask.cuda()
        masked_ids_cuda = mask_input_ids(input_ids_cuda, mask_cuda)
        assert masked_ids_cuda.device == input_ids_cuda.device


def test_mask_input_ids_different_batch_sizes():
    """Test that MaskInputIds works with different batch sizes."""
    mask_fn = MaskInputIds(mask_token_id=103)
    seq_length = 10
    
    for batch_size in [1, 2, 8]:
        input_ids = torch.randint(0, 100, (batch_size, seq_length))
        mask = torch.rand(batch_size, seq_length) > 0.5
        masked_ids = mask_fn(input_ids, mask)
        assert masked_ids.shape == torch.Size([batch_size, seq_length])

