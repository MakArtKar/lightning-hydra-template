"""Unit tests for SampleMask function."""

import pytest
import torch

from ml_core.transforms.diffusion.mdlm import SampleMask


@pytest.fixture
def sample_mask():
    """Create a SampleMask instance.
    
    :return: SampleMask instance.
    """
    return SampleMask()


def test_sample_mask_shape(sample_mask):
    """Test that SampleMask returns correct shape.
    
    :param sample_mask: SampleMask fixture.
    """
    batch_size = 4
    seq_length = 10
    
    input_ids = torch.randint(0, 100, (batch_size, seq_length))
    mask = sample_mask(input_ids)
    
    # Check shape
    assert mask.shape == torch.Size([batch_size, seq_length])
    
    # Check dtype
    assert mask.dtype == torch.bool


def test_sample_mask_has_masked_tokens(sample_mask):
    """Test that SampleMask generates at least some masks.
    
    :param sample_mask: SampleMask fixture.
    """
    batch_size = 4
    seq_length = 20
    
    # Run multiple times to check probabilistic behavior
    num_trials = 10
    total_masked = 0
    
    for _ in range(num_trials):
        input_ids = torch.randint(0, 100, (batch_size, seq_length))
        mask = sample_mask(input_ids)
        total_masked += mask.sum().item()
    
    # With high probability, at least some tokens should be masked across all trials
    assert total_masked > 0


def test_sample_mask_boolean_values(sample_mask):
    """Test that SampleMask returns boolean tensor.
    
    :param sample_mask: SampleMask fixture.
    """
    batch_size = 2
    seq_length = 10
    
    input_ids = torch.randint(0, 100, (batch_size, seq_length))
    mask = sample_mask(input_ids)
    
    # Check all values are 0 or 1
    assert torch.all((mask == 0) | (mask == 1))


def test_sample_mask_different_batch_sizes(sample_mask):
    """Test that SampleMask works with different batch sizes.
    
    :param sample_mask: SampleMask fixture.
    """
    seq_length = 10
    
    for batch_size in [1, 2, 8]:
        input_ids = torch.randint(0, 100, (batch_size, seq_length))
        mask = sample_mask(input_ids)
        assert mask.shape == torch.Size([batch_size, seq_length])


def test_sample_mask_different_seq_lengths(sample_mask):
    """Test that SampleMask works with different sequence lengths.
    
    :param sample_mask: SampleMask fixture.
    """
    batch_size = 4
    
    for seq_length in [5, 10, 20, 50]:
        input_ids = torch.randint(0, 100, (batch_size, seq_length))
        mask = sample_mask(input_ids)
        assert mask.shape == torch.Size([batch_size, seq_length])


def test_sample_mask_device(sample_mask):
    """Test that SampleMask preserves device.
    
    :param sample_mask: SampleMask fixture.
    """
    batch_size = 2
    seq_length = 10
    
    # Test with CPU
    input_ids = torch.randint(0, 100, (batch_size, seq_length))
    mask = sample_mask(input_ids)
    assert mask.device == input_ids.device
    
    # Test with CUDA if available
    if torch.cuda.is_available():
        input_ids_cuda = input_ids.cuda()
        mask_cuda = sample_mask(input_ids_cuda)
        assert mask_cuda.device == input_ids_cuda.device


def test_sample_mask_not_all_masked(sample_mask):
    """Test that SampleMask doesn't mask all tokens (probabilistically).
    
    :param sample_mask: SampleMask fixture.
    """
    batch_size = 4
    seq_length = 20
    
    # Run multiple times
    all_masked_count = 0
    num_trials = 50
    
    for _ in range(num_trials):
        input_ids = torch.randint(0, 100, (batch_size, seq_length))
        mask = sample_mask(input_ids)
        
        # Check if any batch element has all tokens masked
        for b in range(batch_size):
            if mask[b].all():
                all_masked_count += 1
    
    # It's extremely unlikely that all tokens are masked in most trials
    # (would require k=L and all Bernoulli samples to be 1)
    assert all_masked_count < num_trials * batch_size * 0.5

