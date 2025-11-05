"""Unit tests for MaskLoss function."""

import pytest
import torch

from ml_core.transforms.diffusion.mdlm import MaskLoss


@pytest.fixture
def mask_loss():
    """Create a MaskLoss instance with default ignore_index=-100.

    :return: MaskLoss instance.
    """
    return MaskLoss()


def test_mask_loss_shape(mask_loss):
    """Test that MaskLoss returns correct shape.

    :param mask_loss: MaskLoss fixture.
    """
    batch_size = 4
    seq_length = 10

    target = torch.randint(0, 100, (batch_size, seq_length))
    mask = torch.rand(batch_size, seq_length) > 0.5

    masked_target = mask_loss(target, mask)

    # Check shape is preserved
    assert masked_target.shape == target.shape


def test_mask_loss_applies_ignore_index(mask_loss):
    """Test that MaskLoss correctly applies ignore_index to non-masked positions.

    :param mask_loss: MaskLoss fixture.
    """
    batch_size = 2
    seq_length = 10

    target = torch.randint(0, 100, (batch_size, seq_length))

    # Create a specific mask pattern
    mask = torch.zeros(batch_size, seq_length, dtype=torch.bool)
    mask[0, 2] = True
    mask[0, 5] = True
    mask[1, 3] = True

    masked_target = mask_loss(target, mask)

    # Check that masked positions (True) preserve original values
    assert masked_target[0, 2] == target[0, 2]
    assert masked_target[0, 5] == target[0, 5]
    assert masked_target[1, 3] == target[1, 3]

    # Check that non-masked positions (False) have ignore_index
    assert masked_target[0, 0] == -100
    assert masked_target[0, 1] == -100
    assert masked_target[1, 0] == -100


def test_mask_loss_no_mask():
    """Test that MaskLoss applies ignore_index to all positions when no mask is applied."""
    mask_fn = MaskLoss()

    batch_size = 2
    seq_length = 10

    target = torch.randint(0, 100, (batch_size, seq_length))
    mask = torch.zeros(batch_size, seq_length, dtype=torch.bool)

    masked_target = mask_fn(target, mask)

    # Check that all positions have ignore_index (since none are masked)
    assert torch.all(masked_target == -100)


def test_mask_loss_all_mask():
    """Test that MaskLoss preserves all values when all mask is True."""
    mask_fn = MaskLoss()

    batch_size = 2
    seq_length = 10

    target = torch.randint(0, 100, (batch_size, seq_length))
    mask = torch.ones(batch_size, seq_length, dtype=torch.bool)

    masked_target = mask_fn(target, mask)

    # Check that all positions are preserved (since all are masked)
    assert torch.all(masked_target == target)


def test_mask_loss_does_not_modify_original():
    """Test that MaskLoss doesn't modify the original target."""
    mask_fn = MaskLoss()

    batch_size = 2
    seq_length = 10

    target = torch.randint(0, 100, (batch_size, seq_length))
    target_copy = target.clone()
    mask = torch.zeros(batch_size, seq_length, dtype=torch.bool)

    masked_target = mask_fn(target, mask)

    # Check that original target is unchanged
    assert torch.all(target == target_copy)

    # Check that masked_target is different
    assert not torch.all(masked_target == target)


def test_mask_loss_custom_ignore_index():
    """Test that MaskLoss works with custom ignore_index."""
    custom_ignore_index = -999
    mask_fn = MaskLoss(ignore_index=custom_ignore_index)

    batch_size = 2
    seq_length = 10

    target = torch.randint(0, 100, (batch_size, seq_length))
    mask = torch.zeros(batch_size, seq_length, dtype=torch.bool)
    mask[0, 2] = True

    masked_target = mask_fn(target, mask)

    # Check that non-masked positions have custom ignore_index
    assert masked_target[0, 0] == custom_ignore_index
    assert masked_target[0, 1] == custom_ignore_index

    # Check that masked position preserves original value
    assert masked_target[0, 2] == target[0, 2]


def test_mask_loss_device(mask_loss):
    """Test that MaskLoss preserves device.

    :param mask_loss: MaskLoss fixture.
    """
    batch_size = 2
    seq_length = 10

    # Test with CPU
    target = torch.randint(0, 100, (batch_size, seq_length))
    mask = torch.rand(batch_size, seq_length) > 0.5
    masked_target = mask_loss(target, mask)
    assert masked_target.device == target.device

    # Test with CUDA if available
    if torch.cuda.is_available():
        target_cuda = target.cuda()
        mask_cuda = mask.cuda()
        masked_target_cuda = mask_loss(target_cuda, mask_cuda)
        assert masked_target_cuda.device == target_cuda.device


def test_mask_loss_different_batch_sizes():
    """Test that MaskLoss works with different batch sizes."""
    mask_fn = MaskLoss()
    seq_length = 10

    for batch_size in [1, 2, 8]:
        target = torch.randint(0, 100, (batch_size, seq_length))
        mask = torch.rand(batch_size, seq_length) > 0.5
        masked_target = mask_fn(target, mask)
        assert masked_target.shape == torch.Size([batch_size, seq_length])


def test_mask_loss_partial_mask():
    """Test that MaskLoss correctly handles partial masking."""
    mask_fn = MaskLoss()

    batch_size = 2
    seq_length = 10

    target = torch.randint(0, 100, (batch_size, seq_length))

    # Create a partial mask (half masked, half not)
    mask = torch.zeros(batch_size, seq_length, dtype=torch.bool)
    mask[:, :5] = True  # First half is masked

    masked_target = mask_fn(target, mask)

    # Check that masked positions preserve original values
    assert torch.all(masked_target[:, :5] == target[:, :5])

    # Check that non-masked positions have ignore_index
    assert torch.all(masked_target[:, 5:] == -100)


def test_mask_loss_dtype_preservation():
    """Test that MaskLoss preserves dtype of input."""
    mask_fn = MaskLoss()

    batch_size = 2
    seq_length = 10

    # Test with different dtypes
    for dtype in [torch.int32, torch.int64]:
        target = torch.randint(0, 100, (batch_size, seq_length), dtype=dtype)
        mask = torch.rand(batch_size, seq_length) > 0.5

        masked_target = mask_fn(target, mask)

        # Check that dtype is preserved
        assert masked_target.dtype == target.dtype


def test_mask_loss_edge_case_single_masked_position():
    """Test that MaskLoss works correctly with single masked position."""
    mask_fn = MaskLoss()

    batch_size = 2
    seq_length = 10

    target = torch.randint(0, 100, (batch_size, seq_length))

    # Create mask with only one True position
    mask = torch.zeros(batch_size, seq_length, dtype=torch.bool)
    mask[1, 7] = True

    masked_target = mask_fn(target, mask)

    # Check that the single masked position preserves original value
    assert masked_target[1, 7] == target[1, 7]

    # Check that all other positions have ignore_index
    non_masked_positions = ~mask
    assert torch.all(masked_target[non_masked_positions] == -100)
