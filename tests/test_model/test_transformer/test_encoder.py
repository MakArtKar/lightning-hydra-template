"""Unit tests for transformer encoder architecture."""

import os

import pytest
import rootutils
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

# Setup root to get PROJECT_ROOT
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
PROJECT_ROOT = os.environ["PROJECT_ROOT"]


@pytest.fixture(scope="module")
def encoder():
    """Fixture that loads config and returns an Encoder object.

    :return: Encoder instance from x_transformers.
    """
    GlobalHydra.instance().clear()

    config_path = os.path.join(PROJECT_ROOT, "configs")
    
    # Define test parameters
    dim = 16
    depth = 2
    heads = 4
    vocab_size = 200
    max_length = 128
    
    with initialize_config_dir(version_base="1.3", config_dir=config_path):
        # Load the encoder config and provide params via overrides
        cfg = compose(
            config_name="model/transformer/encoder",
            overrides=[
                f"++params.model.dim={dim}",
                f"++params.model.depth={depth}",
                f"++params.model.heads={heads}",
                f"++params.model.num_tokens={vocab_size}",
                f"++params.data.max_length={max_length}",
            ],
        )
        
        # Instantiate the encoder using Hydra
        from hydra.utils import instantiate
        cfg = instantiate(cfg)
        
        yield cfg.model.transformer
    
    GlobalHydra.instance().clear()


def test_encoder_forward(encoder):
    """Test encoder forward pass produces correct output shape.

    :param encoder: The encoder fixture.
    """
    batch_size = 2
    seq_length = 10
    vocab_size = 200
    
    # Create dummy input (token IDs)
    x = torch.randint(0, vocab_size, (batch_size, seq_length))
    
    # Forward pass
    output = encoder(x)
    
    # Check output shape (TransformerWrapper returns logits: [batch, seq, vocab])
    assert output.shape == torch.Size([batch_size, seq_length, vocab_size])


def test_encoder_with_mask(encoder):
    """Test encoder forward pass with attention mask.

    :param encoder: The encoder fixture.
    """
    batch_size = 2
    seq_length = 10
    vocab_size = 200
    
    # Create dummy input (token IDs) and mask
    x = torch.randint(0, vocab_size, (batch_size, seq_length))
    mask = torch.ones(batch_size, seq_length, dtype=torch.bool)
    # Mask out last 3 tokens for first batch element
    mask[0, -3:] = False
    
    # Forward pass with mask
    output = encoder(x, mask=mask)
    
    # Check output shape (TransformerWrapper returns logits: [batch, seq, vocab])
    assert output.shape == torch.Size([batch_size, seq_length, vocab_size])


def test_encoder_different_seq_lengths(encoder):
    """Test encoder handles different sequence lengths.

    :param encoder: The encoder fixture.
    """
    vocab_size = 200
    
    # Test with different sequence lengths
    for seq_length in [5, 10, 20]:
        x = torch.randint(0, vocab_size, (1, seq_length))
        output = encoder(x)
        assert output.shape == torch.Size([1, seq_length, vocab_size])


def test_encoder_batch_processing(encoder):
    """Test encoder processes batches correctly.

    :param encoder: The encoder fixture.
    """
    vocab_size = 200
    seq_length = 10
    
    # Test with different batch sizes
    for batch_size in [1, 4, 8]:
        x = torch.randint(0, vocab_size, (batch_size, seq_length))
        output = encoder(x)
        assert output.shape == torch.Size([batch_size, seq_length, vocab_size])

