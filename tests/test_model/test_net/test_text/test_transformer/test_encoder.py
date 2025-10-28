"""Unit tests for text transformer encoder."""

import hydra
import rootutils
import torch
from hydra import compose, initialize_config_dir

# Setup root directory
PROJECT_ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
CONFIG_DIR = str(PROJECT_ROOT / "configs")


def test_encoder_basic_instantiation():
    """Test basic encoder instantiation from config.

    Verifies that the encoder can be instantiated with default parameters.
    """
    with initialize_config_dir(
        version_base="1.3", config_dir=CONFIG_DIR + "/model/net/text/transformer"
    ):
        cfg = compose(config_name="encoder.yaml")

        # Instantiate encoder
        encoder = hydra.utils.instantiate(cfg)

        assert encoder is not None
        assert hasattr(encoder, "forward")


def test_encoder_forward_pass():
    """Test forward pass with standard input shape.

    Verifies that encoder takes token IDs [B, L] and returns logits [B, L, V].
    """
    with initialize_config_dir(
        version_base="1.3", config_dir=CONFIG_DIR + "/model/net/text/transformer"
    ):
        cfg = compose(config_name="encoder.yaml")

        encoder = hydra.utils.instantiate(cfg)

        # Create sample input: batch_size=2, seq_len=10
        batch_size = 2
        seq_len = 10
        vocab_size = cfg.num_tokens

        # Token IDs should be in range [0, vocab_size)
        token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

        # Forward pass
        logits = encoder(token_ids)

        # Check output shape [B, L, V]
        assert logits.shape == (batch_size, seq_len, vocab_size)
        assert logits.dtype == torch.float32


def test_encoder_with_different_batch_sizes():
    """Test encoder with various batch sizes.

    Verifies that encoder handles different batch sizes correctly.
    """
    with initialize_config_dir(
        version_base="1.3", config_dir=CONFIG_DIR + "/model/net/text/transformer"
    ):
        cfg = compose(config_name="encoder.yaml")

        encoder = hydra.utils.instantiate(cfg)

        seq_len = 20
        vocab_size = cfg.num_tokens

        # Test different batch sizes
        for batch_size in [1, 4, 8]:
            token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
            logits = encoder(token_ids)
            assert logits.shape == (batch_size, seq_len, vocab_size)


def test_encoder_with_max_sequence_length():
    """Test encoder with maximum sequence length.

    Verifies that encoder can handle sequences at maximum length.
    """
    with initialize_config_dir(
        version_base="1.3", config_dir=CONFIG_DIR + "/model/net/text/transformer"
    ):
        cfg = compose(config_name="encoder.yaml")

        encoder = hydra.utils.instantiate(cfg)

        batch_size = 2
        max_seq_len = cfg.max_seq_len
        vocab_size = cfg.num_tokens

        token_ids = torch.randint(0, vocab_size, (batch_size, max_seq_len))
        logits = encoder(token_ids)

        assert logits.shape == (batch_size, max_seq_len, vocab_size)


def test_encoder_with_custom_parameters():
    """Test encoder with overridden parameters.

    Verifies that encoder can be instantiated with custom configuration.
    """
    with initialize_config_dir(
        version_base="1.3", config_dir=CONFIG_DIR + "/model/net/text/transformer"
    ):
        # Override parameters
        cfg = compose(
            config_name="encoder.yaml",
            overrides=[
                "num_tokens=10000",
                "max_seq_len=128",
                "attn_layers.dim=256",
                "attn_layers.depth=4",
                "attn_layers.heads=4",
            ],
        )

        encoder = hydra.utils.instantiate(cfg)

        batch_size = 2
        seq_len = 50
        vocab_size = 10000  # overridden value

        token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        logits = encoder(token_ids)

        assert logits.shape == (batch_size, seq_len, vocab_size)


def test_encoder_gradient_flow():
    """Test that gradients flow through the encoder.

    Verifies that the encoder can be used for training (gradients are computed).
    """
    with initialize_config_dir(
        version_base="1.3", config_dir=CONFIG_DIR + "/model/net/text/transformer"
    ):
        cfg = compose(config_name="encoder.yaml")

        encoder = hydra.utils.instantiate(cfg)

        batch_size = 2
        seq_len = 10
        vocab_size = cfg.num_tokens

        token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

        # Forward pass
        logits = encoder(token_ids)

        # Create dummy loss and backpropagate
        loss = logits.sum()
        loss.backward()

        # Check that at least some parameters have gradients
        has_gradients = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in encoder.parameters()
            if p.requires_grad
        )
        assert has_gradients
