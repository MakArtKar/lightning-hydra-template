"""Unit tests for MDLM model architecture."""

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
def mdlm_forward_fn():
    """Fixture that loads config and returns the forward_fn.

    :return: forward_fn from mdlm model.
    """
    GlobalHydra.instance().clear()

    config_path = os.path.join(PROJECT_ROOT, "configs")

    # Define test parameters
    dim = 16
    depth = 2
    heads = 4
    vocab_size = 200  # Large enough for mask_token_id (typically 103)
    max_length = 128
    tokenizer_model = "bert-base-uncased"

    with initialize_config_dir(version_base="1.3", config_dir=config_path):
        # Load the mdlm config and override values directly
        cfg = compose(
            config_name="model/diffusion/mdlm",
            overrides=[
                # Provide params that the interpolations reference
                f"++params.model.dim={dim}",
                f"++params.model.depth={depth}",
                f"++params.model.heads={heads}",
                f"++params.model.num_tokens={vocab_size}",
                # Provide data params
                f"++params.data.max_length={max_length}",
                # Provide tokenizer for MaskInputIds
                "++params.data.tokenizer._target_=transformers.AutoTokenizer.from_pretrained",
                f"++params.data.tokenizer.pretrained_model_name_or_path={tokenizer_model}",
            ],
        )

        # Instantiate the entire config to properly resolve all interpolations
        from hydra.utils import instantiate

        cfg = instantiate(cfg)

        yield cfg.model.diffusion.forward_fn

    GlobalHydra.instance().clear()


def test_mdlm_forward_fn_keys(mdlm_forward_fn):
    """Test that forward_fn produces correct output keys.

    :param mdlm_forward_fn: The mdlm forward_fn fixture.
    """
    batch_size = 2
    seq_length = 10
    vocab_size = 200

    # Create a dummy batch
    batch = {
        "input_ids": torch.randint(0, vocab_size, (batch_size, seq_length)),
        "attention_mask": torch.ones(batch_size, seq_length, dtype=torch.bool),
    }

    # Forward pass
    output = mdlm_forward_fn(batch)

    # Check for expected keys
    assert "input_ids" in output
    assert "attention_mask" in output
    assert "mask" in output
    assert "masked_input_ids" in output
    assert "logits" in output
    assert "reshaped_logits" in output


def test_mdlm_forward_fn_shapes(mdlm_forward_fn):
    """Test that forward_fn produces correct output shapes.

    :param mdlm_forward_fn: The mdlm forward_fn fixture.
    """
    batch_size = 2
    seq_length = 10
    vocab_size = 200

    # Create a dummy batch
    batch = {
        "input_ids": torch.randint(0, vocab_size, (batch_size, seq_length)),
        "attention_mask": torch.ones(batch_size, seq_length, dtype=torch.bool),
    }

    # Forward pass
    output = mdlm_forward_fn(batch)

    # Check shapes
    assert output["mask"].shape == (batch_size, seq_length)
    assert output["masked_input_ids"].shape == (batch_size, seq_length)
    assert output["logits"].shape == (batch_size, seq_length, vocab_size)
    assert output["reshaped_logits"].shape == (batch_size, vocab_size, seq_length)
