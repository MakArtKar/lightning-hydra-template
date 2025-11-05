"""Unit tests for tokenizer transform."""

import os

import pytest
import rootutils
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

# Setup root to get PROJECT_ROOT
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
PROJECT_ROOT = os.environ["PROJECT_ROOT"]


@pytest.fixture(scope="module")
def tokenizer_transform():
    """Fixture that loads config and returns a WrapTransform object.

    :return: WrapTransform instance configured with tokenizer.
    """
    GlobalHydra.instance().clear()

    config_path = os.path.join(PROJECT_ROOT, "configs")

    # Define test parameters
    max_length = 128
    tokenizer_model = "bert-base-uncased"

    with initialize_config_dir(version_base="1.3", config_dir=config_path):
        # Load the tokenizer transform config and provide params via overrides
        # Use ++ to force-add new params keys since config is in struct mode
        # Use _target_ to instantiate tokenizer from Hydra (strings are primitives)
        cfg = compose(
            config_name="data/transform/tokenizer",
            overrides=[
                f"++params.data.max_length={max_length}",
                "++params.data.tokenizer._target_=transformers.AutoTokenizer.from_pretrained",
                f"++params.data.tokenizer.pretrained_model_name_or_path={tokenizer_model}",
            ],
        )

        # Instantiate the transform using Hydra
        from hydra.utils import instantiate

        transform = instantiate(cfg.data.transform)

        yield transform

    GlobalHydra.instance().clear()


def test_tokenizer_transform_basic(tokenizer_transform):
    """Test tokenizer transform on a simple batch with single text.

    :param tokenizer_transform: The tokenizer transform fixture.
    """
    # Create a dummy batch
    batch = {"text": "This is a test sentence."}

    # Apply the transform
    output = tokenizer_transform(batch)

    # Check that the original key is still present
    assert "text" in output
    assert output["text"] == "This is a test sentence."

    # Check that tokenized outputs are present
    assert "input_ids" in output
    assert "attention_mask" in output

    # Check that outputs are tensors
    assert isinstance(output["input_ids"], torch.Tensor)
    assert isinstance(output["attention_mask"], torch.Tensor)

    # Check shape (should be [1, max_length] because return_tensors="pt" and padding="max_length")
    assert output["input_ids"].shape == torch.Size([1, 128])
    assert output["attention_mask"].shape == torch.Size([1, 128])


def test_tokenizer_transform_batch(tokenizer_transform):
    """Test tokenizer transform on a batch with multiple texts.

    :param tokenizer_transform: The tokenizer transform fixture.
    """
    # Create a batch with multiple texts
    texts = [
        "This is the first sentence.",
        "Here is another one.",
        "And a third sentence for testing.",
    ]
    batch = {"text": texts}

    # Apply the transform
    output = tokenizer_transform(batch)

    # Check that the original key is still present
    assert "text" in output
    assert output["text"] == texts

    # Check that tokenized outputs are present
    assert "input_ids" in output
    assert "attention_mask" in output

    # Check that outputs are tensors
    assert isinstance(output["input_ids"], torch.Tensor)
    assert isinstance(output["attention_mask"], torch.Tensor)

    # Check shape (should be [3, max_length])
    assert output["input_ids"].shape == torch.Size([3, 128])
    assert output["attention_mask"].shape == torch.Size([3, 128])


def test_tokenizer_transform_truncation(tokenizer_transform):
    """Test tokenizer transform truncates long text correctly.

    :param tokenizer_transform: The tokenizer transform fixture.
    """
    # Create a very long text (more than 128 tokens)
    long_text = " ".join(["word"] * 200)
    batch = {"text": long_text}

    # Apply the transform
    output = tokenizer_transform(batch)

    # Check that output is truncated to max_length
    assert output["input_ids"].shape == torch.Size([1, 128])
    assert output["attention_mask"].shape == torch.Size([1, 128])


def test_tokenizer_transform_padding(tokenizer_transform):
    """Test tokenizer transform pads short text correctly.

    :param tokenizer_transform: The tokenizer transform fixture.
    """
    # Create a very short text
    short_text = "Hi"
    batch = {"text": short_text}

    # Apply the transform
    output = tokenizer_transform(batch)

    # Check that output is padded to max_length
    assert output["input_ids"].shape == torch.Size([1, 128])
    assert output["attention_mask"].shape == torch.Size([1, 128])

    # Check that attention mask has 0s for padding
    # The attention mask should have some 0s at the end
    attention_sum = output["attention_mask"].sum().item()
    assert attention_sum < 128  # Should be less than max_length due to padding


def test_tokenizer_transform_empty_text(tokenizer_transform):
    """Test tokenizer transform handles empty text.

    :param tokenizer_transform: The tokenizer transform fixture.
    """
    # Create a batch with empty text
    batch = {"text": ""}

    # Apply the transform
    output = tokenizer_transform(batch)

    # Check that outputs are present and correctly shaped
    assert "input_ids" in output
    assert "attention_mask" in output
    assert output["input_ids"].shape == torch.Size([1, 128])
    assert output["attention_mask"].shape == torch.Size([1, 128])


def test_tokenizer_transform_preserves_other_keys(tokenizer_transform):
    """Test that tokenizer transform preserves other keys in the batch.

    :param tokenizer_transform: The tokenizer transform fixture.
    """
    # Create a batch with additional keys
    batch = {
        "text": "This is a test.",
        "label": 5,
        "metadata": {"id": 123},
    }

    # Apply the transform
    output = tokenizer_transform(batch)

    # Check that all original keys are preserved
    assert "text" in output
    assert "label" in output
    assert "metadata" in output

    # Check that original values are unchanged
    assert output["label"] == 5
    assert output["metadata"] == {"id": 123}

    # Check that new keys are added
    assert "input_ids" in output
    assert "attention_mask" in output
