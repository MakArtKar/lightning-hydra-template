"""Tests for LM1B dataset configuration and tokenization."""

from typing import Dict, List, Optional

import hydra
import pytest
import rootutils
import torch
from datasets import Dataset, DatasetDict, load_dataset
from hydra import compose, initialize_config_dir

from ml_core.data.base_datamodule import BaseDataModule

# Setup root directory
PROJECT_ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
CONFIG_DIR = str(PROJECT_ROOT / "configs" / "data")


@pytest.fixture(scope="module")
def lm1b_dataset_path(tmp_path_factory):
    """Download and save a limited LM1B dataset for testing.
    
    Downloads only 100 train and 10 test samples once per module,
    saves to disk, and returns the path for Hydra config override.
    """
    tmp_dir = tmp_path_factory.mktemp("lm1b_test_data")
    dataset_dict = load_dataset("dvruette/lm1b", split={"train": "train[:100]", "test": "test[:10]"})
    dataset_dict.save_to_disk(str(tmp_dir))
    return str(tmp_dir)


@pytest.fixture(scope="module")
def lm1b_toy_dataset_path(tmp_path_factory):
    """Create and save a tiny toy dataset for transform testing.
    
    Creates a minimal 3-sample dataset for quick transform validation,
    saves to disk, and returns the path for Hydra config override.
    """
    tmp_dir = tmp_path_factory.mktemp("lm1b_toy_data")
    small_data = {
        "text": [
            "Hello, world!",
            "This is a test sentence.",
            "Another example text for testing.",
        ]
    }
    dataset = Dataset.from_dict(small_data)
    dataset_dict = DatasetDict({"train": dataset, "test": dataset})
    dataset_dict.save_to_disk(str(tmp_dir))
    return str(tmp_dir)


def _create_datamodule(dataset_path: str, additional_overrides: Optional[List[str]] = None) -> BaseDataModule:
    """Create a BaseDataModule using Hydra config with dataset path override.
    
    :param dataset_path: Path to pre-saved dataset on disk.
    :param additional_overrides: Additional Hydra config overrides (default: batch_size=8, num_workers=0).
    :return: Instantiated BaseDataModule.
    """
    base_overrides = [
        "hf_dict_dataset._target_=datasets.load_from_disk",  # Override to load from disk
        f"++hf_dict_dataset._args_=[{dataset_path}]",  # Force add _args_ with tmp path
        "~hf_dict_dataset.path",  # Remove path kwarg (not needed for load_from_disk)
        "num_workers=0",
        "persistent_workers=False",  # Required when num_workers=0
    ]
    
    if additional_overrides:
        base_overrides.extend(additional_overrides)
    
    with initialize_config_dir(version_base="1.3", config_dir=CONFIG_DIR):
        cfg = compose(config_name="lm1b", overrides=base_overrides)
        dm = hydra.utils.instantiate(cfg)
    
    return dm


def _check_batch_correctness(
    batch: Dict[str, torch.Tensor],
    expected_batch_size: int,
    expected_seq_length: int,
    check_tokenization: bool = True,
) -> None:
    """Check if batch has correct structure, shapes, and dtypes.
    
    :param batch: Batch dictionary from dataloader.
    :param expected_batch_size: Expected batch size.
    :param expected_seq_length: Expected sequence length.
    :param check_tokenization: Whether to check tokenization outputs (input_ids, etc.).
    """
    # Check text key is preserved
    assert "text" in batch, "Original 'text' key should be preserved"
    
    if check_tokenization:
        # Check all tokenizer outputs
        tokenizer_keys = ["input_ids", "attention_mask", "token_type_ids"]
        
        for key in tokenizer_keys:
            # Check key exists
            assert key in batch, f"Tokenizer should produce '{key}'"
            
            # Check batch size
            assert len(batch[key]) == expected_batch_size, f"{key}: Batch size should be {expected_batch_size}"
            
            # Check sequence length
            assert batch[key].shape[1] == expected_seq_length, f"{key}: Sequence length should be {expected_seq_length}"
            
            # Check dtype
            assert batch[key].dtype == torch.int64, f"{key} should be int64"
        
        # Check that tokenization is meaningful (not all zeros)
        assert batch["input_ids"].sum() > 0, "input_ids should contain actual tokens"
        assert batch["attention_mask"].sum() > 0, "attention_mask should have some active tokens"
    else:
        # Check basic text structure without tokenization
        assert isinstance(batch["text"], list), "Text should be a list of strings"
        assert len(batch["text"]) > 0, "Batch should not be empty"
        assert all(isinstance(text, str) for text in batch["text"]), "All texts should be strings"


def _check_datamodule_structure(dm: BaseDataModule, expected_total_train_val: int, expected_test: int) -> None:
    """Check if BaseDataModule has correct structure and dataset splits.
    
    :param dm: BaseDataModule instance.
    :param expected_total_train_val: Expected total number of train+val samples.
    :param expected_test: Expected number of test samples.
    """
    # Check data splits exist
    assert dm.data_train is not None, "Train dataset should exist"
    assert dm.data_val is not None, "Validation dataset should exist"
    assert dm.data_test is not None, "Test dataset should exist"
    
    # Check dataloaders work
    assert dm.train_dataloader() is not None, "Train dataloader should be created"
    assert dm.val_dataloader() is not None, "Val dataloader should be created"
    assert dm.test_dataloader() is not None, "Test dataloader should be created"
    
    # Verify total number of samples
    total_train_val = len(dm.data_train) + len(dm.data_val)
    assert total_train_val == expected_total_train_val, f"Train+Val should be {expected_total_train_val}, got {total_train_val}"
    assert len(dm.data_test) == expected_test, f"Test should be {expected_test}, got {len(dm.data_test)}"


def test_tokenizer_transform_manually(lm1b_toy_dataset_path: str) -> None:
    """Test TokenizerTransform with manually created small dataset.

    Verifies:
    - Tokenizer produces expected keys (input_ids, attention_mask, token_type_ids)
    - Token shapes and dtypes are correct
    - Batch size is preserved
    
    :param lm1b_toy_dataset_path: Path to pre-saved toy dataset.
    """
    # Create datamodule with toy dataset and custom settings
    dm = _create_datamodule(
        lm1b_toy_dataset_path,
        additional_overrides=[
            "transform.tokenize.max_length=128",
            "batch_size=2",
        ]
    )
    dm.setup()

    # Get a batch and check correctness
    batch = next(iter(dm.train_dataloader()))
    _check_batch_correctness(batch, expected_batch_size=2, expected_seq_length=128, check_tokenization=True)


def test_lm1b_dataset_download_without_transform(lm1b_dataset_path: str) -> None:
    """Test LM1B dataset can be downloaded and loaded without transform.

    Verifies:
    - Dataset loads correctly from HuggingFace
    - Required splits are present
    - Basic structure is correct
    
    :param lm1b_dataset_path: Path to pre-saved limited dataset.
    """
    # Create datamodule without transform
    dm = _create_datamodule(
        lm1b_dataset_path,
        additional_overrides=[
            "transform=null",
            "batch_size=8",
        ]
    )
    dm.setup()

    # Check datamodule structure
    _check_datamodule_structure(dm, expected_total_train_val=100, expected_test=10)

    # Get a batch and check basic structure (no tokenization)
    batch = next(iter(dm.train_dataloader()))
    _check_batch_correctness(batch, expected_batch_size=8, expected_seq_length=0, check_tokenization=False)


@pytest.mark.parametrize("batch_size", [16, 32])
def test_lm1b_full_setup(lm1b_dataset_path: str, batch_size: int) -> None:
    """Test LM1B dataset with full tokenization setup using Hydra config.

    Verifies:
    - Complete pipeline works end-to-end
    - All dataloaders are functional
    - Tokenization is applied correctly
    - Batch sizes are correct

    :param lm1b_dataset_path: Path to pre-saved limited dataset.
    :param batch_size: Batch size to test with.
    """
    # Create datamodule with full tokenization
    dm = _create_datamodule(
        lm1b_dataset_path,
        additional_overrides=[f"batch_size={batch_size}"]
    )
    
    dm.prepare_data()

    # Before setup, data splits should not exist
    assert not dm.data_train and not dm.data_val and not dm.data_test

    # Setup splits data
    dm.setup()

    # Check datamodule structure
    _check_datamodule_structure(dm, expected_total_train_val=100, expected_test=10)

    # Check train batch correctness
    train_batch = next(iter(dm.train_dataloader()))
    _check_batch_correctness(train_batch, expected_batch_size=batch_size, expected_seq_length=512, check_tokenization=True)

    # Test validation dataloader (might have smaller batch)
    val_batch = next(iter(dm.val_dataloader()))
    assert len(val_batch["input_ids"]) <= batch_size, f"Val batch size should be <= {batch_size}"
    assert len(val_batch["input_ids"]) > 0, "Val batch should not be empty"

    # Test test dataloader
    test_batch = next(iter(dm.test_dataloader()))
    assert "input_ids" in test_batch, "Test batch should contain tokenized data"
    assert "text" in test_batch, "Test batch should preserve original text"
