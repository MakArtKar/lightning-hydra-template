"""Unit tests for lm1b dataset."""

import os

import pytest
import rootutils
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

# Setup root to get PROJECT_ROOT
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
PROJECT_ROOT = os.environ["PROJECT_ROOT"]


@pytest.fixture(scope="module")
def lm1b_datamodule():
    """Fixture that loads config and returns a BaseDataModule object.

    :return: BaseDataModule instance configured with lm1b dataset.
    """
    GlobalHydra.instance().clear()

    config_path = os.path.join(PROJECT_ROOT, "configs")
    
    # Define test parameters
    max_length = 128
    tokenizer_model = "bert-base-uncased"
    
    with initialize_config_dir(version_base="1.3", config_dir=config_path):
        # Load the lm1b data config and provide params via overrides
        cfg = compose(
            config_name="data/lm1b",
            overrides=[
                f"++params.data.max_length={max_length}",
                f"++params.data.tokenizer._target_=transformers.AutoTokenizer.from_pretrained",
                f"++params.data.tokenizer.pretrained_model_name_or_path={tokenizer_model}",
                "data.batch_size=2",  # Small batch size for testing
                "data.num_workers=0",  # No multiprocessing for tests
                "data.persistent_workers=False",  # Disable persistent workers for testing
            ],
        )
        
        # Instantiate the datamodule using Hydra
        from hydra.utils import instantiate
        datamodule = instantiate(cfg.data)
        
        yield datamodule
    
    GlobalHydra.instance().clear()


def test_lm1b_datamodule_setup(lm1b_datamodule):
    """Test that datamodule can be setup correctly.

    :param lm1b_datamodule: The lm1b datamodule fixture.
    """
    # Setup the datamodule
    lm1b_datamodule.setup()
    
    # Check that splits are created
    assert lm1b_datamodule.data_train is not None
    assert lm1b_datamodule.data_val is not None
    assert lm1b_datamodule.data_test is not None


def test_lm1b_train_dataloader(lm1b_datamodule):
    """Test that train dataloader returns correct keys.

    :param lm1b_datamodule: The lm1b datamodule fixture.
    """
    # Setup the datamodule
    lm1b_datamodule.setup()
    
    # Get train dataloader
    train_loader = lm1b_datamodule.train_dataloader()
    
    # Get one batch
    batch = next(iter(train_loader))
    
    # Check required keys are present
    assert "input_ids" in batch
    assert "attention_mask" in batch
    assert "text" in batch


def test_lm1b_val_dataloader(lm1b_datamodule):
    """Test that validation dataloader returns correct keys.

    :param lm1b_datamodule: The lm1b datamodule fixture.
    """
    # Setup the datamodule
    lm1b_datamodule.setup()
    
    # Get val dataloader
    val_loader = lm1b_datamodule.val_dataloader()
    
    # Get one batch
    batch = next(iter(val_loader))
    
    # Check required keys are present
    assert "input_ids" in batch
    assert "attention_mask" in batch
    assert "text" in batch


def test_lm1b_test_dataloader(lm1b_datamodule):
    """Test that test dataloader returns correct keys.

    :param lm1b_datamodule: The lm1b datamodule fixture.
    """
    # Setup the datamodule
    lm1b_datamodule.setup()
    
    # Get test dataloader
    test_loader = lm1b_datamodule.test_dataloader()
    
    # Get one batch
    batch = next(iter(test_loader))
    
    # Check required keys are present
    assert "input_ids" in batch
    assert "attention_mask" in batch
    assert "text" in batch

