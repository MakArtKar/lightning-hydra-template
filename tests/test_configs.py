from typing import Dict

import hydra
import pytest
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from tests.conftest import get_all_experiments


@pytest.mark.parametrize("experiment_path", get_all_experiments())
def test_train_config(cfg_train: Dict[str, DictConfig], experiment_path: str) -> None:
    """Tests the training configuration provided by the `cfg_train` pytest fixture.

    :param cfg_train: A dict mapping experiment paths to training configurations.
    :param experiment_path: The experiment path to test.
    """
    cfg_train = cfg_train[experiment_path]

    assert cfg_train
    assert cfg_train.data
    assert cfg_train.model
    assert cfg_train.trainer

    HydraConfig().set_config(cfg_train)

    hydra.utils.instantiate(cfg_train.data)
    hydra.utils.instantiate(cfg_train.model)
    hydra.utils.instantiate(cfg_train.trainer)


@pytest.mark.parametrize("experiment_path", get_all_experiments())
def test_eval_config(cfg_eval: Dict[str, DictConfig], experiment_path: str) -> None:
    """Tests the evaluation configuration provided by the `cfg_eval` pytest fixture.

    :param cfg_eval: A dict mapping experiment paths to evaluation configurations.
    :param experiment_path: The experiment path to test.
    """
    cfg_eval = cfg_eval[experiment_path]

    assert cfg_eval
    assert cfg_eval.data
    assert cfg_eval.model
    assert cfg_eval.trainer

    HydraConfig().set_config(cfg_eval)

    hydra.utils.instantiate(cfg_eval.data)
    hydra.utils.instantiate(cfg_eval.model)
    hydra.utils.instantiate(cfg_eval.trainer)
