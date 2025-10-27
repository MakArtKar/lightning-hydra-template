import os
from pathlib import Path
from typing import Dict

import pytest
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict

from ml_core.eval import evaluate
from ml_core.train import train
from tests.conftest import get_all_experiments


@pytest.mark.parametrize("experiment_path", get_all_experiments())
@pytest.mark.slow
def test_train_eval(
    tmp_path: Path,
    cfg_train: Dict[str, DictConfig],
    cfg_eval: Dict[str, DictConfig],
    experiment_path: str,
) -> None:
    """Tests training and evaluation by training for 1 epoch with `train.py` then evaluating with
    `eval.py`.

    :param tmp_path: The temporary logging path.
    :param cfg_train: A dict mapping experiment paths to training configurations.
    :param cfg_eval: A dict mapping experiment paths to evaluation configurations.
    :param experiment_path: The experiment path to test.
    """
    cfg_train = cfg_train[experiment_path]
    cfg_eval = cfg_eval[experiment_path]

    assert str(tmp_path) == cfg_train.paths.output_dir == cfg_eval.paths.output_dir

    with open_dict(cfg_train):
        cfg_train.trainer.max_epochs = 1
        cfg_train.test = True

    HydraConfig().set_config(cfg_train)
    train_metric_dict, _ = train(cfg_train)

    assert "last.ckpt" in os.listdir(tmp_path / "checkpoints")

    with open_dict(cfg_eval):
        cfg_eval.ckpt_path = str(tmp_path / "checkpoints" / "last.ckpt")

    HydraConfig().set_config(cfg_eval)
    test_metric_dict, _ = evaluate(cfg_eval)

    for key in test_metric_dict:
        assert test_metric_dict[key] > 0.0
        assert abs(train_metric_dict[key].item() - test_metric_dict[key].item()) < 0.001
