"""This file prepares config fixtures for other tests."""

from pathlib import Path
from typing import Dict, List

import pytest
import rootutils
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, open_dict


def get_all_experiments() -> List[str]:
    """Get all experiment config paths from configs/experiment/ directory.

    :return: A list of experiment config names (without .yaml extension).
    """
    root = rootutils.find_root(indicator=".project-root")
    experiment_dir = Path(root) / "configs" / "experiment"

    if not experiment_dir.exists():
        return []

    experiment_paths = []
    # Recursively find all .yaml files in experiment directory
    for yaml_file in experiment_dir.rglob("*.yaml"):
        # Get relative path from experiment directory and remove .yaml extension
        relative_path = yaml_file.relative_to(experiment_dir)
        experiment_name = str(relative_path.with_suffix("")).replace("\\", "/")
        experiment_paths.append(experiment_name)

    return sorted(experiment_paths)


@pytest.fixture(scope="package")
def cfg_train_global() -> Dict[str, DictConfig]:
    """A pytest fixture for setting up a default Hydra DictConfig for training.

    :return: A dict mapping experiment paths to their corresponding DictConfig objects.
    """
    experiment_paths = get_all_experiments()
    configs = {}

    root_dir = str(rootutils.find_root(indicator=".project-root"))

    with initialize(version_base="1.3", config_path="../configs"):
        for experiment_path in experiment_paths:
            overrides = [
                f"experiment={experiment_path}",
                f"paths.root_dir={root_dir}",
                "++trainer.max_epochs=1",
                "++trainer.limit_train_batches=10",
                "++trainer.limit_val_batches=10",
                "++trainer.limit_test_batches=10",
                "++trainer.accelerator=cpu",
                "++trainer.devices=1",
                "++data.num_workers=0",
                "++data.pin_memory=False",
                "++data.persistent_workers=False",
                "++extras.print_config=False",
                "++extras.enforce_tags=False",
                "logger=[]",
            ]

            cfg = compose(config_name="train.yaml", return_hydra_config=True, overrides=overrides)
            configs[experiment_path] = cfg

    return configs


@pytest.fixture(scope="package")
def cfg_eval_global() -> Dict[str, DictConfig]:
    """A pytest fixture for setting up a default Hydra DictConfig for evaluation.

    :return: A dict mapping experiment paths to their corresponding DictConfig objects.
    """
    experiment_paths = get_all_experiments()
    configs = {}

    root_dir = str(rootutils.find_root(indicator=".project-root"))

    with initialize(version_base="1.3", config_path="../configs"):
        for experiment_path in experiment_paths:
            overrides = [
                f"experiment={experiment_path}",
                "ckpt_path=.",
                f"paths.root_dir={root_dir}",
                "++trainer.max_epochs=1",
                "++trainer.limit_test_batches=10",
                "++trainer.accelerator=cpu",
                "++trainer.devices=1",
                "++data.num_workers=0",
                "++data.pin_memory=False",
                "++data.persistent_workers=False",
                "++extras.print_config=False",
                "++extras.enforce_tags=False",
                "logger=[]",
            ]

            cfg = compose(config_name="eval.yaml", return_hydra_config=True, overrides=overrides)
            configs[experiment_path] = cfg

    return configs


@pytest.fixture(scope="function")
def cfg_train(cfg_train_global: Dict[str, DictConfig], tmp_path: Path) -> Dict[str, DictConfig]:
    """A pytest fixture built on top of the `cfg_train_global()` fixture, which accepts a temporary
    logging path `tmp_path` for generating a temporary logging path.

    This is called by each test which uses the `cfg_train` arg. Each test generates its own temporary logging path.

    :param cfg_train_global: The input dict of DictConfig objects to be modified.
    :param tmp_path: The temporary logging path.

    :return: A dict of DictConfig objects with updated output and log directories corresponding to `tmp_path`.
    """
    configs = {}

    for experiment_path, cfg_global in cfg_train_global.items():
        cfg = cfg_global.copy()

        with open_dict(cfg):
            cfg.paths.output_dir = str(tmp_path)
            cfg.paths.log_dir = str(tmp_path)

        configs[experiment_path] = cfg

    yield configs

    GlobalHydra.instance().clear()


@pytest.fixture(scope="function")
def cfg_eval(cfg_eval_global: Dict[str, DictConfig], tmp_path: Path) -> Dict[str, DictConfig]:
    """A pytest fixture built on top of the `cfg_eval_global()` fixture, which accepts a temporary
    logging path `tmp_path` for generating a temporary logging path.

    This is called by each test which uses the `cfg_eval` arg. Each test generates its own temporary logging path.

    :param cfg_eval_global: The input dict of DictConfig objects to be modified.
    :param tmp_path: The temporary logging path.

    :return: A dict of DictConfig objects with updated output and log directories corresponding to `tmp_path`.
    """
    configs = {}

    for experiment_path, cfg_global in cfg_eval_global.items():
        cfg = cfg_global.copy()

        with open_dict(cfg):
            cfg.paths.output_dir = str(tmp_path)
            cfg.paths.log_dir = str(tmp_path)

        configs[experiment_path] = cfg

    yield configs

    GlobalHydra.instance().clear()
