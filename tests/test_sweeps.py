from pathlib import Path

import pytest

from tests.helpers.run_if import RunIf
from tests.helpers.run_sh_command import run_sh_command

startfile = "ml_core/train.py"
overrides = ["logger=[]"]


@RunIf(sh=True)
@pytest.mark.slow
def test_experiments(tmp_path: Path) -> None:
    """Test running all available experiment configs with `fast_dev_run=True.`

    :param tmp_path: The temporary logging path.
    """
    command = [
        startfile,
        "-m",
        "--config-dir=configs/",
        "experiment=glob(*)",
        "hydra.sweep.dir=" + str(tmp_path),
        "++data.num_workers=0",
        "++data.pin_memory=False",
        "++data.persistent_workers=False",
        "++trainer.fast_dev_run=true",
    ] + overrides
    run_sh_command(command)


@RunIf(sh=True)
@pytest.mark.slow
def test_hydra_sweep(tmp_path: Path) -> None:
    """Test default hydra sweep.

    :param tmp_path: The temporary logging path.
    """
    command = [
        startfile,
        "-m",
        "--config-dir=configs/",
        "hydra.sweep.dir=" + str(tmp_path),
        "model.optimizer.lr=0.005,0.01",
        "++data.num_workers=0",
        "++data.pin_memory=False",
        "++data.persistent_workers=False",
        "++trainer.fast_dev_run=true",
    ] + overrides

    run_sh_command(command)


@RunIf(sh=True)
@pytest.mark.slow
def test_hydra_sweep_ddp_sim(tmp_path: Path) -> None:
    """Test default hydra sweep with ddp sim.

    :param tmp_path: The temporary logging path.
    """
    command = [
        startfile,
        "-m",
        "--config-dir=configs/",
        "hydra.sweep.dir=" + str(tmp_path),
        "trainer=ddp_sim",
        "trainer.max_epochs=3",
        "++data.num_workers=0",
        "++data.pin_memory=False",
        "++data.persistent_workers=False",
        "+trainer.limit_train_batches=0.01",
        "+trainer.limit_val_batches=0.1",
        "+trainer.limit_test_batches=0.1",
        "model.optimizer.lr=0.005,0.01,0.02",
    ] + overrides
    run_sh_command(command)


@RunIf(sh=True)
@pytest.mark.slow
def test_optuna_sweep(tmp_path: Path) -> None:
    """Test Optuna hyperparam sweeping.

    :param tmp_path: The temporary logging path.
    """
    command = [
        startfile,
        "-m",
        "--config-dir=configs/",
        "hparams_search=optuna",
        "hydra.sweep.dir=" + str(tmp_path),
        "hydra.sweeper.n_trials=10",
        "hydra.sweeper.sampler.n_startup_trials=5",
        "++data.num_workers=0",
        "++data.pin_memory=False",
        "++data.persistent_workers=False",
        "++trainer.fast_dev_run=true",
    ] + overrides
    run_sh_command(command)


@RunIf(wandb=True, sh=True)
@pytest.mark.slow
def test_optuna_sweep_ddp_sim_wandb(tmp_path: Path) -> None:
    """Test Optuna sweep with wandb logging and ddp sim.

    :param tmp_path: The temporary logging path.
    """
    command = [
        startfile,
        "-m",
        "--config-dir=configs/",
        "hparams_search=optuna",
        "hydra.sweep.dir=" + str(tmp_path),
        "hydra.sweeper.n_trials=5",
        "trainer=ddp_sim",
        "trainer.max_epochs=3",
        "++data.num_workers=0",
        "++data.pin_memory=False",
        "++data.persistent_workers=False",
        "+trainer.limit_train_batches=0.01",
        "+trainer.limit_val_batches=0.1",
        "+trainer.limit_test_batches=0.1",
        "logger=wandb",
    ]
    run_sh_command(command)
