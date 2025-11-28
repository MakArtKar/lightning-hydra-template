from typing import Any, Callable, Literal, Mapping

import torch
from lightning import LightningModule
from torchmetrics import MeanMetric, MetricCollection

from ml_core.models.utils import CriterionsComposition


class BaseLitModule(LightningModule):
    """Generic LightningModule wiring forward, losses, and optimizers.

    Expects a callable `forward_fn`, a `CriterionsComposition` for computing losses,
    and optimizer/scheduler factories. Provides common training/validation/test steps
    and logs with `sync_dist` enabled for DDP. Metrics are handled via MetricsCallback.
    """

    def __init__(
        self,
        forward_fn: Callable,
        criterions: CriterionsComposition,
        optimizer: Callable[[], torch.optim.Optimizer],
        scheduler: (
            Callable[[torch.optim.Optimizer], torch.optim.lr_scheduler.LRScheduler] | None
        ) = None,
        compile: bool = False,
    ) -> None:
        """Initialize the Lightning module wiring.

        :param forward_fn: Callable mapping a batch to outputs (e.g., network forward).
        :param criterions: Composition of losses operating on the output batch mapping.
        :param optimizer: Factory returning a configured optimizer for the model parameters.
        :param scheduler: Optional factory returning an LR scheduler for the optimizer.
        :param compile: Whether to compile ``forward_fn`` with ``torch.compile`` during fit.
        """
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.forward_fn = forward_fn
        self.criterions = criterions

        self.train_losses = MetricCollection(
            {criterion_name: MeanMetric() for criterion_name in self.criterions.keys()}
            | {"total": MeanMetric()}
        ).clone("train/")
        self.val_losses = self.train_losses.clone("val/")
        self.test_losses = self.train_losses.clone("test/")

    def forward(self, batch) -> Mapping[str, Any]:
        """Forward pass producing a dict-like output expected by losses/metrics.

        :param batch: Mapping with input tensors.
        :return: Mapping augmented with model outputs.
        """
        return self.forward_fn(batch)

    def on_train_start(self) -> None:
        """Reset validation loss aggregators at train start."""
        for criterion_name in self.criterions.keys():
            self.val_losses[criterion_name].reset()
        self.val_losses["total"].reset()

    def model_step(
        self, batch: Mapping[str, Any], stage: Literal["train", "val", "test"]
    ) -> Mapping[str, Any]:
        """Shared step computing losses and logging for a stage."""
        batch = self(batch)

        # Losses
        losses = self.criterions(batch)
        loss = getattr(self, f"{stage}_losses")
        for criterion_name in losses.keys():
            loss[criterion_name](losses[criterion_name])
            self.log(
                f"{stage}/{criterion_name}",
                loss[criterion_name],
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )

        return {"loss": losses["total"], **batch}

    def training_step(self, batch: Mapping[str, Any], batch_idx: int) -> torch.Tensor:
        """Lightning training step."""
        return self.model_step(batch, "train")

    def validation_step(self, batch: Mapping[str, Any], batch_idx: int) -> torch.Tensor:
        """Lightning validation step."""
        return self.model_step(batch, "val")

    def test_step(self, batch: Mapping[str, Any], batch_idx: int) -> torch.Tensor:
        """Lightning test step."""
        return self.model_step(batch, "test")

    def setup(self, stage: str) -> None:
        """Optionally compile the wrapped forward callable in fit stage."""
        if self.hparams.compile and stage == "fit":
            self.forward_fn = torch.compile(self.forward_fn)

    def configure_optimizers(self) -> dict[str, Any]:
        """Create optimizer and (optionally) LR scheduler config for Trainer."""
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            # tracked_metric_name is set by BestMetricTrackerCallback in setup
            tracked_metric_name = getattr(self, "tracked_metric_name", None)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": f"val/{tracked_metric_name or 'total'}",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

    def on_validation_epoch_start(self) -> None:
        """Initialize generations dict at start of validation epoch."""
        self.trainer._generations = {}

    def on_test_epoch_start(self) -> None:
        """Initialize generations dict at start of test epoch."""
        self.trainer._generations = {}
