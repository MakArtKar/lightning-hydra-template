from typing import Any, Callable, Literal, Mapping

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric, MinMetric, MetricCollection, Metric

from ml_core.models.utils import CriterionsComposition, MetricsComposition


class BaseLitModule(LightningModule):
    def __init__(
        self,
        forward_fn: Callable,
        criterions: CriterionsComposition,
        optimizer: Callable[[], torch.optim.Optimizer],
        scheduler: Callable[[torch.optim.Optimizer], torch.optim.lr_scheduler.LRScheduler] | None = None,
        metrics: MetricsComposition | None = None,
        tracked_metric_name: str | None = None,
        compile: bool = False,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.forward_fn = forward_fn
        self.criterions = criterions

        if metrics is not None:
            self.train_metrics = metrics.clone("train/")
            self.val_metrics = metrics.clone("val/")
            self.test_metrics = metrics.clone("test/")

        self.train_losses = MetricCollection(
            {criterion_name: MeanMetric() for criterion_name in self.criterions.keys()} | {"total": MeanMetric()}
        ).clone("train/")
        self.val_losses = self.train_losses.clone("val/")
        self.test_losses = self.train_losses.clone("test/")

        self.best_val_tracked_metric = MaxMetric() if metrics and tracked_metric_name else MinMetric()

        if tracked_metric_name is not None and metrics is None:
            raise ValueError("`tracked_metric_name` should be None in case null metrics")

    def forward(self, batch) -> Mapping[str, Any]:
        return self.forward_fn(batch)

    def on_train_start(self) -> None:
        for criterion_name in self.criterions.keys():
            self.val_losses[criterion_name].reset()
        self.val_losses["total"].reset()

        if hasattr(self, "val_metrics"):
            for metric_name in self.val_metrics:
                self.val_metrics[metric_name].reset()
        self.best_val_tracked_metric.reset()

    def model_step(self, batch: Mapping[str, Any], stage: Literal["train", "val", "test"]) -> Mapping[str, Any]:
        batch = self(batch)

        # Losses
        losses = self.criterions(batch)
        loss = getattr(self, f"{stage}_losses")
        for criterion_name in losses.keys():
            loss[criterion_name](losses[criterion_name])
            self.log(f"{stage}/{criterion_name}", loss[criterion_name], on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        # Metrics
        if hasattr(self, f"{stage}_metrics"):
            metrics = getattr(self, f"{stage}_metrics")
            metrics(batch)
            for metric_name in metrics.keys():
                self.log(f"{metric_name}", metrics[metric_name], on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        return {"loss": losses["total"], **batch}

    def training_step(self, batch: Mapping[str, Any], batch_idx: int) -> torch.Tensor:
        return self.model_step(batch, "train")

    def validation_step(self, batch: Mapping[str, Any], batch_idx: int) -> torch.Tensor:
        return self.model_step(batch, "val")

    def test_step(self, batch: Mapping[str, Any], batch_idx: int) -> torch.Tensor:
        return self.model_step(batch, "test")

    def on_validation_epoch_end(self) -> None:
        if self.hparams.tracked_metric_name is None:
            tracked_metric = self.val_losses["total"]
        else:
            tracked_metric = self.val_metrics[self.hparams.tracked_metric_name]
        
        self.best_val_tracked_metric(tracked_metric.compute())
        self.log(f"val/best", self.best_val_tracked_metric.compute(), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def setup(self, stage: str) -> None:
        if self.hparams.compile and stage == "fit":
            self.forward_fn = torch.compile(self.forward_fn)
    
    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": f"val/{self.hparams.tracked_metric_name or 'loss'}",
                    "interval": "epoch",
                    "frequency": 1,
                    "mode": "max" if self.hparams.tracked_metric_name else "min"
                },
            }
        return {"optimizer": optimizer}
