from typing import Any, Literal

from lightning import Callback, LightningModule, Trainer
from torchmetrics import MaxMetric, MinMetric

from ml_core.models.utils import MetricsComposition


class MetricsCallback(Callback):
    """Callback that manages metrics computation and logging for train/val/test stages.

    This callback creates metric collections for each stage and attaches them to the
    LightningModule, then updates and logs them during training. Metrics are kept as
    module attributes to follow Lightning's expected pattern. Also tracks the best
    validation metric for model selection.
    """

    def __init__(
        self,
        metrics: MetricsComposition | None = None,
        tracked_metric_name: str | None = None,
    ) -> None:
        """Initialize the metrics callback.

        :param metrics: Optional composition of metrics to track across stages.
        :param tracked_metric_name: Metric key (without stage prefix) used to track the best
            value on validation; if None, the total validation loss is tracked.
        """
        super().__init__()
        self.metrics = metrics
        self.tracked_metric_name = tracked_metric_name

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        """Create and attach metric collections to the module before training starts.

        :param trainer: The Lightning trainer.
        :param pl_module: The Lightning module to attach metrics to.
        :param stage: Current stage (fit, test, validate, predict).
        """
        # Set tracked_metric_name on the module so it can be used for scheduler config
        if not hasattr(pl_module, "tracked_metric_name"):
            pl_module.tracked_metric_name = self.tracked_metric_name

        # Only create metrics once, and only if they don't already exist
        if self.metrics is not None and not hasattr(pl_module, "train_metrics"):
            pl_module.train_metrics = self.metrics.clone("train/")
            pl_module.val_metrics = self.metrics.clone("val/")
            pl_module.test_metrics = self.metrics.clone("test/")

        # Create best validation metric tracker
        if not hasattr(pl_module, "best_val_tracked_metric"):
            pl_module.best_val_tracked_metric = (
                MaxMetric() if self.tracked_metric_name else MinMetric()
            )

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Reset validation metrics and best metric tracker at the start of training.

        :param trainer: The Lightning trainer.
        :param pl_module: The Lightning module.
        """
        if hasattr(pl_module, "val_metrics"):
            for metric_name in pl_module.val_metrics:
                pl_module.val_metrics[metric_name].reset()

        if hasattr(pl_module, "best_val_tracked_metric"):
            pl_module.best_val_tracked_metric.reset()

    def _on_stage_batch_end(
        self,
        stage: Literal["train", "val", "test"],
        pl_module: LightningModule,
        outputs: dict[str, Any],
    ) -> None:
        """Update and log metrics for a given stage after each batch.

        :param stage: Current stage (train, val, or test).
        :param pl_module: The Lightning module.
        :param outputs: Outputs from the step (includes the augmented batch).
        """
        metrics_attr = f"{stage}_metrics"
        if hasattr(pl_module, metrics_attr):
            metrics = getattr(pl_module, metrics_attr)
            metrics(outputs)
            for metric_name in metrics.keys():
                pl_module.log(
                    f"{metric_name}",
                    metrics[metric_name],
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    sync_dist=True,
                )

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: dict[str, Any],
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Update and log training metrics after each batch.

        :param trainer: The Lightning trainer.
        :param pl_module: The Lightning module.
        :param outputs: Outputs from the training step (includes the augmented batch).
        :param batch: The current batch.
        :param batch_idx: Index of the current batch.
        """
        self._on_stage_batch_end("train", pl_module, outputs)

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: dict[str, Any],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Update and log validation metrics after each batch.

        :param trainer: The Lightning trainer.
        :param pl_module: The Lightning module.
        :param outputs: Outputs from the validation step (includes the augmented batch).
        :param batch: The current batch.
        :param batch_idx: Index of the current batch.
        :param dataloader_idx: Index of the current dataloader.
        """
        self._on_stage_batch_end("val", pl_module, outputs)

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: dict[str, Any],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Update and log test metrics after each batch.

        :param trainer: The Lightning trainer.
        :param pl_module: The Lightning module.
        :param outputs: Outputs from the test step (includes the augmented batch).
        :param batch: The current batch.
        :param batch_idx: Index of the current batch.
        :param dataloader_idx: Index of the current dataloader.
        """
        self._on_stage_batch_end("test", pl_module, outputs)

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Compute and log best validation metric at epoch end.

        :param trainer: The Lightning trainer.
        :param pl_module: The Lightning module.
        """
        if self.tracked_metric_name is None:
            tracked_metric = pl_module.val_losses["total"]
        else:
            if not hasattr(pl_module, "val_metrics"):
                raise ValueError(
                    f"tracked_metric_name '{self.tracked_metric_name}' is set but "
                    "val_metrics not found. Did you set metrics in MetricsCallback?"
                )
            tracked_metric = pl_module.val_metrics[self.tracked_metric_name]

        pl_module.best_val_tracked_metric(tracked_metric.compute())
        pl_module.log(
            "val/best",
            pl_module.best_val_tracked_metric.compute(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

