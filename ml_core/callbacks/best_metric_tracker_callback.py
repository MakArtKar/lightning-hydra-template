from lightning import Callback, LightningModule, Trainer
from torchmetrics import MaxMetric, MinMetric


class BestMetricTrackerCallback(Callback):
    """Callback that tracks the best validation metric for model selection.

    This callback monitors a specific validation metric (or validation loss if none specified) and
    tracks the best value seen during training. This is used by ModelCheckpoint and EarlyStopping
    callbacks to save/select the best model.

    The best metric is logged as "val/best" and can be monitored by other components.
    """

    def __init__(
        self,
        tracked_metric_name: str | None = None,
    ) -> None:
        """Initialize the best metric tracker callback.

        :param tracked_metric_name: Name of the metric to track (without "val/" prefix). If None,
            tracks validation loss instead. For custom metrics, higher is better; for loss, lower
            is better.
        """
        super().__init__()
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

        # Create best validation metric tracker
        # Use MaxMetric for custom metrics (higher is better), MinMetric for loss (lower is better)
        if not hasattr(pl_module, "best_val_tracked_metric"):
            pl_module.best_val_tracked_metric = (
                MaxMetric() if self.tracked_metric_name else MinMetric()
            )

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        # Reset best metric tracker only if NOT resuming from checkpoint
        # (trainer.ckpt_path is set when resuming)
        if hasattr(pl_module, "best_val_tracked_metric"):
            if trainer.ckpt_path is None:
                # Fresh training - reset the tracker for validation sanity checks
                pl_module.best_val_tracked_metric.reset()
            # else: resuming from checkpoint - don't reset, it was loaded from checkpoint

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
