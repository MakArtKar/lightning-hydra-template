from typing import Any, Literal, Mapping

from lightning import Callback, LightningModule, Trainer
from torchmetrics import Metric, MetricCollection


class MetricsCallback(Callback):
    """Callback that manages metrics computation and logging for train/val/test stages.

    This callback creates metric collections for each stage and attaches them to the
    LightningModule, then updates and logs them during training. Metrics are kept as module
    attributes to follow Lightning's expected pattern.

    Supports two computation modes:
    - Batch-end: Metrics computed after each batch (default)
    - Stage-end: Metrics computed at the end of validation/test using generated samples
    """

    def __init__(
        self,
        metrics: Mapping[str, Metric] | None = None,
        mapping: Mapping[str, Mapping[str, str]] | None = None,
        compute_on_batch_end: Mapping[str, bool] | None = None,
    ) -> None:
        """Initialize the metrics callback.

        :param metrics: Mapping from metric name to Metric instances to track.
        :param mapping: Optional mapping defining how to pull inputs from batch for each metric.
            For each metric name, maps metric argument names to batch keys.
        :param compute_on_batch_end: Optional mapping from metric name to bool indicating whether
            to compute the metric after each batch (True, default) or at stage end (False). Stage-
            end metrics use generated samples from trainer._generations.
        """
        super().__init__()
        self.metrics = metrics
        self.mapping = mapping or {}
        self.compute_on_batch_end = compute_on_batch_end or {}

        self._has_on_batch_end_false = any(
            not value for value in self.compute_on_batch_end.values()
        )

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        """Create and attach metric collections to the module before training starts.

        :param trainer: The Lightning trainer.
        :param pl_module: The Lightning module to attach metrics to.
        :param stage: Current stage (fit, test, validate, predict).
        """
        # Only create metrics once, and only if they don't already exist
        if self.metrics is not None and not hasattr(pl_module, "train_metrics"):
            # Convert to plain dict in case it's a DictConfig
            metrics_dict = dict(self.metrics) if self.metrics else {}
            pl_module.train_metrics = MetricCollection(metrics_dict, prefix="train/")
            pl_module.val_metrics = MetricCollection(metrics_dict, prefix="val/")
            pl_module.test_metrics = MetricCollection(metrics_dict, prefix="test/")

    def _calculate_metrics(
        self,
        trainer: Trainer,
        stage: Literal["train", "val", "test"],
        pl_module: LightningModule,
        outputs: dict[str, Any],
        computing_on_batch_end: bool,
    ) -> None:
        """Calculate and log metrics for a given stage.

        :param trainer: The Lightning trainer.
        :param stage: Current stage (train, val, or test).
        :param pl_module: The Lightning module.
        :param outputs: Outputs from the step or generated samples from trainer._generations.
        :param computing_on_batch_end: If True, only compute metrics with
            compute_on_batch_end=True. If False, only compute metrics with
            compute_on_batch_end=False.
        """
        metrics_attr = f"{stage}_metrics"
        if hasattr(pl_module, metrics_attr):
            metrics = getattr(pl_module, metrics_attr)
            # Update each metric with remapped inputs from the batch
            for metric_name in metrics.keys():
                # Remove the stage prefix to get the original metric name for mapping lookup
                original_metric_name = metric_name.replace(f"{stage}/", "")

                # Check if this metric should be computed at this time
                should_compute_on_batch_end = self.compute_on_batch_end.get(
                    original_metric_name, True
                )
                if should_compute_on_batch_end != computing_on_batch_end:
                    continue

                if original_metric_name in self.mapping:
                    # Remap batch keys to metric arguments
                    input_batch = {
                        kwarg_name: outputs[batch_key]
                        for kwarg_name, batch_key in self.mapping[original_metric_name].items()
                    }
                    metrics[metric_name](**input_batch)
                else:
                    # No mapping, pass outputs directly (backward compatibility)
                    metrics[metric_name](outputs)

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
        self._calculate_metrics(trainer, "train", pl_module, outputs, computing_on_batch_end=True)

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
        self._calculate_metrics(trainer, "val", pl_module, outputs, computing_on_batch_end=True)

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
        self._calculate_metrics(trainer, "test", pl_module, outputs, computing_on_batch_end=True)

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Compute stage-end metrics at the end of validation using generated samples.

        :param trainer: The Lightning trainer.
        :param pl_module: The Lightning module.
        """
        # Get generated samples from trainer if available
        if self._has_on_batch_end_false and not hasattr(trainer, "_generations"):
            raise ValueError("Generated samples are not available for stage-end metrics")
        self._calculate_metrics(
            trainer, "val", pl_module, trainer._generations, computing_on_batch_end=False
        )

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Compute stage-end metrics at the end of test using generated samples.

        :param trainer: The Lightning trainer.
        :param pl_module: The Lightning module.
        """
        # Get generated samples from trainer if available
        if self._has_on_batch_end_false and not hasattr(trainer, "_generations"):
            raise ValueError("Generated samples are not available for stage-end metrics")
        self._calculate_metrics(
            trainer, "test", pl_module, trainer._generations, computing_on_batch_end=False
        )
