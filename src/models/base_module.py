from typing import Any, Literal, Mapping

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric, Metric, MinMetric
from torchmetrics.classification.accuracy import Accuracy


class BaseLitModule(LightningModule):
    """Generic LightningModule wrapping a forward module, losses, and metrics.

    This module orchestrates a provided forward function, a configurable set of
    loss functions (with input-to-argument mappings and coefficients), and
    metrics (with stage selection and input mappings). It also tracks a
    validation metric according to a specified mode (min/max).
    """
    def __init__(
        self,
        forward_fn: torch.nn.Module,
        criterion: Mapping[str, tuple[Mapping[str, str], torch.nn.Module]],
        criterion_coefs: Mapping[str, float],
        metrics: Mapping[str, tuple[Mapping[str, str], list[str], Metric]] | None,
        tracked_metric_mode: Literal["max", "min"],
        tracked_metric_name: str | None,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
    ) -> None:
        """Initialize a BaseLitModule.

        :param forward_fn: The callable used as a forward method.
        :param criterion: Mapping from loss name to a tuple of (input mapping,
            loss module). 
            * The input mapping maps batch keys to loss argument names.
            * The loss module - torch.nn.Module loss class
        :param criterion_coefs: Coefficients per loss to combine into total loss.
        :param metrics: Mapping from metric name to a tuple of (input mapping, stages, metric instance).
            * The input mapping maps batch keys to metric argument names.
            * Stages is a list among ["train","val","test"] specifying stages on
                which metric should be calculated.
            * The metric instance - torch.nn.Module metric class
        :param tracked_metric_mode: How to track the best value of the chosen validation metric, either
            "max" or "min".
        :param tracked_metric_name: The metric name (without stage prefix) to track on validation.
        :param optimizer: Optimizer factory (callable) configured via Hydra.
        :param scheduler: LR scheduler factory (callable) configured via Hydra.
        :param compile: If True, compile the forward function during fit.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["forward_fn"])

        self.forward_fn = forward_fn

        # loss function
        self.criterion_names = list(criterion.keys())
        for criterion_name, (criterion_mapping, criterion) in criterion.items():
            setattr(self, f"{criterion_name}_mapping", criterion_mapping)
            setattr(self, criterion_name, criterion)
        self.criterion_coefs = criterion_coefs

        # for averaging loss across batches
        for stage in ["train", "val", "test"]:
            for criterion_name in self.criterion_names:
                setattr(self, f"{stage}_{criterion_name}_loss", MeanMetric())
            setattr(self, f"{stage}_loss", MeanMetric())

        # metric objects for calculating and averaging accuracy across batches
        if metrics is not None:
            for stage in ["train", "val", "test"]:
                setattr(self, f"{stage}_metric_names", [])
                for metric_name, (metric_mapping, metric_stages, metric) in metrics.items():
                    if stage not in metric_stages:
                        continue
                    getattr(self, f"{stage}_metric_names").append(metric_name)
                    setattr(self, f"{stage}_{metric_name}_mapping", metric_mapping)
                    setattr(self, f"{stage}_{metric_name}", metric.clone())

        # for tracking best so far validation accuracy
        if metrics is None and tracked_metric_mode == "max":
            raise ValueError("Tracked metric mode is 'max' but metrics are not provided => trackng loss instead")
        if metrics is None and tracked_metric_name is not None:
            raise ValueError("Tracked metric name is provided but metrics are not provided")
        
        self.val_best = MaxMetric() if tracked_metric_mode == "max" else MinMetric()
        self.tracked_metric_name = tracked_metric_name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        result = self.forward_fn(x)
        if not isinstance(result, Mapping):
            result = {"output": result}
        return result

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        for criterion_name in self.criterion_names:
            getattr(self, f"val_{criterion_name}_loss").reset()
        getattr(self, f"val_loss").reset()

        if hasattr(self, "val_metric_names"):
            for metric_name in self.val_metric_names:
                getattr(self, f"val_{metric_name}").reset()

        self.val_best.reset()

    def _calculate_loss(self, batch: Mapping[str, Any]) -> Mapping[str, torch.Tensor]:
        """Calculate individual losses and aggregate into a total loss.

        :param batch: Model batch dictionary containing all required inputs.
        :return: Mapping with individual losses by name and a "total" key.
        """
        total_loss = 0.
        losses = {}
        for criterion_name in self.criterion_names:
            filtered_batch = {new_key: batch[old_key] for old_key, new_key in getattr(self, f"{criterion_name}_mapping").items()}
            losses[criterion_name] = getattr(self, criterion_name)(**filtered_batch)
            total_loss = total_loss + self.criterion_coefs[criterion_name] * losses[criterion_name]
        losses["total"] = total_loss
        return losses

    def _log_losses(self, losses: Mapping[str, torch.Tensor], stage: Literal["train", "val", "test"]) -> None:
        """Log per-criterion and total losses for a given stage.

        :param losses: Mapping returned by ``_calculate_loss``.
        :param stage: One of "train", "val", or "test".
        """
        for criterion_name in self.criterion_names:
            self.log(f"{stage}/{criterion_name}_loss", losses[criterion_name], on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f"{stage}/loss", losses["total"], on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

    def _update_metrics(self, batch: Mapping[str, Any], stage: Literal["train", "val", "test"]) -> None:
        """Update configured metrics for the specified stage.

        :param batch: Model batch dictionary containing all required inputs.
        :param stage: Stage to update metrics for: "train", "val", or "test".
        """
        if hasattr(self, f"{stage}_metric_names"):
            for metric_name in getattr(self, f"{stage}_metric_names"):
                filtered_batch = {new_key: batch[old_key] for old_key, new_key in getattr(self, f"{stage}_{metric_name}_mapping").items()}
                getattr(self, f"{stage}_{metric_name}")(**filtered_batch)
        
    def _log_metrics(self, stage: Literal["train", "val", "test"]) -> None:
        """Log all configured metrics for the specified stage.

        :param stage: Stage to log metrics for: "train", "val", or "test".
        """
        if hasattr(self, f"{stage}_metric_names"):
            for metric_name in getattr(self, f"{stage}_metric_names"):
                self.log(f"{stage}/{metric_name}", getattr(self, f"{stage}_{metric_name}"), on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

    def model_step(
        self, batch: Mapping[str, Any], stage: Literal["train", "val", "test"]
    ) -> Mapping[str, Any]:
        """Run forward pass, compute losses, update and log metrics for a stage.

        :param batch: Batch dictionary before forward; will be augmented by forward.
        :param stage: "train", "val", or "test".
        :return: Mapping with keys: "loss" (tensor), "batch" (augmented batch),
            and "losses" (mapping of individual losses including "total").
        """
        batch = self.forward_fn(batch)

        losses = self._calculate_loss(batch)
        self._update_metrics(batch, stage)

        self._log_losses(losses, stage)
        self._log_metrics(stage)
        return {"loss": losses["total"], "batch": batch, "losses": losses}

    def training_step(
        self, batch: Mapping[str, Any], batch_idx: int
    ) -> Mapping[str, Any]:
        """Lightning training step.

        :param batch: Input batch.
        :param batch_idx: Batch index (unused).
        :return: Total loss tensor used for optimization.
        """
        return self.model_step(batch, "train")

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(self, batch: Mapping[str, Any], batch_idx: int) -> Mapping[str, Any]:
        """Lightning validation step.

        :param batch: Input batch.
        :param batch_idx: Batch index (unused).
        :return: Step output mapping from ``model_step``.
        """
        return self.model_step(batch, "val")

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        if hasattr(self, f"val_{self.tracked_metric_name}"):
            tracked_metric = getattr(self, f"val_{self.tracked_metric_name}")
        else:
            tracked_metric = getattr(self, f"val_loss")            
        tracked_metric_value = tracked_metric.compute()  # get current val tracked
        self.val_best(tracked_metric_value)  # update best so far val tracked metric
        # log `self.val_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/best", self.val_best.compute(), on_epoch=True, prog_bar=True, sync_dist=True)

    def test_step(self, batch: Mapping[str, Any], batch_idx: int) -> Mapping[str, Any]:
        """Lightning test step.

        :param batch: Input batch.
        :param batch_idx: Batch index (unused).
        :return: Step output mapping from ``model_step``.
        """
        return self.model_step(batch, "test")

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.forward_fn = torch.compile(self.forward_fn)

    def configure_optimizers(self) -> dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
