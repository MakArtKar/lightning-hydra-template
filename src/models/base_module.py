from typing import Any, Literal, Mapping

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric, Metric, MinMetric
from torchmetrics.classification.accuracy import Accuracy


class BaseLitModule(LightningModule):
    def __init__(
        self,
        forward_fn: torch.nn.Module,
        criterion: Mapping[str, tuple[Mapping[str, str], torch.nn.Module]],
        criterion_coefs: Mapping[str, float],
        metrics: Mapping[str, tuple[Mapping[str, str], list[str], Metric]],
        best_metric_mode: Literal["max", "min"],
        best_metric_name: str,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
    ) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

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
        for stage in ["train", "val", "test"]:
            setattr(self, f"{stage}_metric_names", [])
            for metric_name, (metric_mapping, metric_stages, metric) in metrics.items():
                if stage not in metric_stages:
                    continue
                getattr(self, f"{stage}_metric_names").append(metric_name)
                setattr(self, f"{stage}_{metric_name}_mapping", metric_mapping)
                setattr(self, f"{stage}_{metric_name}", metric.clone())

        # for tracking best so far validation accuracy
        self.val_best = MaxMetric() if best_metric_mode == "max" else MinMetric()
        self.best_metric_name = best_metric_name

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

        for metric_name in self.metric_names:
            getattr(self, f"val_{metric_name}").reset()

        self.val_best.reset()

    def _calculate_loss(self, batch: Mapping[str, Any]) -> Mapping[str, torch.Tensor]:
        total_loss = 0.
        losses = {}
        for criterion_name in self.criterion_names:
            filtered_batch = {new_key: batch[old_key] for old_key, new_key in self.criterion_mapping[criterion_name]}
            losses[criterion_name] = getattr(self, criterion_name)(**filtered_batch)
            total_loss = total_loss + self.criterion_coefs[criterion_name] * losses[criterion_name]
        losses["total"] = total_loss
        return losses

    def _log_losses(self, losses: Mapping[str, torch.Tensor], stage: Literal["train", "val", "test"]) -> None:
        for criterion_name in self.criterion_names:
            self.log(f"{stage}/{criterion_name}_loss", losses[criterion_name], on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f"{stage}/loss", losses["total"], on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def _update_metrics(self, batch: Mapping[str, Any], stage: Literal["train", "val", "test"]) -> None:
        for metric_name in getattr(self, f"{stage}_metric_names"):
            filtered_batch = {new_key: batch[old_key] for old_key, new_key in getattr(self, f"{stage}_{metric_name}_mapping")}
            getattr(self, f"{stage}_{metric_name}")(**filtered_batch)
        
    def _log_metrics(self, stage: Literal["train", "val", "test"]) -> None:
        for metric_name in getattr(self, f"{stage}_metric_names"):
            self.log(f"{stage}/{metric_name}", getattr(self, f"{stage}_{metric_name}"), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def model_step(
        self, batch: Mapping[str, Any], stage: Literal["train", "val", "test"]
    ) -> Mapping[str, Any]:
        batch = self.forward_fn(batch)

        losses = self._calculate_loss(batch)
        self._update_metrics(batch, stage)

        self._log_losses(losses, stage)
        self._log_metrics(stage)
        return {"loss": losses["total"], "batch": batch, "losses": losses}

    def training_step(
        self, batch: Mapping[str, Any], batch_idx: int
    ) -> torch.Tensor:
        return self.model_step(batch, "train")

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(self, batch: Mapping[str, Any], batch_idx: int) -> None:
        return self.model_step(batch, "val")

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        best_metric_value = getattr(self, f"val_{self.best_metric_name}").compute()  # get current val acc
        getattr(self, f"val_{self.best_metric_name}")(best_metric_value)  # update best so far val acc
        # log `self.val_{self.best_metric_name}` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/best", getattr(self, f"val_{self.best_metric_name}").compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch: Mapping[str, Any], batch_idx: int) -> None:
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
