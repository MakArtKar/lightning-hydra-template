from typing import Any, Mapping

import torch.nn as nn
from torchmetrics import Metric, MetricCollection


class CriterionsComposition(nn.Module):
    """Compose multiple loss criteria with input mapping and weights.

    The module pulls named tensors from the batch using `mapping`, applies each
    criterion, aggregates a weighted total under `"total"`, and exposes constituent
    loss names via `keys()`.
    """

    def __init__(
        self,
        criterions: Mapping[str, nn.Module],
        weights: Mapping[str, float],
        mapping: Mapping[str, Mapping[str, str]],
    ):
        """Initialize composition with components and their wiring.

        :param criterions: Mapping from loss name to callable loss modules.
        :param weights: Per-loss weights to aggregate into total.
        :param mapping: For each loss, mapping from batch keys to criterion kwargs.
        """
        super().__init__()

        self.criterions = criterions
        self.weights = weights
        self.mapping = mapping

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Compute each loss and the weighted total from a batch mapping."""
        total_loss = 0.0
        losses = {}
        for name, criterion in self.criterions.items():
            input_batch = {
                new_key: batch[old_key]
                for old_key, new_key in self.mapping[name].items()
            }
            losses[name] = criterion(**input_batch)
            total_loss += self.weights[name] * losses[name]
        losses["total"] = total_loss
        return losses

    def keys(self) -> list[str]:
        """Return criterion names for which individual losses are computed."""
        return list(self.criterions.keys())


class MetricsComposition(MetricCollection):
    """Wrap a MetricCollection with batch-field remapping per metric entry."""

    def __init__(
        self,
        metrics: Mapping[str, Metric],
        mapping: Mapping[str, Mapping[str, str]],
    ):
        """Initialize with metrics and how to pull their inputs from batch."""
        super().__init__(dict(metrics))
        self.mapping = mapping

    def forward(self, batch) -> dict[str, Any]:
        """Evaluate metrics using remapped fields and return their values."""
        result = {}
        for name in self._modules.keys():
            input_batch = {
                new_key: batch[old_key]
                for old_key, new_key in self.mapping[name].items()
            }
            result[name] = self._modules[name](**input_batch)
        return result
