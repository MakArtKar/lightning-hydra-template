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
        mapping: Mapping[str, Mapping[str, str]],
        weights: float | Mapping[str, float] | None = None,
    ):
        """Initialize composition with components and their wiring.

        :param criterions: Mapping from loss name to callable loss modules.
        :param mapping: For each loss, mapping from criterion kwargs to batch keys.
        :param weights: Per-loss weights to aggregate into total. If None (by default), all weights
            are 1 / num_losses.
        """
        super().__init__()

        self.criterions = criterions
        self.weights = weights or {}
        self.mapping = mapping

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Compute each loss and the weighted total from a batch mapping."""
        total_loss = 0.0
        losses = {}
        for name, criterion in self.criterions.items():
            input_batch = {
                kwarg_name: batch[batch_key]
                for kwarg_name, batch_key in self.mapping[name].items()
            }
            losses[name] = criterion(**input_batch)
            total_loss += self.weights.get(name, 1.0 / len(self.criterions)) * losses[name]
        losses["total"] = total_loss
        return losses

    def keys(self) -> list[str]:
        """Return criterion names for which individual losses are computed."""
        return list(self.criterions.keys())
