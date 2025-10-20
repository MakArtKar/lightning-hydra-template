from typing import Any, Mapping

import torch.nn as nn
from torchmetrics import MetricCollection, Metric


class CriterionsComposition(nn.Module):
    def __init__(
        self,
        criterions: Mapping[str, nn.Module],
        weights: Mapping[str, float],
        mapping: Mapping[str, Mapping[str, str]],
    ):
        super().__init__()

        self.criterions = criterions
        self.weights = weights
        self.mapping = mapping

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        total_loss = 0.0
        losses = {}
        for name, criterion in self.criterions.items():
            input_batch = {new_key: batch[old_key] for old_key, new_key in self.mapping[name].items()}
            losses[name] = criterion(**input_batch)
            total_loss += self.weights[name] * losses[name]
        losses["total"] = total_loss
        return losses

    def keys(self) -> list[str]:
        return list(self.criterions.keys())


class MetricsComposition(MetricCollection):
    def __init__(
        self,
        metrics: Mapping[str, Metric],
        mapping: Mapping[str, Mapping[str, str]],
    ):
        super().__init__(dict(metrics))
        self.mapping = mapping

    def forward(self, batch) -> dict[str, Any]:
        result = {}
        for name in self._modules.keys():
            input_batch = {new_key: batch[old_key] for old_key, new_key in self.mapping[name].items()}
            result[name] = self._modules[name](**input_batch)
        return result
