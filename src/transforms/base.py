import abc

import torch.nn as nn

from typing import Mapping, Any, Callable


class BaseTransform(nn.Module, abc.ABC):
    @abc.abstractmethod
    def forward(self, batch: Mapping[str, Any]) -> Mapping[str, Any]:
        pass


class TransformCompose(nn.Module):
    def __init__(
        self,
        transforms: Mapping[str, BaseTransform],
    ):
        super().__init__()
        self.transforms = nn.ModuleList(transforms)

    def forward(self, batch: Mapping[str, Any]) -> Mapping[str, Any]:
        for transform in self.transforms:
            batch = batch | transform(batch)
        return batch


class TransformWrapper(nn.Module):
    def __init__(
        self,
        transform_fn: Callable,
        mapping: Mapping[str, str],
        new_key: str | None = None,
    ):
        super().__init__()
        self.transform_fn = transform_fn
        self.mapping = mapping
        self.new_key = new_key

    def forward(self, batch: Mapping[str, Any]) -> Mapping[str, Any]:
        filtered_batch = {new_key: batch[old_key] for old_key, new_key in self.mapping}
        result = self.transform_fn(**filtered_batch)
        if self.new_key is not None:
            result = {self.new_key: result}
        return batch | result
