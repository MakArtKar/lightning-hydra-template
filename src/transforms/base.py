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
        self.transforms = nn.ModuleList(transforms.values())

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
        filtered_batch = {new_key: batch[old_key] for old_key, new_key in self.mapping.items()}
        result = self.transform_fn(**filtered_batch)
        if self.new_key is not None:
            result = {self.new_key: result}
        return batch | result


class ElementwiseTransformWrapper(TransformWrapper):
    def __init__(
        self,
        transform_fn: Callable,
        mapping: Mapping[str, str],
        new_key: str | None = None,
    ):
        super().__init__(self.transform_fn, mapping, new_key)
        self.elementwise_transform_fn = transform_fn

    def _batch_to_list(self, batch: Mapping[str, list[Any]]) -> list[Mapping[str, Any]]:
        examples = []
        keys = batch.keys()
        for values in zip(*batch.values()):
            examples.append({key: value for key, value in zip(keys, values)})
        return examples

    def _list_to_batch(self, examples: list[Mapping[str, Any]]) -> Mapping[str, Any]:
        keys = examples[0].keys()
        result = {}
        for key in keys:
            result[key] = [example[key] for example in examples]
        return result

    def transform_fn(self, **batch) -> Mapping[str, Any]:
        examples = self._batch_to_list(batch)
        outputs = [self.elementwise_transform_fn(**example) for example in examples]
        if self.new_key is None:
            outputs = self._list_to_batch(outputs)
        return outputs
