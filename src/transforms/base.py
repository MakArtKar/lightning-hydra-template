import abc

import torch.nn as nn

from typing import Mapping, Any, Callable


class BaseTransform(nn.Module, abc.ABC):
    """Abstract base class for batch-level transforms.

    Subclasses should implement ``forward`` to accept a batch mapping and
    return a mapping of computed fields to be merged into the batch.
    """
    @abc.abstractmethod
    def forward(self, batch: Mapping[str, Any]) -> Mapping[str, Any]:
        """Apply the transform to a batch.

        :param batch: Mapping of input tensors/objects keyed by string.
        :return: Mapping of outputs to be merged with the input batch.
        """
        pass


class TransformCompose(nn.Module):
    """Compose multiple ``BaseTransform`` instances sequentially.

    Each transform receives the current batch and may add new keys. The final
    batch is returned after all transforms are applied.
    """
    def __init__(
        self,
        **transforms
    ):
        super().__init__()
        self.transforms = nn.ModuleList(transforms.values())

    def forward(self, batch: Mapping[str, Any]) -> Mapping[str, Any]:
        """Apply transforms in order and merge their outputs into the batch."""
        for transform in self.transforms:
            batch = batch | transform(batch)
        return batch


class TransformWrapper(nn.Module):
    """Wrap a callable as a transform with argument mapping and optional key.

    The callable will be invoked with a filtered/mapped subset of the batch.
    If ``new_key`` is provided, the callable's output will be inserted under
    that key; otherwise the callable should return a mapping to merge.
    """
    def __init__(
        self,
        transform_fn: Callable,
        transform_kwargs: Mapping[str, Any] = {},
        mapping: Mapping[str, str] = {},
        new_key: str | None = None,
    ):
        super().__init__()
        self.transform_fn = transform_fn
        self.mapping = mapping
        self.new_key = new_key
        self.transform_kwargs = transform_kwargs
    
    def forward(self, batch: Mapping[str, Any]) -> Mapping[str, Any]:
        """Invoke the wrapped callable using mapped batch inputs.

        :param batch: Input batch mapping.
        :return: The input batch merged with the callable's outputs.
        """
        filtered_batch = {new_key: batch[old_key] for old_key, new_key in self.mapping.items()}
        result = self.transform_fn(**filtered_batch, **self.transform_kwargs)
        if self.new_key is not None:
            result = {self.new_key: result}
        return batch | result


class ElementwiseTransformWrapper(TransformWrapper):
    """Elementwise variant that applies a callable to each example in a batch.

    Converts a batch mapping of lists into a list of per-example mappings,
    applies ``elementwise_transform_fn`` to each example, and converts back.
    """
    def __init__(
        self,
        transform_fn: Callable,
        transform_kwargs: Mapping[str, Any] = {},
        mapping: Mapping[str, str] = {},
        new_key: str | None = None,
    ):
        super().__init__(self.transform_fn, transform_kwargs, mapping, new_key)
        self.elementwise_transform_fn = transform_fn

    def _batch_to_list(self, batch: Mapping[str, list[Any]]) -> list[Mapping[str, Any]]:
        """Convert a batch of lists to a list of per-example dicts."""
        examples = []
        keys = batch.keys()
        for values in zip(*batch.values()):
            examples.append({key: value for key, value in zip(keys, values)})
        return examples

    def _list_to_batch(self, examples: list[Mapping[str, Any]]) -> Mapping[str, Any]:
        """Convert a list of per-example dicts back to a batch of lists."""
        keys = examples[0].keys()
        result = {}
        for key in keys:
            result[key] = [example[key] for example in examples]
        return result

    def transform_fn(self, **batch) -> Mapping[str, Any]:
        """Apply the elementwise callable across the batch.

        :param batch: Keyword-only batch mapping after argument mapping.
        :return: Mapping or list suitable to be merged into the original batch.
        """
        examples = self._batch_to_list(batch)
        outputs = [self.elementwise_transform_fn(**example, **self.transform_kwargs) for example in examples]
        if self.new_key is None:
            outputs = self._list_to_batch(outputs)
        return outputs
