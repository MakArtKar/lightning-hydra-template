"""Composable transforms operating on dict-based mini-batches."""

from typing import Any, Callable, Mapping

import torch.nn as nn


class RenameTransform(nn.Module):
    """Create a view of batch dict with keys renamed per mapping.

    :param mapping: Mapping from new keys to old keys in the batch.
    """

    def __init__(self, mapping: Mapping[str, str]) -> None:
        """Initialize the transform.

        :param mapping: Mapping from desired new keys to existing batch keys.
        """
        super().__init__()
        self.mapping = mapping

    def __call__(self, batch: Mapping[str, Any]) -> Mapping[str, Any]:
        """Return a new dict with keys renamed according to mapping."""
        return {new_key: batch[old_key] for new_key, old_key in self.mapping.items()}


class ComposeTransform(nn.Module):
    """Compose dict-to-dict transforms, merging outputs into the batch.

    :param transforms: Named sub-transforms applied in insertion order.
    """

    def __init__(self, **transforms) -> None:
        """Initialize the composed transform container.

        :param transforms: Keyword-named transforms applied sequentially.
        """
        super().__init__()
        self.transforms = nn.ModuleDict(transforms)

    def __call__(self, batch: Mapping[str, Any]) -> Mapping[str, Any]:
        """Apply contained transforms sequentially, merging their outputs."""
        for transform in self.transforms.values():
            batch = batch | transform(batch)
        return batch


class WrapTransform(nn.Module):
    """Wrap a callable, building kwargs from `mapping` and writing to `new_key`."""

    def __init__(
        self,
        transform: Callable,
        new_key: str | None = None,
        mapping: Mapping[str, str] | None = None,
        transform_kwargs: Mapping[str, Any] | None = None,
        method_name: str | None = None,
    ):
        """Initialize the wrapper.

        :param transform: Callable to execute with remapped inputs.
        :param new_key: Key to place the callable output under. If None, the output is returned as
            is (should be a dict).
        :param mapping: Optional mapping from callable kwargs to batch keys.
        :param transform_kwargs: Optional kwargs to pass to the transform.
        :param method_name: If not None, extract this method from the transform class instance.
        """
        super().__init__()
        if method_name is not None:
            self.transform = getattr(transform, method_name)
        else:
            self.transform = transform
        self.new_key = new_key
        self.mapping = mapping
        self.transform_kwargs = transform_kwargs or {}

    def __call__(self, batch: Mapping[str, Any]) -> Mapping[str, Any]:
        """Build kwargs from mapping, call the underlying transform, store output."""
        if self.mapping is not None:
            input_batch = {
                kwarg_name: batch[batch_key] for kwarg_name, batch_key in self.mapping.items()
            }
        else:
            input_batch = batch
        output = self.transform(**input_batch, **self.transform_kwargs)
        if self.new_key is not None:
            output = {self.new_key: output}
        return {**batch, **output}
