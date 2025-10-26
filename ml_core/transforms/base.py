"""Composable transforms operating on dict-based mini-batches."""

from typing import Any, Callable, Mapping

import torch.nn as nn


class RenameTransform(nn.Module):
    """Create a view of batch dict with keys renamed per mapping.

    :param mapping: Mapping from old keys to new keys to construct.
    """

    def __init__(self, mapping: Mapping[str, str]) -> None:
        """Initialize the transform.

        :param mapping: Mapping from existing keys to desired new keys.
        """
        super().__init__()
        self.mapping = mapping

    def __call__(self, batch: Mapping[str, Any]) -> Mapping[str, Any]:
        """Return a new dict with keys renamed according to mapping."""
        return {new_key: batch[old_key] for old_key, new_key in self.mapping.items()}


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
    """Wrap a callable, building kwargs from `mapping` and writing to `new_key`.

    :param transform: Callable to execute with remapped inputs.
    :param new_key: Key to place the callable output under.
    :param mapping: Optional mapping from batch to callable argument names.
    """

    def __init__(
        self,
        transform: Callable,
        new_key: str,
        mapping: Mapping[str, str] | None = None,
    ):
        """Initialize the wrapper.

        :param transform: Callable to execute with remapped inputs.
        :param new_key: Key to place the callable output under.
        :param mapping: Optional mapping from batch keys to callable kwargs.
        """
        super().__init__()
        self.transform = transform
        self.new_key = new_key
        self.mapping = mapping

    def __call__(self, batch: Mapping[str, Any]) -> Mapping[str, Any]:
        """Build kwargs from mapping, call the underlying transform, store output."""
        if self.mapping is not None:
            input_batch = {
                new_key: batch[old_key] for old_key, new_key in self.mapping.items()
            }
        else:
            input_batch = batch
        output = self.transform(**input_batch)
        return batch | {self.new_key: output}
