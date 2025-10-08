import abc

import torch.nn as nn

from typing import Mapping, Any, Callable, Literal, Iterable


class BaseTransform(nn.Module, abc.ABC):
    """Abstract base class for batch-level transforms.

    Subclasses should implement ``forward`` to accept a batch mapping and
    return a mapping of computed fields to be merged into the batch.
    """
    def _list_to_batch(self, examples: list[Mapping[str, Any]]) -> Mapping[str, Any]:
        """Convert a list of per-example dicts to a mapping key -> batch list."""
        keys = examples[0].keys()
        result = {}
        for key in keys:
            result[key] = [example[key] for example in examples]
        return result

    def _batch_to_list(self, batch: Mapping[str, list[Any]]) -> list[Mapping[str, Any]]:
        """Convert a batch of lists to a list of per-example dicts."""
        examples = []
        keys = batch.keys()
        for values in zip(*batch.values()):
            examples.append({key: value for key, value in zip(keys, values)})
        return examples
    
    def _get_batch_type(self, batch: Mapping[str, Any] | Iterable[Mapping[str, Any]]) -> Literal["batch", "list"]:
        """Get the type of the batch."""
        if isinstance(batch, Mapping):
            return "batch"
        else:
            return "list"

    def _convert_batch_to_type(self, batch: Mapping[str, Any] | Iterable[Mapping[str, Any]], batch_type: Literal["batch", "list"]) -> Mapping[str, Any] | Iterable[Mapping[str, Any]]:
        """Convert the batch to the given type."""
        if batch_type == "batch" and self._get_batch_type(batch) == "list":
            return self._list_to_batch(batch)
        elif batch_type == "list" and self._get_batch_type(batch) == "batch":
            return self._batch_to_list(batch)
        else:
            return batch

    def _merge_batches(
        self,
        first_batch: Mapping[str, Any] | Iterable[Mapping[str, Any]],
        second_batch: Mapping[str, Any] | Iterable[Mapping[str, Any]],
    ) -> Mapping[str, Any] | Iterable[Mapping[str, Any]]:
        """Merge two batches."""
        # saving second batch type
        first_batch = self._convert_batch_to_type(first_batch, self._get_batch_type(second_batch))

        # overriding first batch with second batch
        if self._get_batch_type(first_batch) == "batch":
            return first_batch | second_batch
        else:
            return [first_batch_item | second_batch_item for first_batch_item, second_batch_item in zip(first_batch, second_batch)]

    @abc.abstractmethod
    def forward(self, batch: Mapping[str, Any] | Iterable[Mapping[str, Any]]) -> Mapping[str, Any] | Iterable[Mapping[str, Any]]:
        """Apply the transform to a batch.

        :param batch: Can be a Mapping of input tensors/objects keyed by string or an Iterable of such Mappings.
        :return: Mapping of outputs to be merged with the input batch or an Iterable of such Mappings.
        """
        pass


class TransformCompose(BaseTransform):
    """Compose multiple ``BaseTransform`` instances sequentially.

    Each transform receives the current batch and may add new keys. The final
    batch is returned after all transforms are applied.
    """
    def __init__(
        self,
        transforms: Mapping[str, BaseTransform],
    ):
        super().__init__()
        self.transforms = nn.ModuleList(transforms.values())

    def forward(self, batch: Mapping[str, Any] | Iterable[Mapping[str, Any]]) -> Mapping[str, Any] | Iterable[Mapping[str, Any]]:
        """Apply transforms in order and merge their outputs into the batch."""
        for transform in self.transforms:
            result = transform(batch)
            batch = self._merge_batches(batch, result)
        return batch


class TransformWrapper(BaseTransform):
    """Wraps a transform applied to a whole batch or separately to each sample of the batch."""
    def __init__(
        self,
        transform_fn: Callable,
        mapping: Mapping[str, str] | None = None,
        is_elementwise: bool = False,
        new_key: str | None = None,
        dot_key: str | None = None,
        transform_kwargs: Mapping[str, Any] = {},
        input_format: Literal["batch", "list"] = "batch",
        output_format: Literal["batch", "list"] = "batch",
    ):
        """Initialize a TransformWrapper.

        :param transform_fn: The callable to wrap. Applied to a whole batch or separately to each sample.
        :param mapping: The mapping of input keys to the `transform_fn` kwargs.
        :param is_elementwise: Whether the transform is applied separately to each sample of the batch.
        :param new_key: In case if the result is returned without key - using `new_key` to insert the result under.
        :param dot_key: In case if the result is returned as an object with attributes - using `dot_key` to extract the result.
        :param transform_kwargs: Extra keyword arguments to pass to the `transform_fn`.
        :param input_format: The format of the input batch. `batch` if the input is a whole batch, `list` if the input is a list of per-example dicts.
        :param output_format: The format of the output batch. `batch` if the output is a whole batch, `list` if the output is a list of per-example dicts.
        """
        super().__init__()
        self.transform_fn = transform_fn
        self.mapping = mapping
        self.is_elementwise = is_elementwise
        self.new_key = new_key
        self.dot_key = dot_key
        self.transform_kwargs = transform_kwargs
        self.input_format = input_format
        self.output_format = output_format

        if self.is_elementwise and not self.input_format == "list":
            raise ValueError("Input format must be 'list' when is_elementwise is True.")

    def forward_elementwisely(self, batch: Iterable[Mapping[str, Any]]) -> Mapping[str, Any]:
        """Apply the transform to each example in the batch."""
        outputs = [self.transform_fn(**example, **self.transform_kwargs) for example in batch]
        
        if self.dot_key is not None:
            outputs = [getattr(output, self.dot_key) for output in outputs]
        if self.new_key is None:
            return self._list_to_batch(outputs)
        else:
            return {self.new_key: outputs}

    def forward_batchwise(self, batch: Mapping[str, Any] | Iterable[Mapping[str, Any]]) -> Mapping[str, Any]:
        """Apply the transform to the whole batch."""
        if self.input_format == "batch":
            result = self.transform_fn(**batch, **self.transform_kwargs)
        else:
            # batch is a list of samples
            result = self.transform_fn(batch, **self.transform_kwargs)

        if self.dot_key is not None:
            result = getattr(result, self.dot_key)
        if self.new_key is not None:
            result = {self.new_key: result}
        return result
    
    def forward(self, batch: Mapping[str, Any] | Iterable[Mapping[str, Any]]) -> Mapping[str, Any] | Iterable[Mapping[str, Any]]:
        """Invoke the wrapped callable using mapped batch inputs.

        :param batch: Input batch mapping.
        :return: The input batch merged with the callable's outputs.
        """
        batch = self._convert_batch_to_type(batch, self.input_format)

        if self.mapping is not None:
            if self.input_format == "batch":
                filtered_batch = {new_key: batch[old_key] for old_key, new_key in self.mapping.items()}
            else:
                filtered_batch = [{new_key: batch_item[old_key] for old_key, new_key in self.mapping.items()} for batch_item in batch]
        else:
            filtered_batch = batch

        if self.is_elementwise:
            result = self.forward_elementwisely(filtered_batch)
        else:
            result = self.forward_batchwise(filtered_batch)
        
        batch = self._merge_batches(batch, result)
        batch = self._convert_batch_to_type(batch, self.output_format)
        return batch
