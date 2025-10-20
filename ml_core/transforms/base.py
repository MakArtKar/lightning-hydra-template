from typing import Any, Callable, Mapping

import torch.nn as nn


class RenameTransform(nn.Module):
    def __init__(self, mapping: Mapping[str, str]) -> None:
        super().__init__()
        self.mapping = mapping

    def __call__(self, batch: Mapping[str, Any]) -> Mapping[str, Any]:
        return {new_key: batch[old_key] for old_key, new_key in self.mapping.items()}


class ComposeTransform(nn.Module):
    def __init__(self, **transforms) -> None:
        super().__init__()
        self.transforms = nn.ModuleDict(transforms)

    def __call__(self, batch: Mapping[str, Any]) -> Mapping[str, Any]:
        for transform in self.transforms.values():
            batch = batch | transform(batch)
        return batch


class WrapTransform(nn.Module):
    def __init__(
        self, transform: Callable, new_key: str, mapping: Mapping[str, str] | None = None
    ):
        super().__init__()
        self.transform = transform
        self.new_key = new_key
        self.mapping = mapping

    def __call__(self, batch: Mapping[str, Any]) -> Mapping[str, Any]:
        if self.mapping is not None:
            input_batch = {new_key: batch[old_key] for old_key, new_key in self.mapping.items()}
        else:
            input_batch = batch
        output = self.transform(**input_batch)
        return batch | {self.new_key: output}
