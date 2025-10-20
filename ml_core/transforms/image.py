from typing import Any, Callable, Mapping

import torch.nn as nn

class TorchVisionTransform(nn.Module):
    def __init__(self, transform: Callable, key: str) -> None:
        super().__init__()
        self.transform = transform
        self.key = key

    def __call__(self, batch: Mapping[str, Any]) -> Mapping[str, Any]:
        batch[self.key] = [self.transform(image) for image in batch[self.key]]
        return batch
