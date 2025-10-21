"""Image transforms adapters for dict-based mini-batches."""

from typing import Any, Callable, Mapping

import torch.nn as nn


class TorchVisionTransform(nn.Module):
    """Apply a torchvision-style transform to each image at `key` in batch.

    :param transform: Callable transform to apply to each image.
    :param key: Batch key containing a sequence of images.
    """

    def __init__(self, transform: Callable, key: str) -> None:
        """Initialize the transform wrapper.

        :param transform: Callable to apply to each image.
        :param key: Batch key pointing to the image sequence.
        """
        super().__init__()
        self.transform = transform
        self.key = key

    def __call__(self, batch: Mapping[str, Any]) -> Mapping[str, Any]:
        """Apply the transform to every element under `key` and return the batch.

        :param batch: Mapping including a sequence at `key` to be transformed.
        :return: The same mapping with transformed elements at `key`.
        """
        batch[self.key] = [self.transform(image) for image in batch[self.key]]
        return batch
