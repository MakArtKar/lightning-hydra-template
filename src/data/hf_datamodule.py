from typing import Any, Callable, Dict, Optional, Tuple, Mapping

import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, random_split
from datasets import DatasetDict, Dataset

from src.transforms.base import BaseTransform

class ComposedCollateFn:
    def __init__(self, default_collate_fn: Callable, postprocess_fn: Callable):
        self.default_collate_fn = default_collate_fn
        self.postprocess_fn = postprocess_fn

    def __call__(self, batch: Any) -> Any:
        return self.postprocess_fn(self.default_collate_fn(batch))


class HFDataModule(LightningDataModule):
    """LightningDataModule for datasets provided as a Hugging Face DatasetDict.

    Handles splitting the training split into train/validation sets and exposes
    ready-to-use dataloaders configured via ``dataloader_kwargs``.
    """
    def __init__(
        self,
        hf_dict_dataset: DatasetDict,
        val_ratio: float = 0.1,
        transform: BaseTransform | None = None,
        collate_postprocess: BaseTransform | None = None,
        **dataloader_kwargs: dict[str, Any],
    ) -> None:
        """Initialize a `HFDataModule`.

        :param hf_dict_dataset: The Hugging Face DictDataset.
        :param val_ratio: The ratio of the validation set. Defaults to `0.1`.
        :param dataloader_kwargs: The keyword arguments for the dataloader.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=dataloader_kwargs.keys())

        self.hf_dict_dataset = hf_dict_dataset
        self.val_ratio = val_ratio
        self.dataloader_kwargs = dataloader_kwargs

        if collate_postprocess is not None:
            dataloader_kwargs["collate_fn"] = ComposedCollateFn(
                dataloader_kwargs.get("collate_fn", torch.utils.data.default_collate),
                collate_postprocess,
            )

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        if transform is not None:
            self.hf_dict_dataset.set_transform(transform, output_all_columns=True)

    def setup(self, stage: Optional[str] = None) -> None:
        """Prepare datasets and adjust dataloader settings for the current stage.

        - Divides batch size by world size when running distributed.
        - Splits the HF dataset into train/val/test on first call.

        :param stage: Optional Lightning stage ("fit", "test", "predict").
        """
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            batch_size = self.dataloader_kwargs.get("batch_size", 1)
            if batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.dataloader_kwargs["batch_size"] = batch_size // self.trainer.world_size

        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            self.data_test = self.hf_dict_dataset["test"]
            self.data_train, self.data_val = self.hf_dict_dataset["train"].train_test_split(self.val_ratio, seed=42).values()

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            **self.dataloader_kwargs,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            **self.dataloader_kwargs,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            **self.dataloader_kwargs,
            shuffle=False,
        )
