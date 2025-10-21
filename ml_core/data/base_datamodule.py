"""Generic LightningDataModule wrapping a Hugging Face DatasetDict."""

from typing import Any, Callable, Optional

from datasets import DatasetDict
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset


class BaseDataModule(LightningDataModule):
    """LightningDataModule that splits HF datasets and builds DataLoaders.

    Divides batch size across devices in distributed runs and applies an optional
    transform on-the-fly via `with_transform`.
    """

    def __init__(
        self,
        hf_dict_dataset: DatasetDict,
        val_ratio: float = 0.1,
        transform: Callable | None = None,
        **dataloader_kwargs: dict[str, Any],
    ) -> None:
        """Initialize the data module with datasets and loader settings.

        :param hf_dict_dataset: A preloaded Hugging Face `DatasetDict`.
        :param val_ratio: Fraction of train to use for validation split.
        :param transform: Optional mapping transform applied via `with_transform`.
        :param dataloader_kwargs: Extra arguments forwarded to DataLoader.
        """
        super().__init__()

        self.save_hyperparameters(logger=False, ignore=dataloader_kwargs.keys())

        self.hf_dict_dataset = hf_dict_dataset
        self.val_ratio = val_ratio
        self.transform = transform
        self.dataloader_kwargs = dataloader_kwargs

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        if transform is not None:
            self.hf_dict_dataset = self.hf_dict_dataset.with_transform(
                transform, output_all_columns=True
            )

    def setup(self, stage: str | None = None) -> None:
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
            self.data_train, self.data_val = (
                self.hf_dict_dataset["train"].train_test_split(self.val_ratio, seed=42).values()
            )

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
