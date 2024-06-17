from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import partial
from typing import Any, Dict, Optional, Tuple

import torch
from lightning import LightningDataModule
from lightning.pytorch.utilities import CombinedLoader
from torch.utils.data import DataLoader, Dataset, default_collate

from src.utils import pcd_collate_fn, point_collate_fn


class BaseDataModule(LightningDataModule):
    """`LightningDataModule` for basic datasets.

    A `LightningDataModule` implements 7 key methods:

    ```python
        def prepare_data(self):
        # Things to do on 1 GPU/TPU (not on every GPU/TPU in DDP).
        # Download data, pre-process, split, save to disk, etc...

        def setup(self, stage):
        # Things to do on every process in DDP.
        # Load data, set variables, etc...

        def train_dataloader(self):
        # return train dataloader

        def val_dataloader(self):
        # return validation dataloader

        def test_dataloader(self):
        # return test dataloader

        def predict_dataloader(self):
        # return predict dataloader

        def teardown(self, stage):
        # Called on every process in DDP.
        # Clean up after fit or test.
    ```

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        **kwargs,
    ) -> None:
        """Initialize a `MNISTDataModule`.

        :param data_dir: The data directory. Defaults to `"data/"`.
        :param train_val_test_split: The train, validation and test split. Defaults to `(55_000, 5_000, 10_000)`.
        :param batch_size: The batch size. Defaults to `64`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        # self.batch_size_per_device_train = self.hparams.batch_size_train
        # self.batch_size_per_device_val = self.hparams.batch_size_val
        # self.batch_size_per_device_test = self.hparams.batch_size_test

    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            self.data_train = self.hparams.get("train")
            self.data_val = self.hparams.get("val")
            self.data_test = self.hparams.get("test")

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        if hasattr(self.data_train, "_collate_fn"):
            _collate_fn = self.data_train._collate_fn
        elif "pcd" not in self.data_train.__repr__().lower():
            _collate_fn = default_collate
        else:
            _collate_fn = pcd_collate_fn

        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size_train,
            num_workers=self.hparams.num_workers,
            persistent_workers=(
                True if self.hparams.num_workers > 0 else False
            ),  # https://github.com/Lightning-AI/lightning/issues/10389 , speed up 3x data workers initialization
            shuffle=True,
            pin_memory=self.hparams.pin_memory,
            collate_fn=_collate_fn,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        if hasattr(self.data_val, "_collate_fn"):
            _collate_fn = self.data_val._collate_fn
        elif "pcd" not in self.data_val.__repr__().lower():
            _collate_fn = default_collate
        else:
            _collate_fn = pcd_collate_fn

        num_workers = self.hparams.get("num_workers_val", self.hparams.num_workers)

        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size_val,
            num_workers=num_workers,
            persistent_workers=True if num_workers > 0 else False,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=_collate_fn,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        if hasattr(self.data_train, "_collate_fn"):
            _collate_fn = self.data_train._collate_fn
        elif "pcd" not in self.data_test.__repr__().lower():
            _collate_fn = default_collate
        else:
            _collate_fn = pcd_collate_fn

        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size_test,
            num_workers=self.hparams.num_workers,
            persistent_workers=True if self.hparams.num_workers > 0 else False,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=_collate_fn,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass
