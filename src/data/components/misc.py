from typing import Callable, Iterable

import torch
from torch.utils.data import Dataset, IterableDataset


class DummyDataset(Dataset):
    def __init__(self, size: int):
        self.size = size

    def __getitem__(self, index: int):
        return torch.tensor(index)

    def __len__(self):
        return self.size


class ExperienceSourceDataset(IterableDataset):
    """
    Implementation from PyTorch Lightning Bolts:
    https://github.com/PyTorchLightning/pytorch-lightning-bolts/blob/master/pl_bolts/datamodules/experience_source.py

    Basic experience source dataset. Takes a generate_batch function that returns an iterator.
    The logic for the experience source and how the batch is generated is defined the Lightning model itself
    """

    def __init__(self, generate_batch: Callable):
        self.generate_batch = generate_batch

    def __iter__(self) -> Iterable:
        iterator = self.generate_batch()
        return iterator
