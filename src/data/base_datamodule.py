import argparse
from typing import Tuple, Optional, Union, MutableMapping

import lightning as L
import torch.utils.data as torchdata
from lightning.pytorch.utilities import AttributeDict

from src.data.components import collate_data
from src.data.components.datasets.base import BaseDataset
from src.data.components.utils import split_train_val_test, split_from_file


def get_dataloader(
    dataset: torchdata.Dataset,
    shuffle: bool,
    hp: Union[AttributeDict, MutableMapping, argparse.Namespace],
):
    return torchdata.DataLoader(
        dataset,
        batch_size=hp.batch_size,
        collate_fn=collate_data,
        shuffle=shuffle,
        num_workers=hp.num_workers,
        pin_memory=hp.pin_memory,
        drop_last=False,
    )


class BaseDataModule(L.LightningDataModule):
    dataset: Optional[BaseDataset]
    splits: Optional[dict[str, BaseDataset]]

    def __init__(
        self,
        train_val_split: Tuple[float, float] = (0.8, 0.1),
        split_file: Optional[str] = None,
        seed_split: int = 42,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        self.dataset = None
        self.splits = None

    def setup_dataset(self):
        raise NotImplementedError

    def setup(self, stage: Optional[str] = None):
        self.setup_dataset()

        if self.hparams.split_file is None:
            self.splits = split_train_val_test(
                self.dataset,
                train_val_split=self.hparams.train_val_split,
                split_seed=self.hparams.seed_split,
                only_idx=False,
            )
        else:  # a split file was provided
            print(f"A split file: {self.hparams.split_file} was provided.")
            self.splits = split_from_file(
                self.dataset,
                split_file=self.hparams.split_file,
                split_seed=self.hparams.seed_split,
            )

        assert all(k in self.splits.keys() for k in ["train", "val"])

    def train_dataloader(self):
        return self._dataloader("train")

    def val_dataloader(self):
        return self._dataloader("val")

    def test_dataloader(self):
        return self._dataloader("test")

    def _dataloader(self, split: str):
        if split not in self.splits:
            return None
        return get_dataloader(
            dataset=self.splits[split], shuffle=bool(split == "train"), hp=self.hparams
        )
