import os.path
from typing import Optional

from src.data.ase_datamodule import ASEDataModule
from src.data.components.datasets.suzuki import SuzukiDataset, Suzuki4Dataset, SuzukiFFDataset

__all__ = ["SuzukiDataModule", "Suzuki4DataModule", "SuzukiFFDataModule"]


class SuzukiDataModule(ASEDataModule):
    dataset_cls = SuzukiDataset
    dataset: Optional[SuzukiDataset]
    splits: Optional[dict[str, SuzukiDataset]]

    def prepare_data(self) -> None:
        if not os.path.exists(self.hparams.db_path):
            SuzukiDataset(self.hparams.db_path, download=True, in_memory=False)


class Suzuki4DataModule(ASEDataModule):
    dataset_cls = Suzuki4Dataset
    dataset: Optional[Suzuki4Dataset]
    splits: Optional[dict[str, Suzuki4Dataset]]

    def prepare_data(self) -> None:
        if not os.path.exists(self.hparams.db_path):
            Suzuki4Dataset(self.hparams.db_path, download=True, in_memory=False)


class SuzukiFFDataModule(ASEDataModule):
    dataset_cls = SuzukiFFDataset
    dataset: Optional[SuzukiFFDataset]
    splits: Optional[dict[str, SuzukiFFDataset]]

    def prepare_data(self) -> None:
        if not os.path.exists(self.hparams.db_path):
            SuzukiFFDataset(self.hparams.db_path, download=True, in_memory=False)
