import shutil
from typing import Tuple, Optional, Type

from src.data.base_datamodule import BaseDataModule
from src.data.components.datasets.ase_dataset import ASEDBDataset
from src.data.components.transforms.base import Compose


class ASEDataModule(BaseDataModule):
    dataset_cls: Type[ASEDBDataset] = ASEDBDataset
    dataset: Optional[ASEDBDataset]
    splits: Optional[dict[str, ASEDBDataset]]

    def __init__(
        self,
        db_path: str,
        train_val_split: Tuple[float, float] = (0.8, 0.1),
        split_file: Optional[str] = None,
        seed_split: int = 42,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        transform_compose: Compose = None,
        in_memory: bool = True,
        move_data_to: str = None,
    ):
        super().__init__(
            train_val_split=train_val_split,
            split_file=split_file,
            seed_split=seed_split,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        self.save_hyperparameters(logger=False)
        self.transform_compose = transform_compose

    def setup_dataset(self):
        if self.hparams.move_data_to is not None:
            db_path = shutil.copy2(self.hparams.db_path, self.hparams.move_data_to)
        else:
            db_path = self.hparams.db_path

        self.dataset = self.dataset_cls(
            db_path=db_path,
            transform=self.transform_compose,
            in_memory=self.hparams.in_memory,
        )
