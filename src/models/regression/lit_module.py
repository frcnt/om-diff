from typing import Optional, Union, Callable

import torch
from src.models.base_lit_module import BaseLitModule
from src.models.atomistic import AtomisticModel
from src.models.connectivity import Connectivity
from src.models.components.losses import (
    TimeConditionedRegressor,
    TimeConditionedRegressorLoss,
    RegressorLoss,
)
from src.models.components.noise.model import NormalNoiseModel

from src.data.base_datamodule import BaseDataModule
from src.data.components import Batch


class RegressorLitModule(BaseLitModule):
    def __init__(
            self,
            regressor_net: AtomisticModel,
            train_loss_module: RegressorLoss,
            val_test_loss_module: RegressorLoss,
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler = None,
            dm: BaseDataModule = None,
    ):
        super().__init__(
            optimizer=optimizer,
            scheduler=scheduler,
            dm=dm,
        )
        self.save_hyperparameters(
            ignore=[
                "regressor_net",
                "train_loss_module",
                "val_test_loss_module",
                "dm",
            ]
        )
        self.train_loss_module = train_loss_module
        self.val_test_loss_module = val_test_loss_module
        self.regressor: AtomisticModel = regressor_net

    def forward(self, batch: Batch):
        return self.regressor.forward(batch)

    def step(self, batch, batch_idx, prefix: str):
        is_train = prefix.startswith("train")
        loss_module = self.train_loss_module if is_train else self.val_test_loss_module

        loss, loss_dict = loss_module(self.regressor, batch)
        self.log(
            f"{prefix}/loss",
            loss,
            batch_size=batch.num_data,
            on_step=is_train,
            on_epoch=not is_train,
        )
        self.log_dict(
            {f"{prefix}/{k}": v for k, v in loss_dict.items()},
            batch_size=batch.num_data,
            on_step=is_train,
            on_epoch=not is_train,
        )
        return loss


class TimeConditionedRegressorLitModule(BaseLitModule):
    def __init__(
            self,
            regressor_net: AtomisticModel,
            train_loss_module: TimeConditionedRegressorLoss,
            val_test_loss_module: TimeConditionedRegressorLoss,
            noise_model: NormalNoiseModel,
            optimizer: torch.optim.Optimizer,
            connectivity_module: Optional[Union[Connectivity, Callable[[Batch], Batch]]] = None,
            scheduler: torch.optim.lr_scheduler = None,
            dm: BaseDataModule = None,
    ):
        super().__init__(
            optimizer=optimizer,
            scheduler=scheduler,
            dm=dm,
        )
        self.save_hyperparameters(
            ignore=[
                "regressor_net",
                "train_loss_module",
                "val_test_loss_module",
                "noise_model",
                "connectivity_module",
                "dm",
            ]
        )
        self.train_loss_module = train_loss_module
        self.val_test_loss_module = val_test_loss_module
        self.regressor: TimeConditionedRegressor = TimeConditionedRegressor(
            regressor_net=regressor_net,
            noise_model=noise_model,
            connectivity_module=connectivity_module,
        )

    def forward(self, batch: Batch, ts: torch.Tensor):
        return self.regressor.forward(batch, ts)

    def step(self, batch, batch_idx, prefix: str):
        is_train = prefix.startswith("train")
        loss_module = self.train_loss_module if is_train else self.val_test_loss_module

        loss, loss_dict = loss_module(self.regressor, batch)
        self.log(
            f"{prefix}/loss",
            loss,
            batch_size=batch.num_data,
            on_step=is_train,
            on_epoch=not is_train,
        )
        self.log_dict(
            {f"{prefix}/{k}": v for k, v in loss_dict.items()},
            batch_size=batch.num_data,
            on_step=is_train,
            on_epoch=not is_train,
        )
        return loss
