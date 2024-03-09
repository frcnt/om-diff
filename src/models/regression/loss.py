import copy
from typing import Tuple, Optional

import copy
from typing import Tuple, Optional

import src.models.components.ops as ops
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.components.atomistic import AtomisticModel
from src.models.components.regressor import TimeConditionedRegressor

from src.data.components import Batch


class RegressorLoss(nn.Module):
    def __init__(self, weights: Optional[dict[str, float]] = None):
        super(RegressorLoss, self).__init__()
        if weights is not None:
            self.weights = nn.ParameterDict(
                {
                    k: nn.Parameter(torch.as_tensor(weights[k]), requires_grad=False)
                    for k in weights
                }
            )
        else:
            self.weights = None

    def forward(
            self,
            regressor: AtomisticModel,
            batch: Batch,
    ) -> Tuple[float, dict[str, torch.Tensor]]:
        targets = {key: getattr(batch, key) for key in regressor.model_outputs}

        predicted = regressor.forward(batch)
        loss, losses = self.compute_losses(predicted, targets)

        return loss, losses

    def compute_losses(self, predicted: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]):
        loss, losses = None, {}
        for key in predicted:
            p, t = predicted[key], targets[key]
            loss_key = self.compute_loss(p, t)
            losses[key] = loss_key

            if self.weights is not None and key in self.weights:
                loss_key = loss_key * self.weights[key]

            if loss is None:
                loss = loss_key
            else:
                loss = loss + loss_key

        return loss, losses

    def compute_loss(self, p: torch.Tensor, t: torch.Tensor):
        raise NotImplementedError


class RegressorMSELoss(RegressorLoss):
    def compute_loss(self, p: torch.Tensor, t: torch.Tensor):
        return F.mse_loss(p, t)


class RegressorHuberLoss(RegressorLoss):
    def __init__(
            self,
            weights: Optional[dict[str, float]] = None,
            delta: float = 1.0,
    ):
        super(RegressorHuberLoss, self).__init__(weights=weights)
        self.register_buffer("delta", torch.as_tensor(delta))

    def compute_loss(self, p: torch.Tensor, t: torch.Tensor):
        return F.huber_loss(p, t, delta=self.delta)


class RegressorReverseHuberLoss(RegressorLoss):
    def __init__(
            self,
            weights: Optional[dict[str, float]] = None,
            delta: float = 1.0,
    ):
        super(RegressorReverseHuberLoss, self).__init__(weights=weights)
        self.register_buffer("delta", torch.as_tensor(delta))

    def compute_loss(self, p: torch.Tensor, t: torch.Tensor):
        x = torch.abs(p - t)
        delta = torch.minimum(0.2 * torch.max(x).detach(), self.delta)

        return torch.mean(torch.where(x < delta, x, (x * x + delta * delta) / (2 * delta)))


class TimeConditionedRegressorLoss(RegressorLoss):
    def __init__(self, weights: Optional[dict[str, float]] = None, on_clean: bool = False):
        super(TimeConditionedRegressorLoss, self).__init__(weights)
        self.on_clean = on_clean

    def forward(
            self,
            regressor: TimeConditionedRegressor,
            batch: Batch,
    ) -> Tuple[float, dict[str, torch.Tensor]]:
        targets = {key: getattr(batch, key) for key in regressor.regressor_net.model_outputs}

        if self.on_clean:
            clean_batch = copy.deepcopy(batch)
            ts = torch.zeros(
                (batch.num_data, 1),
                device=batch.node_features.device,
            )
            clean_batch.node_positions = ops.center_splits(
                clean_batch.node_positions, clean_batch.num_nodes
            )
            clean_batch = regressor.update_connectivity(
                clean_batch
            )  # in case the data part does not handle connectivity.
            predicted = regressor.forward(clean_batch, ts)
            clean_loss, clean_losses = self.compute_losses(predicted, targets)
            clean_losses = {f"clean_{key}": clean_losses[key] for key in clean_losses}

        else:
            clean_loss, clean_losses = 0.0, {}

        ts = torch.randint(
            0,  # depends on t0_always
            regressor.timesteps + 1,
            size=(batch.num_data, 1),
            device=batch.node_features.device,
        )
        noisy_inputs = regressor.forward_diffusion(batch, ts)
        predicted = regressor.forward(noisy_inputs, ts)

        loss, losses = self.compute_losses(predicted, targets)

        return loss + clean_loss, {**losses, **clean_losses}

    def compute_loss(self, p: torch.Tensor, t: torch.Tensor):
        raise NotImplementedError


class TimeConditionedRegressorMSELoss(TimeConditionedRegressorLoss):
    def compute_loss(self, p: torch.Tensor, t: torch.Tensor):
        return F.mse_loss(p, t)


class TimeConditionedRegressorHuberLoss(TimeConditionedRegressorLoss):
    def __init__(
            self,
            weights: Optional[dict[str, float]] = None,
            on_clean: bool = False,
            delta: float = 1.0,
    ):
        super(TimeConditionedRegressorHuberLoss, self).__init__(weights=weights, on_clean=on_clean)
        self.register_buffer("delta", torch.as_tensor(delta))

    def compute_loss(self, p: torch.Tensor, t: torch.Tensor):
        return F.huber_loss(p, t, delta=self.delta)
