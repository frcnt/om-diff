from typing import Tuple, Optional, Union, Iterable

import torch
import torch.nn as nn

import src.models.ops as ops
from src.data.components import Batch
from src.models.diffusion.model import OMDiff


class DiffusionLoss(nn.Module):
    def __init__(self):
        super(DiffusionLoss, self).__init__()

    def forward(
            self,
            edm: OMDiff,
            batch: Batch,
    ) -> Tuple[float, dict[str, torch.Tensor]]:
        raise NotImplementedError


class DiffusionL2Loss(DiffusionLoss):
    def __init__(self, weights: Optional[dict[str, float]] = None):
        super(DiffusionL2Loss, self).__init__()
        if weights is None:
            weights = {"node_positions": 1.0, "node_features": 1.0}

        self.weights = nn.ParameterDict(
            {k: nn.Parameter(torch.as_tensor(weights[k]), requires_grad=False) for k in weights}
        )

    def forward(
            self,
            edm: OMDiff,
            batch: Batch,
    ) -> Tuple[float, dict[str, torch.Tensor]]:
        ts = torch.randint(
            0,  # depends on t0_always
            edm.timesteps + 1,
            size=(batch.num_data, 1),
            device=batch.node_features.device,
        )

        noisy_inputs, target_noise, gamma_t = edm.forward_diffusion(batch, ts)
        predicted_noise = edm.forward(noisy_inputs, ts)

        # need to make sure that keys are the same
        # dimensions match batch.num_nodes
        loss, losses = self.l2_loss(
            predicted_noise, target_noise, splits=batch.num_nodes, key_weight=self.weights
        )
        # on a graph basis, need to aggregate
        loss = torch.mean(loss)
        losses = {key: torch.mean(value) for (key, value) in losses.items()}

        return loss, losses

    @staticmethod
    def l2_loss(
            predicted: dict[str, torch.Tensor],
            targets: dict[str, torch.Tensor],
            splits: torch.LongTensor,
            graph_weight: Optional[torch.Tensor] = None,
            key_weight: Optional[Union[dict[str, torch.Tensor], nn.ParameterDict]] = None,
            keys: Iterable[str] = None,
    ) -> Tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if keys is None:
            keys = predicted.keys()

        loss, losses = None, {}
        for key in keys:
            p, t = predicted[key], targets[key]
            node_error = torch.sum(
                torch.square(t - p), dim=-1, keepdim=True
            )  # mean across features
            graph_error = ops.mean_splits(node_error, splits)  # mean across nodes

            if graph_weight is not None:
                graph_error *= graph_weight

            loss_key = 0.5 * graph_error
            losses[key] = loss_key

            if key_weight is not None and key in key_weight:
                loss_key = loss_key * key_weight[key]

            if loss is None:
                loss = loss_key
            else:
                loss = loss + loss_key
        return loss, losses
