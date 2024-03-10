import torch
import torch.nn as nn

import src.models.ops as ops


class GraphLayerNorm(nn.Module):
    """
    Layer normalisation on a per graph basis.
    """

    def __init__(
            self,
            in_channels: int,
            eps: float = 1e-6,
            affine: bool = True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.eps = eps
        self.affine = affine

        if affine:
            self.weight = nn.Parameter(torch.ones(in_channels))
            self.bias = nn.Parameter(torch.zeros(in_channels))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor, splits: torch.Tensor) -> torch.Tensor:
        x = ops.center_splits(x - x.mean(-1, keepdim=True), splits=splits)
        var_x = torch.mean(ops.mean_splits(x * x, splits=splits), dim=-1, keepdim=True)
        var_x = torch.repeat_interleave(var_x, splits, dim=0)

        out = x / torch.sqrt(var_x + self.eps)

        if self.weight is not None and self.bias is not None:
            out = out * self.weight + self.bias

        return out
