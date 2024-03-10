from typing import Optional

import torch
import torch.nn as nn

import src.models.ops as ops


class MLP(nn.Module):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            hidden_dim: Optional[int] = None,
            num_hidden_layers: int = 1,
            activation_cls=nn.SiLU,
    ):
        super().__init__()
        assert num_hidden_layers >= 0
        self.input_dim = input_dim
        self.output_dim = output_dim

        if num_hidden_layers == 0:
            self.nn = nn.Linear(input_dim, output_dim)
        else:
            assert hidden_dim is not None
            self.nn = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                activation_cls(),
                *[
                    nn.Sequential(nn.Linear(hidden_dim, hidden_dim), activation_cls())
                    for _ in range(num_hidden_layers - 1)
                ],
                nn.Linear(hidden_dim, output_dim),
            )

    def forward(self, x):
        return self.nn(x)

