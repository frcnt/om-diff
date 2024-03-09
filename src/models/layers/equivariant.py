import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedEquivariantBlock(nn.Module):
    def __init__(
            self,
            hidden_channels,
            out_channels,
            intermediate_channels=None,
    ):
        super(GatedEquivariantBlock, self).__init__()
        self.out_channels = out_channels

        if intermediate_channels is None:
            intermediate_channels = hidden_channels

        self.vec1_proj = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.vec2_proj = nn.Linear(hidden_channels, out_channels, bias=False)

        self.update_net = nn.Sequential(
            nn.Linear(hidden_channels * 2, intermediate_channels),
            nn.SiLU(),
            nn.Linear(intermediate_channels, out_channels * 2),
        )

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.vec1_proj.weight)
        nn.init.xavier_uniform_(self.vec2_proj.weight)
        nn.init.xavier_uniform_(self.update_net[0].weight)
        self.update_net[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.update_net[2].weight)
        self.update_net[2].bias.data.fill_(0)

    def forward(
            self,
            atom_states_scalar: torch.Tensor,
            atom_states_vector: torch.Tensor,
    ):
        vec1 = torch.norm(self.vec1_proj(atom_states_vector), dim=-2)
        vec2 = self.vec2_proj(atom_states_vector)

        x = torch.cat([atom_states_scalar, vec1], dim=-1)
        x, v = torch.split(self.update_net(x), self.out_channels, dim=-1)
        v = v.unsqueeze(1) * vec2

        x = F.silu(x)

        return x, v


class EquivariantReadout(nn.Module):
    def __init__(
            self,
            hidden_channels,
    ):
        super(EquivariantReadout, self).__init__()
        self.output_network = nn.ModuleList(
            [
                GatedEquivariantBlock(
                    hidden_channels,
                    hidden_channels // 2,
                ),
                GatedEquivariantBlock(hidden_channels // 2, 1),
            ]
        )

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.output_network:
            layer.reset_parameters()

    def forward(
            self,
            atom_states_scalar: torch.Tensor,
            atom_states_vector: torch.Tensor,
    ):
        for layer in self.output_network:
            atom_states_scalar, atom_states_vector = layer(atom_states_scalar, atom_states_vector)
        return atom_states_scalar, atom_states_vector.squeeze()
