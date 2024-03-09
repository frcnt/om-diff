import torch
import torch.nn as nn

from src.data.components import Batch


class Connectivity(nn.Module):
    @torch.no_grad()
    def forward(self, inputs: Batch) -> Batch:
        raise NotImplementedError


def fully_connected(num_nodes: torch.Tensor, keep_loops: bool = False):
    device = num_nodes.device
    offset = 0
    edge_indices, num_edges = [], []
    for n in num_nodes:
        adjacency_matrix = torch.ones((n, n), dtype=torch.bool, device=device)
        if not keep_loops:
            adjacency_matrix.fill_diagonal_(False)
        # Create edges
        edge_index = offset + torch.argwhere(adjacency_matrix)
        edge_indices.append(edge_index)

        offset += n
        num_edges.append(len(edge_index))

    edge_indices = torch.cat(edge_indices, dim=0)
    num_edges = torch.as_tensor(num_edges, device=device)

    return edge_indices, num_edges


class FullyConnected(Connectivity):
    def __init__(self, keep_loops: bool = False):
        super(FullyConnected, self).__init__()
        self.keep_loops = keep_loops

    @torch.no_grad()
    def forward(self, inputs: Batch) -> Batch:
        edge_index, num_edges = fully_connected(inputs.num_nodes, keep_loops=self.keep_loops)
        inputs.edge_index, inputs.num_edges = edge_index, num_edges
        return inputs


def connected_within_cutoff(
    num_nodes: torch.Tensor, positions: torch.Tensor, cutoff: float = 5.0, keep_loops: bool = False
):
    edge_index, num_edges = fully_connected(num_nodes, keep_loops=keep_loops)
    distances = torch.norm(positions[edge_index[:, 0]] - positions[edge_index[:, 1]], p=2, dim=-1)
    mask = distances <= cutoff
    num_edges = torch.as_tensor(
        [torch.sum(split) for split in torch.split(mask, num_edges.tolist())],
        device=num_nodes.device,
        dtype=torch.long,
    )
    return edge_index[mask], num_edges


class ConnectedWithinCutoff(Connectivity):
    def __init__(self, cutoff: float = 5.0, keep_loops: bool = False):
        super(ConnectedWithinCutoff, self).__init__()
        self.cutoff = cutoff
        self.keep_loops = keep_loops

    @torch.no_grad()
    def forward(self, inputs: Batch) -> Batch:
        edge_index, num_edges = connected_within_cutoff(
            inputs.num_nodes, inputs.node_positions, cutoff=self.cutoff, keep_loops=self.keep_loops
        )
        inputs.edge_index, inputs.num_edges = edge_index, num_edges
        return inputs


def connected_with_knn(
    num_nodes: torch.Tensor, positions: torch.Tensor, k: int = 5, keep_loops: bool = False
):
    device = num_nodes.device
    offset = 0
    edge_indices, num_edges = [], []

    start_idx = 0 if keep_loops else 1  # NOTE: index=0 is the node itself
    for p in torch.split(positions, num_nodes.tolist()):
        n = p.shape[0]

        k_ = min(k, n - 1)
        distances = torch.cdist(p, p)

        _, index_to = torch.sort(distances, dim=-1)
        index_to = index_to[:, start_idx : k_ + 1]

        index_from = torch.repeat_interleave(
            torch.arange(index_to.shape[0], device=num_nodes.device).unsqueeze(1),
            repeats=index_to.shape[1],
            dim=1,
        )

        edge_index = torch.stack((index_from, index_to), dim=-1).view(-1, 2)
        edge_index += offset

        edge_indices.append(edge_index)
        num_edges.append(len(edge_index))

        offset += n

    edge_indices = torch.cat(edge_indices, dim=0)
    num_edges = torch.as_tensor(num_edges, device=device)

    return edge_indices, num_edges


class ConnectedWithKNN(Connectivity):
    def __init__(self, k: int = 5, keep_loops: bool = False):
        super(ConnectedWithKNN, self).__init__()
        self.k = k
        self.keep_loops = keep_loops

    @torch.no_grad()
    def forward(self, inputs: Batch) -> Batch:
        edge_index, num_edges = connected_with_knn(
            inputs.num_nodes, inputs.node_positions, k=self.k, keep_loops=self.keep_loops
        )
        inputs.edge_index, inputs.num_edges = edge_index, num_edges
        return inputs
