"""Data object classes."""

from typing import List, Optional

import torch


class BaseData:
    """Base class for data objects.

    Store all tensors in a dict for easy access and enumeration.
    """

    def __init__(self, **kwargs):
        self.tensors = dict()
        for key, value in kwargs.items():
            self.__setattr__(key, value)

    def __getattr__(self, key):
        # try to get from self.tensors and get everything else as normal
        try:
            return self.tensors[key]
        except KeyError:
            return super().__getattr__(key)

    def __setattr__(self, key, value):
        # store tensors in self.tensors and everything else in self.__dict__
        # while ensuring the same key is not in both self.tensors and self.__dict__
        if isinstance(value, torch.Tensor):
            self.tensors[key] = value
            self.__dict__.pop(key, None)  # remove key from __dict__ if it exists
        else:
            super().__setattr__(key, value)
            self.tensors.pop(key, None)  # remove key from tensors if it exists

    def __getstate__(self):
        # pickle
        return self.__dict__

    def __setstate__(self, state: dict):
        # unpickle
        self.__dict__ = state

    def validate(self) -> bool:
        """Validate the data."""
        for key, tensor in self.tensors.items():
            assert isinstance(tensor, torch.Tensor), f"'{key}' is not a tensor!"
        return True

    def to(self, device):
        """Move all tensors to the given device."""
        self.tensors = {k: v.to(device) for k, v in self.tensors.items()}

        return self


class Data(BaseData):
    """A data object describing a homogeneous graph.

    Includes general graph information about: nodes, edges, target labels and global features.

    Args:
        node_features (tensor): Node feature tensor with shape (num_nodes, num_node_features).
        edge_index (tensor): Edge index (adjacency list) tensor with shape (num_edges, 2).
                             Each edge has the form [source_node_index, target_node_index].
        edge_features (tensor): Edge feature tensor with shape (num_edges, num_edge_features).
        targets (tensor): Graph-level or node-level target tensor with arbitrary shape.
        global_features (tensor): Graph-level features.
    """

    def __init__(
        self,
        node_features: torch.Tensor,
        edge_index: Optional[torch.LongTensor] = None,
        edge_features: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        global_features: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # nodes
        self.node_features = node_features
        # TODO: could add another type of node feature specifically for embedding
        # (node_embed_index or node_type)
        # edges
        self.edge_index = (
            torch.tensor([], dtype=torch.long, device=node_features.device)
            if edge_index is None
            else edge_index
        )
        self.edge_features = edge_features
        # global
        self.global_features = global_features
        # arbitrary
        self.targets = targets
        # TODO: could also do node_targets, edge_targets, global_targets
        # Use kwargs to the tensors dict for arbitrary properties

    def validate(self) -> bool:
        """Validate the data."""
        super().validate()
        assert self.num_nodes > 0
        assert self.node_features.shape[0] == self.num_nodes
        assert self.node_features.ndim >= 2
        assert self.edge_index.shape[0] == self.num_edges
        assert self.edge_index.shape[0] == 0 or self.edge_index.shape[1] == 2
        assert self.edge_index.shape[0] == 0 or self.edge_index.max() < self.num_nodes
        if self.edge_features is not None:
            assert self.edge_features.shape[0] == self.num_edges
            assert self.num_edges == 0 or self.edge_features.ndim >= 2
        return True

    @property
    def num_nodes(self) -> int:
        """Get number of nodes in the data."""
        # try to get num_nodes from tensors, else get it from node_features
        return self.tensors.get("num_nodes", self.node_features.shape[0])

    @property
    def num_edges(self) -> int:
        """Get number of edges in the data."""
        # try to get num_edges from tensors, else get it from edge_index
        return self.tensors.get("num_edges", self.edge_index.shape[0])

    @property
    def edge_index_source(self) -> torch.Tensor:
        """Get indices of source nodes."""
        return self.edge_index[:, 0]

    @property
    def edge_index_target(self) -> torch.Tensor:
        """Get indices of target nodes."""
        return self.edge_index[:, 1]


class AtomsData(Data):
    """A data object describing am atoms graph.

    The AtomsData object can represent an isolated molecule, or a periodically repeated structure.
    Loosely based on: https://wiki.fysik.dtu.dk/ase/ase/atoms.html#ase.Atoms

    Args:
        node_positions (tensor): Tensor of node positions with shape (num_nodes, num_dimensions).
    """

    def __init__(self, node_positions: torch.Tensor, **kwargs):
        super().__init__(**kwargs)
        self.node_positions = node_positions

    def validate(self) -> bool:
        """Validate the data."""
        super().validate()
        assert self.node_positions.shape[0] == self.num_nodes
        return True


class Batch(Data):
    """An object representing a batch of data.

    Typically a disjoint union of graphs. See collate_data function.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def validate(self) -> bool:
        """Validate the data."""
        for key, tensor in self.tensors.items():
            assert isinstance(tensor, torch.Tensor), f"'{key}' is not a tensor!"
        assert self.node_features.shape[0] == torch.sum(self.num_nodes)
        assert self.node_features.shape[0] == 0 or self.node_features.ndim >= 2
        assert self.edge_index.shape[0] == torch.sum(self.num_edges)
        assert self.edge_index.shape[0] == 0 or self.edge_index.shape[1] == 2
        assert self.edge_index.shape[0] == 0 or self.edge_index.max() < self.node_features.shape[0]
        assert self.num_data >= 1
        if self.edge_features is not None:
            assert self.edge_features.shape[0] == self.edge_index.shape[0]
            assert self.edge_features.ndim >= 2
        if self.global_features is not None:
            assert self.global_features.shape[0] == self.num_data
        if self.targets is not None:
            assert self.targets.ndim >= 2
        return True

    @property
    def num_data(self) -> int:
        """Get number of data in the batch."""
        return self.num_nodes.shape[0]


def collate_data(list_of_data: List[Data]) -> Batch:
    """Collate a list of data objects into a batch object.

    The input graphs are combined into a single graph as a disjoint union by
    concatenation of all data and appropriate adjustment of the edge_index.

    It is assumed that all the input data objects have the same keys/tensors
    and that tensor shapes line up.

    This function can be used with torch.utils.data.DataLoader.

    Args:
        list_of_data (list): A list of data objects.
    Returns:
        A Batch object representing a disjoint union of the input data.
    """
    device = list_of_data[0].node_features.device

    batch = dict()
    # Add num_nodes and num_edges to the batch
    batch["num_nodes"] = torch.tensor([d.num_nodes for d in list_of_data], device=device)
    batch["num_edges"] = torch.tensor([d.num_edges for d in list_of_data], device=device)
    # Compute edge index offset from num_nodes
    offset = torch.cumsum(batch["num_nodes"], dim=0) - batch["num_nodes"]
    # Add offset edge_index to the batch
    batch["edge_index"] = torch.cat([d.edge_index + offset[i] for i, d in enumerate(list_of_data)])
    batch["index"] = index = torch.repeat_interleave(
        torch.arange(batch["num_nodes"].shape[0], dtype=torch.long), batch["num_nodes"]
    )
    # Concatenate and include remaining tensors in the batch
    for k in list_of_data[0].tensors.keys():  # Use keys from the first data object
        if k not in batch.keys():  # Avoid overwriting for example the edge_index
            try:
                batch[k] = torch.cat([torch.atleast_2d(d.tensors[k]) for d in list_of_data])
            except Exception as e:
                raise Exception(f"Failed to add '{k}' to batch:", e)
    # Create a Batch object from the batch dict and return
    return Batch(**batch)
