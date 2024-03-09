from abc import abstractmethod
from typing import Optional

import torch
import torch.nn.functional as F

from src.data.components import AtomsData
from src.data.components.transforms.base import Transform


class AtomsDataTransform(Transform):
    @abstractmethod
    def __call__(self, atoms_data: AtomsData) -> AtomsData:
        raise NotImplementedError()


def add_fully_connected_edges(atoms_data: AtomsData, keep_loops: bool = False):
    n = atoms_data.num_nodes
    adjacency_matrix = torch.ones((n, n), dtype=torch.bool)
    if not keep_loops:
        adjacency_matrix.fill_diagonal_(False)
    # Create edges
    edge_index = torch.argwhere(adjacency_matrix)
    # Add edges to data object
    atoms_data.edge_index = edge_index
    return atoms_data


class AddFullyConnectedEdgesTransform(AtomsDataTransform):
    """Add edges between all nodes."""

    def __init__(self, keep_loops: bool = False):
        self.keep_loops = keep_loops

    def __call__(self, atoms_data: AtomsData) -> AtomsData:
        return add_fully_connected_edges(atoms_data, keep_loops=self.keep_loops)


def add_edges_within_cutoff_distance(
    atoms_data: AtomsData,
    cutoff: float,
    keep_loops: bool = False,
    use_edge_length_as_edge_feature: bool = False,
):
    # Compute distance matrix (num_nodes, num_nodes) from node positions
    distance_matrix = torch.cdist(atoms_data.node_positions, atoms_data.node_positions)
    # Mask where distances are below cutoff
    adjacency_matrix = distance_matrix <= cutoff
    if not keep_loops:
        adjacency_matrix.fill_diagonal_(False)
    # Create edges
    edge_index = torch.argwhere(adjacency_matrix)
    edge_lengths = distance_matrix[adjacency_matrix]  # (num_edges, 1)
    # Add edges to data object
    atoms_data.edge_index = edge_index
    if use_edge_length_as_edge_feature:
        atoms_data.edge_features = edge_lengths.unsqueeze(1)
    return atoms_data


class AddEdgesWithinCutoffDistanceTransform(AtomsDataTransform):
    def __init__(
        self,
        cutoff: float,
        keep_loops: bool = False,
        use_edge_length_as_edge_feature: bool = False,
    ):
        self.cutoff = cutoff
        self.keep_loops = keep_loops
        self.use_edge_length_as_edge_feature = use_edge_length_as_edge_feature

    def __call__(self, atoms_data: AtomsData) -> AtomsData:
        return add_edges_within_cutoff_distance(
            atoms_data,
            cutoff=self.cutoff,
            keep_loops=self.keep_loops,
            use_edge_length_as_edge_feature=self.use_edge_length_as_edge_feature,
        )


class AddZeroForcesTransform(AtomsDataTransform):
    """Add zero forces to nodes with same dimensionality as 'node_positions'."""

    def __call__(self, atoms_data: AtomsData) -> AtomsData:
        """Add zero forces to nodes with same dimensionality as 'node_positions'.

        Args:
            data: A data object with the 'node_positions' property.
        Returns:
            Data object with forces.
        """
        atoms_data.forces = torch.zeros_like(atoms_data.node_positions)
        return atoms_data


class OneHotNodeFeaturesTransform(AtomsDataTransform):
    """One-hot encode node features.

    Args:
        num_classes (int): Number of classes.
    """

    def __init__(self, num_classes: int = 119, scale: float = 1.0):
        self.num_classes = num_classes
        self.scale = scale

    def __call__(self, atoms_data: AtomsData) -> AtomsData:
        assert atoms_data.node_features.max() < self.num_classes
        atoms_data.node_features = (
            torch.nn.functional.one_hot(
                atoms_data.node_features.squeeze(), self.num_classes
            ).float()
            * self.scale
        )
        return atoms_data


class CollapsedOneHotNodeFeaturesTransform(AtomsDataTransform):
    def __init__(
        self,
        node_labels: list[int],
        node_labels_mask: Optional[list[int]] = None,
        scale: float = 1.0,
    ):
        self.encoder_mapping = {nl: i for (i, nl) in enumerate(sorted(node_labels))}
        self.decoder_mapping = {i: nl for (i, nl) in enumerate(sorted(node_labels))}

        node_features_mask = torch.zeros((1, len(self.encoder_mapping)), dtype=torch.bool)
        if node_labels_mask is not None:
            node_features_mask[:, [self.encoder_mapping[v] for v in node_labels_mask]] = True

        self.node_features_mask = node_features_mask
        self.scale = scale

    def __call__(self, atoms_data: AtomsData) -> AtomsData:
        shape = atoms_data.node_features.shape
        assert len(shape) <= 2
        if len(shape) == 2:
            assert atoms_data.node_features.shape[-1] == 1
        node_features = (
            F.one_hot(
                torch.as_tensor(
                    [self.encoder_mapping[v.item()] for v in atoms_data.node_features.view(-1)]
                ),
                self.num_classes,
            ).float()
            * self.scale
        )

        atoms_data.node_features = node_features
        atoms_data.node_features_mask = self.node_features_mask.repeat(node_features.shape[0], 1)

        return atoms_data

    @property
    def num_classes(self) -> int:
        return len(self.encoder_mapping)

    @staticmethod
    def encode(values, encoder, n):
        return


class AddMetalCenterTransform(AtomsDataTransform):
    def __init__(self, node_labels: list[int], output_key: str):
        self.encoder_mapping = {nl: i for (i, nl) in enumerate(sorted(node_labels))}
        self.output_key = output_key

    def __call__(self, atoms_data: AtomsData) -> AtomsData:
        values = torch.unique(atoms_data.node_features)
        flags = torch.tensor([v.item() in self.encoder_mapping for v in values])

        # assert sum(flags) == 1, f"Only one MC allowed, got: '{flags}' for {values}"

        value = values[flags]
        if len(value) > 1:
            _idx = value[0].item()
        else:
            _idx = value.item()

        idx = self.encoder_mapping[_idx]
        one_hot = F.one_hot(torch.as_tensor(idx), self.num_classes).float()

        setattr(atoms_data, self.output_key, one_hot)

        return atoms_data

    @property
    def num_classes(self) -> int:
        return len(self.encoder_mapping)


class AddNodeMaskTransform(AtomsDataTransform):
    def __init__(self, node_labels: list[int], output_key: str = "node_mask"):
        self.node_labels = node_labels
        self.output_key = output_key

    def __call__(self, atoms_data: AtomsData) -> AtomsData:
        shape = atoms_data.node_features.shape
        assert len(shape) <= 2
        if len(shape) == 2:
            assert atoms_data.node_features.shape[-1] == 1
        mask = torch.as_tensor(
            [[v.item() in self.node_labels] for v in atoms_data.node_features.view(-1)]
        )
        setattr(atoms_data, self.output_key, mask)

        return atoms_data


class CenterAroundTransform(AtomsDataTransform):
    def __init__(
        self,
        node_labels: list[int],
    ):
        self.node_labels = torch.as_tensor(node_labels)

    def __call__(self, atoms_data: AtomsData) -> AtomsData:
        shape = atoms_data.node_features.shape
        assert len(shape) <= 2
        if len(shape) == 2:
            assert atoms_data.node_features.shape[-1] == 1

        isin = torch.isin(atoms_data.node_features.view(-1), self.node_labels)
        if torch.sum(isin) == 1:
            atoms_data.node_positions -= atoms_data.node_positions[isin]

        return atoms_data


class ScaleFeaturesTransform(AtomsDataTransform):
    def __init__(self, input_key: str, scale: float, output_key: str = None):
        self.input_key = input_key
        self.scale = scale
        self.output_key = input_key if output_key is None else output_key

    def __call__(self, atoms_data: AtomsData) -> AtomsData:
        unscaled = getattr(atoms_data, self.input_key)
        setattr(atoms_data, self.output_key, unscaled * self.scale)

        return atoms_data
