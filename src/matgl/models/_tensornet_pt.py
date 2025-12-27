"""Implementation of TensorNet model.

A Cartesian based equivariant GNN model. For more details on TensorNet,
please refer to::

    G. Simeon, G. de. Fabritiis, _TensorNet: Cartesian Tensor Representations for Efficient Learning of Molecular
    Potentials. _arXiv, June 10, 2023, 10.48550/arXiv.2306.06482.

"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

import torch
from torch import nn
from torch_geometric.data import Batch, Data

import matgl
from matgl.config import DEFAULT_ELEMENTS
from matgl.layers import (
    ActivationFunction,
    BondExpansion,
)
from matgl.layers._embedding_pyg import TensorEmbedding
from matgl.layers._graph_convolution_pyg import TensorNetInteraction
from matgl.layers._readout_pyg import (
    WeightedReadOut,
)
from matgl.utils.maths import decompose_tensor, scatter_add, tensor_norm

from ._core import MatGLModel

if TYPE_CHECKING:
    from matgl.graph._converters_pyg import GraphConverter

logger = logging.getLogger(__file__)


def compute_pair_vector_and_distance(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    pbc_offshift: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Calculate bond vectors and distances.

    Args:
        pos: Node positions, shape (num_nodes, 3)
        edge_index: Edge indices, shape (2, num_edges)
        pbc_offshift: Periodic boundary condition offsets, shape (num_edges, 3)

    Returns:
        bond_vec: Bond vectors, shape (num_edges, 3)
        bond_dist: Bond distances, shape (num_edges,)
    """
    src_idx, dst_idx = edge_index[0], edge_index[1]
    src_pos = pos[src_idx]
    dst_pos = pos[dst_idx]

    if pbc_offshift is not None:
        dst_pos = dst_pos + pbc_offshift

    bond_vec = dst_pos - src_pos
    bond_dist = torch.norm(bond_vec, dim=1)

    return bond_vec, bond_dist


class TensorNet(MatGLModel):
    """The main TensorNet model. The official implementation can be found in https://github.com/torchmd/torchmd-net."""

    __version__ = 1

    def __init__(
        self,
        element_types: tuple[str, ...] = DEFAULT_ELEMENTS,
        units: int = 64,
        ntypes_state: int | None = None,
        dim_state_embedding: int = 0,
        dim_state_feats: int | None = None,
        include_state: bool = False,
        nblocks: int = 2,
        num_rbf: int = 32,
        max_n: int = 3,
        max_l: int = 3,
        rbf_type: Literal["Gaussian", "SphericalBessel"] = "Gaussian",
        use_smooth: bool = False,
        activation_type: Literal["swish", "tanh", "sigmoid", "softplus2", "softexp"] = "swish",
        cutoff: float = 5.0,
        equivariance_invariance_group: str = "O(3)",
        dtype: torch.dtype = matgl.float_th,
        width: float = 0.5,
        readout_type: Literal["weighted_atom", "reduce_atom"] = "weighted_atom",
        task_type: Literal["classification", "regression"] = "regression",
        niters_set2set: int = 3,
        nlayers_set2set: int = 3,
        field: Literal["node_feat", "edge_feat"] = "node_feat",
        is_intensive: bool = True,
        ntargets: int = 1,
        **kwargs,
    ):
        r"""

        Args:
            element_types (tuple): List of elements appearing in the dataset. Default to DEFAULT_ELEMENTS.
            units (int, optional): Hidden embedding size.
                (default: :obj:`64`)
            ntypes_state (int): Number of state labels
            dim_state_embedding (int): Number of hidden neurons in state embedding
            dim_state_feats (int): Number of state features after linear layer
            include_state (bool): Whether to include states features
            nblocks (int, optional): The number of interaction layers.
                (default: :obj:`2`)
            num_rbf (int, optional): The number of radial basis Gaussian functions :math:`\mu`.
                (default: :obj:`32`)
            max_n (int): maximum of n in spherical Bessel functions
            max_l (int): maximum of l in spherical Bessel functions
            rbf_type (str): Radial basis function. choose from 'Gaussian' or 'SphericalBessel'
            use_smooth (bool): Whether to use the smooth version of SphericalBessel functions.
                This is particularly important for the smoothness of PES.
            activation_type (str): Activation type. choose from 'swish', 'tanh', 'sigmoid', 'softplus2', 'softexp'
            cutoff (float): cutoff distance for interatomic interactions.
            equivariance_invariance_group (string, optional): Group under whose action on input
                positions internal tensor features will be equivariant and scalar predictions
                will be invariant. O(3) or SO(3).
               (default :obj:`"O(3)"`)
            dtype (torch.dtype): data type for all variables
            width (float): the width of Gaussian radial basis functions
            readout_type (str): Readout function type, `set2set`, `weighted_atom` (default) or `reduce_atom`.
            task_type (str): `classification` or `regression` (default).
            niters_set2set (int): Number of set2set iterations
            nlayers_set2set (int): Number of set2set layers
            field (str): Using either "node_feat" or "edge_feat" for Set2Set and Reduced readout
            is_intensive (bool): Whether the prediction is intensive
            ntargets (int): Number of target properties
            **kwargs: For future flexibility. Not used at the moment.

        """
        super().__init__()

        self.save_args(locals(), kwargs)

        try:
            activation: nn.Module = ActivationFunction[activation_type].value()
        except KeyError:
            raise ValueError(
                f"Invalid activation type, please try using one of {[af.name for af in ActivationFunction]}"
            ) from None

        self.element_types = element_types  # type: ignore

        self.bond_expansion = BondExpansion(
            cutoff=cutoff,
            rbf_type=rbf_type,
            final=cutoff + 1.0,
            num_centers=num_rbf,
            width=width,
            smooth=use_smooth,
            max_n=max_n,
            max_l=max_l,
        )

        assert equivariance_invariance_group in ["O(3)", "SO(3)"], "Unknown group representation. Choose O(3) or SO(3)."

        self.units = units
        self.equivariance_invariance_group = equivariance_invariance_group
        self.num_layers = nblocks
        self.num_rbf = num_rbf
        self.rbf_type = rbf_type
        self.activation = activation
        self.cutoff = cutoff
        self.dim_state_embedding = dim_state_embedding
        self.dim_state_feats = dim_state_feats
        self.include_state = include_state
        self.ntypes_state = ntypes_state
        self.task_type = task_type

        # make sure the number of radial basis functions correct for tensor embedding
        if rbf_type == "SphericalBessel":
            num_rbf = max_n

        self.tensor_embedding = TensorEmbedding(
            units=units,
            degree_rbf=num_rbf,
            activation=activation,
            ntypes_node=len(element_types),
            cutoff=cutoff,
            dtype=dtype,
        )

        self.layers = nn.ModuleList(
            {
                TensorNetInteraction(num_rbf, units, activation, cutoff, equivariance_invariance_group, dtype)
                for _ in range(nblocks)
                if nblocks != 0
            }
        )

        self.out_norm = nn.LayerNorm(3 * units, dtype=dtype)
        self.linear = nn.Linear(3 * units, units, dtype=dtype)
        if is_intensive:
            raise NotImplementedError("Intensive property prediction is not implemented yet.")
        if task_type == "classification":
            raise ValueError("Classification task cannot be extensive.")
        self.final_layer = WeightedReadOut(
            in_feats=units,
            dims=[units, units],
            num_targets=ntargets,  # type: ignore
        )

        self.is_intensive = is_intensive
        self.reset_parameters()

    def reset_parameters(self):
        self.tensor_embedding.reset_parameters()
        for layer in self.layers:
            layer.reset_parameters()
        self.out_norm.reset_parameters()

    def forward(self, g: dict[str, torch.Tensor] | Data, state_attr: torch.Tensor | None = None, **kwargs):
        """
        Args:
            g: Either a PyG Data object or a dict with keys:
                - 'node_type' or 'z': Node types, shape (num_nodes,)
                - 'pos': Node positions, shape (num_nodes, 3)
                - 'edge_index': Edge indices, shape (2, num_edges)
                - 'pbc_offshift': Optional PBC offsets, shape (num_edges, 3)
                - 'batch': Optional batch indices, shape (num_nodes,)
            state_attr: State attrs for a batch of graphs (not used currently).
            **kwargs: For future flexibility. Not used at the moment.

        Returns:
            output: Output property for a batch of graphs
        """
        if isinstance(g, dict):
            z = g.get("node_type", g.get("z"))
            pos = g["pos"]
            edge_index = g["edge_index"]
            pbc_offshift = g.get("pbc_offshift", None)
            batch = g.get("batch", None)
            num_graphs = g.get("num_graphs", None)
        else:
            # PyG Data object - extract tensors
            z = getattr(g, "node_type", getattr(g, "z", None))
            pos = g.pos  # type: ignore[union-attr]
            edge_index = g.edge_index  # type: ignore[union-attr]
            pbc_offshift = getattr(g, "pbc_offshift", None)
            batch = getattr(g, "batch", None)
            num_graphs = getattr(g, "num_graphs", None)

        num_graphs = int(num_graphs) if num_graphs is not None else 1
        # Obtain graph, with distances and relative position vectors
        bond_vec, bond_dist = compute_pair_vector_and_distance(pos, edge_index, pbc_offshift)

        # Expand distances with radial basis functions
        edge_attr = self.bond_expansion(bond_dist)
        # Embedding layer
        X, _ = self.tensor_embedding(z, edge_index, edge_attr, bond_dist, bond_vec, state_attr)
        # Interaction layers
        for layer in self.layers:
            X = layer(edge_index, bond_dist, edge_attr, X)
        scalars, skew_metrices, traceless_tensors = decompose_tensor(X)

        x = torch.cat((tensor_norm(scalars), tensor_norm(skew_metrices), tensor_norm(traceless_tensors)), dim=-1)
        x = self.out_norm(x)
        x = self.linear(x)

        if self.is_intensive:
            raise NotImplementedError("Intensive property prediction is not implemented yet.")
        atomic_energies = self.final_layer(x)
        if isinstance(g, Batch) and hasattr(g, "batch") and g.batch is not None:
            # edge case, if we do squeeze() directly, we will get torch.size([]) and it will crash in the training.
            if atomic_energies.shape == (1, 1):
                atomic_energies = atomic_energies.squeeze(-1)
            else:
                atomic_energies = atomic_energies.squeeze()
            # Batch case: Use scatter_add with batch tensor
            return scatter_add(atomic_energies, batch, num_graphs)
        # Single graph case: Sum all energies (equivalent to scatter_add with all nodes in one graph)
        return torch.sum(atomic_energies, dim=0, keepdim=True).squeeze()

    def predict_structure(
        self,
        structure,
        state_feats: torch.Tensor | None = None,
        graph_converter: GraphConverter | None = None,
    ):
        """Convenience method to directly predict property from structure.

        Args:
            structure: An input crystal/molecule.
            state_feats (torch.tensor): Graph attributes
            graph_converter: Object that implements a get_graph_from_structure.

        Returns:
            output (torch.tensor): output property
        """
        if graph_converter is None:
            from matgl.ext._pymatgen_pyg import Structure2Graph

            graph_converter = Structure2Graph(element_types=self.element_types, cutoff=self.cutoff)  # type: ignore
        g, lat, state_feats_default = graph_converter.get_graph(structure)
        g.pbc_offshift = torch.matmul(g.pbc_offset, lat[0])
        g.pos = g.frac_coords @ lat[0]
        if state_feats is None:
            state_feats = torch.tensor(state_feats_default)
        return self(g=g, state_attr=state_feats).detach()
