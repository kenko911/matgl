from __future__ import annotations

import os

import numpy as np
import pytest
import torch
from pymatgen.core import Lattice, Structure

import matgl

BACKEND = matgl.config.BACKEND

if BACKEND == "DGL":
    from matgl.graph._compute_dgl import compute_pair_vector_and_distance
elif BACKEND == "PYG":
    from matgl.graph._compute_pyg import compute_pair_vector_and_distance  # type: ignore[assignment]
else:
    pytest.skip(f"Unsupported backend: {BACKEND}", allow_module_level=True)

from matgl.models import MEGNet  # noqa: E402


def _device_of(graph):
    if BACKEND == "DGL":
        return graph.device
    return graph.pos.device


def _prep_graph(graph, structure):
    """Attach pos / pbc_offshift / bond_dist for the active backend."""
    lat = torch.tensor(np.array([structure.lattice.matrix]), dtype=matgl.float_th, device=_device_of(graph))
    if BACKEND == "DGL":
        graph.edata["pbc_offshift"] = torch.matmul(graph.edata["pbc_offset"], lat[0])
        graph.ndata["pos"] = graph.ndata["frac_coords"] @ lat[0]
        _, bond_dist = compute_pair_vector_and_distance(graph)
        graph.edata["bond_dist"] = bond_dist
    else:
        graph.pbc_offshift = torch.matmul(graph.pbc_offset, lat[0])
        graph.pos = graph.frac_coords @ lat[0]
        _, bond_dist = compute_pair_vector_and_distance(graph.pos, graph.edge_index, graph.pbc_offshift)
        graph.bond_dist = bond_dist
    return graph


def _make_megnet(**overrides):
    base = {
        "dim_node_embedding": 16,
        "dim_edge_embedding": 100,
        "dim_state_embedding": 2,
        "nblocks": 3,
        "include_state": True,
        "hidden_layer_sizes_input": (64, 32),
        "hidden_layer_sizes_conv": (64, 64, 32),
        "activation_type": "swish",
        "nlayers_set2set": 4,
        "niters_set2set": 3,
        "hidden_layer_sizes_output": (32, 16),
        "is_classification": True,
    }
    base.update(overrides)
    return MEGNet(**base)


def test_megnet(graph_MoS):
    structure, graph, state = graph_MoS
    graph = _prep_graph(graph, structure)
    state_t = torch.tensor(np.array(state), dtype=matgl.float_th)
    output = None
    for act in ["tanh", "sigmoid", "softplus2", "softexp", "swish"]:
        model = _make_megnet(activation_type=act)
        output = model(g=graph, state_attr=state_t)
    with pytest.raises(ValueError, match="Invalid activation type"):
        _ = MEGNet(activation_type="whatever")
    assert torch.numel(output) == 1


def test_megnet_isolated_atom():
    structure = Structure(Lattice.cubic(10.0), ["Mo"], [[0.0, 0, 0]])
    model = _make_megnet(dropout=0.1)
    output = model.predict_structure(structure)
    assert torch.numel(output) == 1


def test_save_load(tmp_path):
    model = _make_megnet(dropout=0.1)
    cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        model.save(".", metadata={"description": "forme model"})
        MEGNet.load(".")
    finally:
        os.chdir(cwd)
