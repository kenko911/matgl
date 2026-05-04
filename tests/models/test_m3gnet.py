from __future__ import annotations

import os

import numpy as np
import pytest
import torch

import matgl

BACKEND = matgl.config.BACKEND

if BACKEND == "DGL":
    from matgl.graph._compute_dgl import compute_pair_vector_and_distance
elif BACKEND == "PYG":
    from matgl.graph._compute_pyg import compute_pair_vector_and_distance  # type: ignore[assignment]
else:
    pytest.skip(f"Unsupported backend: {BACKEND}", allow_module_level=True)

from matgl.models import M3GNet  # noqa: E402


def _device_of(graph):
    if BACKEND == "DGL":
        return graph.device
    return graph.pos.device


def _prep_graph(graph, structure):
    """Attach pos / pbc_offshift / bond_{vec,dist} for the active backend."""
    lat = torch.tensor(np.array([structure.lattice.matrix]), dtype=matgl.float_th, device=_device_of(graph))
    if BACKEND == "DGL":
        graph.edata["pbc_offshift"] = torch.matmul(graph.edata["pbc_offset"], lat[0])
        graph.ndata["pos"] = graph.ndata["frac_coords"] @ lat[0]
        bond_vec, bond_dist = compute_pair_vector_and_distance(graph)
        graph.edata["bond_vec"] = bond_vec
        graph.edata["bond_dist"] = bond_dist
    else:
        graph.pbc_offshift = torch.matmul(graph.pbc_offset, lat[0])
        graph.pos = graph.frac_coords @ lat[0]
        bond_vec, bond_dist = compute_pair_vector_and_distance(graph.pos, graph.edge_index, graph.pbc_offshift)
        graph.bond_vec = bond_vec
        graph.bond_dist = bond_dist
    return graph


def test_model(graph_MoS):
    structure, graph, _ = graph_MoS
    graph = _prep_graph(graph, structure)
    for act in ["swish", "tanh", "sigmoid", "softplus2", "softexp"]:
        model = M3GNet(is_intensive=False, activation_type=act)
        output = model(g=graph)
        assert torch.numel(output) == 1
    model.save(".")
    M3GNet.load(".")
    os.remove("model.pt")
    os.remove("model.json")
    os.remove("state.pt")


def test_exceptions():
    with pytest.raises(ValueError, match="Invalid activation type"):
        _ = M3GNet(element_types=None, is_intensive=False, activation_type="whatever")
    with pytest.raises(ValueError, match=r"Classification task cannot be extensive."):
        _ = M3GNet(element_types=["Mo", "S"], is_intensive=False, task_type="classification")


def test_model_intensive(graph_MoS):
    structure, graph, _ = graph_MoS
    graph = _prep_graph(graph, structure)
    model = M3GNet(element_types=["Mo", "S"], is_intensive=True)
    output = model(g=graph)
    assert torch.numel(output) == 1


def test_model_intensive_reduce_atom(graph_MoS):
    structure, graph, _ = graph_MoS
    graph = _prep_graph(graph, structure)
    model = M3GNet(element_types=["Mo", "S"], is_intensive=True, readout_type="reduce_atom")
    output = model(g=graph)
    assert torch.numel(output) == 1


def test_model_intensive_with_classification(graph_MoS):
    structure, graph, _ = graph_MoS
    graph = _prep_graph(graph, structure)
    model = M3GNet(element_types=["Mo", "S"], is_intensive=True, task_type="classification")
    output = model(g=graph)
    assert torch.numel(output) == 1


def test_model_intensive_set2set_classification(graph_MoS):
    structure, graph, _ = graph_MoS
    graph = _prep_graph(graph, structure)
    kwargs = {
        "element_types": ["Mo", "S"],
        "is_intensive": True,
        "task_type": "classification",
        "readout_type": "set2set",
    }
    if BACKEND == "PYG":
        kwargs["niters_set2set"] = 2
        kwargs["nlayers_set2set"] = 1
    model = M3GNet(**kwargs)
    output = model(g=graph)
    assert torch.numel(output) == 1


def test_predict_structure(graph_MoS):
    structure, _, _ = graph_MoS
    if BACKEND == "DGL":
        models = [M3GNet(is_intensive=False), M3GNet(element_types=["Mo", "S"], is_intensive=True)]
    else:
        models = [M3GNet(element_types=["Mo", "S"], is_intensive=False)]
    for model in models:
        output_final = model.predict_structure(structure)
        assert torch.numel(output_final) == 1


def test_save_load(tmp_path):
    model = M3GNet(element_types=("Mo", "S"), is_intensive=True)
    cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        model.save(".")
        M3GNet.load(".")
    finally:
        os.chdir(cwd)


def test_featurize_structure(graph_MoS):
    if BACKEND != "DGL":
        pytest.skip("predict_structure(return_features=True) feature shape contract is only validated for DGL.")
    structure, _, _ = graph_MoS
    model_intensive = M3GNet(element_types=["Mo", "S"], is_intensive=True)
    model_extensive = M3GNet(is_intensive=False)
    for model in [model_extensive, model_intensive]:
        with pytest.raises(ValueError, match="Invalid output_layers"):
            model.predict_structure(structure, output_layers=["whatever"], return_features=True)
        features = model.predict_structure(structure, return_features=True)
        assert torch.numel(features["bond_expansion"]) == 252
        assert torch.numel(features["three_body_basis"]) == 3276
        for output_layer in ["embedding", "gc_1", "gc_2", "gc_3"]:
            assert torch.numel(features[output_layer]["node_feat"]) == 128
            assert torch.numel(features[output_layer]["edge_feat"]) == 1792
            assert features[output_layer]["state_feat"] is None
        if model.is_intensive:
            assert torch.numel(features["readout"]) == 64
        else:
            assert torch.numel(features["readout"]) == 2
        assert torch.numel(features["final"]) == 1
        assert list(model.predict_structure(structure, output_layers=["gc_1"], return_features=True).keys()) == ["gc_1"]
