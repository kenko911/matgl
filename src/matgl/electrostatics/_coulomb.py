from __future__ import annotations

import math

import dgl
import torch
from matgl.config import COULOMB_CONSTANT
from matgl.layers import MLP
from torch import nn


class Coulomb(nn.Module):
    """Class for calculating electrostatic energy using Coulomb's law."""

    def __init__(self, numerical_noise: float = 1.0e8):
        super().__init__()
        self.numerical_noise = numerical_noise

    def forward(self, g: dgl.DGLGraph, A_matrix: torch.Tensor):
        elec_energy = 0.5 * torch.einsum("i,ij,j", g.ndata["charge"], A_matrix, g.ndata["charge"])

        return elec_energy


class CoulombQeq(nn.Module):
    """Class for calculating Qeq charges in non-periodic systems."""

    def __init__(
        self,
        element_types: tuple,
        dim_node_feats: int,
        units: int,
        is_hardness_envs: bool = False,
        is_sigma_train: bool = False,
        pbc: bool = False,
        numerical_noise: float = 1.0e8,
        energy_scale: float = 1.0,  # std of energy in dataset, eV unit
    ):
        super().__init__()

        self.pbc = pbc
        if self.pbc:
            raise NotImplementedError("Electrostatic correction for periodic systems is not implemented yet!")

        self.energy_scale = energy_scale
        self.scaled_coulomb_factor = COULOMB_CONSTANT / self.energy_scale  # dimensionless
        self.element_types = element_types

        # sigma: species_index (0-indexed) -> covalent radius
        if is_hardness_envs is False:
            hardness = torch.ones(len(element_types))
            self.hardness_readout = torch.nn.Parameter(data=hardness)
        else:
            self.hardness_readout = MLP(
                dim=[dim_node_feats, units, units, 1], activation=nn.Softplus(), activate_last=True
            )

        if is_sigma_train:
            sigma = torch.ones(len(element_types))
            self.sigma = torch.nn.Parameter(data=sigma)

        self.chi_readout = MLP(dims=[dim_node_feats, units, units, 1], activation=nn.Tanhshrink(), activate_last=True)

        self.numerical_noise = numerical_noise
        self.is_hardness_envs = is_hardness_envs

    def forward(self, g_batched: dgl.DGLGraph, total_charge: torch.Tensor) -> torch.Tensor:
        for i, g in enumerate(dgl.unbatch(g_batched)):
            n_atoms = g.num_nodes()
            chi = self.chi_readout(g.data["node_feats"])  # (num_nodes, 1)
            if self.is_hardness_envs:
                hardness = self.hardness_readout(g.ndata["node_feats"])
            else:
                hardness = self.hardness_readout[g.ndata["node_type"]]
            sigma = self.sigma[g.ndata["node_type"]].to(g.device)
            # square here to restrit hardness to be positive!
            #        hardness = torch.square(self.to_hardness[species_idx])  # (num_atoms, )

            # batch-wise pair indices of atoms
            diff = g.ndata["pos"].unsqueeze(1) - g.nata["pos"].unsqueeze(0)
            bond_dist = torch.norm(diff, dim=-1) + self.numerical_noise
            a_matrix = torch.zeros((n_atoms, n_atoms), device=g.device)
            gamma_ij = torch.sqrt(sigma.unsqueeze(1) ** 2 + sigma.unsqueeze(0) ** 2)
            a_matrix += COULOMB_CONSTANT * torch.erf(bond_dist / torch.sqrt(2.0) / gamma_ij) / bond_dist
            A_matrix = a_matrix + torch.diag(hardness + COULOMB_CONSTANT / (math.sqrt(math.pi) * gamma_ij))
            b_matrix = torch.ones((n_atoms + 1, n_atoms + 1), device=g.device)
            b_matrix[:n_atoms, :n_atoms] = A_matrix
            b_matrix[-1, -1] = 0.0
            q_tol = total_charge[i].to(g.device)
            rhs_vec = torch.vstack([-chi, q_tol])
            charges_and_lambda = torch.linalg.solve(torch.unsqueeze(b_matrix, dim=0), torch.unsqueeze(rhs_vec, dim=0))
            charges = torch.squeeze(charges_and_lambda)[:-1]
            g.ndata["charge"] = charges

        return charges, A_matrix
