from __future__ import annotations

import sys

import matgl
import numpy as np
import torch

torch.set_printoptions(threshold=sys.maxsize)

ewaldSummationPrecision = 1e-8


def getOptimalEtaMatrix(lat):
    return 1 / torch.sqrt(2 * torch.tensor(np.pi, dtype=matgl.float_th)) * torch.det(lat) ** (1 / 3)


def getOptimalCutoffReal(eta, prec):
    r = torch.sqrt(2) * eta * torch.sqrt(-torch.log(prec))
    return r


def getOptimalCutoffRecip(eta, prec):
    r = torch.sqrt(2) / eta * torch.sqrt(-torch.log(prec))
    return r


def recLattice(lat):
    B = torch.inverse(lat) * (2 * torch.tensor(np.pi, dtype=matgl.float_th))
    return B


def getNcells(lat, cutoff):
    proj = torch.zeros(3)
    axb = torch.cross(lat[0], lat[1])
    axb = axb / torch.norm(axb)
    axc = torch.cross(lat[0], lat[2])
    axc = axc / torch.norm(axc)
    bxc = torch.cross(lat[1], lat[2])
    bxc = bxc / torch.norm(bxc)
    proj[0] = torch.dot(lat[0], bxc)
    proj[1] = torch.dot(lat[1], axc)
    proj[2] = torch.dot(lat[2], axb)
    n = torch.ceil(cutoff / torch.abs(proj))
    n = n.int()
    return n


def eemMatrixEwald(nat, ats, lat, sigma, hard):
    sigma = torch.tensor(sigma, dtype=matgl.float_th)
    hard = torch.tensor(hard, dtype=matgl.float_th)
    eta = getOptimalEtaMatrix(nat, lat)
    eta = torch.max(eta, 2 * torch.max(sigma))
    Areal = AmatrixReal(nat, ats, lat, sigma, hard, eta)
    Arecip = AmatrixRecip(nat, ats, lat, eta)
    Aself = AmatrixSelf(nat, eta)
    A = Areal + Arecip + Aself
    return A


def AmatrixSelf(nat, eta):
    A = torch.zeros((nat, nat), dtype=matgl.float_th)
    for i in range(nat):
        A[i, i] = -2 / (torch.sqrt(2 * torch.tensor(np.pi, dtype=matgl.float_th)) * eta)
    return A


def AmatrixRecip(nat, ats, lat, eta):
    reclat = recLattice(lat)
    V = torch.det(lat)
    print("V", V)
    A = torch.zeros((nat, nat), dtype=matgl.float_th)
    cutoff = getOptimalCutoffRecip(eta, ewaldSummationPrecision)
    n = getNcells(reclat, cutoff)
    print(n)
    for i in range(-n[0], n[0] + 1):
        for j in range(-n[1], n[1] + 1):
            for k in range(-n[2], n[2] + 1):
                if i != 0 or j != 0 or k != 0:
                    dlat = i * reclat[0] + j * reclat[1] + k * reclat[2]
                    r = torch.sum(dlat**2)
                    if r > cutoff**2:
                        continue
                    factor = torch.exp(-(eta**2) * r / 2) / r
                    for iat in range(nat):
                        kri = torch.sum(dlat * ats[iat])
                        for jat in range(iat, nat):
                            krj = torch.sum(dlat * ats[jat])
                            A[iat, jat] = A[iat, jat] + factor * (
                                torch.cos(kri) * torch.cos(krj) + torch.sin(kri) * torch.sin(krj)
                            )
    for iat in range(nat):
        for jat in range(iat + 1, nat):
            A[jat, iat] = A[iat, jat]
    A = A / V * 4 * torch.tensor(np.pi, dtype=matgl.float_th)
    return A


def AmatrixReal(nat, ats, lat, sigma, hard, eta):
    A = torch.zeros((nat, nat), dtype=matgl.float_th)
    invsqrt2eta = 1 / (torch.sqrt(2) * eta)
    cutoff = getOptimalCutoffReal(eta, ewaldSummationPrecision)
    n = getNcells(lat, cutoff)
    print(eta, cutoff, n)

    for iat in range(nat):
        for i in range(-n[0], n[0] + 1):
            for j in range(-n[1], n[1] + 1):
                for k in range(-n[2], n[2] + 1):
                    dlat = i * lat[0] + j * lat[1] + k * lat[2]
                    for jat in range(iat, nat):
                        if i != 0 or j != 0 or k != 0 or iat != jat:
                            d = ats[iat] - ats[jat] + dlat
                            d2 = torch.sum(d**2)
                            if d2 > cutoff**2:
                                continue

                            r = torch.sqrt(d2)
                            interf = torch.erfc(r * invsqrt2eta)

                            if sigma[0] > 0:
                                gamma = torch.sqrt(sigma[iat] ** 2 + sigma[jat] ** 2)
                                interf = interf - torch.erfc(r / (torch.sqrt(2) * gamma))

                            A[iat, jat] = A[iat, jat] + interf / r

        if sigma[0] > 0:
            A[iat, iat] = (
                A[iat, iat] + 1 / (sigma[iat] * torch.sqrt(torch.tensor(np.pi, dtype=matgl.float_th))) + hard[iat]
            )

    for iat in range(nat):
        for jat in range(iat + 1, nat):
            A[jat, iat] = A[iat, jat]

    return A
