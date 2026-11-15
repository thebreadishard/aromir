"""Helpers the tests share: synthetic Hessians, a seeded rotation, a random network."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from aromir.network.model import Architecture, DeltaHessianNetwork

if TYPE_CHECKING:
    from aromir.molecule import FloatArray, Geometry

Bonds = tuple[tuple[int, int, float], ...]

WATER_BONDS: Bonds = ((0, 1, 0.55), (0, 2, 0.55), (1, 2, 0.10))
"""Atom pairs and force constants (hartree/bohr²) of a valence force field for water."""

BENZENE_XYZ = """12
benzene, Å
C   1.3970   0.0000  0.0
C   0.6985   1.2098  0.0
C  -0.6985   1.2098  0.0
C  -1.3970   0.0000  0.0
C  -0.6985  -1.2098  0.0
C   0.6985  -1.2098  0.0
H   2.4810   0.0000  0.0
H   1.2405   2.1486  0.0
H  -1.2405   2.1486  0.0
H  -2.4810   0.0000  0.0
H  -1.2405  -2.1486  0.0
H   1.2405  -2.1486  0.0
"""
"""Benzene as an xyz file, in Å."""


def bond_force_field(geometry: Geometry, bonds: Bonds) -> FloatArray:
    """Return an exact Cartesian Hessian of harmonic springs along the given atom pairs.

    Such a Hessian is symmetric and invariant to translations and rotations at the
    geometry it is built for, which makes it a known answer for the vibrational code.
    """
    hessian = np.zeros((3 * geometry.n_atoms, 3 * geometry.n_atoms))
    for first, second, constant in bonds:
        direction = geometry.coords_bohr[second] - geometry.coords_bohr[first]
        direction = direction / np.linalg.norm(direction)
        block = constant * np.outer(direction, direction)
        for row, column, sign in (
            (first, first, 1),
            (second, second, 1),
            (first, second, -1),
            (second, first, -1),
        ):
            hessian[3 * row : 3 * row + 3, 3 * column : 3 * column + 3] += sign * block
    return hessian


def rotation_matrix(seed: int) -> FloatArray:
    """Return a proper rotation drawn from a seeded generator."""
    matrix, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
    if np.linalg.det(matrix) < 0:
        matrix[:, 0] = -matrix[:, 0]
    return np.asarray(matrix, dtype=float)


def random_hessian(n_atoms: int, seed: int) -> FloatArray:
    """Return a seeded symmetric matrix with a stiff diagonal, the size of a Hessian.

    The network's arithmetic does not care that it is not a real Hessian, and a test of
    that arithmetic then needs no quantum chemistry.
    """
    noise = np.random.default_rng(seed).normal(scale=0.02, size=(3 * n_atoms, 3 * n_atoms))
    return np.asarray(noise + noise.T + 0.5 * np.eye(3 * n_atoms), dtype=float)


def random_network(architecture: Architecture, seed: int) -> DeltaHessianNetwork:
    """Return a network with every parameter and buffer drawn, scales and spreads positive.

    Nothing is left at its zero initialisation, so every input reaches the output.
    """
    torch.manual_seed(seed)
    network = DeltaHessianNetwork(architecture)
    positive = ("class_scale", "force_constant_scale", "pair_feature_std")
    with torch.no_grad():
        for name, value in network.state_dict().items():
            if name.endswith("centres"):
                continue
            drawn = torch.randn_like(value) * 0.1
            value.copy_(drawn.abs() + 0.5 if name in positive else drawn)
    return network.eval()
