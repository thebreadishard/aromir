"""Everything the network reads of one molecule, built from its geometry and Hessian."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict

from aromir.network import graph, primitives
from aromir.network import pattern as pair_pattern
from aromir.network.pattern import Pattern
from aromir.network.primitives import PrimitiveSettings
from aromir.vibrations import project_rigid_motions

if TYPE_CHECKING:
    from torch import Tensor

    from aromir.molecule import FloatArray, Geometry


class InputSettings(BaseModel):
    """How a network's inputs are built; fixed by the recipe its weights were trained with."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    pattern: Pattern = "f"
    primitives: PrimitiveSettings = PrimitiveSettings()
    parity: bool = False
    """Whether the primitives' parities are needed (a network with the parity-odd factor)."""


@dataclass(frozen=True, eq=False)
class NetworkInputs:
    """One molecule as tensors: N atoms, K primitives, P pattern pairs.

    Attributes:
        atomic_numbers: (N,) long.
        positions: (N, 3) in bohr.
        hessian: The low-level Hessian with rigid motions projected out, (3N, 3N).
        hinge_classes: (N,) long, indices into ``graph.HINGE_CLASSES``.
        wilson_b: (K, 3N).
        force_constants: The low level's internal force constants, (K, K).
        pairs: (P, 2) long, primitive indices i ≤ j.
        pair_classes: (P,) long.
        pair_features: (P, F).
        pool: (K, N); row k averages over the atoms of primitive k.
        odd: (K,) bool, the parity-odd primitives; empty unless the settings ask for parity.
        sines: (K,) sin(value) of the odd primitives; empty likewise.
    """

    atomic_numbers: Tensor
    positions: Tensor
    hessian: Tensor
    hinge_classes: Tensor
    wilson_b: Tensor
    force_constants: Tensor
    pairs: Tensor
    pair_classes: Tensor
    pair_features: Tensor
    pool: Tensor
    odd: Tensor
    sines: Tensor


def _float(array: FloatArray) -> Tensor:
    return torch.tensor(np.asarray(array), dtype=torch.float32)


def _pool_matrix(built: primitives.Primitives) -> Tensor:
    """Return the (K, N) matrix that averages atom features over each primitive's atoms."""
    pool = torch.zeros(len(built.internals), built.built_on.n_atoms, dtype=torch.float32)
    for index in range(len(built.internals)):
        atoms = {int(built.order[atom]) for atom in built.atoms(index)}
        for atom in atoms:
            pool[index, atom] = 1.0 / len(atoms)
    return pool


def build(geometry: Geometry, hessian: FloatArray, settings: InputSettings) -> NetworkInputs:
    """Build the network's inputs for a molecule.

    Args:
        geometry: The optimised geometry.
        hessian: The low-level Cartesian Hessian at that geometry, hartree/bohr².
        settings: The input settings of the network that will read them.
    """
    projected = project_rigid_motions(hessian, geometry)
    built = primitives.build(geometry, settings.primitives)
    force_constants = primitives.internal_force_constants(projected, built.wilson_b)
    pairs = pair_pattern.build(built, force_constants, settings.pattern, settings.primitives)
    if settings.parity:
        odd, sines = primitives.parities(built)
    else:
        odd, sines = np.zeros(0, dtype=bool), np.zeros(0)
    return NetworkInputs(
        atomic_numbers=torch.as_tensor(geometry.atomic_numbers, dtype=torch.long),
        positions=_float(geometry.coords_bohr),
        hessian=_float(projected),
        hinge_classes=torch.as_tensor(graph.hinge_classes(geometry), dtype=torch.long),
        wilson_b=_float(built.wilson_b),
        force_constants=_float(force_constants),
        pairs=torch.as_tensor(pairs.pairs, dtype=torch.long),
        pair_classes=torch.as_tensor(pairs.classes, dtype=torch.long),
        pair_features=torch.as_tensor(pairs.features, dtype=torch.float32),
        pool=_pool_matrix(built),
        odd=torch.as_tensor(odd),
        sines=_float(sines),
    )
