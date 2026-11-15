"""The model's guarantees: a symmetric ΔH that ignores rigid motions and turns with the frame."""

from __future__ import annotations

import numpy as np
import pytest

from aromir import geometry
from aromir.molecule import FloatArray, Geometry
from aromir.network import inputs, weights
from aromir.network.model import Architecture
from aromir.network.primitives import PrimitiveSettings
from aromir.vibrations import rigid_basis

from support import random_hessian, random_network, rotation_matrix

SMALL = Architecture(n_features=16, n_blocks=2, hidden=24)
ARCHITECTURES = {
    "pair_features": SMALL,
    "mean_pooling": SMALL.model_copy(update={"aggregation": "mean"}),
    "plain": SMALL.model_copy(update={"n_pair_features": 0, "sqm_scale": False}),
    "parity_odd": SMALL.model_copy(update={"n_pair_features": 0, "parity_odd": True}),
}
TOLERANCE = 1e-4
"""Relative to the largest element of ΔH; the network computes in single precision."""


def _correction(
    name: str, molecule: Geometry, hessian: FloatArray, primitives: PrimitiveSettings | None = None
) -> FloatArray:
    architecture = ARCHITECTURES[name]
    settings = inputs.InputSettings(
        primitives=primitives or PrimitiveSettings(), parity=architecture.parity_odd
    )
    network = random_network(architecture, seed=0)
    return weights.correction(network, inputs.build(molecule, hessian, settings))


@pytest.fixture(scope="module")
def biphenyl() -> Geometry:
    return geometry.start_from_smiles("c1ccc(cc1)-c1ccccc1")


@pytest.mark.parametrize("name", list(ARCHITECTURES))
def test_the_correction_is_symmetric_and_free_of_rigid_motions(
    name: str, biphenyl: Geometry
) -> None:
    delta_h = _correction(name, biphenyl, random_hessian(biphenyl.n_atoms, seed=1))
    scale = np.abs(delta_h).max()
    assert scale > 0
    assert np.abs(delta_h - delta_h.T).max() <= TOLERANCE * scale
    weights_ = np.sqrt(np.repeat(biphenyl.masses_amu, 3))
    rigid_displacements = rigid_basis(biphenyl) / weights_[:, None]
    leak = np.abs(delta_h @ rigid_displacements).max() / np.abs(rigid_displacements).max()
    assert leak <= TOLERANCE * scale


@pytest.mark.parametrize("name", list(ARCHITECTURES))
def test_the_correction_turns_with_the_molecule(name: str, biphenyl: Geometry) -> None:
    hessian = random_hessian(biphenyl.n_atoms, seed=1)
    rotation = rotation_matrix(seed=4)
    turn = np.kron(np.eye(biphenyl.n_atoms), rotation)
    turned = Geometry(
        biphenyl.symbols, biphenyl.coords_bohr @ rotation.T + np.array([3.0, -1.0, 2.0])
    )
    here = _correction(name, biphenyl, hessian)
    there = _correction(name, turned, turn @ hessian @ turn.T)
    assert np.abs(there - turn @ here @ turn.T).max() <= TOLERANCE * np.abs(here).max()


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_the_order_free_settings_give_one_answer_for_every_atom_order(seed: int) -> None:
    # 2-chlorophenol's bond graph has no symmetry. Atoms the graph cannot tell apart are
    # ordered by the input, which is harmless for a real Hessian (it shares the molecule's
    # symmetry) but not for the random matrix that stands in for one here.
    molecule = geometry.start_from_smiles("Oc1ccccc1Cl")
    order_free = PrimitiveSettings(angle_choice="canonical", dihedral_encoding="cos")
    hessian = random_hessian(molecule.n_atoms, seed=2)
    permutation = np.random.default_rng(seed).permutation(molecule.n_atoms)
    columns = (3 * permutation[:, None] + np.arange(3)).ravel()
    here = _correction("pair_features", molecule, hessian, order_free)
    there = _correction(
        "pair_features",
        molecule.reordered(permutation.tolist()),
        hessian[np.ix_(columns, columns)],
        order_free,
    )
    assert np.abs(there - here[np.ix_(columns, columns)]).max() <= TOLERANCE * np.abs(here).max()


def test_the_low_level_hessian_reaches_the_correction(biphenyl: Geometry) -> None:
    first = _correction("pair_features", biphenyl, random_hessian(biphenyl.n_atoms, seed=1))
    second = _correction("pair_features", biphenyl, random_hessian(biphenyl.n_atoms, seed=2))
    assert np.abs(first - second).max() > 0.01 * np.abs(first).max()


def test_the_parity_odd_factor_does_not_combine_with_pair_features() -> None:
    with pytest.raises(ValueError, match="parity_odd with pair features"):
        Architecture(parity_odd=True)


def test_an_unknown_architecture_setting_is_refused() -> None:
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        Architecture.model_validate({"tensor_input": True})
    with pytest.raises(ValueError, match="aggregation"):
        Architecture.model_validate({"aggregation": "max"})
