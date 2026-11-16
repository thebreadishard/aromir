"""The pseudo-inverse of the Wilson B: its rank is known, so the count decides, not a cutoff."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from aromir import geometry, vibrations
from aromir.errors import InternalCoordinatesError
from aromir.molecule import Geometry
from aromir.network import primitives
from aromir.network.primitives import PrimitiveSettings
from aromir.vibrations import N_RIGID_MOTIONS

if TYPE_CHECKING:
    from aromir.molecule import FloatArray

FIXTURE = Path(__file__).parent / "data" / "network" / "wilson_b_biphenyl_cn.npz"
PIN_TOLERANCE = 1e-10
"""The rank-63 B⁺ of the fixture's matrix on numpy 2.5.4 for Linux differs from the pinned
one by 1.4e-13; with 62 values kept the difference is 2.9, with 64 it is 1e13."""

RAW = PrimitiveSettings()
BENT_LINEAR_GROUPS = [
    "C(#Cc1ccccc1)c1ccccc1",
    "Cc1ccccc1C#Cc1ccccc1",
    "C=C=Cc1ccccc1",
    "C(#Cc1ccccn1)c1ccccc1",
    "O=C=Nc1ccccc1",
]
"""Start geometries in which a linear group is bent by 1e-5 to 1e-2 rad: geomeTRIC's B of
them holds a rotation at up to 1e-4 of its largest singular value."""


def _fixture() -> tuple[FloatArray, FloatArray]:
    """Return the B of 4-cyanobiphenyl as a Linux build made it, and the bounded B⁺ of it."""
    with np.load(FIXTURE) as arrays:
        wilson_b = np.array(arrays["wilson_b"], float)
        return wilson_b, np.array(arrays["pseudo_inverse_windows"], float)


def _benzene_b() -> FloatArray:
    return primitives.build(geometry.start_from_smiles("c1ccccc1"), RAW).wilson_b


def test_the_force_constants_of_a_twisted_biphenyl_do_not_depend_on_the_lapack_build() -> None:
    # With np.linalg.pinv at its default cutoff this fails on numpy 2.5.4's Linux wheels
    # (OpenBLAS 0.3.34, SkylakeX kernel): a null singular value comes back as 7e-15,
    # above the cutoff of 3.9e-15, B⁺ reaches 1e13 and these force constants 1e12.
    wilson_b, pinned = _fixture()
    internal = np.random.default_rng(0).normal(size=(len(wilson_b), len(wilson_b)))
    hessian = wilson_b.T @ (internal + internal.T) @ wilson_b
    force_constants = primitives.internal_force_constants(hessian, wilson_b)
    np.testing.assert_allclose(force_constants, pinned.T @ hessian @ pinned, atol=1e-7)


def test_the_pseudo_inverse_is_the_one_a_build_without_the_residue_returns() -> None:
    wilson_b, pinned = _fixture()
    pseudo_inverse = primitives.pseudo_inverse(wilson_b)
    np.testing.assert_allclose(pseudo_inverse, pinned, atol=PIN_TOLERANCE)
    assert float(np.abs(pseudo_inverse).max()) == pytest.approx(3.296522, abs=1e-6)
    np.testing.assert_allclose(wilson_b @ pseudo_inverse @ wilson_b, wilson_b, atol=1e-10)
    rank = wilson_b.shape[1] - N_RIGID_MOTIONS
    assert float(np.trace(pseudo_inverse @ wilson_b)) == pytest.approx(rank)


def test_a_null_value_above_numpys_cutoff_is_not_inverted() -> None:
    wilson_b = _benzene_b()
    rank = wilson_b.shape[1] - N_RIGID_MOTIONS
    left, _, right = np.linalg.svd(wilson_b, full_matrices=False)
    residue = 1e-9
    with_residue = wilson_b + residue * np.outer(left[:, rank], right[rank])
    assert float(np.abs(np.linalg.pinv(with_residue)).max()) > 0.01 / residue
    np.testing.assert_allclose(
        primitives.pseudo_inverse(with_residue), primitives.pseudo_inverse(wilson_b), atol=1e-10
    )


@pytest.mark.parametrize(
    "smiles", ["c1ccccc1", "Oc1ccccc1", "N#Cc1ccccc1", "c1ccc2ccccc2c1", "Cc1ccccc1Cl"]
)
def test_nothing_changes_where_a_cutoff_inside_the_gap_was_right(smiles: str) -> None:
    wilson_b = primitives.build(geometry.start_from_smiles(smiles), RAW).wilson_b
    np.testing.assert_allclose(
        primitives.pseudo_inverse(wilson_b), np.linalg.pinv(wilson_b, rcond=1e-6), atol=1e-10
    )


def _without_rigid_motions(wilson_b: FloatArray, start: Geometry) -> FloatArray:
    """Return B with its rows made orthogonal to the translations and rotations."""
    weights = vibrations.mass_weights(start.masses_amu)
    rigid, _ = np.linalg.qr(vibrations.rigid_basis(start) / weights[:, None])
    return np.asarray(wilson_b - (wilson_b @ rigid) @ rigid.T, dtype=float)


def _assert_the_rotation_is_dropped(start: Geometry) -> None:
    wilson_b = primitives.build(start, RAW).wilson_b
    pseudo_inverse = primitives.pseudo_inverse(wilson_b)
    clean = primitives.pseudo_inverse(_without_rigid_motions(wilson_b, start))
    largest = float(np.abs(clean).max())
    assert largest < 20.0
    assert float(np.abs(pseudo_inverse - clean).max()) < 0.01 * largest


@pytest.mark.parametrize("smiles", BENT_LINEAR_GROUPS)
def test_the_rotation_in_a_bent_linear_group_is_not_inverted(smiles: str) -> None:
    _assert_the_rotation_is_dropped(geometry.start_from_smiles(smiles))


def test_a_distorted_diphenylacetylene_is_not_refused() -> None:
    # 0.01 bohr of noise bends the C–C≡C–C axis: a null value of 2e-5 of the largest.
    start = geometry.start_from_smiles("C(#Cc1ccccc1)c1ccccc1")
    noise = np.random.default_rng(4).normal(scale=0.01, size=start.coords_bohr.shape)
    _assert_the_rotation_is_dropped(Geometry(start.symbols, start.coords_bohr + noise))


def test_a_linear_molecule_is_refused() -> None:
    acetylene = Geometry(
        ("H", "C", "C", "H"),
        np.array([[0.0, 0.0, -3.14], [0.0, 0.0, -1.13], [0.0, 0.0, 1.13], [0.0, 0.0, 3.14]]),
    )
    wilson_b = primitives.build(acetylene, RAW).wilson_b
    with pytest.raises(InternalCoordinatesError, match="more than 3N − 6"):
        primitives.pseudo_inverse(wilson_b)


def test_primitives_that_miss_an_internal_direction_are_refused() -> None:
    wilson_b = _benzene_b()
    rank = wilson_b.shape[1] - N_RIGID_MOTIONS
    left, values, right = np.linalg.svd(wilson_b, full_matrices=False)
    one_short = wilson_b - values[rank - 1] * np.outer(left[:, rank - 1], right[rank - 1])
    with pytest.raises(InternalCoordinatesError, match="fewer than 3N − 6"):
        primitives.pseudo_inverse(one_short)
    with pytest.raises(InternalCoordinatesError, match="fewer than 3N − 6"):
        primitives.pseudo_inverse(wilson_b[: rank - 1])


@pytest.mark.parametrize("fill", [0.0, np.nan])
def test_a_matrix_that_is_no_b_is_refused(fill: float) -> None:
    with pytest.raises(InternalCoordinatesError, match="zero or not finite"):
        primitives.pseudo_inverse(np.full((40, 36), fill))
