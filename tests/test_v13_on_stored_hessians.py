"""Release v1.3 on stored Hessians: what one molecule's answer does and does not depend on.

A molecule turned in space, its mirror image and another listing of its atoms are the same
molecule. Release v1.3 does not treat them alike in every case: its internal coordinates
measure a linear angle against a fixed axis, take a dihedral with its sign and are chosen by
atom number. Each such case is a test that is expected to fail, strictly: the day a release
answers alike, the test passes, the mark must go, and ``invariance.json`` is measured anew.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from aromir import admission, evaluation, families, geometry, release
from aromir import vibrations as vib
from aromir.constants import ANGSTROM_PER_BOHR
from aromir.molecule import Geometry

from support import rotation_matrix

if TYPE_CHECKING:
    from aromir.molecule import FloatArray

DATA = Path(__file__).parent / "data" / "release"
SAME_CM = 1e-3
"""Two answers to one molecule agree when no band differs by more: single-precision noise
in ΔH moves a band by 1e-5 cm⁻¹."""
N_LISTINGS = 5
BENZENE_BAND_ERROR_CM = {"ring-ip": 1.2020, "CH-stretch": 0.9184, "CH-oop": 1.9816, "other": 0.4655}
BENZENE_FIRST_ORDER_CM = {
    "ring-ip": (0.9594, 2.8141, 1.3519),
    "CH-stretch": (0.6741, 1.4202, 0.7597),
    "CH-oop": (2.9201, 2.1665, 1.4810),
    "other": (1.1005, 1.2143, 0.8034),
}
"""What ``aromir.evaluation`` measured for benzene, a molecule of hold-out (a), with release
v1.3 on 16 November 2026: the band error of the seeds' mean and each seed's first-order
error, rms per family. Its bands: 13 ring-ip, 6 CH-stretch, 6 CH-oop, 5 other."""


@pytest.fixture(scope="module")
def bundled() -> release.Release:
    return release.load(release.bundled())


def _stored(name: str) -> tuple[Geometry, FloatArray]:
    with np.load(DATA / f"stored_{name}.npz") as arrays:
        symbols = tuple(str(symbol) for symbol in arrays["symbols"])
        return Geometry(symbols, np.array(arrays["coords_bohr"])), np.array(arrays["hessian"])


def _corrected(loaded: release.Release, molecule: Geometry, hessian: FloatArray) -> FloatArray:
    answer = loaded.corrections(molecule, hessian).mean
    return np.asarray(vib.vibrational_modes(hessian + answer, molecule).wavenumbers)


def _moved(
    molecule: Geometry, hessian: FloatArray, matrix: FloatArray
) -> tuple[Geometry, FloatArray]:
    """Return the molecule and its Hessian under an orthogonal map of space."""
    blocks = np.kron(np.eye(molecule.n_atoms), matrix)
    return Geometry(molecule.symbols, molecule.coords_bohr @ matrix.T), blocks @ hessian @ blocks.T


def _largest_change(loaded: release.Release, name: str, matrices: list[FloatArray]) -> float:
    molecule, hessian = _stored(name)
    as_stored = _corrected(loaded, molecule, hessian)
    return max(
        float(np.abs(_corrected(loaded, *_moved(molecule, hessian, matrix)) - as_stored).max())
        for matrix in matrices
    )


ROTATIONS = [rotation_matrix(seed) for seed in range(3)]
MIRROR = [np.diag([1.0, 1.0, -1.0])]


def test_a_molecule_without_a_linear_group_turns_with_its_spectrum(
    bundled: release.Release,
) -> None:
    assert _largest_change(bundled, "biphenyl", ROTATIONS) < SAME_CM


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="a linear angle is measured against a fixed axis: turned benzonitrile gets bands "
    "up to 3.7 cm⁻¹ away here, 5.34 over the molecules of invariance.json (the audit's "
    "W1); frame-free linear bends need a retrain",
)
def test_a_nitrile_turns_with_its_spectrum(bundled: release.Release) -> None:
    assert _largest_change(bundled, "benzonitrile", ROTATIONS) < SAME_CM


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="a dihedral enters with its sign: the mirror image of twisted biphenyl gets bands "
    "up to 2.6 cm⁻¹ away (the audit's W3); the cos encoding needs a retrain",
)
def test_a_mirror_image_has_the_same_spectrum(bundled: release.Release) -> None:
    assert _largest_change(bundled, "biphenyl", MIRROR) < SAME_CM


def _as_listed(molecule: Geometry, listing: list[int], path: Path) -> list[int]:
    """Return which atoms of the stored molecule aromir puts where, read from an xyz listing."""
    lines = [str(molecule.n_atoms), "one geometry, listed in another order"]
    for atom in listing:
        x, y, z = molecule.coords_bohr[atom] * ANGSTROM_PER_BOHR
        lines.append(f"{molecule.symbols[atom]} {x:.10f} {y:.10f} {z:.10f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    order = geometry.canonical_order(admission.parse_xyz(path))
    return [listing[position] for position in order]


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="atoms the bond graph cannot tell apart keep the file's order, and the primitives "
    "are chosen by atom number: listings of one biphenyl geometry get bands up to 3.1 cm⁻¹ "
    "apart (the audit's W2); the canonical angle choice needs a retrain",
)
def test_every_listing_of_one_geometry_has_the_same_spectrum(
    bundled: release.Release, tmp_path: Path
) -> None:
    molecule, hessian = _stored("biphenyl")
    rng = np.random.default_rng(7)
    spectra = []
    for number in range(N_LISTINGS):
        listing = [int(atom) for atom in rng.permutation(molecule.n_atoms)]
        order = _as_listed(molecule, listing, tmp_path / f"{number}.xyz")
        index = (3 * np.array(order)[:, None] + np.arange(3)).ravel()
        ordered = molecule.reordered(order)
        spectra.append(_corrected(bundled, ordered, hessian[np.ix_(index, index)]))
    assert float(np.abs(np.array(spectra) - spectra[0]).max()) < SAME_CM


def test_a_holdout_molecule_has_the_errors_the_evaluation_measured(
    bundled: release.Release,
) -> None:
    errors = evaluation.molecule_errors(evaluation.read(DATA / "holdout_a_benzene.npz"), bundled)
    band_family = np.array(errors.band_family)
    first_order_family = np.array(errors.first_order_family)
    counts = {family: int((band_family == family).sum()) for family in families.FAMILIES}
    assert counts == {"ring-ip": 13, "CH-stretch": 6, "CH-oop": 6, "other": 5}
    for family in families.FAMILIES:
        band = errors.band[band_family == family]
        assert float(np.sqrt(np.mean(band**2))) == pytest.approx(
            BENZENE_BAND_ERROR_CM[family], abs=0.01
        )
        per_seed = np.sqrt(
            np.mean(errors.first_order[:, first_order_family == family] ** 2, axis=1)
        )
        assert per_seed.tolist() == pytest.approx(BENZENE_FIRST_ORDER_CM[family], abs=0.01)
