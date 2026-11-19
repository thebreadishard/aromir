"""Releases v1.3 and v1.4 on stored Hessians: what one molecule's answer does not depend on.

A molecule turned in space, its mirror image and another listing of its atoms are the same
molecule. Neither release treats them alike in every case: their internal coordinates
measure a linear angle against a fixed axis, take a dihedral with its sign and are chosen by
atom number. Each such case is a test that is expected to fail, strictly: the day a release
answers alike, the test passes, the mark must go, and ``invariance.json`` is measured anew.
Beside each, the largest change the release gives is pinned, so a release that moves further
or less far is noticed too.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

import numpy as np
import pytest

from aromir import admission, evaluation, families, geometry, release
from aromir import vibrations as vib
from aromir.constants import ANGSTROM_PER_BOHR
from aromir.own_spread import moved

from support import RELEASE_DATA, rotation_matrix, stored_molecule

if TYPE_CHECKING:
    from collections.abc import Callable

    from _pytest.mark import ParameterSet

    from aromir.molecule import FloatArray, Geometry

VERSIONS = ("v1.3", "v1.4")
SAME_CM = 1e-3
"""Two answers to one molecule agree when no band differs by more: single-precision noise
in ΔH moves a band by 1e-5 cm⁻¹."""
N_LISTINGS = 5
LARGEST_CHANGE_CM = {
    "v1.3": {"turned nitrile": 3.7298, "mirrored biphenyl": 2.5996, "listed biphenyl": 3.0594},
    "v1.4": {"turned nitrile": 0.9177, "mirrored biphenyl": 2.6687, "listed biphenyl": 2.8521},
}
"""The largest band change of each case below, cm⁻¹, as each release gave it on 19 November
2026 (v1.3's as on 16 November)."""
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


@cache
def _release(version: str) -> release.Release:
    return release.load(release.bundled().parent / version)


def _corrected(loaded: release.Release, molecule: Geometry, hessian: FloatArray) -> FloatArray:
    answer = loaded.corrections(molecule, hessian).mean
    return np.asarray(vib.vibrational_modes(hessian + answer, molecule).wavenumbers)


def _largest_change(loaded: release.Release, name: str, matrices: list[FloatArray]) -> float:
    molecule, hessian = stored_molecule(name)
    as_stored = _corrected(loaded, molecule, hessian)
    return max(
        float(np.abs(_corrected(loaded, *moved(molecule, hessian, matrix)) - as_stored).max())
        for matrix in matrices
    )


ROTATIONS = [rotation_matrix(seed) for seed in range(3)]
MIRROR = [np.diag([1.0, 1.0, -1.0])]


def _as_listed(molecule: Geometry, listing: list[int], path: Path) -> list[int]:
    """Return which atoms of the stored molecule aromir puts where, read from an xyz listing."""
    lines = [str(molecule.n_atoms), "one geometry, listed in another order"]
    for atom in listing:
        x, y, z = molecule.coords_bohr[atom] * ANGSTROM_PER_BOHR
        lines.append(f"{molecule.symbols[atom]} {x:.10f} {y:.10f} {z:.10f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    order = geometry.canonical_order(admission.read(path).molecule)
    return [listing[position] for position in order]


def _largest_listing_change(loaded: release.Release) -> float:
    """Return the largest band difference between listings of one biphenyl geometry."""
    molecule, hessian = stored_molecule("biphenyl")
    rng = np.random.default_rng(7)
    spectra = []
    with TemporaryDirectory() as folder:
        for number in range(N_LISTINGS):
            listing = [int(atom) for atom in rng.permutation(molecule.n_atoms)]
            order = _as_listed(molecule, listing, Path(folder) / f"{number}.xyz")
            index = (3 * np.array(order)[:, None] + np.arange(3)).ravel()
            ordered = molecule.reordered(order)
            spectra.append(_corrected(loaded, ordered, hessian[np.ix_(index, index)]))
    return float(np.abs(np.array(spectra) - spectra[0]).max())


@cache
def _change(version: str, case: str) -> float:
    loaded = _release(version)
    if case == "turned nitrile":
        return _largest_change(loaded, "benzonitrile", ROTATIONS)
    if case == "mirrored biphenyl":
        return _largest_change(loaded, "biphenyl", MIRROR)
    return _largest_listing_change(loaded)


def _recorded(version: str) -> release.Invariance:
    """Return a release's invariance record, read without loading its networks."""
    path = release.bundled().parent / version / release.INVARIANCE_FILE
    return release.Invariance.model_validate_json(path.read_text(encoding="utf-8"))


def _differs(reason: Callable[[str], str]) -> list[ParameterSet]:
    """Return each version as a parameter expected to fail, with its own reason."""
    return [
        pytest.param(
            version,
            marks=pytest.mark.xfail(strict=True, raises=AssertionError, reason=reason(version)),
        )
        for version in VERSIONS
    ]


@pytest.mark.parametrize("version", VERSIONS)
def test_a_molecule_without_a_linear_group_turns_with_its_spectrum(version: str) -> None:
    assert _largest_change(_release(version), "biphenyl", ROTATIONS) < SAME_CM


NITRILE_REASON = (
    "a linear angle is measured against a fixed axis (the audit's W1); frame-free linear "
    "bends need a retrain"
)


@pytest.mark.parametrize(
    "version",
    _differs(
        lambda version: (
            f"turned benzonitrile gets bands up to "
            f"{LARGEST_CHANGE_CM[version]['turned nitrile']:.2f} cm⁻¹ away here, "
            f"{_recorded(version).rotation.with_linear_group_cm:.2f} over the molecules of "
            f"invariance.json: {NITRILE_REASON}"
        )
    ),
)
def test_a_nitrile_turns_with_its_spectrum(version: str) -> None:
    assert _change(version, "turned nitrile") < SAME_CM


MIRROR_REASON = "a dihedral enters with its sign (the audit's W3); the cos encoding needs a retrain"


@pytest.mark.parametrize(
    "version",
    _differs(
        lambda version: (
            f"the mirror image of twisted biphenyl gets bands up to "
            f"{LARGEST_CHANGE_CM[version]['mirrored biphenyl']:.2f} cm⁻¹ away: {MIRROR_REASON}"
        )
    ),
)
def test_a_mirror_image_has_the_same_spectrum(version: str) -> None:
    assert _change(version, "mirrored biphenyl") < SAME_CM


LISTING_REASON = (
    "atoms the bond graph cannot tell apart keep the file's order, and the primitives are "
    "chosen by atom number (the audit's W2); the canonical angle choice needs a retrain"
)


@pytest.mark.parametrize(
    "version",
    _differs(
        lambda version: (
            f"listings of one biphenyl geometry get bands up to "
            f"{LARGEST_CHANGE_CM[version]['listed biphenyl']:.2f} cm⁻¹ apart: {LISTING_REASON}"
        )
    ),
)
def test_every_listing_of_one_geometry_has_the_same_spectrum(version: str) -> None:
    assert _change(version, "listed biphenyl") < SAME_CM


@pytest.mark.parametrize("version", VERSIONS)
@pytest.mark.parametrize("case", ["turned nitrile", "mirrored biphenyl", "listed biphenyl"])
def test_a_release_moves_a_molecule_by_its_pinned_change(version: str, case: str) -> None:
    assert _change(version, case) == pytest.approx(LARGEST_CHANGE_CM[version][case], abs=0.01)


def test_the_bundled_release_is_among_those_tested() -> None:
    assert release.bundled().name in VERSIONS


def test_a_holdout_molecule_has_the_errors_the_evaluation_measured() -> None:
    bundled = _release("v1.3")
    errors = evaluation.molecule_errors(
        evaluation.read(RELEASE_DATA / "holdout_a_benzene.npz"), bundled
    )
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
