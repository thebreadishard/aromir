"""The family rule, branch by branch, on modes built by hand."""

from __future__ import annotations

import numpy as np
import pytest

from aromir import families, geometry
from aromir.molecule import FloatArray, Geometry
from aromir.vibrations import Modes

from support import BENZENE_XYZ

HYDROGEN = 6
CARBON = 0
IN_PLANE = 0
OUT_OF_PLANE = 2


def _benzene() -> Geometry:
    rows = [line.split() for line in BENZENE_XYZ.strip().split("\n")[2:]]
    return Geometry.from_angstrom(
        tuple(row[0] for row in rows), [[float(x) for x in row[1:]] for row in rows]
    )


def _one_atom_mode(n_atoms: int, atom: int, axis: int) -> FloatArray:
    vector = np.zeros(3 * n_atoms)
    vector[3 * atom + axis] = 1.0
    return vector


def _family(molecule: Geometry, wavenumber: float, atom: int, axis: int) -> str:
    modes = Modes(np.array([wavenumber]), _one_atom_mode(molecule.n_atoms, atom, axis)[:, None])
    return families.classify(molecule, modes)[0]


@pytest.mark.parametrize(
    ("wavenumber", "atom", "axis", "expected"),
    [
        (3100.0, HYDROGEN, IN_PLANE, "CH-stretch"),
        (3100.0, CARBON, IN_PLANE, "other"),  # above every window, and not on hydrogen
        (800.0, HYDROGEN, OUT_OF_PLANE, "CH-oop"),
        (800.0, CARBON, OUT_OF_PLANE, "other"),  # out of plane, but a ring pucker
        (800.0, HYDROGEN, IN_PLANE, "other"),  # in plane and below the ring window
        (1500.0, CARBON, IN_PLANE, "ring-ip"),
        (1500.0, HYDROGEN, IN_PLANE, "ring-ip"),  # an in-plane CH bend counts as ring in-plane
        (1500.0, CARBON, OUT_OF_PLANE, "other"),
        (1000.0, HYDROGEN, OUT_OF_PLANE, "CH-oop"),  # the windows overlap; the oop rule is first
        (1000.0, CARBON, IN_PLANE, "ring-ip"),
        (2000.0, CARBON, IN_PLANE, "other"),
        (2800.0, HYDROGEN, IN_PLANE, "other"),  # the stretch window is open at 2800
    ],
)
def test_each_rule_on_benzene(wavenumber: float, atom: int, axis: int, expected: str) -> None:
    assert _family(_benzene(), wavenumber, atom, axis) == expected


def test_a_twisted_molecule_has_no_out_of_plane_family() -> None:
    biphenyl = geometry.start_from_smiles("c1ccc(cc1)-c1ccccc1")
    hydrogen = biphenyl.symbols.index("H")
    for axis in range(3):
        assert _family(biphenyl, 800.0, hydrogen, axis) == "other"


def test_every_mode_gets_a_family() -> None:
    benzene = _benzene()
    random = np.random.default_rng(0)
    vectors, _ = np.linalg.qr(random.normal(size=(36, 30)))
    modes = Modes(np.sort(random.uniform(400, 3200, size=30)), vectors)
    assigned = families.classify(benzene, modes)
    assert len(assigned) == 30
    assert set(assigned) <= set(families.FAMILIES)
