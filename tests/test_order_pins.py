"""The atom order and the internal coordinates of a few molecules, as the weights were made with.

Release v1.3 answers differently for another atom order and another set of internal
coordinates, and both come from other packages: the order from RDKit's canonical SMILES,
the coordinates from geomeTRIC on RDKit's start geometry. These pins fail when a new
version of either changes them for one of these molecules, which is the moment to measure
the release again before the version is allowed in ``pyproject.toml``.

What is pinned is a digest of geomeTRIC's own description of every primitive in order
("Distance 1-2", "Dihedral 11-5-6-12"): it names the atoms by number, so it sees carbons
that changed places and another choice of angles, which a count of kinds does not. The
symbols and the counts beside it say what changed when a digest does. Recorded with RDKit
2026.9.2 and geomeTRIC 1.1.1.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from typing import TYPE_CHECKING

import pytest

from aromir import admission, geometry, release
from aromir.network import primitives

if TYPE_CHECKING:
    from pathlib import Path

PINS = [
    (
        "c1ccccc1",
        "CCCCCCHHHHHH",
        {"Angle": 12, "Dihedral": 24, "Distance": 12, "OutOfPlane": 6},
        "8a29dc53095653b0",
    ),
    (
        "Oc1ccccc1",
        "OCCCCCCHHHHHH",
        {"Angle": 13, "Dihedral": 26, "Distance": 13, "OutOfPlane": 6},
        "fbdc9025ebd8c517",
    ),
    (
        "N#Cc1ccccc1",
        "NCCCCCCCHHHHH",
        {"Angle": 12, "Dihedral": 24, "Distance": 13, "LinearAngle": 2, "OutOfPlane": 6},
        "aac36f1a329109a7",
    ),
    (
        "c1ccc(-c2ccccc2)cc1",
        "CCCCCCCCCCCCHHHHHHHHHH",
        {"Angle": 24, "Dihedral": 52, "Distance": 23, "OutOfPlane": 12},
        "127bde54247917be",
    ),
    (
        "Cc1ccc2ccccc2c1",
        "CCCCCCCCCCCHHHHHHHHHH",
        {"Angle": 26, "Dihedral": 50, "Distance": 22, "OutOfPlane": 10},
        "92f124889e606d97",
    ),
    (
        "FC(F)(F)c1ccncc1",
        "FCFFCCCNCCHHHH",
        {"Angle": 17, "Dihedral": 26, "Distance": 14, "OutOfPlane": 5},
        "1cb425f6f0d99e77",
    ),
    (
        "Clc1ccsc1",
        "ClCCCSCHHH",
        {"Angle": 9, "Dihedral": 16, "Distance": 9, "OutOfPlane": 4},
        "f1c6c17e5e664788",
    ),
]
BENZENE_HYDROGENS_FIRST = """12
benzene, hydrogens first
H   1.2405  -2.1486  0.0
H  -1.2405  -2.1486  0.0
H  -2.4810   0.0000  0.0
H  -1.2405   2.1486  0.0
H   1.2405   2.1486  0.0
H   2.4810   0.0000  0.0
C   0.6985  -1.2098  0.0
C  -0.6985  -1.2098  0.0
C  -1.3970   0.0000  0.0
C  -0.6985   1.2098  0.0
C   0.6985   1.2098  0.0
C   1.3970   0.0000  0.0
"""


def _digest(internals: list[primitives.Primitive]) -> str:
    described = "\n".join(repr(primitive) for primitive in internals)
    return hashlib.sha256(described.encode()).hexdigest()[:16]


@pytest.mark.parametrize(
    ("smiles", "symbols", "kinds", "digest"), PINS, ids=[pin[0] for pin in PINS]
)
def test_a_smiles_gets_the_atom_order_and_primitives_the_weights_were_made_with(
    smiles: str, symbols: str, kinds: dict[str, int], digest: str
) -> None:
    settings = release.load(release.bundled()).recipes[0].inputs.primitives
    start = geometry.start_from_smiles(admission.admit(admission.parse_smiles(smiles)).smiles)
    assert "".join(start.symbols) == symbols
    built = primitives.build(start, settings)
    counted = Counter(primitives.PRIMITIVE_KINDS[primitives.kind(p)] for p in built.internals)
    assert dict(counted) == kinds
    assert _digest(list(built.internals)) == digest


def test_the_digest_sees_two_atoms_of_one_element_change_places() -> None:
    settings = release.load(release.bundled()).recipes[0].inputs.primitives
    start = geometry.start_from_smiles("Cc1ccc2ccccc2c1")
    order = list(range(start.n_atoms))
    order[1], order[5] = order[5], order[1]
    swapped = type(start)(tuple(start.symbols[i] for i in order), start.coords_bohr[order])
    assert swapped.symbols == start.symbols
    first = _digest(list(primitives.build(start, settings).internals))
    assert _digest(list(primitives.build(swapped, settings).internals)) != first


def test_an_xyz_listing_is_put_in_the_pinned_order(tmp_path: Path) -> None:
    path = tmp_path / "benzene.xyz"
    path.write_text(BENZENE_HYDROGENS_FIRST, encoding="utf-8")
    order = geometry.canonical_order(admission.parse_xyz(path))
    assert order == [8, 7, 6, 11, 10, 9, 2, 1, 0, 5, 4, 3]
