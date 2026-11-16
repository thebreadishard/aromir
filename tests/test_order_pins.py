"""The atom order and the internal coordinates of a few molecules, as the weights were made with.

Release v1.3 answers differently for another atom order and another set of internal
coordinates, and both come from other packages: the order from RDKit's canonical SMILES,
the coordinates from geomeTRIC on RDKit's start geometry. These pins fail when a new
version of either changes them, which is the moment to measure the release again before
the version is allowed in ``pyproject.toml``. Recorded with RDKit 2026.9.2 and geomeTRIC
1.1.1.
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

import pytest

from aromir import admission, geometry, release
from aromir.network import primitives

if TYPE_CHECKING:
    from pathlib import Path

PINS = [
    ("c1ccccc1", "CCCCCCHHHHHH", {"Angle": 12, "Dihedral": 24, "Distance": 12, "OutOfPlane": 6}),
    ("Oc1ccccc1", "OCCCCCCHHHHHH", {"Angle": 13, "Dihedral": 26, "Distance": 13, "OutOfPlane": 6}),
    (
        "N#Cc1ccccc1",
        "NCCCCCCCHHHHH",
        {"Angle": 12, "Dihedral": 24, "Distance": 13, "LinearAngle": 2, "OutOfPlane": 6},
    ),
    (
        "c1ccc(-c2ccccc2)cc1",
        "CCCCCCCCCCCCHHHHHHHHHH",
        {"Angle": 24, "Dihedral": 52, "Distance": 23, "OutOfPlane": 12},
    ),
    (
        "Cc1ccc2ccccc2c1",
        "CCCCCCCCCCCHHHHHHHHHH",
        {"Angle": 26, "Dihedral": 50, "Distance": 22, "OutOfPlane": 10},
    ),
    (
        "FC(F)(F)c1ccncc1",
        "FCFFCCCNCCHHHH",
        {"Angle": 17, "Dihedral": 26, "Distance": 14, "OutOfPlane": 5},
    ),
    ("Clc1ccsc1", "ClCCCSCHHH", {"Angle": 9, "Dihedral": 16, "Distance": 9, "OutOfPlane": 4}),
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


@pytest.mark.parametrize(("smiles", "symbols", "kinds"), PINS, ids=[pin[0] for pin in PINS])
def test_a_smiles_gets_the_atom_order_and_primitives_the_weights_were_made_with(
    smiles: str, symbols: str, kinds: dict[str, int]
) -> None:
    settings = release.load(release.bundled()).recipes[0].inputs.primitives
    start = geometry.start_from_smiles(admission.admit(admission.parse_smiles(smiles)).smiles)
    assert "".join(start.symbols) == symbols
    built = primitives.build(start, settings)
    counted = Counter(primitives.PRIMITIVE_KINDS[primitives.kind(p)] for p in built.internals)
    assert dict(counted) == kinds


def test_an_xyz_listing_is_put_in_the_pinned_order(tmp_path: Path) -> None:
    path = tmp_path / "benzene.xyz"
    path.write_text(BENZENE_HYDROGENS_FIRST, encoding="utf-8")
    order = geometry.canonical_order(admission.parse_xyz(path))
    assert order == [8, 7, 6, 11, 10, 9, 2, 1, 0, 5, 4, 3]
