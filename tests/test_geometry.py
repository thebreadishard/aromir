"""Start geometry: one atom order and one geometry per molecule, however it is written."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from rdkit.Chem import rdDistGeom

from aromir import admission, geometry
from aromir.errors import RefusedError

from support import BENZENE_XYZ

if TYPE_CHECKING:
    from pathlib import Path

PHENANTHRENE_WRITINGS = ("c1ccc2c(c1)ccc1ccccc12", "C1=CC=C2C(=C1)C=CC3=CC=CC=C32")


def test_two_writings_start_from_the_same_geometry_in_the_same_order() -> None:
    starts = [
        geometry.start_from_smiles(admission.admit(admission.parse_smiles(writing)).smiles)
        for writing in PHENANTHRENE_WRITINGS
    ]
    assert starts[0].symbols == starts[1].symbols
    np.testing.assert_array_equal(starts[0].coords_bohr, starts[1].coords_bohr)


def test_heavy_atoms_come_first_then_their_hydrogens() -> None:
    start = geometry.start_from_smiles("Cc1ccccc1")
    assert start.symbols == ("C",) * 7 + ("H",) * 8


def test_the_start_geometry_is_reproducible() -> None:
    first = geometry.start_from_smiles("c1ccncc1")
    second = geometry.start_from_smiles("c1ccncc1")
    np.testing.assert_array_equal(first.coords_bohr, second.coords_bohr)


def test_the_start_geometry_has_sensible_bond_lengths() -> None:
    start = geometry.start_from_smiles("c1ccccc1")
    carbons = start.coords_bohr[:6]
    ring_bonds = np.linalg.norm(carbons - np.roll(carbons, 1, axis=0), axis=1)
    np.testing.assert_allclose(ring_bonds, 2.64, atol=0.05)  # 1.40 Å in bohr


def _shuffled_xyz(tmp_path: Path, seed: int) -> Path:
    lines = BENZENE_XYZ.strip().split("\n")
    atoms = lines[2:]
    order = np.random.default_rng(seed).permutation(len(atoms))
    path = tmp_path / f"benzene_{seed}.xyz"
    path.write_text("\n".join([*lines[:2], *(atoms[k] for k in order)]) + "\n", encoding="utf-8")
    return path


def test_an_xyz_keeps_its_coordinates_and_gets_the_canonical_order(tmp_path: Path) -> None:
    start = geometry.start_from_xyz(admission.parse_xyz(_shuffled_xyz(tmp_path, seed=1)))
    assert start.symbols == ("C",) * 6 + ("H",) * 6
    carbon_hydrogen = np.linalg.norm(start.coords_bohr[:6] - start.coords_bohr[6:], axis=1)
    np.testing.assert_allclose(carbon_hydrogen, 1.084 / 0.529177, atol=1e-3)


def test_the_canonical_order_follows_the_bond_graph_not_the_file(tmp_path: Path) -> None:
    ring_walks = []
    for seed in (1, 2):
        start = geometry.start_from_xyz(admission.parse_xyz(_shuffled_xyz(tmp_path, seed)))
        carbons = start.coords_bohr[:6]
        ring_walks.append(np.linalg.norm(carbons[1:] - carbons[:-1], axis=1))
    # Consecutive carbons are bonded in both: the order walks the ring, whatever the file's.
    for walk in ring_walks:
        np.testing.assert_allclose(walk, 1.397 / 0.529177, atol=1e-3)


def test_a_lone_hydrogen_is_not_silently_dropped() -> None:
    molecule = admission.parse_smiles("[H][H]")
    with pytest.raises(ValueError, match="cannot be ordered"):
        geometry.canonical_order(molecule)


def test_random_coordinates_are_tried_when_the_first_embedding_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    embed = rdDistGeom.EmbedMolecule
    attempts = []

    def fail_first(molecule: Any, **options: Any) -> int:  # noqa: ANN401  RDKit's signature
        attempts.append(options)
        return embed(molecule, **options) if "useRandomCoords" in options else -1

    monkeypatch.setattr(rdDistGeom, "EmbedMolecule", fail_first)
    start = geometry.start_from_smiles("c1ccccc1")
    assert start.n_atoms == 12
    assert [sorted(options) for options in attempts] == [
        ["randomSeed"],
        ["randomSeed", "useRandomCoords"],
    ]


def test_no_embedding_at_all_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rdDistGeom, "EmbedMolecule", lambda _molecule, **_options: -1)
    with pytest.raises(RefusedError, match="no 3D start geometry"):
        geometry.start_from_smiles("c1ccccc1")
