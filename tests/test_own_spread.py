"""A band's licence follows its own spread: how far it moves when its molecule is given another way.

The molecule is turned, mirrored and listed in other atom orders, the networks correct each,
and a band whose position moves by more than the licence line is not licensed. The stored
molecules are read as the pipeline reads an xyz request, so their atoms are in aromir's
canonical order; their spreads on v1.4 are pinned as measured on 19 November 2026.

The band of 9-methylanthracene that moves most, by 9.95 cm⁻¹, is its methyl torsion, which
the stored Hessian has imaginary (−50.8 cm⁻¹): a flat coordinate, never licensed. With that
mode made real at 40 cm⁻¹ (``TORSION``), as at a minimum, the same band moves by 5.16
cm⁻¹, and the rule by the molecule's own spread is what withholds its licence.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest

from aromir import admission, card, certificate, lowlevel, own_spread, release, response
from aromir import vibrations as vib
from aromir.constants import WAVENUMBERS_PER_HARTREE

from support import as_computed, random_hessian, unmoved

if TYPE_CHECKING:
    from aromir.molecule import FloatArray, Geometry

LINE_CM = 3.0
PIN_CM = 0.02
"""How far a pinned spread may lie from the one measured: the networks run in single
precision, and other hardware moves a band by far less."""
LARGEST_CM = {
    "benzene": (0.88, "another listing 1"),
    "benzonitrile": (1.75, "rotation 2"),
    "biphenyl": (2.85, "a swap of equivalent atoms 1"),
    "diphenyl_ether": (1.40, "a swap of equivalent atoms 5"),
    "9_methylanthracene": (9.95, "a swap of equivalent atoms 6"),
}
"""Each stored molecule's largest band spread on v1.4 and the change that gave it."""
BEYOND_THE_LINE = {
    "benzene": 0,
    "benzonitrile": 0,
    "biphenyl": 0,
    "diphenyl_ether": 0,
    "9_methylanthracene": 1,
}
"""How many of its bands move by more than the licence line."""
TORSION = "9-methylanthracene, its torsion real"
TORSION_CM = 40.0


@cache
def _molecule(name: str) -> tuple[str, Geometry, FloatArray]:
    """Return a stored molecule as computed; ``TORSION`` with its softest mode made real."""
    if name != TORSION:
        return as_computed(name)
    smiles, molecule, hessian = as_computed("9_methylanthracene")
    modes = vib.vibrational_modes(hessian, molecule)
    soft = modes.vectors[:, 0] * vib.mass_weights(molecule.masses_amu)
    lift = ((TORSION_CM**2) + modes.wavenumbers[0] ** 2) / WAVENUMBERS_PER_HARTREE**2
    return smiles, molecule, hessian + lift * np.outer(soft, soft)


@cache
def _v14() -> release.Release:
    return release.load(release.bundled().parent / "v1.4")


def _low(hessian: FloatArray) -> lowlevel.LowLevel:
    return lowlevel.LowLevel(
        energy=-1.0,
        hessian=hessian,
        polar_tensor=np.random.default_rng(1).normal(size=(3, len(hessian))),
        refinement=response.Refinement(1e-9, 1e-9, 0, 0),
        sum_rule_violation=1e-7,
        seconds=1.0,
    )


@cache
def _record(name: str) -> dict[str, Any]:
    """Return a stored molecule's certificate record on v1.4."""
    smiles, molecule, hessian = _molecule(name)
    admitted = admission.admit(admission.read(smiles))
    corrections = _v14().corrections(molecule, hessian)
    return certificate.build(admitted, molecule, _low(hessian), corrections, _v14())


def _bands(name: str, spread: own_spread.OwnSpread | None = None) -> list[certificate.Band]:
    """Return a molecule's bands on v1.4 with a given own spread, by default none at all."""
    _, molecule, hessian = _molecule(name)
    corrections = _v14().corrections(molecule, hessian)
    n_bands = 3 * molecule.n_atoms - 6
    return certificate.bands(
        molecule,
        _low(hessian),
        corrections,
        _v14().evaluation,
        inside_training_sizes=True,
        spread=unmoved(n_bands) if spread is None else spread,
    )


@pytest.mark.parametrize("name", list(LARGEST_CM))
def test_each_stored_molecule_gets_its_own_measured_spread(name: str) -> None:
    record = _record(name)
    spreads = [band["own_spread_cm"] for band in record["bands"]]
    largest = max(spreads)
    change = record["bands"][spreads.index(largest)]["own_spread_change"]
    assert largest == pytest.approx(LARGEST_CM[name][0], abs=PIN_CM)
    assert change == LARGEST_CM[name][1]
    beyond = [band for band in record["bands"] if band["own_spread_cm"] > LINE_CM]
    assert len(beyond) == BEYOND_THE_LINE[name]
    assert all(not band["licensed"] for band in beyond)
    within = [band for band in record["bands"] if band["own_spread_cm"] <= LINE_CM]
    unruled = {band.wavenumber_cm: band.licence_note for band in _bands(name)}
    assert all(band["licence_note"] == unruled[band["wavenumber_cm"]] for band in within)


def test_9_methylanthracenes_band_that_moves_10_cm_is_its_flat_torsion() -> None:
    record = _record("9_methylanthracene")
    assert record["molecule"]["linear_group"] is False
    (moved,) = [band for band in record["bands"] if band["own_spread_cm"] > LINE_CM]
    assert moved["b3lyp_cm"] < 0
    assert moved["licence_note"] == certificate.FLAT_COORDINATE
    assert f"| {moved['own_spread_cm']:.2f}, a swap of equivalent atoms 6 |" in (
        certificate.to_markdown(record)
    )


def test_9_methylanthracenes_real_torsion_loses_its_licence_by_its_own_spread() -> None:
    record = _record(TORSION)
    (moved,) = [band for band in record["bands"] if band["own_spread_cm"] > LINE_CM]
    assert moved["b3lyp_cm"] == pytest.approx(TORSION_CM, abs=0.1)
    assert moved["family"] == "other"
    assert moved["own_spread_cm"] == pytest.approx(5.16, abs=PIN_CM)
    note = (
        f"its position moved by {moved['own_spread_cm']:.2f} cm⁻¹ under a swap of equivalent "
        "atoms 6 of this molecule, more than the licence line"
    )
    assert moved["licence_note"] == note
    unruled = {band.wavenumber_cm: band for band in _bands(TORSION)}
    assert unruled[moved["wavenumber_cm"]].licensed
    text = certificate.to_markdown(record)
    assert f"| no: {note} |" in text
    assert f"| {moved['own_spread_cm']:.2f}, a swap of equivalent atoms 6 |" in text
    assert (
        "Each band's own spread was measured for this molecule: the networks corrected it again "
        "given in 16 other ways (3 rotations, 2 mirror images, 3 other listings of the atoms, 8 "
        "swaps of equivalent atoms), and a band's spread is the furthest its position moved; the "
        f"largest is {moved['own_spread_cm']:.2f} cm⁻¹, under a swap of equivalent atoms 6, and 1 "
        "band(s) moved by more than the licence line, so none of them is licensed; the "
        "corrected positions are still printed."
    ) in record["statement"]
    assert sum(band["licensed"] for band in record["bands"]) == 74
    assert (
        "the band is real for the seeds' mean and for every seed, its own position moved"
        in (record["statement"])
    )


def test_the_rule_leaves_the_printed_positions_as_they_are() -> None:
    printed = [band["wavenumber_cm"] for band in _record(TORSION)["bands"]]
    assert printed == [band.wavenumber_cm for band in _bands(TORSION)]


@pytest.mark.parametrize("name", ["benzonitrile", "biphenyl"])
def test_a_molecule_whose_bands_stay_within_the_line_keeps_its_licence(name: str) -> None:
    record = _record(name)
    assert [band["licence_note"] for band in record["bands"]] == [
        band.licence_note for band in _bands(name)
    ]
    assert any(band["licensed"] for band in record["bands"])
    assert "no band moved by more than the licence line" in record["statement"]


def _uniform(value: float, n_bands: int) -> own_spread.OwnSpread:
    return own_spread.OwnSpread(
        spread_cm=(value,) * n_bands,
        change=("rotation 1",) * n_bands,
        tried=("rotation 1",),
        refusal="",
    )


def test_a_spread_at_the_line_keeps_the_licence_and_one_above_withholds_it() -> None:
    n_bands = len(_bands("benzonitrile"))
    plain = [band.licence_note for band in _bands("benzonitrile")]
    at_line = _bands("benzonitrile", _uniform(LINE_CM, n_bands))
    assert [band.licence_note for band in at_line] == plain
    assert any(band.licensed for band in at_line)
    above = _bands("benzonitrile", _uniform(3.01, n_bands))
    note = (
        "its position moved by 3.01 cm⁻¹ under rotation 1 of this molecule, more than the "
        "licence line"
    )
    assert {band.licence_note for band in above} == {note}
    assert [band.wavenumber_cm for band in above] == [band.wavenumber_cm for band in at_line]


def test_a_check_that_cannot_run_licenses_no_band_and_says_why() -> None:
    smiles, in_order, in_order_hessian = as_computed("biphenyl")
    # Two hydrogens of different carbons swapped: no longer the order of the SMILES.
    swapped = list(range(in_order.n_atoms))
    swapped[-2:] = swapped[-1], swapped[-2]
    molecule, hessian = own_spread.reordered(in_order, in_order_hessian, tuple(swapped))
    found = own_spread.measure(_v14(), smiles, molecule, _low(hessian), np.zeros_like(hessian))
    assert found.spread_cm == ()
    assert found.refusal == (
        "a bond of the SMILES is too long in the geometry: its atoms are in another order"
    )
    with pytest.raises(ValueError, match="no spread was measured"):
        _ = found.largest
    admitted = admission.admit(admission.read(smiles))
    corrections = _v14().corrections(molecule, hessian)
    record = certificate.build(admitted, molecule, _low(hessian), corrections, _v14())
    assert {band["licence_note"] for band in record["bands"]} == {certificate.UNCHECKED}
    assert all(band["own_spread_cm"] is None for band in record["bands"])
    assert (
        "How far this molecule's bands move when it is turned, mirrored or listed in another "
        f"order could not be measured ({found.refusal}), so no band is licensed; the corrected "
        "positions are still printed."
    ) in record["statement"]
    assert record["own_spread"]["not_measured_because"] == found.refusal
    assert "| not measured |" in certificate.to_markdown(record)


def test_a_geometry_whose_elements_are_out_of_order_is_refused() -> None:
    smiles, molecule, _ = as_computed("benzonitrile")
    reversed_atoms = molecule.reordered(list(range(molecule.n_atoms))[::-1])
    with pytest.raises(ValueError, match="elements are not in the canonical order"):
        own_spread.bond_graph(smiles, reversed_atoms)


def test_the_check_is_the_same_every_time() -> None:
    smiles, molecule, hessian = as_computed("benzonitrile")
    first = own_spread.measure(_v14(), smiles, molecule, _low(hessian), np.zeros_like(hessian))
    again = own_spread.measure(_v14(), smiles, molecule, _low(hessian), np.zeros_like(hessian))
    assert first == again
    tried = own_spread.changes(own_spread.bond_graph(smiles, molecule))
    assert [change.name for change in tried] == [
        "rotation 1",
        "rotation 2",
        "rotation 3",
        "the mirror image",
        "the mirror image, rotated",
        "another listing 1",
    ]
    rng = np.random.default_rng(own_spread.SEED)
    np.testing.assert_array_equal(tried[0].matrix, own_spread.random_rotation(rng))


def test_the_changes_are_the_same_molecule_and_undo_returns_a_hessian() -> None:
    smiles, molecule, _ = as_computed("9_methylanthracene")
    graph = own_spread.bond_graph(smiles, molecule)
    bonds = {frozenset((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())) for bond in graph.GetBonds()}
    hessian = random_hessian(molecule.n_atoms, seed=4)
    for change in own_spread.changes(graph):
        changed, changed_hessian = change.apply(molecule, hessian)
        np.testing.assert_allclose(change.undo(changed_hessian), hessian, atol=1e-12)
        assert changed.symbols == tuple(molecule.symbols[k] for k in change.order)
        inverse = np.argsort(change.order)
        assert {frozenset(int(inverse[k]) for k in bond) for bond in bonds} == bonds
        determinant = -1.0 if "mirror" in change.name else 1.0
        assert np.linalg.det(change.matrix) == pytest.approx(determinant)


@dataclass(frozen=True)
class _Training:
    soft_mode_repair: bool = False


@dataclass(frozen=True)
class _Networks:
    """Stand-in networks whose answer is a fixed function of the Hessian they read."""

    answer: str
    training: _Training = _Training()

    def corrections(self, molecule: Geometry, hessian: FloatArray) -> release.Corrections:
        # The other answer is stiffer along the laboratory's z axis: no rotation leaves it alone.
        laboratory_z = np.kron(np.eye(molecule.n_atoms), np.diag([0.0, 0.0, 1.0]))
        delta = np.asarray(0.01 * hessian if self.answer == "scaled" else 0.02 * laboratory_z)
        return release.Corrections(delta, (delta, delta))


def _measured(answer: str) -> own_spread.OwnSpread:
    smiles, molecule, hessian = as_computed("benzonitrile")
    networks = cast("release.Release", _Networks(answer))
    correction = networks.corrections(molecule, hessian).mean
    return own_spread.measure(networks, smiles, molecule, _low(hessian), correction)


def test_networks_that_turn_with_the_molecule_move_no_band() -> None:
    found = _measured("scaled")
    assert max(found.spread_cm) == 0.0


def test_networks_tied_to_the_laboratory_axes_move_bands_under_a_rotation_only() -> None:
    found = _measured("laboratory z")
    assert max(found.spread_cm) > LINE_CM
    assert {
        change for change, spread in zip(found.change, found.spread_cm, strict=True) if spread > 0
    } <= {
        "rotation 1",
        "rotation 2",
        "rotation 3",
        "the mirror image, rotated",
    }


def test_the_card_states_the_rule_and_keeps_the_record_as_context() -> None:
    text = card.to_markdown(_v14())
    assert (
        "- **A band's licence follows its own measured spread.** For every molecule, aromir's "
        "certificate has the networks correct the same molecule given in other ways: 3 random "
        "rotations, the mirror image and a rotated mirror image, 4 random listings of the atoms "
        "put into aromir's canonical order and up to 12 swaps of atoms the bond graph cannot "
        "tell apart, drawn with the seed 7. Each band is paired with itself by its motion, and "
        "it is licensed only when its position moved by no more than 3 cm⁻¹ under every one of "
        "them; where that cannot be measured, no band is licensed. The corrected positions are "
        "printed either way. The spreads above are the release's record, measured on a few "
        "molecules: they describe the network and decide no licence."
    ) in text
    assert "11.20 cm⁻¹ on 3 molecule(s) with a linear group" in text
