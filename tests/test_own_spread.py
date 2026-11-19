"""A band's licence follows its own spread: how far it lies from print in another presentation.

The molecule is turned, mirrored and listed in other atom orders, the networks correct each
presentation, and a band whose position lies further from the printed one than the licence
line is not licensed. The stored molecules are read as the pipeline reads an xyz request,
so their atoms are in aromir's canonical order; their spreads on v1.4 are pinned as
measured on 20 November 2026, each with the presentation that gave it: an atom order, or
the rotation's place among the draws. No pin rests on a presentation's ordinal name.

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
import rdkit

from aromir import admission, card, certificate, lowlevel, own_spread, release, response
from aromir import vibrations as vib
from aromir.constants import WAVENUMBERS_PER_HARTREE

from support import as_computed, random_hessian, uniform_spread, unmoved

if TYPE_CHECKING:
    from aromir.molecule import FloatArray, Geometry

LINE_CM = 3.0
PIN_CM = 0.02
"""How far a pinned spread may lie from the one measured: the networks run in single
precision, and other hardware moves a band by far less."""
TORSION = "9-methylanthracene, its torsion real"
TORSION_CM = 40.0
RING_FLIP: tuple[int, ...] = (0, 1, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 15, 17, 16)
RING_FLIP += (26, 25, 24, 23, 22, 21, 20, 19, 18)
"""9-methylanthracene's two outer rings exchanged and two methyl hydrogens with them."""
LARGEST_CM: dict[str, tuple[float, tuple[int, ...] | int]] = {
    "benzene": (0.88, (2, 3, 4, 5, 0, 1, 8, 9, 10, 11, 6, 7)),
    "benzonitrile": (2.78, 7),
    "biphenyl": (
        2.85,
        (0, 11, 10, 3, 4, 5, 6, 7, 8, 9, 2, 1, 12, 21, 20, 15, 16, 17, 18, 19, 14, 13),
    ),
    "diphenyl_ether": (
        1.40,
        (8, 9, 10, 5, 4, 3, 2, 1, 0, 12, 11, 6, 7, 18, 19, 20, 15, 14, 13, 22, 21, 16, 17),
    ),
    "9_methylanthracene": (9.95, RING_FLIP),
    TORSION: (5.16, RING_FLIP),
}
"""Each stored molecule's largest band spread on v1.4 and the presentation that gave it:
the atom order of a swap, or which of the generator's rotations it was (the seventh)."""
BEYOND_THE_LINE = {
    "benzene": 0,
    "benzonitrile": 0,
    "biphenyl": 0,
    "diphenyl_ether": 0,
    "9_methylanthracene": 1,
    TORSION: 1,
}
"""How many of its bands move by more than the licence line."""
SWAPS = {
    "benzene": 11,
    "benzonitrile": 1,
    "biphenyl": 7,
    "diphenyl_ether": 7,
    "9_methylanthracene": 11,
    TORSION: 11,
}
"""Every order each molecule's graph cannot tell from its own: all are tried, so no listing
adds one, whatever way the running RDKit breaks its ties."""


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
        outside_domain="",
        spread=unmoved(n_bands) if spread is None else spread,
    )


def _pinned_change(name: str) -> own_spread.Change:
    """Return the presentation a molecule's pin names, made here and not read from a record."""
    _, molecule, _ = _molecule(name)
    _, pinned = LARGEST_CM[name]
    identity = tuple(range(molecule.n_atoms))
    if isinstance(pinned, tuple):
        return own_spread.Change("pinned", "swap", np.eye(3), pinned)
    rng = np.random.default_rng(own_spread.SEED)
    drawn = [own_spread.random_rotation(rng) for _ in range(pinned)]
    return own_spread.Change("pinned", "rotation", drawn[-1], identity)


def _worst(record: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the band that moved most and the recorded presentation that moved it."""
    band = max(record["bands"], key=lambda band: band["own_spread_cm"])
    (presentation,) = [
        tried
        for tried in record["own_spread"]["presentations"]
        if tried["name"] == band["own_spread_change"]
    ]
    return band, presentation


@pytest.mark.parametrize("name", list(LARGEST_CM))
def test_each_stored_molecule_gets_its_own_measured_spread(name: str) -> None:
    record = _record(name)
    band, _ = _worst(record)
    assert band["own_spread_cm"] == pytest.approx(LARGEST_CM[name][0], abs=PIN_CM)
    beyond = [band for band in record["bands"] if band["own_spread_cm"] > LINE_CM]
    assert len(beyond) == BEYOND_THE_LINE[name]
    assert all(not band["licensed"] for band in beyond)
    within = [band for band in record["bands"] if band["own_spread_cm"] <= LINE_CM]
    unruled = {band.wavenumber_cm: band.licence_note for band in _bands(name)}
    assert all(band["licence_note"] == unruled[band["wavenumber_cm"]] for band in within)
    assert record["own_spread"]["counts"] == {
        "rotation": own_spread.N_ROTATIONS,
        "mirror image": 2,
        "swap": SWAPS[name],
        "listing": 0,
    }


@pytest.mark.parametrize("name", list(LARGEST_CM))
def test_the_pinned_presentation_gives_the_pinned_spread(name: str) -> None:
    smiles, molecule, hessian = _molecule(name)
    pinned = _pinned_change(name)
    printed = vib.vibrational_modes(hessian + _v14().corrections(molecule, hessian).mean, molecule)
    modes = own_spread.corrected_modes(_v14(), molecule, _low(hessian), pinned)
    shifts = own_spread.band_shifts(modes, printed)
    assert float(np.max(shifts)) == pytest.approx(LARGEST_CM[name][0], abs=PIN_CM)
    tried = _record(name)["own_spread"]["presentations"]
    if pinned.kind == "swap":
        assert list(pinned.order) in [made["order"] for made in tried if made["kind"] == "swap"]
        graph = own_spread.graph_of(smiles, molecule)
        found = own_spread.skeleton_symmetries(graph)
        assert pinned.order in own_spread.swaps(found, np.random.default_rng(0))
    else:
        _, number = LARGEST_CM[name]
        assert isinstance(number, int)
        rotations = [made["matrix"] for made in tried if made["kind"] == "rotation"]
        np.testing.assert_array_equal(rotations[number - 1], pinned.matrix)


def test_the_record_says_how_to_make_each_presentation_again() -> None:
    made = _record("9_methylanthracene")["own_spread"]
    assert made["seed"] == own_spread.SEED
    assert made["rdkit"] == rdkit.__version__
    assert made["max_symmetries"] == own_spread.MAX_SYMMETRIES
    assert made["skeleton_symmetries"] == 2
    assert made["equivalent_orders"] == 12
    assert made["symmetries_capped"] is False
    assert made["not_measured_because"] == ""
    orders = [tried["order"] for tried in made["presentations"] if "order" in tried]
    assert len(orders) == 11
    assert all(sorted(order) == list(range(27)) for order in orders)
    matrices = [tried["matrix"] for tried in made["presentations"] if "matrix" in tried]
    assert len(matrices) == own_spread.N_ROTATIONS + 2
    for matrix in matrices:
        np.testing.assert_allclose(np.array(matrix) @ np.array(matrix).T, np.eye(3), atol=1e-12)


def test_9_methylanthracenes_band_that_moves_10_cm_is_its_flat_torsion() -> None:
    record = _record("9_methylanthracene")
    assert record["molecule"]["linear_group"] is False
    moved, presentation = _worst(record)
    assert presentation["order"] == list(RING_FLIP)
    assert moved["b3lyp_cm"] < 0
    assert moved["licence_note"] == certificate.FLAT_COORDINATE
    assert f"| {moved['own_spread_cm']:.2f}, {presentation['name']} |" in (
        certificate.to_markdown(record)
    )


def test_9_methylanthracenes_real_torsion_loses_its_licence_by_its_own_spread() -> None:
    record = _record(TORSION)
    moved, presentation = _worst(record)
    assert presentation["order"] == list(RING_FLIP)
    assert moved["b3lyp_cm"] == pytest.approx(TORSION_CM, abs=0.1)
    assert moved["family"] == "other"
    stated = f"{moved['own_spread_cm']:.2f}"
    note = (
        f"its position moved by {stated} cm⁻¹ under {presentation['name']} of this molecule, "
        "more than the licence line"
    )
    assert moved["licence_note"] == note
    unruled = {band.wavenumber_cm: band for band in _bands(TORSION)}
    assert unruled[moved["wavenumber_cm"]].licensed
    text = certificate.to_markdown(record)
    assert f"| no: {note} |" in text
    assert f"| {stated}, {presentation['name']} |" in text
    assert (
        "its own position moved by no more than the licence line in the 21 presentations tried "
        "(this molecule turned, mirrored or listed in another atom order)"
    ) in record["statement"]
    assert (
        "Each band's own spread was measured for this molecule: the networks corrected it again "
        "in 21 other presentations (8 rotations, 2 mirror images, 11 swaps of equivalent atoms), "
        "and a band's spread is the furthest its position lay from the printed one in any of "
        "them: its distance from the printed presentation, not the range between presentations. "
        f"The largest is {stated} cm⁻¹, under {presentation['name']}, and 1 band(s) moved by "
        "more than the licence line, so none of them is licensed; the corrected positions are "
        "still printed."
    ) in record["statement"]
    assert sum(band["licensed"] for band in record["bands"]) == 74


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


def test_a_spread_at_the_line_keeps_the_licence_and_one_above_withholds_it() -> None:
    n_bands = len(_bands("benzonitrile"))
    plain = [band.licence_note for band in _bands("benzonitrile")]
    at_line = _bands("benzonitrile", uniform_spread(LINE_CM, n_bands))
    assert [band.licence_note for band in at_line] == plain
    assert any(band.licensed for band in at_line)
    above = _bands("benzonitrile", uniform_spread(3.01, n_bands))
    note = (
        "its position moved by 3.01 cm⁻¹ under rotation 1 of this molecule, more than the "
        "licence line"
    )
    assert {band.licence_note for band in above} == {note}
    assert [band.wavenumber_cm for band in above] == [band.wavenumber_cm for band in at_line]


def test_the_line_is_held_against_the_unrounded_spread() -> None:
    # 3.004 rounds to 3.00, the line itself: it is beyond the line, and printed as it is.
    n_bands = len(_bands("benzonitrile"))
    just_above = _bands("benzonitrile", uniform_spread(3.004, n_bands))
    note = (
        "its position moved by 3.004 cm⁻¹ under rotation 1 of this molecule, more than the "
        "licence line"
    )
    assert {band.licence_note for band in just_above} == {note}
    assert {band.own_spread_cm for band in just_above} == {3.004}
    just_below = _bands("benzonitrile", uniform_spread(2.996, n_bands))
    assert any(band.licensed for band in just_below)
    assert {band.own_spread_cm for band in just_below} == {3.0}


@pytest.mark.parametrize(
    ("spread", "text"),
    [
        (1.4, "1.40"),
        (3.0, "3.00"),
        (2.996, "3.00"),
        (3.004, "3.004"),
        (3.00004, "3.00004"),
        (9.951, "9.95"),
    ],
)
def test_a_spread_is_printed_to_two_decimals_unless_they_hide_its_side(
    spread: float, text: str
) -> None:
    assert own_spread.stated(spread, LINE_CM) == text


def _hydrogens_swapped() -> tuple[str, Geometry, FloatArray]:
    """Return biphenyl with two hydrogens of different carbons in each other's place."""
    smiles, in_order, hessian = as_computed("biphenyl")
    swapped = list(range(in_order.n_atoms))
    swapped[-2:] = swapped[-1], swapped[-2]
    return smiles, *own_spread.reordered(in_order, hessian, tuple(swapped))


def test_a_check_that_cannot_run_licenses_no_band_and_says_why() -> None:
    smiles, molecule, hessian = _hydrogens_swapped()
    found = own_spread.measure(_v14(), smiles, molecule, _low(hessian), np.zeros_like(hessian))
    assert found.spread_cm == ()
    assert found.refusal.startswith(
        "the geometry is not the molecule of its SMILES in the canonical atom order: "
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
    assert "in the presentations tried (this molecule turned" in record["statement"]
    assert record["own_spread"]["not_measured_because"] == found.refusal
    assert record["own_spread"]["presentations"] == []
    assert "| not measured |" in certificate.to_markdown(record)


def test_a_geometry_whose_elements_are_out_of_order_is_refused() -> None:
    smiles, molecule, _ = as_computed("benzonitrile")
    reversed_atoms = molecule.reordered(list(range(molecule.n_atoms))[::-1])
    with pytest.raises(ValueError, match="are not those of"):
        own_spread.graph_of(smiles, reversed_atoms)
    with pytest.raises(ValueError, match="does not parse as a SMILES"):
        own_spread.bond_graph("c1ccccc")


def test_the_check_is_the_same_every_time() -> None:
    smiles, molecule, hessian = as_computed("benzonitrile")
    first = own_spread.measure(_v14(), smiles, molecule, _low(hessian), np.zeros_like(hessian))
    again = own_spread.measure(_v14(), smiles, molecule, _low(hessian), np.zeros_like(hessian))
    assert first.spread_cm == again.spread_cm
    assert first.change == again.change
    assert first.record() == again.record()
    assert [change.kind for change in first.tried] == [
        *["rotation"] * own_spread.N_ROTATIONS,
        "mirror image",
        "mirror image",
        "swap",
    ]
    rng = np.random.default_rng(own_spread.SEED)
    np.testing.assert_array_equal(first.tried[0].matrix, own_spread.random_rotation(rng))


def _bonds(graph: Any) -> set[frozenset[int]]:  # noqa: ANN401  an RDKit molecule
    return {frozenset((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())) for bond in graph.GetBonds()}


def _keeps_the_graph(graph: Any, order: tuple[int, ...]) -> bool:  # noqa: ANN401  an RDKit molecule
    """Tell whether an order of the atoms leaves every bond and element where it was."""
    bonds = _bonds(graph)
    place = np.argsort(order)
    same_bonds = {frozenset(int(place[k]) for k in bond) for bond in bonds} == bonds
    symbols = [atom.GetSymbol() for atom in graph.GetAtoms()]
    return same_bonds and [symbols[k] for k in order] == symbols


def test_the_changes_are_the_same_molecule_and_undo_returns_a_hessian() -> None:
    smiles, molecule, _ = as_computed("9_methylanthracene")
    graph = own_spread.graph_of(smiles, molecule)
    hessian = random_hessian(molecule.n_atoms, seed=4)
    for change in own_spread.changes(graph, own_spread.skeleton_symmetries(graph)):
        changed, changed_hessian = change.apply(molecule, hessian)
        np.testing.assert_allclose(change.undo(changed_hessian), hessian, atol=1e-12)
        assert changed.symbols == tuple(molecule.symbols[k] for k in change.order)
        assert _keeps_the_graph(graph, change.order)
        determinant = -1.0 if change.kind == "mirror image" else 1.0
        assert np.linalg.det(change.matrix) == pytest.approx(determinant)
        assert ("order" in change.record()) is (change.kind in ("swap", "listing"))


def _heavy(graph: Any) -> list[int]:  # noqa: ANN401  an RDKit molecule
    return [atom.GetIdx() for atom in graph.GetAtoms() if atom.GetAtomicNum() > 1]


def test_a_small_molecule_gets_every_order_its_graph_cannot_tell_apart() -> None:
    graph = own_spread.bond_graph("Cc1ccccc1")
    found = own_spread.skeleton_symmetries(graph)
    assert (len(found.skeleton) + 1, found.n_exchanges, found.n_orders) == (2, 6, 12)
    orders = own_spread.swaps(found, np.random.default_rng(own_spread.SEED))
    assert len(set(orders)) == 11
    assert orders == sorted(orders)
    assert all(_keeps_the_graph(graph, order) for order in orders)
    heavy = _heavy(graph)
    assert sum(all(order[k] == k for k in heavy) for order in orders) == 5


def test_a_methyl_rich_molecule_gets_swaps_of_its_heavy_atoms_and_of_its_hydrogens() -> None:
    # Hexamethylbenzene has 12 skeleton symmetries and 6 ways to list each methyl's
    # hydrogens: 559,872 orders, of which the first thousand RDKit matches move no carbon.
    graph = own_spread.bond_graph("Cc1c(C)c(C)c(C)c(C)c1C")
    found = own_spread.skeleton_symmetries(graph)
    assert (len(found.skeleton) + 1, found.n_orders, found.capped) == (12, 559872, False)
    orders = own_spread.swaps(found, np.random.default_rng(own_spread.SEED))
    assert len(orders) == 11 + own_spread.N_HYDROGEN_SWAPS
    assert all(_keeps_the_graph(graph, order) for order in orders)
    heavy = _heavy(graph)
    assert sum(any(order[k] != k for k in heavy) for order in orders[:11]) == 11
    assert orders[0] == found.order(found.skeleton[0], {})
    exchanged = orders[11:]
    assert len(set(exchanged)) == own_spread.N_HYDROGEN_SWAPS
    assert not set(exchanged) & set(orders[:11])


def test_more_skeleton_symmetries_than_are_tried_are_drawn() -> None:
    graph = own_spread.bond_graph("CC(C)(C)c1ccc(C(C)(C)C)cc1")
    found = own_spread.skeleton_symmetries(graph)
    assert len(found.skeleton) + 1 == 144
    assert not found.capped
    orders = own_spread.swaps(found, np.random.default_rng(own_spread.SEED))
    again = own_spread.swaps(found, np.random.default_rng(own_spread.SEED))
    assert orders == again
    assert orders != own_spread.swaps(found, np.random.default_rng(own_spread.SEED + 1))
    assert len(orders) == own_spread.N_SWAPS + own_spread.N_HYDROGEN_SWAPS
    assert all(_keeps_the_graph(graph, order) for order in orders)
    skeleton_only = orders[: own_spread.N_SWAPS]
    assert skeleton_only == sorted(skeleton_only)
    assert len(set(skeleton_only)) == own_spread.N_SWAPS
    tried = own_spread.changes(graph, found)
    assert sum(change.kind == "swap" for change in tried) == len(set(orders))


def test_a_molecule_at_the_limit_of_symmetries_is_recorded_as_capped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(own_spread, "MAX_SYMMETRIES", 8)
    graph = own_spread.bond_graph("CC(C)(C)c1ccc(C(C)(C)C)cc1")
    found = own_spread.skeleton_symmetries(graph)
    assert found.capped
    assert len(found.skeleton) + 1 <= 8
    orders = own_spread.swaps(found, np.random.default_rng(own_spread.SEED))
    assert all(_keeps_the_graph(graph, order) for order in orders)


def test_a_symmetry_of_the_skeleton_keeps_each_atoms_hydrogens() -> None:
    # Imidazole's ring maps onto itself with N–H and N exchanged; that is no symmetry.
    imidazole = own_spread.bond_graph("c1c[nH]cn1")
    assert own_spread.skeleton_symmetries(imidazole).skeleton == ()
    nitrobenzene = own_spread.bond_graph("O=[N+]([O-])c1ccccc1")
    assert len(own_spread.skeleton_symmetries(nitrobenzene).skeleton) == 1


def _modes(wavenumbers: list[float], vectors: list[list[float]]) -> vib.Modes:
    return vib.Modes(np.array(wavenumbers), np.array(vectors, dtype=float).T)


def test_two_bands_that_mix_get_the_larger_of_the_pairs_shift_and_their_own() -> None:
    # Printed 1000 and 1002; a presentation shares their motion out 60 to 40 and puts them
    # at 1000 and 1006. As a pair, sorted against sorted, they moved by 0 and 4: 4.
    printed = _modes([1000.0, 1002.0], [[1.0, 0.0], [0.0, 1.0]])
    more, less = 0.6**0.5, 0.4**0.5
    # The mode at 1000 is mostly the band printed there: paired by motion the bands moved
    # by 0 and 4, and the band that stayed takes the pair's 4.
    kept = _modes([1000.0, 1006.0], [[more, less], [less, -more]])
    assert own_spread.band_shifts(kept, printed).tolist() == [4.0, 4.0]
    # The mode at 1006 is mostly the band printed at 1000: paired by motion the bands moved
    # by 6 and 2. The first keeps its own 6, larger than the pair's 4; the second takes 4.
    crossed = _modes([1000.0, 1006.0], [[less, more], [more, -less]])
    assert own_spread.band_shifts(crossed, printed).tolist() == [6.0, 4.0]
    # Half and half, the pairing may fall either way: no band gets less than the pair's 4.
    half = float(np.sqrt(0.5))
    even = _modes([1000.0, 1006.0], [[half, half], [half, -half]])
    assert own_spread.band_shifts(even, printed).tolist() in ([4.0, 4.0], [6.0, 4.0])


def _thinly_mixed(partner_cm: float, other_cm: float) -> vib.Modes:
    """Return six modes, one holding 0.6 of the first printed band and 0.08 of each other.

    The mode set is a reflection of the printed one, so it is orthonormal; the thin mode
    is the second by position and the first printed band's partner by motion.
    """
    thin = np.array([0.6**0.5, *[0.08**0.5] * 5])
    normal = np.eye(6)[0] - thin
    vectors = np.eye(6) - 2.0 * np.outer(normal, normal) / (normal @ normal)
    positions = np.array([other_cm, partner_cm, 1020.0, 1030.0, 1040.0, 1050.0])
    return vib.Modes(positions, vectors[:, [1, 0, 2, 3, 4, 5]])


def test_a_mixed_band_with_no_cluster_gets_the_larger_of_its_paired_and_its_rank_shift() -> None:
    printed = vib.Modes(np.array([1000.0, 1010.0, 1020.0, 1030.0, 1040.0, 1050.0]), np.eye(6))
    # Its partner by motion lies 0.5 away, the mode of its own rank 4 away.
    by_rank = own_spread.band_shifts(_thinly_mixed(1000.5, 996.0), printed)
    assert by_rank[0] == pytest.approx(4.0)
    # Its partner by motion lies 5 away, the mode of its own rank 1 away.
    by_motion = own_spread.band_shifts(_thinly_mixed(1005.0, 1001.0), printed)
    assert by_motion[0] == pytest.approx(5.0)


def test_two_bands_that_keep_their_motion_are_paired_by_it_not_by_rank() -> None:
    printed = _modes([1000.0, 1002.0], [[1.0, 0.0], [0.0, 1.0]])
    unmixed = _modes([1000.0, 1006.0], [[1.0, 0.0], [0.0, 1.0]])
    assert own_spread.band_shifts(unmixed, printed).tolist() == [0.0, 4.0]
    # The two cross: the lower mode of the presentation is the upper printed band.
    crossed = _modes([1001.0, 1003.0], [[0.0, 1.0], [1.0, 0.0]])
    assert own_spread.band_shifts(crossed, printed).tolist() == [3.0, 1.0]
    # A band that shares a little motion with an unmixed neighbour keeps its own shift.
    small = float(np.sqrt(0.05))
    large = float(np.sqrt(0.95))
    nearly = _modes([1000.0, 1006.0], [[large, small], [small, -large]])
    assert own_spread.band_shifts(nearly, printed).tolist() == [0.0, 4.0]


@dataclass(frozen=True)
class _Training:
    soft_mode_repair: bool = False


@dataclass
class _Networks:
    """Stand-in networks whose answer is a fixed function of the Hessian they read."""

    answer: str
    training: _Training = _Training()
    fails_at: int = 0
    failure: type[Exception] = RuntimeError
    calls: int = 0

    def corrections(self, molecule: Geometry, hessian: FloatArray) -> release.Corrections:
        self.calls += 1
        if self.calls == self.fails_at:
            msg = "the internal coordinates could not be built"
            raise self.failure(msg)
        # The other answer is stiffer along the laboratory's z axis: no rotation leaves it alone.
        laboratory_z = np.kron(np.eye(molecule.n_atoms), np.diag([0.0, 0.0, 1.0]))
        delta = np.asarray(0.01 * hessian if self.answer == "scaled" else 0.02 * laboratory_z)
        return release.Corrections(delta, (delta, delta))


def _measured(networks: _Networks) -> own_spread.OwnSpread:
    smiles, molecule, hessian = as_computed("benzonitrile")
    correction = 0.01 * hessian if networks.answer == "scaled" else np.zeros_like(hessian)
    return own_spread.measure(
        cast("release.Release", networks), smiles, molecule, _low(hessian), correction
    )


def test_networks_that_turn_with_the_molecule_move_no_band() -> None:
    found = _measured(_Networks("scaled"))
    assert max(found.spread_cm) < 1e-6


def test_networks_tied_to_the_laboratory_axes_move_bands_under_a_rotation_only() -> None:
    networks = _Networks("laboratory z")
    smiles, molecule, hessian = as_computed("benzonitrile")
    correction = networks.corrections(molecule, hessian).mean
    found = own_spread.measure(
        cast("release.Release", networks), smiles, molecule, _low(hessian), correction
    )
    assert max(found.spread_cm) > LINE_CM
    kinds = {change.name: change.kind for change in found.tried}
    moving = {
        name for name, spread in zip(found.change, found.spread_cm, strict=True) if spread > 1e-6
    }
    assert moving
    assert {kinds[name] for name in moving} <= {"rotation", "mirror image"}
    assert "the mirror image" not in moving


def test_a_presentation_the_networks_cannot_correct_withholds_the_licence_by_name() -> None:
    found = _measured(_Networks("scaled", fails_at=3))
    assert found.spread_cm == ()
    assert found.refusal == (
        "the networks could not correct the molecule given as rotation 3: RuntimeError: the "
        "internal coordinates could not be built"
    )
    n_bands = len(_bands("benzonitrile"))
    refused = _bands("benzonitrile", found)
    assert len(refused) == n_bands
    assert {band.licence_note for band in refused} == {certificate.UNCHECKED}
    assert [band.wavenumber_cm for band in refused] == [
        band.wavenumber_cm for band in _bands("benzonitrile")
    ]


def test_a_defect_in_the_check_is_raised_not_turned_into_a_refusal() -> None:
    with pytest.raises(TypeError, match="could not be built"):
        _measured(_Networks("scaled", fails_at=3, failure=TypeError))


def test_the_card_states_the_rule_and_keeps_the_record_as_context() -> None:
    text = card.to_markdown(_v14())
    assert (
        "- **A band's licence follows its own measured spread.** For every molecule, aromir's "
        "certificate has the networks correct the same molecule given in other ways: 8 random "
        "rotations, the mirror image and a rotated mirror image, 4 random listings of the atoms "
        "put into aromir's canonical order and swaps of atoms the bond graph cannot tell apart "
        "(all of them up to 12; of more, 12 symmetries of the heavy-atom skeleton and 6 "
        "exchanges of hydrogens), drawn with the seed 7."
    ) in text
    assert (
        "it is licensed only when its position lay no more than 3 cm⁻¹ from the printed one in "
        "every presentation tried: the distance from the printed presentation, not the range "
        "between presentations, and a sample, not a bound."
    ) in text
    assert "they describe the network and decide no licence." in text
    assert "11.20 cm⁻¹ on 3 molecule(s) with a linear group" in text
