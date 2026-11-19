"""What a certificate and a card may say: a band's own licence, and what the error leaves out."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from pydantic import ValidationError

from aromir import admission, card, certificate, families, geometry, lowlevel, release, response
from aromir import vibrations as vib
from aromir.errors import RefusedError
from aromir.network import graph
from aromir.network.inputs import InputSettings
from aromir.network.primitives import PrimitiveSettings

from support import TRAINING, bond_force_field, random_hessian, write_release

if TYPE_CHECKING:
    from pathlib import Path

    from aromir.molecule import FloatArray, Geometry


def _spread(
    with_group: float | None, n_with: int, without: float | None, n_without: int
) -> dict[str, Any]:
    return {
        "with_linear_group_cm": with_group,
        "n_with_linear_group": n_with,
        "without_linear_group_cm": without,
        "n_without_linear_group": n_without,
        "n_trials": 8,
    }


INVARIANCE = {
    "network_version": "1.3",
    "rotation": _spread(5.34, 3, 0.0, 4),
    "mirror_image": _spread(5.22, 3, 2.6, 4),
    "xyz_atom_order": _spread(5.27, 3, 3.07, 4),
    "source": "a test",
}
X_H_SPRING = {"C": 0.37, "O": 0.50, "N": 0.45}
"""Force constants, hartree/bohr², that put a C–H stretch near 3100 cm⁻¹ and an O–H or N–H
stretch well above it."""


@pytest.fixture(scope="module")
def loaded(tmp_path_factory: pytest.TempPathFactory) -> release.Release:
    return release.load(write_release(tmp_path_factory.mktemp("release")))


@pytest.fixture(scope="module")
def measured(loaded: release.Release) -> release.Release:
    return replace(loaded, invariance=release.Invariance.model_validate(INVARIANCE))


def _low(hessian: FloatArray) -> lowlevel.LowLevel:
    return lowlevel.LowLevel(
        energy=-1.0,
        hessian=hessian,
        polar_tensor=np.random.default_rng(1).normal(size=(3, len(hessian))),
        refinement=response.Refinement(1e-9, 1e-9, 0, 0),
        sum_rule_violation=1e-7,
        seconds=1.0,
    )


def _springs(start: Geometry) -> FloatArray:
    """Return a Hessian of springs along every bond, stiff along X–H, with no zero mode."""
    bonds = tuple(
        (atom, neighbour, X_H_SPRING[start.symbols[neighbour]])
        if start.symbols[atom] == "H"
        else (atom, neighbour, 0.6)
        for atom, neighbours in enumerate(graph.feature_bonds(start))
        for neighbour in neighbours
        if start.symbols[atom] == "H" or (atom < neighbour and start.symbols[neighbour] != "H")
    )
    floor = vib.project_rigid_motions(0.02 * np.eye(3 * start.n_atoms), start)
    return bond_force_field(start, bonds) + floor


def _record(smiles: str, hessian_of: str, loaded: release.Release) -> dict[str, Any]:
    start = geometry.start_from_smiles(smiles)
    hessian = _springs(start) if hessian_of == "springs" else random_hessian(start.n_atoms, 0)
    admitted = admission.admit(admission.read(smiles))
    zero = np.zeros_like(hessian)
    corrections = release.Corrections(zero, tuple(zero for _ in loaded.networks))
    return certificate.build(admitted, start, _low(hessian), corrections, loaded)


def _modes_on(start: Geometry, shares: dict[int, float]) -> vib.Modes:
    """Return one mode that moves the given atoms with the given shares of its motion."""
    vector = np.zeros((3 * start.n_atoms, 1))
    for atom, share in shares.items():
        vector[3 * atom, 0] = np.sqrt(share)
    return vib.Modes(wavenumbers=np.array([3300.0]), vectors=vector)


def _hydrogen_on(start: Geometry, element: str) -> int:
    heavy = [k for k, symbol in enumerate(start.symbols) if symbol != "H"]
    for atom, symbol in enumerate(start.symbols):
        if symbol != "H":
            continue
        distances = [
            float(np.linalg.norm(start.coords_bohr[k] - start.coords_bohr[atom])) for k in heavy
        ]
        if start.symbols[heavy[int(np.argmin(distances))]] == element:
            return atom
    raise AssertionError(element)


@pytest.mark.parametrize("on_oxygen", [0.0, 1.0, 0.45, 0.05])
def test_the_share_of_hydrogen_motion_off_carbon(on_oxygen: float) -> None:
    phenol = geometry.start_from_smiles("Oc1ccccc1")
    shares = {_hydrogen_on(phenol, "O"): on_oxygen, _hydrogen_on(phenol, "C"): 1.0 - on_oxygen}
    share = families.hydrogen_motion_off_carbon(phenol, _modes_on(phenol, shares))
    assert share.tolist() == pytest.approx([on_oxygen])


def test_a_mode_that_moves_no_hydrogen_has_no_share_off_carbon() -> None:
    phenol = geometry.start_from_smiles("Oc1ccccc1")
    heavy_only = _modes_on(phenol, {phenol.symbols.index("O"): 1.0})
    assert families.hydrogen_motion_off_carbon(phenol, heavy_only).tolist() == [0.0]


@pytest.mark.parametrize(
    ("smiles", "n_stretches", "n_off_carbon"),
    [("Oc1ccccc1", 6, 1), ("Nc1ccccc1", 7, 2), ("c1ccccc1", 6, 0)],
)
def test_a_stretch_on_oxygen_or_nitrogen_is_not_licensed_as_a_ch_stretch(
    smiles: str, n_stretches: int, n_off_carbon: int, loaded: release.Release
) -> None:
    record = _record(smiles, "springs", loaded)
    stretches = [band for band in record["bands"] if band["family"] == "CH-stretch"]
    assert len(stretches) == n_stretches
    refused = [band for band in stretches if not band["licensed"]]
    assert len(refused) == n_off_carbon
    note = "100% of its hydrogen motion is on N–H, O–H or S–H, which the rule counts as CH-stretch"
    assert {band["licence_note"] for band in refused} <= {note}
    highest = sorted(band["wavenumber_cm"] for band in stretches)[n_stretches - n_off_carbon :]
    assert sorted(band["wavenumber_cm"] for band in refused) == highest
    text = certificate.to_markdown(record)
    assert text.count(f"| CH-stretch | no: {note} |") == n_off_carbon
    assert text.count("| CH-stretch | yes |") == n_stretches - n_off_carbon


def _note(mode: certificate.Mode, loaded: release.Release) -> str:
    return certificate.licence_note(mode, loaded.evaluation, inside_training_sizes=True)


def test_a_mixed_stretch_and_a_wag_off_carbon_fail_the_licence(loaded: release.Release) -> None:
    assert _note(certificate.Mode("CH-stretch", 100.0, 3100.0, 3099.0, 0.05), loaded) == ""
    mixed = _note(certificate.Mode("CH-stretch", 100.0, 3100.0, 3099.0, 0.45), loaded)
    assert mixed.startswith("45% of its hydrogen motion")
    wag = _note(certificate.Mode("CH-oop", 100.0, 800.0, 799.0, 0.9), loaded)
    assert wag.endswith("which the rule counts as CH-oop")
    assert _note(certificate.Mode("ring-ip", 100.0, 1200.0, 1199.0, 0.9), loaded) == ""


def test_a_position_just_below_zero_is_imaginary_though_it_rounds_to_zero(
    loaded: release.Release,
) -> None:
    assert (
        _note(certificate.Mode("ring-ip", 100.0, -0.04, -0.04, 0.0), loaded)
        == certificate.IMAGINARY
    )
    for_a_seed = _note(certificate.Mode("ring-ip", 100.0, 12.0, -3.0, 0.0), loaded)
    assert for_a_seed == certificate.IMAGINARY_FOR_A_SEED


def test_a_band_of_an_imaginary_low_level_mode_is_a_flat_coordinate_whatever_its_position(
    loaded: release.Release,
) -> None:
    # The command ends with 'unphysical' for an imaginary corrected band; a flat coordinate
    # is named before that, so a spectrum printed at one does not end that way for it.
    for position in (-8.0, 15.0):
        flat = _note(certificate.Mode("other", -12.0, position, position, 0.0), loaded)
        assert flat == certificate.FLAT_COORDINATE


def test_every_family_has_its_rule_in_words(loaded: release.Release) -> None:
    assert tuple(families.RULES) == families.FAMILIES
    text = certificate.to_markdown(_record("c1ccccc1", "random", loaded))
    for family, rule in families.RULES.items():
        assert f"- {family}: {rule}." in text


def test_the_certificate_says_what_the_error_does_not_cover(measured: release.Release) -> None:
    nitrile = _record("N#Cc1ccccc1", "random", measured)
    assert nitrile["molecule"]["linear_group"] is True
    assert (
        "a rigid rotation of the molecule moved a position by up to 5.34 cm⁻¹, the largest seen "
        "on 3 molecule(s) with a linear group; that is more than the licence line of 3 cm⁻¹"
    ) in nitrile["not_in_the_error"]
    assert "positions up to 5.34 cm⁻¹ away, more than the licence line" in nitrile["statement"]
    benzene = _record("c1ccccc1", "random", measured)
    assert benzene["molecule"]["linear_group"] is False
    assert (
        "a rigid rotation of the molecule moved a position by up to 0.00 cm⁻¹, the largest seen "
        "on 4 molecule(s) without a linear group"
    ) in benzene["not_in_the_error"]
    assert (
        "another listing of the same atoms in an xyz file moved a position by up to 3.07 cm⁻¹, "
        "the largest seen on 4 molecule(s) without a linear group; that is more than the licence "
        "line of 3 cm⁻¹"
    ) in benzene["not_in_the_error"]
    assert "positions up to 3.07 cm⁻¹ away" in benzene["statement"]
    text = certificate.to_markdown(benzene)
    assert "## What the measured error does not cover" in text
    assert f"- {certificate.HEIGHTS_NOTE[0].upper()}{certificate.HEIGHTS_NOTE[1:]}." in text
    assert "One conformer was computed" in text
    assert json.loads(json.dumps(benzene)) == benzene


def test_a_spread_no_molecule_of_the_kind_was_measured_on_is_said_to_be_missing(
    loaded: release.Release,
) -> None:
    partial = INVARIANCE | {"rotation": _spread(None, 0, 0.0, 4)}
    release_ = replace(loaded, invariance=release.Invariance.model_validate(partial))
    nitrile = _record("N#Cc1ccccc1", "random", release_)
    assert not any("rigid rotation" in line for line in nitrile["not_in_the_error"])
    assert (
        "the spread under a rotation, a reflection or another listing of an xyz file's atoms "
        "was not measured on a molecule with a linear group"
    ) in nitrile["not_in_the_error"]
    assert "5.34" not in card.to_markdown(release_)


@pytest.mark.parametrize(
    "spread",
    [
        _spread(-1.0, 3, 0.0, 4),
        _spread(float("inf"), 3, 0.0, 4),
        _spread(None, 3, 0.0, 4),
        _spread(5.0, 0, 0.0, 4),
        _spread(5.0, 3, 0.0, 4) | {"extra": 1},
        _spread(None, 0, None, 0),
    ],
)
def test_a_spread_that_cannot_be_a_measurement_is_refused(spread: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        release.MeasuredSpread.model_validate(spread)


def test_the_atom_order_spread_of_the_evaluation_is_on_the_certificate(
    loaded: release.Release,
) -> None:
    spread = release.AtomOrderSpread(max_cm=10.73, n_molecules=3, n_orders=5, source="x.json")
    evaluation = loaded.evaluation.model_copy(update={"atom_order_spread": spread})
    record = _record("c1ccccc1", "random", replace(loaded, evaluation=evaluation))
    assert (
        "another atom order of the same molecule moved a position by up to 10.73 cm⁻¹, the "
        "largest seen on 3 molecule(s) in arbitrary orders; that is more than the licence line "
        "of 3 cm⁻¹"
    ) in record["not_in_the_error"]
    assert "positions up to 10.73 cm⁻¹ away" in record["statement"]


def test_a_release_without_the_measurements_says_so(loaded: release.Release) -> None:
    record = _record("c1ccccc1", "random", loaded)
    assert loaded.invariance is None
    assert (
        "the spread between atom orders was not measured for this release"
        in (record["not_in_the_error"])
    )
    assert any("a reflection" in line for line in record["not_in_the_error"])
    assert "Not in that error" not in record["statement"]
    assert "by how much was not measured for this release" in card.to_markdown(loaded)


def test_a_twisted_molecule_is_told_it_has_no_out_of_plane_family(
    loaded: release.Release,
) -> None:
    biphenyl = _record("c1ccc(cc1)-c1ccccc1", "random", loaded)
    assert biphenyl["molecule"]["planar"] is False
    assert certificate.NOT_PLANAR_NOTE in biphenyl["not_in_the_error"]
    benzene = _record("c1ccccc1", "random", loaded)
    assert certificate.NOT_PLANAR_NOTE not in benzene["not_in_the_error"]


def test_stereo_the_smiles_leaves_open_is_on_the_certificate(loaded: release.Release) -> None:
    stilbene = _record("C(=Cc1ccccc1)c1ccccc1", "random", loaded)
    assert stilbene["molecule"]["unassigned_stereo"] == ["1 double bond"]
    assert any(
        line.startswith("the input leaves 1 double bond unassigned: one isomer was computed")
        for line in stilbene["not_in_the_error"]
    )
    assigned = _record("C(=C/c1ccccc1)\\c1ccccc1", "random", loaded)
    assert assigned["molecule"]["unassigned_stereo"] == []
    assert not any("unassigned" in line for line in assigned["not_in_the_error"])
    centre = admission.admit(admission.read("CC(N)c1ccccc1"))
    assert centre.unassigned_stereo == ("1 stereocentre",)


def test_a_stereoisomer_of_a_training_molecule_is_not_called_unseen() -> None:
    # Written as admission writes trans-stilbene (its explicit hydrogens turn the marks).
    trained = TRAINING | {"trained_smiles": ["C(=C\\c1ccccc1)/c1ccccc1"]}
    training = release.Training.model_validate(trained)
    assert "a stereoisomer of it" in certificate.acquaintance("C(=Cc1ccccc1)c1ccccc1", training)
    assert "has not seen it" in certificate.acquaintance("c1ccncc1", training)


def test_the_card_states_the_measured_spreads_and_claims_no_more(
    measured: release.Release,
) -> None:
    text = card.to_markdown(measured)
    assert (
        "- **A rigid rotation of the molecule** moved band positions by up to 5.34 cm⁻¹ on 3 "
        "molecule(s) with a linear group and 0.00 cm⁻¹ on 4 molecule(s) without a linear group "
        "(8 trial(s) per molecule). These are the largest seen, not bounds."
    ) in text
    assert "- **The molecule's mirror image** moved band positions by up to 5.22 cm⁻¹" in text
    assert "and a dihedral enters with its sign; the test sets' errors" in text
    assert "Source: a test." in text
    assert "by construction" not in text
    assert "one canonical atom order" not in text
    assert "keep the order of an xyz file" in text


def test_an_order_free_release_does_not_claim_more_than_its_measurements(
    measured: release.Release,
) -> None:
    order_free = InputSettings(
        primitives=PrimitiveSettings(angle_choice="canonical", dihedral_encoding="cos")
    )
    recipes = tuple(recipe.model_copy(update={"inputs": order_free}) for recipe in measured.recipes)
    text = card.to_markdown(replace(measured, recipes=recipes))
    assert "the measurements say how far it is" in text
    assert "does not depend on how" not in text
    assert "fixed axis in space; the test sets' errors" in text
    assert "enters with its sign" not in text


def test_a_release_loads_its_invariance_record(tmp_path: Path) -> None:
    folder = write_release(tmp_path / "release")
    assert release.load(folder).invariance is None
    (folder / release.INVARIANCE_FILE).write_text(json.dumps(INVARIANCE), encoding="utf-8")
    invariance = release.load(folder).invariance
    assert invariance is not None
    assert invariance.rotation.of(linear_group=True) == (5.34, 3)
    assert invariance.rotation.of(linear_group=False) == (0.0, 4)
    assert list(invariance.spreads()) == [
        "a rigid rotation of the molecule",
        "the molecule's mirror image",
        "another listing of the same atoms in an xyz file",
    ]


def test_an_invariance_record_of_another_version_is_refused(tmp_path: Path) -> None:
    folder = write_release(tmp_path / "release")
    other = INVARIANCE | {"network_version": "1.4"}
    (folder / release.INVARIANCE_FILE).write_text(json.dumps(other), encoding="utf-8")
    with pytest.raises(RefusedError, match="different network versions"):
        release.load(folder)


def test_release_v13_states_its_measured_spreads() -> None:
    bundled = release.load(release.bundled().parent / "v1.3")
    assert bundled.invariance is not None
    assert bundled.invariance.rotation.of(linear_group=True) == (5.34, 3)
    assert bundled.invariance.rotation.of(linear_group=False) == (0.0, 4)
    spreads = certificate.spreads_cm(bundled, linear_group=True)
    assert spreads[certificate.ATOM_ORDER] == (10.73, 3)
    assert max(value for value, _ in spreads.values()) > bundled.evaluation.licence_line_cm
