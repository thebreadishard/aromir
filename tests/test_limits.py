"""What a certificate and a card may say: a band's own licence, and what the error leaves out."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from aromir import admission, card, certificate, families, geometry, lowlevel, release, response
from aromir import vibrations as vib
from aromir.errors import RefusedError
from aromir.network import graph

from support import bond_force_field, random_hessian, write_release

if TYPE_CHECKING:
    from pathlib import Path

    from aromir.molecule import FloatArray, Geometry

INVARIANCE = {
    "network_version": "1.3",
    "rotation": {
        "with_linear_group_cm": 5.34,
        "without_linear_group_cm": 0.0,
        "n_molecules": 7,
        "n_trials": 8,
    },
    "mirror_image": {
        "with_linear_group_cm": 5.22,
        "without_linear_group_cm": 2.6,
        "n_molecules": 7,
        "n_trials": 1,
    },
    "xyz_atom_order": {
        "with_linear_group_cm": 5.27,
        "without_linear_group_cm": 3.07,
        "n_molecules": 7,
        "n_trials": 12,
    },
    "source": "a test",
}
X_H_SPRING = {"C": 0.37, "O": 0.50}
"""Force constants, hartree/bohr², that put a C–H stretch near 3100 and an O–H near 3600 cm⁻¹."""


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
        (atom, neighbour, X_H_SPRING.get(start.symbols[neighbour], 0.6))
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
    admitted = admission.admit(admission.parse_smiles(smiles))
    zero = np.zeros_like(hessian)
    corrections = release.Corrections(zero, tuple(zero for _ in loaded.networks))
    return certificate.build(admitted, start, _low(hessian), corrections, loaded)


def test_a_mode_is_carried_by_the_element_its_moving_hydrogens_sit_on() -> None:
    phenol = geometry.start_from_smiles("Oc1ccccc1")
    hydrogens = [k for k, symbol in enumerate(phenol.symbols) if symbol == "H"]
    oxygen = phenol.symbols.index("O")
    on_oxygen = min(
        hydrogens,
        key=lambda k: float(np.linalg.norm(phenol.coords_bohr[k] - phenol.coords_bohr[oxygen])),
    )
    on_carbon = next(k for k in hydrogens if k != on_oxygen)
    vectors = np.zeros((3 * phenol.n_atoms, 2))
    vectors[3 * on_oxygen, 0] = 1.0
    vectors[3 * on_carbon + 1, 1] = 1.0
    modes = vib.Modes(wavenumbers=np.array([3700.0, 3100.0]), vectors=vectors)
    assert families.hydrogen_carriers(phenol, modes) == ["O", "C"]


def test_an_oh_stretch_is_not_licensed_as_a_ch_stretch(loaded: release.Release) -> None:
    record = _record("Oc1ccccc1", "springs", loaded)
    stretches = [band for band in record["bands"] if band["family"] == "CH-stretch"]
    assert len(stretches) == 6
    refused = [band for band in stretches if not band["licensed"]]
    assert len(refused) == 1
    assert refused[0]["licence_note"] == "an O–H stretch, which the rule counts as CH-stretch"
    assert refused[0]["wavenumber_cm"] == max(band["wavenumber_cm"] for band in stretches)
    text = certificate.to_markdown(record)
    assert "| CH-stretch | no: an O–H stretch, which the rule counts as CH-stretch |" in text
    assert text.count("| CH-stretch | yes |") == 5


def test_every_family_has_its_rule_in_words(loaded: release.Release) -> None:
    assert tuple(families.RULES) == families.FAMILIES
    text = certificate.to_markdown(_record("c1ccccc1", "random", loaded))
    for family, rule in families.RULES.items():
        assert f"- {family}: {rule}." in text


def test_the_certificate_says_what_the_error_does_not_cover(measured: release.Release) -> None:
    nitrile = _record("N#Cc1ccccc1", "random", measured)
    assert nitrile["molecule"]["linear_group"] is True
    assert (
        "a rigid rotation of the molecule moves a position by up to 5.34 cm⁻¹ on molecules "
        "with a linear group, more than the licence line of 3 cm⁻¹"
    ) in nitrile["not_in_the_error"]
    assert "positions up to 5.34 cm⁻¹ away, more than the licence line" in nitrile["statement"]
    benzene = _record("c1ccccc1", "random", measured)
    assert benzene["molecule"]["linear_group"] is False
    assert (
        "a rigid rotation of the molecule moves a position by up to 0.00 cm⁻¹ on molecules "
        "without a linear group"
    ) in benzene["not_in_the_error"]
    assert (
        "another listing of the same atoms in an xyz file moves a position by up to 3.07 cm⁻¹ "
        "on molecules without a linear group, more than the licence line of 3 cm⁻¹"
    ) in benzene["not_in_the_error"]
    assert "positions up to 3.07 cm⁻¹ away" in benzene["statement"]
    text = certificate.to_markdown(benzene)
    assert "## What the measured error does not cover" in text
    assert f"- {certificate.HEIGHTS_NOTE[0].upper()}{certificate.HEIGHTS_NOTE[1:]}." in text
    assert "One conformer was computed" in text
    assert json.loads(json.dumps(benzene)) == benzene


def test_the_atom_order_spread_of_the_evaluation_is_on_the_certificate(
    loaded: release.Release,
) -> None:
    spread = release.AtomOrderSpread(max_cm=10.73, n_molecules=3, n_orders=5, source="x.json")
    evaluation = loaded.evaluation.model_copy(update={"atom_order_spread": spread})
    record = _record("c1ccccc1", "random", replace(loaded, evaluation=evaluation))
    assert (
        f"{certificate.ATOM_ORDER} moves a position by up to 10.73 cm⁻¹, more than the licence "
        "line of 3 cm⁻¹"
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


def test_the_card_states_the_measured_spreads_and_claims_no_more(
    measured: release.Release,
) -> None:
    text = card.to_markdown(measured)
    assert (
        "- **A rigid rotation of the molecule** moves band positions by up to 5.34 cm⁻¹ in a "
        "molecule with a linear group (a nitrile, an alkyne) and 0.00 cm⁻¹ in one without "
        "(7 molecules, 8 each)."
    ) in text
    assert "- **The molecule's mirror image** moves band positions by up to 5.22 cm⁻¹" in text
    assert "Source: a test." in text
    assert "by construction" not in text
    assert "one canonical atom order" not in text
    assert "keep the order of an xyz file" in text


def test_a_release_loads_its_invariance_record(tmp_path: Path) -> None:
    folder = write_release(tmp_path / "release")
    assert release.load(folder).invariance is None
    (folder / release.INVARIANCE_FILE).write_text(json.dumps(INVARIANCE), encoding="utf-8")
    invariance = release.load(folder).invariance
    assert invariance is not None
    assert invariance.rotation.of(linear_group=True) == 5.34
    assert invariance.rotation.of(linear_group=False) == 0.0
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


def test_the_bundled_release_states_its_measured_spreads() -> None:
    bundled = release.load(release.bundled())
    assert bundled.invariance is not None
    assert bundled.invariance.rotation.with_linear_group_cm == 5.34
    assert bundled.invariance.rotation.without_linear_group_cm == 0.0
    spreads = certificate.spreads_cm(bundled, linear_group=True)
    assert spreads[certificate.ATOM_ORDER] == 10.73
    assert max(spreads.values()) > bundled.evaluation.licence_line_cm
