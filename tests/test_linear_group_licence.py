"""The licence of a molecule with a linear group follows the release's invariance record.

Every band of such a molecule is not licensed while the record's largest spread for
molecules with a linear group, under a rotation, a mirror image or another xyz listing,
exceeds the licence line, or was not measured; the positions are printed all the same.
"""

from __future__ import annotations

from dataclasses import replace
from functools import cache
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from aromir import admission, card, certificate, geometry, lowlevel, release, response
from aromir import vibrations as vib

from support import random_hessian, stored_molecule

if TYPE_CHECKING:
    from aromir.molecule import FloatArray, Geometry

V14_REASON = (
    "the molecule has a linear group (an angle near 180°, as in a nitrile or an alkyne), and "
    "the measured spread under a rotation, a mirror image or another xyz listing of the atoms "
    "for molecules with one (11.20 cm⁻¹, from the release's invariance record) exceeds the "
    "licence line of 3 cm⁻¹, so the network's correction for this molecule is not licensed; "
    "the corrected positions are still printed"
)
V14_NOTE = (
    "the molecule has a linear group, and molecules with one moved by up to 11.20 cm⁻¹ under "
    "a rotation, a mirror image or another xyz listing, more than the licence line"
)
UNMEASURED_REASON = (
    "the molecule has a linear group (an angle near 180°, as in a nitrile or an alkyne), and "
    "the spread under a rotation, a mirror image or another xyz listing of the atoms was not "
    "measured for this release on a molecule with one, so the network's correction for this "
    "molecule is not licensed; the corrected positions are still printed"
)
DIPHENYLACETYLENE = "C(#Cc1ccccc1)c1ccccc1"


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


STORED_SMILES = {"benzonitrile": "N#Cc1ccccc1", "biphenyl": "c1ccc(cc1)-c1ccccc1"}
BUILT_SMILES = {"diphenylacetylene": DIPHENYLACETYLENE, "benzene": "c1ccccc1"}


def _molecule(name: str) -> tuple[str, Geometry, FloatArray, release.Corrections]:
    """Return a molecule, its Hessian and a correction.

    A stored molecule gets its B3LYP Hessian and v1.4's own correction. One without a stored
    Hessian gets aromir's start geometry, a random Hessian with real modes and no
    correction: v1.4's answer to a random Hessian can make a band imaginary, which the
    licence names before anything else, and the licence does not depend on the answer.
    """
    if name in STORED_SMILES:
        molecule, hessian = stored_molecule(name)
        return STORED_SMILES[name], molecule, hessian, _v14().corrections(molecule, hessian)
    start = geometry.start_from_smiles(BUILT_SMILES[name])
    hessian = vib.project_rigid_motions(random_hessian(start.n_atoms, 3), start)
    zero = np.zeros_like(hessian)
    none = release.Corrections(zero, tuple(zero for _ in _v14().networks))
    return BUILT_SMILES[name], start, hessian, none


def _record(name: str, loaded: release.Release) -> dict[str, Any]:
    """Return a molecule's certificate record."""
    smiles, molecule, hessian, corrections = _molecule(name)
    admitted = admission.admit(admission.read(smiles))
    return certificate.build(admitted, molecule, _low(hessian), corrections, loaded)


def _unruled_bands(name: str) -> list[certificate.Band]:
    """Return a molecule's bands on v1.4 as they would be without the linear-group rule."""
    _, molecule, hessian, corrections = _molecule(name)
    return certificate.bands(
        molecule,
        _low(hessian),
        corrections,
        _v14().evaluation,
        inside_training_sizes=True,
        withheld=None,
    )


def _with_spreads(with_group: float | None, n_with: int) -> release.Invariance:
    """Return a v1.4 invariance record whose rotation spread with a linear group is given."""
    recorded = _v14().invariance
    assert recorded is not None
    rotation = recorded.rotation.model_copy(
        update={"with_linear_group_cm": with_group, "n_with_linear_group": n_with}
    )
    others = {
        "with_linear_group_cm": min(2.5, with_group or 2.5),
        "n_with_linear_group": 3,
    }
    return recorded.model_copy(
        update={
            "rotation": rotation,
            "mirror_image": recorded.mirror_image.model_copy(update=others),
            "xyz_atom_order": recorded.xyz_atom_order.model_copy(update=others),
        }
    )


@pytest.mark.parametrize(
    "name",
    [
        "diphenylacetylene",
        # The rule is by group: benzonitrile's own turns move it less than the line (0.92
        # cm⁻¹ in test_releases_on_stored_hessians), the record's largest is 11.20.
        "benzonitrile",
    ],
)
def test_on_v14_no_band_of_a_molecule_with_a_linear_group_is_licensed_by_its_group(
    name: str,
) -> None:
    record = _record(name, _v14())
    assert record["molecule"]["linear_group"] is True
    assert record["bands"]
    assert all(not band["licensed"] for band in record["bands"])
    assert {band["licence_note"] for band in record["bands"]} == {V14_NOTE}
    assert f"{V14_REASON[0].upper()}{V14_REASON[1:]}." in record["statement"]
    assert "0 of " in record["statement"]
    text = certificate.to_markdown(record)
    assert text.count(f"| no: {V14_NOTE} |") == len(record["bands"])


@pytest.mark.parametrize("name", ["diphenylacetylene", "benzonitrile"])
def test_the_rule_leaves_the_printed_positions_as_they_are(name: str) -> None:
    printed = [band["wavenumber_cm"] for band in _record(name, _v14())["bands"]]
    assert printed == [band.wavenumber_cm for band in _unruled_bands(name)]


@pytest.mark.parametrize("name", ["biphenyl", "benzene"])
def test_a_molecule_without_a_linear_group_keeps_its_licence(name: str) -> None:
    record = _record(name, _v14())
    assert record["molecule"]["linear_group"] is False
    notes = [band["licence_note"] for band in record["bands"]]
    assert notes == [band.licence_note for band in _unruled_bands(name)]
    assert "linear group (an angle near 180°" not in record["statement"]
    assert any(band["licensed"] for band in record["bands"])


@pytest.mark.parametrize("spread", [2.5, 3.0])
def test_a_record_at_or_below_the_line_restores_the_licence(spread: float) -> None:
    within = replace(_v14(), invariance=_with_spreads(spread, 3))
    assert certificate.withheld_for_linear_group(within, linear_group=True) is None
    record = _record("benzonitrile", within)
    assert [band["licence_note"] for band in record["bands"]] == [
        band.licence_note for band in _unruled_bands("benzonitrile")
    ]
    assert all(band["licensed"] for band in record["bands"])
    assert (
        f"for this release it is {spread:.2f} cm⁻¹, so their bands keep the licence; the "
        "corrected positions are printed either way."
    ) in card.to_markdown(within)


def test_a_record_just_above_the_line_withholds_the_licence() -> None:
    above = replace(_v14(), invariance=_with_spreads(3.01, 3))
    withheld = certificate.withheld_for_linear_group(above, linear_group=True)
    assert withheld is not None
    assert withheld.spread_cm == pytest.approx(3.01)
    assert certificate.withheld_for_linear_group(above, linear_group=False) is None


@pytest.mark.parametrize("record_kind", ["no record", "no molecule with one rotated"])
def test_a_release_that_did_not_measure_a_linear_group_licenses_none(record_kind: str) -> None:
    invariance = None if record_kind == "no record" else _with_spreads(None, 0)
    unmeasured = replace(_v14(), invariance=invariance)
    record = _record("benzonitrile", unmeasured)
    assert all(not band["licensed"] for band in record["bands"])
    assert f"{UNMEASURED_REASON[0].upper()}{UNMEASURED_REASON[1:]}." in record["statement"]
    assert (
        "for this release it was not measured, so their bands are not licensed"
        in card.to_markdown(unmeasured)
    )
    benzene = _record("benzene", unmeasured)
    assert [band["licence_note"] for band in benzene["bands"]] == [
        band.licence_note for band in _unruled_bands("benzene")
    ]


def test_the_cards_state_the_rule_with_their_records_numbers() -> None:
    assert "for this release it is 11.20 cm⁻¹, so their bands are not licensed" in card.to_markdown(
        _v14()
    )
    v13 = release.load(release.bundled().parent / "v1.3")
    withheld = certificate.withheld_for_linear_group(v13, linear_group=True)
    assert withheld is not None
    assert withheld.spread_cm == 5.34
