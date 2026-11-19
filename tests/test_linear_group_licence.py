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


def _record_with(
    *, linear_group: bool, rotation: float | None, mirror: float | None, listing: float | None
) -> release.Invariance:
    """Return v1.4's invariance record with one group's three spreads replaced."""
    recorded = _v14().invariance
    assert recorded is not None
    group = "with_linear_group" if linear_group else "without_linear_group"

    def spread(old: release.MeasuredSpread, value: float | None) -> release.MeasuredSpread:
        counts = 0 if value is None else 3
        return old.model_copy(update={f"{group}_cm": value, f"n_{group}": counts})

    return recorded.model_copy(
        update={
            "rotation": spread(recorded.rotation, rotation),
            "mirror_image": spread(recorded.mirror_image, mirror),
            "xyz_atom_order": spread(recorded.xyz_atom_order, listing),
        }
    )


def _within(value: float | None, *, linear_group: bool = True) -> release.Release:
    """Return v1.4 with one group's three spreads at one value."""
    invariance = _record_with(
        linear_group=linear_group, rotation=value, mirror=value, listing=value
    )
    return replace(_v14(), invariance=invariance)


def _capital(sentence: str) -> str:
    return f"{sentence[0].upper()}{sentence[1:]}."


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
    assert _capital(V14_REASON) in record["statement"]
    assert "0 of " in record["statement"]
    assert (
        "molecules with a linear group moved by no more than the licence line under a "
        "rotation, a mirror image or another xyz listing of the atoms in the release's "
        "invariance record"
    ) in record["statement"]
    text = certificate.to_markdown(record)
    assert text.count(f"| no: {V14_NOTE} |") == len(record["bands"])


@pytest.mark.parametrize("name", ["diphenylacetylene", "benzonitrile"])
def test_the_rule_leaves_the_printed_positions_as_they_are(name: str) -> None:
    printed = [band["wavenumber_cm"] for band in _record(name, _v14())["bands"]]
    assert printed == [band.wavenumber_cm for band in _unruled_bands(name)]


@pytest.mark.parametrize("name", ["biphenyl", "benzene"])
def test_on_v14_a_molecule_without_a_linear_group_keeps_its_licence(name: str) -> None:
    # v1.4's largest spread without a linear group is 2.85 cm⁻¹, within the line.
    record = _record(name, _v14())
    assert record["molecule"]["linear_group"] is False
    notes = [band["licence_note"] for band in record["bands"]]
    assert notes == [band.licence_note for band in _unruled_bands(name)]
    assert "the network's correction for this molecule is not licensed" not in record["statement"]
    assert "molecules without a linear group moved by no more than" in record["statement"]
    assert any(band["licensed"] for band in record["bands"])


@pytest.mark.parametrize("spread", [2.5, 3.0])
def test_a_record_at_or_below_the_line_restores_the_licence(spread: float) -> None:
    within = _within(spread)
    assert certificate.withheld_by_spread(within, linear_group=True) is None
    record = _record("benzonitrile", within)
    assert [band["licence_note"] for band in record["bands"]] == [
        band.licence_note for band in _unruled_bands("benzonitrile")
    ]
    assert all(band["licensed"] for band in record["bands"])
    assert (
        f"for this release it is {spread:.2f} cm⁻¹ with a linear group, so their bands keep "
        "the licence and 2.85 cm⁻¹ without a linear group, so their bands keep the licence; "
        "the corrected positions are printed either way."
    ) in card.to_markdown(within)


def test_a_record_just_above_the_line_withholds_the_licence() -> None:
    above = _within(3.01)
    withheld = certificate.withheld_by_spread(above, linear_group=True)
    assert withheld is not None
    assert withheld.spread_cm == pytest.approx(3.01)
    assert certificate.withheld_by_spread(above, linear_group=False) is None
    record = _record("benzonitrile", above)
    assert all(not band["licensed"] for band in record["bands"])
    assert {band["licence_note"] for band in record["bands"]} == {withheld.note}


@pytest.mark.parametrize(
    ("mirror", "listing", "change"),
    [(3.5, 2.5, "a mirror image"), (2.5, 3.5, "another xyz listing")],
)
@pytest.mark.parametrize("linear_group", [True, False], ids=["with", "without"])
def test_a_change_other_than_the_rotation_above_the_line_withholds_the_licence(
    *, mirror: float, listing: float, change: str, linear_group: bool
) -> None:
    invariance = _record_with(
        linear_group=linear_group, rotation=2.5, mirror=mirror, listing=listing
    )
    above = replace(_v14(), invariance=invariance)
    withheld = certificate.withheld_by_spread(above, linear_group=linear_group)
    assert withheld is not None
    assert withheld.spread_cm == 3.5
    name = "benzonitrile" if linear_group else "biphenyl"
    record = _record(name, above)
    assert all(not band["licensed"] for band in record["bands"])
    if not linear_group:
        assert withheld.note == (
            f"molecules without a linear group moved by up to 3.50 cm⁻¹ under {change}, more "
            "than the licence line"
        )


@pytest.mark.parametrize("record_kind", ["no record", "no molecule with one rotated"])
def test_a_release_that_did_not_measure_a_linear_group_licenses_none(record_kind: str) -> None:
    invariance = (
        None
        if record_kind == "no record"
        else _record_with(linear_group=True, rotation=None, mirror=2.5, listing=2.5)
    )
    unmeasured = replace(_v14(), invariance=invariance)
    record = _record("benzonitrile", unmeasured)
    assert all(not band["licensed"] for band in record["bands"])
    assert _capital(UNMEASURED_REASON) in record["statement"]
    assert "with a linear group, so their bands are not licensed" in card.to_markdown(unmeasured)


@pytest.mark.parametrize("record_kind", ["no record", "no molecule without one listed"])
def test_a_molecule_without_a_linear_group_keeps_its_licence_where_it_was_not_measured(
    record_kind: str,
) -> None:
    invariance = (
        None
        if record_kind == "no record"
        else _record_with(linear_group=False, rotation=0.0, mirror=2.5, listing=None)
    )
    unmeasured = replace(_v14(), invariance=invariance)
    assert certificate.withheld_by_spread(unmeasured, linear_group=False) is None
    benzene = _record("benzene", unmeasured)
    assert [band["licence_note"] for band in benzene["bands"]] == [
        band.licence_note for band in _unruled_bands("benzene")
    ]
    if invariance is None:
        assert (
            "for this release it is not measured with a linear group, so their bands are not "
            "licensed and not measured without a linear group, so their bands keep the licence"
        ) in card.to_markdown(unmeasured)


def test_on_v13_a_molecule_without_a_linear_group_is_not_licensed_by_its_group() -> None:
    v13 = release.load(release.bundled().parent / "v1.3")
    record = _record("benzene", v13)
    assert record["molecule"]["linear_group"] is False
    note = (
        "molecules without a linear group moved by up to 3.07 cm⁻¹ under another xyz "
        "listing, more than the licence line"
    )
    assert {band["licence_note"] for band in record["bands"]} == {note}
    reason = (
        "the molecule has no linear group, and the measured spread under another xyz listing "
        "for molecules without one (3.07 cm⁻¹, from the release's invariance record) exceeds "
        "the licence line of 3 cm⁻¹, so the network's correction for this molecule is not "
        "licensed; the corrected positions are still printed"
    )
    assert _capital(reason) in record["statement"]
    assert (
        "for this release it is 5.34 cm⁻¹ with a linear group, so their bands are not "
        "licensed and 3.07 cm⁻¹ without a linear group, so their bands are not licensed"
    ) in card.to_markdown(v13)


def test_the_v14_card_states_the_rule_with_its_records_numbers() -> None:
    assert (
        "for this release it is 11.20 cm⁻¹ with a linear group, so their bands are not "
        "licensed and 2.85 cm⁻¹ without a linear group, so their bands keep the licence"
    ) in card.to_markdown(_v14())


def test_the_rule_reads_every_change_of_the_record(monkeypatch: pytest.MonkeyPatch) -> None:
    recorded = _v14().invariance
    assert recorded is not None
    assert list(certificate.CHANGE_PHRASES) == list(recorded.spreads())
    four = recorded.spreads() | {"a fourth change": recorded.rotation}
    monkeypatch.setattr(release.Invariance, "spreads", lambda _: four)
    with pytest.raises(ValueError, match="a fourth change"):
        certificate.group_spread(_v14(), linear_group=True)
