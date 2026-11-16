"""Admission: every reason to refuse, and what an admitted molecule is called."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from aromir import admission
from aromir.errors import RefusedError

from support import BENZENE_XYZ

if TYPE_CHECKING:
    from pathlib import Path


def test_an_aromatic_molecule_is_admitted_under_its_canonical_name() -> None:
    admitted = admission.admit(admission.parse_smiles("C1=CC=CC=C1"))
    assert admitted == admission.Admission(smiles="c1ccccc1", formula="C6H6", n_atoms=12)


def test_two_writings_get_one_name() -> None:
    first = admission.admit(admission.parse_smiles("c1ccc2ccccc2c1"))
    second = admission.admit(admission.parse_smiles("c1cccc2c1cccc2"))
    assert first == second


@pytest.mark.parametrize(
    ("smiles", "reason"),
    [
        ("CCO", "no aromatic ring"),
        ("c1ccccc1[O-]", "net charge"),
        ("c1ccccc1[CH2]", "unpaired electrons"),
        ("c1ccccc1Br", "element(s) Br outside"),
        ("c1ccccc1.O", "2 separate molecules"),
    ],
)
def test_a_molecule_outside_the_domain_is_refused(smiles: str, reason: str) -> None:
    with pytest.raises(RefusedError, match=reason.replace("(", r"\(").replace(")", r"\)")):
        admission.admit(admission.parse_smiles(smiles))


def test_every_reason_is_reported_at_once() -> None:
    with pytest.raises(RefusedError) as refusal:
        admission.admit(admission.parse_smiles("[O-]CCBr"))
    assert len(refusal.value.reasons) == 3


def test_a_string_that_is_no_smiles_is_refused() -> None:
    with pytest.raises(RefusedError, match="does not parse as a SMILES"):
        admission.parse_smiles("c1ccccc")


def test_an_xyz_file_is_read_with_its_bonds(tmp_path: Path) -> None:
    path = tmp_path / "benzene.xyz"
    path.write_text(BENZENE_XYZ, encoding="utf-8")
    assert admission.admit(admission.parse_xyz(path)).smiles == "c1ccccc1"


def test_an_unreadable_xyz_file_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "broken.xyz"
    path.write_text("this is not an xyz file\n", encoding="utf-8")
    with pytest.raises(RefusedError, match="does not parse as an xyz file"):
        admission.parse_xyz(path)


def test_an_xyz_without_a_neutral_bonding_pattern_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "ammonium.xyz"
    path.write_text(
        "5\nNH4+\nN 0 0 0\nH 0.59 0.59 0.59\nH -0.59 -0.59 0.59\n"
        "H -0.59 0.59 -0.59\nH 0.59 -0.59 -0.59\n",
        encoding="utf-8",
    )
    with pytest.raises(RefusedError, match="no neutral bonding pattern"):
        admission.parse_xyz(path)
