"""The training chemistry: charge separation and aromatic ring sizes the training lists hold."""

from __future__ import annotations

import pytest

from aromir import cli, domain, release

from support import ZWITTERION


@pytest.fixture(scope="module")
def trained() -> list[str]:
    return release.load(release.bundled()).training.trained_smiles


def test_the_bundled_training_list_holds_nitro_groups_an_n_oxide_and_rings_of_4_to_7(
    trained: list[str],
) -> None:
    # Three kinds of charged atom: a nitro group's N+ and O-, and an aromatic N-oxide's n+
    # (whose O- is of the nitro group's kind: a single bond to N+).
    kinds = {site for smiles in trained for site in domain.charge_sites(smiles)}
    assert {(element, charge) for element, charge, *_ in kinds} == {("N", 1), ("O", -1)}
    assert len(kinds) == 3
    sizes = set().union(*(domain.aromatic_ring_sizes(smiles) for smiles in trained))
    assert sizes == {4, 5, 6, 7}
    assert all(domain.outside_training_chemistry(smiles, trained) == () for smiles in trained)


def test_a_zwitterion_is_outside_the_training_chemistry(trained: list[str]) -> None:
    caveats = domain.outside_training_chemistry("[NH3+]c1ccccc1C(=O)[O-]", trained)
    assert caveats == (
        (
            "charge separation of a kind no training molecule holds: C[O-], c[NH3+] (the "
            "charged atoms of the training molecules: [n+][O-], c[N+](=O)[O-], c[n+]([O-])c)"
        ),
    )


@pytest.mark.parametrize("smiles", ["[N-]=[N+]=Nc1ccccc1", "C[S+]([O-])c1ccccc1"])
def test_an_azide_or_a_sulfoxide_is_outside_a_list_of_nitro_groups(
    smiles: str, trained: list[str]
) -> None:
    (caveat,) = domain.outside_training_chemistry(smiles, trained)
    assert caveat.startswith("charge separation of a kind no training molecule holds")


def test_tropone_is_inside_a_list_with_azulenes(trained: list[str]) -> None:
    # RDKit writes azulene's shared bond as single: the ring counts by its atoms.
    assert domain.aromatic_ring_sizes("c1ccc2cccc2cc1") == {5, 7}
    assert domain.outside_training_chemistry("O=c1cccccc1", trained) == ()


def test_a_seven_membered_ring_is_outside_a_list_without_one(trained: list[str]) -> None:
    # A release trained without the azulenes, the list's only rings of seven.
    without = [s for s in trained if 7 not in domain.aromatic_ring_sizes(s)]
    assert len(without) < len(trained)
    assert domain.outside_training_chemistry("O=c1cccccc1", without) == (
        "an aromatic ring of 7 atoms, a size no training molecule has (theirs: 4, 5, 6)",
    )


@pytest.mark.parametrize(
    "smiles", ["O=[N+]([O-])c1ccccc1", "[O-][n+]1ccccc1", "c1ccc2ccccc2c1", "c1ccoc1"]
)
def test_the_kinds_the_training_list_holds_are_inside(smiles: str, trained: list[str]) -> None:
    assert domain.outside_training_chemistry(smiles, trained) == ()


def test_a_release_trained_on_more_chemistry_admits_more() -> None:
    assert domain.outside_training_chemistry("Cc1cccc(=O)cc1", ["O=c1cccccc1"]) == ()


def test_check_prints_the_molecule_and_says_why_no_band_will_be_licensed(
    capsys: pytest.CaptureFixture[str], trained: list[str]
) -> None:
    (caveat,) = domain.outside_training_chemistry(ZWITTERION, trained)
    assert cli.main(["check", ZWITTERION]) == cli.EXIT_DONE
    assert capsys.readouterr().out == (
        "admitted: [NH3+]c1ccccc1C(=O)[O-] (C7H7NO2, 17 atoms)\n"
        f"note: outside the training chemistry, so no band of it will be licensed: {caveat}\n"
    )


def test_a_string_that_is_no_smiles_is_refused() -> None:
    with pytest.raises(ValueError, match="does not parse as a SMILES"):
        domain.charge_sites("c1ccccc")
