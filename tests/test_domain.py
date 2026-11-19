"""The training chemistry: charge separation and aromatic ring sizes the training lists hold."""

from __future__ import annotations

import pytest

from aromir import cli, domain, release


@pytest.fixture(scope="module")
def trained() -> list[str]:
    return release.load(release.bundled()).training.trained_smiles


def test_the_bundled_training_list_holds_nitro_groups_and_an_n_oxide_and_rings_of_5_and_6(
    trained: list[str],
) -> None:
    # Three kinds of charged atom: a nitro group's N+ and O-, and an aromatic N-oxide's n+
    # (whose O- is of the nitro group's kind: a single bond to N+).
    kinds = {site for smiles in trained for site in domain.charge_sites(smiles)}
    assert {(element, charge) for element, charge, *_ in kinds} == {("N", 1), ("O", -1)}
    assert len(kinds) == 3
    sizes = set().union(*(domain.aromatic_ring_sizes(smiles) for smiles in trained))
    assert sizes == {5, 6}
    assert all(domain.outside_training_chemistry(smiles, trained) == () for smiles in trained)


def test_a_zwitterion_is_outside_the_training_chemistry(trained: list[str]) -> None:
    caveats = domain.outside_training_chemistry("[NH3+]c1ccccc1C(=O)[O-]", trained)
    assert caveats == (
        (
            "charge separation of a kind no training molecule holds: C[O-], c[NH3+] (the "
            "charged atoms of the training molecules: C[N+](=O)[O-], [N+][O-], c[n+]([O-])c)"
        ),
    )


@pytest.mark.parametrize("smiles", ["[N-]=[N+]=Nc1ccccc1", "C[S+]([O-])c1ccccc1"])
def test_an_azide_or_a_sulfoxide_is_outside_a_list_of_nitro_groups(
    smiles: str, trained: list[str]
) -> None:
    (caveat,) = domain.outside_training_chemistry(smiles, trained)
    assert caveat.startswith("charge separation of a kind no training molecule holds")


def test_tropone_is_outside_the_training_chemistry(trained: list[str]) -> None:
    assert domain.outside_training_chemistry("O=c1cccccc1", trained) == (
        "an aromatic ring of 7 atoms, a size no training molecule has (theirs: 5, 6)",
    )


@pytest.mark.parametrize(
    "smiles", ["O=[N+]([O-])c1ccccc1", "[O-][n+]1ccccc1", "c1ccc2ccccc2c1", "c1ccoc1"]
)
def test_the_kinds_the_training_list_holds_are_inside(smiles: str, trained: list[str]) -> None:
    assert domain.outside_training_chemistry(smiles, trained) == ()


def test_a_release_trained_on_more_chemistry_admits_more() -> None:
    assert domain.outside_training_chemistry("Cc1cccc(=O)cc1", ["O=c1cccccc1"]) == ()


def test_check_prints_the_molecule_and_says_why_no_band_will_be_licensed(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert cli.main(["check", "O=c1cccccc1"]) == cli.EXIT_DONE
    assert capsys.readouterr().out == (
        "admitted: O=c1cccccc1 (C7H6O, 14 atoms)\n"
        "note: outside the training chemistry, so no band of it will be licensed: an aromatic "
        "ring of 7 atoms, a size no training molecule has (theirs: 5, 6)\n"
    )


def test_a_string_that_is_no_smiles_is_refused() -> None:
    with pytest.raises(ValueError, match="does not parse as a SMILES"):
        domain.charge_sites("c1ccccc")
