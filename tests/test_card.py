"""The model card: written from a release, with nothing typed in."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import TYPE_CHECKING

import pytest

from aromir import card, cli, release
from aromir.network.inputs import InputSettings
from aromir.network.primitives import PrimitiveSettings

from support import TRAINING, write_release

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(scope="module")
def loaded(tmp_path_factory: pytest.TempPathFactory) -> release.Release:
    return release.load(write_release(tmp_path_factory.mktemp("release")))


def test_the_card_states_the_releases_own_numbers(loaded: release.Release) -> None:
    text = card.to_markdown(loaded)
    assert text.startswith("---\nlicense: cc-by-4.0\n")
    assert "# aromir ΔH network v1.3" in text
    assert "- **Seeds:** 2 (0, 1)" in text
    assert "- **Licensed families:** ring-ip, CH-stretch, CH-oop." in text
    assert "- **Not licensed:** other." in text
    seeds = loaded.evaluation.seed_errors_cm["ring-ip"]
    brackets = {name: ", ".join(str(error) for error in seeds[name]) for name in seeds}
    assert f"| ring-ip | 1.77 ({brackets['a']}) | 2.12 ({brackets['b']}) | yes |" in text
    assert "- Elements: C, H, N, O." in text
    assert "- Sizes: 10 to 30 atoms." in text
    assert "- 1 molecules, neutral" in text
    assert "1 in (a), 1 in (b)" in text
    n_parameters = sum(parameter.numel() for parameter in loaded.networks[0].parameters())
    assert f"- {n_parameters:,} parameters per seed." in text
    for sha256 in loaded.tensor_sha256:
        assert f"`{sha256}`" in text
    assert text.endswith("own files.\n")


def test_the_card_says_how_the_training_hessians_were_prepared(loaded: release.Release) -> None:
    assert "without the soft-mode repair" in card.to_markdown(loaded)
    repaired = replace(
        loaded, training=loaded.training.model_copy(update={"soft_mode_repair": True})
    )
    assert "soft modes (below 20 cm⁻¹) measured" in card.to_markdown(repaired)


def test_the_card_warns_about_the_atom_order_unless_the_network_is_order_free(
    loaded: release.Release,
) -> None:
    assert "**The answer depends on how the molecule's atoms are numbered**" in card.to_markdown(
        loaded
    )
    order_free = InputSettings(
        primitives=PrimitiveSettings(angle_choice="canonical", dihedral_encoding="cos")
    )
    recipes = tuple(recipe.model_copy(update={"inputs": order_free}) for recipe in loaded.recipes)
    text = card.to_markdown(replace(loaded, recipes=recipes))
    assert "does not depend on how the molecule's atoms are numbered" in text
    assert "Do not feed the network" not in text


def test_a_release_with_no_licensed_family_says_so(loaded: release.Release) -> None:
    errors = {family: {"a": 9.0, "b": 9.0} for family in loaded.evaluation.errors_cm}
    evaluation = loaded.evaluation.model_copy(update={"errors_cm": errors})
    text = card.to_markdown(replace(loaded, evaluation=evaluation))
    assert "- **Licensed families:** none." in text
    assert "- **Not licensed: ring-ip, CH-stretch, CH-oop, other.**" in text


def test_the_command_prints_or_writes_the_card(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    folder = write_release(tmp_path / "release")
    expected = card.to_markdown(release.load(folder))
    assert cli.main(["card", "--release", str(folder)]) == cli.EXIT_DONE
    assert capsys.readouterr().out == expected
    out = tmp_path / "MODEL_CARD.md"
    assert cli.main(["card", "--release", str(folder), "--out", str(out)]) == cli.EXIT_DONE
    assert out.read_text(encoding="utf-8") == expected
    assert capsys.readouterr().out == f"done: model card in {out}\n"


def test_the_command_refuses_a_folder_that_is_no_release(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["card", "--release", str(tmp_path)]) == cli.EXIT_REFUSED
    assert "no evaluation.json" in capsys.readouterr().err


def test_a_release_record_missing_a_field_names_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    folder = write_release(tmp_path / "release")
    training = {key: value for key, value in TRAINING.items() if key != "soft_mode_repair"}
    (folder / "training.json").write_text(json.dumps(training), encoding="utf-8")
    assert cli.main(["card", "--release", str(folder)]) == cli.EXIT_REFUSED
    assert capsys.readouterr().err == "refused:\n  soft_mode_repair: Field required\n"


def test_the_card_states_the_measured_spread_between_atom_orders(loaded: release.Release) -> None:
    assert "The spread between atom orders was not measured for this release." in (
        card.to_markdown(loaded)
    )
    spread = release.AtomOrderSpread(
        max_cm=2.9, n_molecules=3, n_orders=5, source="order_spread_v1.3.json"
    )
    evaluation = loaded.evaluation.model_copy(update={"atom_order_spread": spread})
    text = card.to_markdown(replace(loaded, evaluation=evaluation))
    assert (
        "Measured: band positions move by up to 2.9 cm⁻¹ between atom orders "
        "(3 molecules, 5 orders each; order_spread_v1.3.json)." in text
    )


def test_a_spread_must_be_a_measurement() -> None:
    with pytest.raises(ValueError, match="greater than or equal to 0"):
        release.AtomOrderSpread(max_cm=-1.0, n_molecules=3, n_orders=5, source="x.json")
    with pytest.raises(ValueError, match="greater than or equal to 1"):
        release.AtomOrderSpread(max_cm=1.0, n_molecules=0, n_orders=5, source="x.json")
