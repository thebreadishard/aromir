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
    seeds = loaded.evaluation.first_order_error_per_seed_cm["ring-ip"]
    brackets = {name: ", ".join(f"{error:.2f}" for error in seeds[name]) for name in seeds}
    assert "| ring-ip | 1.43 [1.13–1.70], 199 bands | 1.91 [1.84–2.00], 1164 bands | yes |" in text
    assert "| other | 2.78 [1.68–3.76], 156 bands | 3.89 [3.50–4.20], 1225 bands | no |" in text
    assert f"| ring-ip | 1.77 ({brackets['a']}) | 2.12 ({brackets['b']}) |" in text
    assert "It is not the error of what aromir prints." in text
    assert "chosen on 16 November 2026" in text
    assert "- Elements: C, H, N, O." in text
    assert "- Sizes: 10 to 30 atoms." in text
    assert "- 1 molecules, neutral" in text
    assert "1 in (a); 1 in (b)" in text
    n_parameters = sum(parameter.numel() for parameter in loaded.networks[0].parameters())
    assert f"- {n_parameters:,} parameters per seed." in text
    for sha256 in loaded.tensor_sha256:
        assert f"`{sha256}`" in text
    assert text.endswith("own files.\n")


def test_a_held_out_molecule_that_was_trained_on_is_not_counted_as_held_out(
    loaded: release.Release,
) -> None:
    leaked = loaded.training.model_copy(
        update={"held_out_smiles": {"a": ["c1ccccc1", "c1ccc2ccccc2c1"], "b": ["Cc1ccccc1"]}}
    )
    text = card.to_markdown(replace(loaded, training=leaked))
    assert (
        "- Held out and never trained on: 1 in (a), not counting 1 the network also trained "
        "on; 1 in (b)." in text
    )


def test_a_hold_out_list_is_counted_by_molecule_and_none_is_said(
    loaded: release.Release,
) -> None:
    twice = {"a": ["c1ccc2ccccc2c1", "c1ccc2ccccc2c1"], "b": ["Cc1ccccc1", "Cc1ccccc1"]}
    training = loaded.training.model_copy(update={"held_out_smiles": twice})
    text = card.to_markdown(replace(loaded, training=training))
    assert (
        "- Held out and never trained on: 0 in (a), not counting 1 the network also trained "
        "on; 1 in (b)." in text
    )
    nothing = loaded.training.model_copy(update={"held_out_smiles": {}})
    assert "- Held out and never trained on: none." in card.to_markdown(
        replace(loaded, training=nothing)
    )


def test_errors_are_written_with_two_decimals(loaded: release.Release) -> None:
    errors = dict(loaded.evaluation.first_order_error_per_seed_mean_cm)
    errors["CH-stretch"] = {"a": 1.2, "b": 2.5}
    evaluation = loaded.evaluation.model_copy(update={"first_order_error_per_seed_mean_cm": errors})
    text = card.to_markdown(replace(loaded, evaluation=evaluation))
    assert "| CH-stretch | 1.20 (" in text
    assert ") | 2.50 (" in text


def test_the_level_is_explained_once_and_named_after(loaded: release.Release) -> None:
    explained = "ωB97X/6-31G* (the DFT proxy: trained on ωB97X − B3LYP Hessians)"
    evaluation = loaded.evaluation.model_copy(update={"higher_level": explained})
    text = card.to_markdown(replace(loaded, evaluation=evaluation))
    assert text.count("(the DFT proxy") == 1
    assert "the difference to ωB97X/6-31G* (the DFT proxy" in text
    assert "- **The level is ωB97X/6-31G*.**" in text
    assert "positions of the seeds' mean against ωB97X/6-31G*, in cm⁻¹" in text


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
    assert "is meant to make the answer independent of how the molecule's atoms" in text
    assert "Do not feed the network" not in text


def test_a_release_with_no_licensed_family_says_so(loaded: release.Release) -> None:
    high = release.BandError(
        rms_cm=9.0, n_modes=10, interval_cm=(8.0, 10.0), share_of_resamples_within_line=0.0
    )
    errors = {
        family: {"a": high, "b": high} for family in loaded.evaluation.band_error_seed_mean_cm
    }
    evaluation = loaded.evaluation.model_copy(update={"band_error_seed_mean_cm": errors})
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
    assert capsys.readouterr().err == (
        f"refused:\n  {folder / 'training.json'}: soft_mode_repair: Field required\n"
    )


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
