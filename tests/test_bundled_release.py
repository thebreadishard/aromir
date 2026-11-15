"""The release inside the package: it loads, is the version named, and its card is its own."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

from aromir import card, cli, release

if TYPE_CHECKING:
    import pytest

N_SEEDS = 3


def test_the_bundled_release_loads_as_the_version_the_package_names() -> None:
    loaded = release.load(release.bundled())
    assert loaded.network_version == release.BUNDLED_VERSION
    assert [recipe.seed for recipe in loaded.recipes] == list(range(N_SEEDS))
    assert loaded.training.network_version == release.BUNDLED_VERSION


def test_each_weight_file_is_the_one_its_recipe_records() -> None:
    loaded = release.load(release.bundled())
    for seed, recorded in zip(range(N_SEEDS), loaded.tensor_sha256, strict=True):
        tensors = release.bundled() / f"seed-{seed}.safetensors"
        assert hashlib.sha256(tensors.read_bytes()).hexdigest() == recorded


def test_the_card_command_without_a_release_prints_the_shipped_card(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert cli.main(["card"]) == cli.EXIT_DONE
    shipped = (release.bundled() / "MODEL_CARD.md").read_text(encoding="utf-8")
    assert capsys.readouterr().out == shipped


def test_the_shipped_model_card_is_the_one_the_release_writes() -> None:
    shipped = (release.bundled() / "MODEL_CARD.md").read_text(encoding="utf-8")
    assert shipped == card.to_markdown(release.load(release.bundled()))
