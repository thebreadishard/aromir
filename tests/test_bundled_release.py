"""The release inside the package: it loads, is the version named, and its card is its own."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from aromir import card, cli, release
from aromir.molecule import Geometry

if TYPE_CHECKING:
    import pytest

N_SEEDS = 3
PLATFORM_PIN = Path(__file__).parent / "data" / "release" / "delta_h_biphenyl_cn_v1.3.npz"
DELTA_H_TOLERANCE = 1e-6
"""hartree/bohr². Single-precision arithmetic on another build: Linux and Windows differ by
7e-9 for this molecule, whose largest ΔH element is 0.040, and by 2.4e-7 at most over 76
molecules of the training corpus."""


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


def test_the_release_corrects_a_twisted_biphenyl_alike_on_every_platform() -> None:
    # The pin is the Windows build's ΔH for 4-cyanobiphenyl. Before
    # `primitives.pseudo_inverse` this failed on numpy 2.5.4's Linux wheels: a null singular
    # value of this molecule's B matrix came back above `np.linalg.pinv`'s default cutoff
    # and every seed's ΔH was 1e29. A numerical step that depends on the platform shows
    # here as a difference above single-precision noise.
    with np.load(PLATFORM_PIN) as pin:
        geometry = Geometry(tuple(str(symbol) for symbol in pin["symbols"]), pin["coords_bohr"])
        hessian, pinned = pin["hessian"], pin["delta_h_windows"]
    corrections = release.load(release.bundled()).corrections(geometry, hessian)
    assert len(corrections.per_seed) == len(pinned) == N_SEEDS
    for ours, theirs in zip(corrections.per_seed, pinned, strict=True):
        assert 0.03 < float(np.abs(theirs).max()) < 0.05
        assert float(np.abs(ours - theirs).max()) <= DELTA_H_TOLERANCE
