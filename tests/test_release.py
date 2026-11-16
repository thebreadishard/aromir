"""A release folder: what it must hold, and what it says about its network."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from aromir import geometry, release
from aromir.errors import RefusedError
from aromir.network import weights
from aromir.network.inputs import InputSettings

from support import (
    RELEASE_DATA,
    SMALL_ARCHITECTURE,
    random_hessian,
    random_network,
    write_release,
)

if TYPE_CHECKING:
    from pathlib import Path


def _evaluation() -> dict[str, Any]:
    record: dict[str, Any] = json.loads(
        (RELEASE_DATA / "evaluation_v1.3.json").read_text(encoding="utf-8")
    )
    return record


def test_a_release_loads_with_its_records(tmp_path: Path) -> None:
    loaded = release.load(write_release(tmp_path))
    assert loaded.network_version == "1.3"
    assert [recipe.seed for recipe in loaded.recipes] == [0, 1]
    assert loaded.training.elements == ["C", "H", "N", "O"]
    assert set(loaded.evaluation.holdouts) == {"a", "b"}


def test_the_licence_follows_the_measured_errors() -> None:
    evaluation = release.Evaluation.model_validate(_evaluation())
    assert evaluation.licence_line_cm == 3.0
    assert evaluation.errors_cm["other"] == {"a": 3.62, "b": 3.89}
    assert [evaluation.licensed(family) for family in ("ring-ip", "CH-stretch", "CH-oop")] == [
        True,
        True,
        True,
    ]
    assert not evaluation.licensed("other")


def test_one_hold_out_above_the_line_withdraws_the_licence() -> None:
    record = _evaluation()
    record["errors_cm"]["ring-ip"]["b"] = 3.01
    assert not release.Evaluation.model_validate(record).licensed("ring-ip")


def test_an_evaluation_must_cover_every_family_and_hold_out() -> None:
    without_family = _evaluation()
    del without_family["errors_cm"]["other"]
    with pytest.raises(ValueError, match="expected every family"):
        release.Evaluation.model_validate(without_family)
    without_holdout = _evaluation()
    del without_holdout["seed_errors_cm"]["other"]["b"]
    with pytest.raises(ValueError, match="other: errors on"):
        release.Evaluation.model_validate(without_holdout)


def test_the_applied_correction_is_the_seeds_mean(tmp_path: Path) -> None:
    loaded = release.load(write_release(tmp_path))
    benzene = geometry.start_from_smiles("c1ccccc1")
    corrections = loaded.corrections(benzene, random_hessian(12, seed=0))
    assert len(corrections.per_seed) == 2
    assert np.abs(corrections.per_seed[0] - corrections.per_seed[1]).max() > 0
    np.testing.assert_allclose(
        corrections.mean, 0.5 * (corrections.per_seed[0] + corrections.per_seed[1])
    )


@pytest.mark.parametrize("missing", ["evaluation.json", "training.json"])
def test_a_release_without_its_records_is_refused(tmp_path: Path, missing: str) -> None:
    folder = write_release(tmp_path)
    (folder / missing).unlink()
    with pytest.raises(RefusedError, match=f"no {missing}"):
        release.load(folder)


def test_a_single_seed_is_refused(tmp_path: Path) -> None:
    with pytest.raises(RefusedError, match="at least 2"):
        release.load(write_release(tmp_path, seeds=(0,)))


def test_parts_of_different_versions_are_refused(tmp_path: Path) -> None:
    with pytest.raises(RefusedError, match=r"different network versions: \['1.3', '1.4'\]"):
        release.load(write_release(tmp_path, version="1.4"))


def test_seeds_built_differently_are_refused(tmp_path: Path) -> None:
    folder = write_release(tmp_path)
    other = weights.Recipe(
        network_version="1.3",
        seed=2,
        architecture=SMALL_ARCHITECTURE,
        inputs=InputSettings(pattern="d"),
    )
    weights.save(random_network(SMALL_ARCHITECTURE, seed=2).state_dict(), folder, "seed-2", other)
    with pytest.raises(RefusedError, match="differ in architecture or input settings"):
        release.load(folder)


def test_a_recipe_under_another_seeds_name_is_refused(tmp_path: Path) -> None:
    folder = write_release(tmp_path)
    again = weights.Recipe(
        network_version="1.3", seed=1, architecture=SMALL_ARCHITECTURE, inputs=InputSettings()
    )
    weights.save(random_network(SMALL_ARCHITECTURE, seed=1).state_dict(), folder, "seed-9", again)
    with pytest.raises(RefusedError, match="holds the recipe of seed 1"):
        release.load(folder)
