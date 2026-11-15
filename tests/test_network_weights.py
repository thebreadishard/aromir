"""Weights on disk: saved with their recipe, checked on loading, converted from a checkpoint."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from safetensors.torch import load_file

from aromir import geometry
from aromir.network import convert, inputs, weights
from aromir.network.model import Architecture
from aromir.network.primitives import PrimitiveSettings

from support import random_hessian, random_network

PINS = Path(__file__).parent / "data" / "network"
SMALL = Architecture(n_features=16, n_blocks=2, hidden=24)
RECIPE = weights.Recipe(
    network_version="1.3", seed=2, architecture=SMALL, inputs=inputs.InputSettings()
)


def _trainer_checkpoint() -> dict[str, Any]:
    """A checkpoint as the trainer saves it, built from the carried-shape pin."""
    record: dict[str, Any] = json.loads((PINS / "carried_shape.json").read_text(encoding="utf-8"))
    return record | {"state": load_file(PINS / "carried_shape.safetensors")}


def test_a_saved_network_loads_to_the_same_correction(tmp_path: Path) -> None:
    weights.save(random_network(SMALL, seed=0).state_dict(), tmp_path, "network", RECIPE)
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "network.json",
        "network.safetensors",
    ]
    loaded, read_back = weights.load(tmp_path / "network.json")
    assert read_back == RECIPE
    assert not loaded.training
    molecule = geometry.start_from_smiles("c1ccccc1")
    built = inputs.build(molecule, random_hessian(12, seed=0), RECIPE.inputs)
    np.testing.assert_array_equal(
        weights.correction(loaded, built),
        weights.correction(random_network(SMALL, seed=0), built),
    )


def test_tensors_that_are_not_the_recipes_are_refused(tmp_path: Path) -> None:
    other = tmp_path / "other"
    other.mkdir()
    weights.save(random_network(SMALL, seed=0).state_dict(), tmp_path, "network", RECIPE)
    weights.save(random_network(SMALL, seed=1).state_dict(), other, "network", RECIPE)
    (other / "network.safetensors").replace(tmp_path / "network.safetensors")
    with pytest.raises(ValueError, match="the recipe names"):
        weights.load(tmp_path / "network.json")


def test_a_state_of_another_shape_is_not_written(tmp_path: Path) -> None:
    carried_size = RECIPE.model_copy(update={"architecture": Architecture()})
    with pytest.raises(RuntimeError, match="size mismatch"):
        weights.save(random_network(SMALL, seed=0).state_dict(), tmp_path, "network", carried_size)
    assert list(tmp_path.iterdir()) == []


def test_a_recipe_whose_inputs_do_not_fit_its_architecture_is_refused() -> None:
    recipe = RECIPE.model_dump()
    cossin = PrimitiveSettings(dihedral_encoding="cossin").model_dump()
    with pytest.raises(ValueError, match="66 pair features, but the input settings build 68"):
        weights.Recipe.model_validate(
            recipe | {"inputs": recipe["inputs"] | {"primitives": cossin}}
        )
    with pytest.raises(ValueError, match="on or off together"):
        weights.Recipe.model_validate(recipe | {"inputs": recipe["inputs"] | {"parity": True}})
    with pytest.raises(ValueError, match="network_version"):
        weights.Recipe.model_validate(recipe | {"network_version": "latest"})


def test_a_checkpoint_is_converted_with_its_settings(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    checkpoint = _trainer_checkpoint()
    torch.save(checkpoint, tmp_path / "checkpoint.pt")
    convert.main([str(tmp_path / "checkpoint.pt"), str(tmp_path), "v", "--network-version", "1.3"])
    assert capsys.readouterr().out == f"v: network 1.3, seed {checkpoint['seed']}\n"
    network, recipe = weights.load(tmp_path / "v.json")
    assert recipe == weights.Recipe(
        network_version="1.3",
        seed=checkpoint["seed"],
        architecture=SMALL,
        inputs=inputs.InputSettings(pattern="f"),
    )
    assert not any(name.startswith("body.") for name in network.state_dict())


def test_the_order_free_settings_survive_the_conversion() -> None:
    record = json.loads((PINS / "order_free.json").read_text(encoding="utf-8"))
    _, settings = convert.settings(record)
    order_free = PrimitiveSettings(angle_choice="canonical", dihedral_encoding="cos")
    assert settings.primitives == order_free


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"head": "cartesian"}, "only the pair"),
        ({"ctor": {"tensor_input": True}}, "rank-2 Hessian input"),
        ({"ctor": {"n_v": 8}}, "as many of each"),
        ({"ctor": {"dropout": 0.1}}, "settings aromir does not know"),
        ({"angle_choice": "canonical"}, "without the other primitive setting"),
    ],
)
def test_a_checkpoint_aromir_cannot_run_is_refused(change: dict[str, Any], reason: str) -> None:
    checkpoint = _trainer_checkpoint()
    constructor = checkpoint["ctor"] | change.pop("ctor", {})
    with pytest.raises(ValueError, match=reason):
        convert.settings(checkpoint | change | {"ctor": constructor})


def test_an_unknown_parameter_is_refused() -> None:
    trainer_state = _trainer_checkpoint()["state"]
    with pytest.raises(ValueError, match=r"does not know: body\.extra\.weight"):
        convert.state(trainer_state | {"body.extra.weight": torch.zeros(1)})
