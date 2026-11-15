"""A trained network on disk: tensors in a safetensors file, the recipe beside it in JSON.

The recipe says how the network is shaped and how its inputs are built; without it the
tensors cannot be used correctly, so the two are loaded together, and the recipe file
carries the tensor file's sha256.
"""

from __future__ import annotations

import hashlib
import os
from typing import TYPE_CHECKING, Self

import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator
from safetensors.torch import load_file, save_file

from aromir.network import pattern
from aromir.network.inputs import InputSettings, NetworkInputs
from aromir.network.model import Architecture, DeltaHessianNetwork

if TYPE_CHECKING:
    from pathlib import Path

    from torch import Tensor

    from aromir.molecule import FloatArray

TENSOR_SUFFIX = ".safetensors"
RECIPE_SUFFIX = ".json"


class Recipe(BaseModel):
    """What a set of weights is: the network's version, shape and input settings."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    network_version: str = Field(pattern=r"^\d+\.\d+$")
    seed: int = Field(ge=0)
    architecture: Architecture
    inputs: InputSettings

    @model_validator(mode="after")
    def _inputs_fit_the_architecture(self) -> Self:
        expected = pattern.n_pair_features(self.inputs.primitives)
        if self.architecture.n_pair_features not in (0, expected):
            msg = (
                f"{self.architecture.n_pair_features} pair features, but the input settings "
                f"build {expected}"
            )
            raise ValueError(msg)
        if self.inputs.parity != self.architecture.parity_odd:
            msg = "the parity inputs and the parity-odd factor must be on or off together"
            raise ValueError(msg)
        return self


class _RecipeFile(BaseModel):
    """The recipe file's content: the recipe, and the tensor file it was written for."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    recipe: Recipe
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(state: dict[str, Tensor], folder: Path, name: str, recipe: Recipe) -> None:
    """Write a network's tensors and recipe as ``<name>.safetensors`` and ``<name>.json``.

    The state must load into a network of the recipe's architecture; that is checked
    before anything is written.
    """
    DeltaHessianNetwork(recipe.architecture).load_state_dict(state, strict=True)
    tensors = folder / (name + TENSOR_SUFFIX)
    temporary = tensors.with_name(tensors.name + ".tmp")
    save_file({key: value.contiguous() for key, value in state.items()}, temporary)
    os.replace(temporary, tensors)  # noqa: PTH105  the atomic move
    stored = _RecipeFile(recipe=recipe, sha256=_sha256(tensors))
    recipe_path = folder / (name + RECIPE_SUFFIX)
    temporary = recipe_path.with_name(recipe_path.name + ".tmp")
    temporary.write_text(stored.model_dump_json(indent=1) + "\n", encoding="utf-8")
    os.replace(temporary, recipe_path)  # noqa: PTH105  the atomic move


def recorded_sha256(recipe_path: Path) -> str:
    """Return the sha256 a recipe file records for its tensor file."""
    stored = _RecipeFile.model_validate_json(recipe_path.read_text(encoding="utf-8"))
    return stored.sha256


def load(recipe_path: Path) -> tuple[DeltaHessianNetwork, Recipe]:
    """Load a network from its recipe file and the tensor file beside it.

    Raises:
        ValueError: The tensor file is not the one the recipe was written for.
    """
    stored = _RecipeFile.model_validate_json(recipe_path.read_text(encoding="utf-8"))
    tensors = recipe_path.with_suffix(TENSOR_SUFFIX)
    found = _sha256(tensors)
    if found != stored.sha256:
        msg = f"{tensors.name} has sha256 {found[:16]}…, the recipe names {stored.sha256[:16]}…"
        raise ValueError(msg)
    network = DeltaHessianNetwork(stored.recipe.architecture)
    network.load_state_dict(load_file(tensors), strict=True)
    network.eval()
    return network, stored.recipe


def correction(network: DeltaHessianNetwork, inputs: NetworkInputs) -> FloatArray:
    """Return one network's ΔH for a molecule, (3N, 3N), hartree/bohr²."""
    network.eval()
    with torch.no_grad():
        delta: FloatArray = network(inputs).numpy().astype(float)
    return delta
