"""A network release: the seeds' weights with what is known about them.

A release is a folder:

    seed-0.safetensors, seed-0.json    one pair per seed (``aromir.network.weights``)
    evaluation.json                    the version's measured error per family and hold-out
    training.json                      the elements, sizes and molecules it was trained on

The two JSON files are written from the training project's own records. Nothing the
certificate states about a network is typed into this package.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from aromir.errors import RefusedError
from aromir.families import FAMILIES, Family
from aromir.network import inputs, weights

if TYPE_CHECKING:
    from pathlib import Path

    from aromir.molecule import FloatArray, Geometry
    from aromir.network.model import DeltaHessianNetwork

EVALUATION_FILE: Final = "evaluation.json"
TRAINING_FILE: Final = "training.json"
SEED_FILES: Final = "seed-*.json"
MIN_SEEDS: Final = 2
"""A band's spread over the seeds needs at least two of them."""


class Holdout(BaseModel):
    """One set of molecules the network never trained on."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    description: str
    n_molecules: int = Field(ge=1)


class AtomOrderSpread(BaseModel):
    """How far the corrected band positions move when a molecule's atoms are renumbered.

    Attributes:
        max_cm: The largest change of any band over all molecules and orders, cm⁻¹.
        n_molecules: The molecules it was measured on.
        n_orders: The atom orders tried per molecule, each as another SMILES writing gives.
        source: The record the numbers were read from.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_cm: float = Field(ge=0)
    n_molecules: int = Field(ge=1)
    n_orders: int = Field(ge=1)
    source: str


class Evaluation(BaseModel):
    """A network version's measured error, as its evaluation records state it.

    Attributes:
        network_version: The version the numbers belong to.
        higher_level: The level of theory the network corrects towards.
        licence_line_cm: A family is licensed when its error is within this on every hold-out.
        holdouts: The hold-out sets, by name.
        errors_cm: Per family and hold-out, the seed-mean rms error of the corrected
            positions against the higher level, cm⁻¹.
        seed_errors_cm: The same per seed.
        coupled_cluster: What is known at the coupled-cluster level.
        sources: The evaluation records the numbers were read from.
        atom_order_spread: The measured spread between atom orders, when it was measured.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    network_version: str
    higher_level: str
    licence_line_cm: float = Field(gt=0)
    holdouts: dict[str, Holdout]
    errors_cm: dict[Family, dict[str, float]]
    seed_errors_cm: dict[Family, dict[str, list[float]]]
    coupled_cluster: str
    sources: list[str]
    atom_order_spread: AtomOrderSpread | None = None

    @model_validator(mode="after")
    def _every_family_on_every_holdout(self) -> Self:
        for table in (self.errors_cm, self.seed_errors_cm):
            if set(table) != set(FAMILIES):
                msg = f"errors for {sorted(table)}, expected every family of {sorted(FAMILIES)}"
                raise ValueError(msg)
            for family, per_holdout in table.items():
                if set(per_holdout) != set(self.holdouts):
                    msg = (
                        f"{family}: errors on {sorted(per_holdout)}, "
                        f"hold-outs {sorted(self.holdouts)}"
                    )
                    raise ValueError(msg)
        return self

    def licensed(self, family: Family) -> bool:
        """Tell whether a family's error is within the licence line on every hold-out."""
        return all(error <= self.licence_line_cm for error in self.errors_cm[family].values())


class Training(BaseModel):
    """What a network version was trained on.

    Attributes:
        network_version: The version the set belongs to.
        elements: The elements of the training molecules.
        min_atoms: The smallest training molecule's atom count.
        max_atoms: The largest's.
        trained_smiles: Canonical SMILES of every molecule a seed fitted or stopped on.
        held_out_smiles: Canonical SMILES of the hold-out molecules, by hold-out.
        soft_mode_repair: Whether the training Hessians had their soft modes repaired; a
            molecule's Hessian is then prepared the same way before the network reads it.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    network_version: str
    elements: list[str]
    min_atoms: int = Field(ge=1)
    max_atoms: int = Field(ge=1)
    trained_smiles: list[str]
    held_out_smiles: dict[str, list[str]]
    soft_mode_repair: bool


def error_text(error_cm: float) -> str:
    """Write a measured error for a table: two decimals always, so 2.50 reads like 2.12."""
    return f"{error_cm:.2f}"


def level_name(higher_level: str) -> str:
    """Return a level of theory without the explanation in brackets that may follow it.

    A document explains the level where it first names it and uses the bare name after.
    The name ends at the first bracket that follows a space, so "CCSD(T)/cc-pVTZ" is kept
    whole.
    """
    return higher_level.split(" (", 1)[0]


@dataclass(frozen=True, eq=False)
class Corrections:
    """The seeds' answers for one molecule.

    Attributes:
        mean: The seed-mean ΔH (3N, 3N), hartree/bohr²: the correction that is applied.
        per_seed: Each seed's ΔH, in the release's seed order.
    """

    mean: FloatArray
    per_seed: tuple[FloatArray, ...]


@dataclass(frozen=True, eq=False)
class Release:
    """A loaded release: its networks, their recipes and file hashes, and its records."""

    networks: tuple[DeltaHessianNetwork, ...]
    recipes: tuple[weights.Recipe, ...]
    tensor_sha256: tuple[str, ...]
    evaluation: Evaluation
    training: Training

    @property
    def network_version(self) -> str:
        """The version every part of the release belongs to."""
        return self.evaluation.network_version

    def corrections(self, geometry: Geometry, hessian: FloatArray) -> Corrections:
        """Return every seed's ΔH for a molecule, and their mean."""
        built = inputs.build(geometry, hessian, self.recipes[0].inputs)
        per_seed = tuple(weights.correction(network, built) for network in self.networks)
        return Corrections(np.mean(per_seed, axis=0), per_seed)


def _reasons_to_refuse(
    recipes: tuple[weights.Recipe, ...], evaluation: Evaluation, training: Training
) -> list[str]:
    reasons = []
    if len(recipes) < MIN_SEEDS:
        reasons.append(f"{len(recipes)} seed(s): a release needs at least {MIN_SEEDS}")
    versions = {recipe.network_version for recipe in recipes}
    versions |= {evaluation.network_version, training.network_version}
    if len(versions) > 1:
        reasons.append(f"its parts are of different network versions: {sorted(versions)}")
    if len({(recipe.architecture, recipe.inputs) for recipe in recipes}) > 1:
        reasons.append("its seeds differ in architecture or input settings")
    if len({recipe.seed for recipe in recipes}) < len(recipes):
        reasons.append("two of its weight files are of the same seed")
    return reasons


def load(folder: Path) -> Release:
    """Load a release folder.

    Raises:
        RefusedError: A file is missing, or the parts do not belong together.
    """
    missing = [name for name in (EVALUATION_FILE, TRAINING_FILE) if not (folder / name).is_file()]
    if missing:
        raise RefusedError([f"{folder}: no {name}" for name in missing])
    evaluation = Evaluation.model_validate_json(
        (folder / EVALUATION_FILE).read_text(encoding="utf-8")
    )
    training = Training.model_validate_json((folder / TRAINING_FILE).read_text(encoding="utf-8"))
    seed_files = sorted(folder.glob(SEED_FILES))
    loaded = [weights.load(path) for path in seed_files]
    recipes = tuple(recipe for _, recipe in loaded)
    reasons = _reasons_to_refuse(recipes, evaluation, training)
    if reasons:
        raise RefusedError([f"{folder}: {reason}" for reason in reasons])
    return Release(
        networks=tuple(network for network, _ in loaded),
        recipes=recipes,
        tensor_sha256=tuple(weights.recorded_sha256(path) for path in seed_files),
        evaluation=evaluation,
        training=training,
    )
