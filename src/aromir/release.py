"""A network release: the seeds' weights with what is known about them.

A release is a folder:

    seed-0.safetensors, seed-0.json    one pair per seed (``aromir.network.weights``)
    evaluation.json                    the version's measured error per family and hold-out
    training.json                      the elements, sizes and molecules it was trained on
    invariance.json                    how far its answer moves under a rotation, a
                                       reflection and another atom listing (optional)
    measured.json                      the output of ``aromir evaluate`` on the hold-outs,
                                       which the evaluation's tables are (optional)

The JSON files are written from the training project's own records. Nothing the
certificate states about a network is typed into this package.

One release ships inside the package (``bundled``), so a corrected spectrum needs no
download and works on a machine without a network connection.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal, Self, TypeVar

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from safetensors import SafetensorError

from aromir.constants import ELEMENTS
from aromir.errors import RefusedError
from aromir.families import FAMILIES, Family
from aromir.network import inputs, weights

if TYPE_CHECKING:
    from aromir.molecule import FloatArray, Geometry
    from aromir.network.model import DeltaHessianNetwork

EVALUATION_FILE: Final = "evaluation.json"
TRAINING_FILE: Final = "training.json"
INVARIANCE_FILE: Final = "invariance.json"
MEASURED_FILE: Final = "measured.json"
"""What ``aromir evaluate`` wrote on the hold-outs: the source of the evaluation record's
error tables, shipped beside it where a release has it."""
SEED_FILES: Final = "seed-*.json"
MIN_SEEDS: Final = 2
"""A band's spread over the seeds needs at least two of them."""
_MEAN_TOLERANCE_CM: Final = 0.011
"""A record holds two decimals: a mean of rounded seeds may differ from the rounded mean."""
_Record = TypeVar("_Record", bound=BaseModel)
BUNDLED_VERSION: Final = "1.3"
"""The network version whose release ships inside the package."""


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

    max_cm: float = Field(ge=0, allow_inf_nan=False)
    n_molecules: int = Field(ge=1)
    n_orders: int = Field(ge=1)
    source: str


class BandError(BaseModel):
    """The error of the corrected band positions of the seeds' mean, for one family on one hold-out.

    Attributes:
        rms_cm: The rms difference to the reference band each corrected band is paired
            with, cm⁻¹.
        n_modes: The bands it is the rms of.
        interval_cm: The 2.5th and 97.5th percentile of that rms when the hold-out's
            molecules are resampled with replacement.
        share_of_resamples_within_line: The share of those resamples below the licence line.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    rms_cm: float = Field(ge=0, allow_inf_nan=False)
    n_modes: int = Field(ge=1)
    interval_cm: tuple[float, float]
    share_of_resamples_within_line: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def _an_interval(self) -> Self:
        low, high = self.interval_cm
        if not (math.isfinite(low) and math.isfinite(high) and 0 <= low <= high):
            msg = f"{self.interval_cm} is no interval of an rms error"
            raise ValueError(msg)
        return self


class Evaluation(BaseModel):
    """A network version's measured error, as its evaluation records state it.

    Two quantities are kept apart by name (``aromir.evaluation`` defines both): the band
    error of the seeds' mean, which is what the product delivers and what the licence is
    read on, and the trainer's first-order error per seed.

    Attributes:
        record_version: The format of this record; 2 is the first that names both.
        network_version: The version the numbers belong to.
        higher_level: The level of theory the network corrects towards.
        licence_line_cm: A family is licensed when its band error is within this on every
            hold-out.
        licence_basis: Which quantity the licence is read on, and since when.
        holdouts: The hold-out sets, by name.
        band_error_seed_mean_cm: Per family and hold-out, the error of the corrected band
            positions of the seeds' mean.
        first_order_error_per_seed_cm: Per family and hold-out, each seed's first-order
            error.
        first_order_error_per_seed_mean_cm: The mean of those over the seeds.
        coupled_cluster: What is known at the coupled-cluster level.
        sources: The evaluation records the numbers were read from.
        atom_order_spread: The measured spread between atom orders, when it was measured.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    record_version: Literal[2]
    network_version: str
    higher_level: str
    licence_line_cm: float = Field(gt=0, allow_inf_nan=False)
    licence_basis: str = Field(min_length=1)
    holdouts: dict[str, Holdout]
    band_error_seed_mean_cm: dict[Family, dict[str, BandError]]
    first_order_error_per_seed_cm: dict[Family, dict[str, list[float]]]
    first_order_error_per_seed_mean_cm: dict[Family, dict[str, float]]
    coupled_cluster: str
    sources: list[str]
    atom_order_spread: AtomOrderSpread | None = None

    @model_validator(mode="before")
    @classmethod
    def _not_the_first_format(cls, record: Any) -> Any:  # noqa: ANN401  a record before it is validated
        if isinstance(record, dict) and "errors_cm" in record:
            msg = (
                "a record of the first format: its `errors_cm` are first-order errors per "
                "seed, which are not the error of what the release delivers and are not read "
                "as it. Measure the release with `aromir evaluate` and write a record of "
                "format 2"
            )
            raise ValueError(msg)
        return record

    @model_validator(mode="after")
    def _every_family_on_every_holdout(self) -> Self:
        tables: tuple[dict[Family, Any], ...] = (
            self.band_error_seed_mean_cm,
            self.first_order_error_per_seed_cm,
            self.first_order_error_per_seed_mean_cm,
        )
        for table in tables:
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

    @model_validator(mode="after")
    def _every_error_is_an_error(self) -> Self:
        for family, per_holdout in self.first_order_error_per_seed_cm.items():
            for holdout, seeds in per_holdout.items():
                mean = self.first_order_error_per_seed_mean_cm[family][holdout]
                if not all(math.isfinite(error) and error >= 0 for error in [*seeds, mean]):
                    msg = "an rms error is negative or not a finite number"
                    raise ValueError(msg)
                if not seeds or abs(mean - sum(seeds) / len(seeds)) > _MEAN_TOLERANCE_CM:
                    msg = f"{family} on ({holdout}): {mean} is not the mean of the seeds' {seeds}"
                    raise ValueError(msg)
        return self

    def licensed(self, family: Family) -> bool:
        """Tell whether a family's band error is within the licence line on every hold-out.

        The comparison is made on the record's numbers, which hold two decimals: a family
        measured at 3.004 cm⁻¹ is recorded as 3.00 and is within a line of 3.
        """
        errors = self.band_error_seed_mean_cm[family].values()
        return all(error.rms_cm <= self.licence_line_cm for error in errors)

    def caveats(self) -> list[str]:
        """Say, for every licensed family on thin ground, what the licence rests on."""
        sentences = []
        for family in FAMILIES:
            if not self.licensed(family):
                continue
            for holdout, error in self.thin_ground(family).items():
                low, high = error.interval_cm
                sentences.append(
                    f"'{family}' is within the licence line on hold-out ({holdout}) at "
                    f"{error.rms_cm:.2f} cm⁻¹, on {self.holdouts[holdout].n_molecules} molecules: "
                    f"resampling those molecules puts the error between {low:.2f} and {high:.2f} "
                    f"cm⁻¹, and {1 - error.share_of_resamples_within_line:.0%} of the resamples "
                    "lie above the line"
                )
        return sentences

    def thin_ground(self, family: Family) -> dict[str, BandError]:
        """Return the hold-outs on which a family's licence could fall with other molecules.

        Those are the hold-outs where the upper end of the resampling interval is above
        the line, for a family that is licensed; a family that is not has no licence to
        lose.
        """
        if not self.licensed(family):
            return {}
        return {
            holdout: error
            for holdout, error in self.band_error_seed_mean_cm[family].items()
            if error.rms_cm <= self.licence_line_cm < error.interval_cm[1]
        }


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
    elements: list[str] = Field(min_length=1)
    min_atoms: int = Field(ge=1)
    max_atoms: int = Field(ge=1)
    trained_smiles: list[str]
    held_out_smiles: dict[str, list[str]]
    soft_mode_repair: bool

    @model_validator(mode="after")
    def _a_set_that_can_exist(self) -> Self:
        unknown = sorted(set(self.elements) - set(ELEMENTS))
        if unknown:
            msg = f"element(s) {', '.join(unknown)} that the package does not know"
            raise ValueError(msg)
        if self.min_atoms > self.max_atoms:
            msg = f"the smallest molecule has {self.min_atoms} atoms, the largest {self.max_atoms}"
            raise ValueError(msg)
        return self

    def n_held_out(self, holdout: str) -> int:
        """Count a hold-out's molecules that no seed trained on under another identifier."""
        return len(set(self.held_out_smiles[holdout]) - set(self.trained_smiles))


class MeasuredSpread(BaseModel):
    """How far the corrected band positions move under one change that leaves a molecule itself.

    The numbers are the largest seen on a handful of molecules, not bounds.

    Attributes:
        with_linear_group_cm: The largest change of any band, cm⁻¹, over the molecules
            measured that have a linear group (a nitrile, an alkyne); None when no such
            molecule was measured.
        n_with_linear_group: How many such molecules were measured.
        without_linear_group_cm: The same over the molecules without one.
        n_without_linear_group: How many of those were measured.
        n_trials: The changes tried per molecule.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    with_linear_group_cm: float | None = Field(ge=0, allow_inf_nan=False)
    n_with_linear_group: int = Field(ge=0)
    without_linear_group_cm: float | None = Field(ge=0, allow_inf_nan=False)
    n_without_linear_group: int = Field(ge=0)
    n_trials: int = Field(ge=1)

    @model_validator(mode="after")
    def _a_number_for_every_molecule_measured(self) -> Self:
        for value, count in (
            (self.with_linear_group_cm, self.n_with_linear_group),
            (self.without_linear_group_cm, self.n_without_linear_group),
        ):
            if (value is None) != (count == 0):
                msg = "a spread is given exactly when at least one molecule was measured"
                raise ValueError(msg)
        if self.n_with_linear_group + self.n_without_linear_group == 0:
            msg = "a spread measured on no molecule is no measurement"
            raise ValueError(msg)
        return self

    def of(self, *, linear_group: bool) -> tuple[float | None, int]:
        """Return the spread for molecules with or without a linear group, and their count."""
        if linear_group:
            return self.with_linear_group_cm, self.n_with_linear_group
        return self.without_linear_group_cm, self.n_without_linear_group


class Invariance(BaseModel):
    """What a network version's answer depends on that the molecule does not.

    A molecule turned in space, its mirror image and another listing of its atoms are the
    same molecule with the same spectrum. A network whose inputs are not built to respect
    that gives each a somewhat different answer; this record says by how much.

    Attributes:
        network_version: The version the numbers belong to.
        rotation: Under a rigid rotation of geometry and Hessian.
        mirror_image: Under a reflection of both.
        xyz_atom_order: Between listings of one geometry's atoms in another order, each
            put into aromir's canonical order.
        source: The record the numbers were read from.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    network_version: str
    rotation: MeasuredSpread
    mirror_image: MeasuredSpread
    xyz_atom_order: MeasuredSpread
    source: str

    def spreads(self) -> dict[str, MeasuredSpread]:
        """Return each measured spread under the words a document names it by."""
        return {
            "a rigid rotation of the molecule": self.rotation,
            "the molecule's mirror image": self.mirror_image,
            "another listing of the same atoms in an xyz file": self.xyz_atom_order,
        }


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
    invariance: Invariance | None = None

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
    recipes: tuple[weights.Recipe, ...],
    evaluation: Evaluation,
    training: Training,
    invariance: Invariance | None,
) -> list[str]:
    reasons = []
    if len(recipes) < MIN_SEEDS:
        reasons.append(f"{len(recipes)} seed(s): a release needs at least {MIN_SEEDS}")
    versions = {recipe.network_version for recipe in recipes}
    versions |= {evaluation.network_version, training.network_version}
    if invariance is not None:
        versions.add(invariance.network_version)
    if len(versions) > 1:
        reasons.append(f"its parts are of different network versions: {sorted(versions)}")
    if len({(recipe.architecture, recipe.inputs) for recipe in recipes}) > 1:
        reasons.append("its seeds differ in architecture or input settings")
    return reasons + _records_that_disagree(evaluation, training)


def _records_that_disagree(evaluation: Evaluation, training: Training) -> list[str]:
    """Name where the evaluation and the training record describe different hold-outs."""
    if set(evaluation.holdouts) != set(training.held_out_smiles):
        names = (
            f"{EVALUATION_FILE} has the hold-outs {sorted(evaluation.holdouts)}, "
            f"{TRAINING_FILE} {sorted(training.held_out_smiles)}"
        )
        return [names]
    reasons = []
    for name, holdout in evaluation.holdouts.items():
        listed = training.n_held_out(name)
        if holdout.n_molecules != listed:
            reasons.append(
                f"hold-out ({name}): {EVALUATION_FILE} counts {holdout.n_molecules} molecules, "
                f"{TRAINING_FILE} lists {listed} that were never trained on"
            )
    return reasons


def _field_reasons(path: Path, invalid: ValidationError) -> list[str]:
    """Return a record's validation errors, each with the file and the field it is in."""
    return [
        ": ".join([str(path), *(str(part) for part in error["loc"]), str(error["msg"])])
        for error in invalid.errors()
    ]


def _record(model: type[_Record], folder: Path, name: str) -> _Record:
    """Read one JSON record of a release; a file that does not hold it is a refusal."""
    path = folder / name
    try:
        return model.model_validate_json(path.read_text(encoding="utf-8"))
    except ValidationError as invalid:
        raise RefusedError(_field_reasons(path, invalid)) from invalid
    except (OSError, UnicodeDecodeError) as error:
        raise RefusedError([f"{path} cannot be read as UTF-8 text: {error}"]) from error


def _load_seeds(folder: Path) -> list[tuple[Path, DeltaHessianNetwork, weights.Recipe]]:
    """Load every seed of a release folder; a file that is missing or damaged is a refusal."""
    loaded = []
    for path in sorted(folder.glob(SEED_FILES)):
        try:
            network, recipe = weights.load(path)
        except ValidationError as invalid:
            raise RefusedError(_field_reasons(path, invalid)) from invalid
        except (OSError, ValueError, RuntimeError, SafetensorError) as error:
            # torch reports tensors that do not fit the recipe's network as a RuntimeError
            raise RefusedError([f"{path}: {error}"]) from error
        if path.name != f"seed-{recipe.seed}.json":
            raise RefusedError([f"{path} holds the recipe of seed {recipe.seed}"])
        loaded.append((path, network, recipe))
    return loaded


def bundled() -> Path:
    """Return the folder of the release that ships inside the package."""
    return Path(__file__).parent / "releases" / f"v{BUNDLED_VERSION}"


def load(folder: Path) -> Release:
    """Load a release folder.

    Raises:
        RefusedError: A file is missing, or the parts do not belong together.
    """
    missing = [name for name in (EVALUATION_FILE, TRAINING_FILE) if not (folder / name).is_file()]
    if missing:
        raise RefusedError([f"{folder}: no {name}" for name in missing])
    evaluation = _record(Evaluation, folder, EVALUATION_FILE)
    training = _record(Training, folder, TRAINING_FILE)
    invariance = None
    if (folder / INVARIANCE_FILE).exists():
        invariance = _record(Invariance, folder, INVARIANCE_FILE)
    loaded = _load_seeds(folder)
    recipes = tuple(recipe for _, _, recipe in loaded)
    reasons = _reasons_to_refuse(recipes, evaluation, training, invariance)
    if reasons:
        raise RefusedError([f"{folder}: {reason}" for reason in reasons])
    return Release(
        networks=tuple(network for _, network, _ in loaded),
        recipes=recipes,
        tensor_sha256=tuple(weights.recorded_sha256(path) for path, _, _ in loaded),
        evaluation=evaluation,
        training=training,
        invariance=invariance,
    )
