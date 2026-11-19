"""A network release: the seeds' weights with what is known about them.

A release is a folder:

    seed-0.safetensors, seed-0.json    one pair per seed (``aromir.network.weights``)
    evaluation.json                    the version's measured error per family and hold-out
    training.json                      the elements, sizes and molecules it was trained on
    invariance.json                    how far its answer moves under a rotation, a
                                       reflection and another atom listing (optional)
    measured.json                      the output of ``aromir evaluate`` on the hold-outs,
                                       which the evaluation's tables are (optional)
    release.json, measurement.json     present together, when the seeds are release
                                       networks: which networks the folder ships, the
                                       checks that let them ship, and the training record
                                       of the measurement networks the errors belong to

A version can be a pair trained on one recipe. Its measurement networks are trained
without the hold-outs, and every error and the licence are read on them; its release
networks are trained on every molecule with labels, hold-outs included, and are what the
folder ships. They have no held-out error of their own, and the certificate says so.

The JSON files are written from the training project's own records. Nothing the
certificate states about a network is typed into this package.

One release ships inside the package (``bundled``), so a corrected spectrum needs no
download and works on a machine without a network connection.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Final, Literal, Self, TypeVar

import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    field_validator,
    model_validator,
)
from safetensors import SafetensorError

from aromir.admission import standard_smiles
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
RELEASE_FILE: Final = "release.json"
MEASUREMENT_FILE: Final = "measurement.json"
RECORDED_FILES: Final = (EVALUATION_FILE, TRAINING_FILE, MEASUREMENT_FILE, MEASURED_FILE)
"""The records ``release.json`` binds by the hash of their content, beside the seeds' tensors."""
RELEASE_CHECKS: Final = (
    "same_recipe",
    "pool_agreement",
    "holdouts_learnt",
    "early_stopping",
    "benzene",
)
"""The checks release networks pass before they ship, as the training project names them."""
_INTERVAL_TOLERANCE_CM: Final = 0.01
"""A record holds two decimals, so an rms may lie that far outside its rounded interval."""
_SHARE_AT_A_PERCENTILE: Final = 0.97
"""A 97.5th percentile below the line puts 97.5 % of the resamples within it, 0.97 in two
decimals. Below, not at: a recorded 3.00 may be a 3.004, which is above a line of 3."""
Hex16 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{16}$")]
Hex64 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
SEED_FILES: Final = "seed-*.json"
MIN_SEEDS: Final = 2
"""A band's spread over the seeds needs at least two of them."""
_MEAN_TOLERANCE_CM: Final = 0.011
"""A record holds two decimals: a mean of rounded seeds may differ from the rounded mean."""
_Record = TypeVar("_Record", bound=BaseModel)
BUNDLED_VERSION: Final = "1.4"
"""The network version whose release ships inside the package."""
TEST_SETS: Final = {"a": "unseen molecules", "b": "held-out scaffolds"}
"""What each hold-out holds, in a reader's words. Readers are told of test sets; code and
records keep the hold-outs."""


def seed_name(seed: int) -> str:
    """Return the name a seed's files and records go by: ``seed-0``."""
    return f"seed-{seed}"


def named_test_set(name: str, *, first: bool = False) -> str:
    """Name a hold-out for a reader: "test set (a)", and at first use what it holds.

    A hold-out without an entry in ``TEST_SETS`` is named by its letter alone.
    """
    named = f"test set ({name})"
    kind = TEST_SETS.get(name)
    return f"{named}: {kind}" if first and kind is not None else named


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
        n_molecules: The hold-out's molecules with at least one band of the family: those
            the rms rests on. The resampling draws from all of the hold-out's molecules, and a
            draw without such a molecule counts as an rms of zero.
        interval_cm: The 2.5th and 97.5th percentile of that rms when the hold-out's
            molecules are resampled with replacement.
        share_of_resamples_within_line: The share of those resamples below the licence line.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    rms_cm: float = Field(ge=0, allow_inf_nan=False)
    n_modes: int = Field(ge=1)
    n_molecules: int = Field(ge=1)
    interval_cm: tuple[float, float]
    share_of_resamples_within_line: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def _an_interval(self) -> Self:
        low, high = self.interval_cm
        if not (math.isfinite(low) and math.isfinite(high) and 0 <= low <= high):
            msg = f"{self.interval_cm} is no interval of an rms error"
            raise ValueError(msg)
        if not low - _INTERVAL_TOLERANCE_CM <= self.rms_cm <= high + _INTERVAL_TOLERANCE_CM:
            msg = f"an rms of {self.rms_cm} outside its own resampling interval {self.interval_cm}"
            raise ValueError(msg)
        if self.n_molecules > self.n_modes:
            msg = f"bands on {self.n_molecules} molecules, and only {self.n_modes} bands"
            raise ValueError(msg)
        return self


class Evaluation(BaseModel):
    """A network version's measured error, as its evaluation records state it.

    Two quantities are kept apart by name (``aromir.evaluation`` defines both): the band
    error of the seeds' mean, which is what the product delivers (for release networks, the
    error bar of it, measured on their measurement networks) and what the licence is
    read on, and the trainer's first-order error per seed.

    Attributes:
        record_version: The format of this record; 2 is the first that names both, 3 the
            first that counts, per family, the molecules with a band of it.
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

    record_version: Literal[3]
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
                "format 3"
            )
            raise ValueError(msg)
        if isinstance(record, dict) and record.get("record_version") == 2:
            msg = (
                "a record of format 2: it does not count the molecules each family's band "
                "error rests on, so a caveat would count the whole hold-out. Measure the "
                "release with `aromir evaluate` and write a record of format 3"
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
    def _shares_fit_the_line(self) -> Self:
        line = self.licence_line_cm
        for family, per_holdout in self.band_error_seed_mean_cm.items():
            for holdout, error in per_holdout.items():
                low, high = error.interval_cm
                share = error.share_of_resamples_within_line
                if (high < line and share < _SHARE_AT_A_PERCENTILE) or (
                    low > line and share > 1 - _SHARE_AT_A_PERCENTILE
                ):
                    msg = (
                        f"{family} on ({holdout}): {share:.0%} of the resamples within the line "
                        f"of {line:g} cm⁻¹ does not fit the interval {error.interval_cm}"
                    )
                    raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _no_more_molecules_than_the_holdout(self) -> Self:
        for family, per_holdout in self.band_error_seed_mean_cm.items():
            for holdout, error in per_holdout.items():
                if error.n_molecules > self.holdouts[holdout].n_molecules:
                    msg = (
                        f"{family} on ({holdout}): bands on {error.n_molecules} molecules of a "
                        f"hold-out of {self.holdouts[holdout].n_molecules}"
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
                    f"'{family}' is within the licence line on {named_test_set(holdout)} at "
                    f"{error.rms_cm:.2f} cm⁻¹, {self._resting_on(family, holdout)}: "
                    "resampling the test set's molecules puts the error between "
                    f"{low:.2f} and {high:.2f} "
                    f"cm⁻¹, and {1 - error.share_of_resamples_within_line:.0%} of the resamples "
                    "lie above the line"
                )
        return sentences

    def _resting_on(self, family: Family, holdout: str) -> str:
        """Say how many of a hold-out's molecules a family's band error rests on."""
        with_bands = self.band_error_seed_mean_cm[family][holdout].n_molecules
        of = self.holdouts[holdout].n_molecules
        if with_bands == of:
            return f"on {of} molecules"
        return (
            f"on {with_bands} of the test set's {of} molecules (the others have no '{family}' band)"
        )

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


class CationTest(BaseModel):
    """What training on a neutral molecule's radical cation was seen to do to that molecule.

    Attributes:
        molecule: The neutral molecule the test was made on.
        without_cm: Its corrected positions' rms error, cm⁻¹, for networks trained without
            the radical cations.
        with_cm: The same for networks trained with them.
        n_networks: The networks the test was made with.
        source: The record the numbers were read from.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    molecule: str = Field(min_length=1)
    without_cm: float = Field(ge=0, allow_inf_nan=False)
    with_cm: float = Field(ge=0, allow_inf_nan=False)
    n_networks: int = Field(ge=1)
    source: str = Field(min_length=1)


def _refuse_nonstandard(smiles: list[str]) -> None:
    """Raise for the first entry that is not in the standard form admission writes.

    Raises:
        ValueError: An entry does not parse, or admission would write it otherwise.
    """
    for entry in smiles:
        standard = standard_smiles(entry)
        if standard != entry:
            msg = f"{entry!r} is not in the standard form admission writes ({standard!r})"
            raise ValueError(msg)


class Training(BaseModel):
    """What a network version was trained on.

    Attributes:
        network_version: The version the set belongs to.
        elements: The elements of the training molecules.
        min_atoms: The smallest training molecule's atom count.
        max_atoms: The largest's.
        trained_smiles: Canonical SMILES of every neutral molecule a seed fitted or
            stopped on.
        held_out_smiles: Canonical SMILES of the hold-out molecules, by hold-out.
        soft_mode_repair: Whether the training Hessians had their soft modes repaired; a
            molecule's Hessian is then prepared the same way before the network reads it.
        cation_parent_smiles: Canonical SMILES of the neutral parents of the radical
            cations a seed fitted or stopped on; None where the record does not say.
        cation_test: What such training was seen to do to a neutral parent; given exactly
            when there are such cations.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    network_version: str
    elements: list[str] = Field(min_length=1)
    min_atoms: int = Field(ge=1)
    max_atoms: int = Field(ge=1)
    trained_smiles: list[str]
    held_out_smiles: dict[str, list[str]]
    soft_mode_repair: bool
    cation_parent_smiles: list[str] | None = None
    cation_test: CationTest | None = None

    # The 'seen' look-up compares an admitted SMILES with these lists, and admission writes
    # a molecule's standard form: an entry in another form could never be found, and
    # rewriting it could merge two entries unseen, so it is refused.
    @field_validator("trained_smiles", "cation_parent_smiles", mode="after")
    @classmethod
    def _standard_list(cls, smiles: list[str] | None) -> list[str] | None:
        _refuse_nonstandard(smiles or [])
        return smiles

    @field_validator("held_out_smiles", mode="after")
    @classmethod
    def _standard_lists(cls, held_out: dict[str, list[str]]) -> dict[str, list[str]]:
        for smiles in held_out.values():
            _refuse_nonstandard(smiles)
        return held_out

    @model_validator(mode="after")
    def _a_set_that_can_exist(self) -> Self:
        unknown = sorted(set(self.elements) - set(ELEMENTS))
        if unknown:
            msg = f"element(s) {', '.join(unknown)} that the package does not know"
            raise ValueError(msg)
        if self.min_atoms > self.max_atoms:
            msg = f"the smallest molecule has {self.min_atoms} atoms, the largest {self.max_atoms}"
            raise ValueError(msg)
        if bool(self.cation_parent_smiles) != (self.cation_test is not None):
            msg = (
                "a record that names radical cations in the training data carries the test "
                "of what they do to their neutral parents, and only such a record does"
            )
            raise ValueError(msg)
        return self

    def n_held_out(self, holdout: str) -> int:
        """Count a hold-out's molecules that no seed trained on under another identifier."""
        return len(set(self.held_out_smiles[holdout]) - set(self.trained_smiles))


class ReleaseNetworks(BaseModel):
    """Which networks a folder ships when they are not the ones its errors were measured on.

    Attributes:
        network_version: The version the pair belongs to.
        n_entries: The labelled entries of the training pool a release network fitted or
            stopped on: a neutral molecule may be an entry under more than one identifier,
            and each radical cation of the training record is one entry of its own (the card
            counts the neutral entries as the rest).
        fitted_share: The share of them each network was fitted on; the others stopped its
            training, a different draw for each seed.
        checks: Each check the release networks had to pass before they ship, by name:
            exactly those of ``RELEASE_CHECKS``.
        seed_sha256: The SHA-256 of each shipped seed's tensor file, by seed name
            (``seed-0``): what binds this record to the weights beside it.
        record_sha256: The SHA-256 of each record of ``RECORDED_FILES``, of its content
            (``record_sha256``): what binds the records to this one.
        release_checkpoints: The training checkpoints of the shipped seeds, by file name,
            with the first 16 hex digits of their SHA-256.
        measurement_checkpoints: The same for the measurement networks, which are not
            shipped.
        source: The record of the checks.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    network_version: str
    n_entries: int = Field(ge=1)
    fitted_share: float = Field(gt=0, lt=1)
    checks: dict[str, bool] = Field(min_length=1)
    seed_sha256: dict[str, Hex64] = Field(min_length=1)
    record_sha256: dict[str, Hex64]
    release_checkpoints: dict[str, Hex16] = Field(min_length=1)
    measurement_checkpoints: dict[str, Hex16] = Field(min_length=1)
    source: str = Field(min_length=1)

    @model_validator(mode="after")
    def _named_as_expected(self) -> Self:
        for what, named, expected in (
            ("release check(s)", self.checks, RELEASE_CHECKS),
            ("record(s) by hash", self.record_sha256, RECORDED_FILES),
        ):
            missing, unknown = set(expected) - set(named), set(named) - set(expected)
            if missing or unknown:
                msg = (
                    f"{what}: {sorted(missing)} missing and {sorted(unknown)} unknown; "
                    f"expected exactly {list(expected)}"
                )
                raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _every_check_passed(self) -> Self:
        failed = sorted(name for name, passed in self.checks.items() if not passed)
        if failed:
            msg = (
                f"release check(s) {', '.join(failed)} failed: such release networks are not "
                "shipped, and the version's measurement networks are the product"
            )
            raise ValueError(msg)
        return self

    @property
    def stopping_share(self) -> float:
        """The share of the molecules that stopped a network's training instead of fitting it."""
        return 1 - self.fitted_share


_SUPERSCRIPT: Final = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")


def pool(training: Training) -> str:
    """Count a training record's molecules: neutral ones and the parents of radical cations."""
    cations = len(training.cation_parent_smiles or ())
    neutral = f"{len(training.trained_smiles)} distinct neutral molecules by SMILES"
    if not cations:
        return neutral
    return f"{neutral} and the radical cations of {cations} molecule{'' if cations == 1 else 's'}"


def made_by(release: Release) -> str:
    """Say which networks make a release's spectrum and which ones its errors belong to.

    For networks whose errors were measured on themselves this is a clause that ends a
    sentence; for release networks it is sentences of their own.
    """
    shipped = release.shipped
    if shipped is None:
        return "with the error measured on molecules the network never saw."
    n = len(release.networks)
    held = shipped.stopping_share
    return (
        f"The spectrum is made by the release networks of version {release.network_version}, "
        f"trained on the {shipped.n_entries} labelled entries of their training pool, the "
        f"test sets' molecules included ({pool(release.training)}): each of the {n} networks "
        f"is fitted on a random {100 * shipped.fitted_share:.0f} % of the entries and stopped "
        f"on the other {100 * held:.0f} %, a different {100 * held:.0f} % per seed, so not every "
        "entry is fitted by every network (with "
        f"{n} networks about {held:g}{str(n).translate(_SUPERSCRIPT)} ≈ "
        f"{100 * held**n:.1f} % by none). Its error bars were measured on the measurement "
        "networks of the same version — the same recipe trained without the test sets' "
        "molecules — on molecules those networks never saw; the release networks themselves "
        "have seen the test sets' molecules, so they have no out-of-sample error of their own."
    )


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
    """A loaded release: its networks, their recipes and file hashes, its folder and records."""

    networks: tuple[DeltaHessianNetwork, ...]
    recipes: tuple[weights.Recipe, ...]
    tensor_sha256: tuple[str, ...]
    folder: Path
    """The folder the release was read from, as an absolute path."""
    evaluation: Evaluation
    training: Training
    invariance: Invariance | None = None
    shipped: ReleaseNetworks | None = None
    measurement: Training | None = None

    def __post_init__(self) -> None:
        """Refuse release networks without their measurement record, or the reverse."""
        if (self.shipped is None) != (self.measurement is None):
            msg = "release networks come with the measurement networks' record, and only they do"
            raise ValueError(msg)

    @property
    def network_version(self) -> str:
        """The version every part of the release belongs to."""
        return self.evaluation.network_version

    @property
    def seed_sha256(self) -> dict[str, str]:
        """Return each seed's tensor hash by seed name, as ``release.json`` binds them."""
        return {
            seed_name(recipe.seed): sha256
            for recipe, sha256 in zip(self.recipes, self.tensor_sha256, strict=True)
        }

    @property
    def is_bundled(self) -> bool:
        """Tell whether the release is the one inside the package."""
        return self.folder == bundled().resolve()

    @property
    def measured_training(self) -> Training:
        """Return the training record the errors belong to.

        That is the measurement networks' where the seeds are release networks, else the
        seeds' own.
        """
        return self.training if self.measurement is None else self.measurement

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
    reasons += _seeds_that_disagree(recipes, evaluation)
    return reasons + _records_that_disagree(evaluation, training)


def _pair_reasons(
    seeds: dict[str, str],
    evaluation: Evaluation,
    training: Training,
    pair: tuple[ReleaseNetworks, Training] | None,
) -> list[str]:
    """Name where release networks and the measurement networks of their errors do not fit.

    ``seeds`` is each seed's tensor hash by seed name; the measurement record's version is
    checked with the other parts' (``_reasons_to_refuse``).
    """
    if pair is None:
        return []
    shipped, measurement = pair
    reasons = []
    others = {shipped.network_version, training.network_version} - {evaluation.network_version}
    if others:
        reasons.append(
            f"{RELEASE_FILE} or {TRAINING_FILE} is of version {sorted(others)}, the evaluation "
            f"of {evaluation.network_version}"
        )
    if training.held_out_smiles:
        reasons.append(f"{TRAINING_FILE} lists hold-outs, and release networks hold nothing out")
    lacking = set(measurement.trained_smiles) - set(training.trained_smiles)
    held_out = {s for smiles in measurement.held_out_smiles.values() for s in smiles}
    for what, missing in (
        ("molecule(s) the measurement networks trained on", lacking),
        ("hold-out molecule(s)", held_out - set(training.trained_smiles)),
    ):
        if missing:
            reasons.append(
                f"the release networks lack {len(missing)} {what}: release networks train on "
                "the measurement set and the hold-outs"
            )
    if shipped.seed_sha256 != seeds:
        reasons.append(
            f"{RELEASE_FILE} names other weights than the seeds beside it: it is the record of "
            "other networks"
        )
    if len(shipped.release_checkpoints) != len(seeds):
        reasons.append(
            f"{RELEASE_FILE} names {len(shipped.release_checkpoints)} checkpoint(s), the "
            f"folder holds {len(seeds)} seed(s)"
        )
    return (
        reasons
        + _scope_reasons(training, measurement)
        + _entry_reasons(shipped, training, evaluation)
    )


def _scope_reasons(training: Training, measurement: Training) -> list[str]:
    """Name where the measurement networks covered another domain than the release networks."""
    compared = {
        "elements": (sorted(measurement.elements), sorted(training.elements)),
        "min_atoms": (measurement.min_atoms, training.min_atoms),
        "max_atoms": (measurement.max_atoms, training.max_atoms),
        "soft_mode_repair": (measurement.soft_mode_repair, training.soft_mode_repair),
    }
    reasons = [
        f"{MEASUREMENT_FILE} gives {name} {mine}, {TRAINING_FILE} {theirs}: the error bars "
        "would belong to networks of another domain"
        for name, (mine, theirs) in compared.items()
        if mine != theirs
    ]
    if measurement.cation_test is not None and measurement.cation_test != training.cation_test:
        reasons.append(
            f"{MEASUREMENT_FILE} gives another cation test than {TRAINING_FILE}: the certificate "
            "would state one the error bars do not belong to"
        )
    lacking = set(measurement.cation_parent_smiles or ()) - set(training.cation_parent_smiles or ())
    if lacking:
        reasons.append(
            f"the release networks lack {len(lacking)} radical cation(s) the measurement "
            "networks trained on"
        )
    return reasons


def _entry_reasons(
    shipped: ReleaseNetworks, training: Training, evaluation: Evaluation
) -> list[str]:
    """Name a count of entries or a measurement checkpoint that the other records contradict."""
    reasons = []
    least = len(set(training.trained_smiles)) + len(training.cation_parent_smiles or ())
    if shipped.n_entries < least:
        reasons.append(
            f"{RELEASE_FILE} counts {shipped.n_entries} entries, fewer than the {least} "
            f"molecules and cations {TRAINING_FILE} lists"
        )
    unnamed = sorted(
        name
        for name in shipped.measurement_checkpoints
        if not any(name in source for source in evaluation.sources)
    )
    if unnamed:
        reasons.append(
            f"{EVALUATION_FILE} does not name {', '.join(unnamed)} as its source: its errors "
            "may be other networks'"
        )
    return reasons


def record_sha256(path: Path) -> str:
    """Return the SHA-256 of a JSON record's content, written out in one canonical way.

    The content as Python's JSON reader gives it is hashed, not the bytes: line endings,
    whitespace, key order and escapes do not change it; a number written with another type
    does (1 and 1.0 differ).
    """
    content = json.loads(path.read_text(encoding="utf-8"))
    canonical = json.dumps(content, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _record_hash_reasons(folder: Path, shipped: ReleaseNetworks) -> list[str]:
    """Name a record that is not the one ``release.json`` was written with."""
    reasons = []
    for name in RECORDED_FILES:
        try:
            digest = record_sha256(folder / name)
        except (OSError, UnicodeDecodeError, ValueError) as error:
            reasons.append(f"{name}, which {RELEASE_FILE} binds by hash, cannot be read: {error}")
            continue
        if digest != shipped.record_sha256[name]:
            reasons.append(
                f"{name} is not the record {RELEASE_FILE} was written with: it was changed after"
            )
    return reasons


def _measured_reasons(folder: Path, evaluation: Evaluation) -> list[str]:
    """Name an evaluation record whose tables are not what ``aromir evaluate`` wrote beside it."""
    path = folder / MEASURED_FILE
    if not path.exists():
        return []
    try:
        measured = json.loads(path.read_text(encoding="utf-8"))
        tables = record_tables(measured)
        again = Evaluation.model_validate(evaluation.model_dump(mode="json") | tables)
        sizes = {name: holdout["n_molecules"] for name, holdout in measured.items()}
    except (OSError, UnicodeDecodeError, ValueError, KeyError, TypeError, RefusedError) as error:
        return [f"{MEASURED_FILE} cannot be read as the output of `aromir evaluate`: {error}"]
    if sizes != {name: holdout.n_molecules for name, holdout in evaluation.holdouts.items()}:
        return [
            f"{MEASURED_FILE} was measured on hold-outs of other sizes than {EVALUATION_FILE} gives"
        ]
    if again == evaluation:
        return []
    reason = (
        f"{EVALUATION_FILE} holds other errors than {MEASURED_FILE}, the measurement it is "
        "written from"
    )
    return [reason]


def record_tables(measured: dict[str, Any]) -> dict[str, Any]:
    """Return what ``evaluate`` measured as the tables an evaluation record holds.

    A record lists every family on every hold-out, so a hold-out without a band of some
    family cannot carry a record.

    Raises:
        RefusedError: A family has no band in a hold-out.
    """
    empty = [
        f"hold-out ({name}) has no {family} band"
        for name, holdout in measured.items()
        for family, entry in holdout["band_error_seed_mean"].items()
        if not entry["n_modes"]
    ]
    if empty:
        raise RefusedError(empty)
    names = {
        "band_error_seed_mean_cm": "band_error_seed_mean",
        "first_order_error_per_seed_cm": "first_order_error_per_seed",
        "first_order_error_per_seed_mean_cm": "first_order_error_per_seed_mean",
    }
    return {
        field: {
            family: {name: holdout[key][family] for name, holdout in measured.items()}
            for family in FAMILIES
        }
        for field, key in names.items()
    }


def _pair(folder: Path) -> tuple[ReleaseNetworks, Training] | None:
    """Read a folder's release and measurement records; one without the other is a refusal."""
    present = [name for name in (RELEASE_FILE, MEASUREMENT_FILE) if (folder / name).exists()]
    if not present:
        return None
    if len(present) == 1:
        other = MEASUREMENT_FILE if present == [RELEASE_FILE] else RELEASE_FILE
        raise RefusedError([f"{folder}: {present[0]} without {other}: release networks need both"])
    shipped = _record(ReleaseNetworks, folder, RELEASE_FILE)
    return shipped, _record(Training, folder, MEASUREMENT_FILE)


def _seeds_that_disagree(recipes: tuple[weights.Recipe, ...], evaluation: Evaluation) -> list[str]:
    """Name an evaluation that was measured on another number of seeds than the release holds."""
    counted = {
        len(seeds)
        for per_holdout in evaluation.first_order_error_per_seed_cm.values()
        for seeds in per_holdout.values()
    }
    if counted == {len(recipes)}:
        return []
    reason = (
        f"{EVALUATION_FILE} gives the first-order error of {sorted(counted)} seed(s), the "
        f"release holds {len(recipes)}: the record was measured on another set of seeds"
    )
    return [reason]


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
        if path.name != f"{seed_name(recipe.seed)}.json":
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
    pair = _pair(folder)
    measured = training if pair is None else pair[1]
    loaded = _load_seeds(folder)
    recipes = tuple(recipe for _, _, recipe in loaded)
    seeds = {seed_name(recipe.seed): weights.recorded_sha256(path) for path, _, recipe in loaded}
    reasons = _reasons_to_refuse(recipes, evaluation, measured, invariance)
    reasons += _pair_reasons(seeds, evaluation, training, pair)
    reasons += _measured_reasons(folder, evaluation)
    if pair is not None:
        reasons += _record_hash_reasons(folder, pair[0])
    if reasons:
        raise RefusedError([f"{folder}: {reason}" for reason in reasons])
    return Release(
        networks=tuple(network for _, network, _ in loaded),
        recipes=recipes,
        tensor_sha256=tuple(seeds.values()),
        folder=folder.resolve(),
        evaluation=evaluation,
        training=training,
        invariance=invariance,
        shipped=None if pair is None else pair[0],
        measurement=None if pair is None else pair[1],
    )
