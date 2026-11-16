"""From one molecule to its spectrum, written to a result folder.

A molecule outside the domain is refused before the first computation, and a result is
never overwritten. Each step is saved when it ends — the optimised geometry, the Hessian,
the measured soft modes — so that a failure after hours of quantum chemistry loses none of
them: the same request into the same folder takes up where the run stopped.

A saved step is one file, written through a temporary file and moved into place, that
says of itself which attempt and which geometry it belongs to. No two files have to agree
for a step to count, so a run killed between two writes resumes from a step that is
whole or not there.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from dataclasses import asdict, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Self, TextIO

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from aromir import (
    admission,
    certificate,
    geometry,
    lowlevel,
    optimisation,
    records,
    repair,
    response,
    spectrum,
    vibrations,
)
from aromir.dft import BASIS, FUNCTIONAL, GRID, Resources
from aromir.errors import AromirError, NotAMinimumError, RefusedError
from aromir.molecule import Geometry

if TYPE_CHECKING:
    from aromir.release import Release

GEOMETRY_FILE: Final = "geometry.json"
LOW_LEVEL_FILE: Final = "lowlevel.npz"
CORRECTION_FILE: Final = "correction.npz"
BASELINE_FILE: Final = "baseline.json"
CERTIFICATE_FILE: Final = "certificate.json"
CERTIFICATE_TEXT_FILE: Final = "certificate.md"
QUANTUM_LOG_FILE: Final = "pyscf.log"
RUN_FILE: Final = "run.json"
"""What the run was asked, and whether it has stepped off a saddle point."""
SADDLE_POINT_PREFIX: Final = "saddle_point_"
"""The name a saddle point's geometry and Hessian are kept under."""

OFF_THE_SADDLE_BOHR: Final = 0.1
"""How far the atom that moves most is displaced along an imaginary mode before the one
second optimisation. On hydrogen peroxide's planar saddle point (B3LYP/6-31G*) this step
leads the optimiser to the skew minimum; a flatter saddle may need more, and the run then
ends as not a minimum, with everything kept."""

_LEFTOVERS: Final = (QUANTUM_LOG_FILE, f"{RUN_FILE}.tmp")
"""What a run killed before its first step leaves behind: no reason to refuse the folder."""


class MoleculeInput(BaseModel):
    """A molecule as the user gives it: a SMILES string or an xyz file, never both."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    smiles: str | None = None
    xyz: Path | None = None

    @model_validator(mode="after")
    def _exactly_one(self) -> Self:
        if (self.smiles is None) == (self.xyz is None):
            msg = "give exactly one of a SMILES and an xyz file"
            raise ValueError(msg)
        return self


class Request(BaseModel):
    """One molecule to compute, and where its result goes."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    molecule: MoleculeInput
    out: Path
    resources: Resources = Resources()
    outside_training_sizes: bool = False
    """Compute a corrected spectrum although no band of it can be licensed."""


class _SaddlePoint(BaseModel):
    """The saddle point a run has stepped off, and where its second optimisation starts."""

    model_config = ConfigDict(extra="forbid")

    imaginary_cm: float
    restart: list[list[float]]


class _Run(BaseModel):
    """The state of a run that is not in its step files: the request and the saddle step."""

    model_config = ConfigDict(extra="forbid")

    asked: dict[str, Any]
    left_saddle_point: _SaddlePoint | None = None

    @property
    def attempt(self) -> int:
        """1 for the optimisation from the start geometry, 2 for the one off a saddle."""
        return 1 if self.left_saddle_point is None else 2

    def save(self, folder: Path) -> None:
        records.write_json(folder / RUN_FILE, self.model_dump(mode="json"))


def admit(molecule: MoleculeInput) -> tuple[admission.Admission, Geometry]:
    """Parse and check a molecule, and build its start geometry in the canonical order.

    Raises:
        RefusedError: The molecule is outside the supported domain, or does not parse.
    """
    if molecule.smiles is not None:
        admitted = admission.admit(admission.parse_smiles(molecule.smiles))
        return admitted, geometry.start_from_smiles(admitted.smiles)
    if molecule.xyz is None or not molecule.xyz.is_file():
        raise RefusedError([f"{molecule.xyz}: no such xyz file"])
    parsed = admission.parse_xyz(molecule.xyz)
    return admission.admit(parsed), geometry.start_from_xyz(parsed)


def _asked(command: str, admitted: admission.Admission, start: Geometry) -> dict[str, Any]:
    """Return what identifies a run: the command, the molecule, its start and the level."""
    # Adding 0.0 turns a rounded -0.0 into 0.0: the two differ in their bytes only.
    coordinates = (np.round(start.coords_bohr, 6) + 0.0).tobytes()
    start_hash = hashlib.sha256(" ".join(start.symbols).encode() + coordinates).hexdigest()
    return {
        "command": command,
        "smiles": admitted.smiles,
        "start_geometry_sha256": start_hash,
        "level": {"functional": FUNCTIONAL, "basis": BASIS, "grid": list(GRID)},
    }


def _damaged(folder: Path, what: str, error: Exception) -> RefusedError:
    reason = (
        f"{what} in {folder} cannot be read ({type(error).__name__}: {error}): the folder "
        "holds a damaged run; delete it to start over"
    )
    return RefusedError([reason])


def _open_folder(out: Path, asked: dict[str, Any]) -> tuple[Path, _Run]:
    """Return the result folder and the state of the run it holds, new or to be resumed.

    Raises:
        RefusedError: The folder holds a result, another request's run, a damaged run or
            other files.
    """
    folder = out.resolve()
    if folder.exists() and not folder.is_dir():
        raise RefusedError([f"{folder} is a file: --out names a folder"])
    held = [] if not folder.exists() else [path.name for path in folder.iterdir()]
    if all(name in _LEFTOVERS for name in held):
        return folder, _Run(asked=asked)
    if BASELINE_FILE in held or CERTIFICATE_FILE in held:
        raise RefusedError([f"{folder} holds a result: a result is never overwritten"])
    if RUN_FILE not in held:
        raise RefusedError([f"{folder} is not empty: a result is never overwritten"])
    try:
        state = _Run.model_validate_json((folder / RUN_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError, ValidationError) as error:
        raise _damaged(folder, RUN_FILE, error) from error
    if state.asked != asked:
        differing = sorted(key for key in asked if state.asked.get(key) != asked[key])
        reason = (
            f"{folder} holds the unfinished run of another request "
            f"({state.asked.get('command')} {state.asked.get('smiles')}; it differs in "
            f"{', '.join(differing)}): only the same command on the same molecule, start "
            "geometry and level takes it up"
        )
        raise RefusedError([reason])
    return folder, state


def _optimised(
    start: Geometry, folder: Path, state: _Run, resources: Resources, log: TextIO
) -> optimisation.Optimisation:
    """Return the optimised geometry: the saved one of this attempt, or a new one, saved."""
    path = folder / GEOMETRY_FILE
    if path.is_file():
        try:
            stored = json.loads(path.read_text(encoding="utf-8"))
            if stored["attempt"] == state.attempt:
                minimum = Geometry(tuple(stored["symbols"]), np.array(stored["coords_bohr"]))
                return optimisation.Optimisation(minimum, **stored["optimisation"])
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise _damaged(folder, GEOMETRY_FILE, error) from error
    optimised = optimisation.optimise(start, resources, log)
    record = optimised.geometry.to_dict() | {
        "made_with": records.provenance(),
        "optimisation": {
            "energy": optimised.energy,
            "max_force": optimised.max_force,
            "n_steps": optimised.n_steps,
        },
        "attempt": state.attempt,
    }
    records.write_json(path, record)
    return optimised


def _save_low_level(
    folder: Path,
    minimum: Geometry,
    attempt: int,
    low: lowlevel.LowLevel,
    made_with: dict[str, Any],
) -> None:
    """Write the whole low-level step as one file: arrays, checks, and whose it is.

    ``made_with`` names the run that computed the analytic Hessian and, once the soft
    modes are measured, the run that measured them.
    """
    measured = "soft modes" in made_with
    arrays: dict[str, Any] = {
        "hessian": low.hessian,
        "polar_tensor": low.polar_tensor,
        "coords_bohr": minimum.coords_bohr,
        "attempt": np.array(attempt),
        "soft_modes_measured": np.array(measured),
        "energy": np.array(low.energy),
        "refinement": np.array(list(asdict(low.refinement).values()), dtype=float),
        "sum_rule_violation": np.array(low.sum_rule_violation),
        "seconds": np.array(low.seconds),
        "made_with": np.array(json.dumps(made_with)),
    }
    if low.analytic_hessian is not None:
        arrays["analytic_hessian"] = low.analytic_hessian
    if low.soft_modes is not None:
        arrays["soft_modes"] = np.array(list(asdict(low.soft_modes).values()), dtype=float)
    records.write_arrays(folder / LOW_LEVEL_FILE, arrays)


def _saved_low_level(
    folder: Path, minimum: Geometry, attempt: int
) -> tuple[lowlevel.LowLevel, dict[str, Any]] | None:
    """Return this attempt's saved low level of this geometry and what made it.

    None when there is none: no file, another attempt's, or another geometry's.
    """
    path = folder / LOW_LEVEL_FILE
    if not path.is_file():
        return None
    try:
        with np.load(path) as arrays:
            saved = {name: np.array(arrays[name]) for name in arrays.files}
        if int(saved["attempt"]) != attempt or not np.array_equal(
            saved["coords_bohr"], minimum.coords_bohr
        ):
            return None
        before, after, rounds, columns = saved["refinement"].tolist()
        soft_modes = None
        if "soft_modes" in saved:
            n_directions, *positions = saved["soft_modes"].tolist()
            soft_modes = repair.Repair(int(n_directions), *positions)
        low = lowlevel.LowLevel(
            energy=float(saved["energy"]),
            hessian=saved["hessian"],
            polar_tensor=saved["polar_tensor"],
            refinement=response.Refinement(before, after, int(rounds), int(columns)),
            sum_rule_violation=float(saved["sum_rule_violation"]),
            seconds=float(saved["seconds"]),
            soft_modes=soft_modes,
            analytic_hessian=saved.get("analytic_hessian"),
        )
        made_with: dict[str, Any] = json.loads(str(saved["made_with"]))
        expected = (3 * minimum.n_atoms, 3 * minimum.n_atoms)
        if low.hessian.shape != expected or low.polar_tensor.shape != (3, expected[0]):
            msg = f"arrays of shape {low.hessian.shape} and {low.polar_tensor.shape}"
            raise ValueError(msg)  # noqa: TRY301  a damaged file like the others below
        if bool(saved["soft_modes_measured"]) != ("soft modes" in made_with):
            msg = "its record of the soft-mode step contradicts itself"
            raise ValueError(msg)  # noqa: TRY301  the same
    except (OSError, ValueError, KeyError, TypeError, EOFError, zipfile.BadZipFile) as error:
        raise _damaged(folder, LOW_LEVEL_FILE, error) from error
    return low, made_with


def _low_level(
    minimum: Geometry, folder: Path, state: _Run, resources: Resources, log: TextIO
) -> lowlevel.LowLevel:
    """Return the low level with its soft modes measured, saving each of the two steps.

    The analytic Hessian is on disk before the repair starts, so a repair that fails —
    for want of memory, say — costs the repair alone.
    """
    saved = _saved_low_level(folder, minimum, state.attempt)
    if saved is None:
        low = lowlevel.compute(minimum, resources, log)
        made_with = {"low level": records.provenance()}
        _save_low_level(folder, minimum, state.attempt, low, made_with)
    else:
        low, made_with = saved
    if "soft modes" not in made_with:
        repaired, record = repair.repair(minimum, low.hessian, resources, log)
        if record is not None:
            low = replace(low, hessian=repaired, analytic_hessian=low.hessian, soft_modes=record)
        made_with |= {"soft modes": records.provenance()}
        _save_low_level(folder, minimum, state.attempt, low, made_with)
    return low


def _attempt(
    start: Geometry, folder: Path, state: _Run, resources: Resources, log: TextIO
) -> tuple[optimisation.Optimisation, lowlevel.LowLevel, vibrations.Modes]:
    left = state.left_saddle_point
    begin = start if left is None else Geometry(start.symbols, np.array(left.restart))
    optimised = _optimised(begin, folder, state, resources, log)
    low = _low_level(optimised.geometry, folder, state, resources, log)
    return optimised, low, vibrations.vibrational_modes(low.hessian, optimised.geometry)


def _step_off(folder: Path, state: _Run, saddle: Geometry, modes: vibrations.Modes) -> None:
    """Keep the saddle point's files under another name and record where to start again.

    The copies are made first and the state is written last: until then a resumed run
    finds the first attempt whole, and after it the first attempt's files no longer count,
    because they say they are of attempt 1.
    """
    for name in (GEOMETRY_FILE, LOW_LEVEL_FILE):
        (folder / f"{SADDLE_POINT_PREFIX}{name}").write_bytes((folder / name).read_bytes())
    step, _ = vibrations.displacement(modes.vectors[:, 0], saddle.masses_amu, OFF_THE_SADDLE_BOHR)
    state.left_saddle_point = _SaddlePoint(
        imaginary_cm=float(modes.wavenumbers[0]),
        restart=(saddle.coords_bohr + step).tolist(),
    )
    state.save(folder)


def _compute(
    start: Geometry, folder: Path, state: _Run, resources: Resources
) -> tuple[optimisation.Optimisation, lowlevel.LowLevel]:
    """Optimise and compute the low level, each step saved; step off a saddle point once.

    Whether the geometry is a minimum is read from the Hessian with its soft modes
    measured: the analytic one can make a soft mode imaginary that is not. From a saddle
    point the optimisation, the Hessian and the repair are run a second time, which
    doubles the cost and can still end without a minimum.

    Raises:
        NotAMinimumError: The geometry still has an imaginary mode after one step along it
            and a second optimisation.
    """
    folder.mkdir(parents=True, exist_ok=True)
    state.save(folder)
    with (folder / QUANTUM_LOG_FILE).open("a", encoding="utf-8") as log:
        optimised, low, modes = _attempt(start, folder, state, resources, log)
        if modes.n_imaginary and state.left_saddle_point is None:
            _step_off(folder, state, optimised.geometry, modes)
            optimised, low, modes = _attempt(start, folder, state, resources, log)
    if modes.n_imaginary:
        msg = (
            f"{modes.n_imaginary} imaginary mode(s), the lowest at {modes.wavenumbers[0]:.0f} "
            "cm⁻¹, also after one step along it and a second optimisation: no minimum was "
            f"found. The geometries and the Hessians are kept in {folder}"
        )
        raise NotAMinimumError(msg)
    return optimised, low


def _made_with(folder: Path) -> dict[str, Any]:
    """Return what made each saved step of a finished run, read from the step files."""
    steps: dict[str, Any] = {}
    for prefix in (SADDLE_POINT_PREFIX, ""):
        geometry_file = folder / f"{prefix}{GEOMETRY_FILE}"
        if not geometry_file.is_file():
            continue
        stored = json.loads(geometry_file.read_text(encoding="utf-8"))
        attempt = stored["attempt"]
        steps[f"optimisation {attempt}"] = stored["made_with"]
        with np.load(folder / f"{prefix}{LOW_LEVEL_FILE}") as arrays:
            made_with = json.loads(str(arrays["made_with"]))
        steps |= {f"{name} {attempt}": made for name, made in made_with.items()}
    return steps


def _computation_record(
    optimised: optimisation.Optimisation, low: lowlevel.LowLevel, state: _Run, folder: Path
) -> dict[str, Any]:
    left = state.left_saddle_point
    return {
        "level": {"functional": FUNCTIONAL, "basis": BASIS, "grid": list(GRID)},
        "optimisation": {
            "energy_hartree": optimised.energy,
            "max_force_hartree_per_bohr": optimised.max_force,
            "n_steps": optimised.n_steps,
            "left_saddle_point_at_cm": None if left is None else left.imaginary_cm,
        },
        "checks": {
            "response_residual_before": low.refinement.residual_before,
            "response_residual_after": low.refinement.residual_after,
            "response_batched_rounds": low.refinement.batched_rounds,
            "response_single_columns": low.refinement.single_columns,
            "sum_rule_violation_e": low.sum_rule_violation,
        },
        "soft_mode_repair": None if low.soft_modes is None else asdict(low.soft_modes),
        "seconds_low_level": round(low.seconds, 1),
        "steps": _made_with(folder),
    }


def _bands_record(bands: spectrum.Bands) -> list[dict[str, float]]:
    return [
        {"wavenumber_cm": round(float(position), 2), "intensity_km_mol": round(float(height), 3)}
        for position, height in zip(bands.wavenumbers, bands.intensities, strict=True)
    ]


def run_baseline(request: Request) -> dict[str, Any]:
    """Compute a molecule's low-level spectrum and write its result folder.

    Returns:
        The record written to ``baseline.json``.

    Raises:
        RefusedError: The molecule is outside the domain, or the folder holds a result or
            another request's unfinished run.
        ConvergenceError: The SCF or the optimisation did not converge.
        NotAMinimumError: No minimum was found.
        ResponseError: The dipole derivatives failed their checks.
    """
    admitted, start = admit(request.molecule)
    folder, state = _open_folder(request.out, _asked("baseline", admitted, start))
    launched = records.provenance()
    optimised, low = _compute(start, folder, state, request.resources)
    bands = spectrum.harmonic_bands(low.hessian, optimised.geometry, low.polar_tensor)
    record = {
        "molecule": {
            "smiles": admitted.smiles,
            "formula": admitted.formula,
            "n_atoms": admitted.n_atoms,
        },
        **_computation_record(optimised, low, state, folder),
        "harmonic": {
            "note": "B3LYP harmonic positions, uncorrected",
            "bands": _bands_record(bands),
        },
        "scaled": {
            "note": "baseline: harmonic positions times the scale factor of each region",
            "region_edges_cm": list(spectrum.SCALING_REGION_EDGES),
            "factors": list(spectrum.SCALING_FACTORS),
            "bands": _bands_record(spectrum.scaled(bands)),
        },
        "provenance": launched,
    }
    records.write_json(folder / BASELINE_FILE, record)
    return record


def _untrained_elements(start: Geometry, release: Release) -> list[str]:
    unseen = sorted(set(start.symbols) - set(release.training.elements))
    if not unseen:
        return []
    trained = ", ".join(sorted(release.training.elements))
    return [f"element(s) {', '.join(unseen)} outside the network's training set ({trained})"]


def _untrained_size(start: Geometry, release: Release) -> list[str]:
    training = release.training
    if training.min_atoms <= start.n_atoms <= training.max_atoms:
        return []
    reason = (
        f"{start.n_atoms} atoms: the network was trained on {training.min_atoms} to "
        f"{training.max_atoms}, so no band of this molecule would be licensed; "
        "--outside-training-sizes computes it all the same"
    )
    return [reason]


def _no_network_inputs(start: Geometry, release: Release) -> list[str]:
    """Name why the network could not read the molecule, found on its start geometry.

    The internal coordinates and their pseudo-inverse need no Hessian to fail, so they are
    built once on the start geometry, before hours of quantum chemistry are spent on a
    molecule the network would then refuse.
    """
    try:
        release.corrections(start, np.eye(3 * start.n_atoms))
    except (AromirError, ValueError) as error:
        reason = (
            f"the network cannot read this molecule's start geometry: {error}. If a linear "
            "group is bent only in this start geometry, straighten it or give the SMILES"
        )
        return [reason]
    return []


def _saddle_note(state: _Run) -> list[str]:
    left = state.left_saddle_point
    if left is None:
        return []
    note = (
        f"the start geometry led to a saddle point (a mode at {left.imaginary_cm:.0f} cm⁻¹): "
        f"the atoms were moved {OFF_THE_SADDLE_BOHR:g} bohr along that mode and optimised "
        "again, so the structure computed is the minimum that step led to, one of two"
    )
    return [note]


def run_spectrum(request: Request, release: Release) -> dict[str, Any]:
    """Compute a molecule's corrected spectrum and certificate and write its result folder.

    The network reads the kind of Hessian it was trained on: with its soft modes measured
    when the release says so, pyscf's analytic one otherwise. Its correction is added to
    the Hessian with the soft modes measured, which is also the one that says whether the
    geometry is a minimum.

    Returns:
        The record written to ``certificate.json``.

    Raises:
        RefusedError: The molecule is outside the domain, the network's training elements
            or sizes, the network cannot read it, or the folder holds a result or another
            request's unfinished run.
        ConvergenceError: The SCF or the optimisation did not converge.
        NotAMinimumError: No minimum was found.
        ResponseError: The dipole derivatives failed their checks.
    """
    admitted, start = admit(request.molecule)
    reasons = _untrained_elements(start, release)
    if not request.outside_training_sizes:
        reasons += _untrained_size(start, release)
    if reasons:
        raise RefusedError(reasons)
    folder, state = _open_folder(request.out, _asked("spectrum", admitted, start))
    # pyscf is told its threads by the resources; torch takes every core unless told too.
    # The setting is torch's own, for the whole process, and stays after this call.
    torch.set_num_threads(request.resources.threads)
    reasons = _no_network_inputs(start, release)
    if reasons:
        raise RefusedError(reasons)
    launched = records.provenance()
    optimised, low = _compute(start, folder, state, request.resources)
    as_trained = low.as_trained(soft_modes_measured=release.training.soft_mode_repair)
    corrections = release.corrections(optimised.geometry, as_trained)
    records.write_arrays(
        folder / CORRECTION_FILE,
        {"mean": corrections.mean, "per_seed": np.stack(corrections.per_seed)},
    )
    record = certificate.build(admitted, optimised.geometry, low, corrections, release)
    record["not_in_the_error"] += _saddle_note(state)
    record |= {
        "computation": _computation_record(optimised, low, state, folder),
        "provenance": launched,
    }
    # The record is written last: it is what makes the folder a result, and a run killed
    # before it is taken up again and writes both files.
    records.write_text(folder / CERTIFICATE_TEXT_FILE, certificate.to_markdown(record))
    records.write_json(folder / CERTIFICATE_FILE, record)
    return record
