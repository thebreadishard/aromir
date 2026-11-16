"""From one molecule to its spectrum, written to a result folder.

A molecule outside the domain is refused before the first computation, and a result is
never overwritten. Each step is saved when it ends — the optimised geometry, the Hessian,
the measured soft modes — so that a failure after hours of quantum chemistry loses none of
them: the same request into the same folder takes up where the run stopped.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Self, TextIO

import numpy as np
from pydantic import BaseModel, ConfigDict, model_validator

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
"""What the run was asked and which steps have ended: the record a later run resumes from."""

OFF_THE_SADDLE_BOHR: Final = 0.1
"""How far the atom that moves most is displaced along an imaginary mode before the one
second optimisation: far enough to leave a saddle point's flat top, small beside a bond."""


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
    coordinates = np.round(start.coords_bohr, 6).tobytes()
    start_hash = hashlib.sha256(" ".join(start.symbols).encode() + coordinates).hexdigest()
    return {
        "command": command,
        "smiles": admitted.smiles,
        "start_geometry_sha256": start_hash,
        "level": {"functional": FUNCTIONAL, "basis": BASIS, "grid": list(GRID)},
    }


def _open_folder(out: Path, asked: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    """Return the result folder and the state of the run it holds, new or to be resumed.

    Raises:
        RefusedError: The folder holds a result, another request's run or other files.
    """
    folder = out.resolve()
    if folder.exists() and not folder.is_dir():
        raise RefusedError([f"{folder} is a file: --out names a folder"])
    if not folder.exists() or not any(folder.iterdir()):
        return folder, {"asked": asked}
    if any((folder / name).exists() for name in (BASELINE_FILE, CERTIFICATE_FILE)):
        raise RefusedError([f"{folder} holds a result: a result is never overwritten"])
    if not (folder / RUN_FILE).is_file():
        raise RefusedError([f"{folder} is not empty: a result is never overwritten"])
    state: dict[str, Any] = json.loads((folder / RUN_FILE).read_text(encoding="utf-8"))
    if state.get("asked") != asked:
        before = state.get("asked", {})
        reason = (
            f"{folder} holds the unfinished run of another request "
            f"({before.get('command')} {before.get('smiles')}): only the same command on the "
            "same molecule and start geometry takes it up"
        )
        raise RefusedError([reason])
    return folder, state


def _optimised(
    start: Geometry, folder: Path, state: dict[str, Any], resources: Resources, log: TextIO
) -> optimisation.Optimisation:
    """Return the optimised geometry: the saved one, or a new one that is saved at once."""
    saved = state.get("optimisation")
    if saved is not None and (folder / GEOMETRY_FILE).is_file():
        stored = json.loads((folder / GEOMETRY_FILE).read_text(encoding="utf-8"))
        minimum = Geometry(tuple(stored["symbols"]), np.array(stored["coords_bohr"], float))
        return optimisation.Optimisation(minimum, **saved)
    optimised = optimisation.optimise(start, resources, log)
    records.write_json(folder / GEOMETRY_FILE, optimised.geometry.to_dict())
    state["optimisation"] = {
        "energy": optimised.energy,
        "max_force": optimised.max_force,
        "n_steps": optimised.n_steps,
    }
    records.write_json(folder / RUN_FILE, state)
    return optimised


def _save_low_level(
    folder: Path, state: dict[str, Any], low: lowlevel.LowLevel, *, soft_modes_measured: bool
) -> None:
    arrays = {"hessian": low.hessian, "polar_tensor": low.polar_tensor}
    if low.analytic_hessian is not None:
        arrays["analytic_hessian"] = low.analytic_hessian
    records.write_arrays(folder / LOW_LEVEL_FILE, arrays)
    state["low_level"] = {
        "energy": low.energy,
        "refinement": asdict(low.refinement),
        "sum_rule_violation": low.sum_rule_violation,
        "seconds": low.seconds,
        "soft_modes_measured": soft_modes_measured,
        "soft_modes": None if low.soft_modes is None else asdict(low.soft_modes),
    }
    records.write_json(folder / RUN_FILE, state)


def _saved_low_level(folder: Path, saved: dict[str, Any]) -> lowlevel.LowLevel:
    with np.load(folder / LOW_LEVEL_FILE) as arrays:
        hessian, polar_tensor = np.array(arrays["hessian"]), np.array(arrays["polar_tensor"])
        analytic = np.array(arrays["analytic_hessian"]) if "analytic_hessian" in arrays else None
    soft_modes = saved["soft_modes"]
    return lowlevel.LowLevel(
        energy=saved["energy"],
        hessian=hessian,
        polar_tensor=polar_tensor,
        refinement=response.Refinement(**saved["refinement"]),
        sum_rule_violation=saved["sum_rule_violation"],
        seconds=saved["seconds"],
        soft_modes=None if soft_modes is None else repair.Repair(**soft_modes),
        analytic_hessian=analytic,
    )


def _low_level(
    minimum: Geometry, folder: Path, state: dict[str, Any], resources: Resources, log: TextIO
) -> lowlevel.LowLevel:
    """Return the low level with its soft modes measured, saving each of the two steps.

    The analytic Hessian is on disk before the repair starts, so a repair that fails —
    for want of memory, say — costs the repair alone.
    """
    saved = state.get("low_level")
    if saved is not None and (folder / LOW_LEVEL_FILE).is_file():
        low = _saved_low_level(folder, saved)
    else:
        low = lowlevel.compute(minimum, resources, log, repair_soft_modes=False)
        _save_low_level(folder, state, low, soft_modes_measured=False)
    if not state["low_level"]["soft_modes_measured"]:
        repaired, record = repair.repair(minimum, low.hessian, resources, log)
        if record is not None:
            low = replace(low, hessian=repaired, analytic_hessian=low.hessian, soft_modes=record)
        _save_low_level(folder, state, low, soft_modes_measured=True)
    return low


def _compute(
    start: Geometry, folder: Path, state: dict[str, Any], resources: Resources
) -> tuple[optimisation.Optimisation, lowlevel.LowLevel]:
    """Optimise and compute the low level, each step saved; step off a saddle point once.

    Whether the geometry is a minimum is read from the Hessian with its soft modes
    measured: the analytic one can make a soft mode imaginary that is not.

    Raises:
        NotAMinimumError: The geometry still has an imaginary mode after one step along it
            and a second optimisation.
    """
    folder.mkdir(parents=True, exist_ok=True)
    records.write_json(folder / RUN_FILE, state)
    with (folder / QUANTUM_LOG_FILE).open("a", encoding="utf-8") as log:
        for _attempt in range(2):
            left = state.get("left_saddle_point")
            begin = start if left is None else Geometry(start.symbols, np.array(left["restart"]))
            optimised = _optimised(begin, folder, state, resources, log)
            low = _low_level(optimised.geometry, folder, state, resources, log)
            modes = vibrations.vibrational_modes(low.hessian, optimised.geometry)
            if not modes.n_imaginary or left is not None:
                break
            step, _ = vibrations.displacement(
                modes.vectors[:, 0], optimised.geometry.masses_amu, OFF_THE_SADDLE_BOHR
            )
            # The saddle point's own files stay, under another name; the steps start over.
            for name in (GEOMETRY_FILE, LOW_LEVEL_FILE):
                (folder / name).replace(folder / f"saddle_point_{name}")
            state.pop("optimisation")
            state.pop("low_level")
            state["left_saddle_point"] = {
                "imaginary_cm": float(modes.wavenumbers[0]),
                "restart": (optimised.geometry.coords_bohr + step).tolist(),
            }
            records.write_json(folder / RUN_FILE, state)
    if modes.n_imaginary:
        msg = (
            f"{modes.n_imaginary} imaginary mode(s), the lowest at {modes.wavenumbers[0]:.0f} "
            "cm⁻¹, also after one step along it and a second optimisation: no minimum was "
            f"found. The geometry and the Hessian are kept in {folder}"
        )
        raise NotAMinimumError(msg)
    return optimised, low


def _computation_record(
    optimised: optimisation.Optimisation, low: lowlevel.LowLevel, state: dict[str, Any]
) -> dict[str, Any]:
    left = state.get("left_saddle_point")
    return {
        "level": {"functional": FUNCTIONAL, "basis": BASIS, "grid": list(GRID)},
        "optimisation": {
            "energy_hartree": optimised.energy,
            "max_force_hartree_per_bohr": optimised.max_force,
            "n_steps": optimised.n_steps,
            "left_saddle_point_at_cm": None if left is None else left["imaginary_cm"],
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
        **_computation_record(optimised, low, state),
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
        return [f"the network cannot read this molecule's start geometry: {error}"]
    return []


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
    record = certificate.build(admitted, optimised.geometry, low, corrections, release) | {
        "computation": _computation_record(optimised, low, state),
        "provenance": launched,
    }
    records.write_json(folder / CERTIFICATE_FILE, record)
    records.write_text(folder / CERTIFICATE_TEXT_FILE, certificate.to_markdown(record))
    return record
