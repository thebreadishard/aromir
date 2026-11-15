"""From one molecule to its spectrum, written to a result folder.

Every refusal comes before the first computation, and a result is never overwritten.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, model_validator

from aromir import (
    admission,
    certificate,
    geometry,
    lowlevel,
    optimisation,
    records,
    spectrum,
    vibrations,
)
from aromir.dft import BASIS, FUNCTIONAL, GRID, Resources
from aromir.errors import NotAMinimumError, RefusedError

if TYPE_CHECKING:
    from aromir.molecule import Geometry
    from aromir.release import Release

GEOMETRY_FILE: Final = "geometry.json"
LOW_LEVEL_FILE: Final = "lowlevel.npz"
CORRECTION_FILE: Final = "correction.npz"
BASELINE_FILE: Final = "baseline.json"
CERTIFICATE_FILE: Final = "certificate.json"
CERTIFICATE_TEXT_FILE: Final = "certificate.md"
QUANTUM_LOG_FILE: Final = "pyscf.log"


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


def _new_folder(out: Path) -> Path:
    folder = out.resolve()
    if folder.exists() and any(folder.iterdir()):
        raise RefusedError([f"{folder} is not empty: a result is never overwritten"])
    return folder


def _compute(
    start: Geometry, folder: Path, resources: Resources, *, repair_soft_modes: bool
) -> tuple[optimisation.Optimisation, lowlevel.LowLevel]:
    """Optimise, compute the low level and write both; refuse a geometry that is no minimum."""
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / QUANTUM_LOG_FILE).open("w", encoding="utf-8") as log:
        optimised = optimisation.optimise(start, resources, log)
        records.write_json(folder / GEOMETRY_FILE, optimised.geometry.to_dict())
        low = lowlevel.compute(
            optimised.geometry, resources, log, repair_soft_modes=repair_soft_modes
        )
    n_imaginary = vibrations.vibrational_modes(low.hessian, optimised.geometry).n_imaginary
    if n_imaginary:
        msg = f"{n_imaginary} imaginary mode(s): the optimised geometry is not a minimum"
        raise NotAMinimumError(msg)
    records.write_arrays(
        folder / LOW_LEVEL_FILE, {"hessian": low.hessian, "polar_tensor": low.polar_tensor}
    )
    return optimised, low


def _computation_record(
    optimised: optimisation.Optimisation, low: lowlevel.LowLevel
) -> dict[str, Any]:
    return {
        "level": {"functional": FUNCTIONAL, "basis": BASIS, "grid": list(GRID)},
        "optimisation": {
            "energy_hartree": optimised.energy,
            "max_force_hartree_per_bohr": optimised.max_force,
            "n_steps": optimised.n_steps,
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
        RefusedError: The molecule is outside the domain, or the folder holds a result.
        ConvergenceError: The SCF or the optimisation did not converge.
        NotAMinimumError: The optimised geometry has an imaginary mode.
        ResponseError: The dipole derivatives failed their checks.
    """
    admitted, start = admit(request.molecule)
    folder = _new_folder(request.out)
    launched = records.provenance()
    optimised, low = _compute(start, folder, request.resources, repair_soft_modes=True)
    bands = spectrum.harmonic_bands(low.hessian, optimised.geometry, low.polar_tensor)
    record = {
        "molecule": {
            "smiles": admitted.smiles,
            "formula": admitted.formula,
            "n_atoms": admitted.n_atoms,
        },
        **_computation_record(optimised, low),
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


def run_spectrum(request: Request, release: Release) -> dict[str, Any]:
    """Compute a molecule's corrected spectrum and certificate and write its result folder.

    Returns:
        The record written to ``certificate.json``.

    Raises:
        RefusedError: The molecule is outside the domain or the network's training
            elements, or the folder holds a result.
        ConvergenceError: The SCF or the optimisation did not converge.
        NotAMinimumError: The optimised geometry has an imaginary mode.
        ResponseError: The dipole derivatives failed their checks.
    """
    admitted, start = admit(request.molecule)
    reasons = _untrained_elements(start, release)
    if reasons:
        raise RefusedError(reasons)
    folder = _new_folder(request.out)
    launched = records.provenance()
    optimised, low = _compute(
        start, folder, request.resources, repair_soft_modes=release.training.soft_mode_repair
    )
    corrections = release.corrections(optimised.geometry, low.hessian)
    records.write_arrays(
        folder / CORRECTION_FILE,
        {"mean": corrections.mean, "per_seed": np.stack(corrections.per_seed)},
    )
    record = certificate.build(admitted, optimised.geometry, low, corrections, release) | {
        "computation": _computation_record(optimised, low),
        "provenance": launched,
    }
    records.write_json(folder / CERTIFICATE_FILE, record)
    records.write_text(folder / CERTIFICATE_TEXT_FILE, certificate.to_markdown(record))
    return record
