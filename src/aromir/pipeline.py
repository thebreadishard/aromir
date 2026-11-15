"""From one molecule to its low-level spectrum, written to a result folder.

Every refusal comes before the first computation, and a result is never overwritten.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Self

from pydantic import BaseModel, ConfigDict, model_validator

from aromir import admission, geometry, lowlevel, optimisation, records, spectrum, vibrations
from aromir.dft import BASIS, FUNCTIONAL, GRID, Resources
from aromir.errors import NotAMinimumError, RefusedError

if TYPE_CHECKING:
    from aromir.molecule import Geometry

GEOMETRY_FILE: Final = "geometry.json"
LOW_LEVEL_FILE: Final = "lowlevel.npz"
BASELINE_FILE: Final = "baseline.json"
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


def _bands_record(bands: spectrum.Bands) -> list[dict[str, float]]:
    return [
        {"wavenumber_cm": round(float(position), 2), "intensity_km_mol": round(float(height), 3)}
        for position, height in zip(bands.wavenumbers, bands.intensities, strict=True)
    ]


def _baseline_record(
    admitted: admission.Admission,
    optimised: optimisation.Optimisation,
    low: lowlevel.LowLevel,
    bands: spectrum.Bands,
) -> dict[str, Any]:
    return {
        "molecule": {
            "smiles": admitted.smiles,
            "formula": admitted.formula,
            "n_atoms": admitted.n_atoms,
        },
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
        "seconds_low_level": round(low.seconds, 1),
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
    }


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
    folder = request.out.resolve()
    if folder.exists() and any(folder.iterdir()):
        raise RefusedError([f"{folder} is not empty: a result is never overwritten"])
    launched = records.provenance()
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / QUANTUM_LOG_FILE).open("w", encoding="utf-8") as log:
        optimised = optimisation.optimise(start, request.resources, log)
        records.write_json(folder / GEOMETRY_FILE, optimised.geometry.to_dict())
        low = lowlevel.compute(optimised.geometry, request.resources, log)
    n_imaginary = vibrations.vibrational_modes(low.hessian, optimised.geometry).n_imaginary
    if n_imaginary:
        msg = f"{n_imaginary} imaginary mode(s): the optimised geometry is not a minimum"
        raise NotAMinimumError(msg)
    records.write_arrays(
        folder / LOW_LEVEL_FILE, {"hessian": low.hessian, "polar_tensor": low.polar_tensor}
    )
    bands = spectrum.harmonic_bands(low.hessian, optimised.geometry, low.polar_tensor)
    record = _baseline_record(admitted, optimised, low, bands) | {"provenance": launched}
    records.write_json(folder / BASELINE_FILE, record)
    return record
