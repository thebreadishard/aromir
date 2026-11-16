"""The repair of a Hessian's soft modes.

pyscf's analytic DFT Hessian leaves out the derivative of the quadrature weights: the
integration grid moves with the atoms, and the Hessian does not know. Stiff modes do not
feel it. A mode below about 20 cm⁻¹ can be off by several cm⁻¹, or come out imaginary
when it is not.

The repair measures the Hessian along those modes instead. For each soft direction u the
column H u is the central difference of two gradients that do include the grid's
response, at geometries displaced along u. The measured columns then replace the
analytic Hessian's rows and columns along the soft directions; everything orthogonal to
them is kept. A molecule without a soft mode is left as it is, at no cost.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, TextIO

import numpy as np

from aromir import vibrations
from aromir.dft import GRID, Resources, Scf, build_molecule, converged_scf
from aromir.errors import AromirError
from aromir.molecule import Geometry

if TYPE_CHECKING:
    from aromir.molecule import FloatArray

SOFT_BELOW_CM: Final = 20.0
"""Above this the measured change stays under 0.5 cm⁻¹; below it, it reaches several."""

STEP_BOHR: Final = 0.01
"""The largest atomic displacement of a finite-difference step."""

DISPLACED_ENERGY_TOLERANCE: Final = 1e-9
DISPLACED_GRADIENT_TOLERANCE: Final = 1e-7
"""SCF thresholds of a displaced point, set by the finite difference's error budget.

A gradient uncertain by 1e-7 over a step of mass-weighted length near 1.5 leaves 3e-8
hartree/bohr² in a measured column, far below what the analytic Hessian itself carries.
Large molecules cannot reach a Hessian's own thresholds at all: their energy fluctuates
by more than 1e-11 hartree from one iteration to the next.
"""

_BYTES_PER_FLOAT: Final = 8
_BYTES_PER_MB: Final = 1e6


class GridResponseMemoryError(AromirError):
    """The grid-response gradient needs more memory than the calculation may use."""


@dataclass(frozen=True)
class Repair:
    """What the repair did to a Hessian.

    Attributes:
        n_directions: The soft modes whose columns were measured.
        lowest_before_cm: The lowest mode of the analytic Hessian.
        lowest_after_cm: The lowest mode of the repaired Hessian.
        largest_change_cm: The largest change of any mode's position, modes paired by rank.
    """

    n_directions: int
    lowest_before_cm: float
    lowest_after_cm: float
    largest_change_cm: float


def _check_memory(scf: Scf, resources: Resources) -> None:
    """Refuse a gradient whose arrays would not fit, before pyscf allocates them.

    pyscf evaluates the grid response on all of one atom's grid points at once, whatever
    its ``max_memory`` says: per point, ten values per basis function and the derivatives
    of the atomic partition (3 N² + 5 N numbers).
    """
    molecule = scf.mol
    per_point = 10 * molecule.nao + 3 * molecule.natm**2 + 5 * molecule.natm
    needed_mb = _BYTES_PER_FLOAT * GRID[0] * GRID[1] * per_point / _BYTES_PER_MB
    if needed_mb > resources.max_memory_mb:
        msg = (
            f"the soft-mode repair needs about {needed_mb:.0f} MB for the grid-response "
            f"gradient of {molecule.natm} atoms; {resources.max_memory_mb} MB are allowed "
            "(raise --max-memory-mb; the same command into the same folder then takes the "
            "run up from the saved Hessian)"
        )
        raise GridResponseMemoryError(msg)


def _gradient_with_grid_response(
    geometry: Geometry, resources: Resources, log: TextIO, start_density: FloatArray | None
) -> tuple[FloatArray, FloatArray]:
    """Return the nuclear gradient including the grid's response, and the SCF density.

    The call is the one of pyscf's ``grad/test/test_rks.py``:
    ``mf.nuc_grad_method().set(grid_response=True).kernel()``.
    """
    scf = converged_scf(
        build_molecule(geometry, resources, log),
        energy_tolerance=DISPLACED_ENERGY_TOLERANCE,
        gradient_tolerance=DISPLACED_GRADIENT_TOLERANCE,
        start_density=start_density,
    )
    _check_memory(scf, resources)
    gradient = scf.nuc_grad_method().set(grid_response=True).kernel()
    return np.asarray(gradient, dtype=float), np.asarray(scf.make_rdm1(), dtype=float)


def measured_columns(
    geometry: Geometry, directions: FloatArray, resources: Resources, log: TextIO
) -> FloatArray:
    """Return the mass-weighted Hessian's columns along each direction, shape (3N, k).

    Two gradients per direction, each SCF started from the previous point's density.
    """
    columns = []
    density = None
    for direction in directions.T:
        step, length = vibrations.displacement(direction, geometry.masses_amu, STEP_BOHR)
        gradients = []
        for sign in (1.0, -1.0):
            displaced = Geometry(geometry.symbols, geometry.coords_bohr + sign * step)
            gradient, density = _gradient_with_grid_response(displaced, resources, log, density)
            gradients.append(gradient)
        columns.append(vibrations.column(gradients[0], gradients[1], length, geometry.masses_amu))
    return np.column_stack(columns)


def repair(
    geometry: Geometry,
    hessian: FloatArray,
    resources: Resources,
    log: TextIO,
    soft_below_cm: float = SOFT_BELOW_CM,
) -> tuple[FloatArray, Repair | None]:
    """Return the Hessian with its soft modes measured, and the record of the repair.

    Args:
        geometry: The geometry the Hessian belongs to.
        hessian: The analytic Cartesian Hessian, hartree/bohr².
        resources: What the displaced calculations may use.
        log: Where pyscf's own log goes.
        soft_below_cm: Modes below this, in absolute value, are measured. The default is
            the rule; a test passes a higher value to measure a stiff mode, where the
            analytic Hessian is right and the two must agree.

    Returns:
        The repaired Hessian and its record, or the Hessian itself and None when the
        molecule has no soft mode.

    Raises:
        GridResponseMemoryError: The gradient's arrays would exceed the allowed memory.
        ConvergenceError: A displaced point's SCF did not converge.
    """
    before = vibrations.vibrational_modes(hessian, geometry)
    directions = before.vectors[:, np.abs(before.wavenumbers) < soft_below_cm]
    if directions.shape[1] == 0:
        return hessian, None
    columns = measured_columns(geometry, directions, resources, log)
    repaired = vibrations.with_measured_columns(hessian, geometry.masses_amu, directions, columns)
    after = vibrations.vibrational_modes(repaired, geometry)
    record = Repair(
        n_directions=int(directions.shape[1]),
        lowest_before_cm=float(before.wavenumbers[0]),
        lowest_after_cm=float(after.wavenumbers[0]),
        largest_change_cm=float(np.abs(after.wavenumbers - before.wavenumbers).max()),
    )
    return repaired, record
