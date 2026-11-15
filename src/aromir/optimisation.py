"""The geometry optimisation at the low level, by geomeTRIC through pyscf."""

from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, TextIO

import numpy as np
from pyscf.geomopt import geometric_solver

from aromir.dft import Resources, build_molecule, converged_scf
from aromir.errors import ConvergenceError
from aromir.molecule import Geometry

if TYPE_CHECKING:
    from collections.abc import Iterator

CONVERGENCE_SET: Final = "GAU_TIGHT"
"""geomeTRIC's tight criteria: a harmonic spectrum needs a well-converged minimum."""

MAX_STEPS: Final = 100


@dataclass(frozen=True, eq=False)
class Optimisation:
    """An optimised geometry and the record of how it was reached.

    Attributes:
        geometry: The minimum, atoms in the start geometry's order.
        energy: The total energy at the minimum, hartree.
        max_force: The largest gradient component at the minimum, hartree/bohr.
        n_steps: The optimiser's steps.
    """

    geometry: Geometry
    energy: float
    max_force: float
    n_steps: int


@contextlib.contextmanager
def _geometric_output_to(log: TextIO) -> Iterator[None]:
    """Send geomeTRIC's progress report to the run's log instead of the terminal.

    geomeTRIC configures the root logger to write to ``sys.stderr`` each time it runs.
    That stream is pointed at the log for the duration, and the caller's own logging
    configuration is put back afterwards.
    """
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    try:
        with contextlib.redirect_stderr(log):
            yield
    finally:
        root.handlers[:] = handlers
        root.setLevel(level)


def optimise(start: Geometry, resources: Resources, log: TextIO) -> Optimisation:
    """Optimise a geometry at the low level.

    pyscf's ``geometric_solver.kernel`` is called rather than its ``optimize`` wrapper:
    the wrapper returns the last geometry whether or not geomeTRIC converged, and only
    ``kernel`` hands back the verdict. The energy and forces of the record are recomputed
    at the final geometry, so they describe the geometry that is returned.

    Raises:
        ConvergenceError: The optimisation did not converge within ``MAX_STEPS``.
    """
    steps = 0

    def count_step(_environment: dict[str, object]) -> None:
        nonlocal steps
        steps += 1

    scf = converged_scf(build_molecule(start, resources, log))
    with _geometric_output_to(log):
        converged, optimised = geometric_solver.kernel(
            scf,
            assert_convergence=True,
            callback=count_step,
            convergence_set=CONVERGENCE_SET,
            maxsteps=MAX_STEPS,
        )
    if not converged:
        msg = f"the geometry optimisation did not converge in {MAX_STEPS} steps"
        raise ConvergenceError(msg)
    geometry = Geometry(start.symbols, optimised.atom_coords())
    final = converged_scf(build_molecule(geometry, resources, log))
    forces = final.nuc_grad_method().kernel()
    return Optimisation(geometry, float(final.e_tot), float(np.abs(forces).max()), steps)
