"""The geometry optimisation at the low level, by geomeTRIC through pyscf.

An optimisation is asked for geomeTRIC's tight criteria on energy, forces and displacement.
On a flat coordinate — a nearly free torsion — the first two are met long before the third:
the optimiser's trust radius shrinks and every step is a crawl of a size the displacement
criterion does not accept. After ``MAX_STEPS`` of that the optimisation goes on with
geomeTRIC's same test and the same thresholds for the energy and the forces, and without
the displacement criterion; the result says which of the two it converged under.
"""

from __future__ import annotations

import contextlib
import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, TextIO

import numpy as np
from pyscf.geomopt import geometric_solver

from aromir.constants import ANGSTROM_PER_BOHR
from aromir.dft import Resources, build_molecule, converged_scf
from aromir.errors import ConvergenceError
from aromir.molecule import Geometry

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from aromir.molecule import FloatArray

CONVERGENCE_SET: Final = "GAU_TIGHT"
"""geomeTRIC's tight criteria: a harmonic spectrum needs a well-converged minimum."""

MAX_STEPS: Final = 100
"""The optimiser's steps under all of the tight criteria: geomeTRIC's ``maxiter``, which
takes one gradient more than it counts steps. An optimisation taken up has the steps left
that its gradients so far have not used."""

FLAT_STEPS: Final = 20
"""The steps it may take after those without the displacement criterion. A geometry that
has the energy and the forces converged passes at its first step; the rest is room for one
that has not."""

NO_DISPLACEMENT_CRITERION: Final[dict[str, float]] = {
    "convergence_drms": math.inf,
    "convergence_dmax": math.inf,
}
"""geomeTRIC's own keywords for the two displacement thresholds, set so that any step meets
them (``geometric/params.py``, ``convergence_criteria``: 'convergence_drms' and
'convergence_dmax' replace the set's values). Its test stays the one it always applies —
energy change, rms and largest gradient, rms and largest displacement, all five
(``geometric/optimize.py``, ``evaluateStep``: ``criteria_met``) — with the set's thresholds
for the first three. pyscf hands the keywords through: ``geometric_solver.kernel`` passes
its ``**kwargs`` on to ``geometric.optimize.run_optimizer``."""

ALL_CRITERIA: Final = "energy, forces and displacement"
ENERGY_AND_FORCES: Final = "energy and forces"
"""The two ways an optimisation ends: under all of ``CONVERGENCE_SET`` within ``MAX_STEPS``,
or after those under ``CONVERGENCE_SET`` with ``NO_DISPLACEMENT_CRITERION``."""


@dataclass(frozen=True)
class Progress:
    """Where an optimisation stands after a gradient: enough to take it up again.

    The optimiser's own Hessian estimate and trust radius are not kept; an optimisation
    taken up starts them afresh at the saved geometry, which is the last one a gradient
    was computed at — also when the optimiser would have rejected that step.

    Attributes:
        coords_bohr: The geometry of the last gradient, atoms in the start's order.
        n_steps: The gradients taken so far.
    """

    coords_bohr: list[list[float]]
    n_steps: int


@dataclass(frozen=True, eq=False)
class Optimisation:
    """An optimised geometry and the record of how it was reached.

    Attributes:
        geometry: Where the optimisation ended, atoms in the start geometry's order.
        energy: The total energy there, hartree.
        max_force: The largest gradient component there, hartree/bohr.
        n_steps: The optimiser's gradients, over every run that took part.
        criteria: ``ALL_CRITERIA`` or ``ENERGY_AND_FORCES``: what the optimiser converged on.
        last_step: For an optimisation that converged on energy and forces alone, the
            change between its last two gradients: of the energy (hartree), the rms and
            the largest atomic force (hartree/bohr) at the end, and the rms and the largest
            distance an atom moved (ångström). They describe the end for a reader; the
            optimiser's own displacement is measured after aligning the two geometries.
    """

    geometry: Geometry
    energy: float
    max_force: float
    n_steps: int
    criteria: str = ALL_CRITERIA
    last_step: dict[str, float] | None = None


class _Trail:
    """What the optimiser reports at each gradient: counted, kept, and the last two held."""

    def __init__(self, n_steps: int, keep: Callable[[Progress], None] | None) -> None:
        self.n_steps = n_steps
        self._keep = keep
        self._points: list[tuple[FloatArray, float, FloatArray]] = []

    def __call__(self, environment: dict[str, Any]) -> None:
        """Take pyscf's callback: the local variables of its gradient call."""
        self.n_steps += 1
        coords = np.array(environment["coords"], dtype=float).reshape(-1, 3)
        gradient = np.array(environment["gradients"], dtype=float).reshape(-1, 3)
        self._points = [*self._points[-1:], (coords, float(environment["energy"]), gradient)]
        if self._keep is not None:
            self._keep(Progress(coords.tolist(), self.n_steps))

    def last_step(self) -> dict[str, float]:
        """Return the change between the last two gradients.

        Raises:
            ConvergenceError: Fewer than two gradients were taken, so there is no step.
        """
        if len(self._points) < 2:
            msg = "the optimiser reported fewer than two gradients: there is no last step"
            raise ConvergenceError(msg)
        (before, energy_before, _), (after, energy, gradient) = self._points
        moved = np.linalg.norm(after - before, axis=1) * ANGSTROM_PER_BOHR
        forces = np.linalg.norm(gradient, axis=1)
        return {
            "energy_change_hartree": energy - energy_before,
            "rms_force_hartree_per_bohr": float(np.sqrt(np.mean(forces**2))),
            "max_force_hartree_per_bohr": float(np.max(forces)),
            "rms_displacement_angstrom": float(np.sqrt(np.mean(moved**2))),
            "max_displacement_angstrom": float(np.max(moved)),
        }


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


def _run(
    start: Geometry, resources: Resources, log: TextIO, trail: _Trail, **settings: float
) -> tuple[bool, Geometry]:
    """Run geomeTRIC once from a geometry; return its verdict and where it stopped.

    pyscf's ``geometric_solver.kernel`` is called rather than its ``optimize`` wrapper:
    the wrapper returns the last geometry whether or not geomeTRIC converged, and only
    ``kernel`` hands back the verdict.
    """
    scf = converged_scf(build_molecule(start, resources, log))
    with _geometric_output_to(log):
        converged, stopped = geometric_solver.kernel(
            scf,
            assert_convergence=True,
            callback=trail,
            convergence_set=CONVERGENCE_SET,
            **settings,
        )
    return bool(converged), Geometry(start.symbols, stopped.atom_coords())


def optimise(
    start: Geometry,
    resources: Resources,
    log: TextIO,
    *,
    taken_up: Progress | None = None,
    keep: Callable[[Progress], None] | None = None,
) -> Optimisation:
    """Optimise a geometry at the low level.

    The energy and forces of the record are recomputed at the final geometry, so they
    describe the geometry that is returned.

    Args:
        start: Where the optimisation starts.
        resources: What the calculation may use.
        log: Where pyscf and geomeTRIC report.
        taken_up: The saved progress of an earlier, unfinished run of this optimisation;
            its geometry replaces ``start`` and its steps count.
        keep: Called after every gradient with the progress to save.

    Raises:
        ConvergenceError: The optimisation converged neither in ``MAX_STEPS`` under all
            criteria nor in ``FLAT_STEPS`` more on energy and forces.
    """
    trail = _Trail(0 if taken_up is None else taken_up.n_steps, keep)
    at = start if taken_up is None else Geometry(start.symbols, np.array(taken_up.coords_bohr))
    converged, criteria = False, ALL_CRITERIA
    if trail.n_steps < MAX_STEPS:
        converged, at = _run(at, resources, log, trail, maxsteps=MAX_STEPS - trail.n_steps)
    if not converged:
        criteria = ENERGY_AND_FORCES
        log.write(
            f"\naromir: not converged on {ALL_CRITERIA} after {trail.n_steps} gradients; "
            f"going on without the displacement criterion ({ENERGY_AND_FORCES})\n"
        )
        converged, at = _run(
            at, resources, log, trail, maxsteps=FLAT_STEPS, **NO_DISPLACEMENT_CRITERION
        )
    if not converged:
        msg = (
            f"the geometry optimisation did not converge in {MAX_STEPS} steps, nor in "
            f"{FLAT_STEPS} more on {ENERGY_AND_FORCES} alone ({trail.n_steps} gradients in all)"
        )
        raise ConvergenceError(msg)
    final = converged_scf(build_molecule(at, resources, log))
    forces = final.nuc_grad_method().kernel()
    return Optimisation(
        geometry=at,
        energy=float(final.e_tot),
        max_force=float(np.abs(forces).max()),
        n_steps=trail.n_steps,
        criteria=criteria,
        last_step=trail.last_step() if criteria == ENERGY_AND_FORCES else None,
    )
