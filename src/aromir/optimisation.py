"""The geometry optimisation at the low level, by geomeTRIC through pyscf.

An optimisation is asked for geomeTRIC's tight criteria on energy, forces and displacement.
On a flat coordinate — a nearly free torsion — the first two are met long before the third:
the optimiser's trust radius shrinks and every step is a crawl of a size the displacement
criterion does not accept. After ``MAX_STEPS`` of that the optimisation goes on under
geomeTRIC's own Q-Chem-style test at the same thresholds, which converges on the forces
and either the energy or the displacement, and the result says which of the two it was.
"""

from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, TextIO, cast

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
"""The gradients an optimisation may take under all of the tight criteria."""

FLAT_STEPS: Final = 20
"""The gradients it may take after those under geomeTRIC's Q-Chem-style test. A geometry
that has the energy and the forces converged passes that test at its first step; the rest
is room for one that has not."""

ALL_CRITERIA: Final = "energy, forces and displacement"
ENERGY_AND_FORCES: Final = "energy and forces"
"""The two ways an optimisation ends: under all of ``CONVERGENCE_SET``, or under
geomeTRIC's ``qccnv`` at the same thresholds (``geometric/params.py``: 'Q-Chem style
convergence criteria (i.e. gradient and either energy or displacement)';
``geometric/optimize.py``, ``evaluate_OPT_step``: converged when the rms gradient and
either the rms displacement or the energy change are within theirs). pyscf hands the
keyword through: ``geometric_solver.kernel`` passes its ``**kwargs`` to geomeTRIC's
``run_optimizer``, as its docstring's ``conv_params`` example does."""


Q_CHEM_VERDICT: Final = "Converged! (Q-Chem style"
"""How geomeTRIC's report says that it converged on its Q-Chem-style test and not on every
criterion, which it reports as 'Converged! =D' (``geometric/optimize.py``,
``evaluate_OPT_step``)."""


@dataclass(frozen=True)
class Progress:
    """Where an optimisation stands after a gradient: enough to take it up again.

    The optimiser's own Hessian estimate and trust radius are not kept; an optimisation
    taken up starts them afresh at the saved geometry.

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
        geometry: The minimum, atoms in the start geometry's order.
        energy: The total energy at the minimum, hartree.
        max_force: The largest gradient component at the minimum, hartree/bohr.
        n_steps: The optimiser's gradients, over every run that took part.
        criteria: ``ALL_CRITERIA`` or ``ENERGY_AND_FORCES``: what the optimiser converged on.
        last_step: The optimiser's last step, for the record of an optimisation that
            converged on energy and forces alone: the energy change (hartree), the rms and
            the largest atomic force (hartree/bohr) and atomic displacement (ångström).
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

    def last_step(self) -> dict[str, float] | None:
        """Return the size of the last step, in the quantities geomeTRIC's criteria read."""
        if len(self._points) < 2:
            return None
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


class _Report:
    """geomeTRIC's report on its way to the log, read for the test it converged on."""

    def __init__(self, log: TextIO) -> None:
        self._log = log
        self.on_energy_and_forces = False

    def write(self, text: str) -> int:
        self.on_energy_and_forces |= Q_CHEM_VERDICT in text
        return self._log.write(text)

    def flush(self) -> None:
        self._log.flush()


@contextlib.contextmanager
def _geometric_output_to(log: TextIO) -> Iterator[_Report]:
    """Send geomeTRIC's progress report to the run's log instead of the terminal.

    geomeTRIC configures the root logger to write to ``sys.stderr`` each time it runs.
    That stream is pointed at the log for the duration, and the caller's own logging
    configuration is put back afterwards.
    """
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    report = _Report(log)
    try:
        with contextlib.redirect_stderr(cast("TextIO", report)):
            yield report
    finally:
        root.handlers[:] = handlers
        root.setLevel(level)


def _run(
    start: Geometry, resources: Resources, log: TextIO, trail: _Trail, **settings: object
) -> tuple[bool, Geometry, bool]:
    """Run geomeTRIC once from a geometry.

    Returns its verdict, where it stopped, and whether it said it converged on its
    Q-Chem-style test rather than on every criterion.

    pyscf's ``geometric_solver.kernel`` is called rather than its ``optimize`` wrapper:
    the wrapper returns the last geometry whether or not geomeTRIC converged, and only
    ``kernel`` hands back the verdict.
    """
    scf = converged_scf(build_molecule(start, resources, log))
    with _geometric_output_to(log) as report:
        converged, stopped = geometric_solver.kernel(
            scf,
            assert_convergence=True,
            callback=trail,
            convergence_set=CONVERGENCE_SET,
            **settings,
        )
    where = Geometry(start.symbols, stopped.atom_coords())
    return bool(converged), where, report.on_energy_and_forces


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
    converged, on_energy_and_forces = False, False
    if trail.n_steps < MAX_STEPS:
        converged, at, _ = _run(at, resources, log, trail, maxsteps=MAX_STEPS - trail.n_steps)
    if not converged:
        log.write(
            f"\naromir: not converged on {ALL_CRITERIA} after {trail.n_steps} gradients; "
            f"going on under geomeTRIC's Q-Chem-style test ({ENERGY_AND_FORCES})\n"
        )
        converged, at, on_energy_and_forces = _run(
            at, resources, log, trail, maxsteps=FLAT_STEPS, qccnv=True
        )
    if not converged:
        msg = (
            f"the geometry optimisation did not converge in {MAX_STEPS} steps, nor in "
            f"{FLAT_STEPS} more on {ENERGY_AND_FORCES} alone"
        )
        raise ConvergenceError(msg)
    final = converged_scf(build_molecule(at, resources, log))
    forces = final.nuc_grad_method().kernel()
    return Optimisation(
        geometry=at,
        energy=float(final.e_tot),
        max_force=float(np.abs(forces).max()),
        n_steps=trail.n_steps,
        criteria=ENERGY_AND_FORCES if on_energy_and_forces else ALL_CRITERIA,
        last_step=trail.last_step() if on_energy_and_forces else None,
    )
