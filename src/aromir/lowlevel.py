"""The low level's Hessian and dipole derivatives of one geometry, with their checks."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, TextIO

from aromir import repair, response
from aromir.dft import Resources, build_molecule, converged_scf
from aromir.errors import ResponseError

if TYPE_CHECKING:
    from aromir.molecule import FloatArray, Geometry


@dataclass(frozen=True, eq=False)
class LowLevel:
    """The B3LYP result the correction and the spectrum are built on.

    Attributes:
        energy: Total energy, hartree.
        hessian: Cartesian Hessian, hartree/bohr², shape (3N, 3N), symmetric: pyscf's
            analytic one, or that with its soft modes measured once the repair has run.
        polar_tensor: Atomic polar tensor ∂μ/∂R in e, shape (3, 3N).
        refinement: The residual check of the response the polar tensor rests on.
        sum_rule_violation: The polar tensor's translation sum rule, in e.
        seconds: Wall time of the calculation.
        soft_modes: The record of the soft-mode repair; None when none was needed or asked.
        analytic_hessian: pyscf's analytic Hessian as it was before the repair changed it;
            None when ``hessian`` is that Hessian.
    """

    energy: float
    hessian: FloatArray
    polar_tensor: FloatArray
    refinement: response.Refinement
    sum_rule_violation: float
    seconds: float
    soft_modes: repair.Repair | None = None
    analytic_hessian: FloatArray | None = None

    def as_trained(self, *, soft_modes_measured: bool) -> FloatArray:
        """Return the Hessian of the kind a network was trained on.

        A network trained on analytic Hessians reads the analytic one, also where the
        repair has since measured this molecule's soft modes.
        """
        if soft_modes_measured or self.analytic_hessian is None:
            return self.hessian
        return self.analytic_hessian


def _failed_checks(refinement: response.Refinement, violation: float) -> list[str]:
    failures = []
    if refinement.residual_after > response.RESIDUAL_BOUND:
        failures.append(
            f"response residual {refinement.residual_after:.1e} above "
            f"{response.RESIDUAL_BOUND:g} after {refinement.batched_rounds} batched rounds "
            f"and {refinement.single_columns} single-column solves"
        )
    if violation > response.SUM_RULE_LIMIT:
        failures.append(
            f"translation sum rule violated by {violation:.1e} e "
            f"(limit {response.SUM_RULE_LIMIT:g} e)"
        )
    return failures


def compute(geometry: Geometry, resources: Resources, log: TextIO) -> LowLevel:
    """Compute the analytic Hessian and the polar tensor at a geometry.

    The orbital response is solved once. The Hessian is built on pyscf's own solution; the
    polar tensor on the solution after the residual check, because it is the more sensitive
    of the two. The Hessian's soft modes are measured afterwards (``aromir.repair``), as
    a step of its own, so that this result is saved before that one can fail.

    Raises:
        ConvergenceError: The SCF did not converge.
        ResponseError: The polar tensor failed its residual bound or its sum rule.
    """
    started = time.monotonic()
    scf = converged_scf(build_molecule(geometry, resources, log))
    hessian_method = scf.Hessian()
    solved = response.solve(hessian_method)
    hessian = response.hessian(hessian_method, solved)
    refined, refinement = response.refine(hessian_method, solved)
    tensor = response.polar_tensor(scf, refined)
    violation = response.sum_rule_violation(tensor)
    failures = _failed_checks(refinement, violation)
    if failures:
        raise ResponseError("; ".join(failures))
    return LowLevel(
        energy=float(scf.e_tot),
        hessian=hessian,
        polar_tensor=tensor,
        refinement=refinement,
        sum_rule_violation=violation,
        seconds=time.monotonic() - started,
    )
