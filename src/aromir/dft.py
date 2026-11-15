"""The low level of theory: B3LYP/6-31G* in pyscf, converged tightly.

The level is not a setting. The learned correction is trained on Hessians of exactly this
level, so another functional, basis or grid would feed it inputs it has never seen. Only
the machine's resources are the caller's to choose.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final, TextIO

from pydantic import BaseModel, ConfigDict, Field
from pyscf import dft, gto, lib

from aromir.errors import ConvergenceError

if TYPE_CHECKING:
    from aromir.molecule import Geometry

FUNCTIONAL: Final = "b3lyp"
BASIS: Final = "6-31g*"
CARTESIAN_D: Final = True
"""Six Cartesian d functions, as the basis set was defined."""

GRID: Final = (99, 590)
"""Radial and angular points per atom, unpruned."""

ENERGY_TOLERANCE: Final = 1e-11
"""The SCF's energy threshold in hartree: a Hessian needs a tightly converged density."""

SECOND_ORDER_MACRO_ITERATIONS: Final = 50
PYSCF_LOG_LEVEL: Final = 5
"""Every SCF cycle and the Hessian's step timers: a long run shows that it computes."""

Scf = Any
"""A pyscf restricted Kohn-Sham object (pyscf ships no type information)."""


class Resources(BaseModel):
    """What a calculation may use of the machine."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    threads: int = Field(default=4, ge=1, le=256)
    max_memory_mb: int = Field(default=4000, ge=500)


def build_molecule(geometry: Geometry, resources: Resources, log: TextIO) -> gto.Mole:
    """Build the pyscf molecule at the low level's basis, with pyscf's own log on ``log``."""
    lib.num_threads(resources.threads)
    molecule = gto.Mole()
    molecule.stdout = log
    molecule.verbose = PYSCF_LOG_LEVEL
    molecule.build(
        atom=[
            (symbol, tuple(xyz))
            for symbol, xyz in zip(geometry.symbols, geometry.coords_bohr.tolist(), strict=True)
        ],
        unit="Bohr",
        basis=BASIS,
        cart=CARTESIAN_D,
        symmetry=False,
        charge=0,
        spin=0,
        max_memory=resources.max_memory_mb,
    )
    return molecule


def _finish_by_second_order(scf: Scf) -> None:
    """Finish a stalled DIIS run with pyscf's second-order solver, then restart the plain SCF.

    The call follows pyscf's ``examples/scf/22-newton.py``
    (``mf.newton().kernel(mo_coeff, mo_occ)``). The second-order object copies the caller's
    ``conv_tol_grad`` and cannot always reach a tight one; ``None`` gives it pyscf's own
    √conv_tol. The restart from its density keeps ``scf.converged`` the plain object's own
    verdict at the caller's thresholds.
    """
    second_order = scf.newton()
    second_order.max_cycle = SECOND_ORDER_MACRO_ITERATIONS
    second_order.conv_tol_grad = None
    second_order.kernel(scf.mo_coeff, scf.mo_occ)
    if second_order.converged:
        scf.kernel(dm0=second_order.make_rdm1())


def converged_scf(molecule: gto.Mole) -> Scf:
    """Run the low level's Kohn-Sham SCF to convergence.

    Raises:
        ConvergenceError: Neither DIIS nor the second-order finish converged.
    """
    scf = dft.RKS(molecule)
    scf.xc = FUNCTIONAL
    scf.grids.atom_grid = GRID
    scf.grids.prune = None
    scf.conv_tol = ENERGY_TOLERANCE
    scf.kernel()
    if not scf.converged:
        _finish_by_second_order(scf)
    if not scf.converged:
        msg = f"the {FUNCTIONAL} SCF did not converge to {ENERGY_TOLERANCE:g} hartree"
        raise ConvergenceError(msg)
    return scf
