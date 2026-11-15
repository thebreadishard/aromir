"""The orbital response a Hessian solves, kept, checked and used for dipole derivatives.

An analytic Hessian solves the coupled-perturbed equations for the response of the
orbitals to every nuclear displacement. The same response gives the atomic polar tensor
(∂μ/∂R) at no extra cost, so it is solved once and handed to both.

pyscf's Krylov solver can stop before its solution satisfies those equations (pyscf issue
2702). A Hessian barely feels that; a polar tensor does. ``refine`` therefore checks the
response against the equations themselves and corrects it until the residual is below
``RESIDUAL_BOUND``; the caller decides what a residual above the bound means.

The calls into pyscf follow pyscf's own ``hessian/rhf.py`` (``HessianBase.kernel``,
exercised by ``hessian/test/test_rks.py``) and pyscf-properties'
``prop/infrared/rhf.py``, which keeps the response the same way. Closed shells only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, NamedTuple

import numpy as np
from pyscf import lib
from pyscf.hessian import rhf as rhf_hessian
from scipy.sparse.linalg import LinearOperator, gmres

if TYPE_CHECKING:
    from aromir.molecule import FloatArray

RESIDUAL_BOUND: Final = 1e-6
"""Largest element of the response equations' residual that a polar tensor may rest on."""

SUM_RULE_LIMIT: Final = 5e-4
"""Largest violation, in e, of the translation sum rule Σ_A ∂μ/∂R_A = 0 of a neutral molecule."""

_BATCHED_ROUNDS: Final = 3
_KRYLOV_TOLERANCE: Final = 1e-9
_KRYLOV_CYCLES: Final = 50
_GMRES_TOLERANCE: Final = 1e-10
_GMRES_ITERATIONS: Final = 300
_GMRES_RESTART: Final = 60

HessianMethod = Any
"""A pyscf Hessian object of a converged restricted calculation."""


class Response(NamedTuple):
    """The orbital response as pyscf's ``solve_mo1`` returns it.

    Attributes:
        mo1: Per atom, ∂C_occ/∂R in the AO basis, shape (3, n_ao, n_occ).
        mo_e1: Per atom, the first-order orbital energies.
        h1ao: Per atom, the derivative Fock matrices the equations are solved for.
    """

    mo1: list[FloatArray]
    mo_e1: list[FloatArray]
    h1ao: Any


@dataclass(frozen=True)
class Refinement:
    """What the residual check found and did.

    Attributes:
        residual_before: Largest residual element of pyscf's own solution.
        residual_after: Largest residual element of the refined solution.
        batched_rounds: Defect-correction rounds over all right-hand sides at once.
        single_columns: Right-hand sides that then needed a solve of their own.
    """

    residual_before: float
    residual_after: float
    batched_rounds: int
    single_columns: int


def solve(hessian_method: HessianMethod) -> Response:
    """Solve the response of every atom: the first two steps of pyscf's Hessian kernel."""
    scf = hessian_method.base
    h1ao = hessian_method.make_h1(scf.mo_coeff, scf.mo_occ, None, hessian_method.atmlst)
    mo1, mo_e1 = hessian_method.solve_mo1(
        scf.mo_energy, scf.mo_coeff, scf.mo_occ, h1ao, None, hessian_method.atmlst
    )
    return Response(mo1, mo_e1, h1ao)


def hessian(hessian_method: HessianMethod, response: Response) -> FloatArray:
    """Return the Cartesian Hessian (3N, 3N) built on a response that is already solved.

    These are the remaining steps of pyscf's Hessian kernel. The low level has neither a
    dispersion nor a counterpoise term, so none is added.
    """
    scf = hessian_method.base
    electronic = hessian_method.hess_elec(
        scf.mo_energy,
        scf.mo_coeff,
        scf.mo_occ,
        mo1=response.mo1,
        mo_e1=response.mo_e1,
        h1ao=response.h1ao,
        atmlst=hessian_method.atmlst,
    )
    blocks = electronic + hessian_method.hess_nuc(hessian_method.mol, atmlst=hessian_method.atmlst)
    n_coordinates = 3 * hessian_method.mol.natm
    cartesian = blocks.transpose(0, 2, 1, 3).reshape(n_coordinates, n_coordinates)
    return np.asarray(0.5 * (cartesian + cartesian.T), dtype=float)


class _Equations:
    """The coupled-perturbed equations of all 3N displacements, in the MO basis.

    With x = Cᵀ S mo1, pyscf's ``scf/cphf.py`` ``solve_withs1`` solves, per right-hand side,
    x + A x = b on the virtual rows (A the induced potential divided by the orbital-energy
    gaps) and sets x = −s1/2 on the occupied rows.
    """

    def __init__(self, hessian_method: HessianMethod, h1ao: Any) -> None:  # noqa: ANN401  pyscf's per-atom container
        scf = hessian_method.base
        molecule = scf.mol
        self._coefficients = scf.mo_coeff
        self._overlap = scf.get_ovlp()
        self._occupied = scf.mo_occ > 0
        self._virtual = scf.mo_occ == 0
        energies = scf.mo_energy
        self._inverse_gaps = 1.0 / (
            energies[self._virtual][:, None] - energies[self._occupied][None, :]
        )
        self._induced = rhf_hessian.gen_vind(scf, scf.mo_coeff, scf.mo_occ)
        self.shape = (self._coefficients.shape[1], int(self._occupied.sum()))
        self.right_hand_sides = self._right_hand_sides(molecule, h1ao, energies)

    def _right_hand_sides(self, molecule: Any, h1ao: Any, energies: FloatArray) -> FloatArray:  # noqa: ANN401
        coefficients, occupied = self._coefficients, self._coefficients[:, self._occupied]
        overlap_gradient = -molecule.intor("int1e_ipovlp", comp=3)
        per_atom = []
        for atom, (_, _, first, last) in enumerate(molecule.aoslice_by_atom()):
            s1_ao = np.zeros((3, *self._overlap.shape))
            s1_ao[:, first:last] += overlap_gradient[:, first:last]
            s1_ao[:, :, first:last] += overlap_gradient[:, first:last].transpose(0, 2, 1)
            s1 = np.einsum("pq,xqr,ri->xpi", coefficients.T, s1_ao, occupied)
            h1 = np.einsum("pq,xqr,ri->xpi", coefficients.T, np.asarray(h1ao[atom]), occupied)
            b = h1 - s1 * energies[self._occupied]
            b[:, self._virtual] *= -self._inverse_gaps
            b[:, self._occupied] = -0.5 * s1[:, self._occupied]
            per_atom.append(b)
        return np.concatenate(per_atom)

    def to_mo_basis(self, mo1: list[FloatArray]) -> FloatArray:
        """Stack a per-atom AO response as x (3N, n_mo, n_occ), occupied rows as pyscf sets them."""
        x = np.concatenate(
            [np.einsum("pq,qr,xri->xpi", self._coefficients.T, self._overlap, m) for m in mo1]
        )
        x[:, self._occupied] = self.right_hand_sides[:, self._occupied]
        return x

    def to_ao_basis(self, x: FloatArray) -> list[FloatArray]:
        """Return x in pyscf's per-atom AO layout."""
        return [
            np.einsum("pq,xqi->xpi", self._coefficients, x[first : first + 3])
            for first in range(0, len(x), 3)
        ]

    def apply(self, flat: FloatArray) -> FloatArray:
        """Return A x for flattened x (rows, n_mo·n_occ): zero on the occupied rows."""
        result = self._induced(flat.reshape(-1, *self.shape)).reshape(-1, *self.shape)
        result[:, self._virtual] *= self._inverse_gaps
        result[:, self._occupied] = 0.0
        return np.asarray(result.reshape(len(result), -1), dtype=float)

    def residual(self, x: FloatArray) -> FloatArray:
        """Return b − x − A x, which is zero on the occupied rows."""
        flat = x.reshape(len(x), -1)
        return np.asarray(self.right_hand_sides - x - self.apply(flat).reshape(x.shape))


def _largest(residual: FloatArray) -> float:
    return float(np.abs(residual).max())


def _correct_all_at_once(equations: _Equations, x: FloatArray) -> FloatArray:
    """One round of defect correction: solve (1 + A) δ = r for every right-hand side together."""
    residual = equations.residual(x)
    correction = lib.krylov(
        equations.apply,
        residual.reshape(len(residual), -1),
        tol=_KRYLOV_TOLERANCE,
        max_cycle=_KRYLOV_CYCLES,
    )
    return np.asarray(x + correction.reshape(x.shape))


def _correct_one_by_one(
    equations: _Equations, x: FloatArray, bound: float
) -> tuple[FloatArray, int]:
    """Solve each right-hand side still above the bound alone, by restarted GMRES.

    The batched solver can stall where a single right-hand side converges.
    """
    residual = equations.residual(x)
    dimension = int(np.prod(equations.shape))
    operator = LinearOperator(
        (dimension, dimension),
        matvec=lambda y: y + equations.apply(y.reshape(1, -1)).ravel(),
        dtype=float,
    )
    above = np.flatnonzero(np.abs(residual.reshape(len(residual), -1)).max(axis=1) > bound)
    corrected = x.copy()
    for column in above:
        correction, _ = gmres(
            operator,
            residual[column].ravel(),
            rtol=_GMRES_TOLERANCE,
            atol=0.0,
            maxiter=_GMRES_ITERATIONS,
            restart=_GMRES_RESTART,
        )
        corrected[column] += correction.reshape(equations.shape)
    return corrected, len(above)


def refine(
    hessian_method: HessianMethod, response: Response, bound: float = RESIDUAL_BOUND
) -> tuple[list[FloatArray], Refinement]:
    """Check a response against its own equations and correct it where it is off.

    Up to three rounds of batched defect correction run while the residual exceeds the
    bound; right-hand sides still above it are then solved one by one.

    Returns:
        The refined ``mo1`` in pyscf's layout, and the record of the residuals. Nothing is
        raised when the bound is not reached: the record says so and the caller decides.
    """
    equations = _Equations(hessian_method, response.h1ao)
    x = equations.to_mo_basis(response.mo1)
    before = after = _largest(equations.residual(x))
    rounds = 0
    while after > bound and rounds < _BATCHED_ROUNDS:
        x = _correct_all_at_once(equations, x)
        after = _largest(equations.residual(x))
        rounds += 1
    single_columns = 0
    if after > bound:
        x, single_columns = _correct_one_by_one(equations, x, bound)
        after = _largest(equations.residual(x))
    return equations.to_ao_basis(x), Refinement(before, after, rounds, single_columns)


def polar_tensor(scf: Any, mo1: list[FloatArray]) -> FloatArray:  # noqa: ANN401  a pyscf SCF object
    """Return the atomic polar tensor P (3, 3N), ∂μ_t/∂R_Ax in e, from a solved response.

    ∂μ_t/∂R_Ax = Z_A δ_tx − Tr(∂D/∂R_Ax · r_t) − Tr(D · ∂r_t/∂R_Ax): the nuclear charge,
    the response of the density, and the basis functions that move with atom A.
    """
    molecule = scf.mol
    occupied = scf.mo_coeff[:, scf.mo_occ > 0]
    density = scf.make_rdm1()
    position = molecule.intor("int1e_r", comp=3)
    n_ao = molecule.nao
    position_gradient = molecule.intor("int1e_irp", comp=9).reshape(3, 3, n_ao, n_ao)
    charges = molecule.atom_charges()
    tensor = np.zeros((3, 3 * molecule.natm))
    for atom, (_, _, first, last) in enumerate(molecule.aoslice_by_atom()):
        for axis in range(3):
            density_response = 2.0 * (mo1[atom][axis] @ occupied.T)
            density_response = density_response + density_response.T
            column = 3 * atom + axis
            tensor[:, column] -= np.einsum("pq,tpq->t", density_response, position)
            moving_functions = position_gradient[:, axis, :, first:last]
            tensor[:, column] += 2.0 * np.einsum(
                "pq,tpq->t", density[:, first:last], moving_functions
            )
        tensor[:, 3 * atom : 3 * atom + 3] += charges[atom] * np.eye(3)
    return tensor


def sum_rule_violation(tensor: FloatArray) -> float:
    """Return max |Σ_A ∂μ/∂R_A| in e: zero for an exact polar tensor of a neutral molecule."""
    n_atoms = tensor.shape[1] // 3
    return float(np.abs(tensor.reshape(3, n_atoms, 3).sum(axis=1)).max())
