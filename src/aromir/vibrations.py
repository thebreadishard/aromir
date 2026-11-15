"""Harmonic vibrations of a Cartesian Hessian: rigid motions, projection, normal modes.

Everything here is linear algebra on arrays; no quantum chemistry is imported. A Hessian is
Cartesian, in hartree/bohr², with shape (3N, 3N); masses are in atomic mass units.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from aromir.constants import ELECTRON_MASSES_PER_AMU, WAVENUMBERS_PER_HARTREE

if TYPE_CHECKING:
    from aromir.molecule import FloatArray, Geometry

N_RIGID_MOTIONS = 6
"""Translations and rotations of a non-linear molecule."""

_RANK_TOLERANCE = 1e-8
_ORTHONORMAL_TOLERANCE = 1e-10


@dataclass(frozen=True, eq=False)
class Modes:
    """The vibrational modes of a Hessian.

    Attributes:
        wavenumbers: Signed harmonic wavenumbers in cm⁻¹, ascending, shape (3N−6,); an
            imaginary mode is negative.
        vectors: Mass-weighted, orthonormal mode vectors as columns, shape (3N, 3N−6), each
            orthogonal to the rigid motions.
    """

    wavenumbers: FloatArray
    vectors: FloatArray

    @property
    def n_imaginary(self) -> int:
        """How many modes have a negative curvature."""
        return int((self.wavenumbers < 0).sum())


def signed_wavenumber(curvature: FloatArray) -> FloatArray:
    """Convert mass-weighted curvatures (hartree / (bohr² mₑ)) to signed wavenumbers in cm⁻¹."""
    return np.sign(curvature) * np.sqrt(np.abs(curvature)) * WAVENUMBERS_PER_HARTREE


def mass_weights(masses_amu: FloatArray) -> FloatArray:
    """Return √m per Cartesian coordinate, in √(electron masses), shape (3N,)."""
    return np.sqrt(np.repeat(np.asarray(masses_amu, dtype=float) * ELECTRON_MASSES_PER_AMU, 3))


def rigid_basis(geometry: Geometry) -> FloatArray:
    """Return an orthonormal basis (3N, 6) of the mass-weighted translations and rotations.

    Raises:
        ValueError: The molecule is linear (or a single atom): it has fewer than six rigid
            motions, which this package's 3N−6 convention does not cover.
    """
    masses = geometry.masses_amu
    root_mass = np.sqrt(masses)[:, None]
    centred = geometry.coords_bohr - masses @ geometry.coords_bohr / masses.sum()
    axes = np.eye(3)
    translations = [np.tile(axis, (geometry.n_atoms, 1)) for axis in axes]
    rotations = [np.cross(axis, centred) for axis in axes]
    motions = np.array([(motion * root_mass).ravel() for motion in translations + rotations]).T
    if np.linalg.matrix_rank(motions, tol=_RANK_TOLERANCE) < N_RIGID_MOTIONS:
        msg = "a linear molecule has no six rigid motions; only non-linear molecules are supported"
        raise ValueError(msg)
    basis, _ = np.linalg.qr(motions)
    return np.asarray(basis, dtype=float)


def _mass_weighted(hessian: FloatArray, masses_amu: FloatArray) -> FloatArray:
    weights = mass_weights(masses_amu)
    return 0.5 * (hessian + hessian.T) / np.outer(weights, weights)


def project_rigid_motions(hessian: FloatArray, geometry: Geometry) -> FloatArray:
    """Return the Hessian with translations and rotations projected out.

    The projector acts on the mass-weighted Hessian and the result is transformed back, so
    the projected Hessian has six exact zero modes and the vibrational ones untouched.
    """
    weights = mass_weights(geometry.masses_amu)
    rigid = rigid_basis(geometry)
    projector = np.eye(len(weights)) - rigid @ rigid.T
    projected = projector @ _mass_weighted(hessian, geometry.masses_amu) @ projector
    return np.asarray(projected * np.outer(weights, weights), dtype=float)


def vibrational_modes(hessian: FloatArray, geometry: Geometry) -> Modes:
    """Diagonalise a Hessian in the space orthogonal to the rigid motions.

    Working in that (3N−6)-dimensional space returns exactly the vibrational modes, however
    soft: no mode is told from a rigid motion by a wavenumber threshold.
    """
    rigid = rigid_basis(geometry)
    full, _ = np.linalg.qr(np.hstack([rigid, np.eye(rigid.shape[0])]))
    internal = full[:, N_RIGID_MOTIONS:]
    curvature, vectors = np.linalg.eigh(
        internal.T @ _mass_weighted(hessian, geometry.masses_amu) @ internal
    )
    return Modes(signed_wavenumber(curvature), np.asarray(internal @ vectors, dtype=float))


def displacement(
    direction: FloatArray, masses_amu: FloatArray, largest_step_bohr: float
) -> tuple[FloatArray, float]:
    """Return a Cartesian step along a mass-weighted direction, and its mass-weighted length.

    The step, shape (N, 3) in bohr, moves the atom that moves most by ``largest_step_bohr``.
    """
    cartesian = (direction / mass_weights(masses_amu)).reshape(-1, 3)
    length = largest_step_bohr / float(np.max(np.linalg.norm(cartesian, axis=1)))
    return length * cartesian, length


def column(
    gradient_plus: FloatArray, gradient_minus: FloatArray, length: float, masses_amu: FloatArray
) -> FloatArray:
    """Return the mass-weighted Hessian's column along a direction, by central differences.

    The gradients (hartree/bohr) are those at plus and minus ``length`` along the
    mass-weighted direction.
    """
    difference = np.ravel(gradient_plus) - np.ravel(gradient_minus)
    return np.asarray(difference / mass_weights(masses_amu) / (2 * length), dtype=float)


def with_measured_columns(
    hessian: FloatArray, masses_amu: FloatArray, directions: FloatArray, columns: FloatArray
) -> FloatArray:
    """Return the Hessian with its mass-weighted rows and columns along U replaced by C.

    For orthonormal mass-weighted directions U (3N, k) and measured columns C = H U,

        H' = Q H Q + C Uᵀ + U Cᵀ − U S Uᵀ,   Q = 1 − U Uᵀ,   S = (UᵀC + CᵀU) / 2,

    which keeps the Hessian outside span(U) and takes the measurement inside it.

    Raises:
        ValueError: The directions are not orthonormal, or the shapes differ.
    """
    if directions.shape != columns.shape:
        msg = f"directions {directions.shape} and columns {columns.shape} differ in shape"
        raise ValueError(msg)
    k = directions.shape[1]
    if np.abs(directions.T @ directions - np.eye(k)).max() > _ORTHONORMAL_TOLERANCE:
        msg = "the directions are not orthonormal"
        raise ValueError(msg)
    weights = mass_weights(masses_amu)
    outside = np.eye(len(weights)) - directions @ directions.T
    inside = 0.5 * (directions.T @ columns + columns.T @ directions)
    patched = (
        outside @ _mass_weighted(hessian, masses_amu) @ outside
        + columns @ directions.T
        + directions @ columns.T
        - directions @ inside @ directions.T
    )
    return np.asarray(0.5 * (patched + patched.T) * np.outer(weights, weights), dtype=float)
