"""Redundant primitive internal coordinates, as geomeTRIC builds them.

The network predicts its correction in internal coordinates: bonds, angles, dihedrals and
out-of-plane bends. Their Wilson B matrix carries a correction ΔF to Cartesian
coordinates as ΔH = Bᵀ ΔF B, which is symmetric and free of translations and rotations
by construction (but for a bent linear angle, whose row holds a small share of a
rotation: 5e-4 at the start geometry of phenyl isocyanate).

geomeTRIC keeps two of the three angles at a planar centre and chooses by atom number, so
two atom orders of one molecule get different primitives. The ``canonical`` angle choice
builds the primitives on the atoms in a canonical order instead. A dihedral near ±π jumps
by 2π under a tiny distortion; the ``cos`` encoding gives the features cos φ, which does
not jump, and ``cossin`` adds sin φ.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, Literal

import numpy as np
from geometric.internal import (
    Angle,
    Dihedral,
    Distance,
    LinearAngle,
    OutOfPlane,
    PrimitiveInternalCoordinates,
)
from geometric.molecule import Molecule
from pydantic import BaseModel, ConfigDict
from rdkit import Chem

from aromir.constants import ANGSTROM_PER_BOHR
from aromir.errors import InternalCoordinatesError
from aromir.network import graph
from aromir.vibrations import N_RIGID_MOTIONS

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from aromir.molecule import FloatArray, Geometry

Primitive = Any
"""One geomeTRIC primitive (geomeTRIC ships no type information)."""

PRIMITIVE_KINDS: Final = ("Distance", "Angle", "Dihedral", "OutOfPlane", "LinearAngle", "other")
BOND: Final = 0
ANGLE: Final = 1
DIHEDRAL: Final = 2
OUT_OF_PLANE: Final = 3
LINEAR_ANGLE: Final = 4
OTHER: Final = 5

_PARITY_TOLERANCE: Final = 1e-6

_SMALLEST_REAL: Final = 1e-6
"""Singular value 3N − 6 of B, as a share of the largest, must lie above this. A real value
falls with the length of the molecule — 1.0e-2 at the least over the 1,240 molecules of the
networks' training corpus (60 atoms at most), 1.4e-3 for a rod of eight rings and 96 atoms
— and a value that is no direction of the primitives is rounding residue, 1e-15."""

_LARGEST_NULL: Final = 1e-3
"""Singular value 3N − 5 of B, as a share of the largest, must lie at or below this. Over
the corpus every such value is below 4e-15, but for diphenylacetylene, 3.5e-8: geomeTRIC
measures a linear angle against a fixed axis, and where two such groups share an axis that
is bent by θ rad a rotation enters B at about 0.01 θ of the largest value. Start geometries
of unsymmetrical alkynes and allenes reach 4e-4; the limit is a bend of about 0.1 rad."""

_LARGEST_NULL_OF_REAL: Final = 0.1
"""Singular value 3N − 5 of B must also lie at or below this share of value 3N − 6: the
error of dropping a rotation follows their ratio. Up to 60 atoms the largest ratio measured
is 0.064 (1,2-bis(phenylethynyl)benzene, B⁺ off by 0.5 %); a rod of twelve rings joined by
alkynes (144 atoms) has 0.2 at its start geometry, and its B⁺ would be off by 23 %."""


class PrimitiveSettings(BaseModel):
    """How the primitives are built and how a dihedral's value enters the features."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    angle_choice: Literal["geometric", "canonical"] = "geometric"
    dihedral_encoding: Literal["raw", "cos", "cossin"] = "raw"


@dataclass(frozen=True, eq=False)
class Primitives:
    """A molecule's primitives and the geometry they were built on.

    Attributes:
        internals: geomeTRIC's primitives, K of them.
        built_on: The geometry in the order the primitives number its atoms.
        order: ``built_on``'s atom k is the input geometry's atom ``order[k]``.
        wilson_b: The Wilson B matrix (K, 3N), columns in the input geometry's atom order.
    """

    internals: list[Primitive]
    built_on: Geometry
    order: NDArray[np.int64]
    wilson_b: FloatArray

    def atoms(self, index: int) -> tuple[int, ...]:
        """Return a primitive's atoms as indices into ``built_on``."""
        primitive = self.internals[index]
        return tuple(getattr(primitive, key) for key in "abcd" if hasattr(primitive, key))

    def value(self, index: int) -> float:
        """Return a primitive's value at ``built_on`` (bohr or radians)."""
        return float(self.internals[index].value(self.built_on.coords_bohr.ravel()))


def kind(primitive: Primitive) -> int:
    """Return a primitive's index into ``PRIMITIVE_KINDS``."""
    for index, cls in enumerate((Distance, Angle, Dihedral, OutOfPlane, LinearAngle)):
        if isinstance(primitive, cls):
            return index
    return OTHER


def canonical_order(geometry: Geometry) -> NDArray[np.int64]:
    """Return the atom indices in RDKit's canonical rank order on the feature bond graph.

    Atoms the ranks cannot tell apart are symmetry-equivalent, so another tie-break
    changes the primitive set only by a symmetry of the molecule.
    """
    molecule = Chem.RWMol()
    for symbol in geometry.symbols:
        molecule.AddAtom(Chem.Atom(symbol))
    for atom, neighbours in enumerate(graph.feature_bonds(geometry)):
        for neighbour in neighbours:
            if atom < neighbour:
                molecule.AddBond(atom, neighbour, Chem.BondType.SINGLE)
    molecule.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(molecule)
    ranks = list(Chem.CanonicalRankAtoms(molecule, breakTies=True))
    return np.argsort(ranks, kind="stable")


def _geometric_coordinates(geometry: Geometry) -> Any:  # noqa: ANN401  a geomeTRIC object
    molecule = Molecule()
    molecule.elem = list(geometry.symbols)
    molecule.xyzs = [geometry.coords_bohr * ANGSTROM_PER_BOHR]
    return PrimitiveInternalCoordinates(molecule, build=True, connect=True, addcart=False)


def build(geometry: Geometry, settings: PrimitiveSettings) -> Primitives:
    """Build a geometry's primitives and their Wilson B matrix."""
    if settings.angle_choice == "canonical":
        order = canonical_order(geometry)
    else:
        order = np.arange(geometry.n_atoms)
    built_on = geometry.reordered(order.tolist())
    coordinates = _geometric_coordinates(built_on)
    b_built = np.asarray(coordinates.wilsonB(built_on.coords_bohr.ravel()), dtype=float)
    wilson_b = np.empty_like(b_built)
    wilson_b[:, (3 * order[:, None] + np.arange(3)).ravel()] = b_built
    return Primitives(list(coordinates.Internals), built_on, order, wilson_b)


def pseudo_inverse(wilson_b: FloatArray) -> FloatArray:
    """Return B⁺ from exactly the 3N − 6 largest singular values of a Wilson B matrix.

    Internal coordinates do not change under the six rigid motions, so of B's 3N columns
    3N − 6 are independent and every further singular value is rounding residue, or the
    small share of a rotation that a bent linear angle lets in. A cutoff relative to the
    largest value sits inside that residue: at ``np.linalg.pinv``'s default of 1e-15 one
    LAPACK build returns a null value of 9e-16 and drops it, another returns 7e-15 for the
    same matrix and keeps it, and B⁺ reaches 1e13. The count is known, so the count
    decides, and the values on either side of it are checked against where they belong.

    Args:
        wilson_b: (K, 3N), the derivatives of K primitives to the Cartesian coordinates of
            a non-linear molecule.

    Raises:
        InternalCoordinatesError: B is not finite, or the primitives span fewer than
            3N − 6 directions, or more.
    """
    n_primitives, n_cartesian = wilson_b.shape
    n_internal = n_cartesian - N_RIGID_MOTIONS
    if n_internal < 1 or n_primitives < n_internal:
        msg = (
            f"{n_primitives} primitives on {n_cartesian} Cartesian coordinates span fewer "
            f"than 3N − 6 = {n_internal} directions"
        )
        raise InternalCoordinatesError(msg)
    if not np.isfinite(wilson_b).all() or not wilson_b.any():
        msg = "the Wilson B matrix is zero or not finite"
        raise InternalCoordinatesError(msg)
    left, values, right = np.linalg.svd(wilson_b, full_matrices=False)
    if values[n_internal - 1] <= _SMALLEST_REAL * values[0]:
        msg = (
            f"the primitives span fewer than 3N − 6 = {n_internal} directions: singular "
            f"value {n_internal} of B is {values[n_internal - 1]:.1e}, the largest "
            f"{values[0]:.1e}"
        )
        raise InternalCoordinatesError(msg)
    if len(values) > n_internal and values[n_internal] > _LARGEST_NULL * values[0]:
        msg = (
            f"the primitives span more than 3N − 6 = {n_internal} directions: singular "
            f"value {n_internal + 1} of B is {values[n_internal]:.1e}, the largest "
            f"{values[0]:.1e}; a linear molecule, or a linear group (an alkyne between two "
            "groups, an allene) bent by more than about 6°, whose angles' rows then hold "
            "a rotation"
        )
        raise InternalCoordinatesError(msg)
    if (
        len(values) > n_internal
        and values[n_internal] > _LARGEST_NULL_OF_REAL * values[n_internal - 1]
    ):
        msg = (
            f"singular values {n_internal} and {n_internal + 1} of B, "
            f"{values[n_internal - 1]:.1e} and {values[n_internal]:.1e}, are too close to "
            "tell the last internal direction from a rotation that bent linear groups let "
            "in (a long molecule with several alkynes or allenes)"
        )
        raise InternalCoordinatesError(msg)
    kept = (right[:n_internal].T / values[:n_internal]) @ left[:, :n_internal].T
    return np.asarray(kept, dtype=float)


def internal_force_constants(hessian: FloatArray, wilson_b: FloatArray) -> FloatArray:
    """Return F = B⁺ᵀ H B⁺, the minimum-norm internal form of a Cartesian Hessian."""
    inverse = pseudo_inverse(wilson_b)
    return np.asarray(inverse.T @ hessian @ inverse, dtype=float)


def parities(primitives: Primitives) -> tuple[NDArray[np.bool_], FloatArray]:
    """Return which primitives change sign under a reflection, and sin(value) of those.

    Decided numerically, not by kind: each B row at the geometry mirrored through z = 0 is
    compared with the mirrored B row. sin(value) is the pseudo-scalar an odd primitive
    carries; it is zero for the even ones and for a planar molecule.

    Raises:
        ValueError: A primitive is neither even nor odd, which no primitive should be.
    """
    coordinates = _geometric_coordinates(primitives.built_on)
    x = primitives.built_on.coords_bohr
    mirror = np.diag([1.0, 1.0, -1.0])
    b_here = np.asarray(coordinates.wilsonB(x.ravel()))
    b_mirrored = np.asarray(coordinates.wilsonB((x @ mirror).ravel()))
    expected = (b_here.reshape(len(b_here), -1, 3) @ mirror).reshape(len(b_here), -1)
    norms = np.linalg.norm(b_mirrored, axis=1) * np.linalg.norm(expected, axis=1) + 1e-300
    cosine = (b_mirrored * expected).sum(axis=1) / norms
    worst = float(np.abs(np.abs(cosine) - 1.0).max())
    if worst > _PARITY_TOLERANCE:
        msg = f"a primitive is neither even nor odd under a reflection (|cos| off by {worst:.1e})"
        raise ValueError(msg)
    odd = cosine < 0
    sines = np.array(
        [np.sin(primitives.value(k)) if odd[k] else 0.0 for k in range(len(odd))], dtype=float
    )
    return odd, sines
