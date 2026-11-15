"""A molecule's nuclei: which atoms, and where."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from aromir.constants import ANGSTROM_PER_BOHR, ELEMENTS

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, eq=False)
class Geometry:
    """Element symbols and Cartesian coordinates in bohr, in one fixed atom order.

    Attributes:
        symbols: One element symbol per atom, as in ``ELEMENTS`` ("C", "Cl").
        coords_bohr: Positions, shape (N, 3).
    """

    symbols: tuple[str, ...]
    coords_bohr: FloatArray

    def __post_init__(self) -> None:
        """Refuse an element the package does not know, and coordinates of the wrong shape."""
        unknown = sorted(set(self.symbols) - set(ELEMENTS))
        if unknown:
            msg = f"unsupported element(s) {unknown}; supported: {sorted(ELEMENTS)}"
            raise ValueError(msg)
        coords = np.array(self.coords_bohr, dtype=float)
        if coords.shape != (len(self.symbols), 3):
            msg = f"coordinates of shape {coords.shape} for {len(self.symbols)} atoms"
            raise ValueError(msg)
        coords.setflags(write=False)
        object.__setattr__(self, "coords_bohr", coords)

    @classmethod
    def from_angstrom(cls, symbols: tuple[str, ...], coords_angstrom: Any) -> Geometry:  # noqa: ANN401  array-like
        """Build a geometry from coordinates in Ångström."""
        return cls(symbols, np.asarray(coords_angstrom, dtype=float) / ANGSTROM_PER_BOHR)

    @property
    def n_atoms(self) -> int:
        """The number of atoms."""
        return len(self.symbols)

    @property
    def masses_amu(self) -> FloatArray:
        """Each atom's mass in atomic mass units, shape (N,)."""
        return np.array([ELEMENTS[s].mass_amu for s in self.symbols])

    @property
    def atomic_numbers(self) -> NDArray[np.int64]:
        """Each atom's atomic number, shape (N,)."""
        return np.array([ELEMENTS[s].atomic_number for s in self.symbols], dtype=np.int64)

    def reordered(self, order: list[int]) -> Geometry:
        """The same molecule with atom ``order[k]`` of this geometry in position k."""
        if sorted(order) != list(range(self.n_atoms)):
            msg = f"{order} is not a permutation of the {self.n_atoms} atoms"
            raise ValueError(msg)
        return Geometry(tuple(self.symbols[k] for k in order), self.coords_bohr[order])

    def to_dict(self) -> dict[str, Any]:
        """A JSON-ready record: symbols, coordinates and masses."""
        return {
            "symbols": list(self.symbols),
            "coords_bohr": self.coords_bohr.tolist(),
            "masses_amu": self.masses_amu.tolist(),
        }
