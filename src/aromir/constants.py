"""Physical constants and the supported elements, each defined once.

CODATA 2018 values. Atomic units are used throughout the package: energies in hartree,
lengths in bohr, dipole derivatives in elementary charges. Masses are kept in atomic mass
units and converted where a formula needs electron masses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

ELECTRON_MASSES_PER_AMU: Final = 1822.888486209
"""Electron masses in one atomic mass unit (1 / 5.485 799 090 65e-4)."""

WAVENUMBERS_PER_HARTREE: Final = 219474.6313705
"""cm⁻¹ per hartree (E_h / hc)."""

ANGSTROM_PER_BOHR: Final = 0.529177210903
"""Ångström per bohr."""

KM_PER_MOL_PER_SQUARED_DIPOLE_DERIVATIVE: Final = 974.8802
"""km/mol per (e/√amu)²: the double-harmonic intensity of a mode with ∂μ/∂Q = 1 e/√amu.

1 e/√amu is 2.541746 D / 0.529177 Å = 4.80320 D/Å per √amu, and an intensity of
1 (D/Å)²/amu is 42.2561 km/mol.
"""


@dataclass(frozen=True)
class Element:
    """An element the package computes: its atomic number and most abundant isotope's mass."""

    atomic_number: int
    mass_amu: float


ELEMENTS: Final[dict[str, Element]] = {
    "H": Element(1, 1.00782503223),
    "C": Element(6, 12.0),
    "N": Element(7, 14.00307400443),
    "O": Element(8, 15.99491461957),
    "F": Element(9, 18.99840316273),
    "S": Element(16, 31.9720711744),
    "Cl": Element(17, 34.968852682),
}
"""The elements of the network's training corpus, keyed by symbol (NIST isotope masses)."""
