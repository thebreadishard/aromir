"""The family of a vibrational mode: CH stretch, CH out-of-plane, ring in-plane, or other.

The network's error is measured per family, so the certificate names each band's family
by the same rule the evaluation used. The rule reads the low level's modes: where a mode
lies, how much of its motion is on hydrogens, and how much of it points out of the
molecular plane.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, Literal

import numpy as np

if TYPE_CHECKING:
    from aromir.molecule import FloatArray, Geometry
    from aromir.vibrations import Modes

Family = Literal["CH-stretch", "CH-oop", "ring-ip", "other"]

FAMILIES: Final[tuple[Family, ...]] = ("ring-ip", "CH-stretch", "CH-oop", "other")

PLANAR_WITHIN_BOHR: Final = 0.5
"""A molecule is planar when no atom lies further than this from its plane."""

_STRETCH_ABOVE: Final = 2800.0
_STRETCH_HYDROGEN_SHARE: Final = 0.5
_OOP_WINDOW: Final = (600.0, 1050.0)
_OOP_SHARE: Final = 0.6
_OOP_HYDROGEN_SHARE: Final = 0.3
_RING_WINDOW: Final = (950.0, 1750.0)
_RING_OOP_SHARE: Final = 0.3

RULES: Final[dict[Family, str]] = {
    "ring-ip": (
        "between 950 and 1750 cm⁻¹ and in the molecular plane: ring stretches and in-plane "
        "bends, and with them every other mode there, such as a C=O stretch or a CH₃ "
        "deformation; in a non-planar molecule every mode in that window"
    ),
    "CH-stretch": (
        "above 2800 cm⁻¹ with most of its motion on hydrogens: the C–H stretches, and by "
        "the same rule an O–H or N–H stretch"
    ),
    "CH-oop": (
        "between 600 and 1050 cm⁻¹, out of the molecular plane and on hydrogens; a "
        "non-planar molecule has none, and its C–H wags fall under other or ring-ip"
    ),
    "other": "every mode the three rules above do not take",
}
"""What each family's name stands for: the rule, in the words a reader of a band needs.
The names are the evaluation's; the rule decides."""


def _plane_normal(geometry: Geometry) -> FloatArray | None:
    """Return the molecular plane's normal, or None for a molecule that is not planar.

    The normal is the principal axis of the largest moment of inertia.
    """
    masses = geometry.masses_amu
    centred = geometry.coords_bohr - masses @ geometry.coords_bohr / masses.sum()
    inertia = sum(
        mass * (np.dot(r, r) * np.eye(3) - np.outer(r, r))
        for r, mass in zip(centred, masses, strict=True)
    )
    moments, axes = np.linalg.eigh(inertia)
    normal = axes[:, np.argmax(moments)]
    if float(np.abs(centred @ normal).max()) >= PLANAR_WITHIN_BOHR:
        return None
    return np.asarray(normal, dtype=float)


def is_planar(geometry: Geometry) -> bool:
    """Tell whether the family rule treats the molecule as planar."""
    return _plane_normal(geometry) is not None


def hydrogen_carriers(geometry: Geometry, modes: Modes) -> list[str]:
    """Return, per mode, the element whose hydrogens carry most of the mode's hydrogen motion.

    A hydrogen belongs to the heavy atom nearest to it. The answer tells a C–H stretch from
    an O–H or N–H stretch, which the family rule does not; it is "" for a molecule without
    hydrogens.
    """
    symbols = np.array(geometry.symbols)
    hydrogens = np.flatnonzero(symbols == "H")
    heavy = np.flatnonzero(symbols != "H")
    if not len(hydrogens) or not len(heavy):
        return [""] * modes.vectors.shape[1]
    x = geometry.coords_bohr
    distances = np.linalg.norm(x[hydrogens][:, None, :] - x[heavy][None, :, :], axis=-1)
    carrier = symbols[heavy[np.argmin(distances, axis=1)]]
    amplitude = (modes.vectors.reshape(geometry.n_atoms, 3, -1) ** 2).sum(axis=1)[hydrogens]
    elements = sorted(set(carrier))
    shares = np.array([amplitude[carrier == element].sum(axis=0) for element in elements])
    return [elements[int(index)] for index in np.argmax(shares, axis=0)]


def classify(geometry: Geometry, modes: Modes) -> list[Family]:
    """Return the family of each mode of the low level, in the modes' order.

    The first rule that applies: a CH stretch lies above 2800 cm⁻¹ with most of its motion
    on hydrogens; a CH out-of-plane bend lies between 600 and 1050 cm⁻¹, points out of the
    plane and moves hydrogens; a ring in-plane mode lies between 950 and 1750 cm⁻¹ and
    stays in the plane. In a non-planar molecule no mode counts as out of plane.
    """
    per_atom = modes.vectors.reshape(geometry.n_atoms, 3, -1)
    amplitude = (per_atom**2).sum(axis=1)
    on_hydrogen = amplitude[np.array(geometry.symbols) == "H"].sum(axis=0)
    normal = _plane_normal(geometry)
    if normal is None:
        out_of_plane = np.zeros(modes.vectors.shape[1])
    else:
        out_of_plane = ((per_atom * normal[None, :, None]).sum(axis=1) ** 2).sum(axis=0)
    families: list[Family] = []
    for position, hydrogen, oop in zip(modes.wavenumbers, on_hydrogen, out_of_plane, strict=True):
        if position > _STRETCH_ABOVE and hydrogen > _STRETCH_HYDROGEN_SHARE:
            families.append("CH-stretch")
        elif (
            _OOP_WINDOW[0] < position < _OOP_WINDOW[1]
            and oop > _OOP_SHARE
            and hydrogen > _OOP_HYDROGEN_SHARE
        ):
            families.append("CH-oop")
        elif _RING_WINDOW[0] < position < _RING_WINDOW[1] and oop < _RING_OOP_SHARE:
            families.append("ring-ip")
        else:
            families.append("other")
    return families
