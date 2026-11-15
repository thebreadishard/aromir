"""Infrared bands from a Hessian and dipole derivatives, and the curve drawn from them.

Intensities are double-harmonic: A_k = 974.88 km/mol · |P L_k / √m|², with P the atomic
polar tensor (∂μ/∂R, shape (3, 3N), in e) and L_k a mass-weighted mode vector.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import numpy as np

from aromir.constants import KM_PER_MOL_PER_SQUARED_DIPOLE_DERIVATIVE
from aromir.vibrations import vibrational_modes

if TYPE_CHECKING:
    from aromir.molecule import FloatArray, Geometry

FWHM: Final = 10.0
"""Full width at half maximum of the drawn Lorentzian, cm⁻¹."""

GRID: Final = (500.0, 3500.0, 1.0)
"""The drawn range: first wavenumber, end (exclusive) and step, cm⁻¹."""

SCALING_REGION_EDGES: Final = (1111.0, 2500.0)
"""cm⁻¹; the regions of Bauschlicher et al. (2018): below, between (edges included), above."""

SCALING_FACTORS: Final = (0.956, 0.952, 0.960)
"""The NASA Ames PAH database's B3LYP scale factor of each region."""


@dataclass(frozen=True, eq=False)
class Bands:
    """A stick spectrum.

    Attributes:
        wavenumbers: Band positions in cm⁻¹, ascending, shape (K,).
        intensities: Band heights in km/mol, shape (K,).
    """

    wavenumbers: FloatArray
    intensities: FloatArray

    def __post_init__(self) -> None:
        """Refuse positions and heights that do not pair up."""
        if self.wavenumbers.shape != self.intensities.shape or self.wavenumbers.ndim != 1:
            msg = (
                f"positions {self.wavenumbers.shape} and heights {self.intensities.shape} "
                "are not two equally long vectors"
            )
            raise ValueError(msg)


def harmonic_bands(hessian: FloatArray, geometry: Geometry, polar_tensor: FloatArray) -> Bands:
    """Return the 3N−6 harmonic bands of a Hessian with their double-harmonic intensities.

    Args:
        hessian: Cartesian Hessian in hartree/bohr², shape (3N, 3N).
        geometry: The geometry the Hessian was computed at.
        polar_tensor: Atomic polar tensor ∂μ/∂R in e, shape (3, 3N).
    """
    n_coordinates = 3 * geometry.n_atoms
    if polar_tensor.shape != (3, n_coordinates):
        msg = f"polar tensor of shape {polar_tensor.shape}, expected (3, {n_coordinates})"
        raise ValueError(msg)
    modes = vibrational_modes(hessian, geometry)
    root_mass_amu = np.sqrt(np.repeat(geometry.masses_amu, 3))[:, None]
    dipole_derivatives = polar_tensor @ (modes.vectors / root_mass_amu)
    intensities = KM_PER_MOL_PER_SQUARED_DIPOLE_DERIVATIVE * (dipole_derivatives**2).sum(axis=0)
    return Bands(modes.wavenumbers, intensities)


def scaled(bands: Bands) -> Bands:
    """Return the bands with each position multiplied by its region's scale factor.

    This is the conventional correction of B3LYP harmonic positions, shown as a baseline
    beside the learned correction and never in its place.
    """
    low, high = SCALING_REGION_EDGES
    position = bands.wavenumbers
    region = np.where(position < low, 0, np.where(position > high, 2, 1))
    return Bands(np.array(SCALING_FACTORS)[region] * position, bands.intensities)


def grid() -> FloatArray:
    """Return the wavenumbers the spectrum is drawn on."""
    return np.arange(*GRID)


def broadened(bands: Bands, wavenumbers: FloatArray, fwhm: float = FWHM) -> FloatArray:
    """Return the Lorentzian-broadened spectrum on the given wavenumbers.

    Each band becomes a Lorentzian of unit area times its intensity, so the curve is in
    km/mol per cm⁻¹.
    """
    if fwhm <= 0:
        msg = f"a line width of {fwhm} cm⁻¹ is not positive"
        raise ValueError(msg)
    half_width = 0.5 * fwhm
    offsets = wavenumbers[None, :] - bands.wavenumbers[:, None]
    lines = bands.intensities[:, None] * half_width / np.pi / (offsets**2 + half_width**2)
    return np.asarray(lines.sum(axis=0), dtype=float)
