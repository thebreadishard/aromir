"""Spectrum: intensities against a hand calculation, region scaling, line shape."""

from __future__ import annotations

import numpy as np
import pytest

from aromir import spectrum, vibrations
from aromir.constants import KM_PER_MOL_PER_SQUARED_DIPOLE_DERIVATIVE
from aromir.molecule import FloatArray, Geometry

from support import WATER_BONDS, bond_force_field, rotation_matrix


def test_intensity_matches_the_definition(water: Geometry, water_hessian: FloatArray) -> None:
    polar_tensor = np.random.default_rng(0).normal(size=(3, 9))
    bands = spectrum.harmonic_bands(water_hessian, water, polar_tensor)
    modes = vibrations.vibrational_modes(water_hessian, water)
    first_mode = modes.vectors[:, 0] / np.sqrt(np.repeat(water.masses_amu, 3))
    expected = KM_PER_MOL_PER_SQUARED_DIPOLE_DERIVATIVE * np.sum((polar_tensor @ first_mode) ** 2)
    assert bands.intensities[0] == pytest.approx(expected, rel=1e-12)
    np.testing.assert_array_equal(bands.wavenumbers, modes.wavenumbers)


def test_a_zero_polar_tensor_gives_dark_bands(water: Geometry, water_hessian: FloatArray) -> None:
    bands = spectrum.harmonic_bands(water_hessian, water, np.zeros((3, 9)))
    np.testing.assert_array_equal(bands.intensities, 0.0)


def test_a_polar_tensor_of_the_wrong_shape_is_refused(
    water: Geometry, water_hessian: FloatArray
) -> None:
    with pytest.raises(ValueError, match="polar tensor of shape"):
        spectrum.harmonic_bands(water_hessian, water, np.zeros((3, 6)))


def test_intensities_do_not_depend_on_the_frame(water: Geometry) -> None:
    rotation = rotation_matrix(seed=5)
    polar_tensor = np.random.default_rng(1).normal(size=(3, 3, 3))  # (dipole, atom, coordinate)
    rotated_tensor = np.einsum("ts,sak,jk->taj", rotation, polar_tensor, rotation)
    moved = Geometry(water.symbols, water.coords_bohr @ rotation.T)
    here = spectrum.harmonic_bands(
        bond_force_field(water, WATER_BONDS), water, polar_tensor.reshape(3, 9)
    )
    there = spectrum.harmonic_bands(
        bond_force_field(moved, WATER_BONDS), moved, rotated_tensor.reshape(3, 9)
    )
    np.testing.assert_allclose(there.intensities, here.intensities, rtol=1e-9)


def test_bands_refuse_unpaired_positions_and_heights() -> None:
    with pytest.raises(ValueError, match="equally long"):
        spectrum.Bands(np.zeros(3), np.zeros(2))


def test_each_region_gets_its_factor_and_heights_stay() -> None:
    bands = spectrum.Bands(
        np.array([800.0, 1111.0, 2500.0, 3100.0]), np.array([1.0, 2.0, 3.0, 4.0])
    )
    scaled = spectrum.scaled(bands)
    np.testing.assert_allclose(
        scaled.wavenumbers, [800.0 * 0.956, 1111.0 * 0.952, 2500.0 * 0.952, 3100.0 * 0.960]
    )
    np.testing.assert_array_equal(scaled.intensities, bands.intensities)


def test_the_drawn_grid_is_the_registered_one() -> None:
    wavenumbers = spectrum.grid()
    assert wavenumbers[0] == 500.0
    assert wavenumbers[-1] == 3499.0
    assert len(wavenumbers) == 3000


def test_a_broadened_band_keeps_its_area_and_width() -> None:
    bands = spectrum.Bands(np.array([1000.0]), np.array([7.0]))
    wavenumbers = np.arange(0.0, 2000.0, 0.01)
    curve = spectrum.broadened(bands, wavenumbers, fwhm=10.0)
    # A Lorentzian's tails beyond ±1000 cm⁻¹ hold 2/π · atan(γ/1000) of its area: 0.32 %.
    assert np.sum(curve) * 0.01 == pytest.approx(7.0, rel=4e-3)
    peak = float(np.max(curve))
    assert peak == pytest.approx(7.0 / (np.pi * 5.0), rel=1e-9)
    above_half = wavenumbers[curve >= 0.5 * peak]
    assert above_half[-1] - above_half[0] == pytest.approx(10.0, abs=0.02)


def test_broadening_adds_bands() -> None:
    wavenumbers = spectrum.grid()
    first = spectrum.Bands(np.array([900.0]), np.array([2.0]))
    second = spectrum.Bands(np.array([1500.0]), np.array([5.0]))
    both = spectrum.Bands(np.array([900.0, 1500.0]), np.array([2.0, 5.0]))
    np.testing.assert_allclose(
        spectrum.broadened(both, wavenumbers),
        spectrum.broadened(first, wavenumbers) + spectrum.broadened(second, wavenumbers),
    )


def test_a_line_width_must_be_positive() -> None:
    bands = spectrum.Bands(np.array([900.0]), np.array([2.0]))
    with pytest.raises(ValueError, match="not positive"):
        spectrum.broadened(bands, spectrum.grid(), fwhm=0.0)
