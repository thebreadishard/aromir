"""Spectrum: intensities against a hand calculation, region scaling, mode partners."""

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


def test_a_stretch_between_two_point_charges_has_the_textbook_intensity() -> None:
    """Pin one intensity in km/mol by hand, without the package's constant or formula.

    A spring joins C (12 amu) and O (15.99491462 amu); H, a third atom, makes the molecule
    bent and is neither bonded nor charged. The polar tensor is that of point charges,
    +1 e on C and -1 e on O: ∂μ/∂R_C = +I, ∂μ/∂R_O = -I, ∂μ/∂R_H = 0.

    The stretch keeps the centre of mass: x_C = -a Q e, x_O = +b Q e along the bond e, with
    m_C a = m_O b and m_C a² + m_O b² = 1 (Q mass-weighted). Then a + b = 1/√μ with the
    reduced mass μ = m_C m_O / (m_C + m_O) = 12 · 15.99491462 / 27.99491462 = 6.8562086
    amu, and ∂μ/∂Q = (+1)(-a) e + (-1)(+b) e = -(a + b) e, so |∂μ/∂Q|² = 1/μ (e²/amu).

    The double-harmonic intensity is N_A / (12 ε₀ c²) |∂μ/∂Q|². In SI (CODATA 2018),
    N_A e² / (12 ε₀ c² amu) = 6.02214076e23 · (1.602176634e-19)² / (12 · 8.8541878128e-12
    · 299792458² · 1.66053906660e-27) = 974 880 m/mol = 974.88 km/mol per (e/√amu)², so
    A = 974.88 / 6.8562086 = 142.19 km/mol.
    """
    co = Geometry.from_angstrom(
        ("C", "O", "H"), [[0.0, 0.0, 0.0], [1.13, 0.0, 0.0], [-1.0, 1.5, 0.0]]
    )
    hessian = bond_force_field(co, ((0, 1, 0.5),))
    polar_tensor = np.hstack([np.eye(3), -np.eye(3), np.zeros((3, 3))])
    bands = spectrum.harmonic_bands(hessian, co, polar_tensor)
    assert bands.wavenumbers[-1] > 1000.0  # the stretch: the other two modes are free
    assert bands.intensities[-1] == pytest.approx(142.19, rel=1e-4)


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
        np.array([800.0, 1111.2, 2500.0, 3100.0]), np.array([1.0, 2.0, 3.0, 4.0])
    )
    scaled = spectrum.scaled(bands)
    np.testing.assert_allclose(
        scaled.wavenumbers, [800.0 * 0.979, 1111.2 * 0.969, 2500.0 * 0.969, 3100.0 * 0.960]
    )
    np.testing.assert_array_equal(scaled.intensities, bands.intensities)


def test_partners_follow_the_motion_not_the_position(
    water: Geometry, water_hessian: FloatArray
) -> None:
    modes = vibrations.vibrational_modes(water_hessian, water)
    swapped = vibrations.Modes(modes.wavenumbers, modes.vectors[:, [2, 0, 1]])
    assert spectrum.partners(modes, modes).tolist() == [0, 1, 2]
    assert spectrum.partners(swapped, modes).tolist() == [2, 0, 1]


def test_seed_ranges_span_the_seeds_band_by_band(
    water: Geometry, water_hessian: FloatArray
) -> None:
    stiffer, softer = 0.02 * water_hessian, -0.02 * water_hessian
    ranges = spectrum.seed_ranges(water_hessian, water, [stiffer, softer])
    plain = vibrations.vibrational_modes(water_hessian, water).wavenumbers
    np.testing.assert_allclose(ranges[:, 0], plain * np.sqrt(0.98), rtol=1e-9)
    np.testing.assert_allclose(ranges[:, 1], plain * np.sqrt(1.02), rtol=1e-9)


def test_the_scale_factors_are_those_of_the_basis_set_the_hessian_is_computed_with() -> None:
    # Bauschlicher et al. (2018), ApJS 234, 32, Table 2, row 6-31G*; the row of 4-31G
    # (0.956, 0.952, 0.960) would put a 900 cm⁻¹ band 20.7 cm⁻¹ lower.
    assert spectrum.SCALING_FACTORS == (0.979, 0.969, 0.960)
    assert pytest.approx((1111.111, 2500.0), abs=1e-3) == spectrum.SCALING_REGION_EDGES
    bands = spectrum.Bands(np.array([900.0, 1111.0, 1500.0, 3200.0]), np.ones(4))
    np.testing.assert_allclose(
        spectrum.scaled(bands).wavenumbers, [881.1, 1087.669, 1453.5, 3072.0], atol=1e-3
    )
