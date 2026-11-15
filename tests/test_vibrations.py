"""Vibrations: known answers, invariances, and the refusal of a linear molecule."""

from __future__ import annotations

import numpy as np
import pytest

from aromir import vibrations
from aromir.constants import ELECTRON_MASSES_PER_AMU, WAVENUMBERS_PER_HARTREE
from aromir.molecule import FloatArray, Geometry

from support import WATER_BONDS, bond_force_field, rotation_matrix


def test_rigid_basis_is_orthonormal_and_costs_no_energy(
    water: Geometry, water_hessian: FloatArray
) -> None:
    rigid = vibrations.rigid_basis(water)
    np.testing.assert_allclose(rigid.T @ rigid, np.eye(6), atol=1e-12)
    weights = vibrations.mass_weights(water.masses_amu)
    mass_weighted = water_hessian / np.outer(weights, weights)
    np.testing.assert_allclose(mass_weighted @ rigid, 0.0, atol=1e-12)


def test_a_linear_molecule_is_refused() -> None:
    carbon_dioxide = Geometry(("O", "C", "O"), np.array([[0, 0, -2.2], [0, 0, 0], [0, 0, 2.2]]))
    with pytest.raises(ValueError, match="linear"):
        vibrations.rigid_basis(carbon_dioxide)


def test_modes_are_orthonormal_and_orthogonal_to_rigid_motions(
    water: Geometry, water_hessian: FloatArray
) -> None:
    modes = vibrations.vibrational_modes(water_hessian, water)
    assert modes.wavenumbers.shape == (3,)
    np.testing.assert_allclose(modes.vectors.T @ modes.vectors, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(vibrations.rigid_basis(water).T @ modes.vectors, 0.0, atol=1e-12)


def test_mode_reproduces_a_single_spring() -> None:
    # Two hydrogens on a spring, held off the axis by a third, uncoupled atom: the one
    # non-zero mode is the diatomic's, ω = √(k/μ) with μ = m/2.
    geometry = Geometry(("H", "H", "C"), np.array([[0, 0, 0], [1.4, 0, 0], [0, 3.0, 0]]))
    constant = 0.37
    hessian = bond_force_field(geometry, ((0, 1, constant),))
    expected = (
        np.sqrt(constant / (0.5 * geometry.masses_amu[0] * ELECTRON_MASSES_PER_AMU))
        * WAVENUMBERS_PER_HARTREE
    )
    modes = vibrations.vibrational_modes(hessian, geometry)
    assert modes.wavenumbers[-1] == pytest.approx(expected, rel=1e-12)
    np.testing.assert_allclose(modes.wavenumbers[:-1], 0.0, atol=1e-4)


def test_wavenumbers_do_not_depend_on_the_frame(water: Geometry) -> None:
    rotation = rotation_matrix(seed=3)
    moved = Geometry(water.symbols, water.coords_bohr @ rotation.T + np.array([1.0, -2.0, 0.5]))
    here = vibrations.vibrational_modes(bond_force_field(water, WATER_BONDS), water)
    there = vibrations.vibrational_modes(bond_force_field(moved, WATER_BONDS), moved)
    np.testing.assert_allclose(there.wavenumbers, here.wavenumbers, rtol=1e-10)


def test_projection_removes_a_rigid_contamination(
    water: Geometry, water_hessian: FloatArray
) -> None:
    weights = vibrations.mass_weights(water.masses_amu)
    rigid = vibrations.rigid_basis(water)
    contamination = (rigid @ rigid.T) * 1e-3 * np.outer(weights, weights)
    projected = vibrations.project_rigid_motions(water_hessian + contamination, water)
    np.testing.assert_allclose(projected, water_hessian, atol=1e-12)


def test_modes_ignore_a_rigid_contamination(water: Geometry, water_hessian: FloatArray) -> None:
    weights = vibrations.mass_weights(water.masses_amu)
    rigid = vibrations.rigid_basis(water)
    contamination = (rigid @ rigid.T) * 1e-3 * np.outer(weights, weights)
    clean = vibrations.vibrational_modes(water_hessian, water)
    dirty = vibrations.vibrational_modes(water_hessian + contamination, water)
    np.testing.assert_allclose(dirty.wavenumbers, clean.wavenumbers, rtol=1e-10)


def test_an_imaginary_mode_is_negative_and_counted(
    water: Geometry, water_hessian: FloatArray
) -> None:
    modes = vibrations.vibrational_modes(-water_hessian, water)
    assert modes.n_imaginary == 3
    assert (modes.wavenumbers < 0).all()
    assert vibrations.vibrational_modes(water_hessian, water).n_imaginary == 0
