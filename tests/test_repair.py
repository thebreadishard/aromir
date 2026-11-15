"""The soft-mode repair: its arithmetic exactly, its measurement where the answer is known."""

from __future__ import annotations

import io
from typing import TYPE_CHECKING

import numpy as np
import pytest

from aromir import lowlevel, repair, vibrations
from aromir.dft import Resources

if TYPE_CHECKING:
    from aromir.molecule import FloatArray, Geometry

RESOURCES = Resources(threads=2)
STIFF_BELOW_CM = 2000.0
"""Makes water's bend (1679 cm⁻¹) count as soft, so the repair measures a mode whose
analytic curvature is right."""


def test_exact_columns_leave_the_hessian_unchanged(
    water: Geometry, water_hessian: FloatArray
) -> None:
    modes = vibrations.vibrational_modes(water_hessian, water)
    directions = modes.vectors[:, :2]
    weights = vibrations.mass_weights(water.masses_amu)
    exact = (water_hessian / np.outer(weights, weights)) @ directions
    repaired = vibrations.with_measured_columns(water_hessian, water.masses_amu, directions, exact)
    np.testing.assert_allclose(repaired, water_hessian, atol=1e-14)


def test_a_measured_curvature_replaces_the_modes_own(
    water: Geometry, water_hessian: FloatArray
) -> None:
    modes = vibrations.vibrational_modes(water_hessian, water)
    softest = modes.vectors[:, :1]
    weights = vibrations.mass_weights(water.masses_amu)
    stiffer = 1.21 * (water_hessian / np.outer(weights, weights)) @ softest
    repaired = vibrations.with_measured_columns(water_hessian, water.masses_amu, softest, stiffer)
    after = vibrations.vibrational_modes(repaired, water).wavenumbers
    np.testing.assert_allclose(after, modes.wavenumbers * [1.1, 1.0, 1.0], rtol=1e-10)


def test_columns_must_fit_orthonormal_directions(
    water: Geometry, water_hessian: FloatArray
) -> None:
    directions = vibrations.vibrational_modes(water_hessian, water).vectors[:, :2]
    with pytest.raises(ValueError, match="differ in shape"):
        vibrations.with_measured_columns(
            water_hessian, water.masses_amu, directions, directions[:, :1]
        )
    with pytest.raises(ValueError, match="not orthonormal"):
        vibrations.with_measured_columns(
            water_hessian, water.masses_amu, 2 * directions, directions
        )


def test_a_step_moves_the_fastest_atom_by_the_step_length(water: Geometry) -> None:
    direction = np.random.default_rng(0).normal(size=9)
    step, length = vibrations.displacement(direction, water.masses_amu, 0.01)
    assert float(np.max(np.linalg.norm(step, axis=1))) == pytest.approx(0.01)
    weights = vibrations.mass_weights(water.masses_amu)
    np.testing.assert_allclose(step.ravel() * weights, length * direction)


def test_a_column_is_the_central_difference_of_the_gradients(
    water: Geometry, water_hessian: FloatArray
) -> None:
    direction = vibrations.vibrational_modes(water_hessian, water).vectors[:, 0]
    step, length = vibrations.displacement(direction, water.masses_amu, 0.01)
    gradient = water_hessian @ step.ravel()  # a harmonic surface: g(x) = H x
    measured = vibrations.column(gradient, -gradient, length, water.masses_amu)
    weights = vibrations.mass_weights(water.masses_amu)
    np.testing.assert_allclose(
        measured, (water_hessian / np.outer(weights, weights)) @ direction, atol=1e-16
    )


@pytest.fixture(scope="module")
def analytic(request: pytest.FixtureRequest) -> tuple[Geometry, lowlevel.LowLevel]:
    water: Geometry = request.getfixturevalue("water_for_module")
    return water, lowlevel.compute(water, RESOURCES, io.StringIO(), repair_soft_modes=False)


@pytest.mark.quantum
def test_the_measurement_agrees_with_the_analytic_hessian_on_a_stiff_mode(
    analytic: tuple[Geometry, lowlevel.LowLevel],
) -> None:
    water, low = analytic
    repaired, record = repair.repair(
        water, low.hessian, RESOURCES, io.StringIO(), soft_below_cm=STIFF_BELOW_CM
    )
    assert record is not None
    assert record.n_directions == 1
    assert record.largest_change_cm < 0.05
    before = vibrations.vibrational_modes(low.hessian, water).wavenumbers
    after = vibrations.vibrational_modes(repaired, water).wavenumbers
    np.testing.assert_allclose(after, before, atol=0.05)
    assert np.abs(repaired - low.hessian).max() < 1e-4 * np.abs(low.hessian).max()


@pytest.mark.quantum
def test_a_molecule_without_a_soft_mode_is_left_alone(
    analytic: tuple[Geometry, lowlevel.LowLevel],
) -> None:
    water, low = analytic
    repaired, record = repair.repair(water, low.hessian, RESOURCES, io.StringIO())
    assert record is None
    assert repaired is low.hessian


@pytest.mark.quantum
def test_a_gradient_that_would_not_fit_is_refused(
    analytic: tuple[Geometry, lowlevel.LowLevel], monkeypatch: pytest.MonkeyPatch
) -> None:
    water, low = analytic
    monkeypatch.setattr(repair, "GRID", (9900, 5900))  # a grid water's 4 GB cannot hold
    with pytest.raises(repair.GridResponseMemoryError, match="raise --max-memory-mb"):
        repair.repair(
            water,
            low.hessian,
            RESOURCES,
            io.StringIO(),
            soft_below_cm=STIFF_BELOW_CM,
        )
