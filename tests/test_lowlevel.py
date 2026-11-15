"""The low level on water: each quantity against an independent counterpart.

Water is not in the package's domain, but the quantum chemistry does not know that, and
it runs in seconds.
"""

from __future__ import annotations

import io

import numpy as np
import pytest

from aromir import lowlevel, response
from aromir.dft import Resources, build_molecule, converged_scf
from aromir.errors import ResponseError
from aromir.molecule import FloatArray, Geometry

pytestmark = pytest.mark.quantum

RESOURCES = Resources(threads=2)
STEP_BOHR = 1e-3


@pytest.fixture(scope="module")
def water_geometry() -> Geometry:
    return Geometry(
        ("O", "H", "H"),
        np.array([[0.0, 0.0, 0.2217], [0.0, 1.4309, -0.8867], [0.0, -1.4309, -0.8867]]),
    )


@pytest.fixture(scope="module")
def low(water_geometry: Geometry) -> lowlevel.LowLevel:
    return lowlevel.compute(water_geometry, RESOURCES, io.StringIO(), repair_soft_modes=True)


def _dipole(geometry: Geometry) -> FloatArray:
    scf = converged_scf(build_molecule(geometry, RESOURCES, io.StringIO()))
    return np.asarray(scf.dip_moment(unit="AU", verbose=0), dtype=float)


def test_the_hessian_is_pyscfs_own(water_geometry: Geometry, low: lowlevel.LowLevel) -> None:
    scf = converged_scf(build_molecule(water_geometry, RESOURCES, io.StringIO()))
    blocks = scf.Hessian().kernel()
    reference = blocks.transpose(0, 2, 1, 3).reshape(9, 9)
    np.testing.assert_allclose(low.hessian, 0.5 * (reference + reference.T), atol=1e-9)


def test_the_polar_tensor_matches_finite_differences_of_the_dipole(
    water_geometry: Geometry, low: lowlevel.LowLevel
) -> None:
    finite_difference = np.zeros((3, 9))
    for coordinate in range(9):
        step = np.zeros(9)
        step[coordinate] = STEP_BOHR
        plus = Geometry(water_geometry.symbols, water_geometry.coords_bohr + step.reshape(3, 3))
        minus = Geometry(water_geometry.symbols, water_geometry.coords_bohr - step.reshape(3, 3))
        finite_difference[:, coordinate] = (_dipole(plus) - _dipole(minus)) / (2 * STEP_BOHR)
    np.testing.assert_allclose(low.polar_tensor, finite_difference, atol=5e-5)


def test_the_checks_pass_and_are_recorded(low: lowlevel.LowLevel) -> None:
    assert low.refinement.residual_after <= response.RESIDUAL_BOUND
    assert low.sum_rule_violation <= response.SUM_RULE_LIMIT
    assert low.energy == pytest.approx(-76.40, abs=0.02)
    assert low.seconds > 0


def test_a_spoiled_response_is_repaired(water_geometry: Geometry, low: lowlevel.LowLevel) -> None:
    scf = converged_scf(build_molecule(water_geometry, RESOURCES, io.StringIO()))
    hessian_method = scf.Hessian()
    solved = response.solve(hessian_method)
    noise = np.random.default_rng(0)
    spoiled = solved._replace(
        mo1=[block + 1e-3 * noise.normal(size=block.shape) for block in solved.mo1]
    )
    refined, record = response.refine(hessian_method, spoiled)
    assert record.residual_before > 1e-4
    assert record.residual_after <= response.RESIDUAL_BOUND
    assert record.batched_rounds >= 1
    np.testing.assert_allclose(response.polar_tensor(scf, refined), low.polar_tensor, atol=1e-5)


def test_single_column_solves_take_over_when_batched_rounds_are_not_enough(
    water_geometry: Geometry, low: lowlevel.LowLevel, monkeypatch: pytest.MonkeyPatch
) -> None:
    scf = converged_scf(build_molecule(water_geometry, RESOURCES, io.StringIO()))
    hessian_method = scf.Hessian()
    solved = response.solve(hessian_method)
    noise = np.random.default_rng(1)
    spoiled = solved._replace(
        mo1=[block + 1e-3 * noise.normal(size=block.shape) for block in solved.mo1]
    )
    monkeypatch.setattr(response, "_BATCHED_ROUNDS", 0)
    refined, record = response.refine(hessian_method, spoiled)
    assert record.batched_rounds == 0
    assert record.single_columns == 9
    assert record.residual_after <= response.RESIDUAL_BOUND
    np.testing.assert_allclose(response.polar_tensor(scf, refined), low.polar_tensor, atol=1e-5)


def test_a_failed_check_names_itself() -> None:
    passed = response.Refinement(1e-9, 1e-9, 0, 0)
    stuck = response.Refinement(1e-4, 2e-6, 3, 4)
    assert lowlevel._failed_checks(passed, 1e-6) == []  # noqa: SLF001  the checks' own unit
    failures = lowlevel._failed_checks(stuck, 1e-3)  # noqa: SLF001
    assert len(failures) == 2
    assert "response residual 2.0e-06" in failures[0]
    assert "sum rule violated by 1.0e-03" in failures[1]


def test_a_failed_check_refuses_the_result(
    water_geometry: Geometry, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(response, "SUM_RULE_LIMIT", 0.0)
    with pytest.raises(ResponseError, match="sum rule"):
        lowlevel.compute(water_geometry, RESOURCES, io.StringIO(), repair_soft_modes=False)
