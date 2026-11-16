"""The optimisation on water: it reaches a minimum, and says so when it does not."""

from __future__ import annotations

import io
import logging

import numpy as np
import pytest

from aromir import optimisation, vibrations
from aromir.dft import Resources
from aromir.errors import ConvergenceError
from aromir.lowlevel import compute
from aromir.molecule import Geometry

pytestmark = pytest.mark.quantum

RESOURCES = Resources(threads=2)


@pytest.fixture(scope="module")
def distorted_water() -> Geometry:
    return Geometry(("O", "H", "H"), np.array([[0, 0, 0.3], [0.1, 1.6, -0.9], [0, -1.3, -1.0]]))


def test_the_optimised_geometry_is_a_minimum(
    distorted_water: Geometry, capsys: pytest.CaptureFixture[str]
) -> None:
    log = io.StringIO()
    handlers_before = logging.getLogger().handlers[:]
    result = optimisation.optimise(distorted_water, RESOURCES, log)
    assert "Converged!" in log.getvalue()
    assert capsys.readouterr().err == ""
    assert logging.getLogger().handlers == handlers_before
    assert result.geometry.symbols == distorted_water.symbols
    assert result.max_force < 5e-5
    assert result.n_steps >= 2
    low = compute(result.geometry, RESOURCES, io.StringIO())
    assert low.energy == pytest.approx(result.energy, abs=1e-9)
    modes = vibrations.vibrational_modes(low.hessian, result.geometry)
    assert modes.n_imaginary == 0
    np.testing.assert_allclose(modes.wavenumbers, [1713, 3727, 3849], atol=15)


def test_running_out_of_steps_is_an_error(
    distorted_water: Geometry, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(optimisation, "MAX_STEPS", 1)
    monkeypatch.setattr(optimisation, "FLAT_STEPS", 1)
    with pytest.raises(ConvergenceError, match="did not converge in 1 steps, nor in 1 more"):
        optimisation.optimise(distorted_water, RESOURCES, io.StringIO())


def test_geometric_takes_its_q_chem_style_test_through_pyscf(
    distorted_water: Geometry, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Two gradients are not enough under any test; the stage that follows runs under
    # geomeTRIC's Q-Chem-style test, and geomeTRIC's own report says so.
    monkeypatch.setattr(optimisation, "MAX_STEPS", 2)
    log = io.StringIO()
    kept: list[optimisation.Progress] = []
    result = optimisation.optimise(distorted_water, RESOURCES, log, keep=kept.append)
    assert "Q-Chem style convergence criteria requested" in log.getvalue()
    said_flat = optimisation.Q_CHEM_VERDICT in log.getvalue()
    assert (result.criteria == optimisation.ENERGY_AND_FORCES) == said_flat
    assert (result.last_step is not None) == said_flat
    assert result.n_steps == len(kept) == kept[-1].n_steps
    assert result.max_force < 5e-5
    np.testing.assert_allclose(kept[-1].coords_bohr, result.geometry.coords_bohr, atol=1e-10)
