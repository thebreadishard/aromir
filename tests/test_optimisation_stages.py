"""The optimisation's two stages and its saved progress, with geomeTRIC replaced by a stand-in.

The stand-in takes its verdicts from a list, walks a fixed number of gradients from where it
is started and reports each one the way pyscf's solver does: these tests are of what aromir
does around geomeTRIC, not of geomeTRIC.
"""

from __future__ import annotations

import io
import math
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from pyscf.geomopt import geometric_solver

from aromir import optimisation
from aromir.constants import ANGSTROM_PER_BOHR
from aromir.dft import Resources
from aromir.errors import ConvergenceError
from aromir.molecule import Geometry

if TYPE_CHECKING:
    from collections.abc import Callable

    from aromir.molecule import FloatArray

GRADIENTS = 4
STEP_BOHR = 0.001
FORCE = 1e-6
WATER = Geometry(("O", "H", "H"), np.array([[0, 0, 0.2], [0, 1.4, -0.9], [0, -1.4, -0.9]]))


class _Scf:
    """What aromir reads of a converged SCF: its energy and its forces."""

    def __init__(self, geometry: Geometry) -> None:
        self.geometry = geometry
        self.e_tot = -76.0

    def nuc_grad_method(self) -> _Scf:
        return self

    def kernel(self) -> FloatArray:
        return np.full_like(self.geometry.coords_bohr, 2 * FORCE)


class _Stopped:
    def __init__(self, coords: FloatArray) -> None:
        self._coords = coords

    def atom_coords(self) -> FloatArray:
        return self._coords


class _Solver:
    """geomeTRIC's stand-in: its verdicts to come and the calls it got."""

    def __init__(self) -> None:
        self.verdicts: list[bool] = []
        self.calls: list[dict[str, Any]] = []

    def kernel(
        self,
        scf: _Scf,
        *,
        assert_convergence: bool,
        callback: Callable[[dict[str, Any]], None],
        convergence_set: str,
        maxsteps: int,
        **settings: object,
    ) -> tuple[bool, _Stopped]:
        assert assert_convergence
        coords = scf.geometry.coords_bohr.copy()
        self.calls.append(
            {"start": coords.copy(), "set": convergence_set, "maxsteps": maxsteps, **settings}
        )
        for gradient in range(GRADIENTS):
            coords = coords + STEP_BOHR
            energy = -76.0 - 1e-9 * gradient
            callback({"coords": coords, "energy": energy, "gradients": np.full_like(coords, FORCE)})
        return self.verdicts.pop(0), _Stopped(coords)


@pytest.fixture
def solver(monkeypatch: pytest.MonkeyPatch) -> _Solver:
    stand_in = _Solver()
    monkeypatch.setattr(geometric_solver, "kernel", stand_in.kernel)
    monkeypatch.setattr(optimisation, "build_molecule", lambda geometry, _resources, _log: geometry)
    monkeypatch.setattr(optimisation, "converged_scf", _Scf)
    return stand_in


def _optimise(**progress: Any) -> optimisation.Optimisation:  # noqa: ANN401  the keywords under test
    return optimisation.optimise(WATER, Resources(), io.StringIO(), **progress)


def test_an_optimisation_that_converges_is_asked_for_every_criterion(solver: _Solver) -> None:
    solver.verdicts = [True]
    result = _optimise()
    assert len(solver.calls) == 1
    assert solver.calls[0]["set"] == "GAU_TIGHT"
    assert solver.calls[0]["maxsteps"] == optimisation.MAX_STEPS
    assert "convergence_drms" not in solver.calls[0]
    assert "convergence_dmax" not in solver.calls[0]
    assert result.criteria == optimisation.ALL_CRITERIA
    assert result.last_step is None
    assert result.n_steps == GRADIENTS
    np.testing.assert_allclose(
        result.geometry.coords_bohr, WATER.coords_bohr + GRADIENTS * STEP_BOHR
    )
    assert result.max_force == pytest.approx(2 * FORCE)


def test_one_that_does_not_goes_on_under_geometrics_own_test_of_energy_and_forces(
    solver: _Solver,
) -> None:
    solver.verdicts = [False, True]
    log = io.StringIO()
    result = optimisation.optimise(WATER, Resources(), log)
    first, second = solver.calls
    assert (second["convergence_drms"], second["convergence_dmax"]) == (math.inf, math.inf)
    assert "qccnv" not in second
    assert second["set"] == "GAU_TIGHT"
    assert second["maxsteps"] == optimisation.FLAT_STEPS
    np.testing.assert_allclose(second["start"], first["start"] + GRADIENTS * STEP_BOHR)
    assert result.criteria == optimisation.ENERGY_AND_FORCES
    assert result.n_steps == 2 * GRADIENTS
    assert "going on without the displacement criterion" in log.getvalue()
    last = result.last_step
    assert last is not None
    assert last["energy_change_hartree"] == pytest.approx(-1e-9)
    assert last["max_displacement_angstrom"] == pytest.approx(
        np.sqrt(3) * STEP_BOHR * ANGSTROM_PER_BOHR
    )
    assert last["rms_force_hartree_per_bohr"] == pytest.approx(np.sqrt(3) * FORCE)


def test_one_that_converges_on_neither_is_an_error(solver: _Solver) -> None:
    solver.verdicts = [False, False]
    with pytest.raises(ConvergenceError, match="in 100 steps, nor in 20 more on energy and forces"):
        _optimise()


def test_every_gradient_is_kept(solver: _Solver) -> None:
    solver.verdicts = [True]
    kept: list[optimisation.Progress] = []
    _optimise(keep=kept.append)
    assert [progress.n_steps for progress in kept] == [1, 2, 3, 4]
    np.testing.assert_allclose(kept[-1].coords_bohr, WATER.coords_bohr + GRADIENTS * STEP_BOHR)


def test_a_saved_progress_is_where_the_optimisation_goes_on_with_the_steps_it_has_left(
    solver: _Solver,
) -> None:
    solver.verdicts = [True]
    reached = optimisation.Progress((WATER.coords_bohr + 0.5).tolist(), 40)
    result = _optimise(taken_up=reached)
    assert solver.calls[0]["maxsteps"] == optimisation.MAX_STEPS - 40
    np.testing.assert_allclose(solver.calls[0]["start"], WATER.coords_bohr + 0.5)
    assert result.n_steps == 40 + GRADIENTS


def test_a_progress_that_has_used_its_steps_goes_on_under_energy_and_forces(
    solver: _Solver,
) -> None:
    solver.verdicts = [True]
    reached = optimisation.Progress(WATER.coords_bohr.tolist(), optimisation.MAX_STEPS)
    result = _optimise(taken_up=reached)
    assert [call.get("convergence_dmax") for call in solver.calls] == [math.inf]
    assert result.criteria == optimisation.ENERGY_AND_FORCES
