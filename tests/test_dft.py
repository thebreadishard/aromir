"""The SCF on water: the second-order finish, and the error when nothing converges."""

from __future__ import annotations

import io
from typing import TYPE_CHECKING, Any

import pytest
from pyscf import dft as pyscf_dft

from aromir import dft
from aromir.errors import ConvergenceError

if TYPE_CHECKING:
    from aromir.molecule import Geometry

pytestmark = pytest.mark.quantum

RESOURCES = dft.Resources(threads=2)


def starve_diis(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give the plain SCF three cycles, which is not enough to converge water."""
    plain_rks = pyscf_dft.RKS

    def three_cycles(molecule: Any) -> Any:  # noqa: ANN401  pyscf objects carry no types
        scf = plain_rks(molecule)
        scf.max_cycle = 3
        return scf

    monkeypatch.setattr(pyscf_dft, "RKS", three_cycles)


def test_resources_are_validated() -> None:
    with pytest.raises(ValueError, match="greater than or equal to 500"):
        dft.Resources(max_memory_mb=10)
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        dft.Resources.model_validate({"cores": 4})


def test_a_stalled_scf_is_finished_by_the_second_order_solver(
    water: Geometry, monkeypatch: pytest.MonkeyPatch
) -> None:
    plain = dft.converged_scf(dft.build_molecule(water, RESOURCES, io.StringIO()))
    starve_diis(monkeypatch)
    finished = dft.converged_scf(dft.build_molecule(water, RESOURCES, io.StringIO()))
    assert finished.converged
    assert finished.e_tot == pytest.approx(plain.e_tot, abs=1e-9)


def test_an_scf_that_never_converges_is_an_error(
    water: Geometry, monkeypatch: pytest.MonkeyPatch
) -> None:
    starve_diis(monkeypatch)
    monkeypatch.setattr(dft, "SECOND_ORDER_MACRO_ITERATIONS", 1)
    with pytest.raises(ConvergenceError, match="did not converge"):
        dft.converged_scf(dft.build_molecule(water, RESOURCES, io.StringIO()))
