"""A real saddle point is left: planar hydrogen peroxide, with the quantum chemistry itself."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest

from aromir import pipeline, vibrations
from aromir.dft import Resources
from aromir.molecule import Geometry

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.quantum

RESOURCES = Resources(threads=2)
PLANAR_TRANS: Geometry = Geometry.from_angstrom(
    ("O", "O", "H", "H"),
    np.array([[0.0, 0.725, 0.0], [0.0, -0.725, 0.0], [0.955, 0.893, 0.0], [-0.955, -0.893, 0.0]]),
)
"""Hydrogen peroxide with both hydrogens in one plane, trans: the top of its torsion barrier.
Nothing pulls an atom out of the plane, so the optimiser stays on the saddle point."""


def _out_of_plane(geometry: Geometry) -> float:
    """Return how far the fourth atom lies from the plane of the first three, bohr."""
    first, second, third, fourth = geometry.coords_bohr
    normal = np.cross(second - first, third - first)
    return float(abs((fourth - first) @ normal) / np.linalg.norm(normal))


def test_the_run_steps_off_a_real_saddle_point_and_finds_the_minimum(tmp_path: Path) -> None:
    state = pipeline._Run(asked={"command": "a test"})  # noqa: SLF001  the pipeline's own state
    optimised, low = pipeline._compute(PLANAR_TRANS, tmp_path, state, RESOURCES)  # noqa: SLF001
    left = state.left_saddle_point
    assert left is not None
    assert left.imaginary_cm < -50.0
    saddle = json.loads((tmp_path / "saddle_point_geometry.json").read_text(encoding="utf-8"))
    assert saddle["attempt"] == 1
    assert _out_of_plane(Geometry(tuple(saddle["symbols"]), np.array(saddle["coords_bohr"]))) < 1e-3
    assert optimised.energy < saddle["optimisation"]["energy"] - 1e-4
    assert _out_of_plane(optimised.geometry) > 0.5
    assert vibrations.vibrational_modes(low.hessian, optimised.geometry).n_imaginary == 0
