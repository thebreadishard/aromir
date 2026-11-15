"""Geometry: validated on construction, immutable afterwards."""

from __future__ import annotations

import numpy as np
import pytest

from aromir.constants import ANGSTROM_PER_BOHR
from aromir.molecule import Geometry


def test_unknown_element_is_refused() -> None:
    with pytest.raises(ValueError, match="unsupported element"):
        Geometry(("Xe",), np.zeros((1, 3)))


def test_wrong_shape_is_refused() -> None:
    with pytest.raises(ValueError, match="coordinates of shape"):
        Geometry(("O", "H"), np.zeros((3, 3)))


def test_coordinates_cannot_be_changed(water: Geometry) -> None:
    with pytest.raises(ValueError, match="read-only"):
        water.coords_bohr[0, 0] = 1.0


def test_coordinates_are_copied_from_the_caller() -> None:
    coords = np.zeros((1, 3))
    geometry = Geometry(("C",), coords)
    coords[0, 0] = 5.0
    assert geometry.coords_bohr[0, 0] == 0.0


def test_from_angstrom_converts_to_bohr() -> None:
    geometry = Geometry.from_angstrom(("C",), [[ANGSTROM_PER_BOHR, 0.0, 0.0]])
    assert geometry.coords_bohr[0, 0] == pytest.approx(1.0)


def test_masses_and_atomic_numbers_follow_the_symbols(water: Geometry) -> None:
    assert water.atomic_numbers.tolist() == [8, 1, 1]
    assert water.masses_amu == pytest.approx([15.99491461957, 1.00782503223, 1.00782503223])


def test_reordered_moves_symbols_and_coordinates_together(water: Geometry) -> None:
    reordered = water.reordered([2, 0, 1])
    assert reordered.symbols == ("H", "O", "H")
    np.testing.assert_array_equal(reordered.coords_bohr[1], water.coords_bohr[0])


def test_reordered_refuses_a_non_permutation(water: Geometry) -> None:
    with pytest.raises(ValueError, match="not a permutation"):
        water.reordered([0, 0, 1])


def test_to_dict_round_trips_through_plain_types(water: Geometry) -> None:
    record = water.to_dict()
    again = Geometry(tuple(record["symbols"]), np.array(record["coords_bohr"]))
    np.testing.assert_array_equal(again.coords_bohr, water.coords_bohr)
    assert record["masses_amu"] == water.masses_amu.tolist()
