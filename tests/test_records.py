"""Result files: complete, readable back, and stamped with what made them."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np

import aromir
from aromir import records

if TYPE_CHECKING:
    from pathlib import Path


def test_json_is_written_whole_and_keeps_unicode(tmp_path: Path) -> None:
    path = tmp_path / "record.json"
    records.write_json(path, {"note": "ω in cm⁻¹", "value": 1.5})
    assert json.loads(path.read_text(encoding="utf-8")) == {"note": "ω in cm⁻¹", "value": 1.5}
    assert "cm⁻¹" in path.read_text(encoding="utf-8")
    assert [entry.name for entry in tmp_path.iterdir()] == ["record.json"]


def test_arrays_round_trip_and_leave_no_temporary_file(tmp_path: Path) -> None:
    path = tmp_path / "arrays.npz"
    hessian = np.arange(9.0).reshape(3, 3)
    records.write_arrays(path, {"hessian": hessian})
    with np.load(path) as arrays:
        np.testing.assert_array_equal(arrays["hessian"], hessian)
    assert [entry.name for entry in tmp_path.iterdir()] == ["arrays.npz"]


def test_provenance_names_the_versions_and_the_time() -> None:
    stamp = records.provenance()
    assert stamp["packages"]["aromir"] == aromir.__version__
    assert set(stamp["packages"]) == {"aromir", "pyscf", "geometric", "rdkit", "numpy", "scipy"}
    assert stamp["timestamp_utc"].endswith("+00:00")
