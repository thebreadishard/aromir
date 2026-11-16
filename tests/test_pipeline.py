"""The pipeline's own logic, with the quantum chemistry replaced by a synthetic result."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import replace
from typing import TYPE_CHECKING, TextIO

import numpy as np
import pytest
import torch
from pydantic import ValidationError

from aromir import (
    certificate,
    lowlevel,
    optimisation,
    pipeline,
    release,
    repair,
    response,
    spectrum,
    vibrations,
)
from aromir.dft import Resources
from aromir.errors import InternalCoordinatesError, NotAMinimumError, RefusedError

from support import BENZENE_XYZ, bond_force_field, write_release

if TYPE_CHECKING:
    from pathlib import Path

    from aromir.molecule import FloatArray, Geometry


def _ring_force_field(geometry: Geometry) -> tuple[tuple[int, int, float], ...]:
    """Springs between all atom pairs: a stable synthetic Hessian for any geometry."""
    n = geometry.n_atoms
    return tuple((i, j, 0.3) for i in range(n) for j in range(i + 1, n))


def _synthetic_low_level(geometry: Geometry) -> lowlevel.LowLevel:
    return lowlevel.LowLevel(
        energy=-1.0,
        hessian=bond_force_field(geometry, _ring_force_field(geometry)),
        polar_tensor=np.random.default_rng(0).normal(size=(3, 3 * geometry.n_atoms)),
        refinement=response.Refinement(1e-9, 1e-9, 0, 0),
        sum_rule_violation=1e-7,
        seconds=0.1,
    )


@pytest.fixture
def synthetic_chemistry(monkeypatch: pytest.MonkeyPatch) -> Counter[str]:
    """Replace the optimisation, the low level and the repair by instant, synthetic results.

    Returns the count of calls of each of the three steps.
    """
    calls: Counter[str] = Counter()

    def optimise(start: Geometry, _resources: Resources, log: TextIO) -> optimisation.Optimisation:
        calls["optimise"] += 1
        log.write("optimised\n")
        return optimisation.Optimisation(start, energy=-1.0, max_force=1e-6, n_steps=3)

    def compute(
        geometry: Geometry, _resources: Resources, _log: TextIO, *, repair_soft_modes: bool
    ) -> lowlevel.LowLevel:
        assert not repair_soft_modes, "the pipeline saves the analytic Hessian first"
        calls["low level"] += 1
        return _synthetic_low_level(geometry)

    def measure(
        _geometry: Geometry, hessian: FloatArray, _resources: Resources, _log: TextIO
    ) -> tuple[FloatArray, repair.Repair | None]:
        calls["repair"] += 1
        return hessian, None

    monkeypatch.setattr(optimisation, "optimise", optimise)
    monkeypatch.setattr(lowlevel, "compute", compute)
    monkeypatch.setattr(repair, "repair", measure)
    return calls


def _benzene(out: Path) -> pipeline.Request:
    return pipeline.Request(molecule=pipeline.MoleculeInput(smiles="c1ccccc1"), out=out)


def test_a_molecule_is_a_smiles_or_an_xyz_never_both_or_neither(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="exactly one"):
        pipeline.MoleculeInput()
    with pytest.raises(ValidationError, match="exactly one"):
        pipeline.MoleculeInput(smiles="c1ccccc1", xyz=tmp_path / "x.xyz")


def test_an_unknown_setting_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        pipeline.Request.model_validate(
            {"molecule": {"smiles": "c1ccccc1"}, "out": tmp_path, "functional": "pbe"}
        )


def test_a_missing_xyz_file_is_refused(tmp_path: Path) -> None:
    with pytest.raises(RefusedError, match="no such xyz file"):
        pipeline.admit(pipeline.MoleculeInput(xyz=tmp_path / "absent.xyz"))


def test_an_xyz_file_is_admitted_with_its_own_geometry(tmp_path: Path) -> None:
    path = tmp_path / "benzene.xyz"
    path.write_text(BENZENE_XYZ, encoding="utf-8")
    admitted, start = pipeline.admit(pipeline.MoleculeInput(xyz=path))
    assert admitted.formula == "C6H6"
    assert start.symbols == ("C",) * 6 + ("H",) * 6


@pytest.mark.usefixtures("synthetic_chemistry")
def test_a_run_writes_its_result_folder(tmp_path: Path) -> None:
    record = pipeline.run_baseline(_benzene(tmp_path / "benzene"))
    folder = tmp_path / "benzene"
    assert sorted(path.name for path in folder.iterdir()) == [
        "baseline.json",
        "geometry.json",
        "lowlevel.npz",
        "pyscf.log",
        "run.json",
    ]
    assert json.loads((folder / "baseline.json").read_text(encoding="utf-8")) == record
    assert record["molecule"] == {"smiles": "c1ccccc1", "formula": "C6H6", "n_atoms": 12}
    assert record["provenance"]["packages"]["aromir"]
    with np.load(folder / "lowlevel.npz") as arrays:
        assert arrays["hessian"].shape == (36, 36)
        assert arrays["polar_tensor"].shape == (3, 36)


@pytest.mark.usefixtures("synthetic_chemistry")
def test_the_record_lists_every_band_harmonic_and_scaled(tmp_path: Path) -> None:
    record = pipeline.run_baseline(_benzene(tmp_path / "benzene"))
    harmonic, scaled = record["harmonic"]["bands"], record["scaled"]["bands"]
    assert len(harmonic) == len(scaled) == 3 * 12 - 6
    positions = [band["wavenumber_cm"] for band in harmonic]
    assert positions == sorted(positions)
    heights = np.array([band["intensity_km_mol"] for band in harmonic])
    expected = spectrum.scaled(spectrum.Bands(np.array(positions), heights))
    np.testing.assert_allclose(
        [band["wavenumber_cm"] for band in scaled], expected.wavenumbers, atol=0.011
    )
    assert [band["intensity_km_mol"] for band in scaled] == heights.tolist()


@pytest.mark.usefixtures("synthetic_chemistry")
def test_a_result_is_never_overwritten(tmp_path: Path) -> None:
    pipeline.run_baseline(_benzene(tmp_path / "benzene"))
    with pytest.raises(RefusedError, match="never overwritten"):
        pipeline.run_baseline(_benzene(tmp_path / "benzene"))


def test_a_refused_molecule_leaves_no_folder(tmp_path: Path) -> None:
    request = pipeline.Request(molecule=pipeline.MoleculeInput(smiles="CCO"), out=tmp_path / "x")
    with pytest.raises(RefusedError, match="no aromatic ring"):
        pipeline.run_baseline(request)
    assert not (tmp_path / "x").exists()


def _unstable_for(n_calls: int, monkeypatch: pytest.MonkeyPatch) -> list[Geometry]:
    """Make the low level's first ``n_calls`` Hessians all-imaginary; return where it was asked."""
    stable = lowlevel.compute
    asked_at: list[Geometry] = []

    def compute(
        geometry: Geometry, resources: Resources, log: TextIO, *, repair_soft_modes: bool
    ) -> lowlevel.LowLevel:
        asked_at.append(geometry)
        result = stable(geometry, resources, log, repair_soft_modes=repair_soft_modes)
        if len(asked_at) > n_calls:
            return result
        return replace(result, hessian=-result.hessian)

    monkeypatch.setattr(lowlevel, "compute", compute)
    return asked_at


def test_a_saddle_point_that_stays_gets_no_spectrum_and_keeps_its_hessian(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, synthetic_chemistry: Counter[str]
) -> None:
    _unstable_for(2, monkeypatch)
    folder = tmp_path / "benzene"
    with pytest.raises(NotAMinimumError, match="also after one step along it"):
        pipeline.run_baseline(_benzene(folder))
    assert synthetic_chemistry == {"optimise": 2, "low level": 2, "repair": 2}
    assert sorted(path.name for path in folder.iterdir()) == [
        "geometry.json",
        "lowlevel.npz",
        "pyscf.log",
        "run.json",
        "saddle_point_geometry.json",
        "saddle_point_lowlevel.npz",
    ]
    with np.load(folder / "lowlevel.npz") as arrays:
        assert arrays["hessian"].shape == (36, 36)


def test_a_saddle_point_is_left_along_its_mode_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, synthetic_chemistry: Counter[str]
) -> None:
    asked_at = _unstable_for(1, monkeypatch)
    record = pipeline.run_baseline(_benzene(tmp_path / "benzene"))
    assert synthetic_chemistry == {"optimise": 2, "low level": 2, "repair": 2}
    assert record["optimisation"]["left_saddle_point_at_cm"] < 0
    moved = np.linalg.norm(asked_at[1].coords_bohr - asked_at[0].coords_bohr, axis=1)
    assert float(np.max(moved)) == pytest.approx(pipeline.OFF_THE_SADDLE_BOHR)
    assert (
        vibrations.vibrational_modes(
            _synthetic_low_level(asked_at[1]).hessian, asked_at[1]
        ).n_imaginary
        == 0
    )


def test_a_run_that_failed_is_taken_up_where_it_stopped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, synthetic_chemistry: Counter[str]
) -> None:
    working = repair.repair

    def out_of_memory(*_arguments: object) -> None:
        msg = "the grid-response gradient needs more memory"
        raise repair.GridResponseMemoryError(msg)

    monkeypatch.setattr(repair, "repair", out_of_memory)
    folder = tmp_path / "benzene"
    with pytest.raises(repair.GridResponseMemoryError):
        pipeline.run_baseline(_benzene(folder))
    assert sorted(path.name for path in folder.iterdir()) == [
        "geometry.json",
        "lowlevel.npz",
        "pyscf.log",
        "run.json",
    ]
    monkeypatch.setattr(repair, "repair", working)
    record = pipeline.run_baseline(_benzene(folder))
    assert synthetic_chemistry == {"optimise": 1, "low level": 1, "repair": 1}
    assert (folder / "baseline.json").is_file()
    assert record["optimisation"]["n_steps"] == 3
    assert record["checks"]["sum_rule_violation_e"] == 1e-7
    fresh = pipeline.run_baseline(_benzene(tmp_path / "fresh"))
    assert record["harmonic"] == fresh["harmonic"]


def test_an_unfinished_run_is_not_taken_up_by_another_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, synthetic_chemistry: Counter[str]
) -> None:
    def stop(*_arguments: object) -> None:
        raise repair.GridResponseMemoryError

    monkeypatch.setattr(repair, "repair", stop)
    folder = tmp_path / "benzene"
    with pytest.raises(repair.GridResponseMemoryError):
        pipeline.run_baseline(_benzene(folder))
    other = pipeline.Request(molecule=pipeline.MoleculeInput(smiles="c1ccncc1"), out=folder)
    with pytest.raises(
        RefusedError, match=r"unfinished run of another request \(baseline c1ccccc1\)"
    ):
        pipeline.run_baseline(other)
    loaded = release.load(write_release(tmp_path / "release"))
    with pytest.raises(RefusedError, match="unfinished run of another request"):
        pipeline.run_spectrum(_benzene(folder), loaded)
    assert synthetic_chemistry["optimise"] == 1


def test_a_folder_with_other_files_is_not_a_result_folder(tmp_path: Path) -> None:
    folder = tmp_path / "notes"
    folder.mkdir()
    (folder / "todo.txt").write_text("mine", encoding="utf-8")
    with pytest.raises(RefusedError, match="is not empty: a result is never overwritten"):
        pipeline.run_baseline(_benzene(folder))


def test_what_the_network_cannot_read_is_refused_before_the_quantum_chemistry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, synthetic_chemistry: Counter[str]
) -> None:
    loaded = release.load(write_release(tmp_path / "release"))

    def unreadable(_self: release.Release, _geometry: Geometry, _hessian: FloatArray) -> None:
        msg = "the primitives span more than 3N − 6 = 30 directions"
        raise InternalCoordinatesError(msg)

    monkeypatch.setattr(release.Release, "corrections", unreadable)
    with pytest.raises(RefusedError, match="cannot read this molecule's start geometry: the prim"):
        pipeline.run_spectrum(_benzene(tmp_path / "benzene"), loaded)
    assert synthetic_chemistry == {}
    assert not (tmp_path / "benzene").exists()


@pytest.mark.usefixtures("synthetic_chemistry")
def test_a_spectrum_run_writes_the_correction_and_the_certificate(tmp_path: Path) -> None:
    loaded = release.load(write_release(tmp_path / "release"))
    record = pipeline.run_spectrum(_benzene(tmp_path / "benzene"), loaded)
    folder = tmp_path / "benzene"
    assert sorted(path.name for path in folder.iterdir()) == [
        "certificate.json",
        "certificate.md",
        "correction.npz",
        "geometry.json",
        "lowlevel.npz",
        "pyscf.log",
        "run.json",
    ]
    assert json.loads((folder / "certificate.json").read_text(encoding="utf-8")) == record
    assert len(record["bands"]) == 3 * 12 - 6
    assert record["computation"]["level"]["functional"] == "b3lyp"
    assert record["provenance"]["packages"]["torch"]
    with np.load(folder / "correction.npz") as arrays:
        assert arrays["per_seed"].shape == (2, 36, 36)
        np.testing.assert_allclose(arrays["mean"], arrays["per_seed"].mean(axis=0))
    text = (folder / "certificate.md").read_text(encoding="utf-8")
    assert text == certificate.to_markdown(record)


def test_an_element_the_network_never_trained_on_is_refused(tmp_path: Path) -> None:
    loaded = release.load(write_release(tmp_path / "release"))
    request = pipeline.Request(
        molecule=pipeline.MoleculeInput(smiles="Clc1ccccc1"), out=tmp_path / "x"
    )
    with pytest.raises(RefusedError, match=r"Cl outside the network's training set \(C, H, N, O\)"):
        pipeline.run_spectrum(request, loaded)
    assert not (tmp_path / "x").exists()


def test_the_baseline_asks_for_the_repair(
    tmp_path: Path, synthetic_chemistry: Counter[str]
) -> None:
    record = pipeline.run_baseline(_benzene(tmp_path / "benzene"))
    assert synthetic_chemistry == {"optimise": 1, "low level": 1, "repair": 1}
    assert record["soft_mode_repair"] is None  # the synthetic low level repaired nothing


@pytest.mark.parametrize("trained_on_repaired", [False, True])
@pytest.mark.usefixtures("synthetic_chemistry")
def test_the_network_reads_the_kind_of_hessian_it_was_trained_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, trained_on_repaired: bool
) -> None:
    def measure(
        geometry: Geometry, hessian: FloatArray, _resources: Resources, _log: TextIO
    ) -> tuple[FloatArray, repair.Repair]:
        stiffer = vibrations.project_rigid_motions(0.01 * np.eye(len(hessian)), geometry)
        return hessian + stiffer, repair.Repair(2, -8.0, 14.0, 22.0)

    monkeypatch.setattr(repair, "repair", measure)
    read: list[FloatArray] = []
    corrections = release.Release.corrections

    def spy(self: release.Release, geometry: Geometry, hessian: FloatArray) -> release.Corrections:
        read.append(hessian)
        return corrections(self, geometry, hessian)

    monkeypatch.setattr(release.Release, "corrections", spy)
    loaded = release.load(write_release(tmp_path / "release"))
    training = loaded.training.model_copy(update={"soft_mode_repair": trained_on_repaired})
    folder = tmp_path / "benzene"
    record = pipeline.run_spectrum(_benzene(folder), replace(loaded, training=training))
    with np.load(folder / "lowlevel.npz") as arrays:
        measured, analytic = arrays["hessian"], arrays["analytic_hessian"]
    assert float(np.abs(measured - analytic).max()) > 1e-3
    np.testing.assert_array_equal(read[-1], measured if trained_on_repaired else analytic)
    assert record["computation"]["soft_mode_repair"]["n_directions"] == 2
    assert record["soft_modes"].startswith("2 mode(s) below 20 cm⁻¹ were measured")
    assert ("The network read the Hessian without that repair" in record["soft_modes"]) is (
        not trained_on_repaired
    )
    assert "*Soft modes:* 2 mode(s) below 20 cm⁻¹" in (folder / "certificate.md").read_text(
        encoding="utf-8"
    )


@pytest.mark.usefixtures("synthetic_chemistry")
def test_the_network_runs_on_the_threads_the_request_allows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    asked: list[int] = []
    monkeypatch.setattr(torch, "set_num_threads", asked.append)
    loaded = release.load(write_release(tmp_path / "release"))
    request = pipeline.Request(
        molecule=pipeline.MoleculeInput(smiles="c1ccccc1"),
        out=tmp_path / "benzene",
        resources=Resources(threads=3),
    )
    pipeline.run_spectrum(request, loaded)
    assert asked == [3]
