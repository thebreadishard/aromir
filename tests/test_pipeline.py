"""The pipeline's own logic, with the quantum chemistry replaced by a synthetic result."""

from __future__ import annotations

import json
import tokenize
import zlib
from collections import Counter
from dataclasses import replace
from typing import TYPE_CHECKING, Any, TextIO

import numpy as np
import pytest
import torch
from pydantic import ValidationError

from aromir import (
    certificate,
    lowlevel,
    optimisation,
    pipeline,
    records,
    release,
    repair,
    response,
    spectrum,
    vibrations,
)
from aromir.dft import Resources
from aromir.errors import InternalCoordinatesError, NotAMinimumError, RefusedError
from aromir.molecule import Geometry

from support import BENZENE_XYZ, bond_force_field, write_release

if TYPE_CHECKING:
    from pathlib import Path

    from aromir.molecule import FloatArray


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

    def compute(geometry: Geometry, _resources: Resources, _log: TextIO) -> lowlevel.LowLevel:
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

    def compute(geometry: Geometry, resources: Resources, log: TextIO) -> lowlevel.LowLevel:
        asked_at.append(geometry)
        result = stable(geometry, resources, log)
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
        RefusedError,
        match=r"unfinished run of another request \(baseline c1ccccc1; it differs in smiles",
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


def test_a_kill_between_two_writes_does_not_feed_the_network_a_measured_hessian(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, synthetic_chemistry: Counter[str]
) -> None:
    def measure(
        geometry: Geometry, hessian: FloatArray, _resources: Resources, _log: TextIO
    ) -> tuple[FloatArray, repair.Repair]:
        synthetic_chemistry["repair"] += 1
        stiffer = vibrations.project_rigid_motions(0.01 * np.eye(len(hessian)), geometry)
        return hessian + stiffer, repair.Repair(2, -8.0, 14.0, 22.0)

    monkeypatch.setattr(repair, "repair", measure)
    write_arrays = records.write_arrays

    def killed_once_the_measured_hessian_is_saved(
        path: Path, arrays: dict[str, FloatArray]
    ) -> None:
        write_arrays(path, arrays)
        if path.name == pipeline.LOW_LEVEL_FILE and "analytic_hessian" in arrays:
            raise KeyboardInterrupt

    monkeypatch.setattr(records, "write_arrays", killed_once_the_measured_hessian_is_saved)
    loaded = release.load(write_release(tmp_path / "release"))
    folder = tmp_path / "benzene"
    with pytest.raises(KeyboardInterrupt):
        pipeline.run_spectrum(_benzene(folder), loaded)
    monkeypatch.setattr(records, "write_arrays", write_arrays)
    read: list[FloatArray] = []
    corrections = release.Release.corrections

    def spy(self: release.Release, geometry: Geometry, hessian: FloatArray) -> release.Corrections:
        read.append(hessian)
        return corrections(self, geometry, hessian)

    monkeypatch.setattr(release.Release, "corrections", spy)
    record = pipeline.run_spectrum(_benzene(folder), loaded)
    assert synthetic_chemistry == {"optimise": 1, "low level": 1, "repair": 1}
    analytic = _synthetic_low_level(Geometry(*_saved_geometry(folder))).hessian
    np.testing.assert_array_equal(read[-1], analytic)
    assert record["computation"]["soft_mode_repair"]["n_directions"] == 2


def _saved_geometry(folder: Path) -> tuple[tuple[str, ...], FloatArray]:
    stored = json.loads((folder / "geometry.json").read_text(encoding="utf-8"))
    return tuple(stored["symbols"]), np.array(stored["coords_bohr"])


@pytest.mark.parametrize(
    ("name", "content", "said"),
    [
        ("run.json", "[]", "run.json in"),
        ("run.json", "{", "run.json in"),
        ("lowlevel.npz", "not a zip file", "lowlevel.npz in"),
        ("lowlevel.npz", "", "lowlevel.npz in"),
        ("geometry.json", "{}", "geometry.json in"),
    ],
)
def test_a_damaged_run_is_refused_in_words(
    name: str,
    content: str,
    said: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    synthetic_chemistry: Counter[str],
) -> None:
    def stop(*_arguments: object) -> None:
        raise repair.GridResponseMemoryError

    working = repair.repair
    monkeypatch.setattr(repair, "repair", stop)
    folder = tmp_path / "benzene"
    with pytest.raises(repair.GridResponseMemoryError):
        pipeline.run_baseline(_benzene(folder))
    monkeypatch.setattr(repair, "repair", working)
    (folder / name).write_text(content, encoding="utf-8")
    with pytest.raises(
        RefusedError, match="holds a damaged run; delete it to start over"
    ) as refusal:
        pipeline.run_baseline(_benzene(folder))
    assert said in refusal.value.reasons[0]
    assert synthetic_chemistry["optimise"] == 1


@pytest.mark.usefixtures("synthetic_chemistry")
def test_a_folder_a_killed_run_left_nothing_in_is_used(tmp_path: Path) -> None:
    folder = tmp_path / "benzene"
    folder.mkdir()
    (folder / "pyscf.log").write_text("started", encoding="utf-8")
    (folder / "run.json.tmp").write_text("{", encoding="utf-8")
    pipeline.run_baseline(_benzene(folder))
    assert (folder / "baseline.json").is_file()
    assert (folder / "pyscf.log").read_text(encoding="utf-8").startswith("started")


def test_a_hessian_is_not_reused_for_another_geometry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, synthetic_chemistry: Counter[str]
) -> None:
    def stop(*_arguments: object) -> None:
        raise repair.GridResponseMemoryError

    working = repair.repair
    monkeypatch.setattr(repair, "repair", stop)
    folder = tmp_path / "benzene"
    with pytest.raises(repair.GridResponseMemoryError):
        pipeline.run_baseline(_benzene(folder))
    monkeypatch.setattr(repair, "repair", working)
    optimise = optimisation.optimise

    def elsewhere(start: Geometry, resources: Resources, log: TextIO) -> optimisation.Optimisation:
        found = optimise(start, resources, log)
        return replace(found, geometry=Geometry(start.symbols, start.coords_bohr * 1.01))

    monkeypatch.setattr(optimisation, "optimise", elsewhere)
    (folder / "geometry.json").unlink()
    pipeline.run_baseline(_benzene(folder))
    assert synthetic_chemistry == {"optimise": 2, "low level": 2, "repair": 1}


def test_the_certificate_says_when_the_structure_is_one_off_a_saddle_point(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, synthetic_chemistry: Counter[str]
) -> None:
    _unstable_for(1, monkeypatch)
    loaded = release.load(write_release(tmp_path / "release"))
    record = pipeline.run_spectrum(_benzene(tmp_path / "benzene"), loaded)
    assert synthetic_chemistry["optimise"] == 2
    assert any(
        line.startswith("the start geometry led to a saddle point (a mode at -")
        for line in record["not_in_the_error"]
    )
    text = (tmp_path / "benzene" / "certificate.md").read_text(encoding="utf-8")
    assert "- The start geometry led to a saddle point" in text
    assert sorted(record["computation"]["steps"]) == [
        "low level 1",
        "low level 2",
        "optimisation 1",
        "optimisation 2",
        "soft modes 1",
        "soft modes 2",
    ]


def test_the_same_start_geometry_is_the_same_request_whatever_the_sign_of_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, synthetic_chemistry: Counter[str]
) -> None:
    def stop(*_arguments: object) -> None:
        raise repair.GridResponseMemoryError

    working = repair.repair
    monkeypatch.setattr(repair, "repair", stop)
    folder = tmp_path / "benzene"
    with pytest.raises(repair.GridResponseMemoryError):
        pipeline.run_baseline(_benzene(folder))
    monkeypatch.setattr(repair, "repair", working)
    admit = pipeline.admit

    def a_hair_off(molecule: pipeline.MoleculeInput) -> tuple[object, Geometry]:
        admitted, start = admit(molecule)
        return admitted, Geometry(
            start.symbols, np.where(start.coords_bohr == 0, -0.0, start.coords_bohr) - 1e-9
        )

    monkeypatch.setattr(pipeline, "admit", a_hair_off)
    pipeline.run_baseline(_benzene(folder))
    assert synthetic_chemistry["optimise"] == 1


def test_a_folder_with_somebody_elses_temporary_file_is_not_taken(
    tmp_path: Path, synthetic_chemistry: Counter[str]
) -> None:
    folder = tmp_path / "work"
    folder.mkdir()
    (folder / "thesis.tmp").write_text("mine", encoding="utf-8")
    with pytest.raises(RefusedError, match="is not empty: a result is never overwritten"):
        pipeline.run_baseline(_benzene(folder))
    assert (folder / "thesis.tmp").read_text(encoding="utf-8") == "mine"
    assert not synthetic_chemistry


def test_a_run_without_a_soft_mode_is_not_measured_twice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, synthetic_chemistry: Counter[str]
) -> None:
    # The repair found nothing to measure and said so in the step file; a failure after it
    # (here: the network) must not send the resumed run through the repair again.
    loaded = release.load(write_release(tmp_path / "release"))
    working = release.Release.corrections
    calls = []

    def fails_after_the_start(
        self: release.Release, geometry: Geometry, hessian: FloatArray
    ) -> release.Corrections:
        calls.append(1)
        if len(calls) == 2:
            msg = "the network broke"
            raise InternalCoordinatesError(msg)
        return working(self, geometry, hessian)

    monkeypatch.setattr(release.Release, "corrections", fails_after_the_start)
    folder = tmp_path / "benzene"
    with pytest.raises(InternalCoordinatesError):
        pipeline.run_spectrum(_benzene(folder), loaded)
    assert synthetic_chemistry == {"optimise": 1, "low level": 1, "repair": 1}
    record = pipeline.run_spectrum(_benzene(folder), loaded)
    assert synthetic_chemistry == {"optimise": 1, "low level": 1, "repair": 1}
    assert sorted(record["computation"]["steps"]) == [
        "low level 1",
        "optimisation 1",
        "soft modes 1",
    ]
    assert all(step["packages"]["aromir"] for step in record["computation"]["steps"].values())


def test_a_step_names_the_run_that_made_it_in_its_own_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, synthetic_chemistry: Counter[str]
) -> None:
    # A kill between a step's file and anything written after it loses nothing of the
    # record: what made the step is in the step file.
    def stop(*_arguments: object) -> None:
        raise repair.GridResponseMemoryError

    working = repair.repair
    monkeypatch.setattr(repair, "repair", stop)
    folder = tmp_path / "benzene"
    with pytest.raises(repair.GridResponseMemoryError):
        pipeline.run_baseline(_benzene(folder))
    stored = json.loads((folder / "geometry.json").read_text(encoding="utf-8"))
    assert stored["made_with"]["packages"]["pyscf"]
    with np.load(folder / "lowlevel.npz") as arrays:
        assert list(json.loads(str(arrays["made_with"]))) == ["low level"]
    monkeypatch.setattr(repair, "repair", working)
    record = pipeline.run_baseline(_benzene(folder))
    assert sorted(record["steps"]) == ["low level 1", "optimisation 1", "soft modes 1"]
    assert synthetic_chemistry["optimise"] == 1


def test_a_run_killed_before_the_certificates_record_is_taken_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, synthetic_chemistry: Counter[str]
) -> None:
    # The record is what makes a folder a result, so it is written last: a kill at the
    # record leaves the text and an unfinished run, which the same request finishes
    # without any quantum chemistry.
    loaded = release.load(write_release(tmp_path / "release"))
    folder = tmp_path / "benzene"
    write_json = records.write_json

    def killed_at_the_record(path: Path, record: dict[str, Any]) -> None:
        if path.name == pipeline.CERTIFICATE_FILE:
            raise KeyboardInterrupt
        write_json(path, record)

    monkeypatch.setattr(records, "write_json", killed_at_the_record)
    with pytest.raises(KeyboardInterrupt):
        pipeline.run_spectrum(_benzene(folder), loaded)
    assert (folder / "certificate.md").is_file()
    assert not (folder / "certificate.json").exists()
    monkeypatch.setattr(records, "write_json", write_json)
    pipeline.run_spectrum(_benzene(folder), loaded)
    assert (folder / "certificate.json").is_file()
    assert synthetic_chemistry == {"optimise": 1, "low level": 1, "repair": 1}


def test_the_start_geometry_is_named_the_same_whatever_the_sign_of_its_zeros() -> None:
    admitted, start = pipeline.admit(pipeline.MoleculeInput(smiles="c1ccccc1"))
    flat = start.coords_bohr.copy()
    flat[:, 2] = 0.0
    below = flat.copy()
    below[:, 2] = -1e-9
    above = Geometry(start.symbols, flat)
    asked = pipeline._asked("baseline", admitted, above)  # noqa: SLF001  the identity itself
    assert asked == pipeline._asked("baseline", admitted, Geometry(start.symbols, below))  # noqa: SLF001
    elsewhere = flat.copy()
    elsewhere[0, 0] += 1e-4
    assert asked != pipeline._asked("baseline", admitted, Geometry(start.symbols, elsewhere))  # noqa: SLF001


def _stopped_before_the_repair(folder: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Leave a run with its optimisation and its analytic Hessian saved."""

    def stop(*_arguments: object) -> None:
        raise repair.GridResponseMemoryError

    working = repair.repair
    monkeypatch.setattr(repair, "repair", stop)
    with pytest.raises(repair.GridResponseMemoryError):
        pipeline.run_baseline(_benzene(folder))
    monkeypatch.setattr(repair, "repair", working)


@pytest.mark.parametrize(
    ("change", "said"),
    [
        ({"hessian": np.zeros((3, 3))}, "arrays of shape (3, 3)"),
        ({"soft_modes_measured": np.array(True)}, "contradicts itself"),  # noqa: FBT003  a stored flag
        ({"made_with": np.array("{}")}, "does not name the run that computed it"),
        ({"made_with": np.array("[]")}, "not a record"),
    ],
    ids=["another shape", "measured without a run", "no low-level run", "no record"],
)
def test_a_low_level_file_that_cannot_be_this_runs_is_a_damaged_run(
    change: dict[str, FloatArray],
    said: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    synthetic_chemistry: Counter[str],
) -> None:
    folder = tmp_path / "benzene"
    _stopped_before_the_repair(folder, monkeypatch)
    with np.load(folder / "lowlevel.npz") as arrays:
        saved = {name: np.array(arrays[name]) for name in arrays.files}
    records.write_arrays(folder / "lowlevel.npz", saved | change)
    with pytest.raises(RefusedError, match="holds a damaged run") as refusal:
        pipeline.run_baseline(_benzene(folder))
    assert said in refusal.value.reasons[0]
    assert synthetic_chemistry == {"optimise": 1, "low level": 1}


@pytest.mark.parametrize(
    "raised",
    [
        zlib.error("Error -3 while decompressing data: invalid stored block lengths"),
        tokenize.TokenError("EOF in multi-line statement", (2, 0)),
        NotImplementedError("zip file version 21.0"),
        RuntimeError("File 'hessian.npy' is encrypted, password required for extraction"),
    ],
    ids=["zlib", "npy header", "zip version", "encryption flag"],
)
def test_a_low_level_file_that_numpy_cannot_open_is_a_damaged_run(
    raised: Exception,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    synthetic_chemistry: Counter[str],
) -> None:
    # What flipped bytes of a real file made numpy raise, each given back by a stand-in.
    folder = tmp_path / "benzene"
    _stopped_before_the_repair(folder, monkeypatch)

    def does_not_open(*_arguments: object) -> None:
        raise raised

    monkeypatch.setattr(np, "load", does_not_open)
    with pytest.raises(RefusedError, match=r"lowlevel\.npz in .* cannot be read") as refusal:
        pipeline.run_baseline(_benzene(folder))
    assert type(raised).__name__ in refusal.value.reasons[0]
    assert "holds a damaged run" in refusal.value.reasons[0]
    assert synthetic_chemistry == {"optimise": 1, "low level": 1}


def test_a_geometry_file_that_does_not_name_its_run_is_refused_before_the_next_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, synthetic_chemistry: Counter[str]
) -> None:
    # A folder of an earlier aromir: nothing is computed on top of a step that cannot go
    # into the record, or every later resume would end at the same missing name.
    folder = tmp_path / "benzene"
    _stopped_before_the_repair(folder, monkeypatch)
    stored = json.loads((folder / "geometry.json").read_text(encoding="utf-8"))
    del stored["made_with"]
    (folder / "geometry.json").write_text(json.dumps(stored), encoding="utf-8")
    (folder / "lowlevel.npz").unlink()
    with pytest.raises(RefusedError, match=r"geometry\.json in .* holds a damaged run"):
        pipeline.run_baseline(_benzene(folder))
    assert synthetic_chemistry == {"optimise": 1, "low level": 1}
    assert not (folder / "lowlevel.npz").exists()


def test_a_saddle_points_file_that_does_not_name_its_run_is_refused_before_the_second_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, synthetic_chemistry: Counter[str]
) -> None:
    _unstable_for(1, monkeypatch)
    optimise = optimisation.optimise

    def killed_at_the_second(start: Geometry, resources: Resources, log: TextIO) -> object:
        if synthetic_chemistry["optimise"]:
            raise KeyboardInterrupt
        return optimise(start, resources, log)

    monkeypatch.setattr(optimisation, "optimise", killed_at_the_second)
    folder = tmp_path / "benzene"
    with pytest.raises(KeyboardInterrupt):
        pipeline.run_baseline(_benzene(folder))
    monkeypatch.setattr(optimisation, "optimise", optimise)
    kept = folder / "saddle_point_geometry.json"
    stored = json.loads(kept.read_text(encoding="utf-8"))
    del stored["made_with"]
    kept.write_text(json.dumps(stored), encoding="utf-8")
    said = r"saddle_point_geometry\.json or saddle_point_lowlevel\.npz in .* holds a damaged run"
    with pytest.raises(RefusedError, match=said):
        pipeline.run_baseline(_benzene(folder))
    assert synthetic_chemistry == {"optimise": 1, "low level": 1, "repair": 1}
