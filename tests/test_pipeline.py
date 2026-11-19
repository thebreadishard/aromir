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
from aromir.constants import WAVENUMBERS_PER_HARTREE
from aromir.dft import Resources
from aromir.errors import InternalCoordinatesError, NotAMinimumError, RefusedError
from aromir.molecule import Geometry

from support import BENZENE_XYZ, bond_force_field, write_release

if TYPE_CHECKING:
    from collections.abc import Callable
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

    def optimise(
        start: Geometry, _resources: Resources, log: TextIO, **_progress: object
    ) -> optimisation.Optimisation:
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
    saddle = vibrations.vibrational_modes(-_synthetic_low_level(asked_at[0]).hessian, asked_at[0])
    step = pipeline._step_off_bohr(saddle, asked_at[0])  # noqa: SLF001  the rule itself
    assert step < pipeline.LARGEST_STEP_OFF_BOHR
    assert float(np.max(moved)) == pytest.approx(step)
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


def test_a_run_of_an_older_standard_form_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, synthetic_chemistry: Counter[str]
) -> None:
    def stop(*_arguments: object) -> None:
        raise repair.GridResponseMemoryError

    monkeypatch.setattr(repair, "repair", stop)
    folder = tmp_path / "benzene"
    with pytest.raises(repair.GridResponseMemoryError):
        pipeline.run_baseline(_benzene(folder))
    # An earlier aromir kept atom-map numbers: it named mapped benzene c1cc[cH:1]cc1.
    run = json.loads((folder / pipeline.RUN_FILE).read_text(encoding="utf-8"))
    run["asked"]["smiles"] = "c1cc[cH:1]cc1"
    (folder / pipeline.RUN_FILE).write_text(json.dumps(run), encoding="utf-8")
    with pytest.raises(RefusedError, match="reads in another standard form") as refusal:
        pipeline.run_baseline(_benzene(folder))
    assert "delete the folder to start over, or finish the run with the version" in str(
        refusal.value.reasons
    )
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

    def elsewhere(
        start: Geometry, resources: Resources, log: TextIO, **_progress: object
    ) -> optimisation.Optimisation:
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


def test_a_field_this_version_does_not_know_is_no_damaged_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, synthetic_chemistry: Counter[str]
) -> None:
    # A step file that reads cleanly but names a field another version wrote: the error is
    # the code's, and the folder's hours of DFT must not be called damaged.
    folder = tmp_path / "benzene"
    _stopped_before_the_repair(folder, monkeypatch)
    stored = json.loads((folder / "geometry.json").read_text(encoding="utf-8"))
    stored["optimisation"]["a_field_renamed_later"] = 1
    (folder / "geometry.json").write_text(json.dumps(stored), encoding="utf-8")
    with pytest.raises(TypeError, match="a_field_renamed_later"):
        pipeline.run_baseline(_benzene(folder))
    assert synthetic_chemistry == {"optimise": 1, "low level": 1}


@pytest.mark.parametrize(
    ("name", "edit"),
    [
        ("geometry.json", lambda stored: [stored]),
        ("geometry.json", lambda stored: stored | {"symbols": 5}),
        ("optimisation_progress.json", lambda stored: stored | {"n_steps": None}),
    ],
    ids=["a list at the top", "symbols a number", "no number of steps"],
)
def test_a_step_file_of_the_wrong_types_is_a_damaged_run(
    name: str,
    edit: Callable[[dict[str, Any]], object],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    synthetic_chemistry: Counter[str],
) -> None:
    folder = tmp_path / "benzene"
    _stopped_before_the_repair(folder, monkeypatch)
    stored = json.loads((folder / "geometry.json").read_text(encoding="utf-8"))
    if name == "optimisation_progress.json":
        (folder / "geometry.json").unlink()
        (folder / "lowlevel.npz").unlink()
        stored = {"attempt": stored["attempt"], "coords_bohr": stored["coords_bohr"], "n_steps": 3}
    (folder / name).write_text(json.dumps(edit(stored)), encoding="utf-8")
    with pytest.raises(RefusedError, match=rf"{name} in .* holds a damaged run"):
        pipeline.run_baseline(_benzene(folder))
    assert synthetic_chemistry == {"optimise": 1, "low level": 1}


def test_a_bug_in_building_the_saved_low_level_is_raised_as_itself(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, synthetic_chemistry: Counter[str]
) -> None:
    folder = tmp_path / "benzene"
    _stopped_before_the_repair(folder, monkeypatch)

    def broken(**_fields: object) -> None:
        msg = "a bug in the code, not in the file"
        raise RuntimeError(msg)

    monkeypatch.setattr(lowlevel, "LowLevel", broken)
    with pytest.raises(RuntimeError, match="a bug in the code"):
        pipeline.run_baseline(_benzene(folder))
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

    def killed_at_the_second(
        start: Geometry, resources: Resources, log: TextIO, **_progress: object
    ) -> object:
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


def test_an_optimisation_that_was_killed_is_taken_up_where_it_stood(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, synthetic_chemistry: Counter[str]
) -> None:
    optimise = optimisation.optimise
    taken_up_from: list[optimisation.Progress | None] = []

    def killed_after_three(
        start: Geometry,
        resources: Resources,
        log: TextIO,
        *,
        taken_up: optimisation.Progress | None,
        keep: Callable[[optimisation.Progress], None],
    ) -> optimisation.Optimisation:
        taken_up_from.append(taken_up)
        if taken_up is None:
            keep(optimisation.Progress((start.coords_bohr + 0.01).tolist(), 3))
            raise KeyboardInterrupt
        return optimise(Geometry(start.symbols, np.array(taken_up.coords_bohr)), resources, log)

    monkeypatch.setattr(optimisation, "optimise", killed_after_three)
    folder = tmp_path / "benzene"
    with pytest.raises(KeyboardInterrupt):
        pipeline.run_baseline(_benzene(folder))
    stored = json.loads((folder / pipeline.PROGRESS_FILE).read_text(encoding="utf-8"))
    assert (stored["attempt"], stored["n_steps"]) == (1, 3)
    pipeline.run_baseline(_benzene(folder))
    reached = taken_up_from[1]
    assert reached is not None
    assert reached.n_steps == 3
    assert reached.coords_bohr == stored["coords_bohr"]
    assert not (folder / pipeline.PROGRESS_FILE).exists()
    geometry = json.loads((folder / "geometry.json").read_text(encoding="utf-8"))
    np.testing.assert_allclose(geometry["coords_bohr"], stored["coords_bohr"])
    assert synthetic_chemistry["optimise"] == 1


def test_a_saved_progress_that_is_not_this_molecules_is_a_damaged_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, synthetic_chemistry: Counter[str]
) -> None:
    def killed(*_arguments: object, **_progress: object) -> None:
        raise KeyboardInterrupt

    working = optimisation.optimise
    monkeypatch.setattr(optimisation, "optimise", killed)
    folder = tmp_path / "benzene"
    with pytest.raises(KeyboardInterrupt):
        pipeline.run_baseline(_benzene(folder))
    monkeypatch.setattr(optimisation, "optimise", working)
    saved = {"attempt": 1, "coords_bohr": [[0.0, 0.0, 0.0]], "n_steps": 3}
    (folder / pipeline.PROGRESS_FILE).write_text(json.dumps(saved), encoding="utf-8")
    with pytest.raises(RefusedError, match=r"optimisation_progress\.json in .* damaged run"):
        pipeline.run_baseline(_benzene(folder))
    assert not synthetic_chemistry


def test_the_step_off_a_saddle_point_is_sized_from_its_curvature_and_capped() -> None:
    _, benzene = pipeline.admit(pipeline.MoleculeInput(smiles="c1ccccc1"))
    along = vibrations.vibrational_modes(_synthetic_low_level(benzene).hessian, benzene).vectors[
        :, :1
    ]
    stiff = pipeline._step_off_bohr(vibrations.Modes(np.array([-400.0]), along), benzene)  # noqa: SLF001
    _, length = vibrations.displacement(along[:, 0], benzene.masses_amu, stiff)
    lowering = 0.5 * (400.0 / WAVENUMBERS_PER_HARTREE) ** 2 * length**2
    assert lowering == pytest.approx(pipeline.SADDLE_ENERGY_LOWERING_HARTREE)
    assert stiff < pipeline.LARGEST_STEP_OFF_BOHR
    soft = pipeline._step_off_bohr(vibrations.Modes(np.array([-12.0]), along), benzene)  # noqa: SLF001
    assert soft == pipeline.LARGEST_STEP_OFF_BOHR


def _imaginary_at(monkeypatch: pytest.MonkeyPatch, *wavenumbers: float) -> None:
    """Give every low-level Hessian its lowest modes at these imaginary positions, cm⁻¹."""
    stable = lowlevel.compute

    def compute(geometry: Geometry, resources: Resources, log: TextIO) -> lowlevel.LowLevel:
        result = stable(geometry, resources, log)
        modes = vibrations.vibrational_modes(result.hessian, geometry)
        hessian = result.hessian.copy()
        for index, wavenumber in enumerate(wavenumbers):
            weighted = modes.vectors[:, index] * vibrations.mass_weights(geometry.masses_amu)
            now = (modes.wavenumbers[index] / WAVENUMBERS_PER_HARTREE) ** 2
            wanted = -((wavenumber / WAVENUMBERS_PER_HARTREE) ** 2)
            hessian += (wanted - now) * np.outer(weighted, weighted)
        return replace(result, hessian=hessian)

    monkeypatch.setattr(lowlevel, "compute", compute)


def _flat_at_minus_ten(monkeypatch: pytest.MonkeyPatch) -> None:
    _imaginary_at(monkeypatch, -10.0)


# Exactly -20 cm⁻¹ does not survive the arithmetic that makes a Hessian of it (it came back
# about 3e-10 on the soft side), so these cases stand just beyond the line; the line itself
# is the next test's.
@pytest.mark.parametrize(
    "wavenumbers",
    [(-25.0,), (-20.01,), (-30.0, -10.0)],
    ids=["one", "just beyond the line", "one of two"],
)
def test_an_imaginary_mode_beyond_the_soft_mode_line_still_gets_no_spectrum(
    wavenumbers: tuple[float, ...],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    synthetic_chemistry: Counter[str],
) -> None:
    _imaginary_at(monkeypatch, *wavenumbers)
    said = f"{len(wavenumbers)} imaginary mode\\(s\\), the lowest at {wavenumbers[0]:.0f} cm"
    with pytest.raises(NotAMinimumError, match=said):
        pipeline.run_baseline(_benzene(tmp_path / "benzene"))
    assert synthetic_chemistry == {"optimise": 2, "low level": 2, "repair": 2}
    assert not (tmp_path / "benzene" / "baseline.json").exists()


def test_a_mode_exactly_at_the_soft_mode_line_gets_no_spectrum(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, synthetic_chemistry: Counter[str]
) -> None:
    # 'At or beyond the line refuses': the mode is set to the line where the run reads it.
    _imaginary_at(monkeypatch, -19.0)
    modes_of = vibrations.vibrational_modes

    def at_the_line(hessian: FloatArray, geometry: Geometry) -> vibrations.Modes:
        modes = modes_of(hessian, geometry)
        if modes.wavenumbers[0] >= 0:
            return modes
        wavenumbers = modes.wavenumbers.copy()
        wavenumbers[0] = -repair.SOFT_BELOW_CM
        return vibrations.Modes(wavenumbers, modes.vectors)

    monkeypatch.setattr(vibrations, "vibrational_modes", at_the_line)
    with pytest.raises(NotAMinimumError, match=r"1 imaginary mode\(s\), the lowest at -20 cm"):
        pipeline.run_baseline(_benzene(tmp_path / "benzene"))
    assert synthetic_chemistry == {"optimise": 2, "low level": 2, "repair": 2}


def test_the_first_attempts_progress_is_not_the_second_attempts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, synthetic_chemistry: Counter[str]
) -> None:
    # A progress file that outlived the first optimisation says so itself: the second
    # optimisation, off the saddle point, does not start from it.
    _unstable_for(1, monkeypatch)
    optimise = optimisation.optimise
    step_off = pipeline._step_off  # noqa: SLF001  the moment between the two attempts
    taken_up_from: list[object] = []
    folder = tmp_path / "benzene"

    def watched(
        start: Geometry, resources: Resources, log: TextIO, **progress: object
    ) -> optimisation.Optimisation:
        taken_up_from.append(progress["taken_up"])
        return optimise(start, resources, log)

    def leaving_a_stale_file(
        folder: Path,
        state: Any,  # noqa: ANN401  the run's own state, a private class
        saddle: Geometry,
        modes: vibrations.Modes,
    ) -> None:
        step_off(folder, state, saddle, modes)
        stale = {"attempt": 1, "coords_bohr": saddle.coords_bohr.tolist(), "n_steps": 7}
        (folder / pipeline.PROGRESS_FILE).write_text(json.dumps(stale), encoding="utf-8")

    monkeypatch.setattr(optimisation, "optimise", watched)
    monkeypatch.setattr(pipeline, "_step_off", leaving_a_stale_file)
    pipeline.run_baseline(_benzene(folder))
    assert taken_up_from == [None, None]
    assert synthetic_chemistry["optimise"] == 2


@pytest.mark.parametrize(
    ("n_steps", "coordinate"),
    [(0, 0.0), (-5, 0.0), (3, float("nan"))],
    ids=["no gradient", "fewer than none", "no number"],
)
def test_a_saved_progress_that_cannot_be_a_runs_is_a_damaged_run(
    n_steps: int, coordinate: float, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class StoppedError(Exception):
        """The stand-in's end: not an interrupt, which would end the test session if it escaped."""

    def stopped(start: Geometry, *_arguments: object, **progress: Any) -> None:  # noqa: ANN401
        coords = start.coords_bohr.copy()
        coords[0, 0] += coordinate
        progress["keep"](optimisation.Progress(coords.tolist(), n_steps))
        raise StoppedError

    monkeypatch.setattr(optimisation, "optimise", stopped)
    folder = tmp_path / "benzene"
    with pytest.raises(StoppedError):
        pipeline.run_baseline(_benzene(folder))
    with pytest.raises(RefusedError, match=r"optimisation_progress\.json in .* damaged run"):
        pipeline.run_baseline(_benzene(folder))


def test_the_progress_of_the_optimisation_off_a_saddle_point_is_the_second_attempts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, synthetic_chemistry: Counter[str]
) -> None:
    # The case the saved progress is for: a run stopped in the optimisation that follows
    # the step off a saddle point goes on there, not at the first attempt's start.
    _unstable_for(1, monkeypatch)
    optimise = optimisation.optimise
    taken_up_from: list[optimisation.Progress | None] = []

    class StoppedError(Exception):
        """The stand-in's end."""

    def stopped_in_the_second(
        start: Geometry,
        resources: Resources,
        log: TextIO,
        **progress: Any,  # noqa: ANN401  the keywords under test
    ) -> optimisation.Optimisation:
        taken_up_from.append(progress["taken_up"])
        if synthetic_chemistry["optimise"] == 1 and progress["taken_up"] is None:
            progress["keep"](optimisation.Progress(start.coords_bohr.tolist(), 60))
            raise StoppedError
        return optimise(start, resources, log)

    monkeypatch.setattr(optimisation, "optimise", stopped_in_the_second)
    folder = tmp_path / "benzene"
    with pytest.raises(StoppedError):
        pipeline.run_baseline(_benzene(folder))
    stored = json.loads((folder / pipeline.PROGRESS_FILE).read_text(encoding="utf-8"))
    assert (stored["attempt"], stored["n_steps"]) == (2, 60)
    pipeline.run_baseline(_benzene(folder))
    assert [reached and reached.n_steps for reached in taken_up_from] == [None, None, 60]
    assert synthetic_chemistry["optimise"] == 2


def test_a_flat_coordinate_gets_its_spectrum_with_that_band_named_and_unlicensed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, synthetic_chemistry: Counter[str]
) -> None:
    _flat_at_minus_ten(monkeypatch)
    loaded = release.load(write_release(tmp_path / "release"))
    record = pipeline.run_spectrum(_benzene(tmp_path / "benzene"), loaded)
    assert synthetic_chemistry == {"optimise": 2, "low level": 2, "repair": 2}
    flat = [band for band in record["bands"] if band["b3lyp_cm"] < 0]
    assert [band["b3lyp_cm"] for band in flat] == [-10.0]
    assert flat[0]["licence_note"] == certificate.FLAT_COORDINATE
    assert not flat[0]["licensed"]
    assert "1 band(s) lie on a flat coordinate" in record["statement"]
    assert any("is not a band position and is never licensed" in w for w in record["warnings"])
    assert record["computation"]["optimisation"]["left_saddle_point_at_cm"] == pytest.approx(-10.0)
    text = (tmp_path / "benzene" / "certificate.md").read_text(encoding="utf-8")
    assert "| no: a flat coordinate: its low-level mode is imaginary" in text


def test_the_baseline_names_a_flat_coordinate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, synthetic_chemistry: Counter[str]
) -> None:
    _flat_at_minus_ten(monkeypatch)
    record = pipeline.run_baseline(_benzene(tmp_path / "benzene"))
    assert synthetic_chemistry == {"optimise": 2, "low level": 2, "repair": 2}
    assert record["notes"][0].startswith(
        "the molecule is flat along 1 coordinate(s): the mode(s) at -10.0 cm⁻¹ are imaginary"
    )
    assert record["harmonic"]["bands"][0]["wavenumber_cm"] == pytest.approx(-10.0, abs=0.05)
    for kind in ("harmonic", "scaled"):
        marked = [band["flat_coordinate"] for band in record[kind]["bands"]]
        assert marked == [True] + [False] * (len(marked) - 1)


@pytest.mark.parametrize(
    ("further", "how"),
    [
        (2, "not met on a flat coordinate"),
        (9, "not met, and 9 more gradients were needed for the energy and the forces"),
    ],
    ids=["one step was enough", "more were needed"],
)
def test_a_geometry_converged_on_energy_and_forces_says_so_with_its_numbers(
    further: int,
    how: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    synthetic_chemistry: Counter[str],
) -> None:
    last_step = {
        "energy_change_hartree": -6.7e-10,
        "rms_force_hartree_per_bohr": 1.3e-6,
        "max_force_hartree_per_bohr": 2.3e-6,
        "rms_displacement_angstrom": 4.2e-4,
        "max_displacement_angstrom": 9.1e-4,
    }

    def crawled(
        start: Geometry, *_arguments: object, **_progress: object
    ) -> optimisation.Optimisation:
        synthetic_chemistry["optimise"] += 1
        return optimisation.Optimisation(
            start, -1.0, 1e-6, 104, optimisation.ENERGY_AND_FORCES, last_step, further
        )

    monkeypatch.setattr(optimisation, "optimise", crawled)
    folder = tmp_path / "benzene"
    _stopped_before_the_repair(folder, monkeypatch)
    loaded_back = pipeline.run_baseline(_benzene(folder))
    assert synthetic_chemistry["optimise"] == 1
    assert loaded_back["optimisation"]["converged_on"] == optimisation.ENERGY_AND_FORCES
    assert loaded_back["optimisation"]["last_step"] == last_step
    assert loaded_back["optimisation"]["gradients_without_displacement"] == further
    said = (
        "the geometry is converged on energy and forces; the displacement criterion was "
        f"{how} in 100 steps and was not applied after them (104 gradients "
        "in all; geomeTRIC's GAU_TIGHT thresholds for the energy change and for the rms and "
        "the largest force were met; between the last two gradients the energy changed by "
        "-6.7e-10 hartree and the atoms moved by 4.2e-04 rms and 9.1e-04 Å at most, with "
        "forces of 1.3e-06 rms and 2.3e-06 hartree/bohr at most at the end)"
    )
    assert loaded_back["notes"] == [said]
    record = pipeline.run_spectrum(
        _benzene(tmp_path / "again"), release.load(write_release(tmp_path / "release"))
    )
    assert loaded_back["notes"][0] in record["not_in_the_error"]
    assert "- The geometry is converged on energy and forces;" in (
        tmp_path / "again" / "certificate.md"
    ).read_text(encoding="utf-8")
