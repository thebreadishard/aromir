"""The pipeline's own logic, with the quantum chemistry replaced by a synthetic result."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import TYPE_CHECKING, TextIO

import numpy as np
import pytest
from pydantic import ValidationError

from aromir import certificate, lowlevel, optimisation, pipeline, release, response, spectrum
from aromir.errors import NotAMinimumError, RefusedError

from support import BENZENE_XYZ, bond_force_field, write_release

if TYPE_CHECKING:
    from pathlib import Path

    from aromir.dft import Resources
    from aromir.molecule import Geometry


def _ring_force_field(geometry: Geometry) -> tuple[tuple[int, int, float], ...]:
    """Springs between all atom pairs: a stable synthetic Hessian for any geometry."""
    n = geometry.n_atoms
    return tuple((i, j, 0.3) for i in range(n) for j in range(i + 1, n))


@pytest.fixture
def synthetic_chemistry(monkeypatch: pytest.MonkeyPatch) -> list[bool]:
    """Replace the optimisation and the low level by instant, synthetic results.

    Returns the list that records, per low-level call, whether the soft-mode repair was asked for.
    """
    asked_for_repair: list[bool] = []

    def optimise(start: Geometry, _resources: Resources, log: TextIO) -> optimisation.Optimisation:
        log.write("optimised\n")
        return optimisation.Optimisation(start, energy=-1.0, max_force=1e-6, n_steps=3)

    def compute(
        geometry: Geometry, _resources: Resources, _log: TextIO, *, repair_soft_modes: bool
    ) -> lowlevel.LowLevel:
        asked_for_repair.append(repair_soft_modes)
        return lowlevel.LowLevel(
            energy=-1.0,
            hessian=bond_force_field(geometry, _ring_force_field(geometry)),
            polar_tensor=np.random.default_rng(0).normal(size=(3, 3 * geometry.n_atoms)),
            refinement=response.Refinement(1e-9, 1e-9, 0, 0),
            sum_rule_violation=1e-7,
            seconds=0.1,
        )

    monkeypatch.setattr(optimisation, "optimise", optimise)
    monkeypatch.setattr(lowlevel, "compute", compute)
    return asked_for_repair


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


@pytest.mark.usefixtures("synthetic_chemistry")
def test_a_saddle_point_gets_no_spectrum(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    stable = lowlevel.compute

    def unstable(
        geometry: Geometry, resources: Resources, log: TextIO, *, repair_soft_modes: bool
    ) -> lowlevel.LowLevel:
        result = stable(geometry, resources, log, repair_soft_modes=repair_soft_modes)
        return lowlevel.LowLevel(
            result.energy,
            -result.hessian,
            result.polar_tensor,
            result.refinement,
            result.sum_rule_violation,
            result.seconds,
        )

    monkeypatch.setattr(lowlevel, "compute", unstable)
    with pytest.raises(NotAMinimumError, match="imaginary mode"):
        pipeline.run_baseline(_benzene(tmp_path / "benzene"))
    assert not (tmp_path / "benzene" / "baseline.json").exists()


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


def test_the_baseline_asks_for_the_repair(tmp_path: Path, synthetic_chemistry: list[bool]) -> None:
    record = pipeline.run_baseline(_benzene(tmp_path / "benzene"))
    assert synthetic_chemistry == [True]
    assert record["soft_mode_repair"] is None  # the synthetic low level repaired nothing


@pytest.mark.parametrize("trained_on_repaired", [False, True])
def test_a_release_decides_whether_its_network_gets_a_repaired_hessian(
    tmp_path: Path, synthetic_chemistry: list[bool], *, trained_on_repaired: bool
) -> None:
    loaded = release.load(write_release(tmp_path / "release"))
    training = loaded.training.model_copy(update={"soft_mode_repair": trained_on_repaired})
    chosen = replace(loaded, training=training)
    pipeline.run_spectrum(_benzene(tmp_path / "benzene"), chosen)
    assert synthetic_chemistry == [trained_on_repaired]
