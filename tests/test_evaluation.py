"""The two errors of a release: of the seeds' mean in the bands, and of each seed to first order."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
import pytest

from aromir import (
    admission,
    certificate,
    cli,
    evaluation,
    families,
    geometry,
    lowlevel,
    release,
    response,
)
from aromir import vibrations as vib
from aromir.errors import RefusedError

from support import random_hessian, write_release

if TYPE_CHECKING:
    from pathlib import Path

    from aromir.molecule import FloatArray, Geometry

N_SEEDS = 2
SMILES = ("c1ccccc1", "Oc1ccccc1", "c1ccncc1")


@pytest.fixture(scope="module")
def loaded(tmp_path_factory: pytest.TempPathFactory) -> release.Release:
    return release.load(write_release(tmp_path_factory.mktemp("release")))


def _stable_hessian(start: Geometry, seed: int) -> FloatArray:
    """Return a Hessian with real, well separated modes: random, with the rigid motions out."""
    return vib.project_rigid_motions(random_hessian(start.n_atoms, seed) + 0.0, start)


def _reference(smiles: str, loaded: release.Release, *, off_by: float) -> evaluation.Reference:
    """Return a molecule whose reference is the seeds' mean answer plus a stated extra."""
    start = geometry.start_from_smiles(smiles)
    hessian = _stable_hessian(start, seed=len(smiles))
    answer = loaded.corrections(start, hessian).mean
    extra = vib.project_rigid_motions(off_by * np.eye(len(hessian)), start)
    return evaluation.Reference(smiles, start, hessian, hessian + answer + extra)


def test_a_molecule_is_written_and_read_back(tmp_path: Path, loaded: release.Release) -> None:
    reference = _reference("c1ccccc1", loaded, off_by=0.0)
    evaluation.write(tmp_path / "benzene.npz", reference)
    again = evaluation.read(tmp_path / "benzene.npz")
    assert again.name == "benzene"
    assert again.geometry.symbols == reference.geometry.symbols
    np.testing.assert_array_equal(again.hessian, reference.hessian)
    np.testing.assert_array_equal(again.reference_hessian, reference.reference_hessian)


def test_the_first_order_shift_is_the_shift_of_a_small_change() -> None:
    start = geometry.start_from_smiles("c1ccccc1")
    hessian = _stable_hessian(start, seed=1)
    low = vib.vibrational_modes(hessian, start)
    change = vib.project_rigid_motions(1e-6 * random_hessian(start.n_atoms, 2), start)
    exact = vib.vibrational_modes(hessian + change, start).wavenumbers - low.wavenumbers
    first = evaluation.first_order_shifts(change, low, start.masses_amu)
    np.testing.assert_allclose(first, exact, atol=1e-5 * float(np.abs(exact).max()) + 1e-9)


def test_the_two_errors_are_different_quantities(loaded: release.Release) -> None:
    # The reference is exactly what the seeds' mean answers: the product has no error on
    # it. Each seed alone is still off, so the first-order error per seed is not zero.
    errors = evaluation.molecule_errors(_reference("c1ccccc1", loaded, off_by=0.0), loaded)
    assert errors.first_order.shape == (N_SEEDS, 30)
    np.testing.assert_allclose(errors.band, 0.0, atol=1e-6)
    assert float(np.abs(errors.first_order).max()) > 1.0
    np.testing.assert_allclose(errors.first_order.mean(axis=0), 0.0, atol=1e-6)
    assert sorted(errors.band_family) == sorted(errors.first_order_family)


def test_an_error_of_the_seeds_mean_is_seen_in_the_bands(loaded: release.Release) -> None:
    exact = evaluation.molecule_errors(_reference("c1ccccc1", loaded, off_by=0.0), loaded)
    off = evaluation.molecule_errors(_reference("c1ccccc1", loaded, off_by=0.01), loaded)
    assert float(np.abs(off.band).max()) > 1.0
    assert float(np.abs(exact.band).max()) < 1e-6


def _holdouts(folder: Path, loaded: release.Release) -> Path:
    for name, off_by in (("a", 0.0), ("b", 0.01)):
        (folder / name).mkdir(parents=True)
        for number, smiles in enumerate(SMILES):
            reference = _reference(smiles, loaded, off_by=off_by)
            evaluation.write(folder / name / f"{number:03d}.npz", reference)
    return folder


def test_a_folder_of_holdouts_is_measured_per_family(
    tmp_path: Path, loaded: release.Release
) -> None:
    measured = evaluation.evaluate(loaded, _holdouts(tmp_path / "holdouts", loaded))
    assert list(measured) == ["a", "b"]
    n_modes = sum(3 * geometry.start_from_smiles(smiles).n_atoms - 6 for smiles in SMILES)
    for name in ("a", "b"):
        holdout = measured[name]
        assert holdout["n_molecules"] == len(SMILES)
        assert list(holdout["band_error_seed_mean"]) == list(families.FAMILIES)
        counted = sum(entry["n_modes"] for entry in holdout["band_error_seed_mean"].values())
        assert counted == n_modes
        for family in families.FAMILIES:
            entry = holdout["band_error_seed_mean"][family]
            if not entry["n_modes"]:
                assert entry == {"rms_cm": None, "n_modes": 0, "n_molecules": 0}
                assert holdout["first_order_error_per_seed"][family] == [None] * N_SEEDS
                continue
            low, high = entry["interval_cm"]
            assert 0 <= low <= high
            assert 0 <= entry["share_of_resamples_within_line"] <= 1
            assert len(holdout["first_order_error_per_seed"][family]) == N_SEEDS
    with_modes = [
        f for f in families.FAMILIES if measured["a"]["band_error_seed_mean"][f]["n_modes"]
    ]
    assert len(with_modes) >= 2
    exact = [measured["a"]["band_error_seed_mean"][f]["rms_cm"] for f in with_modes]
    off = [measured["b"]["band_error_seed_mean"][f]["rms_cm"] for f in with_modes]
    assert max(exact) == 0.0
    assert min(off) > 0.5
    assert measured == evaluation.evaluate(loaded, tmp_path / "holdouts")
    assert json.loads(json.dumps(measured)) == measured


def test_the_interval_is_over_molecules_and_seeded() -> None:
    per_molecule = [np.full(10, 1.0), np.full(10, 2.0), np.full(10, 5.0)]
    low, high, share = evaluation.interval(per_molecule, line_cm=3.0)
    assert (low, high) == (1.0, 5.0)
    # within the line of 3 when no molecule of error 5 is drawn, (2/3)³, or when it is drawn
    # once beside the error-1 molecule twice, which is an rms of exactly 3: 3/27
    assert share == pytest.approx(11 / 27, abs=0.03)
    assert evaluation.interval(per_molecule, line_cm=3.0) == (low, high, share)
    alike = [np.full(10, 2.0)] * 3
    assert evaluation.interval(alike, line_cm=3.0) == (2.0, 2.0, 1.0)


@pytest.mark.parametrize("what", ["no hold-out", "empty hold-out", "half a molecule"])
def test_a_folder_that_holds_no_measurement_is_refused(
    what: str, tmp_path: Path, loaded: release.Release
) -> None:
    folder = tmp_path / "holdouts"
    folder.mkdir()
    if what != "no hold-out":
        (folder / "a").mkdir()
    if what == "half a molecule":
        np.savez(folder / "a" / "x.npz", symbols=np.array(["C"]), coords_bohr=np.zeros((1, 3)))
    said = {
        "no hold-out": "no test-set folder in it",
        "empty hold-out": "no molecule",
        "half a molecule": "no hessian, reference_hessian",
    }[what]
    with pytest.raises(RefusedError, match=said):
        evaluation.evaluate(loaded, folder)


def test_the_command_writes_what_it_measured_and_never_over_a_file(
    tmp_path: Path, loaded: release.Release, capsys: pytest.CaptureFixture[str]
) -> None:
    folder = _holdouts(tmp_path / "holdouts", loaded)
    release_folder = write_release(tmp_path / "release")
    out = tmp_path / "measured.json"
    asked = ["evaluate", str(folder), "--release", str(release_folder), "--out", str(out)]
    assert cli.main(asked) == cli.EXIT_DONE
    assert capsys.readouterr().out == f"done: 2 test set(s) measured, in {out}\n"
    assert json.loads(out.read_text(encoding="utf-8")) == evaluation.evaluate(loaded, folder)
    assert cli.main(asked) == cli.EXIT_REFUSED
    assert "exists: a result is never overwritten" in capsys.readouterr().err


def test_release_v13_is_licensed_on_what_it_delivers() -> None:
    bundled = release.load(release.bundled().parent / "v1.3")
    recorded = bundled.evaluation
    assert recorded.record_version == 3
    assert "chosen on 16 November 2026" in recorded.licence_basis
    assert all(recorded.licensed(family) for family in families.FAMILIES)
    assert recorded.band_error_seed_mean_cm["other"]["a"].interval_cm == (1.68, 3.76)
    assert recorded.first_order_error_per_seed_mean_cm["other"] == {"a": 3.62, "b": 3.89}
    start = geometry.start_from_smiles("c1ccccc1")
    admitted = admission.admit(admission.read("c1ccccc1"))
    hessian = _stable_hessian(start, seed=3)
    low = certificate_low_level(hessian)
    record = certificate.build(admitted, start, low, bundled.corrections(start, hessian), bundled)
    assert record["accuracy"]["licensed_families"] == list(families.FAMILIES)
    plain = (
        "'other' is within the licence line on test set (a) at 2.78 cm⁻¹, on 8 molecules: "
        "resampling the test set's molecules puts the error between 1.68 and 3.76 cm⁻¹, and 30% of "
        "the "
        "resamples lie above the line."
    )
    assert plain[0].upper() + plain[1:] in record["statement"]
    text = certificate.to_markdown(record)
    assert "| other | 2.78 [1.68–3.76], 156 bands | 2.77 [2.56–2.97], 1225 bands | yes |" in text
    assert "| other | 3.62 (4.09, 3.27, 3.51) | 3.89 (3.65, 4.51, 3.51) |" in text
    assert record["accuracy"]["on_thin_ground"] == ["other"]
    # v1.3's record moves molecules without a linear group by up to 3.07 cm⁻¹, above the
    # line, so no band of benzene is licensed; without the record the bands show the
    # families' licence, thin ground included.
    assert not any(band["licensed"] for band in record["bands"])
    unrecorded = replace(bundled, invariance=None)
    record = certificate.build(
        admitted, start, low, bundled.corrections(start, hessian), unrecorded
    )
    text = certificate.to_markdown(record)
    others = [band for band in record["bands"] if band["family"] == "other"]
    assert others
    assert text.count("| other | yes, on thin ground (see above) |") == len(others)
    assert "in cm⁻¹: what this certificate's positions are. In square brackets" in text
    assert "| ring-ip | yes |" in text
    assert "chosen on 16 November 2026" in text


def certificate_low_level(hessian: FloatArray) -> lowlevel.LowLevel:
    return lowlevel.LowLevel(
        energy=-1.0,
        hessian=hessian,
        polar_tensor=np.random.default_rng(1).normal(size=(3, len(hessian))),
        refinement=response.Refinement(1e-9, 1e-9, 0, 0),
        sum_rule_violation=1e-7,
        seconds=1.0,
    )


def test_what_was_measured_becomes_the_tables_of_a_record(
    tmp_path: Path, loaded: release.Release
) -> None:
    measured = evaluation.evaluate(loaded, _holdouts(tmp_path / "holdouts", loaded))
    missing = [
        family
        for family in families.FAMILIES
        if not measured["a"]["band_error_seed_mean"][family]["n_modes"]
    ]
    if missing:
        with pytest.raises(RefusedError, match=rf"hold-out \(a\) has no {missing[0]} band"):
            release.record_tables(measured)
        return
    tables = release.record_tables(measured)
    assert list(tables["band_error_seed_mean_cm"]) == list(families.FAMILIES)
    assert list(tables["first_order_error_per_seed_mean_cm"]["other"]) == ["a", "b"]


def test_a_family_counts_only_the_molecules_with_a_band_of_it() -> None:
    with_oop = evaluation.MoleculeErrors(
        band=np.array([1.0, 2.0]),
        band_family=("CH-oop", "other"),
        first_order=np.zeros((2, 2)),
        first_order_family=("CH-oop", "other"),
    )
    without = evaluation.MoleculeErrors(
        band=np.array([1.0]),
        band_family=("other",),
        first_order=np.zeros((2, 1)),
        first_order_family=("other",),
    )
    measured = evaluation.holdout_errors([with_oop, without, without], line_cm=3.0)
    band = measured["band_error_seed_mean"]
    assert measured["n_molecules"] == 3
    assert (band["CH-oop"]["n_modes"], band["CH-oop"]["n_molecules"]) == (1, 1)
    assert (band["other"]["n_modes"], band["other"]["n_molecules"]) == (3, 3)
    assert (band["ring-ip"]["n_modes"], band["ring-ip"]["n_molecules"]) == (0, 0)


def test_the_v13_record_is_what_the_tables_of_a_measurement_look_like() -> None:
    recorded = release.load(release.bundled().parent / "v1.3").evaluation
    measured = {
        name: {
            "n_molecules": holdout.n_molecules,
            "band_error_seed_mean": {
                family: recorded.band_error_seed_mean_cm[family][name].model_dump(mode="json")
                for family in families.FAMILIES
            },
            "first_order_error_per_seed": {
                family: recorded.first_order_error_per_seed_cm[family][name]
                for family in families.FAMILIES
            },
            "first_order_error_per_seed_mean": {
                family: recorded.first_order_error_per_seed_mean_cm[family][name]
                for family in families.FAMILIES
            },
        }
        for name, holdout in recorded.holdouts.items()
    }
    tables = release.record_tables(measured)
    again = recorded.model_dump(mode="json") | tables
    assert release.Evaluation.model_validate(again) == recorded


def test_a_molecule_at_no_minimum_of_the_low_level_is_refused(loaded: release.Release) -> None:
    reference = _reference("c1ccccc1", loaded, off_by=0.0)
    softest = vib.vibrational_modes(reference.hessian, reference.geometry).vectors[:, 0]
    direction = softest * vib.mass_weights(reference.geometry.masses_amu)
    unstable = reference.hessian - 2.0 * np.outer(direction, direction)
    saddle = evaluation.Reference(
        "saddle", reference.geometry, unstable, reference.reference_hessian
    )
    with pytest.raises(RefusedError, match="saddle: the low-level Hessian has 1 imaginary mode"):
        evaluation.molecule_errors(saddle, loaded)


def test_the_first_order_shift_of_an_imaginary_mode_has_the_sign_of_its_curvature() -> None:
    start = geometry.start_from_smiles("c1ccccc1")
    hessian = _stable_hessian(start, seed=1)
    softest = vib.vibrational_modes(hessian, start).vectors[:, 0]
    direction = softest * vib.mass_weights(start.masses_amu)
    unstable = hessian - 2.0 * np.outer(direction, direction)
    low = vib.vibrational_modes(unstable, start)
    assert low.wavenumbers[0] < 0
    stiffer = 1e-6 * np.outer(direction, direction)
    exact = vib.vibrational_modes(unstable + stiffer, start).wavenumbers[0] - low.wavenumbers[0]
    first = evaluation.first_order_shifts(stiffer, low, start.masses_amu)[0]
    assert exact > 0
    assert first == pytest.approx(exact, rel=1e-3)


@pytest.mark.parametrize("content", [b"PK\x03\x04 not a zip file at all", b""])
def test_a_damaged_molecule_file_is_refused_in_words(
    content: bytes, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "x.npz").write_bytes(content)
    asked = ["evaluate", str(tmp_path), "--release", str(release.bundled().parent / "v1.3")]
    assert cli.main(asked) == cli.EXIT_REFUSED
    assert "x.npz cannot be read as a test-set molecule" in capsys.readouterr().err


def test_the_command_checks_where_it_writes_before_it_measures(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "nowhere" / "measured.json"
    assert cli.main(["evaluate", str(tmp_path / "absent"), "--out", str(out)]) == cli.EXIT_REFUSED
    assert capsys.readouterr().err == f"refused:\n  {out.parent}: no such folder\n"


def test_a_family_at_the_line_is_within_it_in_every_resample() -> None:
    at_the_line = [np.full(5, 3.0)] * 3
    assert evaluation.interval(at_the_line, line_cm=3.0) == (3.0, 3.0, 1.0)


def test_an_unlicensed_family_stands_on_no_ground_to_lose(loaded: release.Release) -> None:
    recorded = loaded.evaluation
    assert not recorded.licensed("other")
    assert recorded.thin_ground("other") == {}
    assert recorded.caveats() == []
