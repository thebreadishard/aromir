"""The command line: what it prints and how it exits."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

import aromir
from aromir import admission, certificate, cli, pipeline, release
from aromir.errors import ConvergenceError, InternalCoordinatesError

from support import write_release

if TYPE_CHECKING:
    from pathlib import Path


def test_check_admits_a_supported_molecule(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["check", "c1ccc2ccccc2c1"]) == cli.EXIT_DONE
    assert capsys.readouterr().out == "admitted: c1ccc2ccccc2c1 (C10H8, 18 atoms)\n"


def test_check_says_that_it_cleared_atom_map_numbers(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["check", "[cH:1]1ccccc1"]) == cli.EXIT_DONE
    assert capsys.readouterr().out == (
        f"admitted: c1ccccc1 (C6H6, 12 atoms)\nnote: {admission.ATOM_MAPS_CLEARED}\n"
    )


def test_check_refuses_with_every_reason(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["check", "CC[O-]"]) == cli.EXIT_REFUSED
    printed = capsys.readouterr().err
    assert "net charge" in printed
    assert "no aromatic ring" in printed


OCTYLNAPHTHALENE = "CCCCCCCCc1ccc2ccccc2c1"
"""C18H24, 42 atoms: above the bundled release's 8 to 40."""


def test_check_refuses_a_molecule_larger_than_the_training_sizes_as_spectrum_does(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["check", OCTYLNAPHTHALENE]) == cli.EXIT_REFUSED
    checked = capsys.readouterr().err
    assert "42 atoms: the network was trained on 8 to 40" in checked
    reached = []

    def compute(*_arguments: object, **_options: object) -> None:
        reached.append(True)

    monkeypatch.setattr(pipeline, "_compute", compute)
    out = tmp_path / "octylnaphthalene"
    assert cli.main(["spectrum", OCTYLNAPHTHALENE, "--out", str(out)]) == cli.EXIT_REFUSED
    assert capsys.readouterr().err == checked
    assert reached == []


def test_check_with_the_size_flag_admits_and_says_no_band_is_licensed(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert cli.main(["check", OCTYLNAPHTHALENE, "--outside-training-sizes"]) == cli.EXIT_DONE
    assert capsys.readouterr().out == (
        "admitted: CCCCCCCCc1ccc2ccccc2c1 (C18H24, 42 atoms)\n"
        "note: 42 atoms: the network was trained on 8 to 40, so no band of this molecule would "
        "be licensed; --outside-training-sizes computes it all the same\n"
    )


def test_check_refuses_a_molecule_the_network_cannot_read(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def unreadable(_self: release.Release, _geometry: object, _hessian: object) -> None:
        msg = "the primitives span more than 3N − 6 = 30 directions"
        raise InternalCoordinatesError(msg)

    monkeypatch.setattr(release.Release, "corrections", unreadable)
    assert cli.main(["check", "c1ccccc1"]) == cli.EXIT_REFUSED
    assert "the network cannot read this molecule's start geometry: the prim" in (
        capsys.readouterr().err
    )


def test_check_reads_the_release_it_is_given(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    folder = write_release(tmp_path / "release")
    assert cli.main(["check", "c1ccccc1", "--release", str(folder)]) == cli.EXIT_DONE
    assert capsys.readouterr().out == "admitted: c1ccccc1 (C6H6, 12 atoms)\n"
    assert cli.main(["check", "c1ccoc1", "--release", str(folder)]) == cli.EXIT_REFUSED
    assert "9 atoms: the network was trained on 10 to 30" in capsys.readouterr().err


def test_an_invalid_setting_is_refused_before_anything_runs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = cli.main(["baseline", "c1ccccc1", "--out", str(tmp_path / "x"), "--threads", "0"])
    assert exit_code == cli.EXIT_REFUSED
    assert "greater than or equal to 1" in capsys.readouterr().err
    assert not (tmp_path / "x").exists()


def test_baseline_reports_where_the_result_is(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    requests = []

    def run(request: pipeline.Request) -> dict[str, Any]:
        requests.append(request)
        return {"harmonic": {"bands": [{}] * 30}}

    monkeypatch.setattr(pipeline, "run_baseline", run)
    out = tmp_path / "benzene"
    exit_code = cli.main(["baseline", "c1ccccc1", "--out", str(out), "--max-memory-mb", "8000"])
    assert exit_code == cli.EXIT_DONE
    assert capsys.readouterr().out == f"done: 30 bands in {out / 'baseline.json'}\n"
    assert requests[0].resources.max_memory_mb == 8000
    assert requests[0].molecule.smiles == "c1ccccc1"


def test_a_failed_calculation_exits_with_its_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def run(_request: pipeline.Request) -> dict[str, Any]:
        message = "the b3lyp SCF did not converge"
        raise ConvergenceError(message)

    monkeypatch.setattr(pipeline, "run_baseline", run)
    assert cli.main(["baseline", "c1ccccc1", "--out", str(tmp_path / "x")]) == cli.EXIT_FAILED
    assert capsys.readouterr().err == "failed: the b3lyp SCF did not converge\n"


def test_a_molecule_must_be_given(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        cli.main(["check"])
    assert "required" in capsys.readouterr().err


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        cli.main(["--version"])
    assert capsys.readouterr().out == f"aromir {aromir.__version__}\n"


def test_spectrum_reports_the_licensed_bands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    loaded_from = []

    def load(folder: Path) -> str:
        loaded_from.append(folder)
        return "the release"

    def run(_request: pipeline.Request, loaded: str) -> dict[str, Any]:
        assert loaded == "the release"
        notes = ("", "", "the family's error is above the licence line")
        return {"bands": [{"licensed": not note, "licence_note": note} for note in notes]}

    monkeypatch.setattr(release, "load", load)
    monkeypatch.setattr(pipeline, "run_spectrum", run)
    out = tmp_path / "benzene"
    arguments = ["spectrum", "c1ccccc1", "--out", str(out), "--release", str(tmp_path / "r")]
    assert cli.main(arguments) == cli.EXIT_DONE
    assert loaded_from == [tmp_path / "r"]
    assert capsys.readouterr().out == (
        f"done: 3 bands, 2 of them licensed; certificate in {out / 'certificate.md'}\n"
    )


def test_a_spectrum_with_a_flat_coordinate_is_done_and_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # A flat band's corrected number may be negative: it is no band position, so the run
    # is done, and the line says what the certificate says.
    def run(_request: pipeline.Request, _loaded: str) -> dict[str, Any]:
        notes = (certificate.FLAT_COORDINATE, "")
        return {"bands": [{"licensed": not note, "licence_note": note} for note in notes]}

    monkeypatch.setattr(release, "load", lambda _folder: "the release")
    monkeypatch.setattr(pipeline, "run_spectrum", run)
    out = tmp_path / "vinylfuran"
    arguments = ["spectrum", "C=Cc1ccoc1", "--out", str(out), "--release", str(tmp_path / "r")]
    assert cli.main(arguments) == cli.EXIT_DONE
    assert capsys.readouterr().out == (
        f"done: 2 bands, 1 of them licensed; certificate in {out / 'certificate.md'}; "
        "1 band(s) on a flat coordinate, which are no band positions\n"
    )


def test_a_flat_coordinate_beside_an_imaginary_band_is_unphysical_and_names_both(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def run(_request: pipeline.Request, _loaded: str) -> dict[str, Any]:
        notes = (certificate.FLAT_COORDINATE, certificate.IMAGINARY, "")
        return {"bands": [{"licensed": not note, "licence_note": note} for note in notes]}

    monkeypatch.setattr(release, "load", lambda _folder: "the release")
    monkeypatch.setattr(pipeline, "run_spectrum", run)
    out = tmp_path / "both"
    arguments = ["spectrum", "C=Cc1ccoc1", "--out", str(out), "--release", str(tmp_path / "r")]
    assert cli.main(arguments) == cli.EXIT_UNPHYSICAL
    said = capsys.readouterr()
    assert said.out == ""
    assert said.err == (
        "unphysical: 1 corrected band(s) have an imaginary position; 3 bands, 1 of them "
        f"licensed; certificate in {out / 'certificate.md'}; 1 band(s) on a flat "
        "coordinate, which are no band positions\n"
    )


def test_without_a_release_named_the_one_inside_the_package_is_used(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loaded_from = []

    def load(folder: Path) -> str:
        loaded_from.append(folder)
        return "the release"

    def run(_request: pipeline.Request, _loaded: str) -> dict[str, Any]:
        return {"bands": []}

    monkeypatch.setattr(release, "load", load)
    monkeypatch.setattr(pipeline, "run_spectrum", run)
    assert cli.main(["spectrum", "c1ccccc1", "--out", str(tmp_path / "x")]) == cli.EXIT_DONE
    assert loaded_from == [release.bundled()]
