"""The command line: what it prints and how it exits."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

import aromir
from aromir import cli, pipeline, release
from aromir.errors import ConvergenceError

if TYPE_CHECKING:
    from pathlib import Path


def test_check_admits_a_supported_molecule(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["check", "c1ccc2ccccc2c1"]) == cli.EXIT_DONE
    assert capsys.readouterr().out == "admitted: c1ccc2ccccc2c1 (C10H8, 18 atoms)\n"


def test_check_refuses_with_every_reason(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["check", "CC[O-]"]) == cli.EXIT_REFUSED
    printed = capsys.readouterr().err
    assert "net charge" in printed
    assert "no aromatic ring" in printed


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
