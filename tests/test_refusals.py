"""Inputs that are refused with a reason, and failures that never leave as a traceback."""

from __future__ import annotations

import json
import shutil
from typing import TYPE_CHECKING, Any

import pytest
from rdkit import Chem
from rdkit.Chem import rdDistGeom

from aromir import admission, cli, pipeline, release
from aromir.errors import RefusedError

from support import BENZENE_XYZ, TRAINING, write_release

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


@pytest.mark.parametrize(
    ("smiles", "label"),
    [
        ("[2H]c1ccccc1", "2H"),
        ("[2H]c1c([2H])c([2H])c([2H])c([2H])c1[2H]", "2H"),
        ("[13cH]1ccccc1", "13C"),
        ("[3H]c1ccccc1", "3H"),
    ],
)
def test_an_isotope_label_is_refused(smiles: str, label: str) -> None:
    with pytest.raises(RefusedError, match=rf"isotope label\(s\) {label}: every atom is computed"):
        admission.admit(admission.parse_smiles(smiles))


@pytest.mark.parametrize("smiles", ["c1ccccc1 benzene", "c1ccccc1\n", " c1ccccc1", "c1ccccc1é", ""])
def test_a_smiles_with_anything_beside_it_is_refused(smiles: str) -> None:
    with pytest.raises(RefusedError, match="is not a SMILES alone"):
        admission.parse_smiles(smiles)


def test_an_xyz_file_without_atoms_is_told_so(tmp_path: Path) -> None:
    path = tmp_path / "empty.xyz"
    path.write_text("0\nnothing\n", encoding="utf-8")
    with pytest.raises(RefusedError) as refusal:
        admission.admit(admission.parse_xyz(path))
    assert refusal.value.reasons in (
        ("the input holds no atom",),
        (f"{path} does not parse as an xyz file",),
    )


def test_an_xyz_file_whose_comment_is_not_utf8_is_read(tmp_path: Path) -> None:
    path = tmp_path / "latin1.xyz"
    path.write_bytes(BENZENE_XYZ.replace("benzene", "benzène, Å").encode("latin-1"))
    assert admission.admit(admission.parse_xyz(path)).formula == "C6H6"


def test_an_xyz_geometry_names_its_double_bonds_isomer(tmp_path: Path) -> None:
    from aromir import geometry  # noqa: PLC0415  only this test embeds a molecule
    from aromir.constants import ANGSTROM_PER_BOHR  # noqa: PLC0415

    start = geometry.start_from_smiles("C(=C/c1ccccc1)\\c1ccccc1")
    lines = [str(start.n_atoms), "E-stilbene"]
    for symbol, position in zip(start.symbols, start.coords_bohr * ANGSTROM_PER_BOHR, strict=True):
        lines.append(f"{symbol} {position[0]:.8f} {position[1]:.8f} {position[2]:.8f}")
    path = tmp_path / "stilbene.xyz"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    admitted = admission.admit(admission.parse_xyz(path))
    assert admitted.unassigned_stereo == ()
    # RDKit writes the E isomer's two bond marks either way round; the isomer is what counts.
    double_bonds = [
        bond.GetStereo()
        for bond in Chem.MolFromSmiles(admitted.smiles).GetBonds()
        if bond.GetStereo() != Chem.BondStereo.STEREONONE
    ]
    assert double_bonds == [Chem.BondStereo.STEREOE]


def _main(arguments: list[str], capsys: pytest.CaptureFixture[str]) -> tuple[int, str]:
    code = cli.main(arguments)
    return code, capsys.readouterr().err


def test_no_start_geometry_is_a_refusal_on_the_command_line(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(rdDistGeom, "EmbedMolecule", lambda _molecule, **_options: -1)
    code, error = _main(["check", "c1ccccc1"], capsys)
    assert code == cli.EXIT_REFUSED
    assert error == (
        "refused:\n  RDKit finds no 3D start geometry for c1ccccc1: give the molecule as an "
        "xyz file\n"
    )


@pytest.mark.parametrize("command", ["baseline", "spectrum"])
def test_out_naming_a_file_is_refused(
    command: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "a_file"
    out.write_text("x", encoding="utf-8")
    code, error = _main([command, "c1ccccc1", "--out", str(out)], capsys)
    assert code == cli.EXIT_REFUSED
    assert error == f"refused:\n  {out} is a file: --out names a folder\n"


def test_a_molecule_outside_the_training_sizes_needs_the_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    folder = write_release(tmp_path / "release")
    asked = ["spectrum", "c1ccoc1", "--out", str(tmp_path / "furan"), "--release", str(folder)]
    code, error = _main(asked, capsys)
    assert code == cli.EXIT_REFUSED
    assert error == (
        "refused:\n  9 atoms: the network was trained on 10 to 30, so no band of this molecule "
        "would be licensed; --outside-training-sizes computes it all the same\n"
    )
    assert not (tmp_path / "furan").exists()
    reached = []

    def compute(*_arguments: object, **_options: object) -> None:
        reached.append(True)
        raise RefusedError(["stopped at the first computation"])

    monkeypatch.setattr(pipeline, "_compute", compute)
    code, error = _main([*asked, "--outside-training-sizes"], capsys)
    assert reached == [True]
    assert "stopped at the first computation" in error


def _spectrum_with(
    notes: tuple[str, ...], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> list[str]:
    def run(_request: pipeline.Request, _loaded: str) -> dict[str, Any]:
        return {"bands": [{"licensed": not note, "licence_note": note} for note in notes]}

    monkeypatch.setattr(release, "load", lambda _folder: "the release")
    monkeypatch.setattr(pipeline, "run_spectrum", run)
    return ["spectrum", "c1ccccc1", "--out", str(tmp_path / "out")]


def test_an_imaginary_corrected_band_is_not_done(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    arguments = _spectrum_with(("imaginary position", "", ""), monkeypatch, tmp_path)
    assert cli.main(arguments) == cli.EXIT_UNPHYSICAL
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith(
        "unphysical: 1 corrected band(s) have an imaginary position; 3 bands, 2 of them licensed"
    )


def test_a_band_imaginary_for_one_seed_only_is_unlicensed_but_done(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    arguments = _spectrum_with(("imaginary for at least one seed", ""), monkeypatch, tmp_path)
    assert cli.main(arguments) == cli.EXIT_DONE
    assert capsys.readouterr().out.startswith("done: 2 bands, 1 of them licensed")


def test_the_card_is_not_written_over_a_file_or_into_a_missing_folder(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    existing = tmp_path / "CARD.md"
    existing.write_text("mine", encoding="utf-8")
    code, error = _main(["card", "--out", str(existing)], capsys)
    assert code == cli.EXIT_REFUSED
    assert error == f"refused:\n  {existing} exists: a result is never overwritten\n"
    assert existing.read_text(encoding="utf-8") == "mine"
    missing = tmp_path / "nowhere" / "CARD.md"
    code, error = _main(["card", "--out", str(missing)], capsys)
    assert code == cli.EXIT_REFUSED
    assert error == f"refused:\n  {missing.parent}: no such folder\n"


def _truncate(folder: Path) -> None:
    data = (folder / "seed-1.safetensors").read_bytes()
    (folder / "seed-1.safetensors").write_bytes(data[: len(data) // 2])


def _damage_with_matching_hash(folder: Path) -> None:
    import hashlib  # noqa: PLC0415  only this damage needs it

    data = (folder / "seed-1.safetensors").read_bytes()[:2000]
    (folder / "seed-1.safetensors").write_bytes(data)
    recipe = json.loads((folder / "seed-1.json").read_text(encoding="utf-8"))
    recipe["sha256"] = hashlib.sha256(data).hexdigest()
    (folder / "seed-1.json").write_text(json.dumps(recipe), encoding="utf-8")


def _latin1_record(folder: Path) -> None:
    data = (folder / "evaluation.json").read_bytes()
    damaged = data.replace(b"network_version", "network_vérsion".encode("latin-1"))
    (folder / "evaluation.json").write_bytes(damaged)


def _edit(name: str, change: Callable[[dict[str, Any]], None]) -> Callable[[Path], None]:
    def damage(folder: Path) -> None:
        record = json.loads((folder / name).read_text(encoding="utf-8"))
        change(record)
        (folder / name).write_text(json.dumps(record), encoding="utf-8")

    return damage


def _wider_network(record: dict[str, Any]) -> None:
    record["recipe"]["architecture"]["n_features"] *= 2


def _negative_error(record: dict[str, Any]) -> None:
    record["errors_cm"]["other"]["a"] = -1.0


def _too_many(record: dict[str, Any]) -> None:
    record["holdouts"]["b"]["n_molecules"] = 3900


def _another_holdout(record: dict[str, Any]) -> None:
    record["held_out_smiles"]["c"] = ["c1ccncc1"]


@pytest.mark.parametrize(
    ("damage", "said"),
    [
        (_truncate, "seed-1.safetensors has sha256"),
        (lambda folder: (folder / "seed-0.safetensors").unlink(), "seed-0.safetensors"),
        (_damage_with_matching_hash, "seed-1.json: "),
        (_latin1_record, "evaluation.json cannot be read as UTF-8 text"),
        (_edit("seed-0.json", _wider_network), "seed-0.json: Error(s) in loading state_dict"),
        (
            _edit("training.json", lambda record: record.update(elements=[])),
            "training.json: elements",
        ),
        (lambda folder: (folder / "invariance.json").mkdir(), "invariance.json cannot be read"),
        (_edit("evaluation.json", _negative_error), "an rms error is negative or not a finite"),
        (_edit("evaluation.json", _too_many), "counts 3900 molecules, training.json lists 1"),
        (_edit("training.json", _another_holdout), "has the hold-outs ['a', 'b'], training.json"),
        (
            _edit("training.json", lambda record: record.update(elements=["C", "H", "Xx"])),
            "element(s) Xx that the package does not know",
        ),
        (
            _edit("training.json", lambda record: record.update(min_atoms=40, max_atoms=3)),
            "the smallest molecule has 40 atoms, the largest 3",
        ),
        (
            _edit("seed-0.json", lambda record: record["recipe"].update(seed=7)),
            "seed-0.json holds the recipe of seed 7",
        ),
    ],
)
def test_a_release_that_is_damaged_or_contradicts_itself_is_refused_in_one_line(
    damage: Callable[[Path], None],
    said: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    folder = write_release(tmp_path / "release")
    damage(folder)
    code, error = _main(["card", "--release", str(folder)], capsys)
    assert code == cli.EXIT_REFUSED
    assert error.startswith("refused:\n  ")
    assert said in error
    assert "Traceback" not in error


def test_the_bundled_release_passes_its_own_cross_checks(tmp_path: Path) -> None:
    copy = tmp_path / "v1.3"
    shutil.copytree(release.bundled(), copy)
    loaded = release.load(copy)
    for name, holdout in loaded.evaluation.holdouts.items():
        assert holdout.n_molecules == loaded.training.n_held_out(name)
    assert set(TRAINING) == set(loaded.training.model_dump())


def test_an_error_nobody_foresaw_is_one_line_with_its_kind(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def broken(_molecule: pipeline.MoleculeInput) -> None:
        msg = "the disk is gone"
        raise OSError(msg)

    monkeypatch.setattr(pipeline, "admit", broken)
    code, error = _main(["check", "c1ccccc1"], capsys)
    assert code == cli.EXIT_FAILED
    assert error == "failed: OSError: the disk is gone\n"


def test_the_xyz_file_of_an_amidine_is_admitted_and_ordered(tmp_path: Path) -> None:
    # RDKit keeps the imine hydrogen as the mark of the C=N bond's stereo when it removes
    # the others; the heavy-atom skeleton the atom order is built on must not hold it.
    from aromir import geometry  # noqa: PLC0415  only this test embeds a molecule
    from aromir.constants import ANGSTROM_PER_BOHR  # noqa: PLC0415

    start = geometry.start_from_smiles("N=C(N)c1ccccc1")
    lines = [str(start.n_atoms), "benzamidine"]
    for symbol, position in zip(start.symbols, start.coords_bohr * ANGSTROM_PER_BOHR, strict=True):
        lines.append(f"{symbol} {position[0]:.8f} {position[1]:.8f} {position[2]:.8f}")
    path = tmp_path / "benzamidine.xyz"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    admitted, ordered = pipeline.admit(pipeline.MoleculeInput(xyz=path))
    assert admitted.smiles == "N=C(N)c1ccccc1"
    assert admitted.unassigned_stereo == ()
    n_heavy = sum(symbol != "H" for symbol in ordered.symbols)
    assert set(ordered.symbols[:n_heavy]) == {"C", "N"}
    assert set(ordered.symbols[n_heavy:]) == {"H"}


@pytest.mark.parametrize("error", [ValueError("a shape is wrong"), ZeroDivisionError("by zero")])
def test_a_numerical_error_is_one_line_unless_the_traceback_is_asked_for(
    error: Exception, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def broken(_molecule: pipeline.MoleculeInput) -> None:
        raise error

    monkeypatch.setattr(pipeline, "admit", broken)
    monkeypatch.delenv(cli.TRACEBACK_VARIABLE, raising=False)
    code, said = _main(["check", "c1ccccc1"], capsys)
    assert code == cli.EXIT_FAILED
    assert said == f"failed: {type(error).__name__}: {error}\n"
    monkeypatch.setenv(cli.TRACEBACK_VARIABLE, "1")
    with pytest.raises(type(error)):
        cli.main(["check", "c1ccccc1"])


def test_an_error_in_the_code_itself_is_not_dressed_as_a_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def bug(_molecule: pipeline.MoleculeInput) -> None:
        msg = "a key the code forgot"
        raise KeyError(msg)

    monkeypatch.setattr(pipeline, "admit", bug)
    with pytest.raises(KeyError):
        cli.main(["check", "c1ccccc1"])


def test_an_imine_isomer_named_through_its_hydrogen_is_said_to_be_left_open() -> None:
    # The admitted SMILES is written without the hydrogen that carries the stereo mark,
    # and the start geometry is embedded from that SMILES: the isomer is not what the
    # user wrote, and the certificate must not pass over it.
    for written in ("[H]/N=C(/C)c1ccccc1", "[H]/N=C(\\C)c1ccccc1"):
        admitted = admission.admit(admission.parse_smiles(written))
        assert admitted.smiles == "CC(=N)c1ccccc1"
        assert admitted.unassigned_stereo == ("1 double bond",)
