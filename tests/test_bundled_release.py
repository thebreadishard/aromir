"""The release inside the package: it loads, is the version named, and its card is its own."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem

from aromir import card, cli, release
from aromir.molecule import Geometry

N_SEEDS = 3
PINS = Path(__file__).parent / "data" / "release"
PLATFORM_PINS = (
    ("1.3", "delta_h_biphenyl_cn_v1.3.npz", "delta_h_windows"),
    ("1.4", "delta_h_biphenyl_cn_v1.4.npz", "delta_h_linux"),
)
"""Each shipped release's ΔH for 4-cyanobiphenyl, computed on the platform the key names."""
RELEASE_CHECKPOINTS = {
    "E7_rungC_v14s3clean_c00_control_release_model_n1132_seed0.pt": "6f52c9d8570fbcf5",
    "E7_rungC_v14s3clean_c00_control_release_model_n1132_seed1.pt": "1e820b8755078b90",
    "E7_rungC_v14s3clean_c00_control_release_model_n1132_seed2.pt": "bc4de0b6458e44f2",
}
MEASUREMENT_CHECKPOINTS = {
    "E7_rungC_v14s3clean_c00_control_model_n1082_seed0.pt": "89574ed1b69c4e10",
    "E7_rungC_v14s3clean_c00_control_model_n1082_seed1.pt": "03dbbf9f29b59186",
    "E7_rungC_v14s3clean_c00_control_model_n1082_seed2.pt": "8dd85f3f1ba5bbeb",
}
"""The training checkpoints v1.4 was converted from, as the training project's certificate of
benzene on 18 November 2026 names them."""
DELTA_H_TOLERANCE = 1e-6
"""hartree/bohr². Single-precision arithmetic on another build: Linux and Windows differ by
7e-9 for this molecule, whose largest ΔH element is 0.040, and by 2.4e-7 at most over 76
molecules of the training corpus."""


def test_the_bundled_release_loads_as_the_version_the_package_names() -> None:
    loaded = release.load(release.bundled())
    assert loaded.network_version == release.BUNDLED_VERSION
    assert [recipe.seed for recipe in loaded.recipes] == list(range(N_SEEDS))
    assert loaded.training.network_version == release.BUNDLED_VERSION


def test_each_weight_file_is_the_one_its_recipe_records() -> None:
    loaded = release.load(release.bundled())
    for seed, recorded in zip(range(N_SEEDS), loaded.tensor_sha256, strict=True):
        tensors = release.bundled() / f"seed-{seed}.safetensors"
        assert hashlib.sha256(tensors.read_bytes()).hexdigest() == recorded


def test_the_card_command_without_a_release_prints_the_shipped_card(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert cli.main(["card"]) == cli.EXIT_DONE
    shipped = (release.bundled() / "MODEL_CARD.md").read_text(encoding="utf-8")
    assert capsys.readouterr().out == shipped


def test_the_shipped_model_card_is_the_one_the_release_writes() -> None:
    shipped = (release.bundled() / "MODEL_CARD.md").read_text(encoding="utf-8")
    assert shipped == card.to_markdown(release.load(release.bundled()))


@pytest.mark.parametrize(("version", "name", "key"), PLATFORM_PINS)
def test_each_release_corrects_a_twisted_biphenyl_alike_on_every_platform(
    version: str, name: str, key: str
) -> None:
    # The pin is the Windows build's ΔH for 4-cyanobiphenyl. Before
    # `primitives.pseudo_inverse` this failed on numpy 2.5.4's Linux wheels: a null singular
    # value of this molecule's B matrix came back above `np.linalg.pinv`'s default cutoff
    # and every seed's ΔH was 1e29. A numerical step that depends on the platform shows
    # here as a difference above single-precision noise.
    with np.load(PINS / name) as pin:
        geometry = Geometry(tuple(str(symbol) for symbol in pin["symbols"]), pin["coords_bohr"])
        hessian, pinned = pin["hessian"], pin[key]
    folder = release.bundled().parent / f"v{version}"
    corrections = release.load(folder).corrections(geometry, hessian)
    assert len(corrections.per_seed) == len(pinned) == N_SEEDS
    for ours, theirs in zip(corrections.per_seed, pinned, strict=True):
        assert 0.03 < float(np.abs(theirs).max()) < 0.05
        assert float(np.abs(ours - theirs).max()) <= DELTA_H_TOLERANCE


def test_the_bundled_release_ships_release_networks_and_names_both_kinds() -> None:
    loaded = release.load(release.bundled())
    assert loaded.shipped is not None
    assert loaded.measurement is not None
    assert loaded.shipped.release_checkpoints == RELEASE_CHECKPOINTS
    assert loaded.shipped.measurement_checkpoints == MEASUREMENT_CHECKPOINTS
    assert loaded.shipped.n_entries == 1132
    seeds = {
        f"seed-{recipe.seed}": sha
        for recipe, sha in zip(loaded.recipes, loaded.tensor_sha256, strict=True)
    }
    assert loaded.shipped.seed_sha256 == seeds
    assert loaded.shipped.fitted_share == 0.85
    assert set(loaded.shipped.checks) == {
        "same_recipe",
        "pool_agreement",
        "holdouts_learnt",
        "early_stopping",
        "benzene",
    }
    held_out = {s for smiles in loaded.measurement.held_out_smiles.values() for s in smiles}
    assert held_out <= set(loaded.training.trained_smiles)
    assert set(loaded.measurement.trained_smiles) <= set(loaded.training.trained_smiles)


def test_the_bundled_release_networks_have_no_error_of_their_own(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "a").mkdir()
    assert cli.main(["evaluate", str(tmp_path)]) == cli.EXIT_REFUSED
    assert "their error there is no measurement" in capsys.readouterr().err


def test_v14s_caveats_count_the_molecules_with_a_ch_oop_band() -> None:
    caveats = release.load(release.bundled()).evaluation.caveats()
    assert any(
        caveat.startswith("'CH-oop' is within the licence line on hold-out (a) at ")
        and "on 5 of the hold-out's 8 molecules (the others have no 'CH-oop' band): " in caveat
        for caveat in caveats
    )


def test_the_evaluation_record_is_the_measurement_that_ships_beside_it() -> None:
    folder = release.bundled()
    measured = json.loads((folder / release.MEASURED_FILE).read_text(encoding="utf-8"))
    recorded = release.load(folder).evaluation.model_dump(mode="json")
    for field, table in release.record_tables(measured).items():
        assert recorded[field] == table
    for name, holdout in measured.items():
        assert recorded["holdouts"][name]["n_molecules"] == holdout["n_molecules"]


def test_every_smiles_of_the_release_is_in_rdkits_present_canonical_form() -> None:
    loaded = release.load(release.bundled())
    listed = [
        *(
            s
            for training in (loaded.training, loaded.measured_training)
            for s in training.trained_smiles
        ),
        *(s for held in loaded.measured_training.held_out_smiles.values() for s in held),
        *(loaded.training.cation_parent_smiles or ()),
    ]
    assert len(listed) > 700
    rewritten = [s for s in listed if Chem.MolToSmiles(Chem.MolFromSmiles(s)) != s]
    assert rewritten == []
