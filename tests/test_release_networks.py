"""Release networks beside measurement networks: one is shipped, the other has the errors."""

from __future__ import annotations

import dataclasses
import json
import re
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from aromir import admission, card, certificate, evaluation, geometry, lowlevel, release, response
from aromir.errors import RefusedError
from aromir.network import weights

from support import TRAINING, measured_of, random_hessian, write_release

if TYPE_CHECKING:
    from pathlib import Path

    from aromir.molecule import Geometry

CATION_TEST = {
    "molecule": "benzene",
    "without_cm": 1.47,
    "with_cm": 5.37,
    "n_networks": 1,
    "source": "a made-up test",
}
RELEASE_TRAINING: dict[str, Any] = TRAINING | {
    "trained_smiles": ["Cc1ccccc1", "c1ccc2ccccc2c1", "c1ccccc1"],
    "held_out_smiles": {},
    "cation_parent_smiles": ["c1ccncc1"],
    "cation_test": CATION_TEST,
}
"""The release networks trained on the measurement networks' naphthalene, on both hold-out
molecules and on pyridine's radical cation, not on pyridine itself."""
SHIPPED: dict[str, Any] = {
    "network_version": "1.3",
    "n_entries": 4,
    "fitted_share": 0.85,
    "checks": dict.fromkeys(release.RELEASE_CHECKS, True),
    "release_checkpoints": {"release_seed0.pt": "0" * 16, "release_seed1.pt": "1" * 16},
    "measurement_checkpoints": {"measured_seed0.pt": "2" * 16, "measured_seed1.pt": "3" * 16},
    "source": "a made-up record",
}
MEASURED_BY = "a made-up measurement of measured_seed0.pt and measured_seed1.pt"
"""The evaluation's source that names the measurement checkpoints."""


def write_pair(
    folder: Path,
    training: dict[str, Any] | None = None,
    shipped: dict[str, Any] | None = None,
    measurement: dict[str, Any] | None = None,
) -> Path:
    """Write a folder of release networks for ``TRAINING``'s measurement networks."""
    write_release(folder)
    path = folder / release.EVALUATION_FILE
    evaluation_record = json.loads(path.read_text(encoding="utf-8"))
    evaluation_record["sources"] = [*evaluation_record["sources"], MEASURED_BY]
    path.write_text(json.dumps(evaluation_record), encoding="utf-8")
    (folder / release.MEASURED_FILE).write_text(
        json.dumps(measured_of(evaluation_record)), encoding="utf-8"
    )
    (folder / release.MEASUREMENT_FILE).write_text(
        json.dumps(TRAINING if measurement is None else measurement), encoding="utf-8"
    )
    record = RELEASE_TRAINING if training is None else training
    (folder / release.TRAINING_FILE).write_text(json.dumps(record), encoding="utf-8")
    hashes = {path.stem: weights.recorded_sha256(path) for path in folder.glob("seed-*.json")}
    bound = {name: release.record_sha256(folder / name) for name in release.RECORDED_FILES}
    written = {"seed_sha256": hashes, "record_sha256": bound} | SHIPPED | (shipped or {})
    (folder / release.RELEASE_FILE).write_text(json.dumps(written))
    return folder


@pytest.fixture(scope="module")
def pair(tmp_path_factory: pytest.TempPathFactory) -> release.Release:
    return release.load(write_pair(tmp_path_factory.mktemp("pair")))


def test_a_pair_loads_with_the_errors_on_the_measurement_networks(pair: release.Release) -> None:
    assert pair.shipped is not None
    assert pair.measurement is not None
    assert pair.measured_training is pair.measurement
    assert pair.measured_training.held_out_smiles == TRAINING["held_out_smiles"]
    assert pair.training.trained_smiles == RELEASE_TRAINING["trained_smiles"]


def test_a_folder_of_measured_networks_has_no_pair(tmp_path: Path) -> None:
    loaded = release.load(write_release(tmp_path))
    assert loaded.shipped is None
    assert loaded.measurement is None
    assert loaded.measured_training is loaded.training
    assert release.made_by(loaded) == "with the error measured on molecules the network never saw."


def test_the_sentence_says_what_made_the_spectrum_and_what_the_errors_are_of(
    pair: release.Release,
) -> None:
    sentence = release.made_by(pair)
    assert sentence.startswith(
        "The spectrum is made by the release networks of version 1.3, trained on the 4 "
        "labelled entries of their training pool, the hold-out molecules included (3 distinct "
        "neutral molecules by SMILES and the radical cations of 1 molecule): each of the 2 "
        "networks is fitted on a random 85 % of the entries and stopped on the other 15 %, a "
        "different 15 % per seed,"
    )
    assert f"(with 2 networks about 0.15² ≈ {100 * (1 - 0.85) ** 2:.1f} % by none)" in sentence
    assert sentence.endswith(
        "the release networks themselves have seen the hold-out molecules, so they have no "
        "held-out error of their own."
    )


@pytest.mark.parametrize("present", [release.RELEASE_FILE, release.MEASUREMENT_FILE])
def test_one_record_of_the_pair_without_the_other_is_refused(tmp_path: Path, present: str) -> None:
    folder = write_pair(tmp_path)
    other = release.MEASUREMENT_FILE if present == release.RELEASE_FILE else release.RELEASE_FILE
    (folder / other).unlink()
    with pytest.raises(RefusedError, match=f"{present} without {other}"):
        release.load(folder)


def test_release_networks_that_failed_a_check_are_not_shipped(tmp_path: Path) -> None:
    failed = SHIPPED | {"checks": SHIPPED["checks"] | {"early_stopping": False}}
    with pytest.raises(RefusedError, match="release check\\(s\\) early_stopping failed"):
        release.load(write_pair(tmp_path, shipped=failed))


@pytest.mark.parametrize(
    ("training", "shipped", "reason"),
    [
        (
            RELEASE_TRAINING | {"trained_smiles": ["c1ccccc1"]},
            SHIPPED,
            "the release networks lack 1 molecule\\(s\\) the measurement networks trained on",
        ),
        (
            RELEASE_TRAINING | {"held_out_smiles": {"a": ["c1ccccc1"]}},
            SHIPPED,
            "release networks hold nothing out",
        ),
        (
            RELEASE_TRAINING,
            SHIPPED | {"release_checkpoints": {"release_seed0.pt": "0" * 16}},
            "names 1 checkpoint\\(s\\), the folder holds 2 seed\\(s\\)",
        ),
        (RELEASE_TRAINING, SHIPPED | {"network_version": "1.4"}, "is of version \\['1.4'\\]"),
        (
            RELEASE_TRAINING | {"network_version": "1.4"},
            SHIPPED,
            "is of version \\['1.4'\\], the evaluation of 1.3",
        ),
        (
            RELEASE_TRAINING | {"trained_smiles": ["c1ccc2ccccc2c1", "c1ccccc1"]},
            SHIPPED,
            "the release networks lack 1 hold-out molecule\\(s\\)",
        ),
        (
            RELEASE_TRAINING,
            SHIPPED | {"seed_sha256": {"seed-0": "0" * 64, "seed-1": "1" * 64}},
            "names other weights than the seeds beside it",
        ),
        (
            RELEASE_TRAINING | {"soft_mode_repair": True},
            SHIPPED,
            "measurement.json gives soft_mode_repair False, training.json True",
        ),
        (
            RELEASE_TRAINING | {"max_atoms": TRAINING["max_atoms"] + 1},
            SHIPPED,
            "measurement.json gives max_atoms",
        ),
        (
            RELEASE_TRAINING | {"elements": [*TRAINING["elements"], "S"]},
            SHIPPED,
            "measurement.json gives elements",
        ),
        (
            RELEASE_TRAINING | {"min_atoms": TRAINING["min_atoms"] - 1},
            SHIPPED,
            "measurement.json gives min_atoms",
        ),
        (RELEASE_TRAINING, SHIPPED | {"n_entries": 2}, "counts 2 entries, fewer than the 4"),
        (
            RELEASE_TRAINING,
            SHIPPED | {"measurement_checkpoints": {"a.pt": "2" * 16, "b.pt": "3" * 16}},
            "does not name a.pt, b.pt as its source",
        ),
        (
            RELEASE_TRAINING,
            SHIPPED | {"checks": {"same_recipe": True}},
            "release check\\(s\\): \\[.*'benzene'.*\\] missing",
        ),
        (
            RELEASE_TRAINING,
            SHIPPED
            | {"release_checkpoints": {"release_seed0.pt": "zz", "release_seed1.pt": "1" * 16}},
            "String should match pattern",
        ),
    ],
)
def test_a_pair_that_does_not_fit_together_is_refused(
    tmp_path: Path, training: dict[str, Any], shipped: dict[str, Any], reason: str
) -> None:
    with pytest.raises(RefusedError, match=reason):
        release.load(write_pair(tmp_path, training=training, shipped=shipped))


@pytest.mark.parametrize(
    ("measurement", "reason"),
    [
        (
            TRAINING | {"cation_parent_smiles": ["c1ccsc1"], "cation_test": CATION_TEST},
            "the release networks lack 1 radical cation\\(s\\)",
        ),
        (
            TRAINING
            | {"cation_parent_smiles": ["c1ccncc1"], "cation_test": CATION_TEST | {"with_cm": 6.0}},
            "measurement.json gives another cation test",
        ),
    ],
)
def test_measurement_networks_of_other_cations_are_refused(
    tmp_path: Path, measurement: dict[str, Any], reason: str
) -> None:
    with pytest.raises(RefusedError, match=reason):
        release.load(write_pair(tmp_path, measurement=measurement))


def test_measurement_networks_without_cations_beside_release_networks_with_them_load(
    pair: release.Release,
) -> None:
    # the certificate's cation clause is the release record's (certificate.acquaintance)
    assert pair.measurement is not None
    assert pair.measurement.cation_test is None
    assert pair.training.cation_test is not None


@pytest.mark.parametrize("name", list(release.RECORDED_FILES))
def test_a_record_changed_after_the_release_was_written_is_refused(
    tmp_path: Path, name: str
) -> None:
    folder = write_pair(tmp_path)
    path = folder / name
    record = json.loads(path.read_text(encoding="utf-8"))
    if "trained_smiles" in record:  # the same set with a molecule listed twice: valid, and changed
        record["trained_smiles"] = [*record["trained_smiles"], record["trained_smiles"][0]]
    elif "coupled_cluster" in record:
        record["coupled_cluster"] += " (edited)"
    else:
        record["a"]["band_error_seed_mean"]["ring-ip"]["n_modes"] += 0.0  # 199 -> 199.0
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    said = re.escape(f"{name} is not the record release.json was written")
    with pytest.raises(RefusedError, match=said):
        release.load(folder)


def test_a_record_the_release_binds_cannot_be_missing(tmp_path: Path) -> None:
    folder = write_pair(tmp_path)
    (folder / release.MEASURED_FILE).unlink()
    with pytest.raises(RefusedError, match=re.escape("measured.json, which release.json binds")):
        release.load(folder)


def test_the_card_of_release_networks_names_every_record(pair: release.Release) -> None:
    text = card.to_markdown(pair)
    assert "`release.json`, `measurement.json` hold the records this card is written from" in text
    assert "`release.json` binds the other records by the SHA-256 of their content." in text


def test_a_measurement_record_of_another_version_is_refused(tmp_path: Path) -> None:
    other = TRAINING | {"network_version": "1.4"}
    with pytest.raises(RefusedError, match="its parts are of different network versions"):
        release.load(write_pair(tmp_path, measurement=other))


def test_release_networks_come_with_their_measurement_record(pair: release.Release) -> None:
    with pytest.raises(ValueError, match="come with the measurement networks' record"):
        dataclasses.replace(pair, measurement=None)


def test_a_stereoisomer_in_the_release_networks_data_is_not_called_unseen() -> None:
    trained = RELEASE_TRAINING | {
        "trained_smiles": [*RELEASE_TRAINING["trained_smiles"], "C(=C/c1ccccc1)\\c1ccccc1"]
    }
    both = release.Training.model_validate(trained)
    measured = release.Training.model_validate(TRAINING)
    said = certificate.acquaintance("C(=Cc1ccccc1)c1ccccc1", both, measured)
    assert said.endswith("is in the release networks' training data")


def test_a_cation_test_on_several_networks_says_how_many() -> None:
    test = release.CationTest.model_validate(CATION_TEST | {"n_networks": 3})
    record = release.Training.model_validate(RELEASE_TRAINING | {"cation_test": test.model_dump()})
    said = certificate.acquaintance("c1ccncc1", record, release.Training.model_validate(TRAINING))
    assert "from 1.5 to 5.4 cm⁻¹ for 3 networks," in said


@pytest.mark.parametrize(
    "record",
    [
        RELEASE_TRAINING | {"cation_test": None},
        RELEASE_TRAINING | {"cation_parent_smiles": []},
    ],
)
def test_cations_and_their_test_come_together(record: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="carries the test"):
        release.Training.model_validate(record)


@pytest.mark.parametrize(
    ("smiles", "expected"),
    [
        ("c1ccccc1", certificate.RELEASE_TRAINED),
        ("c1ccc2ccccc2c1", certificate.RELEASE_TRAINED),
        ("c1ccsc1", certificate.UNSEEN),
        (
            "c1ccncc1",
            (
                f"{certificate.UNSEEN_NEUTRAL}; its radical cation is in the networks' training "
                "data; in a test, training on benzene's cation raised neutral benzene's error "
                "from 1.5 to 5.4 cm⁻¹ for one network, so this molecule's error may be larger "
                "than the measured one"
            ),
        ),
    ],
)
def test_the_seen_line_of_release_networks(
    pair: release.Release, smiles: str, expected: str
) -> None:
    assert certificate.acquaintance(smiles, pair.training, pair.measurement) == expected


def test_a_molecule_trained_on_beside_its_cation_gets_no_cation_clause() -> None:
    both = release.Training.model_validate(
        RELEASE_TRAINING | {"cation_parent_smiles": ["c1ccccc1"]}
    )
    measured = release.Training.model_validate(TRAINING)
    assert certificate.acquaintance("c1ccccc1", both, measured) == certificate.RELEASE_TRAINED
    alone = release.Training.model_validate(
        TRAINING | {"cation_parent_smiles": ["c1ccccc1"], "cation_test": CATION_TEST}
    )
    assert certificate.acquaintance("c1ccccc1", alone).startswith(
        "in hold-out (a): never trained on; "
    )


def test_the_measurement_networks_hold_out_molecules_that_the_release_networks_trained_on(
    pair: release.Release,
) -> None:
    text = card.to_markdown(pair)
    assert f"- **Networks:** {release.made_by(pair)}" in text
    assert "on molecules the measurement networks never trained on" in text
    assert (
        "- Held out from the measurement networks, on which every error is read: 1 in (a); 1 "
        "in (b); the release networks trained on these molecules." in text
    )
    assert "Held out and never trained on" not in text
    assert (
        "- 4 labelled entries of the training pool, each a neutral closed-shell molecule or the "
        "radical cation of one, with an aromatic ring: 3 distinct neutral molecules by SMILES "
        "and the radical cations of 1 molecule. The measurement networks, on which every error "
        "is read, trained on 1 of those neutral molecules; the release networks on these and on "
        "the hold-out molecules." in text
    )


def test_the_card_says_when_a_molecule_is_an_entry_twice(pair: release.Release) -> None:
    assert pair.shipped is not None
    shipped = pair.shipped.model_copy(update={"n_entries": pair.shipped.n_entries + 2})
    text = card.to_markdown(dataclasses.replace(pair, shipped=shipped))
    assert "(the 5 neutral entries hold 3 distinct molecules: a molecule may be an entry" in text


def test_the_certificate_of_release_networks_names_both(
    pair: release.Release, benzene_geometry: Geometry
) -> None:
    low = _low(benzene_geometry)
    admitted = admission.admit(admission.parse_smiles("c1ccccc1"))
    corrections = pair.corrections(benzene_geometry, low.hessian)
    record = certificate.build(admitted, benzene_geometry, low, corrections, pair)
    level = pair.evaluation.higher_level
    assert f"towards {level}. {release.made_by(pair)} " in record["statement"]
    assert record["network"]["kind"] == "release"
    assert record["network"]["errors_measured_on"] == SHIPPED["measurement_checkpoints"]
    assert record["molecule"]["seen"] == certificate.RELEASE_TRAINED
    text = certificate.to_markdown(record)
    assert "and the error bar of this certificate's positions. In square brackets" in text
    assert "what this certificate's positions are" not in text


def test_the_error_of_release_networks_on_the_hold_outs_is_not_measured(
    pair: release.Release, tmp_path: Path
) -> None:
    with pytest.raises(RefusedError, match="which the package does not ship"):
        evaluation.evaluate(pair, tmp_path)


@pytest.fixture(scope="module")
def benzene_geometry() -> Geometry:
    return geometry.start_from_smiles("c1ccccc1")


def _low(molecule: Geometry) -> lowlevel.LowLevel:
    """A made-up low level, as test_certificate builds one."""
    return lowlevel.LowLevel(
        energy=-232.0,
        hessian=random_hessian(molecule.n_atoms, seed=0),
        polar_tensor=np.random.default_rng(1).normal(size=(3, 3 * molecule.n_atoms)),
        refinement=response.Refinement(1e-9, 1e-9, 0, 0),
        sum_rule_violation=1e-7,
        seconds=1.0,
    )
