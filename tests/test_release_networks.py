"""Release networks beside measurement networks: one is shipped, the other has the errors."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from aromir import admission, card, certificate, evaluation, geometry, lowlevel, release, response
from aromir.errors import RefusedError

from support import TRAINING, random_hessian, write_release

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
    "n_molecules": 4,
    "fitted_share": 0.85,
    "checks": {"same_recipe": True, "early_stopping": True},
    "release_checkpoints": {"release_seed0.pt": "0" * 16, "release_seed1.pt": "1" * 16},
    "measurement_checkpoints": {"seed0.pt": "2" * 16, "seed1.pt": "3" * 16},
    "source": "a made-up record",
}


def write_pair(
    folder: Path, training: dict[str, Any] | None = None, shipped: dict[str, Any] | None = None
) -> Path:
    """Write a folder of release networks for ``TRAINING``'s measurement networks."""
    write_release(folder)
    (folder / release.MEASUREMENT_FILE).write_text(json.dumps(TRAINING), encoding="utf-8")
    record = RELEASE_TRAINING if training is None else training
    (folder / release.TRAINING_FILE).write_text(json.dumps(record), encoding="utf-8")
    (folder / release.RELEASE_FILE).write_text(json.dumps(SHIPPED if shipped is None else shipped))
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
        "The spectrum is made by the release networks of version 1.3, trained on all 4 "
        "molecules with labels: each of the 2 networks is fitted on a random 85 % of the "
        "molecules and stopped on the other 15 %, a different 15 % per seed,"
    )
    assert f"(with 2 networks about 0.15² ≈ {100 * (1 - 0.85) ** 2:.1f} % by none)" in sentence
    assert sentence.endswith(
        "the release networks themselves have seen every molecule we hold labels for, so "
        "they have no held-out error of their own."
    )


@pytest.mark.parametrize("present", [release.RELEASE_FILE, release.MEASUREMENT_FILE])
def test_one_record_of_the_pair_without_the_other_is_refused(tmp_path: Path, present: str) -> None:
    folder = write_pair(tmp_path)
    other = release.MEASUREMENT_FILE if present == release.RELEASE_FILE else release.RELEASE_FILE
    (folder / other).unlink()
    with pytest.raises(RefusedError, match=f"{present} without {other}"):
        release.load(folder)


def test_release_networks_that_failed_a_check_are_not_shipped(tmp_path: Path) -> None:
    failed = SHIPPED | {"checks": {"same_recipe": True, "early_stopping": False}}
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
        (RELEASE_TRAINING, SHIPPED | {"network_version": "1.4"}, "are of other versions"),
    ],
)
def test_a_pair_that_does_not_fit_together_is_refused(
    tmp_path: Path, training: dict[str, Any], shipped: dict[str, Any], reason: str
) -> None:
    with pytest.raises(RefusedError, match=reason):
        release.load(write_pair(tmp_path, training=training, shipped=shipped))


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
        "- Held out and never trained on: 1 in (a); 1 in (b) — held out from the measurement "
        "networks only; the release networks trained on them." in text
    )
    assert (
        "- 4 molecules with labels, each with an aromatic ring: 3 neutral closed-shell "
        "molecules by SMILES, and the radical cations of 1 of them; the measurement networks, "
        "on which every error is read, trained on 1 of the neutral ones." in text
    )


def test_the_certificate_of_release_networks_names_both(
    pair: release.Release, benzene_geometry: Geometry
) -> None:
    low = _low(benzene_geometry)
    admitted = admission.admit(admission.parse_smiles("c1ccccc1"))
    corrections = pair.corrections(benzene_geometry, low.hessian)
    record = certificate.build(admitted, benzene_geometry, low, corrections, pair)
    assert f", {release.made_by(pair)} " in record["statement"]
    assert "never saw. " not in record["statement"].split(release.made_by(pair))[0]
    assert record["network"]["kind"] == "release"
    assert record["network"]["errors_measured_on"] == SHIPPED["measurement_checkpoints"]
    assert record["molecule"]["seen"] == certificate.RELEASE_TRAINED


def test_the_error_of_release_networks_on_the_hold_outs_is_not_measured(
    pair: release.Release, tmp_path: Path
) -> None:
    with pytest.raises(RefusedError, match="their error there is no measurement"):
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
