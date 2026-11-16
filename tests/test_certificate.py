"""The certificate: every band with its family, its licence and the seeds' spread."""

from __future__ import annotations

import json
import re
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
import pytest

from aromir import admission, certificate, families, geometry, lowlevel, release, response, spectrum
from aromir import vibrations as vib

from support import TRAINING, random_hessian, write_release

if TYPE_CHECKING:
    from pathlib import Path

    from aromir.molecule import Geometry

N_BANDS = 30


@pytest.fixture(scope="module")
def loaded(tmp_path_factory: pytest.TempPathFactory) -> release.Release:
    return release.load(write_release(tmp_path_factory.mktemp("release")))


@pytest.fixture(scope="module")
def benzene() -> Geometry:
    return geometry.start_from_smiles("c1ccccc1")


@pytest.fixture(scope="module")
def low(benzene: Geometry) -> lowlevel.LowLevel:
    return lowlevel.LowLevel(
        energy=-232.0,
        hessian=random_hessian(benzene.n_atoms, seed=0),
        polar_tensor=np.random.default_rng(1).normal(size=(3, 36)),
        refinement=response.Refinement(1e-9, 1e-9, 0, 0),
        sum_rule_violation=1e-7,
        seconds=1.0,
    )


def _no_correction() -> release.Corrections:
    zero = np.zeros((36, 36))
    return release.Corrections(zero, (zero, zero))


def test_without_a_correction_every_band_is_its_own_b3lyp_band(
    benzene: Geometry, low: lowlevel.LowLevel, loaded: release.Release
) -> None:
    rows = certificate.bands(
        benzene, low, _no_correction(), loaded.evaluation, inside_training_sizes=True
    )
    plain = spectrum.harmonic_bands(low.hessian, benzene, low.polar_tensor)
    scaled = spectrum.scaled(plain)
    assert len(rows) == N_BANDS
    np.testing.assert_allclose([row.wavenumber_cm for row in rows], plain.wavenumbers, atol=0.05)
    np.testing.assert_allclose([row.b3lyp_cm for row in rows], plain.wavenumbers, atol=0.05)
    np.testing.assert_allclose([row.scaled_b3lyp_cm for row in rows], scaled.wavenumbers, atol=0.05)
    np.testing.assert_allclose(
        [row.intensity_km_mol for row in rows], plain.intensities, atol=0.005
    )
    assert all(row.seed_range_cm == (row.wavenumber_cm, row.wavenumber_cm) for row in rows)


def test_each_band_carries_its_familys_record(
    benzene: Geometry, low: lowlevel.LowLevel, loaded: release.Release
) -> None:
    corrections = loaded.corrections(benzene, low.hessian)
    rows = certificate.bands(
        benzene, low, corrections, loaded.evaluation, inside_training_sizes=True
    )
    positions = [row.wavenumber_cm for row in rows]
    assert positions == sorted(positions)
    for row in rows:
        assert row.licensed == (row.family != "other")
        assert row.measured_error_cm == loaded.evaluation.errors_cm[row.family]
        assert row.seed_range_cm[0] <= row.seed_range_cm[1]
    low_families = families.classify(benzene, vib.vibrational_modes(low.hessian, benzene))
    assert sorted(row.family for row in rows) == sorted(low_families)


def test_the_seeds_range_is_their_lowest_and_highest_position(
    benzene: Geometry, low: lowlevel.LowLevel, loaded: release.Release
) -> None:
    corrections = loaded.corrections(benzene, low.hessian)
    rows = certificate.bands(
        benzene, low, corrections, loaded.evaluation, inside_training_sizes=True
    )
    per_seed = [
        vib.vibrational_modes(low.hessian + delta, benzene).wavenumbers
        for delta in corrections.per_seed
    ]
    np.testing.assert_allclose(
        [row.seed_range_cm[0] for row in rows], np.minimum(*per_seed), atol=0.05
    )
    np.testing.assert_allclose(
        [row.seed_range_cm[1] for row in rows], np.maximum(*per_seed), atol=0.05
    )
    assert max(row.seed_range_cm[1] - row.seed_range_cm[0] for row in rows) > 0


@pytest.mark.parametrize(
    ("smiles", "expected"),
    [
        ("c1ccc2ccccc2c1", "in the network's training set"),
        ("c1ccccc1", "in hold-out (a)"),
        ("Cc1ccccc1", "in hold-out (b)"),
        ("c1ccncc1", "the network has not seen it"),
    ],
)
def test_the_certificate_says_whether_the_network_met_the_molecule(
    smiles: str, expected: str
) -> None:
    training = release.Training.model_validate(TRAINING)
    assert expected in certificate.acquaintance(smiles, training)


def test_the_record_and_its_text(
    benzene: Geometry, low: lowlevel.LowLevel, loaded: release.Release
) -> None:
    admitted = admission.admit(admission.parse_smiles("c1ccccc1"))
    corrections = loaded.corrections(benzene, low.hessian)
    record = certificate.build(admitted, benzene, low, corrections, loaded)
    assert record["molecule"]["inside_training_sizes"] is True
    assert record["molecule"]["training_atoms"] == [10, 30]
    assert record["network"] == {
        "version": "1.3",
        "seeds": [0, 1],
        "higher_level": loaded.evaluation.higher_level,
    }
    assert record["accuracy"]["licensed_families"] == ["ring-ip", "CH-stretch", "CH-oop"]
    n_licensed = sum(band["licensed"] for band in record["bands"])
    assert 0 < n_licensed < N_BANDS
    assert f"{n_licensed} of {N_BANDS} bands are licensed" in record["statement"]
    assert "on every hold-out (ring-ip, CH-stretch, CH-oop)" in record["statement"]
    assert "Not in that error" not in record["statement"]
    assert "harmonic" in record["positions"]
    assert not any("imaginary (negative) position" in warning for warning in record["warnings"])
    assert json.loads(json.dumps(record)) == record
    text = certificate.to_markdown(record)
    assert text.startswith("# Certificate: c1ccccc1\n")
    assert "the training set spans 10–30 atoms: inside" in text
    seeds = loaded.evaluation.seed_errors_cm["other"]
    in_brackets = {name: ", ".join(f"{error:.2f}" for error in seeds[name]) for name in seeds}
    assert f"| other | 3.62 ({in_brackets['a']}) | 3.89 ({in_brackets['b']}) | no |" in text
    assert ") | 2.70 (" in text
    assert "(the DFT proxy" in record["statement"]
    assert text.count("(the DFT proxy") == 1
    band_rows = [line for line in text.split("\n") if line.count("|") == 9 and "–" in line]
    assert len(band_rows) == N_BANDS
    error_cells = [line.split("|")[-2].strip() for line in band_rows]
    assert all(re.fullmatch(r"\d+\.\d\d / \d+\.\d\d", cell) for cell in error_cells)


def test_a_molecule_outside_the_training_sizes_is_marked(
    low: lowlevel.LowLevel, benzene: Geometry, loaded: release.Release, tmp_path: Path
) -> None:
    small_training = loaded.training.model_copy(update={"min_atoms": 20})
    narrow = replace(loaded, training=small_training)
    admitted = admission.admit(admission.parse_smiles("c1ccccc1"))
    corrections = narrow.corrections(benzene, low.hessian)
    record = certificate.build(admitted, benzene, low, corrections, narrow)
    assert record["molecule"]["inside_training_sizes"] is False
    assert "20–30 atoms: OUTSIDE: no band is licensed" in certificate.to_markdown(record)
    assert not any(band["licensed"] for band in record["bands"])
    notes = {band["licence_note"] for band in record["bands"] if band["family"] != "other"}
    assert notes == {"the molecule is outside the training sizes"}
    assert "0 of 30 bands are licensed" in record["statement"]
    assert tmp_path.exists()


def test_an_imaginary_corrected_band_is_a_warning(
    benzene: Geometry, low: lowlevel.LowLevel, loaded: release.Release
) -> None:
    admitted = admission.admit(admission.parse_smiles("c1ccccc1"))
    softest = vib.vibrational_modes(low.hessian, benzene).vectors[:, 0]
    weights_ = vib.mass_weights(benzene.masses_amu)
    direction = softest * weights_
    unstable = -2.0 * np.outer(direction, direction)
    corrections = release.Corrections(unstable, (unstable, unstable))
    record = certificate.build(admitted, benzene, low, corrections, loaded)
    assert record["bands"][0]["wavenumber_cm"] < 0
    assert record["bands"][0]["licensed"] is False
    assert record["bands"][0]["licence_note"] == "imaginary position"
    assert "| no: imaginary position |" in certificate.to_markdown(record)
    assert record["warnings"] == [
        (
            "1 corrected band(s) have an imaginary (negative) position: the correction made a "
            "soft mode unstable, and those bands are not physical"
        )
    ]
    assert "**Warning:** 1 corrected band(s)" in certificate.to_markdown(record)
