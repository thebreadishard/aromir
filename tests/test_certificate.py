"""The certificate: every band with its family, its licence and the seeds' spread."""

from __future__ import annotations

import json
import re
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
import pytest
from pydantic import ValidationError
from rdkit import rdBase

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
        benzene, low, _no_correction(), loaded.evaluation, inside_training_sizes=True, withheld=None
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
        benzene, low, corrections, loaded.evaluation, inside_training_sizes=True, withheld=None
    )
    positions = [row.wavenumber_cm for row in rows]
    assert positions == sorted(positions)
    for row in rows:
        assert row.licensed == (row.family != "other")
        band_errors = loaded.evaluation.band_error_seed_mean_cm[row.family]
        assert row.measured_error_cm == {name: error.rms_cm for name, error in band_errors.items()}
        assert row.seed_range_cm[0] <= row.seed_range_cm[1]
    low_families = families.classify(benzene, vib.vibrational_modes(low.hessian, benzene))
    assert sorted(row.family for row in rows) == sorted(low_families)


def test_the_seeds_range_is_their_lowest_and_highest_position(
    benzene: Geometry, low: lowlevel.LowLevel, loaded: release.Release
) -> None:
    corrections = loaded.corrections(benzene, low.hessian)
    rows = certificate.bands(
        benzene, low, corrections, loaded.evaluation, inside_training_sizes=True, withheld=None
    )
    per_seed = [
        vib.vibrational_modes(low.hessian + delta, benzene).wavenumbers
        for delta in corrections.per_seed
    ]
    positions = np.array([row.wavenumber_cm for row in rows])
    np.testing.assert_allclose(
        [row.seed_range_cm[0] for row in rows],
        np.minimum(np.minimum(*per_seed), positions),
        atol=0.05,
    )
    np.testing.assert_allclose(
        [row.seed_range_cm[1] for row in rows],
        np.maximum(np.maximum(*per_seed), positions),
        atol=0.05,
    )
    assert max(row.seed_range_cm[1] - row.seed_range_cm[0] for row in rows) > 0


@pytest.mark.parametrize(
    ("smiles", "expected"),
    [
        ("c1ccc2ccccc2c1", "in the network's training set"),
        ("c1ccccc1", "in test set (a)"),
        ("Cc1ccccc1", "in test set (b)"),
        ("c1ccncc1", "the network has not seen it"),
    ],
)
def test_the_certificate_says_whether_the_network_met_the_molecule(
    smiles: str, expected: str
) -> None:
    training = release.Training.model_validate(TRAINING)
    assert expected in certificate.acquaintance(smiles, training)


@pytest.mark.parametrize(
    ("field", "entry"),
    [
        ("trained_smiles", ["[cH:1]1ccc2ccccc2c1"]),
        ("trained_smiles", ["CS(=O)c1ccccc1"]),
        ("held_out_smiles", {"a": ["N#[N+][N-]c1ccccc1"], "b": ["Cc1ccccc1"]}),
    ],
)
def test_a_training_record_not_in_the_form_admission_writes_is_refused(
    field: str, entry: object
) -> None:
    with pytest.raises(
        ValidationError, match="is not in the standard form admission writes"
    ) as refusal:
        release.Training.model_validate(TRAINING | {field: entry})
    assert f"read with RDKit {rdBase.rdkitVersion})" in str(refusal.value)


def test_a_training_record_is_read_by_molecule_not_by_spelling() -> None:
    # Kekulé benzene and stilbene with its bond directions written the other way round
    # are the molecules admission writes, spelt otherwise: read in the running RDKit's form.
    record = release.Training.model_validate(
        TRAINING
        | {
            "trained_smiles": ["C1=CC=CC=C1", "C(/c1ccccc1)=C\\c1ccccc1"],
            "held_out_smiles": {"a": ["C1=CC=CC=C1"], "b": []},
        }
    )
    stilbene = admission.standard_smiles("C(=C\\c1ccccc1)/c1ccccc1")
    assert record.trained_smiles == ["c1ccccc1", stilbene]
    assert record.held_out_smiles["a"] == ["c1ccccc1"]
    assert record.n_held_out("a") == 0


def test_two_spellings_of_one_molecule_in_a_list_are_refused() -> None:
    with pytest.raises(ValidationError, match="are one molecule, 'c1ccccc1'") as refusal:
        release.Training.model_validate(TRAINING | {"trained_smiles": ["C1=CC=CC=C1", "c1ccccc1"]})
    assert f"(read with RDKit {rdBase.rdkitVersion})" in str(refusal.value)


def test_an_entry_listed_twice_is_refused() -> None:
    with pytest.raises(ValidationError, match="'c1ccccc1' is listed twice"):
        release.Training.model_validate(TRAINING | {"trained_smiles": ["c1ccccc1", "c1ccccc1"]})


@pytest.mark.parametrize("entry", ["", " ", "c1ccccc1 junk", " c1ccccc1", "c1ccccc1\n"])
def test_an_entry_that_is_not_one_smiles_is_refused(entry: str) -> None:
    with pytest.raises(ValidationError, match="is not one SMILES"):
        release.Training.model_validate(TRAINING | {"trained_smiles": [entry]})


def test_a_training_record_entry_that_does_not_parse_names_the_rdkit_that_read_it() -> None:
    with pytest.raises(ValidationError, match="does not parse as a SMILES") as refusal:
        release.Training.model_validate(TRAINING | {"trained_smiles": ["c1ccccc"]})
    assert f"read with RDKit {rdBase.rdkitVersion}" in str(refusal.value)


def test_a_molecule_is_found_by_its_fixed_h_key_where_its_name_differs() -> None:
    # Hexaazidobenzene has more charged atoms than the resonance step enumerates, so it
    # keeps the structure it was given: one azide written N#[N+][N-] names it otherwise.
    azide, other_azide = "N=[N+]=[N-]", "[N-][N+]#N"
    usual = admission.standard_smiles(
        f"c1({azide})c({azide})c({azide})c({azide})c({azide})c1{azide}"
    )
    other = admission.standard_smiles(
        f"c1({other_azide})c({azide})c({azide})c({azide})c({azide})c1{azide}"
    )
    assert other != usual
    training = release.Training.model_validate(TRAINING | {"trained_smiles": [usual]})
    assert "in the network's training set" in certificate.acquaintance(other, training)
    # The fixed-hydrogen layer keeps tautomers apart: 2-pyridone does not stand for
    # 2-hydroxypyridine.
    pyridone = release.Training.model_validate(TRAINING | {"trained_smiles": ["O=c1cccc[nH]1"]})
    assert certificate.acquaintance("Oc1ccccn1", pyridone) == certificate.UNSEEN


def _record(trained: list[str], held_out: list[str] | None = None) -> release.Training:
    """Return a training record with these trained and test-set (a) molecules, in standard form."""
    standard = admission.standard_smiles
    return release.Training.model_validate(
        TRAINING
        | {
            "trained_smiles": [standard(s) for s in trained] or ["c1ccc2ccccc2c1"],
            "held_out_smiles": {"a": [standard(s) for s in held_out or []], "b": ["Cc1ccccc1"]},
        }
    )


HELD_OUT_STEREOISOMER = (
    "the molecule with other or no stereo labels (a stereoisomer, or a configuration this "
    "request leaves open) is in test set (a): never trained on"
)


def test_a_stereo_free_row_stands_for_both_mirror_images() -> None:
    # An xyz sulfoxide names its sulfur's configuration; the training SMILES leaves it
    # open, and the two configurations are mirror images with one infrared spectrum.
    labelled = admission.standard_smiles("C[S@+]([O-])c1ccccc1")
    assert "@" in labelled
    trained = _record(["CS(=O)c1ccccc1"])
    assert "in the network's training set" in certificate.acquaintance(labelled, trained)
    measured = _record([])
    assert certificate.acquaintance(labelled, trained, measured) == certificate.RELEASE_TRAINED
    held = _record([], ["CS(=O)c1ccccc1"])
    assert certificate.acquaintance(labelled, held).startswith("in test set (a): never trained on")
    assert certificate.acquaintance("CC(O)c1ccccc1", trained) == certificate.UNSEEN


def test_a_diastereomer_of_a_trained_or_held_out_molecule_is_a_stereoisomer() -> None:
    one, other = "C[C@H](O)[C@H](C)c1ccccc1", "C[C@H](O)[C@@H](C)c1ccccc1"
    other = admission.standard_smiles(other)
    assert certificate.acquaintance(other, _record([one])) == certificate.STEREOISOMER
    assert "in the network's training set" in certificate.acquaintance(
        admission.standard_smiles(one), _record([one])
    )
    assert certificate.acquaintance(other, _record([], [one])) == HELD_OUT_STEREOISOMER


@pytest.mark.parametrize(
    ("request_smiles", "row"),
    [
        ("C(=Cc1ccccc1)c1ccccc1", "C(=C\\c1ccccc1)/c1ccccc1"),
        ("CC(N)C(C)c1ccccc1", "C[C@H](N)[C@H](C)c1ccccc1"),
        ("C[C@H](O)[C@@H](C)c1ccccc1", "CC(O)C(C)c1ccccc1"),
    ],
    ids=["stilbene", "unlabelled request", "two centres"],
)
def test_a_configuration_that_is_not_a_mirror_image_is_a_stereoisomer_never_unseen(
    request_smiles: str, row: str
) -> None:
    asked = admission.standard_smiles(request_smiles)
    assert certificate.acquaintance(asked, _record([row])) == certificate.STEREOISOMER
    release_said = certificate.acquaintance(asked, _record([row]), _record([]))
    assert release_said == certificate.STEREOISOMER_RELEASE
    assert certificate.acquaintance(asked, _record([], [row])) == HELD_OUT_STEREOISOMER


def test_only_one_tetrahedral_centre_counts_as_mirror_images() -> None:
    assert certificate.only_mirror_images("C[S@+]([O-])c1ccccc1")
    assert certificate.only_mirror_images("C[C@H](O)c1ccccc1")
    assert not certificate.only_mirror_images("C[C@H](O)[C@@H](C)c1ccccc1")
    assert not certificate.only_mirror_images("C(=C\\c1ccccc1)/c1ccccc1")
    assert not certificate.only_mirror_images("c1ccccc1")


def test_the_certificate_says_what_reading_the_input_changed(
    benzene: Geometry, low: lowlevel.LowLevel, loaded: release.Release
) -> None:
    admitted = admission.admit(admission.read("[cH:1]1ccccc1"))
    corrections = loaded.corrections(benzene, low.hessian)
    record = certificate.build(admitted, benzene, low, corrections, loaded)
    assert record["molecule"]["input_changes"] == [admission.ATOM_MAPS_CLEARED]
    text = certificate.to_markdown(record)
    assert text.startswith("# Certificate: c1ccccc1\n")
    note = admission.ATOM_MAPS_CLEARED
    assert f"\n- {note[0].upper()}{note[1:]}.\n" in text


def test_the_record_and_its_text(
    benzene: Geometry, low: lowlevel.LowLevel, loaded: release.Release
) -> None:
    admitted = admission.admit(admission.read("c1ccccc1"))
    corrections = loaded.corrections(benzene, low.hessian)
    record = certificate.build(admitted, benzene, low, corrections, loaded)
    assert record["molecule"]["inside_training_sizes"] is True
    assert record["molecule"]["training_atoms"] == [10, 30]
    assert record["network"] == {
        "version": "1.3",
        "seeds": [0, 1],
        "tensor_sha256": {
            "seed-0": loaded.tensor_sha256[0],
            "seed-1": loaded.tensor_sha256[1],
        },
        "release": loaded.folder.name,
        "inside_package": False,
        "higher_level": loaded.evaluation.higher_level,
        "kind": "measurement",
        "errors_measured_on": None,
    }
    assert record["accuracy"]["licensed_families"] == ["ring-ip", "CH-stretch", "CH-oop"]
    n_licensed = sum(band["licensed"] for band in record["bands"])
    assert 0 < n_licensed < N_BANDS
    assert f"{n_licensed} of {N_BANDS} bands are licensed" in record["statement"]
    assert "on every test set (ring-ip, CH-stretch, CH-oop)" in record["statement"]
    assert (
        "never saw. Those molecules are test set (a): unseen molecules and test set (b): held-out "
        "scaffolds. "
    ) in record["statement"]
    assert record["statement"].endswith(
        "The errors are against ωB97X/6-31G*, not against experiment: a "
        "corrected position estimates that level's harmonic position, which for nearly all "
        "modes lies above the measured band, so compare a measured spectrum with the scaled "
        "B3LYP positions, printed beside each band as the baseline."
    )
    assert "Not in that error" not in record["statement"]
    assert "harmonic" in record["positions"]
    assert not any("imaginary (negative) position" in warning for warning in record["warnings"])
    assert json.loads(json.dumps(record)) == record
    text = certificate.to_markdown(record)
    assert text.startswith("# Certificate: c1ccccc1\n")
    assert "the training set spans 10–30 atoms: inside" in text
    assert (
        f"*Network:* release v1.3, `{loaded.folder.name}` (a folder outside the package); the "
        f"seeds' tensor files by SHA-256: seed-0 `{loaded.tensor_sha256[0]}`, seed-1 "
        f"`{loaded.tensor_sha256[1]}`.\n"
    ) in text
    assert str(loaded.folder.parent) not in json.dumps(record) + text
    seeds = loaded.evaluation.first_order_error_per_seed_cm["other"]
    in_brackets = {name: ", ".join(f"{error:.2f}" for error in seeds[name]) for name in seeds}
    assert "| other | 2.78 [1.68–3.76], 156 bands | 3.89 [3.50–4.20], 1225 bands | no |" in text
    means = loaded.evaluation.first_order_error_per_seed_mean_cm["other"]
    assert (
        f"| other | {means['a']:.2f} ({in_brackets['a']}) | {means['b']:.2f} ({in_brackets['b']}) |"
        in text
    )
    assert "It is not the error of what is printed above." in text
    assert "The licence is read on the band error of the seeds' mean" in text
    wag_on_b = loaded.evaluation.first_order_error_per_seed_mean_cm["CH-oop"]["b"]
    assert f") | {wag_on_b:.2f} (" in text
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
    admitted = admission.admit(admission.read("c1ccccc1"))
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
    admitted = admission.admit(admission.read("c1ccccc1"))
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


def test_the_printed_position_lies_inside_its_range(
    benzene: Geometry, low: lowlevel.LowLevel, loaded: release.Release
) -> None:
    # Two seeds that couple the two softest modes with opposite signs: each seed pushes
    # the two bands apart, their mean leaves them where they were, outside both seeds'.
    vectors = vib.vibrational_modes(low.hessian, benzene).vectors
    weights_ = vib.mass_weights(benzene.masses_amu)
    first, second = vectors[:, 0] * weights_, vectors[:, 1] * weights_
    coupling = 0.05 * (np.outer(first, second) + np.outer(second, first))
    corrections = release.Corrections(np.zeros_like(coupling), (coupling, -coupling))
    per_seed = [
        vib.vibrational_modes(low.hessian + delta, benzene).wavenumbers
        for delta in corrections.per_seed
    ]
    plain = vib.vibrational_modes(low.hessian, benzene).wavenumbers
    np.testing.assert_allclose(per_seed[0], per_seed[1], atol=1e-6)
    assert float(np.abs(plain - per_seed[0]).max()) > 1.0
    rows = certificate.bands(
        benzene, low, corrections, loaded.evaluation, inside_training_sizes=True, withheld=None
    )
    for row in rows:
        assert row.seed_range_cm[0] <= row.wavenumber_cm <= row.seed_range_cm[1]
