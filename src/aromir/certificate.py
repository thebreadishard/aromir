"""The certificate of a result: each band with what it rests on.

A corrected spectrum is only as good as the network behind it, and the network is known
unevenly: well for some kinds of vibration, less well for others, not at all outside its
training domain. The certificate states, band by band, the family the band belongs to,
the error measured for that family on molecules the network never saw, whether the family
is licensed, and how far the network's seeds disagree. Every number about the network
comes from the release's records.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Final

from aromir import families, spectrum, vibrations

if TYPE_CHECKING:
    from collections.abc import Sequence

    from aromir.admission import Admission
    from aromir.lowlevel import LowLevel
    from aromir.molecule import Geometry
    from aromir.release import Corrections, Evaluation, Release, Training

POSITIONS_NOTE: Final = "harmonic positions: the anharmonic shift is not included in this version"
LEVEL_NOTE: Final = "B3LYP/6-31G* corrected"


@dataclass(frozen=True)
class Band:
    """One band of the corrected spectrum.

    Attributes:
        wavenumber_cm: The corrected harmonic position.
        intensity_km_mol: The double-harmonic height.
        seed_range_cm: The lowest and highest position of this band over the seeds.
        b3lyp_cm: The position of the uncorrected mode that moves most like it.
        scaled_b3lyp_cm: That mode's position under the conventional scale factors.
        family: The mode family, by the rule the network's error is measured in.
        licensed: Whether that family's error is within the licence line on every hold-out.
        measured_error_cm: The family's seed-mean error per hold-out.
    """

    wavenumber_cm: float
    intensity_km_mol: float
    seed_range_cm: tuple[float, float]
    b3lyp_cm: float
    scaled_b3lyp_cm: float
    family: families.Family
    licensed: bool
    measured_error_cm: dict[str, float]


def bands(
    geometry: Geometry, low: LowLevel, corrections: Corrections, evaluation: Evaluation
) -> list[Band]:
    """Return the corrected spectrum's bands in ascending order, each with its record."""
    low_modes = vibrations.vibrational_modes(low.hessian, geometry)
    low_families = families.classify(geometry, low_modes)
    scaled = spectrum.scaled(spectrum.harmonic_bands(low.hessian, geometry, low.polar_tensor))
    corrected_hessian = low.hessian + corrections.mean
    corrected = spectrum.harmonic_bands(corrected_hessian, geometry, low.polar_tensor)
    partner = spectrum.partners(
        vibrations.vibrational_modes(corrected_hessian, geometry), low_modes
    )
    ranges = spectrum.seed_ranges(low.hessian, geometry, corrections.per_seed)
    rows = []
    for index, low_index in enumerate(partner):
        family = low_families[low_index]
        rows.append(
            Band(
                wavenumber_cm=round(float(corrected.wavenumbers[index]), 1),
                intensity_km_mol=round(float(corrected.intensities[index]), 2),
                seed_range_cm=(
                    round(float(ranges[index, 0]), 1),
                    round(float(ranges[index, 1]), 1),
                ),
                b3lyp_cm=round(float(low_modes.wavenumbers[low_index]), 1),
                scaled_b3lyp_cm=round(float(scaled.wavenumbers[low_index]), 1),
                family=family,
                licensed=evaluation.licensed(family),
                measured_error_cm=dict(evaluation.errors_cm[family]),
            )
        )
    return rows


def acquaintance(smiles: str, training: Training) -> str:
    """Say whether the network has met a molecule, by its canonical SMILES."""
    if smiles in training.trained_smiles:
        return "in the network's training set: the hold-out error is not this molecule's measure"
    for holdout, held_out in training.held_out_smiles.items():
        if smiles in held_out:
            return f"in hold-out ({holdout}): never trained on; the measured error includes it"
    return "not in the training set or a hold-out: the network has not seen it"


def _domain(admitted: Admission, training: Training) -> dict[str, Any]:
    inside = training.min_atoms <= admitted.n_atoms <= training.max_atoms
    return {
        "training_atoms": [training.min_atoms, training.max_atoms],
        "inside_training_sizes": inside,
        "seen": acquaintance(admitted.smiles, training),
    }


def _statement(admitted: Admission, release: Release, licensed: Sequence[str]) -> str:
    evaluation = release.evaluation
    return (
        f"{admitted.smiles}: the harmonic infrared spectrum of ΔH network "
        f"v{release.network_version} ({len(release.networks)} seeds), {LEVEL_NOTE} towards "
        f"{evaluation.higher_level}, with the error measured on molecules the network never "
        f"saw. Licensed for {', '.join(licensed) or 'no family'}: a family is licensed when "
        f"its error is within {evaluation.licence_line_cm:g} cm⁻¹ on every hold-out. "
        f"Coupled cluster: {evaluation.coupled_cluster}. The scaled B3LYP positions are a "
        "baseline for comparison only."
    )


def _warnings(rows: list[Band]) -> list[str]:
    """Return what a reader must know before using the bands at all."""
    n_imaginary = sum(row.wavenumber_cm < 0 for row in rows)
    if not n_imaginary:
        return []
    return [
        (
            f"{n_imaginary} corrected band(s) have an imaginary (negative) position: the "
            "correction made a soft mode unstable, and those bands are not physical"
        )
    ]


def build(
    admitted: Admission,
    geometry: Geometry,
    low: LowLevel,
    corrections: Corrections,
    release: Release,
) -> dict[str, Any]:
    """Build the certificate's record for one computed molecule."""
    evaluation = release.evaluation
    licensed = [family for family in families.FAMILIES if evaluation.licensed(family)]
    rows = bands(geometry, low, corrections, evaluation)
    return {
        "statement": _statement(admitted, release, licensed),
        "molecule": asdict(admitted) | _domain(admitted, release.training),
        "network": {
            "version": release.network_version,
            "seeds": [recipe.seed for recipe in release.recipes],
            "higher_level": evaluation.higher_level,
        },
        "positions": POSITIONS_NOTE,
        "warnings": _warnings(rows),
        "bands": [asdict(row) | {"seed_range_cm": list(row.seed_range_cm)} for row in rows],
        "accuracy": evaluation.model_dump(mode="json") | {"licensed_families": licensed},
    }


def _band_lines(record: dict[str, Any]) -> list[str]:
    holdouts = list(record["accuracy"]["holdouts"])
    error_header = " / ".join(f"({name})" for name in holdouts)
    lines = [
        (
            "| position (cm⁻¹) | height (km/mol) | seeds' range | B3LYP | scaled B3LYP | family "
            f"| licensed | measured error {error_header}, cm⁻¹ |"
        ),
        "|---|---|---|---|---|---|---|---|",
    ]
    for band in record["bands"]:
        low, high = band["seed_range_cm"]
        errors = " / ".join(str(band["measured_error_cm"][name]) for name in holdouts)
        lines.append(
            f"| {band['wavenumber_cm']} | {band['intensity_km_mol']} | {low}–{high} "
            f"| {band['b3lyp_cm']} | {band['scaled_b3lyp_cm']} | {band['family']} "
            f"| {'yes' if band['licensed'] else 'no'} | {errors} |"
        )
    return lines


def _accuracy_lines(accuracy: dict[str, Any]) -> list[str]:
    holdouts = accuracy["holdouts"]
    header = " | ".join(f"({name}), {h['n_molecules']} molecules" for name, h in holdouts.items())
    lines = [f"| family | {header} | licensed |", "|---|" + "---|" * (len(holdouts) + 1)]
    for family, errors in accuracy["errors_cm"].items():
        per_seed = accuracy["seed_errors_cm"][family]
        cells = " | ".join(
            f"{errors[name]} ({', '.join(str(x) for x in per_seed[name])})" for name in holdouts
        )
        licensed = "yes" if family in accuracy["licensed_families"] else "no"
        lines.append(f"| {family} | {cells} | {licensed} |")
    lines += ["", *(f"- ({name}): {h['description']}" for name, h in holdouts.items())]
    return [*lines, f"- sources: {'; '.join(accuracy['sources'])}"]


def to_markdown(record: dict[str, Any]) -> str:
    """Render a certificate record as a Markdown document."""
    molecule, network = record["molecule"], record["network"]
    low, high = molecule["training_atoms"]
    size = "inside" if molecule["inside_training_sizes"] else "OUTSIDE"
    lines = [
        f"# Certificate: {molecule['smiles']}",
        "",
        record["statement"],
        "",
        "## Molecule",
        "",
        (
            f"- {molecule['formula']}, {molecule['n_atoms']} atoms; the training set spans "
            f"{low}–{high} atoms: {size}"
        ),
        f"- {molecule['seen']}",
        "",
        (
            f"## Spectrum, band by band (network v{network['version']}, "
            f"seeds {', '.join(str(seed) for seed in network['seeds'])})"
        ),
        "",
        *_band_lines(record),
        "",
        f"*Positions:* {record['positions']}.",
        *(f"**Warning:** {warning}." for warning in record["warnings"]),
        "",
        "## How well the network is known",
        "",
        (
            f"Seed-mean rms error of the corrected positions against {network['higher_level']}, "
            "in cm⁻¹, each seed's value in brackets."
        ),
        "",
        *_accuracy_lines(record["accuracy"]),
        "",
    ]
    return "\n".join(lines)
