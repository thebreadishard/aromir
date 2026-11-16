"""The certificate of a result: each band with what it rests on.

A corrected spectrum is only as good as the network behind it, and the network is known
unevenly: well for some kinds of vibration, less well for others, not at all outside its
training domain. The certificate states, band by band, the family the band belongs to,
the error measured for that family on molecules the network never saw, whether the band is
licensed for this molecule, and how far the network's seeds disagree. It also states what
that error does not cover: how far the positions move when the same molecule is turned,
mirrored or listed in another atom order. Every number about the network comes from the
release's records.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Final

from aromir import families, spectrum, vibrations
from aromir.network import primitives
from aromir.release import error_text, level_name

if TYPE_CHECKING:
    from aromir.admission import Admission
    from aromir.lowlevel import LowLevel
    from aromir.molecule import Geometry
    from aromir.release import Corrections, Evaluation, Release, Training

POSITIONS_NOTE: Final = "harmonic positions: the anharmonic shift is not included in this version"
LEVEL_NOTE: Final = "B3LYP/6-31G* corrected"
ATOM_ORDER: Final = "another atom order of the same molecule (aromir computes in one)"
HEIGHTS_NOTE: Final = "the heights are B3LYP's: the dipole derivatives are not corrected"
CONFORMER_NOTE: Final = (
    "one conformer was computed, the one the optimisation reached from aromir's start "
    "geometry; another conformer has another spectrum"
)
NOT_PLANAR_NOTE: Final = (
    "the molecule is not planar, so no band counts as CH-oop: its C–H wags are printed as "
    "other or, between 950 and 1050 cm⁻¹, as ring-ip"
)


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
        licensed: Whether the family's measured error may be read as this band's: the
            family is within the licence line on every hold-out, the molecule lies inside
            the training sizes, and the band is real and of the kind the family names.
        licence_note: Why the band is not licensed; empty when it is.
        measured_error_cm: The family's seed-mean error per hold-out.
    """

    wavenumber_cm: float
    intensity_km_mol: float
    seed_range_cm: tuple[float, float]
    b3lyp_cm: float
    scaled_b3lyp_cm: float
    family: families.Family
    licensed: bool
    licence_note: str
    measured_error_cm: dict[str, float]


def _licence_note(
    family: families.Family,
    position_cm: float,
    carrier: str,
    evaluation: Evaluation,
    *,
    inside_training_sizes: bool,
) -> str:
    """Return the first reason a band is not licensed, or "" when it is."""
    if position_cm < 0:
        return "imaginary position"
    if not evaluation.licensed(family):
        return "the family's error is above the licence line"
    if not inside_training_sizes:
        return "the molecule is outside the training sizes"
    if family == "CH-stretch" and carrier != "C":
        return f"an {carrier}–H stretch, which the rule counts as CH-stretch"
    return ""


def bands(
    geometry: Geometry,
    low: LowLevel,
    corrections: Corrections,
    evaluation: Evaluation,
    *,
    inside_training_sizes: bool = True,
) -> list[Band]:
    """Return the corrected spectrum's bands in ascending order, each with its record."""
    low_modes = vibrations.vibrational_modes(low.hessian, geometry)
    low_families = families.classify(geometry, low_modes)
    carriers = families.hydrogen_carriers(geometry, low_modes)
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
        position = round(float(corrected.wavenumbers[index]), 1)
        note = _licence_note(
            family,
            position,
            carriers[low_index],
            evaluation,
            inside_training_sizes=inside_training_sizes,
        )
        rows.append(
            Band(
                wavenumber_cm=position,
                intensity_km_mol=round(float(corrected.intensities[index]), 2),
                seed_range_cm=(
                    round(float(ranges[index, 0]), 1),
                    round(float(ranges[index, 1]), 1),
                ),
                b3lyp_cm=round(float(low_modes.wavenumbers[low_index]), 1),
                scaled_b3lyp_cm=round(float(scaled.wavenumbers[low_index]), 1),
                family=family,
                licensed=not note,
                licence_note=note,
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


def has_linear_group(geometry: Geometry, release: Release) -> bool:
    """Tell whether the network's internal coordinates of a molecule hold a linear angle."""
    built = primitives.build(geometry, release.recipes[0].inputs.primitives)
    return any(primitives.kind(p) == primitives.LINEAR_ANGLE for p in built.internals)


def _domain(admitted: Admission, geometry: Geometry, release: Release) -> dict[str, Any]:
    training = release.training
    inside = training.min_atoms <= admitted.n_atoms <= training.max_atoms
    return {
        "training_atoms": [training.min_atoms, training.max_atoms],
        "inside_training_sizes": inside,
        "seen": acquaintance(admitted.smiles, training),
        "linear_group": has_linear_group(geometry, release),
        "planar": families.is_planar(geometry),
    }


def spreads_cm(release: Release, *, linear_group: bool) -> dict[str, float]:
    """Return how far a molecule's positions move under each change that leaves it itself.

    The family errors do not hold these: they were measured in one orientation and one
    atom order per molecule.
    """
    spreads = {}
    order = release.evaluation.atom_order_spread
    if order is not None:
        spreads[ATOM_ORDER] = order.max_cm
    if release.invariance is not None:
        for name, spread in release.invariance.spreads().items():
            spreads[name] = spread.of(linear_group=linear_group)
    return spreads


def _not_in_the_error(release: Release, molecule: dict[str, Any]) -> list[str]:
    """Return what a reader must add to a band's measured error, one sentence each."""
    line = release.evaluation.licence_line_cm
    group = "with" if molecule["linear_group"] else "without"
    lines = []
    for name, value in spreads_cm(release, linear_group=molecule["linear_group"]).items():
        measured_on = "" if name == ATOM_ORDER else f" on molecules {group} a linear group"
        above = f", more than the licence line of {line:g} cm⁻¹" if value > line else ""
        lines.append(f"{name} moves a position by up to {value:.2f} cm⁻¹{measured_on}{above}")
    if release.evaluation.atom_order_spread is None:
        lines.append("the spread between atom orders was not measured for this release")
    if release.invariance is None:
        lines.append(
            "the spread under a rotation, a reflection and another listing of an xyz file's "
            "atoms was not measured for this release"
        )
    lines += [HEIGHTS_NOTE, CONFORMER_NOTE]
    if not molecule["planar"]:
        lines.append(NOT_PLANAR_NOTE)
    return lines


def _statement(
    admitted: Admission, release: Release, rows: list[Band], molecule: dict[str, Any]
) -> str:
    evaluation = release.evaluation
    line = evaluation.licence_line_cm
    within = [family for family in families.FAMILIES if evaluation.licensed(family)]
    spreads = spreads_cm(release, linear_group=molecule["linear_group"])
    beyond = ""
    if spreads and max(spreads.values()) > line:
        beyond = (
            "Not in that error: the same molecule turned, mirrored or listed in another "
            f"atom order gets positions up to {max(spreads.values()):.2f} cm⁻¹ away, more "
            "than the licence line. "
        )
    return (
        f"{admitted.smiles}: the harmonic infrared spectrum of ΔH network "
        f"v{release.network_version} ({len(release.networks)} seeds), {LEVEL_NOTE} towards "
        f"{evaluation.higher_level}, with the error measured on molecules the network never "
        f"saw. {sum(row.licensed for row in rows)} of {len(rows)} bands are licensed: a band "
        f"is licensed when its family's error is within {line:g} cm⁻¹ on every hold-out "
        f"({', '.join(within) or 'no family'}), the molecule lies inside the training sizes, "
        f"and the band is real and of the kind the family names. {beyond}"
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
    within = [family for family in families.FAMILIES if evaluation.licensed(family)]
    molecule = asdict(admitted) | _domain(admitted, geometry, release)
    rows = bands(
        geometry,
        low,
        corrections,
        evaluation,
        inside_training_sizes=molecule["inside_training_sizes"],
    )
    return {
        "statement": _statement(admitted, release, rows, molecule),
        "molecule": molecule,
        "network": {
            "version": release.network_version,
            "seeds": [recipe.seed for recipe in release.recipes],
            "higher_level": evaluation.higher_level,
        },
        "positions": POSITIONS_NOTE,
        "warnings": _warnings(rows),
        "bands": [asdict(row) | {"seed_range_cm": list(row.seed_range_cm)} for row in rows],
        "not_in_the_error": _not_in_the_error(release, molecule),
        "accuracy": evaluation.model_dump(mode="json") | {"licensed_families": within},
    }


def _band_lines(record: dict[str, Any]) -> list[str]:
    holdouts = list(record["accuracy"]["holdouts"])
    error_header = " / ".join(f"({name})" for name in holdouts)
    lines = [
        (
            "| position (cm⁻¹) | height (km/mol) | seeds' range | B3LYP | scaled B3LYP | family "
            f"| licensed | family's measured error {error_header}, cm⁻¹ |"
        ),
        "|---|---|---|---|---|---|---|---|",
    ]
    for band in record["bands"]:
        low, high = band["seed_range_cm"]
        errors = " / ".join(error_text(band["measured_error_cm"][name]) for name in holdouts)
        licensed = "yes" if band["licensed"] else f"no: {band['licence_note']}"
        lines.append(
            f"| {band['wavenumber_cm']} | {band['intensity_km_mol']} | {low}–{high} "
            f"| {band['b3lyp_cm']} | {band['scaled_b3lyp_cm']} | {band['family']} "
            f"| {licensed} | {errors} |"
        )
    return lines


def _accuracy_lines(accuracy: dict[str, Any]) -> list[str]:
    holdouts = accuracy["holdouts"]
    header = " | ".join(f"({name}), {h['n_molecules']} molecules" for name, h in holdouts.items())
    lines = [
        f"| family | {header} | within the licence line |",
        "|---|" + "---|" * (len(holdouts) + 1),
    ]
    for family, errors in accuracy["errors_cm"].items():
        per_seed = accuracy["seed_errors_cm"][family]
        cells = " | ".join(
            f"{error_text(errors[name])} ({', '.join(error_text(x) for x in per_seed[name])})"
            for name in holdouts
        )
        licensed = "yes" if family in accuracy["licensed_families"] else "no"
        lines.append(f"| {family} | {cells} | {licensed} |")
    lines += ["", *(f"- ({name}): {h['description']}" for name, h in holdouts.items())]
    lines.append(f"- sources: {'; '.join(accuracy['sources'])}")
    lines += ["", "A family is a rule, and its name is the evaluation's:", ""]
    return [*lines, *(f"- {family}: {rule}." for family, rule in families.RULES.items())]


def to_markdown(record: dict[str, Any]) -> str:
    """Render a certificate record as a Markdown document."""
    molecule, network = record["molecule"], record["network"]
    low, high = molecule["training_atoms"]
    size = "inside" if molecule["inside_training_sizes"] else "OUTSIDE: no band is licensed"
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
        "## What the measured error does not cover",
        "",
        (
            "Each hold-out molecule was read in one orientation and one atom order. For this "
            "molecule:"
        ),
        "",
        *(f"- {line[0].upper()}{line[1:]}." for line in record["not_in_the_error"]),
        "",
        "## How well the network is known",
        "",
        (
            "Seed-mean rms error of the corrected positions against "
            f"{level_name(network['higher_level'])}, in cm⁻¹, each seed's value in brackets."
        ),
        "",
        *_accuracy_lines(record["accuracy"]),
        "",
    ]
    return "\n".join(lines)
