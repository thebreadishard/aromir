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

from rdkit import Chem

from aromir import families, repair, spectrum, vibrations
from aromir.network import primitives
from aromir.release import error_text, level_name

if TYPE_CHECKING:
    from aromir.admission import Admission
    from aromir.lowlevel import LowLevel
    from aromir.molecule import Geometry
    from aromir.release import Corrections, Evaluation, Release, Training

POSITIONS_NOTE: Final = "harmonic positions: the anharmonic shift is not included in this version"
LEVEL_NOTE: Final = "B3LYP/6-31G* corrected"
ATOM_ORDER: Final = "another atom order of the same molecule"
HEIGHTS_NOTE: Final = "the heights are B3LYP's: the dipole derivatives are not corrected"
CONFORMER_NOTE: Final = (
    "one conformer was computed, the one the optimisation reached from aromir's start "
    "geometry; another conformer has another spectrum"
)
NOT_PLANAR_NOTE: Final = (
    "the molecule is not planar, so no band counts as CH-oop: its C–H wags are printed as "
    "other or, between 950 and 1050 cm⁻¹, as ring-ip"
)
C_H_FAMILIES: Final[tuple[families.Family, ...]] = ("CH-stretch", "CH-oop")
"""The families whose name promises carbon-bound hydrogens."""
FLAT_COORDINATE: Final = (
    "a flat coordinate: its low-level mode is imaginary, so this is no band position"
)
"""The molecule is at no minimum along this mode, which is softer than the soft-mode line:
the pipeline prints the spectrum there and claims nothing for this band."""
IMAGINARY: Final = "imaginary position"
IMAGINARY_FOR_A_SEED: Final = "imaginary for at least one seed"
OFF_CARBON_SHARE: Final = 0.1
"""A band of a C–H family is not licensed when more than this share of its hydrogen motion
is on hydrogens bound to another element: an O–H or N–H stretch or wag, or a mode that
mixes one with the C–H motion."""


@dataclass(frozen=True)
class Band:
    """One band of the corrected spectrum.

    Attributes:
        wavenumber_cm: The corrected harmonic position.
        intensity_km_mol: The double-harmonic height.
        seed_range_cm: The lowest and highest position of this band over the seeds and
            their mean. The printed position is an eigenvalue of the Hessian with the mean
            correction, which can lie outside the seeds' own positions.
        b3lyp_cm: The position of the uncorrected mode that moves most like it.
        scaled_b3lyp_cm: That mode's position under the conventional scale factors.
        family: The mode family, by the rule the network's error is measured in.
        licensed: Whether the band passes every condition of the licence: its family's
            error is within the licence line on every hold-out, the molecule lies inside
            the training sizes, its low-level mode is real, the band is real for the
            seed mean and for every seed,
            and a band of a C–H family moves carbon-bound hydrogens.
        licence_note: The first condition the band fails; empty when it is licensed.
        measured_error_cm: The family's band error of the seeds' mean, per hold-out.
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


@dataclass(frozen=True)
class Mode:
    """What the licence reads of one band before rounding.

    Attributes:
        family: The band's mode family.
        low_level_cm: The position of the uncorrected mode that moves most like it.
        position_cm: The corrected position of the seeds' mean.
        lowest_seed_cm: The lowest position of the band over the seeds.
        off_carbon: The share of its hydrogen motion on hydrogens not bound to carbon.
    """

    family: families.Family
    low_level_cm: float
    position_cm: float
    lowest_seed_cm: float
    off_carbon: float


def licence_note(mode: Mode, evaluation: Evaluation, *, inside_training_sizes: bool) -> str:
    """Return the first condition of the licence a band fails, or "" when it passes all."""
    off_carbon = (
        f"{mode.off_carbon:.0%} of its hydrogen motion is on N–H, O–H or S–H, which "
        f"the rule counts as {mode.family}"
    )
    conditions = (
        (mode.low_level_cm < 0, FLAT_COORDINATE),
        (mode.position_cm < 0, IMAGINARY),
        (mode.lowest_seed_cm < 0, IMAGINARY_FOR_A_SEED),
        (not evaluation.licensed(mode.family), "the family's error is above the licence line"),
        (not inside_training_sizes, "the molecule is outside the training sizes"),
        (mode.family in C_H_FAMILIES and mode.off_carbon > OFF_CARBON_SHARE, off_carbon),
    )
    return next((note for failed, note in conditions if failed), "")


def bands(
    geometry: Geometry,
    low: LowLevel,
    corrections: Corrections,
    evaluation: Evaluation,
    *,
    inside_training_sizes: bool,
) -> list[Band]:
    """Return the corrected spectrum's bands in ascending order, each with its record."""
    low_modes = vibrations.vibrational_modes(low.hessian, geometry)
    low_families = families.classify(geometry, low_modes)
    off_carbon = families.hydrogen_motion_off_carbon(geometry, low_modes)
    scaled = spectrum.scaled(spectrum.harmonic_bands(low.hessian, geometry, low.polar_tensor))
    corrected_hessian = low.hessian + corrections.mean
    corrected = spectrum.harmonic_bands(corrected_hessian, geometry, low.polar_tensor)
    partner = spectrum.partners(
        vibrations.vibrational_modes(corrected_hessian, geometry), low_modes
    )
    ranges = spectrum.seed_ranges(low.hessian, geometry, corrections.per_seed)
    rows = []
    for index, low_index in enumerate(partner):
        mode = Mode(
            family=low_families[low_index],
            low_level_cm=float(low_modes.wavenumbers[low_index]),
            position_cm=float(corrected.wavenumbers[index]),
            lowest_seed_cm=float(ranges[index, 0]),
            off_carbon=float(off_carbon[low_index]),
        )
        note = licence_note(mode, evaluation, inside_training_sizes=inside_training_sizes)
        rows.append(
            Band(
                wavenumber_cm=round(mode.position_cm, 1),
                intensity_km_mol=round(float(corrected.intensities[index]), 2),
                seed_range_cm=(
                    round(min(float(ranges[index, 0]), mode.position_cm), 1),
                    round(max(float(ranges[index, 1]), mode.position_cm), 1),
                ),
                b3lyp_cm=round(float(low_modes.wavenumbers[low_index]), 1),
                scaled_b3lyp_cm=round(float(scaled.wavenumbers[low_index]), 1),
                family=mode.family,
                licensed=not note,
                licence_note=note,
                measured_error_cm={
                    holdout: error.rms_cm
                    for holdout, error in evaluation.band_error_seed_mean_cm[mode.family].items()
                },
            )
        )
    return rows


def _without_stereo(smiles: str) -> str:
    """Return a SMILES with its stereo labels dropped, canonical again."""
    molecule = Chem.MolFromSmiles(smiles)
    return smiles if molecule is None else str(Chem.MolToSmiles(molecule, isomericSmiles=False))


def acquaintance(smiles: str, training: Training) -> str:
    """Say whether the network has met a molecule, by its canonical SMILES.

    A molecule written without its stereo labels is compared without them too, so that a
    trained stereoisomer is not reported as never seen.
    """
    if smiles in training.trained_smiles:
        return "in the network's training set: the hold-out error is not this molecule's measure"
    for holdout, held_out in training.held_out_smiles.items():
        if smiles in held_out:
            return f"in hold-out ({holdout}): never trained on; the measured error includes it"
    plain = _without_stereo(smiles)
    if plain in {_without_stereo(trained) for trained in training.trained_smiles}:
        return (
            "a stereoisomer of it, or the molecule itself written with other stereo labels, "
            "is in the network's training set"
        )
    return "not in the training set or a hold-out: the network has not seen it"


def has_linear_group(geometry: Geometry, release: Release) -> bool:
    """Tell whether the network's internal coordinates of a molecule hold a linear angle."""
    built = primitives.build(geometry, release.recipes[0].inputs.primitives)
    return any(primitives.kind(p) == primitives.LINEAR_ANGLE for p in built.internals)


def _molecule(admitted: Admission, geometry: Geometry, release: Release) -> dict[str, Any]:
    training = release.training
    inside = training.min_atoms <= admitted.n_atoms <= training.max_atoms
    return asdict(admitted) | {
        "unassigned_stereo": list(admitted.unassigned_stereo),
        "training_atoms": [training.min_atoms, training.max_atoms],
        "inside_training_sizes": inside,
        "seen": acquaintance(admitted.smiles, training),
        "linear_group": has_linear_group(geometry, release),
        "planar": families.is_planar(geometry),
    }


def spreads_cm(release: Release, *, linear_group: bool) -> dict[str, tuple[float, int]]:
    """Return how far positions were seen to move under each change that leaves a molecule itself.

    Each entry is the largest change, cm⁻¹, and the number of molecules it was seen on:
    for the atom order all that were measured, for the others those that, like the
    molecule at hand, have or lack a linear group. The family errors do not hold these:
    each hold-out molecule was read in one orientation and one atom order.
    """
    spreads = {}
    order = release.evaluation.atom_order_spread
    if order is not None:
        spreads[ATOM_ORDER] = (order.max_cm, order.n_molecules)
    if release.invariance is not None:
        for name, spread in release.invariance.spreads().items():
            value, n_molecules = spread.of(linear_group=linear_group)
            if value is not None:
                spreads[name] = (value, n_molecules)
    return spreads


def _not_in_the_error(release: Release, molecule: dict[str, Any]) -> list[str]:
    """Return what a reader must add to a band's measured error, one sentence each."""
    line = release.evaluation.licence_line_cm
    group = "with" if molecule["linear_group"] else "without"
    lines = []
    for name, (value, n) in spreads_cm(release, linear_group=molecule["linear_group"]).items():
        # The atom orders tried were arbitrary ones; aromir itself writes a SMILES one way.
        which = " in arbitrary orders" if name == ATOM_ORDER else f" {group} a linear group"
        above = f"; that is more than the licence line of {line:g} cm⁻¹" if value > line else ""
        lines.append(
            f"{name} moved a position by up to {value:.2f} cm⁻¹, the largest seen on {n} "
            f"molecule(s){which}{above}"
        )
    if release.evaluation.atom_order_spread is None:
        lines.append("the spread between atom orders was not measured for this release")
    if release.invariance is None:
        lines.append(
            "the spread under a rotation, a reflection and another listing of an xyz file's "
            "atoms was not measured for this release"
        )
    elif any(
        spread.of(linear_group=molecule["linear_group"])[0] is None
        for spread in release.invariance.spreads().values()
    ):
        lines.append(
            "the spread under a rotation, a reflection or another listing of an xyz file's "
            f"atoms was not measured on a molecule {group} a linear group"
        )
    lines += [HEIGHTS_NOTE, CONFORMER_NOTE]
    if molecule["unassigned_stereo"]:
        lines.append(
            f"the input leaves {' and '.join(molecule['unassigned_stereo'])} unassigned: one "
            "isomer was computed, the one the start geometry happened to be"
        )
    if not molecule["planar"]:
        lines.append(NOT_PLANAR_NOTE)
    return lines


def _statement(
    admitted: Admission, release: Release, rows: list[Band], molecule: dict[str, Any]
) -> str:
    evaluation = release.evaluation
    line = evaluation.licence_line_cm
    within = [family for family in families.FAMILIES if evaluation.licensed(family)]
    seen = [
        value for value, _ in spreads_cm(release, linear_group=molecule["linear_group"]).values()
    ]
    thin = "".join(f"{caveat[0].upper()}{caveat[1:]}. " for caveat in evaluation.caveats())
    beyond = ""
    if seen and max(seen) > line:
        beyond = (
            "Not in that error: the same molecule turned, mirrored or listed in another "
            f"atom order was seen to get positions up to {max(seen):.2f} cm⁻¹ away, more "
            "than the licence line. "
        )
    n_flat = sum(row.licence_note == FLAT_COORDINATE for row in rows)
    flat = ""
    if n_flat:
        flat = (
            f"{n_flat} band(s) lie on a flat coordinate, along which the molecule is at no "
            "minimum of the low level: the number printed there is no band position and "
            "nothing is claimed for it. "
        )
    return (
        f"{admitted.smiles}: the harmonic infrared spectrum of ΔH network "
        f"v{release.network_version} ({len(release.networks)} seeds), {LEVEL_NOTE} towards "
        f"{evaluation.higher_level}, with the error measured on molecules the network never "
        f"saw. {sum(row.licensed for row in rows)} of {len(rows)} bands are licensed: a band "
        f"is licensed when its family's error is within {line:g} cm⁻¹ on every hold-out "
        f"({', '.join(within) or 'no family'}), the molecule lies inside the training sizes, "
        "its low-level mode is real, the band is real for the seeds' mean and for every "
        "seed, and a band of a C–H "
        "family moves carbon-bound "
        f"hydrogens. {flat}{thin}{beyond}"
        f"Coupled cluster: {evaluation.coupled_cluster}. The scaled B3LYP positions are a "
        "baseline for comparison only."
    )


def _soft_mode_note(low: LowLevel, release: Release) -> str:
    """Say what the soft-mode repair did to this molecule's Hessian, or "" for nothing."""
    measured = low.soft_modes
    if measured is None:
        return ""
    note = (
        f"{measured.n_directions} mode(s) below {repair.SOFT_BELOW_CM:g} cm⁻¹ were measured from "
        "gradients that include the integration grid's response; the lowest moved from "
        f"{measured.lowest_before_cm:.1f} to {measured.lowest_after_cm:.1f} cm⁻¹"
    )
    if release.training.soft_mode_repair:
        return note
    return (
        f"{note}. The network read the Hessian without that repair, the kind it was trained "
        "on, and its correction was added to the measured one; the family errors were "
        "measured without the repair, so they say nothing of these modes"
    )


def _warnings(rows: list[Band]) -> list[str]:
    """Return what a reader must know before using the bands at all."""
    n_flat = sum(row.licence_note == FLAT_COORDINATE for row in rows)
    n_imaginary = sum(row.licence_note == IMAGINARY for row in rows)
    n_for_a_seed = sum(row.licence_note == IMAGINARY_FOR_A_SEED for row in rows)
    warnings = []
    if n_flat:
        warnings.append(
            f"{n_flat} band(s) lie on a flat coordinate: the geometry is no minimum along it "
            f"at the low level (an imaginary mode softer than {repair.SOFT_BELOW_CM:g} cm⁻¹, "
            "also after a step off the saddle point and a second optimisation). The number "
            "printed for such a band is not a band position and is never licensed"
        )
    if n_imaginary:
        warnings.append(
            f"{n_imaginary} corrected band(s) have an imaginary (negative) position: the "
            "correction made a soft mode unstable, and those bands are not physical"
        )
    if n_for_a_seed:
        warnings.append(
            f"{n_for_a_seed} corrected band(s) are imaginary for at least one seed, though "
            "not for the seeds' mean"
        )
    return warnings


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
    molecule = _molecule(admitted, geometry, release)
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
        "soft_modes": _soft_mode_note(low, release),
        "warnings": _warnings(rows),
        "bands": [asdict(row) | {"seed_range_cm": list(row.seed_range_cm)} for row in rows],
        "not_in_the_error": _not_in_the_error(release, molecule),
        "accuracy": evaluation.model_dump(mode="json")
        | {
            "licensed_families": within,
            "on_thin_ground": [f for f in within if evaluation.thin_ground(f)],
            "caveats": evaluation.caveats(),
        },
    }


def _band_lines(record: dict[str, Any]) -> list[str]:
    holdouts = list(record["accuracy"]["holdouts"])
    error_header = " / ".join(f"({name})" for name in holdouts)
    lines = [
        (
            "| position (cm⁻¹) | height (km/mol) | range over seeds and mean | B3LYP "
            "| scaled B3LYP | family "
            f"| licensed | family's measured error {error_header}, cm⁻¹ |"
        ),
        "|---|---|---|---|---|---|---|---|",
    ]
    for band in record["bands"]:
        low, high = band["seed_range_cm"]
        errors = " / ".join(error_text(band["measured_error_cm"][name]) for name in holdouts)
        licensed = "yes" if band["licensed"] else f"no: {band['licence_note']}"
        if band["licensed"] and band["family"] in record["accuracy"]["on_thin_ground"]:
            licensed = "yes, on thin ground (see above)"
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
    for family, errors in accuracy["band_error_seed_mean_cm"].items():
        cells = " | ".join(
            f"{error_text(errors[name]['rms_cm'])} "
            f"[{error_text(errors[name]['interval_cm'][0])}–"
            f"{error_text(errors[name]['interval_cm'][1])}], {errors[name]['n_modes']} bands"
            for name in holdouts
        )
        licensed = "yes" if family in accuracy["licensed_families"] else "no"
        lines.append(f"| {family} | {cells} | {licensed} |")
    lines += ["", *(f"- {caveat[0].upper()}{caveat[1:]}." for caveat in accuracy["caveats"])]
    lines += [*(f"- ({name}): {h['description']}" for name, h in holdouts.items())]
    lines.append(f"- sources: {'; '.join(accuracy['sources'])}")
    lines += [
        "",
        (
            "For comparison, the first-order error of each seed alone, the number a training "
            "run reports: its mean over the seeds, each seed's value in brackets. It is not "
            "the error of what is printed above."
        ),
        "",
        f"| family | {header} |",
        "|---|" + "---|" * len(holdouts),
    ]
    for family, means in accuracy["first_order_error_per_seed_mean_cm"].items():
        per_seed = accuracy["first_order_error_per_seed_cm"][family]
        cells = " | ".join(
            f"{error_text(means[name])} ({', '.join(error_text(x) for x in per_seed[name])})"
            for name in holdouts
        )
        lines.append(f"| {family} | {cells} |")
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
        *([f"*Soft modes:* {record['soft_modes']}."] if record["soft_modes"] else []),
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
            "The rms error of the corrected band positions of the seeds' mean against "
            f"{level_name(network['higher_level'])}, in cm⁻¹: what this certificate's "
            "positions are. In square brackets the 95 % interval when the hold-out's "
            "molecules are resampled. The licence is read on "
            f"{record['accuracy']['licence_basis']}."
        ),
        "",
        *_accuracy_lines(record["accuracy"]),
        "",
    ]
    return "\n".join(lines)
