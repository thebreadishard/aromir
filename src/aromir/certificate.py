"""The certificate of a result: each band with what it rests on.

A corrected spectrum is only as good as the network behind it, and the network is known
unevenly: well for some kinds of vibration, less well for others, not at all outside its
training domain. The certificate states, band by band, the family the band belongs to,
the error measured for that family on molecules the networks behind that error never saw
(for release networks: the version's measurement networks), whether the band is
licensed for this molecule, and how far the network's seeds disagree. It also states what
that error does not cover: how far the positions move when the same molecule is turned,
mirrored or listed in another atom order; where that is more than the licence line for
molecules with a linear group, no band of such a molecule is licensed. Every number about
the network comes from the release's records.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import cache
from typing import TYPE_CHECKING, Any, Final

from rdkit import Chem

from aromir import domain, families, repair, spectrum, vibrations
from aromir.admission import fixed_h_key
from aromir.network import primitives
from aromir.release import error_text, level_name, made_by, named_test_set

if TYPE_CHECKING:
    from collections.abc import Iterable

    from aromir.admission import Admission, Mol
    from aromir.lowlevel import LowLevel
    from aromir.molecule import Geometry
    from aromir.release import CationTest, Corrections, Evaluation, Release, Training

POSITIONS_NOTE: Final = "harmonic positions: the anharmonic shift is not included in this version"
LEVEL_NOTE: Final = "B3LYP/6-31G* corrected"
ATOM_ORDER: Final = "another atom order of the same molecule"
HEIGHTS_NOTE: Final = (
    "the heights are computed on the corrected modes with B3LYP's dipole derivatives, which "
    "are not corrected: a height carries no error bar"
)
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
OUTSIDE_SIZES: Final = "the molecule is outside the training sizes"
OUTSIDE_CHEMISTRY: Final = "the molecule holds chemistry no training molecule does"
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
            the training sizes and the training chemistry (``aromir.domain``), its
            low-level mode is real, the band is real for the
            seed mean and for every seed, the molecule's group (with or without a
            linear group) has its recorded spread within the licence line
            (``withheld_by_spread``),
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


@dataclass(frozen=True)
class Withheld:
    """Why no band of a molecule is licensed: its group's recorded spread.

    Attributes:
        spread_cm: The recorded spread above the licence line; None when it was not
            measured.
        note: What each band's licence column says.
        reason: The sentence the certificate's statement gives.
    """

    spread_cm: float | None
    note: str
    reason: str


CHANGE_PHRASES: Final = {
    "a rigid rotation of the molecule": "a rotation",
    "the molecule's mirror image": "a mirror image",
    "another listing of the same atoms in an xyz file": "another xyz listing",
}
"""The changes of the invariance record (``Invariance.spreads``) the licence reads, each
with the phrase its sentences use. A user's input can make all three: an xyz file in any
orientation, the other configuration of a molecule whose one stereo element is one centre
(``only_mirror_images``) and any listing of an xyz file's atoms. The evaluation's arbitrary
SMILES orders are not among them: aromir writes every SMILES of a molecule in one order."""
CHANGES: Final = ", ".join(list(CHANGE_PHRASES.values())[:-1]) + (
    f" or {list(CHANGE_PHRASES.values())[-1]}"
)
SPREAD_CHANGES: Final = f"{CHANGES} of the atoms"
STILL_PRINTED: Final = (
    "so the network's correction for this molecule is not licensed; the corrected positions "
    "are still printed"
)


@dataclass(frozen=True)
class GroupSpread:
    """What a release's invariance record says of the molecules of one group.

    Attributes:
        spread_cm: The largest spread, cm⁻¹, over the changes of ``CHANGE_PHRASES``
            measured on molecules of the group; None when none was.
        change: The phrase of the change that gave it; empty when none was measured.
        complete: Whether every one of those changes was measured on the group.
    """

    spread_cm: float | None
    change: str
    complete: bool


def group_spread(release: Release, *, linear_group: bool) -> GroupSpread:
    """Return the largest spread recorded for molecules with or without a linear group.

    Raises:
        ValueError: The record names other changes than ``CHANGE_PHRASES``: a change
            the licence does not read would otherwise stay out of the largest.
    """
    if release.invariance is None:
        return GroupSpread(spread_cm=None, change="", complete=False)
    spreads = release.invariance.spreads()
    if list(spreads) != list(CHANGE_PHRASES):
        msg = (
            f"the invariance record names the changes {list(spreads)}, the licence reads "
            f"{list(CHANGE_PHRASES)}"
        )
        raise ValueError(msg)
    measured = {
        CHANGE_PHRASES[name]: value
        for name, spread in spreads.items()
        if (value := spread.of(linear_group=linear_group)[0]) is not None
    }
    if not measured:
        return GroupSpread(spread_cm=None, change="", complete=False)
    change = max(measured, key=measured.__getitem__)
    return GroupSpread(
        spread_cm=measured[change], change=change, complete=len(measured) == len(spreads)
    )


def _too_far(spread: GroupSpread, line: float, *, linear_group: bool) -> Withheld:
    """Say that a molecule's group moved by more than the licence line."""
    value = spread.spread_cm
    if linear_group:
        return Withheld(
            spread_cm=value,
            note=(
                "the molecule has a linear group, and molecules with one moved by up to "
                f"{value:.2f} cm⁻¹ under {CHANGES}, more than the licence line"
            ),
            reason=(
                "the molecule has a linear group (an angle near 180°, as in a nitrile or an "
                f"alkyne), and the measured spread under {SPREAD_CHANGES} for molecules with "
                f"one ({value:.2f} cm⁻¹, from the release's invariance record) exceeds the "
                f"licence line of {line:g} cm⁻¹, {STILL_PRINTED}"
            ),
        )
    return Withheld(
        spread_cm=value,
        note=(
            f"molecules without a linear group moved by up to {value:.2f} cm⁻¹ under "
            f"{spread.change}, more than the licence line"
        ),
        reason=(
            "the molecule has no linear group, and the measured spread under "
            f"{spread.change} for molecules without one ({value:.2f} cm⁻¹, from the "
            f"release's invariance record) exceeds the licence line of {line:g} cm⁻¹, "
            f"{STILL_PRINTED}"
        ),
    )


UNMEASURED: Final = Withheld(
    spread_cm=None,
    note=(
        "the molecule has a linear group, and the spread of molecules with one under "
        f"{CHANGES} was not measured for this release"
    ),
    reason=(
        "the molecule has a linear group (an angle near 180°, as in a nitrile or an "
        f"alkyne), and the spread under {SPREAD_CHANGES} was not measured for this "
        f"release on a molecule with one, {STILL_PRINTED}"
    ),
)


def withheld_by_spread(release: Release, *, linear_group: bool) -> Withheld | None:
    """Say why a molecule's bands are not licensed for its group's spread; None when they may be.

    A molecule is licensed only while the release's invariance record shows the largest
    spread of its group (with or without a linear group) under ``CHANGE_PHRASES`` within
    the licence line. A release whose record does not measure every change on molecules
    with a linear group licenses no such molecule: a licence is a claim that needs its
    measurement, and v1.3 and v1.4, the releases measured, were both above the line. A
    molecule without one keeps its licence where its group was not measured, as before
    releases had the record: for it the licence is withheld only on a measured spread.
    """
    line = release.evaluation.licence_line_cm
    spread = group_spread(release, linear_group=linear_group)
    if spread.spread_cm is not None and spread.spread_cm > line:
        return _too_far(spread, line, linear_group=linear_group)
    if linear_group and not spread.complete:
        return UNMEASURED
    return None


def licence_note(
    mode: Mode,
    evaluation: Evaluation,
    *,
    outside_domain: str,
    withheld: Withheld | None,
) -> str:
    """Return the first condition of the licence a band fails, or "" when it passes all.

    ``outside_domain`` says why the molecule lies outside the training domain
    (``domain_note``); it is empty for a molecule inside it.
    """
    off_carbon = (
        f"{mode.off_carbon:.0%} of its hydrogen motion is on N–H, O–H or S–H, which "
        f"the rule counts as {mode.family}"
    )
    conditions = (
        (mode.low_level_cm < 0, FLAT_COORDINATE),
        (mode.position_cm < 0, IMAGINARY),
        (mode.lowest_seed_cm < 0, IMAGINARY_FOR_A_SEED),
        (withheld is not None, "" if withheld is None else withheld.note),
        (not evaluation.licensed(mode.family), "the family's error is above the licence line"),
        (bool(outside_domain), outside_domain),
        (mode.family in C_H_FAMILIES and mode.off_carbon > OFF_CARBON_SHARE, off_carbon),
    )
    return next((note for failed, note in conditions if failed), "")


def bands(
    geometry: Geometry,
    low: LowLevel,
    corrections: Corrections,
    evaluation: Evaluation,
    *,
    outside_domain: str,
    withheld: Withheld | None,
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
        note = licence_note(mode, evaluation, outside_domain=outside_domain, withheld=withheld)
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


UNSEEN: Final = "not in the training set or a test set: the network has not seen it"
UNSEEN_NEUTRAL: Final = (
    "not in the training set or a test set as a neutral molecule: the network has not seen it "
    "as one"
)
RELEASE_TRAINED: Final = (
    "in the release networks' training data: the error bars are not an out-of-sample error "
    "for this molecule; expect it to be smaller"
)


@cache
def _keys(listed: tuple[str, ...]) -> frozenset[str]:
    return frozenset(key for smiles in listed if (key := fixed_h_key(smiles)) is not None)


def _among(smiles: str, listed: Iterable[str]) -> bool:
    """Tell whether a molecule is in a list: by its standard SMILES, else by its FixedH key.

    The key finds a molecule whose admitted structure depends on how it was given: one
    past the resonance step's bounds, which keeps the structure it was given.
    """
    listed = tuple(listed)
    if smiles in listed:
        return True
    key = fixed_h_key(smiles)
    return key is not None and key in _keys(listed)


def only_mirror_images(smiles: str) -> bool:
    """Tell whether a molecule's one stereo element is one tetrahedral centre.

    Its configurations are then a molecule and its mirror image, whose infrared spectra
    are the same. Every potential stereo element counts, specified or not
    (``Chem.FindPotentialStereo``, as in RDKit's ``Code/GraphMol/Wrap/rough_test.py``,
    ``testFindPotentialStereo``).
    """
    molecule: Mol | None = Chem.MolFromSmiles(smiles)  # RDKit's stubs omit the None
    if molecule is None:
        return False
    elements = list(Chem.FindPotentialStereo(molecule))
    return len(elements) == 1 and elements[0].type == Chem.StereoType.Atom_Tetrahedral


def _covered(smiles: str, listed: Iterable[str]) -> bool:
    """Tell whether a list holds a molecule, or a row without stereo labels that stands for it.

    A row without labels stands for both mirror images: an xyz sulfoxide names its
    configuration where the training SMILES leave it open. It does not stand for a
    configuration of a molecule with an E/Z bond or two or more centres, which is another
    molecule with another spectrum (``_a_stereoisomer_in`` names it).
    """
    listed = tuple(listed)
    if _among(smiles, listed):
        return True
    plain = _without_stereo(smiles)
    return plain != smiles and only_mirror_images(smiles) and _among(plain, listed)


def _a_stereoisomer_in(smiles: str, listed: Iterable[str]) -> bool:
    """Tell whether a list holds the molecule with other, no or more stereo labels.

    Cautious, as the training project's module 08: a labelled mirror image of a labelled
    row is called a stereoisomer here, while a stereo-free row covers both (``_covered``).
    """
    plain = _without_stereo(smiles)
    return _among(plain, (_without_stereo(entry) for entry in listed))


STEREOISOMER: Final = (
    "the molecule with other or no stereo labels (a stereoisomer, or a configuration this "
    "request leaves open) is in the network's training set"
)
STEREOISOMER_RELEASE: Final = (
    "the molecule with other or no stereo labels (a stereoisomer, or a configuration this "
    "request leaves open) is in the release networks' training data"
)


def _met(smiles: str, training: Training) -> str:
    """Say whether the networks of a training record have met a molecule.

    A molecule held out, or trained on, with other or no stereo labels says so: never
    unseen.
    """
    if _covered(smiles, training.trained_smiles):
        return "in the network's training set: the test sets' error is not this molecule's measure"
    for holdout, held_out in training.held_out_smiles.items():
        if _covered(smiles, held_out):
            return f"in {named_test_set(holdout)}: never trained on; the measured error includes it"
    if _a_stereoisomer_in(smiles, training.trained_smiles):
        return STEREOISOMER
    for holdout, held_out in training.held_out_smiles.items():
        if _a_stereoisomer_in(smiles, held_out):
            return (
                "the molecule with other or no stereo labels (a stereoisomer, or a configuration "
                f"this request leaves open) is in {named_test_set(holdout)}: never trained on"
            )
    return UNSEEN


def _cation_clause(test: CationTest) -> str:
    networks = "one network" if test.n_networks == 1 else f"{test.n_networks} networks"
    return (
        "its radical cation is in the networks' training data; in a test, training on "
        f"{test.molecule}'s cation raised neutral {test.molecule}'s error from "
        f"{test.without_cm:.1f} to {test.with_cm:.1f} cm⁻¹ for {networks}, so this "
        "molecule's error may be larger than the measured one"
    )


def acquaintance(smiles: str, training: Training, measured: Training | None = None) -> str:
    """Say whether the networks have met a molecule, by its standard SMILES or FixedH key.

    ``training`` is the record of the networks that make the spectrum. ``measured`` is
    given when they are release networks: the record of the measurement networks their
    error bars belong to. A molecule in the release networks' training data is said to be
    there, for those error bars were measured on networks that did not train on it; any
    other molecule is read against the record the errors belong to. A row without stereo
    labels stands for both mirror images of a molecule whose one stereo element is one
    tetrahedral centre (``only_mirror_images``); any other stereoisomer of a trained or
    held-out molecule, and a request that leaves open what a row names, is said to be one,
    never unseen. A neutral molecule whose radical cation the networks trained on, while
    they did not train on the molecule itself, gets the clause of the cation test.
    """
    trained = _covered(smiles, training.trained_smiles)
    if measured is not None and trained:
        return RELEASE_TRAINED
    if measured is not None and _a_stereoisomer_in(smiles, training.trained_smiles):
        return STEREOISOMER_RELEASE
    said = _met(smiles, training if measured is None else measured)
    test = training.cation_test
    if test is None or trained or not _covered(smiles, training.cation_parent_smiles or ()):
        return said
    return f"{UNSEEN_NEUTRAL if said == UNSEEN else said}; {_cation_clause(test)}"


def has_linear_group(geometry: Geometry, release: Release) -> bool:
    """Tell whether the network's internal coordinates of a molecule hold a linear angle."""
    built = primitives.build(geometry, release.recipes[0].inputs.primitives)
    return any(primitives.kind(p) == primitives.LINEAR_ANGLE for p in built.internals)


def _molecule(admitted: Admission, geometry: Geometry, release: Release) -> dict[str, Any]:
    training = release.training
    inside = training.min_atoms <= admitted.n_atoms <= training.max_atoms
    return asdict(admitted) | {
        "unassigned_stereo": list(admitted.unassigned_stereo),
        "input_changes": list(admitted.input_changes),
        "training_atoms": [training.min_atoms, training.max_atoms],
        "inside_training_sizes": inside,
        "outside_training_chemistry": list(
            domain.outside_training_chemistry(admitted.smiles, training.trained_smiles)
        ),
        "seen": acquaintance(admitted.smiles, training, release.measurement),
        "linear_group": has_linear_group(geometry, release),
        "planar": families.is_planar(geometry),
    }


def domain_note(molecule: dict[str, Any]) -> str:
    """Say why a molecule's record puts it outside the training domain; "" when inside."""
    if not molecule["inside_training_sizes"]:
        return OUTSIDE_SIZES
    if molecule["outside_training_chemistry"]:
        return OUTSIDE_CHEMISTRY
    return ""


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
        unmeasured = (
            "the spread between arbitrary atom orders (the evaluation's atom-order spread) was "
            "not measured for this release"
        )
        if release.invariance is not None:
            unmeasured += (
                "; the spread between xyz listings put into aromir's canonical order is the "
                "one above"
            )
        lines.append(unmeasured)
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
    admitted: Admission,
    release: Release,
    rows: list[Band],
    molecule: dict[str, Any],
    withheld: Withheld | None,
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
    test_sets = " and ".join(named_test_set(name, first=True) for name in evaluation.holdouts)
    spread = "" if withheld is None else f"{withheld.reason[0].upper()}{withheld.reason[1:]}. "
    group = "with" if molecule["linear_group"] else "without"
    return (
        f"{admitted.smiles}: the harmonic infrared spectrum of ΔH network "
        f"v{release.network_version} ({len(release.networks)} seeds), {LEVEL_NOTE} towards "
        f"{evaluation.higher_level}{', ' if release.shipped is None else '. '}{made_by(release)} "
        f"Those molecules are {test_sets}. "
        f"{sum(row.licensed for row in rows)} of {len(rows)} bands are licensed: a band "
        f"is licensed when its family's error is within {line:g} cm⁻¹ on every test set "
        f"({', '.join(within) or 'no family'}), the molecule lies inside the training sizes "
        "and holds no charge separation or aromatic ring size the training molecules lack, "
        "its low-level mode is real, the band is real for the seeds' mean and for every "
        f"seed, molecules {group} a linear group moved by no more than the licence line "
        f"under {SPREAD_CHANGES} in the release's invariance record, and a band of a C–H "
        "family moves carbon-bound "
        f"hydrogens. {spread}{flat}{thin}{beyond}"
        f"Coupled cluster: {evaluation.coupled_cluster}. {_not_experiment(evaluation)}"
    )


def _not_experiment(evaluation: Evaluation) -> str:
    """Say what the errors are against, and what to compare a measured spectrum with."""
    return (
        f"The errors are against {level_name(evaluation.higher_level)}, not against "
        "experiment: a corrected position estimates that level's harmonic position, which for "
        "nearly all modes lies above the measured band, so compare a measured spectrum with "
        "the scaled B3LYP positions, printed beside each band as the baseline."
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
    withheld = withheld_by_spread(release, linear_group=molecule["linear_group"])
    rows = bands(
        geometry,
        low,
        corrections,
        evaluation,
        outside_domain=domain_note(molecule),
        withheld=withheld,
    )
    return {
        "statement": _statement(admitted, release, rows, molecule, withheld),
        "molecule": molecule,
        "network": {
            "version": release.network_version,
            "seeds": [recipe.seed for recipe in release.recipes],
            "tensor_sha256": release.seed_sha256,
            "release": release.release_name,
            "inside_package": release.inside_package,
            "higher_level": evaluation.higher_level,
            "kind": "measurement" if release.shipped is None else "release",
            "errors_measured_on": (
                None if release.shipped is None else release.shipped.measurement_checkpoints
            ),
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
    error_header = "on test set " + " / ".join(f"({name})" for name in holdouts)
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
    header = " | ".join(
        f"{named_test_set(name)}, {h['n_molecules']} molecules" for name, h in holdouts.items()
    )
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
    lines += [
        *(
            f"- {named_test_set(name, first=True)} — {h['description']}"
            for name, h in holdouts.items()
        )
    ]
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


def _whose_error(kind: str) -> str:
    """Say whose error the table is: the printed positions', or their error bar."""
    if kind == "measurement":
        return "what this certificate's positions are"
    return (
        "the error of the version's measurement networks, the same recipe trained without the "
        "test sets, and the error bar of this certificate's positions"
    )


def _network_line(network: dict[str, Any]) -> str:
    """Say which release and which weights, by their tensors' SHA-256, made the spectrum."""
    where = "inside the package" if network["inside_package"] else "a folder outside the package"
    tensors = ", ".join(f"{name} `{sha256}`" for name, sha256 in network["tensor_sha256"].items())
    return (
        f"*Network:* release v{network['version']}, `{network['release']}` ({where}); the "
        f"seeds' tensor files by SHA-256: {tensors}."
    )


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
        *(
            f"- Outside the training chemistry: {caveat}: no band is licensed."
            for caveat in molecule["outside_training_chemistry"]
        ),
        f"- {molecule['seen']}",
        *(f"- {change[0].upper()}{change[1:]}." for change in molecule["input_changes"]),
        "",
        (
            f"## Spectrum, band by band (network v{network['version']}, "
            f"seeds {', '.join(str(seed) for seed in network['seeds'])})"
        ),
        "",
        *_band_lines(record),
        "",
        _network_line(network),
        "",
        f"*Positions:* {record['positions']}.",
        *([f"*Soft modes:* {record['soft_modes']}."] if record["soft_modes"] else []),
        *(f"**Warning:** {warning}." for warning in record["warnings"]),
        "",
        "## What the measured error does not cover",
        "",
        (
            "Each test-set molecule was read in one orientation and one atom order. For this "
            "molecule:"
        ),
        "",
        *(f"- {line[0].upper()}{line[1:]}." for line in record["not_in_the_error"]),
        "",
        "## How well the network is known",
        "",
        (
            "The rms error of the corrected band positions of the seeds' mean against "
            f"{level_name(network['higher_level'])}, in cm⁻¹: "
            f"{_whose_error(network['kind'])}. In square brackets the 95 % interval when the "
            "test set's molecules are resampled. The licence is read on "
            f"{record['accuracy']['licence_basis']}."
        ),
        "",
        *_accuracy_lines(record["accuracy"]),
        "",
    ]
    return "\n".join(lines)
