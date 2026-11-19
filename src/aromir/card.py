"""The model card of a network release, written from the release itself.

A model card says what a network is, what it was trained on, how well it is known and
where it stops being known. Every number on this one is read from the release's files:
the recipes, the evaluation record and the training record. The prose around the numbers
is fixed here; a statement that depends on how the network was built is chosen from the
recipe, not written by hand.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from aromir import dft, families, repair
from aromir.release import (
    EVALUATION_FILE,
    INVARIANCE_FILE,
    MEASURED_FILE,
    MEASUREMENT_FILE,
    RECORDED_FILES,
    RELEASE_FILE,
    TRAINING_FILE,
    error_text,
    level_name,
    made_by,
    named_test_set,
    pool,
)

if TYPE_CHECKING:
    from aromir.release import Evaluation, Release, Training

LICENCE: Final = "cc-by-4.0"
REPOSITORY: Final = "https://github.com/thebreadishard/aromir"


def _front_matter(release: Release) -> list[str]:
    """Return the metadata block model hubs read."""
    return [
        "---",
        f"license: {LICENCE}",
        "library_name: aromir",
        "tags:",
        "  - chemistry",
        "  - infrared-spectroscopy",
        "  - hessian",
        "  - equivariant",
        f"model_name: aromir ΔH network v{release.network_version}",
        "---",
    ]


def _summary(release: Release) -> list[str]:
    evaluation = release.evaluation
    licensed = [family for family in families.FAMILIES if evaluation.licensed(family)]
    unlicensed = [family for family in families.FAMILIES if family not in licensed]
    test_sets = " and on ".join(named_test_set(name, first=True) for name in evaluation.holdouts)
    return [
        f"# aromir ΔH network v{release.network_version}",
        "",
        (
            "A correction to the Hessian of an aromatic molecule. The network reads a cheap "
            f"{dft.FUNCTIONAL.upper()}/{dft.BASIS.upper()} Hessian and returns ΔH, the difference "
            f"to {evaluation.higher_level}. The corrected Hessian gives harmonic infrared band "
            "positions closer to that level at the cost of the cheap one."
        ),
        "",
        (
            f"- **Seeds:** {len(release.networks)} "
            f"({', '.join(str(recipe.seed) for recipe in release.recipes)}); the applied "
            "correction is their mean, and their spread is reported per band."
        ),
        *([] if release.shipped is None else [f"- **Networks:** {made_by(release)}"]),
        (
            f"- **Licensed families:** {', '.join(licensed) or 'none'} (within "
            f"{evaluation.licence_line_cm:g} cm⁻¹ on {test_sets})."
        ),
        f"- **Not licensed:** {', '.join(unlicensed) or 'none'}.",
        *(f"- **On thin ground:** {caveat}." for caveat in evaluation.caveats()),
        (
            f"- **Use it through:** [aromir]({REPOSITORY}), which builds the inputs, applies the "
            "correction and writes a certificate for every band."
        ),
    ]


def _evaluation_table(evaluation: Evaluation) -> list[str]:
    holdouts = evaluation.holdouts
    header = " | ".join(
        f"{named_test_set(name)}, {h.n_molecules} molecules" for name, h in holdouts.items()
    )
    lines = [f"| family | {header} | licensed |", "|---|" + "---|" * (len(holdouts) + 1)]
    for family in families.FAMILIES:
        errors = evaluation.band_error_seed_mean_cm[family]
        cells = " | ".join(
            f"{error_text(errors[name].rms_cm)} [{error_text(errors[name].interval_cm[0])}–"
            f"{error_text(errors[name].interval_cm[1])}], {errors[name].n_modes} bands"
            for name in holdouts
        )
        licensed = "yes" if evaluation.licensed(family) else "no"
        lines.append(f"| {family} | {cells} | {licensed} |")
    return lines


def _first_order_table(evaluation: Evaluation) -> list[str]:
    holdouts = evaluation.holdouts
    header = " | ".join(named_test_set(name) for name in holdouts)
    lines = [f"| family | {header} |", "|---|" + "---|" * len(holdouts)]
    for family in families.FAMILIES:
        means = evaluation.first_order_error_per_seed_mean_cm[family]
        per_seed = evaluation.first_order_error_per_seed_cm[family]
        cells = " | ".join(
            f"{error_text(means[name])} ({', '.join(error_text(x) for x in per_seed[name])})"
            for name in holdouts
        )
        lines.append(f"| {family} | {cells} |")
    return lines


def _evaluation(release: Release) -> list[str]:
    evaluation = release.evaluation
    networks = (
        "the network never trained on: the error of what aromir prints"
        if release.shipped is None
        else "the measurement networks never trained on — the same recipe trained without "
        "the test sets, whose error is the error bar of what aromir prints; the release "
        "networks aromir runs trained on these molecules too"
    )
    return [
        "## How well it is known",
        "",
        (
            "The rms error of the corrected band positions of the seeds' mean against "
            f"{level_name(evaluation.higher_level)}, in cm⁻¹, on molecules {networks}. "
            "In square brackets the 95 % "
            "interval when the test set's molecules are resampled. The licence is read on "
            f"{evaluation.licence_basis}."
        ),
        "",
        *_evaluation_table(evaluation),
        "",
        *(
            f"- {named_test_set(name, first=True)} — {h.description}"
            for name, h in evaluation.holdouts.items()
        ),
        "",
        (
            "For comparison, the first-order error of each seed alone, the number a training "
            "run reports and the one earlier versions of this card printed as the error: its "
            "mean over the seeds, each seed's value in brackets. It is not the error of what "
            "aromir prints."
        ),
        "",
        *_first_order_table(evaluation),
        "",
        (
            "A family is licensed when the band error of the seeds' mean, as recorded to two "
            "decimals, is within "
            f"{evaluation.licence_line_cm:g} cm⁻¹ on every test set. aromir's certificate marks "
            "each band with its family, that family's error and whether the band itself is "
            "licensed, which asks more: see the limits below."
        ),
        "",
        f"Coupled cluster: {evaluation.coupled_cluster}.",
        "",
        f"Sources: {'; '.join(evaluation.sources)}.",
    ]


def _held_out(training: Training) -> str:
    """Count each hold-out's molecules the network did not train on, and name the others.

    A molecule can sit in a hold-out and, under another identifier, in the training set.
    It was trained on, so it is not counted as held out; the card says how many there are.
    """
    trained = set(training.trained_smiles)
    parts = []
    for name, smiles in training.held_out_smiles.items():
        held_out = set(smiles)
        n_trained = len(held_out & trained)
        part = f"{len(held_out) - n_trained} in {named_test_set(name)}"
        if n_trained:
            part += f", not counting {n_trained} the network also trained on"
        parts.append(part)
    return "; ".join(parts) or "none"


def _molecule_count(release: Release) -> str:
    training = release.training
    shipped = release.shipped
    if shipped is None:
        n = len(training.trained_smiles)
        return f"{n} molecules, neutral and closed-shell, each with an aromatic ring."
    neutral = shipped.n_entries - len(training.cation_parent_smiles or ())
    distinct = len(set(training.trained_smiles))
    twice = (
        f" (the {neutral} neutral entries hold {distinct} distinct molecules: a molecule may "
        "be an entry under more than one identifier)"
        if neutral != distinct
        else ""
    )
    return (
        f"{shipped.n_entries} labelled entries of the training pool, each a neutral "
        f"closed-shell molecule or the radical cation of one, with an aromatic ring: "
        f"{pool(training)}{twice}. The measurement networks, on which every error is read, "
        "trained on "
        f"{len(release.measured_training.trained_smiles)} of those neutral molecules; the "
        "release networks on all of the entries, the test sets' molecules included."
    )


def _training(release: Release) -> list[str]:
    training = release.training
    held_out = f"- Held out and never trained on: {_held_out(training)}."
    if release.shipped is not None:
        held_out = (
            "- Held out from the measurement networks, on which every error is read: "
            f"{_held_out(release.measured_training)}; the release networks trained on these "
            "molecules."
        )
    prepared = (
        "with their soft modes (below "
        f"{repair.SOFT_BELOW_CM:g} cm⁻¹) measured from gradients that include the grid's response"
        if training.soft_mode_repair
        else "as pyscf's analytic Hessians, without the soft-mode repair"
    )
    return [
        "## What it was trained on",
        "",
        f"- {_molecule_count(release)}",
        f"- Elements: {', '.join(training.elements)}.",
        f"- Sizes: {training.min_atoms} to {training.max_atoms} atoms.",
        held_out,
        (
            f"- The training Hessians were used {prepared}; aromir gives the network a "
            "molecule's Hessian prepared the same way. It measures the soft modes in either "
            "case, to decide whether the geometry is a minimum and for the positions it "
            "prints: the correction is added to the Hessian with its soft modes measured."
        ),
    ]


def _measured_spread(release: Release) -> str:
    spread = release.evaluation.atom_order_spread
    if spread is None:
        return "The spread between atom orders was not measured for this release."
    return (
        f"Measured: band positions move by up to {spread.max_cm:g} cm⁻¹ between atom orders "
        f"({spread.n_molecules} molecules, {spread.n_orders} orders each; {spread.source})."
    )


def _order_statement(release: Release) -> str:
    primitives = release.recipes[0].inputs.primitives
    if primitives.angle_choice == "canonical" and primitives.dihedral_encoding != "raw":
        design = (
            "The internal coordinates are built on the atoms in a canonical order and "
            "dihedrals enter by their cosine, which is meant to make the answer independent "
            "of how the molecule's atoms are numbered; the measurements say how far it is."
        )
    else:
        design = (
            "**The answer depends on how the molecule's atoms are numbered**: the internal "
            "coordinates are chosen by atom number, and a dihedral near ±180° enters by its "
            "raw value. aromir orders the atoms of every input by the bond graph first, so "
            "every SMILES of a molecule gets one order; atoms the graph cannot tell apart "
            "keep the order of an xyz file, so two listings of one geometry can still "
            "differ. Do not feed the network a molecule in another order."
        )
    return f"{design} {_measured_spread(release)}"


def _sameness(release: Release) -> list[str]:
    """Return what the answer depends on that the molecule does not, as measured."""
    invariance = release.invariance
    if invariance is None:
        unmeasured = (
            "- **The same molecule turned or mirrored** may get another answer; by how much "
            "was not measured for this release."
        )
        return [unmeasured]
    lines = []
    for name, spread in invariance.spreads().items():
        seen = " and ".join(
            f"{value:.2f} cm⁻¹ on {n} molecule(s) {group} a linear group"
            for group, (value, n) in (
                ("with", spread.of(linear_group=True)),
                ("without", spread.of(linear_group=False)),
            )
            if value is not None
        )
        lines.append(
            f"- **{name[0].upper()}{name[1:]}** moved band positions by up to {seen} "
            f"({spread.n_trials} trial(s) per molecule). These are the largest seen, not bounds."
        )
    reason = (
        "A linear angle (in a nitrile, an alkyne) is measured against a fixed axis in "
        "space, and a dihedral enters with its sign"
        if release.recipes[0].inputs.primitives.dihedral_encoding == "raw"
        else "A linear angle (in a nitrile, an alkyne) is measured against a fixed axis in space"
    )
    return [
        *lines,
        f"  {reason}; the test sets' errors hold none of this. Source: {invariance.source}.",
    ]


def _limits(release: Release) -> list[str]:
    evaluation, training = release.evaluation, release.training
    unlicensed = [f for f in families.FAMILIES if not evaluation.licensed(f)]
    return [
        "## Where it stops being known",
        "",
        (
            f"- **The level is {level_name(evaluation.higher_level)}.** The errors above are "
            "against that level, not against experiment."
        ),
        "- **Positions are harmonic.** No anharmonic shift is applied.",
        (
            "- **Neutral, closed-shell, aromatic molecules only**, of the elements "
            f"{', '.join(training.elements)}. aromir refuses anything else before computing."
        ),
        (
            f"- **Sizes outside {training.min_atoms} to {training.max_atoms} atoms are "
            "extrapolation.** `aromir spectrum` refuses them unless "
            "`--outside-training-sizes` is given; its certificate then licenses none of "
            "their bands."
        ),
        (
            f"- **Not licensed: {', '.join(unlicensed) or 'no family'}.** Bands of an unlicensed "
            "family are printed with their measured error and marked."
        ),
        f"- {_order_statement(release)}",
        *_sameness(release),
        (
            "- **Band heights have no error bar.** They are computed on the corrected modes "
            "with the cheap level's dipole derivatives, which are not corrected."
        ),
        (
            "- **A family is a rule, not a kind of bond.** "
            + " ".join(f"{family}: {rule}." for family, rule in families.RULES.items())
            + " aromir's certificate licenses a band only when its family is within the "
            "licence line, the molecule lies inside the training sizes, its low-level mode "
            "is real, the band is real for "
            "the seeds' mean and for every seed, and a CH-stretch or CH-oop band moves "
            "carbon-bound hydrogens. A "
            "ring-ip band is licensed as the rule takes it, C=O stretches included."
        ),
    ]


def _architecture(release: Release) -> list[str]:
    recipe = release.recipes[0]
    architecture, inputs = recipe.architecture, recipe.inputs
    n_parameters = sum(parameter.numel() for parameter in release.networks[0].parameters())
    extras = [
        name
        for name, on in (
            ("the pair features", architecture.n_pair_features > 0),
            ("a learned share of the low level's own force constant", architecture.sqm_scale),
            ("the parity-odd factor", architecture.parity_odd),
        )
        if on
    ]
    return [
        "## What it is",
        "",
        (
            "An equivariant message-passing encoder (PaiNN type) over the atoms, and a head that "
            "predicts one number per pair of internal coordinates: an entry of ΔF. The Cartesian "
            "correction is ΔH = Bᵀ ΔF B with B the Wilson matrix, so it is symmetric and free "
            "of translations. It is free of rotations and turns with the molecule only where "
            "no linear angle is among the internal coordinates: see the limits above."
        ),
        "",
        (
            f"- Encoder: {architecture.n_blocks} blocks, {architecture.n_features} scalar and "
            f"{architecture.n_features} vector channels, {architecture.aggregation} pooling."
        ),
        f"- Head: hidden width {architecture.hidden}, with {', '.join(extras) or 'no extras'}.",
        (
            f"- Pair pattern {inputs.pattern}; primitives by the {inputs.primitives.angle_choice} "
            f"angle choice, dihedrals {inputs.primitives.dihedral_encoding}."
        ),
        f"- {n_parameters:,} parameters per seed.",
    ]


def _files(release: Release) -> list[str]:
    lines = ["## Files", "", "| file | seed | sha256 |", "|---|---|---|"]
    lines += [
        f"| seed-{recipe.seed}.safetensors | {recipe.seed} | `{sha256}` |"
        for recipe, sha256 in zip(release.recipes, release.tensor_sha256, strict=True)
    ]
    records = [EVALUATION_FILE, TRAINING_FILE]
    records += [INVARIANCE_FILE] if release.invariance is not None else []
    records += [RELEASE_FILE, MEASUREMENT_FILE] if release.shipped is not None else []
    named = [f"`{name}`" for name in RECORDED_FILES]
    # invariance.json is not among them: a reader is told which records the hashes guard.
    bound = (
        f" `{RELEASE_FILE}` binds {', '.join(named[:-1])} and {named[-1]} by the SHA-256 of "
        "their content."
        if release.shipped is not None
        else ""
    )
    return [
        *lines,
        "",
        (
            "Each `seed-<n>.json` beside its tensors holds the recipe and that sha256; "
            f"{', '.join(f'`{name}`' for name in records)} hold the records this card is "
            f"written from, and `{MEASURED_FILE}`, where the folder has it, is the output of "
            "`aromir evaluate` the error tables are copied from: the loader refuses tables "
            f"that differ from it.{bound}"
        ),
    ]


def _use(release: Release) -> list[str]:
    return [
        "## Use",
        "",
        "```console",
        "$ pip install torch --index-url https://download.pytorch.org/whl/cpu",
        f"$ pip install git+{REPOSITORY}",
        '$ aromir spectrum "c1ccc2ccccc2c1" --out results/naphthalene',
        "```",
        "",
        (
            "aromir uses the release inside the package; `--release <folder>` names another, "
            "such as a copy of this one."
        ),
        "",
        "## Licence",
        "",
        f"Weights and records: CC BY 4.0. Code: MIT, at {REPOSITORY}.",
        "",
        (
            f"This card was written by `aromir card` from release v{release.network_version}'s "
            "own files."
        ),
    ]


def to_markdown(release: Release) -> str:
    """Render a release's model card as a Markdown document."""
    sections = (
        _front_matter(release),
        _summary(release),
        _evaluation(release),
        _training(release),
        _limits(release),
        _architecture(release),
        _files(release),
        _use(release),
    )
    return "\n\n".join("\n".join(section) for section in sections) + "\n"
