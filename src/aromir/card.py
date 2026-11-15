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

if TYPE_CHECKING:
    from aromir.release import Evaluation, Release

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
        f"- **Licensed families:** {', '.join(licensed) or 'none'}.",
        f"- **Not licensed:** {', '.join(unlicensed) or 'none'}.",
        (
            f"- **Use it through:** [aromir]({REPOSITORY}), which builds the inputs, applies the "
            "correction and writes a certificate for every band."
        ),
    ]


def _evaluation_table(evaluation: Evaluation) -> list[str]:
    holdouts = evaluation.holdouts
    header = " | ".join(f"({name}), {h.n_molecules} molecules" for name, h in holdouts.items())
    lines = [f"| family | {header} | licensed |", "|---|" + "---|" * (len(holdouts) + 1)]
    for family in families.FAMILIES:
        cells = " | ".join(
            f"{evaluation.errors_cm[family][name]} "
            f"({', '.join(str(x) for x in evaluation.seed_errors_cm[family][name])})"
            for name in holdouts
        )
        licensed = "yes" if evaluation.licensed(family) else "no"
        lines.append(f"| {family} | {cells} | {licensed} |")
    return lines


def _evaluation(release: Release) -> list[str]:
    evaluation = release.evaluation
    return [
        "## How well it is known",
        "",
        (
            "The rms error of the corrected band positions against "
            f"{evaluation.higher_level}, in cm⁻¹, on molecules the network never trained on. "
            "The seed mean, with each seed's value in brackets."
        ),
        "",
        *_evaluation_table(evaluation),
        "",
        *(f"- ({name}): {h.description}" for name, h in evaluation.holdouts.items()),
        "",
        (
            "A family is licensed when its seed-mean error is within "
            f"{evaluation.licence_line_cm:g} cm⁻¹ on every hold-out. aromir's certificate marks "
            "each band with its family, that family's error and whether it is licensed."
        ),
        "",
        f"Coupled cluster: {evaluation.coupled_cluster}.",
        "",
        f"Sources: {'; '.join(evaluation.sources)}.",
    ]


def _training(release: Release) -> list[str]:
    training = release.training
    held_out = ", ".join(
        f"{len(smiles)} in ({name})" for name, smiles in training.held_out_smiles.items()
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
        (
            f"- {len(training.trained_smiles)} molecules, neutral and closed-shell, each with an "
            "aromatic ring."
        ),
        f"- Elements: {', '.join(training.elements)}.",
        f"- Sizes: {training.min_atoms} to {training.max_atoms} atoms.",
        f"- Held out and never trained on: {held_out}.",
        (
            f"- The training Hessians were used {prepared}; aromir prepares a molecule's Hessian "
            "the same way before the network reads it."
        ),
    ]


def _order_statement(release: Release) -> str:
    primitives = release.recipes[0].inputs.primitives
    if primitives.angle_choice == "canonical" and primitives.dihedral_encoding != "raw":
        return (
            "The internal coordinates are built on the atoms in a canonical order and "
            "dihedrals enter by their cosine, so the answer does not depend on how the "
            "molecule's atoms are numbered."
        )
    return (
        "**The answer depends on how the molecule's atoms are numbered**: the internal "
        "coordinates are chosen by atom number, and a dihedral near ±180° enters by its raw "
        "value. aromir therefore puts every input, SMILES or xyz, into one canonical atom "
        "order first. Do not feed the network a molecule in another order."
    )


def _limits(release: Release) -> list[str]:
    evaluation, training = release.evaluation, release.training
    unlicensed = [f for f in families.FAMILIES if not evaluation.licensed(f)]
    return [
        "## Where it stops being known",
        "",
        (
            f"- **The level is {evaluation.higher_level}.** The errors above are against that "
            "level, not against experiment."
        ),
        "- **Positions are harmonic.** No anharmonic shift is applied.",
        (
            "- **Neutral, closed-shell, aromatic molecules only**, of the elements "
            f"{', '.join(training.elements)}. aromir refuses anything else before computing."
        ),
        (
            f"- **Sizes outside {training.min_atoms} to {training.max_atoms} atoms are "
            "extrapolation.** aromir computes them and marks the certificate."
        ),
        (
            f"- **Not licensed: {', '.join(unlicensed) or 'no family'}.** Bands of an unlicensed "
            "family are printed with their measured error and marked."
        ),
        f"- {_order_statement(release)}",
        "- **Band heights are the cheap level's.** The dipole derivatives are not corrected.",
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
            "correction is ΔH = Bᵀ ΔF B with B the Wilson matrix, so it is symmetric, free of "
            "translations and rotations, and turns with the molecule by construction."
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
    return [
        *lines,
        "",
        (
            "Each `seed-<n>.json` beside its tensors holds the recipe and that sha256; "
            "`evaluation.json` and `training.json` hold the records this card is written from."
        ),
    ]


def _use(release: Release) -> list[str]:
    return [
        "## Use",
        "",
        "```console",
        f"$ pip install git+{REPOSITORY}",
        '$ aromir spectrum "c1ccc2ccccc2c1" --out results/naphthalene --release <this folder>',
        "```",
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
