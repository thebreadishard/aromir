"""The command line: ``aromir check``, ``aromir baseline`` and ``aromir spectrum``."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Final

from pydantic import ValidationError

import aromir
from aromir import card, certificate, evaluation, pipeline, records, release
from aromir.dft import Resources
from aromir.errors import AromirError, RefusedError

EXIT_DONE: Final = 0
EXIT_FAILED: Final = 1
EXIT_REFUSED: Final = 2
TRACEBACK_VARIABLE: Final = "AROMIR_TRACEBACK"
"""Set in the environment, an unforeseen error keeps its traceback: for a bug report."""
EXIT_UNPHYSICAL: Final = 3
"""The result is written, but a corrected band is imaginary: not a spectrum to use."""


RELEASE_HELP: Final = (
    "the folder of a network release (default: the release inside the package, "
    f"v{release.BUNDLED_VERSION})"
)


def _parser() -> argparse.ArgumentParser:
    defaults = Resources()
    parser = argparse.ArgumentParser(prog="aromir", description=aromir.__doc__)
    parser.add_argument("--version", action="version", version=f"aromir {aromir.__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    molecule = argparse.ArgumentParser(add_help=False)
    source = molecule.add_mutually_exclusive_group(required=True)
    source.add_argument("smiles", nargs="?", help="the molecule as a SMILES string")
    source.add_argument("--xyz", type=Path, help="the molecule as an xyz file in Å")

    commands.add_parser(
        "check", parents=[molecule], help="say whether a molecule is supported; compute nothing"
    )
    baseline = commands.add_parser(
        "baseline",
        parents=[molecule],
        help="optimise, compute the B3LYP Hessian and write the harmonic and scaled spectra",
    )
    spectrum = commands.add_parser(
        "spectrum",
        parents=[molecule],
        help="compute the corrected spectrum with a network release and write its certificate",
    )
    spectrum.add_argument("--release", type=Path, help=RELEASE_HELP)
    spectrum.add_argument(
        "--outside-training-sizes",
        action="store_true",
        help="compute a molecule smaller or larger than the network's training molecules, "
        "although none of its bands can be licensed",
    )
    for computing in (baseline, spectrum):
        computing.add_argument(
            "--out",
            type=Path,
            required=True,
            help="the result folder: new, empty, or holding the unfinished run of this request",
        )
        computing.add_argument(
            "--threads",
            type=int,
            default=defaults.threads,
            help="threads for the quantum chemistry and the network",
        )
        computing.add_argument(
            "--max-memory-mb",
            type=int,
            default=defaults.max_memory_mb,
            help="what pyscf plans its arrays for: a target of the quantum chemistry, not a "
            "cap the operating system enforces",
        )
    evaluate = commands.add_parser(
        "evaluate",
        help="measure a release on hold-out molecules with stored Hessians: the error of what "
        "it delivers, per family, and the first-order error of each seed",
    )
    evaluate.add_argument(
        "holdouts",
        type=Path,
        help="a folder of hold-out folders of .npz molecules (see aromir.evaluation)",
    )
    evaluate.add_argument("--release", type=Path, help=RELEASE_HELP)
    evaluate.add_argument("--out", type=Path, help="the JSON file to write (default: print)")
    model_card = commands.add_parser(
        "card", help="write the model card of a network release from the release's own files"
    )
    model_card.add_argument("--release", type=Path, help=RELEASE_HELP)
    model_card.add_argument("--out", type=Path, help="the file to write (default: print)")
    return parser


def _release(arguments: argparse.Namespace) -> release.Release:
    """Load the release the command names, or the one that ships inside the package."""
    return release.load(arguments.release or release.bundled())


def _write_card(arguments: argparse.Namespace) -> None:
    out = arguments.out
    reasons = []
    if out is not None and out.exists():
        reasons.append(f"{out} exists: a result is never overwritten")
    if out is not None and not out.parent.is_dir():
        reasons.append(f"{out.parent}: no such folder")
    if reasons:
        raise RefusedError(reasons)
    text = card.to_markdown(_release(arguments))
    if out is None:
        print(text, end="")  # noqa: T201  the command's answer
        return
    records.write_text(out, text)
    print(f"done: model card in {out}")  # noqa: T201  the command's answer


def _spectrum(arguments: argparse.Namespace, molecule: pipeline.MoleculeInput) -> int:
    request = pipeline.Request(
        molecule=molecule,
        out=arguments.out,
        resources=Resources(threads=arguments.threads, max_memory_mb=arguments.max_memory_mb),
        outside_training_sizes=arguments.outside_training_sizes,
    )
    record = pipeline.run_spectrum(request, _release(arguments))
    n_licensed = sum(band["licensed"] for band in record["bands"])
    summary = (
        f"{len(record['bands'])} bands, {n_licensed} of them licensed; "
        f"certificate in {request.out / pipeline.CERTIFICATE_TEXT_FILE}"
    )
    n_imaginary = sum(band["licence_note"] == certificate.IMAGINARY for band in record["bands"])
    if n_imaginary:
        # The result is written, but it is no spectrum to use as it stands.
        print(  # noqa: T201  the command's answer
            f"unphysical: {n_imaginary} corrected band(s) have an imaginary position; {summary}",
            file=sys.stderr,
        )
        return EXIT_UNPHYSICAL
    print(f"done: {summary}")  # noqa: T201  the command's answer
    return EXIT_DONE


def _evaluate(arguments: argparse.Namespace) -> None:
    out = arguments.out
    if out is not None and out.exists():
        raise RefusedError([f"{out} exists: a result is never overwritten"])
    if out is not None and not out.parent.is_dir():
        raise RefusedError([f"{out.parent}: no such folder"])
    measured = evaluation.evaluate(_release(arguments), arguments.holdouts)
    if out is None:
        print(json.dumps(measured, indent=1, ensure_ascii=False))  # noqa: T201  the command's answer
        return
    records.write_json(out, measured)
    print(f"done: {len(measured)} hold-out(s) measured, in {out}")  # noqa: T201  the command's answer


def _run(arguments: argparse.Namespace) -> int:
    if arguments.command == "card":
        _write_card(arguments)
        return EXIT_DONE
    if arguments.command == "evaluate":
        _evaluate(arguments)
        return EXIT_DONE
    molecule = pipeline.MoleculeInput(smiles=arguments.smiles, xyz=arguments.xyz)
    if arguments.command == "check":
        admitted, _start = pipeline.admit(molecule)
        print(f"admitted: {admitted.smiles} ({admitted.formula}, {admitted.n_atoms} atoms)")  # noqa: T201  the command's answer
        return EXIT_DONE
    if arguments.command == "spectrum":
        return _spectrum(arguments, molecule)
    resources = Resources(threads=arguments.threads, max_memory_mb=arguments.max_memory_mb)
    request = pipeline.Request(molecule=molecule, out=arguments.out, resources=resources)
    record = pipeline.run_baseline(request)
    n_bands = len(record["harmonic"]["bands"])
    print(f"done: {n_bands} bands in {request.out / pipeline.BASELINE_FILE}")  # noqa: T201  the command's answer
    return EXIT_DONE


def main(argv: list[str] | None = None) -> int:
    """Run the command line; return the process's exit code.

    No failure a user can cause leaves as a traceback: a refusal names its reasons (exit
    2), anything else is one line with the error's kind (exit 1), unless the environment
    variable AROMIR_TRACEBACK asks for the traceback.
    """
    arguments = _parser().parse_args(argv)
    try:
        return _run(arguments)
    except ValidationError as invalid:
        reasons = [
            ": ".join([*(str(part) for part in error["loc"]), str(error["msg"])])
            for error in invalid.errors()
        ]
        print("refused:", *reasons, sep="\n  ", file=sys.stderr)  # noqa: T201  the command's answer
        return EXIT_REFUSED
    except RefusedError as refusal:
        print("refused:", *refusal.reasons, sep="\n  ", file=sys.stderr)  # noqa: T201  the command's answer
        return EXIT_REFUSED
    except AromirError as failure:
        print(f"failed: {failure}", file=sys.stderr)  # noqa: T201  the command's answer
        return EXIT_FAILED
    except (OSError, ValueError, ArithmeticError) as failure:
        if os.environ.get(TRACEBACK_VARIABLE, "0") not in ("", "0"):
            raise
        # A file that cannot be read or written, or a numerical step that broke: the
        # kind is named because the message alone may not say what happened.
        print(f"failed: {type(failure).__name__}: {failure}", file=sys.stderr)  # noqa: T201  the command's answer
        return EXIT_FAILED


if __name__ == "__main__":
    sys.exit(main())
