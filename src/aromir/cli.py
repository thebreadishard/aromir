"""The command line: ``aromir check``, ``aromir baseline`` and ``aromir spectrum``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Final

from pydantic import ValidationError

import aromir
from aromir import card, pipeline, records, release
from aromir.dft import Resources
from aromir.errors import AromirError, RefusedError

EXIT_DONE: Final = 0
EXIT_FAILED: Final = 1
EXIT_REFUSED: Final = 2


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
    for computing in (baseline, spectrum):
        computing.add_argument(
            "--out", type=Path, required=True, help="the result folder (empty or new)"
        )
        computing.add_argument("--threads", type=int, default=defaults.threads)
        computing.add_argument("--max-memory-mb", type=int, default=defaults.max_memory_mb)
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
    text = card.to_markdown(_release(arguments))
    if arguments.out is None:
        print(text, end="")  # noqa: T201  the command's answer
        return
    records.write_text(arguments.out, text)
    print(f"done: model card in {arguments.out}")  # noqa: T201  the command's answer


def _run(arguments: argparse.Namespace) -> None:
    if arguments.command == "card":
        _write_card(arguments)
        return
    molecule = pipeline.MoleculeInput(smiles=arguments.smiles, xyz=arguments.xyz)
    if arguments.command == "check":
        admitted, _start = pipeline.admit(molecule)
        print(f"admitted: {admitted.smiles} ({admitted.formula}, {admitted.n_atoms} atoms)")  # noqa: T201  the command's answer
        return
    resources = Resources(threads=arguments.threads, max_memory_mb=arguments.max_memory_mb)
    request = pipeline.Request(molecule=molecule, out=arguments.out, resources=resources)
    if arguments.command == "baseline":
        record = pipeline.run_baseline(request)
        n_bands = len(record["harmonic"]["bands"])
        print(f"done: {n_bands} bands in {request.out / pipeline.BASELINE_FILE}")  # noqa: T201  the command's answer
        return
    record = pipeline.run_spectrum(request, _release(arguments))
    n_licensed = sum(band["licensed"] for band in record["bands"])
    print(  # noqa: T201  the command's answer
        f"done: {len(record['bands'])} bands, {n_licensed} of them in a licensed family; "
        f"certificate in {request.out / pipeline.CERTIFICATE_TEXT_FILE}"
    )


def main(argv: list[str] | None = None) -> int:
    """Run the command line; return the process's exit code."""
    arguments = _parser().parse_args(argv)
    try:
        _run(arguments)
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
    return EXIT_DONE


if __name__ == "__main__":
    sys.exit(main())
