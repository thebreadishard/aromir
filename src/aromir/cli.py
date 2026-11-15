"""The command line: ``aromir check`` and ``aromir baseline``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Final

from pydantic import ValidationError

import aromir
from aromir import pipeline
from aromir.dft import Resources
from aromir.errors import AromirError, RefusedError

EXIT_DONE: Final = 0
EXIT_FAILED: Final = 1
EXIT_REFUSED: Final = 2


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
    baseline.add_argument(
        "--out", type=Path, required=True, help="the result folder (empty or new)"
    )
    baseline.add_argument("--threads", type=int, default=defaults.threads)
    baseline.add_argument("--max-memory-mb", type=int, default=defaults.max_memory_mb)
    return parser


def _run(arguments: argparse.Namespace) -> None:
    molecule = pipeline.MoleculeInput(smiles=arguments.smiles, xyz=arguments.xyz)
    if arguments.command == "check":
        admitted, _start = pipeline.admit(molecule)
        print(f"admitted: {admitted.smiles} ({admitted.formula}, {admitted.n_atoms} atoms)")  # noqa: T201  the command's answer
        return
    resources = Resources(threads=arguments.threads, max_memory_mb=arguments.max_memory_mb)
    request = pipeline.Request(molecule=molecule, out=arguments.out, resources=resources)
    record = pipeline.run_baseline(request)
    n_bands = len(record["harmonic"]["bands"])
    print(f"done: {n_bands} bands in {request.out / pipeline.BASELINE_FILE}")  # noqa: T201  the command's answer


def main(argv: list[str] | None = None) -> int:
    """Run the command line; return the process's exit code."""
    arguments = _parser().parse_args(argv)
    try:
        _run(arguments)
    except ValidationError as invalid:
        reasons = [str(error["msg"]) for error in invalid.errors()]
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
