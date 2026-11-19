"""Measure a release's invariance record: how far its answer moves for the same molecule.

    python scripts/measure_invariance.py <corpus molecules folder> <manifest.csv> <invariance.json>
        [--release <release folder>]

The design is the one release v1.3's ``invariance.json`` was measured with, fixed before any
other release was measured: the seven molecules below, on their stored B3LYP/6-31G*
geometries and analytic Hessians (rigid motions projected out) from the training corpus,
public in thebreadishard/capstone-data (``modules/05_support_predictor/corpus/molecules/
<id>/hessian_b3lyp_analytic.npz`` beside ``geometry.json``, and ``corpus/manifest.csv``),
and per molecule, from one random generator seeded 7 and drawn in this order:

- 12 xyz files of the geometry with the atoms listed in a random order, each read the way
  aromir reads a request (``admission.read``) and put into its canonical order; the spread
  is the largest band difference of any distinct order from the first one met;
- 8 random rigid rotations of geometry and Hessian; the spread is the largest band
  difference from the molecule as stored;
- the mirror image (z → −z), against the molecule as stored.

A record keeps, per change, the largest spread over the molecules with a linear group (a
nitrile or an alkyne) and over those without. The script runs no quantum chemistry: it
applies the release's seed-mean correction to stored Hessians, seconds per molecule.

The record names each seed's tensor hash (``tensor_sha256``), which the release's loader
checks against the seeds beside it, and the commit the script ran at, read from git; a
tree with a changed tracked file, or an untracked file under src/ or scripts/, is refused,
for then no commit holds the code that measured.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from aromir import admission, records, release, vibrations
from aromir import geometry as geo
from aromir.constants import ANGSTROM_PER_BOHR
from aromir.molecule import Geometry

if TYPE_CHECKING:
    from aromir.molecule import FloatArray

MOLECULES = (
    ("c1ccccc1", "benzene"),
    ("N#Cc1ccccc1", "benzonitrile"),
    ("c1ccc(-c2ccccc2)cc1", "biphenyl"),
    ("Cc1ccc2ccccc2c1", "2-methylnaphthalene"),
    ("C#Cc1ccccc1", "phenylacetylene"),
    ("C(#Cc1ccccc1)c1ccccc1", "diphenylacetylene"),
    ("Oc1ccccc1", "phenol"),
)
"""v1.3's measured set, in the order its random draws were made: the molecules of its
audit's list that had a stored analytic Hessian."""
SEED = 7
N_LISTINGS = 12
N_ROTATIONS = 8
LINEAR_BOND = "#"
"""A triple bond in the SMILES: the nitriles and alkynes whose angle the network reads
against a fixed axis."""
HESSIAN_FILE = "hessian_b3lyp_analytic.npz"
MIRROR = np.diag([1.0, 1.0, -1.0])
REPOSITORY = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Spreads:
    """One molecule's largest band difference, cm⁻¹, under each change."""

    rotation: float
    mirror_image: float
    xyz_atom_order: float


def stored_molecule(folder: Path) -> tuple[Geometry, FloatArray]:
    """Return a corpus molecule's geometry and its projected analytic B3LYP Hessian."""
    stored = json.loads((folder / "geometry.json").read_text(encoding="utf-8"))
    symbols = tuple(symbol[0].upper() + symbol[1:].lower() for symbol in stored["symbols"])
    with np.load(folder / HESSIAN_FILE) as arrays:
        hessian = np.array(arrays["H_projected"], dtype=float)
    return Geometry(symbols, np.array(stored["coords_bohr"], dtype=float)), hessian


def corpus_ids(corpus: Path, manifest: Path) -> dict[str, str]:
    """Return the first corpus entry with an analytic Hessian for each SMILES of the set."""
    wanted = {smiles for smiles, _ in MOLECULES}
    found: dict[str, str] = {}
    with manifest.open(encoding="utf-8", newline="") as rows:
        for row in csv.DictReader(rows):
            if row["smiles"] in wanted and (corpus / row["id"] / HESSIAN_FILE).is_file():
                found.setdefault(row["smiles"], row["id"])
    missing = sorted(smiles for smiles in wanted if smiles not in found)
    if missing:
        msg = f"no stored analytic Hessian for {missing}: the registered set cannot be measured"
        raise SystemExit(msg)
    return found


def corrected(loaded: release.Release, molecule: Geometry, hessian: FloatArray) -> FloatArray:
    """Return the band positions, cm⁻¹, of a Hessian with the release's correction added."""
    answer = loaded.corrections(molecule, hessian).mean
    return np.asarray(vibrations.vibrational_modes(hessian + answer, molecule).wavenumbers)


def reordered(
    molecule: Geometry, hessian: FloatArray, order: tuple[int, ...]
) -> tuple[Geometry, FloatArray]:
    """Return the molecule with stored atom ``order[k]`` in position k."""
    index = (3 * np.array(order)[:, None] + np.arange(3)).ravel()
    return molecule.reordered(list(order)), hessian[np.ix_(index, index)]


def moved(
    molecule: Geometry, hessian: FloatArray, matrix: FloatArray
) -> tuple[Geometry, FloatArray]:
    """Return the molecule and its Hessian under an orthogonal map of space."""
    blocks = np.kron(np.eye(molecule.n_atoms), matrix)
    return Geometry(molecule.symbols, molecule.coords_bohr @ matrix.T), blocks @ hessian @ blocks.T


def random_rotation(rng: np.random.Generator) -> FloatArray:
    """Return a random proper rotation (determinant +1)."""
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    return np.asarray(q * np.sign(np.linalg.det(q)))


def listed_orders(
    molecule: Geometry, rng: np.random.Generator, scratch: Path
) -> list[tuple[int, ...]]:
    """Return the distinct atom orders aromir gives random xyz listings, first met first."""
    orders: dict[tuple[int, ...], None] = {}
    path = scratch / "listing.xyz"
    for _ in range(N_LISTINGS):
        listing = rng.permutation(molecule.n_atoms)
        lines = [str(molecule.n_atoms), "one geometry, listed in another order"]
        for atom in listing:
            x, y, z = molecule.coords_bohr[atom] * ANGSTROM_PER_BOHR
            lines.append(f"{molecule.symbols[atom]} {x:.10f} {y:.10f} {z:.10f}")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        canonical = geo.canonical_order(admission.read(path).molecule)
        orders.setdefault(tuple(int(listing[k]) for k in canonical), None)
    return list(orders)


def largest_difference(spectra: list[FloatArray], reference: FloatArray) -> float:
    """Return the largest band difference, cm⁻¹, of any spectrum from the reference."""
    return max(float(np.abs(spectrum - reference).max()) for spectrum in spectra)


def spreads(
    loaded: release.Release,
    stored: tuple[Geometry, FloatArray],
    rng: np.random.Generator,
    scratch: Path,
) -> Spreads:
    """Return one molecule's spreads, drawing listings first, then rotations."""
    molecule, hessian = stored
    as_stored = corrected(loaded, molecule, hessian)
    orders = listed_orders(molecule, rng, scratch)
    listed = [corrected(loaded, *reordered(molecule, hessian, order)) for order in orders]
    turned = [
        corrected(loaded, *moved(molecule, hessian, random_rotation(rng)))
        for _ in range(N_ROTATIONS)
    ]
    mirrored = corrected(loaded, *moved(molecule, hessian, MIRROR))
    return Spreads(
        rotation=largest_difference(turned, as_stored),
        mirror_image=largest_difference([mirrored], as_stored),
        xyz_atom_order=largest_difference(listed, listed[0]),
    )


def spread_record(
    per_molecule: dict[str, float], linear: set[str], n_trials: int
) -> dict[str, object]:
    """Return one change's record: the largest spread with and without a linear group."""
    record: dict[str, object] = {}
    for key, members in (
        ("with_linear_group", [m for m in per_molecule if m in linear]),
        ("without_linear_group", [m for m in per_molecule if m not in linear]),
    ):
        record[f"{key}_cm"] = round(max(per_molecule[m] for m in members), 2) if members else None
        record[f"n_{key}"] = len(members)
    return record | {"n_trials": n_trials}


def source_text(loaded: release.Release, folder: Path, commit: str) -> str:
    """Say how, when, on which networks and on which molecules the record was measured."""
    kind = "release networks" if loaded.shipped is not None else "networks"
    with_group = [name for smiles, name in MOLECULES if LINEAR_BOND in smiles]
    without = [name for smiles, name in MOLECULES if LINEAR_BOND not in smiles]
    return (
        f"scripts/measure_invariance.py of aromir at {commit}, "
        f"{datetime.now(UTC).date().isoformat()}, on the {kind} {folder.name} ships (their "
        f"tensors are named in tensor_sha256): {', '.join(with_group[:-1])} and "
        f"{with_group[-1]} with a linear group; {', '.join(without[:-1])} and {without[-1]} "
        "without; stored B3LYP geometries and Hessians of thebreadishard/capstone-data "
        "(modules/05_support_predictor/corpus/molecules), the trials of v1.3's record"
    )


def finished_record(
    measured: dict[str, object], loaded: release.Release, source: str
) -> dict[str, object]:
    """Return a record's spreads with the weights they were measured on and its source, checked.

    The record names each seed's tensor hash as the release's loader reads it, so a
    release folder refuses the record beside other weights.
    """
    record = measured | {"tensor_sha256": loaded.seed_sha256, "source": source}
    release.Invariance.model_validate(record)
    return record


def _git(*arguments: str) -> str:
    """Return a git command's output in this repository.

    autocrlf is set so that a checkout written with Windows line ends, read by a Linux git
    (WSL on the same folder), is not taken for a changed tree.
    """
    git = shutil.which("git")
    if git is None:
        msg = "git is not on the PATH: the commit the record names cannot be read"
        raise SystemExit(msg)
    done = subprocess.run(  # noqa: S603  a fixed git command, no user input
        [git, "-c", "core.autocrlf=true", *arguments],
        cwd=REPOSITORY,
        capture_output=True,
        text=True,
        check=True,
    )
    return done.stdout


def clean_commit() -> str:
    """Return the commit the script runs at; refuse a tree that differs from it."""
    changed = [
        line[3:]
        for line in _git("status", "--porcelain").splitlines()
        if not line.startswith("??") or line[3:].startswith(("src/", "scripts/"))
    ]
    if changed:
        msg = f"the tree differs from its commit in {changed}: commit first, then measure"
        raise SystemExit(msg)
    return _git("rev-parse", "HEAD").strip()


def measure(arguments: argparse.Namespace) -> dict[str, object]:
    """Measure every molecule of the set and return the release's invariance record."""
    commit = clean_commit()
    loaded = release.load(arguments.release)
    ids = corpus_ids(arguments.corpus, arguments.manifest)
    rng = np.random.default_rng(SEED)
    measured: dict[str, Spreads] = {}
    with tempfile.TemporaryDirectory() as scratch:
        for smiles, name in MOLECULES:
            stored = stored_molecule(arguments.corpus / ids[smiles])
            measured[name] = spreads(loaded, stored, rng, Path(scratch))
            print(f"{name:20s} {ids[smiles]}: {measured[name]}", flush=True)
    linear = {name for smiles, name in MOLECULES if LINEAR_BOND in smiles}
    record: dict[str, object] = {"network_version": loaded.network_version}
    for change, n_trials in (
        ("rotation", N_ROTATIONS),
        ("mirror_image", 1),
        ("xyz_atom_order", N_LISTINGS),
    ):
        per_molecule = {name: getattr(found, change) for name, found in measured.items()}
        record[change] = spread_record(per_molecule, linear, n_trials)
    return finished_record(record, loaded, source_text(loaded, arguments.release, commit))


def main() -> None:
    """Measure the record and write it where the command line says."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("corpus", type=Path, help="the corpus's molecules folder")
    parser.add_argument("manifest", type=Path, help="the corpus's manifest.csv")
    parser.add_argument("out", type=Path, help="where invariance.json is written")
    parser.add_argument("--release", type=Path, default=release.bundled())
    arguments = parser.parse_args()
    torch.set_num_threads(2)
    records.write_json(arguments.out, measure(arguments))
    print(f"wrote {arguments.out}", flush=True)


if __name__ == "__main__":
    main()
