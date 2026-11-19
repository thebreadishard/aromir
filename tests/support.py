"""Helpers the tests share: synthetic Hessians, a seeded rotation, a random network."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from rdkit import Chem

from aromir import admission, geometry, own_spread, release
from aromir.constants import ANGSTROM_PER_BOHR
from aromir.families import FAMILIES
from aromir.molecule import Geometry
from aromir.network import weights
from aromir.network.inputs import InputSettings
from aromir.network.model import Architecture, DeltaHessianNetwork

if TYPE_CHECKING:
    from aromir.molecule import FloatArray

Bonds = tuple[tuple[int, int, float], ...]

WATER_BONDS: Bonds = ((0, 1, 0.55), (0, 2, 0.55), (1, 2, 0.10))
"""Atom pairs and force constants (hartree/bohr²) of a valence force field for water."""

BENZENE_XYZ = """12
benzene, Å
C   1.3970   0.0000  0.0
C   0.6985   1.2098  0.0
C  -0.6985   1.2098  0.0
C  -1.3970   0.0000  0.0
C  -0.6985  -1.2098  0.0
C   0.6985  -1.2098  0.0
H   2.4810   0.0000  0.0
H   1.2405   2.1486  0.0
H  -1.2405   2.1486  0.0
H  -2.4810   0.0000  0.0
H  -1.2405  -2.1486  0.0
H   1.2405  -2.1486  0.0
"""
"""Benzene as an xyz file, in Å."""


ZWITTERION = "[NH3+]c1ccccc1C(=O)[O-]"
"""Anthranilic acid's zwitterion: in the gas phase B3LYP moves a proton back to the acid."""


def proton_moved(smiles: str) -> Geometry:
    """Return a molecule's start geometry with an N–H proton moved to 1.0 Å from an O⁻.

    The proton transfer B3LYP makes of a gas-phase zwitterion, given as a geometry: no
    quantum chemistry is needed to test what follows from it.
    """
    start = geometry.start_from_smiles(smiles)
    molecule = Chem.AddHs(Chem.MolFromSmiles(smiles))
    proton = next(
        atom.GetIdx()
        for atom in molecule.GetAtoms()
        if atom.GetSymbol() == "H" and atom.GetNeighbors()[0].GetSymbol() == "N"
    )
    oxygen = next(atom.GetIdx() for atom in molecule.GetAtoms() if atom.GetFormalCharge() < 0)
    coords = start.coords_bohr.copy()
    away = coords[proton] - coords[oxygen]
    coords[proton] = coords[oxygen] + away / np.linalg.norm(away) / ANGSTROM_PER_BOHR
    return Geometry(start.symbols, coords)


def bond_force_field(geometry: Geometry, bonds: Bonds) -> FloatArray:
    """Return an exact Cartesian Hessian of harmonic springs along the given atom pairs.

    Such a Hessian is symmetric and invariant to translations and rotations at the
    geometry it is built for, which makes it a known answer for the vibrational code.
    """
    hessian = np.zeros((3 * geometry.n_atoms, 3 * geometry.n_atoms))
    for first, second, constant in bonds:
        direction = geometry.coords_bohr[second] - geometry.coords_bohr[first]
        direction = direction / np.linalg.norm(direction)
        block = constant * np.outer(direction, direction)
        for row, column, sign in (
            (first, first, 1),
            (second, second, 1),
            (first, second, -1),
            (second, first, -1),
        ):
            hessian[3 * row : 3 * row + 3, 3 * column : 3 * column + 3] += sign * block
    return hessian


def rotation_matrix(seed: int) -> FloatArray:
    """Return a proper rotation drawn from a seeded generator."""
    matrix, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
    if np.linalg.det(matrix) < 0:
        matrix[:, 0] = -matrix[:, 0]
    return np.asarray(matrix, dtype=float)


def random_hessian(n_atoms: int, seed: int) -> FloatArray:
    """Return a seeded symmetric matrix with a stiff diagonal, the size of a Hessian.

    The network's arithmetic does not care that it is not a real Hessian, and a test of
    that arithmetic then needs no quantum chemistry.
    """
    noise = np.random.default_rng(seed).normal(scale=0.02, size=(3 * n_atoms, 3 * n_atoms))
    return np.asarray(noise + noise.T + 0.5 * np.eye(3 * n_atoms), dtype=float)


def random_network(architecture: Architecture, seed: int) -> DeltaHessianNetwork:
    """Return a network with every parameter and buffer drawn, scales and spreads positive.

    Nothing is left at its zero initialisation, so every input reaches the output.
    """
    torch.manual_seed(seed)
    network = DeltaHessianNetwork(architecture)
    positive = ("class_scale", "force_constant_scale", "pair_feature_std")
    with torch.no_grad():
        for name, value in network.state_dict().items():
            if name.endswith("centres"):
                continue
            drawn = torch.randn_like(value) * 0.1
            value.copy_(drawn.abs() + 0.5 if name in positive else drawn)
    return network.eval()


RELEASE_DATA = Path(__file__).parent / "data" / "release"


def stored_molecule(name: str) -> tuple[Geometry, FloatArray]:
    """Return a stored molecule of the test data and its B3LYP Hessian (see its README)."""
    return _stored(RELEASE_DATA / f"stored_{name}.npz")


def _stored(path: Path) -> tuple[Geometry, FloatArray]:
    with np.load(path) as arrays:
        symbols = tuple(str(symbol) for symbol in arrays["symbols"])
        return Geometry(symbols, np.array(arrays["coords_bohr"])), np.array(arrays["hessian"])


def as_computed(name: str) -> tuple[str, Geometry, FloatArray]:
    """Return a stored molecule as the pipeline would have it: SMILES, geometry, Hessian.

    The geometry is read the way an xyz request is, so its atoms are in aromir's canonical
    order and the SMILES is the one admission writes. "benzene" is hold-out (a)'s benzene
    with its B3LYP Hessian.
    """
    path = RELEASE_DATA / ("holdout_a_benzene.npz" if name == "benzene" else f"stored_{name}.npz")
    molecule, hessian = _stored(path)
    with TemporaryDirectory() as folder:
        xyz = Path(folder) / "molecule.xyz"
        lines = [str(molecule.n_atoms), name]
        for symbol, (x, y, z) in zip(
            molecule.symbols, molecule.coords_bohr * ANGSTROM_PER_BOHR, strict=True
        ):
            lines.append(f"{symbol} {x:.10f} {y:.10f} {z:.10f}")
        xyz.write_text("\n".join(lines) + "\n", encoding="utf-8")
        reading = admission.read(xyz)
    order = tuple(geometry.canonical_order(reading.molecule))
    ordered, ordered_hessian = own_spread.reordered(molecule, hessian, order)
    return admission.admit(reading).smiles, ordered, ordered_hessian


def unmoved(n_bands: int) -> own_spread.OwnSpread:
    """Return a molecule's own spread as it would be for a network that moves no band."""
    return own_spread.OwnSpread(
        spread_cm=(0.0,) * n_bands,
        change=("rotation 1",) * n_bands,
        tried=("rotation 1",),
        refusal="",
    )


SMALL_ARCHITECTURE = Architecture(n_features=16, n_blocks=2, hidden=24)
TRAINING: dict[str, Any] = {
    "network_version": "1.3",
    "elements": ["C", "H", "N", "O"],
    "min_atoms": 10,
    "max_atoms": 30,
    "trained_smiles": ["c1ccc2ccccc2c1"],
    "held_out_smiles": {"a": ["c1ccccc1"], "b": ["Cc1ccccc1"]},
    "soft_mode_repair": False,
}
"""A made-up training set: benzene held out, naphthalene trained on, no chlorine."""


UNLICENSED_OTHER = {
    "rms_cm": 3.89,
    "interval_cm": [3.5, 4.2],
    "share_of_resamples_within_line": 0.0,
}
"""The made-up release keeps one family above the licence line, so that a test sees both kinds
of band; v1.3 itself has all four within it."""


def measured_of(evaluation: dict[str, Any]) -> dict[str, Any]:
    """Return what ``aromir evaluate`` writes for the tables of an evaluation record."""
    tables = {
        "band_error_seed_mean": "band_error_seed_mean_cm",
        "first_order_error_per_seed": "first_order_error_per_seed_cm",
        "first_order_error_per_seed_mean": "first_order_error_per_seed_mean_cm",
    }
    return {
        name: {"n_molecules": holdout["n_molecules"]}
        | {
            key: {family: evaluation[field][family][name] for family in FAMILIES}
            for key, field in tables.items()
        }
        for name, holdout in evaluation["holdouts"].items()
    }


def write_release(folder: Path, seeds: tuple[int, ...] = (0, 1), version: str = "1.3") -> Path:
    """Write a release folder: v1.3's evaluation record, ``TRAINING`` and random small seeds.

    The evaluation's hold-out counts are set to ``TRAINING``'s one molecule each: a release
    whose two records count differently does not load, and neither does one whose
    evaluation names another number of seeds than it holds. The family 'other' is put
    above the licence line and the atom-order spread is left unmeasured.
    """
    folder.mkdir(parents=True, exist_ok=True)
    evaluation = json.loads((RELEASE_DATA / "evaluation_v1.3.json").read_text(encoding="utf-8"))
    for name, held_out in TRAINING["held_out_smiles"].items():
        evaluation["holdouts"][name]["n_molecules"] = len(held_out)
        for per_holdout in evaluation["band_error_seed_mean_cm"].values():
            per_holdout[name]["n_molecules"] = min(per_holdout[name]["n_molecules"], len(held_out))
    evaluation["band_error_seed_mean_cm"]["other"]["b"] |= UNLICENSED_OTHER
    means = evaluation["first_order_error_per_seed_mean_cm"]
    for family, per_holdout in evaluation["first_order_error_per_seed_cm"].items():
        for name, errors in per_holdout.items():
            per_holdout[name] = errors[: len(seeds)]
            means[family][name] = round(sum(per_holdout[name]) / len(per_holdout[name]), 2)
    evaluation.pop("atom_order_spread")
    (folder / "evaluation.json").write_text(json.dumps(evaluation), encoding="utf-8")
    (folder / "training.json").write_text(json.dumps(TRAINING), encoding="utf-8")
    for seed in seeds:
        recipe = weights.Recipe(
            network_version=version,
            seed=seed,
            architecture=SMALL_ARCHITECTURE,
            inputs=InputSettings(),
        )
        state = random_network(SMALL_ARCHITECTURE, seed).state_dict()
        weights.save(state, folder, f"seed-{seed}", recipe)
    return folder


SMALL_RINGS = (
    "c1ccc(cc1)C12CC1C2",
    "c1ccc(cc1)C12C(C)(C)C1C2",
    "c1ccc(cc1)C12C3C4C1C5C2C3C45",
    "c1ccc(cc1)C12CC(C1)C2",
)
"""Phenylbicyclo[1.1.0]butane and its 2,2-dimethyl analogue, whose central bond joins two
atoms that share two neighbours; phenylcubane; and phenylbicyclo[1.1.1]pentane, whose
bridgeheads share three neighbours and lie 1.95 Å apart, within bonding distance."""


def training_biphenylenes() -> list[str]:
    """Return the bundled release's training molecules with a ring of four aromatic atoms."""
    four = Chem.MolFromSmarts("a1aaa1")
    trained = release.load(release.bundled()).training.trained_smiles
    return [s for s in trained if Chem.MolFromSmiles(s).HasSubstructMatch(four)]


def distorted(start: Geometry) -> list[Geometry]:
    """Return a start geometry as it is, scaled by 0.995, and with 0.02 Å noise (5 seeds).

    The diagonals of a biphenylene's four-membered ring lie at 1.969 to 1.973 Å in its
    start geometry, at RDKit's bonding limit of 1.97 Å for two carbons: these variants put
    them on both sides of it.
    """
    noisy = [
        start.coords_bohr
        + np.random.default_rng(seed).normal(
            scale=0.02 / ANGSTROM_PER_BOHR, size=(start.n_atoms, 3)
        )
        for seed in range(5)
    ]
    return [
        Geometry(start.symbols, coords)
        for coords in [start.coords_bohr, start.coords_bohr * 0.995, *noisy]
    ]
