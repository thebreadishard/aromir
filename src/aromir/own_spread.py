"""How far one molecule's corrected bands move when the same molecule is given another way.

The network's answer is not yet invariant: a molecule turned in space, its mirror image or
its atoms listed in another order can get other corrected positions (a sign of the
network's internal coordinates that the molecule does not fix). A release's invariance
record measures that on a handful of molecules; this module measures it on the molecule
at hand, so that each band's licence can rest on its own number. No quantum chemistry is
run: the changes are applied to the molecule's optimised geometry and to the Hessian the
networks read, the networks correct each, and each corrected Hessian is brought back to
the molecule as computed.

The changes, drawn from one generator seeded ``SEED`` for every molecule, in this order:

- ``N_ROTATIONS`` random proper rotations;
- the mirror image (z → −z), and the mirror image of one more random rotation;
- ``N_LISTINGS`` random listings of the atoms, each put into aromir's canonical order
  (``geometry.canonical_order``), as an xyz file listed that way would be;
- up to ``N_SWAPS`` swaps of atoms the bond graph cannot tell apart (graph automorphisms:
  a methyl group's hydrogens, the two halves of a symmetric ring system): every order the
  canonical order can give a molecule is one of them. Where the graph has more, those
  tried are drawn from the first ``MAX_SYMMETRIES`` RDKit enumerates.

A listing that gives the molecule's own order, or one already tried, is not tried again.
Each change's corrected Hessian is the seeds' mean correction of the changed molecule,
mapped back (Kᵀ ΔH K for an orthogonal map K applied atom by atom; the 3×3 blocks put back
for a listing) and added to the low-level Hessian. Its modes are paired with the printed
modes by largest squared overlap (``spectrum.partners``), not by rank, and a band's spread
is the largest change of its position over every change, with the change that gave it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import numpy as np
from rdkit import Chem

from aromir import spectrum, vibrations
from aromir.constants import ANGSTROM_PER_BOHR
from aromir.errors import AromirError
from aromir.geometry import canonical_order
from aromir.molecule import Geometry

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from aromir.admission import Mol
    from aromir.lowlevel import LowLevel
    from aromir.molecule import FloatArray
    from aromir.release import Release

SEED: Final = 7
N_ROTATIONS: Final = 8
"""A linear angle is measured against a fixed axis of space, so how far a molecule with a
linear group moves depends on where that axis falls: three rotations gave benzonitrile
1.75 cm⁻¹ on v1.4, eight and twenty-four both 2.78."""
N_LISTINGS: Final = 4
N_SWAPS: Final = 12
MAX_SYMMETRIES: Final = 1000
"""How many graph automorphisms are enumerated at most; the swaps are drawn from them."""
MIRROR: Final = np.diag([1.0, 1.0, -1.0])
BOND_SLACK: Final = 1.3
"""A bond of the molecule's graph may be at most this many times the sum of its atoms'
covalent radii long in the geometry: a geometry in another atom order than the graph's
fails it."""
DECIMALS: Final = 2
"""A spread is stated, and held against the licence line, to two decimals."""


def random_rotation(rng: np.random.Generator) -> FloatArray:
    """Return a random proper rotation (determinant +1)."""
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    return np.asarray(q * np.sign(np.linalg.det(q)))


def moved(
    molecule: Geometry, hessian: FloatArray, matrix: FloatArray
) -> tuple[Geometry, FloatArray]:
    """Return the molecule and its Hessian under an orthogonal map of space."""
    blocks = np.kron(np.eye(molecule.n_atoms), matrix)
    return Geometry(molecule.symbols, molecule.coords_bohr @ matrix.T), blocks @ hessian @ blocks.T


def coordinate_index(order: tuple[int, ...]) -> NDArray[np.int64]:
    """Return the Cartesian coordinates of the atoms in ``order``, three per atom."""
    return np.asarray((3 * np.array(order)[:, None] + np.arange(3)).ravel())


def reordered(
    molecule: Geometry, hessian: FloatArray, order: tuple[int, ...]
) -> tuple[Geometry, FloatArray]:
    """Return the molecule with atom ``order[k]`` in position k."""
    index = coordinate_index(order)
    return molecule.reordered(list(order)), hessian[np.ix_(index, index)]


@dataclass(frozen=True, eq=False)
class Change:
    """One way of giving the same molecule: its atoms listed in another order, then moved.

    Attributes:
        name: The words a certificate names it by.
        matrix: An orthogonal map of space, applied to every atom.
        order: Atom ``order[k]`` of the molecule is put in position k.
    """

    name: str
    matrix: FloatArray
    order: tuple[int, ...]

    def apply(self, molecule: Geometry, hessian: FloatArray) -> tuple[Geometry, FloatArray]:
        """Return the changed molecule and its Hessian."""
        return moved(*reordered(molecule, hessian, self.order), self.matrix)

    def undo(self, delta: FloatArray) -> FloatArray:
        """Return a Hessian of the changed molecule as one of the molecule itself."""
        blocks = np.kron(np.eye(len(self.order)), self.matrix)
        index = coordinate_index(self.order)
        back = np.zeros_like(delta)
        back[np.ix_(index, index)] = blocks.T @ delta @ blocks
        return back


def bond_graph(smiles: str, molecule: Geometry) -> Mol:
    """Return the molecule's graph with hydrogens, in aromir's canonical atom order.

    That is the order of the SMILES aromir writes with its hydrogens added after it, the
    order every geometry the pipeline computes is in.

    Raises:
        ValueError: The SMILES does not parse, or the geometry is not in that order: an
            element differs or a bond of the graph is too long in it.
    """
    parsed: Mol | None = Chem.MolFromSmiles(smiles)  # RDKit's stubs omit the None
    if parsed is None:
        msg = f"{smiles!r} does not parse as a SMILES"
        raise ValueError(msg)
    graph = Chem.AddHs(parsed)
    if tuple(atom.GetSymbol() for atom in graph.GetAtoms()) != molecule.symbols:
        msg = "the geometry's elements are not in the canonical order of its SMILES"
        raise ValueError(msg)
    table = Chem.GetPeriodicTable()
    for bond in graph.GetBonds():
        first, second = bond.GetBeginAtom(), bond.GetEndAtom()
        reach = BOND_SLACK * sum(
            table.GetRcovalent(atom.GetAtomicNum()) for atom in (first, second)
        )
        vector = molecule.coords_bohr[first.GetIdx()] - molecule.coords_bohr[second.GetIdx()]
        if float(np.linalg.norm(vector)) * ANGSTROM_PER_BOHR > reach:
            msg = "a bond of the SMILES is too long in the geometry: its atoms are in another order"
            raise ValueError(msg)
    return graph


def _listings(graph: Mol, rng: np.random.Generator) -> list[tuple[int, ...]]:
    """Return the canonical orders of random listings, in terms of the molecule's atoms."""
    orders = []
    for _ in range(N_LISTINGS):
        listing = [int(atom) for atom in rng.permutation(graph.GetNumAtoms())]
        relisted = Chem.RenumberAtoms(graph, listing)
        orders.append(tuple(listing[k] for k in canonical_order(relisted)))
    return orders


def _swaps(graph: Mol, rng: np.random.Generator) -> list[tuple[int, ...]]:
    """Return up to ``N_SWAPS`` graph automorphisms other than the identity, drawn."""
    found = graph.GetSubstructMatches(
        graph, uniquify=False, useChirality=False, maxMatches=MAX_SYMMETRIES
    )
    identity = tuple(range(graph.GetNumAtoms()))
    others = sorted({tuple(int(k) for k in match) for match in found} - {identity})
    if len(others) <= N_SWAPS:
        return others
    drawn = rng.choice(len(others), size=N_SWAPS, replace=False)
    return [others[k] for k in sorted(int(k) for k in drawn)]


def changes(graph: Mol) -> list[Change]:
    """Return the changes the check tries, in the order they are drawn."""
    rng = np.random.default_rng(SEED)
    identity = tuple(range(graph.GetNumAtoms()))
    tried = [
        Change(f"rotation {k + 1}", random_rotation(rng), identity) for k in range(N_ROTATIONS)
    ]
    tried.append(Change("the mirror image", MIRROR, identity))
    tried.append(Change("the mirror image, rotated", MIRROR @ random_rotation(rng), identity))
    seen = {identity}
    for kind, orders in (
        ("another listing", _listings(graph, rng)),
        ("a swap of equivalent atoms", _swaps(graph, rng)),
    ):
        new = [order for order in dict.fromkeys(orders) if order not in seen]
        seen.update(new)
        tried += [Change(f"{kind} {k + 1}", np.eye(3), order) for k, order in enumerate(new)]
    return tried


KINDS: Final = {
    "rotation": ("rotation", "rotations"),
    "the mirror image": ("mirror image", "mirror images"),
    "another listing": ("other listing of the atoms", "other listings of the atoms"),
    "a swap of equivalent atoms": ("swap of equivalent atoms", "swaps of equivalent atoms"),
}
"""The kinds of change, by the start of their names, with the words a count of them takes."""


def describe(tried: tuple[str, ...]) -> str:
    """Say how many changes of each kind were tried, as "8 rotations, 2 mirror images"."""
    parts = []
    for start, (one, many) in KINDS.items():
        n = sum(name.startswith(start) for name in tried)
        if n:
            parts.append(f"{n} {one if n == 1 else many}")
    return ", ".join(parts)


@dataclass(frozen=True)
class OwnSpread:
    """How far each printed band moved over the changes, or why that was not measured.

    Attributes:
        spread_cm: Per band, in the order of the printed spectrum, its largest change of
            position, cm⁻¹, to ``DECIMALS``; empty when the check could not run.
        change: Per band, the change that gave it.
        tried: The names of every change tried.
        refusal: Why the check could not run; empty when it ran.
    """

    spread_cm: tuple[float, ...]
    change: tuple[str, ...]
    tried: tuple[str, ...]
    refusal: str

    @property
    def largest(self) -> tuple[float, str]:
        """The largest spread of any band and the change that gave it.

        Raises:
            ValueError: The check did not run, so nothing was measured.
        """
        if not self.spread_cm:
            msg = f"no spread was measured: {self.refusal}"
            raise ValueError(msg)
        k = int(np.argmax(self.spread_cm))
        return self.spread_cm[k], self.change[k]


def _corrected_modes(
    release: Release, molecule: Geometry, low: LowLevel, change: Change
) -> vibrations.Modes:
    """Return the modes of the low-level Hessian with the changed molecule's correction."""
    as_trained = low.as_trained(soft_modes_measured=release.training.soft_mode_repair)
    answer = release.corrections(*change.apply(molecule, as_trained)).mean
    return vibrations.vibrational_modes(low.hessian + change.undo(answer), molecule)


def measure(
    release: Release, smiles: str, molecule: Geometry, low: LowLevel, correction: FloatArray
) -> OwnSpread:
    """Return how far the bands of ``low.hessian + correction`` move over the changes.

    Args:
        release: The networks whose answer is checked.
        smiles: The molecule's SMILES as admission writes it.
        molecule: Its geometry, in aromir's canonical atom order.
        low: Its low-level Hessians.
        correction: The printed correction, the seeds' mean for the molecule as computed.
    """
    printed = vibrations.vibrational_modes(low.hessian + correction, molecule)
    try:
        tried = changes(bond_graph(smiles, molecule))
        moved_modes = [_corrected_modes(release, molecule, low, change) for change in tried]
    except (AromirError, ValueError) as error:
        return OwnSpread(spread_cm=(), change=(), tried=(), refusal=str(error))
    shifts = np.zeros((len(tried), len(printed.wavenumbers)))
    for row, modes in zip(shifts, moved_modes, strict=True):
        partner = spectrum.partners(modes, printed)
        row[partner] = np.abs(modes.wavenumbers - printed.wavenumbers[partner])
    worst = shifts.argmax(axis=0)
    return OwnSpread(
        spread_cm=tuple(round(float(x), DECIMALS) for x in shifts.max(axis=0)),
        change=tuple(tried[k].name for k in worst),
        tried=tuple(change.name for change in tried),
        refusal="",
    )
