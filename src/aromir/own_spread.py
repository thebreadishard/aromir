"""How far one molecule's corrected bands move when the same molecule is given another way.

The network's answer is not yet invariant: a molecule turned in space, its mirror image or
its atoms listed in another order can get other corrected positions (a sign of the
network's internal coordinates that the molecule does not fix). A release's invariance
record measures that on a handful of molecules; this module measures it on the molecule
at hand, so that each band's licence can rest on its own number. No quantum chemistry is
run: each presentation is the molecule's optimised geometry and the Hessian the networks
read, changed alike; the networks correct it, and the corrected Hessian is brought back to
the molecule as computed.

The presentations, with every draw from one generator seeded ``SEED``, in this order:

1. ``N_ROTATIONS`` random proper rotations (``random_rotation``, one draw each);
2. the mirror image (z → −z; no draw), and the mirror image of one more random rotation;
3. swaps of atoms the bond graph cannot tell apart. The symmetries of the heavy-atom
   skeleton are enumerated whole (``skeleton_symmetries``; those that keep each atom's
   element, charge, aromaticity and number of hydrogens), sorted, and the hydrogens on one
   heavy atom may be exchanged freely. Where skeleton symmetries and hydrogen exchanges
   together give at most ``N_SWAPS`` orders besides the molecule's own, all are tried, in
   sorted order, and nothing is drawn. Otherwise: every skeleton symmetry with the
   hydrogens following their heavy atom, or ``N_SWAPS`` of them drawn without replacement
   (``Generator.choice``, taken in sorted order) where there are more; then, for a
   molecule with a heavy atom that carries two or more hydrogens, ``N_HYDROGEN_SWAPS``
   draws of one skeleton symmetry (``Generator.integers``; the first is the molecule's
   own) with the hydrogens of each such atom, in ascending atom order, permuted
   (``Generator.permutation``);
4. ``N_LISTINGS`` random listings of the atoms (``Generator.permutation``), each put into
   aromir's canonical order (``geometry.canonical_order``), as an xyz file listed that way
   would be. Every order the canonical order can give is a swap of step 3; which one a
   listing gives depends on how the running RDKit breaks its ties, so the listings come
   last and no other presentation depends on them.

An order already tried, or the molecule's own, is not tried again. The certificate records
every presentation tried: a rotation by its matrix, an order by its atoms.

Each presentation's corrected Hessian is the seeds' mean correction of the changed
molecule, mapped back (Kᵀ ΔH K for an orthogonal map K applied atom by atom; the 3×3
blocks put back for an order) and added to the low-level Hessian. Its modes are paired
with the printed modes by largest squared overlap (``spectrum.partners``), not by rank, and
a band's shift is the distance of its partner from the printed position. Where a band's
partner holds less than ``PAIRED_OVERLAP`` of its motion, the pairing is not trusted: the
band and every band its partner shares ``MIXING_SHARE`` of motion with form a cluster,
whose printed positions and partners are compared in ascending order, and each band of the
cluster gets the cluster's largest shift. A band's spread is its largest shift over the
presentations: its distance from the printed presentation, not the range between
presentations.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, Literal

import numpy as np
import rdkit
from rdkit import Chem
from scipy.sparse.csgraph import connected_components

from aromir import spectrum, vibrations
from aromir.admission import without_hydrogens
from aromir.errors import AromirError
from aromir.geometry import canonical_order, changed_bonds
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
N_HYDROGEN_SWAPS: Final = 6
MAX_SYMMETRIES: Final = 1000
"""How many symmetries of the heavy-atom skeleton are enumerated at most. Coronene has 12
and 1,4-di-tert-butylbenzene 144; a molecule that reaches the limit is recorded as capped,
and its swaps are then drawn from those RDKit happened to enumerate first."""
MIRROR: Final = np.diag([1.0, 1.0, -1.0])
PAIRED_OVERLAP: Final = 0.7
"""A band is paired with one mode of a presentation when that mode holds at least this
share of its motion. On v1.4 the lowest shares of the stored molecules are 0.51 (benzene),
0.58 (biphenyl) and 0.64 (diphenyl ether)."""
MIXING_SHARE: Final = 0.1
"""Two bands belong to one cluster when the partner of one holds this share of the other."""
DECIMALS: Final = 2
"""The decimals a spread is stated with (``stated``); the licence reads it unrounded."""
UNBUILT: Final = (AromirError, ValueError, ArithmeticError, RuntimeError)
"""The errors that mean the networks could not correct one presentation: aromir's own, a
value the inputs refuse, and geomeTRIC's and torch's, which raise ``RuntimeError``. A
presentation that fails so withholds the licence; any other error is a defect and is
raised."""

Kind = Literal["rotation", "mirror image", "swap", "listing"]
COUNTED: Final[dict[Kind, tuple[str, str]]] = {
    "rotation": ("rotation", "rotations"),
    "mirror image": ("mirror image", "mirror images"),
    "swap": ("swap of equivalent atoms", "swaps of equivalent atoms"),
    "listing": ("other listing of the atoms", "other listings of the atoms"),
}
"""The words a count of each kind of presentation takes."""


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
    """One presentation of the molecule: its atoms listed in another order, then moved.

    Attributes:
        name: The words a certificate names it by.
        kind: What kind of presentation it is.
        matrix: An orthogonal map of space, applied to every atom.
        order: Atom ``order[k]`` of the molecule is put in position k.
    """

    name: str
    kind: Kind
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

    def record(self) -> dict[str, Any]:
        """Return what identifies the presentation: its map of space, or its atom order."""
        if self.kind in ("swap", "listing"):
            return {"name": self.name, "kind": self.kind, "order": list(self.order)}
        return {"name": self.name, "kind": self.kind, "matrix": self.matrix.tolist()}


def bond_graph(smiles: str) -> Mol:
    """Return a molecule's graph with its hydrogens, in aromir's canonical atom order.

    That is the order every geometry the pipeline computes is in: the heavy atoms as the
    canonical SMILES writes them, then each heavy atom's hydrogens
    (``geometry.start_from_smiles``).

    Raises:
        ValueError: The SMILES does not parse.
    """
    parsed: Mol | None = Chem.MolFromSmiles(smiles)  # RDKit's stubs omit the None
    if parsed is None:
        msg = f"{smiles!r} does not parse as a SMILES"
        raise ValueError(msg)
    return Chem.AddHs(parsed)


def graph_of(smiles: str, molecule: Geometry) -> Mol:
    """Return the bond graph of a geometry that is in the canonical order of its SMILES.

    Raises:
        ValueError: The geometry's atoms or bonds are not those of the SMILES in its order
            (``geometry.changed_bonds``).
    """
    changed = changed_bonds(smiles, molecule)
    if changed:
        msg = (
            "the geometry is not the molecule of its SMILES in the canonical atom order: "
            f"{'; '.join(changed)}"
        )
        raise ValueError(msg)
    return bond_graph(smiles)


@dataclass(frozen=True)
class Symmetries:
    """The orders of a molecule's atoms that its bond graph cannot tell apart.

    Attributes:
        n_atoms: The number of atoms, hydrogens included.
        skeleton: Each symmetry of the heavy-atom skeleton other than the identity, as the
            heavy atom put in each heavy atom's place, in the graph's atom indices, sorted.
        hydrogens: The hydrogens of each heavy atom that carries any, ascending.
        capped: Whether the enumeration stopped at ``MAX_SYMMETRIES``.
    """

    n_atoms: int
    skeleton: tuple[dict[int, int], ...]
    hydrogens: dict[int, tuple[int, ...]]
    capped: bool

    @property
    def n_exchanges(self) -> int:
        """In how many ways the hydrogens can be listed on their own heavy atoms."""
        return math.prod(math.factorial(len(found)) for found in self.hydrogens.values())

    @property
    def n_orders(self) -> int:
        """How many orders the graph cannot tell apart, the molecule's own included."""
        return (len(self.skeleton) + 1) * self.n_exchanges

    def order(
        self, skeleton: dict[int, int], exchange: dict[int, tuple[int, ...]]
    ) -> tuple[int, ...]:
        """Return the atom order of a skeleton symmetry with the hydrogens exchanged.

        The hydrogens of a heavy atom take the places of the hydrogens of the atom it
        replaces, in ascending order unless ``exchange`` permutes them.
        """
        order = list(range(self.n_atoms))
        for place, atom in skeleton.items():
            order[place] = atom
        for place, found in self.hydrogens.items():
            taken = self.hydrogens[skeleton.get(place, place)]
            permuted = exchange.get(place, tuple(range(len(found))))
            for position, hydrogen in enumerate(found):
                order[hydrogen] = taken[permuted[position]]
        return tuple(order)


def _alike(first: Any, second: Any) -> bool:  # noqa: ANN401  two RDKit atoms
    """Tell whether two heavy atoms are of one kind for a symmetry of the skeleton."""
    return bool(
        first.GetAtomicNum() == second.GetAtomicNum()
        and first.GetFormalCharge() == second.GetFormalCharge()
        and first.GetIsAromatic() == second.GetIsAromatic()
        and first.GetTotalNumHs(includeNeighbors=True)
        == second.GetTotalNumHs(includeNeighbors=True)
    )


def skeleton_symmetries(graph: Mol) -> Symmetries:
    """Return the symmetries of a molecule's heavy-atom skeleton and its hydrogens.

    RDKit matches the skeleton onto itself (``GetSubstructMatches`` with
    ``uniquify=False``, the call of ``testSubstructMatchAcetate`` in RDKit's
    ``Code/GraphMol/Wrap/rough_test.py``, which counts every match of a query's atoms and
    not one per set of atoms); a match is kept when every atom lands on one of its own kind
    (``_alike``), which a match alone does not ask of the hydrogens.
    """
    heavy = [atom.GetIdx() for atom in graph.GetAtoms() if atom.GetAtomicNum() > 1]
    skeleton = without_hydrogens(graph)
    if skeleton.GetNumAtoms() != len(heavy):
        msg = "a hydrogen is not an ordinary one on a single heavy atom"
        raise ValueError(msg)
    matches = skeleton.GetSubstructMatches(
        skeleton, uniquify=False, useChirality=False, maxMatches=MAX_SYMMETRIES
    )
    kept = sorted(
        tuple(int(k) for k in match)
        for match in matches
        if all(
            _alike(graph.GetAtomWithIdx(heavy[k]), graph.GetAtomWithIdx(heavy[image]))
            for k, image in enumerate(match)
        )
    )
    hydrogens = {
        atom: tuple(
            sorted(
                neighbour.GetIdx()
                for neighbour in graph.GetAtomWithIdx(atom).GetNeighbors()
                if neighbour.GetAtomicNum() == 1
            )
        )
        for atom in heavy
    }
    return Symmetries(
        n_atoms=graph.GetNumAtoms(),
        skeleton=tuple(
            {heavy[k]: heavy[image] for k, image in enumerate(match)}
            for match in kept
            if match != tuple(range(len(heavy)))
        ),
        hydrogens={atom: found for atom, found in hydrogens.items() if found},
        capped=len(matches) >= MAX_SYMMETRIES,
    )


def _every_order(found: Symmetries) -> list[tuple[int, ...]]:
    """Return every order of the atoms the graph cannot tell apart but its own, sorted."""
    places = sorted(found.hydrogens)
    exchanges = [
        dict(zip(places, permuted, strict=True))
        for permuted in itertools.product(
            *(itertools.permutations(range(len(found.hydrogens[place]))) for place in places)
        )
    ]
    orders = {
        found.order(skeleton, exchange)
        for skeleton in ({}, *found.skeleton)
        for exchange in exchanges
    }
    identity = found.order({}, {})
    return sorted(orders - {identity})


def _drawn_orders(found: Symmetries, rng: np.random.Generator) -> list[tuple[int, ...]]:
    """Return the swaps of a molecule with more orders than are all tried, as drawn."""
    skeleton = list(found.skeleton)
    if len(skeleton) > N_SWAPS:
        drawn = rng.choice(len(skeleton), size=N_SWAPS, replace=False)
        skeleton = [skeleton[k] for k in sorted(int(k) for k in drawn)]
    orders = [found.order(symmetry, {}) for symmetry in skeleton]
    places = sorted(place for place, held in found.hydrogens.items() if len(held) > 1)
    if not places:
        return orders
    with_own = [{}, *found.skeleton]
    for _ in range(N_HYDROGEN_SWAPS):
        symmetry = with_own[int(rng.integers(len(with_own)))]
        exchange = {
            place: tuple(int(k) for k in rng.permutation(len(found.hydrogens[place])))
            for place in places
        }
        orders.append(found.order(symmetry, exchange))
    return orders


def swaps(found: Symmetries, rng: np.random.Generator) -> list[tuple[int, ...]]:
    """Return the swaps of equivalent atoms the check tries (the module's step 3)."""
    if not found.capped and found.n_orders - 1 <= N_SWAPS:
        return _every_order(found)
    return _drawn_orders(found, rng)


def _listings(graph: Mol, rng: np.random.Generator) -> list[tuple[int, ...]]:
    """Return the canonical orders of random listings, in terms of the molecule's atoms."""
    orders = []
    for _ in range(N_LISTINGS):
        listing = [int(atom) for atom in rng.permutation(graph.GetNumAtoms())]
        relisted = Chem.RenumberAtoms(graph, listing)
        orders.append(tuple(listing[k] for k in canonical_order(relisted)))
    return orders


def changes(graph: Mol, found: Symmetries) -> list[Change]:
    """Return the presentations the check tries, in the order they are drawn.

    Raises:
        ValueError: A hydrogen of the graph has no place in the canonical order.
    """
    rng = np.random.default_rng(SEED)
    identity = tuple(range(graph.GetNumAtoms()))
    tried = [
        Change(f"rotation {k + 1}", "rotation", random_rotation(rng), identity)
        for k in range(N_ROTATIONS)
    ]
    tried.append(Change("the mirror image", "mirror image", MIRROR, identity))
    turned = MIRROR @ random_rotation(rng)
    tried.append(Change("the mirror image, rotated", "mirror image", turned, identity))
    seen = {identity}
    named: tuple[tuple[Kind, str, list[tuple[int, ...]]], ...] = (
        ("swap", "a swap of equivalent atoms", swaps(found, rng)),
        ("listing", "another listing", _listings(graph, rng)),
    )
    for kind, words, orders in named:
        new = [order for order in dict.fromkeys(orders) if order not in seen]
        seen.update(new)
        tried += [Change(f"{words} {k + 1}", kind, np.eye(3), order) for k, order in enumerate(new)]
    return tried


def describe(tried: tuple[Change, ...]) -> str:
    """Say how many presentations of each kind were tried, as "8 rotations, 2 mirror images"."""
    parts = []
    for kind, (one, many) in COUNTED.items():
        n = sum(change.kind == kind for change in tried)
        if n:
            parts.append(f"{n} {one if n == 1 else many}")
    return ", ".join(parts)


def stated(spread_cm: float, line_cm: float) -> str:
    """Return a spread as it is printed: to ``DECIMALS``, or to as many as show its side.

    A spread just beyond the licence line is not printed as the line itself: 3.004 against
    a line of 3 is "3.004", not "3.00".
    """
    for decimals in range(DECIMALS, 12):
        text = f"{spread_cm:.{decimals}f}"
        if (float(text) > line_cm) == (spread_cm > line_cm):
            return text
    return repr(spread_cm)


@dataclass(frozen=True, eq=False)
class OwnSpread:
    """How far each printed band moved over the presentations, or why that was not measured.

    Attributes:
        spread_cm: Per band, in the order of the printed spectrum, its largest distance
            from the printed position, cm⁻¹, unrounded; empty when the check did not run.
        change: Per band, the name of the presentation that gave it.
        tried: Every presentation tried.
        symmetries: The molecule's graph symmetries; None when the check did not get there.
        refusal: Why the check could not run; empty when it ran.
    """

    spread_cm: tuple[float, ...]
    change: tuple[str, ...]
    tried: tuple[Change, ...]
    symmetries: Symmetries | None
    refusal: str

    @property
    def largest(self) -> tuple[float, str]:
        """The largest spread of any band and the presentation that gave it.

        Raises:
            ValueError: The check did not run, so nothing was measured.
        """
        if not self.spread_cm:
            msg = f"no spread was measured: {self.refusal}"
            raise ValueError(msg)
        k = int(np.argmax(self.spread_cm))
        return self.spread_cm[k], self.change[k]

    def record(self) -> dict[str, Any]:
        """Return what a certificate keeps of the check, enough to make each presentation again.

        The RDKit version is named because the listings, and the swaps of a molecule whose
        symmetries reached ``MAX_SYMMETRIES``, depend on it.
        """
        found = self.symmetries
        return {
            "seed": SEED,
            "rdkit": rdkit.__version__,
            "max_symmetries": MAX_SYMMETRIES,
            "skeleton_symmetries": None if found is None else len(found.skeleton) + 1,
            "equivalent_orders": None if found is None else found.n_orders,
            "symmetries_capped": None if found is None else found.capped,
            "paired_overlap": PAIRED_OVERLAP,
            "mixing_share": MIXING_SHARE,
            "counts": {kind: sum(c.kind == kind for c in self.tried) for kind in COUNTED},
            "presentations": [change.record() for change in self.tried],
            "not_measured_because": self.refusal,
        }


def _not_measured(reason: str) -> OwnSpread:
    return OwnSpread(spread_cm=(), change=(), tried=(), symmetries=None, refusal=reason)


def band_shifts(modes: vibrations.Modes, printed: vibrations.Modes) -> FloatArray:
    """Return how far each printed band lies from its place in one presentation's modes.

    A band whose partner holds at least ``PAIRED_OVERLAP`` of its motion is at the distance
    of that partner. Bands that mix — two near-degenerate bands whose motion a presentation
    shares out anew — have no partner of their own, and the pairing may give the band that
    moved the partner that stayed: they are compared as a cluster, the printed positions
    and the partners' in ascending order, and each gets the cluster's largest distance.
    """
    overlap = (modes.vectors.T @ printed.vectors) ** 2
    partner = spectrum.partners(modes, printed)
    shifts = np.zeros(len(printed.wavenumbers))
    shifts[partner] = np.abs(modes.wavenumbers - printed.wavenumbers[partner])
    of_band = np.argsort(partner)
    share = overlap[of_band]
    mixed = np.flatnonzero(np.diag(share) < PAIRED_OVERLAP)
    if not mixed.size:
        return shifts
    linked = share >= MIXING_SHARE
    _, cluster_of = connected_components(linked | linked.T, directed=False)
    for cluster in np.unique(cluster_of[mixed]):
        bands = np.flatnonzero(cluster_of == cluster)
        here = np.sort(printed.wavenumbers[bands])
        there = np.sort(modes.wavenumbers[of_band[bands]])
        shifts[bands] = np.abs(there - here).max()
    return shifts


def corrected_modes(
    release: Release, molecule: Geometry, low: LowLevel, change: Change
) -> vibrations.Modes:
    """Return the modes of the low-level Hessian with the changed molecule's correction."""
    as_trained = low.as_trained(soft_modes_measured=release.training.soft_mode_repair)
    answer = release.corrections(*change.apply(molecule, as_trained)).mean
    return vibrations.vibrational_modes(low.hessian + change.undo(answer), molecule)


def measure(
    release: Release, smiles: str, molecule: Geometry, low: LowLevel, correction: FloatArray
) -> OwnSpread:
    """Return how far the bands of ``low.hessian + correction`` move over the presentations.

    A molecule whose presentations cannot be drawn, or one of whose presentations the
    networks cannot correct (``UNBUILT``), is not measured, and the reason names which.

    Args:
        release: The networks whose answer is checked.
        smiles: The molecule's SMILES as admission writes it.
        molecule: Its geometry, in aromir's canonical atom order.
        low: Its low-level Hessians.
        correction: The printed correction, the seeds' mean for the molecule as computed.
    """
    printed = vibrations.vibrational_modes(low.hessian + correction, molecule)
    try:
        graph = graph_of(smiles, molecule)
        found = skeleton_symmetries(graph)
        tried = changes(graph, found)
    except ValueError as error:
        return _not_measured(str(error))
    shifts = np.zeros((len(tried), len(printed.wavenumbers)))
    for row, change in zip(shifts, tried, strict=True):
        try:
            modes = corrected_modes(release, molecule, low, change)
        except UNBUILT as error:
            return _not_measured(
                f"the networks could not correct the molecule given as {change.name}: "
                f"{type(error).__name__}: {error}"
            )
        row[:] = band_shifts(modes, printed)
    worst = shifts.argmax(axis=0)
    return OwnSpread(
        spread_cm=tuple(float(x) for x in shifts.max(axis=0)),
        change=tuple(tried[k].name for k in worst),
        tried=tuple(tried),
        symmetries=found,
        refusal="",
    )
