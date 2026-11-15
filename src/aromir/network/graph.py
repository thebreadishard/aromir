"""The bond graph of a geometry and what the network reads from it.

Bonds are told from distances and covalent radii. Two thresholds are in use because the
network's two graph inputs were defined at different times and the weights are trained
on each as it is: the primitive and pair features at 1.2 times the radii, the hinge
classes at 1.25 times.

The functions iterate over sets of atom indices on purpose: the weights are trained on
features built in exactly this order, and the regression pins hold it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import numpy as np

from aromir.constants import ANGSTROM_PER_BOHR

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from aromir.molecule import Geometry

Adjacency = list[set[int]]

COVALENT_RADIUS_ANGSTROM: Final = {
    "H": 0.31,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "F": 0.57,
    "S": 1.05,
    "Cl": 1.02,
}

FEATURE_BOND_TOLERANCE: Final = 1.2
HINGE_BOND_TOLERANCE: Final = 1.25

SMALLEST_RING: Final = 5
LARGEST_RING: Final = 7

HYDROGEN_CLASSES: Final = ("H_solo", "H_duo", "H_trio", "H_quartet", "H_subst")
CARBON_CLASSES: Final = ("C_fused", "C_edgeH", "C_edgeX", "C_nonring_sp2", "C_sp3")
HETERO_CLASSES: Final = ("X_ring", "X_nonring")
ATOM_CLASSES: Final = HYDROGEN_CLASSES + CARBON_CLASSES + HETERO_CLASSES

HINGE_CLASSES: Final = ("none", "rotor", "ring5", "ring4", "linear", "sp3")

_SP3_NEIGHBOURS: Final = 4
_FUSED_RINGS: Final = 2
_LINEAR_ANGLE_DEGREES: Final = 170.0
_TRIPLE_BOND_ANGSTROM: Final = 1.25
_PATH_LIMIT: Final = 8
_FOUR_RING: Final = 4
_FIVE_RING: Final = 5


def _distances_angstrom(geometry: Geometry) -> NDArray[np.float64]:
    x = geometry.coords_bohr * ANGSTROM_PER_BOHR
    return np.asarray(np.linalg.norm(x[:, None, :] - x[None, :, :], axis=-1), dtype=float)


def _radius_sums(geometry: Geometry) -> NDArray[np.float64]:
    radii = np.array([COVALENT_RADIUS_ANGSTROM[symbol] for symbol in geometry.symbols])
    return np.asarray(radii[:, None] + radii[None, :], dtype=float)


def _adjacency(bonded: NDArray[np.bool_]) -> Adjacency:
    adjacency: Adjacency = [set() for _ in range(len(bonded))]
    for first in range(len(bonded)):
        for second in range(first + 1, len(bonded)):
            if bonded[first, second]:
                adjacency[first].add(second)
                adjacency[second].add(first)
    return adjacency


def feature_bonds(geometry: Geometry) -> Adjacency:
    """Return the bond graph the primitive and pair features are built on."""
    limit = FEATURE_BOND_TOLERANCE * _radius_sums(geometry)
    return _adjacency(_distances_angstrom(geometry) < limit)


def hinge_bonds(geometry: Geometry) -> Adjacency:
    """Return the bond graph the hinge classes are built on."""
    limit = HINGE_BOND_TOLERANCE * _radius_sums(geometry)
    return _adjacency(_distances_angstrom(geometry) <= limit)


def rings(adjacency: Adjacency) -> list[frozenset[int]]:
    """Return every simple cycle of five to seven atoms, each once."""
    found: set[frozenset[int]] = set()
    for start in range(len(adjacency)):
        stack = [(start, [start])]
        while stack:
            atom, path = stack.pop()
            for neighbour in adjacency[atom]:
                if neighbour == start and len(path) >= 3:
                    found.add(frozenset(path))
                elif neighbour not in path and len(path) < LARGEST_RING and neighbour > start:
                    stack.append((neighbour, [*path, neighbour]))
    return [ring for ring in found if SMALLEST_RING <= len(ring) <= LARGEST_RING]


def ring_cycles(adjacency: Adjacency, ring_sets: list[frozenset[int]]) -> list[list[int]]:
    """Return each ring's atoms in the order a walk around the ring meets them."""
    cycles = []
    for ring in ring_sets:
        current = min(ring)
        order = [current]
        previous = None
        while True:
            onward = [
                atom
                for atom in adjacency[current]
                if atom in ring and atom != previous and atom not in order
            ]
            if not onward:
                break
            previous, current = current, onward[0]
            order.append(current)
        cycles.append(order if len(order) == len(ring) else sorted(ring))
    return cycles


def rings_per_atom(n_atoms: int, ring_sets: list[frozenset[int]]) -> NDArray[np.int64]:
    """Return how many rings each atom belongs to."""
    counts = np.zeros(n_atoms, dtype=np.int64)
    for ring in ring_sets:
        for atom in ring:
            counts[atom] += 1
    return counts


def graph_distances(adjacency: Adjacency) -> NDArray[np.int64]:
    """Return all-pairs shortest path lengths in bonds; −1 between disconnected atoms."""
    n_atoms = len(adjacency)
    distances = -np.ones((n_atoms, n_atoms), dtype=np.int64)
    for source in range(n_atoms):
        distances[source, source] = 0
        queue = [source]
        for atom in queue:
            for neighbour in adjacency[atom]:
                if distances[source, neighbour] < 0:
                    distances[source, neighbour] = distances[source, atom] + 1
                    queue.append(neighbour)
    return distances


def _heavy_atom_class(symbol: str, n_neighbours: int, n_rings: int, *, has_hydrogen: bool) -> str:
    if symbol != "C":
        return "X_ring" if n_rings else "X_nonring"
    if n_neighbours >= _SP3_NEIGHBOURS:
        return "C_sp3"
    if n_rings >= _FUSED_RINGS:
        return "C_fused"
    if n_rings == 1:
        return "C_edgeH" if has_hydrogen else "C_edgeX"
    return "C_nonring_sp2"


def _hydrogen_run(
    carbon: int, ring_ch: set[int], adjacency: Adjacency, ring_sets: list[frozenset[int]]
) -> int:
    """Count the adjacent ring C-H carbons in one row that ``carbon`` belongs to."""
    run = {carbon}
    frontier = [carbon]
    while frontier:
        atom = frontier.pop()
        for neighbour in adjacency[atom]:
            shares_a_ring = any(atom in ring and neighbour in ring for ring in ring_sets)
            if neighbour in ring_ch and neighbour not in run and shares_a_ring:
                run.add(neighbour)
                frontier.append(neighbour)
    return len(run)


def atom_classes(
    symbols: tuple[str, ...], adjacency: Adjacency, ring_sets: list[frozenset[int]]
) -> list[str]:
    """Return each atom's environment class, one of ``ATOM_CLASSES``.

    A ring hydrogen is classed by how many C-H carbons stand in a row along its ring
    (solo, duo, trio, quartet); a carbon by its ring role; a heteroatom by ring membership.
    """
    n_rings = rings_per_atom(len(symbols), ring_sets)
    hydrogen_on = [any(symbols[b] == "H" for b in adjacency[a]) for a in range(len(symbols))]
    ring_ch = {a for a, s in enumerate(symbols) if s == "C" and n_rings[a] == 1 and hydrogen_on[a]}
    classes = []
    for atom, symbol in enumerate(symbols):
        if symbol != "H":
            classes.append(
                _heavy_atom_class(
                    symbol,
                    len(adjacency[atom]),
                    int(n_rings[atom]),
                    has_hydrogen=hydrogen_on[atom],
                )
            )
            continue
        carbon = next(iter(adjacency[atom])) if adjacency[atom] else None
        if carbon is None or carbon not in ring_ch:
            classes.append("H_subst")
            continue
        run = _hydrogen_run(carbon, ring_ch, adjacency, ring_sets)
        classes.append(HYDROGEN_CLASSES[min(run, 4) - 1])
    return classes


def _path_length(
    adjacency: Adjacency,
    start: int,
    end: int,
    *,
    skip_atom: int | None = None,
    skip_bond: frozenset[int] | None = None,
) -> int | None:
    """Return the shortest path length avoiding an atom or a bond; None beyond eight bonds."""
    seen = {start}
    frontier = [start]
    length = 0
    while frontier and length < _PATH_LIMIT:
        length += 1
        reached = []
        for atom in frontier:
            for neighbour in adjacency[atom]:
                blocked = neighbour == skip_atom or frozenset((atom, neighbour)) == skip_bond
                if blocked or neighbour in seen:
                    continue
                if neighbour == end:
                    return length
                seen.add(neighbour)
                reached.append(neighbour)
        frontier = reached
    return None


def _smallest_ring(adjacency: Adjacency, atom: int) -> int:
    """Return the size of the smallest ring through an atom, 0 without one."""
    neighbours = sorted(adjacency[atom])
    sizes = [
        length + 2
        for u in neighbours
        for v in neighbours
        if u < v
        for length in [_path_length(adjacency, u, v, skip_atom=atom)]
        if length is not None
    ]
    return min(sizes, default=0)


def _is_linear(geometry: Geometry, adjacency: Adjacency, atom: int, heavy: list[bool]) -> bool:
    """Tell a two-coordinate atom with a straight angle, or the end of a triple bond."""
    x = geometry.coords_bohr * ANGSTROM_PER_BOHR
    neighbours = sorted(adjacency[atom])
    if len(neighbours) == 2:
        u, v = x[neighbours[0]] - x[atom], x[neighbours[1]] - x[atom]
        cosine = np.clip(u @ v / np.linalg.norm(u) / np.linalg.norm(v), -1, 1)
        if float(np.degrees(np.arccos(cosine))) > _LINEAR_ANGLE_DEGREES:
            return True
    return (
        len(neighbours) == 1
        and heavy[neighbours[0]]
        and float(np.linalg.norm(x[neighbours[0]] - x[atom])) < _TRIPLE_BOND_ANGSTROM
    )


def _is_rotor(adjacency: Adjacency, atom: int, heavy: list[bool], heavy_degree: list[int]) -> bool:
    """Tell an atom on a bond between two branched heavy atoms that closes no ring."""
    if heavy_degree[atom] < 2:
        return False
    return any(
        heavy[other]
        and heavy_degree[other] >= 2
        and _path_length(adjacency, atom, other, skip_bond=frozenset((atom, other))) is None
        for other in sorted(adjacency[atom])
    )


def hinge_classes(geometry: Geometry) -> NDArray[np.int64]:
    """Return each atom's hinge class as an index into ``HINGE_CLASSES``.

    The classes mark where soft modes come from: a small ring, a linear group, a bond
    between ring systems, an sp³ carbon. The first that applies, in that order, is taken.
    """
    adjacency = hinge_bonds(geometry)
    heavy = [symbol != "H" for symbol in geometry.symbols]
    heavy_degree = [sum(heavy[b] for b in adjacency[a]) for a in range(geometry.n_atoms)]
    classes = np.zeros(geometry.n_atoms, dtype=np.int64)
    for atom, symbol in enumerate(geometry.symbols):
        ring = _smallest_ring(adjacency, atom)
        if ring == _FOUR_RING:
            name = "ring4"
        elif ring == _FIVE_RING:
            name = "ring5"
        elif heavy[atom] and _is_linear(geometry, adjacency, atom, heavy):
            name = "linear"
        elif heavy[atom] and _is_rotor(adjacency, atom, heavy, heavy_degree):
            name = "rotor"
        elif symbol == "C" and len(adjacency[atom]) == _SP3_NEIGHBOURS:
            name = "sp3"
        else:
            name = "none"
        classes[atom] = HINGE_CLASSES.index(name)
    return classes
