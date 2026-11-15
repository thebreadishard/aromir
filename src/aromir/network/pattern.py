"""Which pairs of primitives the network corrects, and the features of each pair.

The correction ΔF is sparse: an entry exists for every primitive with itself, for two
primitives that share an atom, for two bonds of one ring, and for two primitives whose
atoms lie at most ``reach`` bonds apart. Everything else stays zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal

import numpy as np

from aromir.network import graph
from aromir.network.primitives import BOND, DIHEDRAL, OUT_OF_PLANE, PRIMITIVE_KINDS, kind

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from aromir.molecule import FloatArray
    from aromir.network.primitives import Primitive, Primitives, PrimitiveSettings

Pattern = Literal["c", "d", "e", "f"]

REACH: Final[dict[str, int]] = {"c": 0, "d": 1, "e": 2, "f": 3}
"""Largest bond-graph distance between two disjoint primitives a pattern still pairs."""

FEATURE_ELEMENTS: Final = ("H", "C", "N", "O", "S", "F", "Cl")

PAIR_CLASSES: Final = (
    "diag_bond",
    "diag_angle",
    "diag_dihedral",
    "diag_other",
    "off_bondbond",
    "off_other",
    "off_dist1",
    "off_dist2",
    "off_dist3",
)
_DIAGONAL_CLASS: Final = {BOND: 0, 1: 1, DIHEDRAL: 2}
_DIAGONAL_OTHER: Final = 3
_BOND_BOND: Final = 4
_OFF_OTHER: Final = 5
_FIRST_DISTANCE_CLASS: Final = 5  # a pair d bonds apart is class 5 + d

_ELEMENT_SLOT: Final = len(PRIMITIVE_KINDS)
_RING_FLAG_SLOT: Final = _ELEMENT_SLOT + len(FEATURE_ELEMENTS)
_VALUE_SLOT: Final = _RING_FLAG_SLOT + 1
_FORCE_CONSTANT_SLOT: Final = _RING_FLAG_SLOT + 2
_ATOM_CLASS_SLOT: Final = _RING_FLAG_SLOT + 3
_RING_COUNT_SLOT: Final = _ATOM_CLASS_SLOT + len(graph.ATOM_CLASSES)
N_PRIMITIVE_FEATURES: Final = _RING_COUNT_SLOT + 1
N_PAIR_EXTRAS: Final = 8


def n_pair_features(settings: PrimitiveSettings) -> int:
    """Return the length of a pair's feature vector under the given encoding."""
    n_primitive = N_PRIMITIVE_FEATURES + (settings.dihedral_encoding == "cossin")
    return 2 * n_primitive + N_PAIR_EXTRAS


@dataclass(frozen=True, eq=False)
class PairPattern:
    """The pattern of one molecule.

    Attributes:
        pairs: Primitive index pairs (i ≤ j), shape (P, 2).
        classes: Each pair's index into ``PAIR_CLASSES``, shape (P,).
        features: Each pair's feature vector, shape (P, ``n_pair_features``), float32.
    """

    pairs: NDArray[np.int64]
    classes: NDArray[np.int64]
    features: NDArray[np.float32]


@dataclass(frozen=True, eq=False)
class _Topology:
    """What the pattern reads of the bond graph."""

    atom_classes: list[str]
    rings_per_atom: NDArray[np.int64]
    ring_bonds: dict[frozenset[int], dict[int, int]]
    """Per ring bond: the rings it belongs to, with its position in each ring's cycle."""
    cycle_lengths: list[int]
    distances: NDArray[np.int64]


def _topology(primitives: Primitives) -> _Topology:
    geometry = primitives.built_on
    adjacency = graph.feature_bonds(geometry)
    ring_sets = graph.rings(adjacency)
    cycles = graph.ring_cycles(adjacency, ring_sets)
    ring_bonds: dict[frozenset[int], dict[int, int]] = {}
    for ring, cycle in enumerate(cycles):
        for position, atom in enumerate(cycle):
            bond = frozenset((atom, cycle[(position + 1) % len(cycle)]))
            ring_bonds.setdefault(bond, {})[ring] = position
    return _Topology(
        atom_classes=graph.atom_classes(geometry.symbols, adjacency, ring_sets),
        rings_per_atom=graph.rings_per_atom(geometry.n_atoms, ring_sets),
        ring_bonds=ring_bonds,
        cycle_lengths=[len(cycle) for cycle in cycles],
        distances=graph.graph_distances(adjacency),
    )


def _in_ring(primitive: Primitive, primitive_kind: int, topology: _Topology) -> bool:
    """Tell whether a primitive lies in a ring: its bond or bonds, or its central atom."""
    ring_bonds = topology.ring_bonds
    if primitive_kind == BOND:
        return frozenset((primitive.a, primitive.b)) in ring_bonds
    if primitive_kind == 1:
        first, second = frozenset((primitive.a, primitive.b)), frozenset((primitive.b, primitive.c))
        return first in ring_bonds and second in ring_bonds
    if primitive_kind == DIHEDRAL:
        return frozenset((primitive.b, primitive.c)) in ring_bonds
    if primitive_kind == OUT_OF_PLANE:
        return bool(topology.rings_per_atom[primitive.a] > 0)
    return False


def _value_slots(value: float, primitive_kind: int, encoding: str) -> tuple[float, float]:
    """Return a primitive's value slot and sine slot; only a dihedral is encoded."""
    if primitive_kind != DIHEDRAL or encoding == "raw":
        return value, 0.0
    if encoding == "cos":
        return float(np.cos(value)), 0.0
    return float(np.cos(value)), float(np.sin(value))


def _primitive_features(
    primitives: Primitives, topology: _Topology, diagonal: FloatArray, encoding: str
) -> NDArray[np.float32]:
    """Return one feature vector per primitive.

    The slots: kind (one-hot), element counts, ring flag, value, the low level's own force
    constant, atom-class counts, the atoms' ring count and, under ``cossin``, sin φ.
    """
    symbols = primitives.built_on.symbols
    width = N_PRIMITIVE_FEATURES + (encoding == "cossin")
    features = np.zeros((len(primitives.internals), width), dtype=np.float32)
    for index, primitive in enumerate(primitives.internals):
        row = features[index]
        primitive_kind = kind(primitive)
        value, sine = _value_slots(primitives.value(index), primitive_kind, encoding)
        row[primitive_kind] = 1.0
        for atom in primitives.atoms(index):
            row[_ELEMENT_SLOT + FEATURE_ELEMENTS.index(symbols[atom])] += 1.0
            row[_ATOM_CLASS_SLOT + graph.ATOM_CLASSES.index(topology.atom_classes[atom])] += 1.0
            row[_RING_COUNT_SLOT] += topology.rings_per_atom[atom]
        row[_RING_FLAG_SLOT] = float(_in_ring(primitive, primitive_kind, topology))
        row[_VALUE_SLOT] = value
        row[_FORCE_CONSTANT_SLOT] = float(diagonal[index])
        if encoding == "cossin":
            row[N_PRIMITIVE_FEATURES] = sine
    return features


def _ring_path(first: frozenset[int], second: frozenset[int], topology: _Topology) -> int | None:
    """Return how many bonds apart two bonds lie along a ring they share; None without one."""
    rings_first = topology.ring_bonds.get(first, {})
    rings_second = topology.ring_bonds.get(second, {})
    common = set(rings_first) & set(rings_second)
    if not common:
        return None
    return min(
        min(
            abs(rings_first[ring] - rings_second[ring]),
            topology.cycle_lengths[ring] - abs(rings_first[ring] - rings_second[ring]),
        )
        for ring in common
    )


@dataclass(frozen=True)
class _Pair:
    """One pair the pattern holds: its class and its relation, as the features state it."""

    pair_class: int
    shared_atoms: int
    same_ring: bool
    ring_path: int


def _diagonal_pair(primitive_kind: int, n_atoms: int) -> _Pair:
    pair_class = _DIAGONAL_CLASS.get(primitive_kind, _DIAGONAL_OTHER)
    return _Pair(pair_class, n_atoms, same_ring=False, ring_path=0)


def _off_diagonal_pair(
    atoms: tuple[frozenset[int], frozenset[int]],
    both_bonds: bool,  # noqa: FBT001  one of two facts about the pair, passed positionally
    topology: _Topology,
    reach: int,
) -> _Pair | None:
    """Classify two distinct primitives, or return None when the pattern does not pair them."""
    shared = len(atoms[0] & atoms[1])
    ring_path = _ring_path(*atoms, topology) if both_bonds else None
    same_ring = ring_path is not None
    if shared == 0 and not same_ring:
        apart = min(topology.distances[a, b] for a in atoms[0] for b in atoms[1])
        if 1 <= apart <= reach:
            return _Pair(_FIRST_DISTANCE_CLASS + int(apart), 0, same_ring=False, ring_path=0)
        return None
    pair_class = _BOND_BOND if both_bonds else _OFF_OTHER
    return _Pair(pair_class, shared, same_ring, ring_path or 0)


def _extras(pair: _Pair, force_constants: FloatArray, i: int, j: int) -> list[float]:
    return [
        pair.shared_atoms,
        float(pair.same_ring),
        float(pair.ring_path == 1),
        float(pair.ring_path == 2),
        float(pair.ring_path >= 3),
        force_constants[i, j],
        force_constants[i, i] * force_constants[j, j],
        float(i == j),
    ]


def build(
    primitives: Primitives,
    force_constants: FloatArray,
    pattern: Pattern,
    settings: PrimitiveSettings,
) -> PairPattern:
    """Build a molecule's pair pattern with the features of every pair.

    Args:
        primitives: The molecule's primitives.
        force_constants: The low level's internal force constants F (K, K).
        pattern: Which disjoint pairs are held, by ``REACH``.
        settings: The dihedral encoding of the features.
    """
    topology = _topology(primitives)
    single = _primitive_features(
        primitives, topology, np.diag(force_constants), settings.dihedral_encoding
    )
    kinds = [kind(primitive) for primitive in primitives.internals]
    atoms = [frozenset(primitives.atoms(k)) for k in range(len(kinds))]
    pairs, classes, features = [], [], []
    for i in range(len(kinds)):
        for j in range(i, len(kinds)):
            if i == j:
                held: _Pair | None = _diagonal_pair(kinds[i], len(atoms[i]))
            else:
                both_bonds = kinds[i] == BOND and kinds[j] == BOND
                held = _off_diagonal_pair(
                    (atoms[i], atoms[j]), both_bonds, topology, REACH[pattern]
                )
            if held is None:
                continue
            extras = np.array(_extras(held, force_constants, i, j), dtype=np.float32)
            pairs.append((i, j))
            classes.append(held.pair_class)
            features.append(
                np.concatenate([single[i] + single[j], np.abs(single[i] - single[j]), extras])
            )
    return PairPattern(
        np.array(pairs, dtype=np.int64),
        np.array(classes, dtype=np.int64),
        np.stack(features).astype(np.float32),
    )
