"""The bond graph: rings, atom classes and hinge classes on molecules whose answer is known."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from aromir import geometry
from aromir.network import graph


def _classes(smiles: str) -> Counter[str]:
    start = geometry.start_from_smiles(smiles)
    adjacency = graph.feature_bonds(start)
    return Counter(graph.atom_classes(start.symbols, adjacency, graph.rings(adjacency)))


def _hinges(smiles: str) -> Counter[str]:
    start = geometry.start_from_smiles(smiles)
    return Counter(graph.HINGE_CLASSES[index] for index in graph.hinge_classes(start))


def test_benzene_is_one_ring_of_bonded_carbons() -> None:
    benzene = geometry.start_from_smiles("c1ccccc1")
    adjacency = graph.feature_bonds(benzene)
    assert [len(neighbours) for neighbours in adjacency] == [3] * 6 + [1] * 6
    ring_sets = graph.rings(adjacency)
    assert ring_sets == [frozenset(range(6))]
    cycle = graph.ring_cycles(adjacency, ring_sets)[0]
    assert sorted(cycle) == list(range(6))
    assert all(cycle[k] in adjacency[cycle[k - 1]] for k in range(6))


def test_the_enclosing_ring_of_a_fused_pair_is_not_a_ring() -> None:
    naphthalene = geometry.start_from_smiles("c1ccc2ccccc2c1")
    ring_sets = graph.rings(graph.feature_bonds(naphthalene))
    assert sorted(len(ring) for ring in ring_sets) == [6, 6]
    assert graph.rings_per_atom(18, ring_sets).tolist().count(2) == 2


def test_graph_distances_count_bonds() -> None:
    benzene = geometry.start_from_smiles("c1ccccc1")
    distances = graph.graph_distances(graph.feature_bonds(benzene))
    assert distances[0, 0] == 0
    assert distances[0, 3] == 3  # para carbons
    assert int(np.max(distances)) == 5  # para hydrogens
    assert (distances == distances.T).all()


def test_disconnected_atoms_are_at_distance_minus_one() -> None:
    assert graph.graph_distances([{1}, {0}, set()]).tolist() == [
        [0, 1, -1],
        [1, 0, -1],
        [-1, -1, 0],
    ]


@pytest.mark.parametrize(
    ("smiles", "expected"),
    [
        ("c1ccccc1", {"C_edgeH": 6, "H_quartet": 6}),
        ("c1ccc2ccccc2c1", {"C_edgeH": 8, "C_fused": 2, "H_quartet": 8}),
        (
            "Cc1ccccc1",
            {"C_sp3": 1, "C_edgeX": 1, "C_edgeH": 5, "H_subst": 3, "H_quartet": 5},
        ),
        (
            "c1ccc2sccc2c1",
            {"C_edgeH": 6, "C_fused": 2, "X_ring": 1, "H_quartet": 4, "H_duo": 2},
        ),
        (
            "N#Cc1ccccc1",
            {"X_nonring": 1, "C_nonring_sp2": 1, "C_edgeX": 1, "C_edgeH": 5, "H_quartet": 5},
        ),
        ("Cc1cc(C)cc(C)c1", {"C_sp3": 3, "C_edgeX": 3, "C_edgeH": 3, "H_subst": 9, "H_solo": 3}),
        ("Cc1ccc(C)c(C)c1C", {"C_sp3": 4, "C_edgeX": 4, "C_edgeH": 2, "H_subst": 12, "H_duo": 2}),
    ],
)
def test_atom_classes(smiles: str, expected: dict[str, int]) -> None:
    assert _classes(smiles) == expected


@pytest.mark.parametrize(
    ("smiles", "expected"),
    [
        ("c1ccccc1", {"none": 12}),
        ("Cc1ccccc1", {"sp3": 1, "none": 14}),
        ("N#Cc1ccccc1", {"linear": 2, "rotor": 1, "none": 10}),
        ("c1ccc(cc1)-c1ccccc1", {"rotor": 2, "none": 20}),
        ("c1ccc2sccc2c1", {"ring5": 5, "none": 10}),
        ("c1ccc2c(c1)c1ccccc12", {"ring4": 4, "none": 16}),
    ],
)
def test_hinge_classes(smiles: str, expected: dict[str, int]) -> None:
    assert _hinges(smiles) == expected
