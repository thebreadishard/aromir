"""Chemistry an admitted molecule holds that the network's training molecules do not.

Admission takes any neutral, closed-shell aromatic molecule of the supported elements, but
the training molecules hold less than that: charge separation only of a few kinds (nitro
groups, an aromatic N-oxide) and aromatic rings of a few sizes. A molecule beyond them is
computed and printed, and named as outside the training chemistry; no band of it is
licensed, as for a molecule outside the training sizes, since the measured error rests on
molecules of neither kind. What counts as inside is read from the release's own training
list, not from a hand-kept one, so a release trained on more chemistry admits more.
"""

from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING, Any

from rdkit import Chem

if TYPE_CHECKING:
    from collections.abc import Iterable

    from aromir.admission import Mol

ChargeSite = tuple[str, int, bool, int, tuple[tuple[str, int, str], ...]]
"""A charged atom's kind: its element, formal charge, aromaticity and hydrogens, and each
neighbour's element and formal charge with the type of the bond to it."""


def _molecule(smiles: str) -> Mol:
    molecule: Mol | None = Chem.MolFromSmiles(smiles)  # RDKit's stubs omit the None
    if molecule is None:
        msg = f"{smiles!r} does not parse as a SMILES"
        raise ValueError(msg)
    return molecule


def _site(atom: Any) -> ChargeSite:  # noqa: ANN401  an RDKit atom
    molecule = atom.GetOwningMol()
    neighbours = sorted(
        (
            neighbour.GetSymbol(),
            neighbour.GetFormalCharge(),
            str(molecule.GetBondBetweenAtoms(atom.GetIdx(), neighbour.GetIdx()).GetBondType()),
        )
        for neighbour in atom.GetNeighbors()
    )
    return (
        atom.GetSymbol(),
        atom.GetFormalCharge(),
        atom.GetIsAromatic(),
        atom.GetTotalNumHs(),
        tuple(neighbours),
    )


def charge_sites(smiles: str) -> dict[ChargeSite, str]:
    """Return each kind of charged atom a SMILES holds, with a fragment SMILES that shows it.

    The fragment is the charged atom with its neighbours, as ``c[N+](=O)[O-]``; of two
    atoms of one kind the smaller fragment is kept, so the choice is the order's no more.

    Raises:
        ValueError: The string does not parse as a SMILES.
    """
    molecule = _molecule(smiles)
    sites: dict[ChargeSite, str] = {}
    for atom in molecule.GetAtoms():
        if atom.GetFormalCharge():
            shown = [atom.GetIdx(), *(neighbour.GetIdx() for neighbour in atom.GetNeighbors())]
            fragment = str(Chem.MolFragmentToSmiles(molecule, atomsToUse=shown))
            site = _site(atom)
            sites[site] = min(sites.get(site, fragment), fragment)
    return sites


def aromatic_ring_sizes(smiles: str) -> frozenset[int]:
    """Return the sizes of the rings of a SMILES's smallest set whose bonds are all aromatic.

    Raises:
        ValueError: The string does not parse as a SMILES.
    """
    molecule = _molecule(smiles)
    return frozenset(
        len(ring)
        for ring in molecule.GetRingInfo().BondRings()
        if all(molecule.GetBondWithIdx(bond).GetIsAromatic() for bond in ring)
    )


@cache
def _training_chemistry(
    trained: tuple[str, ...],
) -> tuple[dict[ChargeSite, str], frozenset[int]]:
    sites: dict[ChargeSite, str] = {}
    sizes: set[int] = set()
    for smiles in trained:
        for site, shown in charge_sites(smiles).items():
            sites[site] = min(sites.get(site, shown), shown)
        sizes |= aromatic_ring_sizes(smiles)
    return sites, frozenset(sizes)


def outside_training_chemistry(smiles: str, trained: Iterable[str]) -> tuple[str, ...]:
    """Name what a molecule holds that no training molecule does, one sentence per kind.

    Two kinds are looked for: a charged atom whose element, charge, aromaticity,
    hydrogens and bonded neighbours no training molecule's charged atom shares (the
    charge separation of a zwitterion, an azide, an ylide or a sulfoxide in a release
    trained on nitro groups alone), and an aromatic ring of a size no training molecule
    has (tropone's seven-membered ring). Both lists of the molecule and of the training
    set are read from SMILES in the standard form admission writes.

    Raises:
        ValueError: A string does not parse as a SMILES.
    """
    known_sites, known_sizes = _training_chemistry(tuple(trained))
    caveats = []
    unseen = sorted(
        {shown for site, shown in charge_sites(smiles).items() if site not in known_sites}
    )
    if unseen:
        held = ", ".join(sorted(set(known_sites.values()))) or "none"
        caveats.append(
            f"charge separation of a kind no training molecule holds: {', '.join(unseen)} "
            f"(the charged atoms of the training molecules: {held})"
        )
    sizes = sorted(aromatic_ring_sizes(smiles) - known_sizes)
    if sizes:
        held_sizes = ", ".join(str(size) for size in sorted(known_sizes)) or "none"
        caveats.append(
            f"an aromatic ring of {' and '.join(str(size) for size in sizes)} atoms, a size no "
            f"training molecule has (theirs: {held_sizes})"
        )
    return tuple(caveats)
