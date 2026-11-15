"""A start geometry in one canonical atom order, whatever way the molecule was written.

The learned correction is not yet invariant to the atom order, so every input must reach
it in the same order: the heavy atoms as RDKit's canonical SMILES writes them, then each
heavy atom's hydrogens. A SMILES gets a force-field geometry in that order; an xyz file
keeps its own coordinates, reordered.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from rdkit import Chem
from rdkit.Chem import rdDistGeom, rdForceFieldHelpers

from aromir.molecule import Geometry

if TYPE_CHECKING:
    from aromir.admission import Mol

EMBEDDING_SEED: Final = 0
"""The random seed of the distance-geometry embedding: one SMILES, one start geometry."""

FORCE_FIELD_ITERATIONS: Final = 2000
"""Upper limit of the MMFF94 relaxation's steps."""

_EMBEDDED = 0


def canonical_order(molecule: Mol) -> list[int]:
    """Return the molecule's atom indices in the canonical order.

    Hydrogens on one heavy atom are equivalent in the bond graph and keep their input order.

    Raises:
        ValueError: A hydrogen is not an ordinary one on a single heavy atom (a lone,
            bridging or isotope-labelled hydrogen), so the order has no place for it.
    """
    heavy = [atom.GetIdx() for atom in molecule.GetAtoms() if atom.GetAtomicNum() > 1]
    skeleton = Chem.RemoveHs(molecule)
    if skeleton.GetNumAtoms() != len(heavy):
        msg = "a hydrogen is not an ordinary one on a single heavy atom: it cannot be ordered"
        raise ValueError(msg)
    Chem.MolToSmiles(skeleton)
    written = [heavy[k] for k in skeleton.GetPropsAsDict(True, True)["_smilesAtomOutputOrder"]]  # noqa: FBT003  RDKit's positional flags: private and computed properties
    hydrogens = [
        neighbour.GetIdx()
        for centre in written
        for neighbour in sorted(
            molecule.GetAtomWithIdx(centre).GetNeighbors(), key=lambda atom: atom.GetIdx()
        )
        if neighbour.GetAtomicNum() == 1
    ]
    return written + hydrogens


def _symbols(molecule: Mol) -> tuple[str, ...]:
    return tuple(atom.GetSymbol() for atom in molecule.GetAtoms())


def start_from_smiles(canonical_smiles: str) -> Geometry:
    """Embed a canonical SMILES in 3D (ETKDG) and relax it with MMFF94.

    The atoms come out in the canonical order because RDKit numbers a molecule parsed from
    its canonical SMILES in the order that SMILES writes it.

    Raises:
        ValueError: RDKit finds no 3D embedding.
    """
    molecule = Chem.AddHs(Chem.MolFromSmiles(canonical_smiles))
    if rdDistGeom.EmbedMolecule(molecule, randomSeed=EMBEDDING_SEED) != _EMBEDDED:
        embedded = rdDistGeom.EmbedMolecule(
            molecule, randomSeed=EMBEDDING_SEED, useRandomCoords=True
        )
        if embedded != _EMBEDDED:
            msg = f"no 3D embedding found for {canonical_smiles}"
            raise ValueError(msg)
    rdForceFieldHelpers.MMFFOptimizeMolecule(molecule, maxIters=FORCE_FIELD_ITERATIONS)
    return Geometry.from_angstrom(_symbols(molecule), molecule.GetConformer().GetPositions())


def start_from_xyz(molecule: Mol) -> Geometry:
    """Return an xyz molecule's own geometry with its atoms in the canonical order."""
    as_read = Geometry.from_angstrom(_symbols(molecule), molecule.GetConformer().GetPositions())
    return as_read.reordered(canonical_order(molecule))
