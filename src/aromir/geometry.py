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

from aromir.admission import perceive_connectivity, without_hydrogens
from aromir.constants import ANGSTROM_PER_BOHR
from aromir.errors import RefusedError
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
    skeleton = without_hydrogens(molecule)
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
        RefusedError: RDKit finds no 3D embedding, so there is no geometry to start from.
    """
    molecule = Chem.AddHs(Chem.MolFromSmiles(canonical_smiles))
    if rdDistGeom.EmbedMolecule(molecule, randomSeed=EMBEDDING_SEED) != _EMBEDDED:
        embedded = rdDistGeom.EmbedMolecule(
            molecule, randomSeed=EMBEDDING_SEED, useRandomCoords=True
        )
        if embedded != _EMBEDDED:
            reason = (
                f"RDKit finds no 3D start geometry for {canonical_smiles}: give the molecule "
                "as an xyz file"
            )
            raise RefusedError([reason])
    rdForceFieldHelpers.MMFFOptimizeMolecule(molecule, maxIters=FORCE_FIELD_ITERATIONS)
    return Geometry.from_angstrom(_symbols(molecule), molecule.GetConformer().GetPositions())


def _pairs(molecule: Mol) -> set[tuple[int, int]]:
    pairs = set()
    for bond in molecule.GetBonds():
        first, second = sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
        pairs.add((first, second))
    return pairs


def changed_bonds(admitted_smiles: str, geometry: Geometry) -> list[str]:
    """Name the bonds a geometry in the canonical order has formed or broken.

    The admitted molecule's bonds are read from its canonical SMILES, whose atoms with
    their hydrogens added are in the canonical order (``start_from_smiles``); the
    geometry's are perceived from its interatomic distances as an xyz file's are
    (``admission.perceive_connectivity``; RDKit's ``testCtDConnectivity`` compares the
    bonds so perceived on a geometry with those of its SMILES). An optimisation that moved
    a proton, opened a ring or joined two atoms has computed another molecule than the one
    admitted.

    Raises:
        ValueError: The geometry's atoms are not the SMILES's, in its order.
    """
    admitted = Chem.AddHs(Chem.MolFromSmiles(admitted_smiles))
    if _symbols(admitted) != geometry.symbols:
        msg = f"the geometry's atoms {geometry.symbols} are not those of {admitted_smiles}"
        raise ValueError(msg)
    angstrom = geometry.coords_bohr * ANGSTROM_PER_BOHR
    rows = [
        f"{s} {x:.10f} {y:.10f} {z:.10f}"
        for s, (x, y, z) in zip(geometry.symbols, angstrom, strict=True)
    ]
    perceived = perceive_connectivity(
        Chem.MolFromXYZBlock("\n".join([str(geometry.n_atoms), "", *rows]) + "\n")
    )
    expected, found = _pairs(admitted), _pairs(perceived)

    def named(pair: tuple[int, int]) -> str:
        return "–".join(f"{geometry.symbols[k]}{k + 1}" for k in pair)

    return [f"the bond {named(pair)} broke" for pair in sorted(expected - found)] + [
        f"a bond {named(pair)} formed" for pair in sorted(found - expected)
    ]


def start_from_xyz(molecule: Mol) -> Geometry:
    """Return an xyz molecule's own geometry with its atoms in the canonical order."""
    as_read = Geometry.from_angstrom(_symbols(molecule), molecule.GetConformer().GetPositions())
    return as_read.reordered(canonical_order(molecule))
