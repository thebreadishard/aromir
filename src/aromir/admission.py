"""Which molecules the package computes, decided before anything is computed.

The network is trained on neutral, closed-shell aromatic molecules of a few elements. A
request outside that domain is refused with every reason found, not answered badly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from rdkit import Chem
from rdkit.Chem import rdDetermineBonds
from rdkit.Chem.rdMolDescriptors import CalcMolFormula

from aromir.constants import ELEMENTS
from aromir.errors import RefusedError

if TYPE_CHECKING:
    from pathlib import Path

Mol = Any
"""An RDKit molecule with explicit hydrogens (RDKit ships no type information)."""


@dataclass(frozen=True)
class Admission:
    """An admitted molecule, as the result names it.

    Attributes:
        smiles: RDKit's canonical SMILES; every writing of the molecule gives the same one.
        formula: The molecular formula.
        n_atoms: The number of atoms, hydrogens included.
    """

    smiles: str
    formula: str
    n_atoms: int


def parse_smiles(smiles: str) -> Mol:
    """Parse a SMILES string into a molecule with explicit hydrogens.

    Raises:
        RefusedError: The string is not a valid SMILES.
    """
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise RefusedError([f"{smiles!r} does not parse as a SMILES"])
    return Chem.AddHs(molecule)


def parse_xyz(path: Path) -> Mol:
    """Read an xyz file (Å) and perceive its bonds from the geometry, as a neutral molecule.

    Raises:
        RefusedError: The file does not parse, or no neutral bonding pattern fits it.
    """
    molecule = Chem.MolFromXYZFile(str(path))
    if molecule is None:
        raise RefusedError([f"{path} does not parse as an xyz file"])
    try:
        rdDetermineBonds.DetermineBonds(molecule, charge=0)
    except ValueError as error:
        raise RefusedError([f"{path}: no neutral bonding pattern fits the geometry"]) from error
    return molecule


def _reasons_to_refuse(molecule: Mol) -> list[str]:
    atoms = list(molecule.GetAtoms())
    reasons = []
    n_fragments = len(Chem.GetMolFrags(molecule))
    if n_fragments > 1:
        reasons.append(f"{n_fragments} separate molecules: one molecule at a time")
    unsupported = sorted({atom.GetSymbol() for atom in atoms} - set(ELEMENTS))
    if unsupported:
        reasons.append(
            f"element(s) {', '.join(unsupported)} outside the supported set "
            f"({', '.join(sorted(ELEMENTS))})"
        )
    if sum(atom.GetFormalCharge() for atom in atoms) != 0:
        reasons.append("the molecule carries a net charge: only neutral molecules are supported")
    if any(atom.GetNumRadicalElectrons() for atom in atoms):
        reasons.append("the molecule has unpaired electrons: only closed shells are supported")
    if not any(atom.GetIsAromatic() for atom in atoms):
        reasons.append("no aromatic ring: the network is trained on aromatic molecules only")
    return reasons


def admit(molecule: Mol) -> Admission:
    """Check a parsed molecule against the supported domain.

    Raises:
        RefusedError: With every reason the molecule is outside the domain.
    """
    reasons = _reasons_to_refuse(molecule)
    if reasons:
        raise RefusedError(reasons)
    return Admission(
        smiles=Chem.MolToSmiles(Chem.RemoveHs(molecule)),
        formula=CalcMolFormula(molecule),
        n_atoms=molecule.GetNumAtoms(),
    )
