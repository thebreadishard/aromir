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
        unassigned_stereo: What the input leaves open, such as "1 double bond": the
            computed geometry is then one isomer of several the SMILES stands for.
    """

    smiles: str
    formula: str
    n_atoms: int
    unassigned_stereo: tuple[str, ...] = ()


def parse_smiles(smiles: str) -> Mol:
    """Parse a SMILES string into a molecule with explicit hydrogens.

    Raises:
        RefusedError: The string is not one valid SMILES and nothing else.
    """
    # RDKit reads a SMILES up to the first blank and takes the rest for a name, so
    # "c1ccccc1 benzene" would be answered as benzene without a word.
    if not smiles or not smiles.isascii() or any(character.isspace() for character in smiles):
        reason = (
            f"{smiles!r} is not a SMILES alone: it is empty, or holds a blank, a name after "
            "the SMILES or a character outside ASCII"
        )
        raise RefusedError([reason])
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise RefusedError([f"{smiles!r} does not parse as a SMILES"])
    return Chem.AddHs(molecule)


def parse_xyz(path: Path) -> Mol:
    """Read an xyz file (Å) and perceive its bonds from the geometry, as a neutral molecule.

    Raises:
        RefusedError: The file does not parse, or no neutral bonding pattern fits it.
    """
    try:
        # The comment line is free text and often not UTF-8 (a name, an Å): nothing of it
        # is used, so a byte that does not decode is replaced, not refused.
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as error:
        raise RefusedError([f"{path} cannot be read: {error}"]) from error
    molecule = Chem.MolFromXYZBlock(text)
    if molecule is None:
        raise RefusedError([f"{path} does not parse as an xyz file"])
    try:
        rdDetermineBonds.DetermineBonds(molecule, charge=0)
    except ValueError as error:
        raise RefusedError([f"{path}: no neutral bonding pattern fits the geometry"]) from error
    return molecule


def without_hydrogens(molecule: Mol) -> Mol:
    """Return a molecule's heavy-atom skeleton.

    RDKit keeps a hydrogen that defines a double bond's stereo, as on the N–H of an imine
    or an amidine read from an xyz file; the skeleton has none.
    """
    parameters: Any = Chem.RemoveHsParameters()  # RDKit's stubs mistype its attributes
    parameters.removeDefiningBondStereo = True
    return Chem.RemoveHs(molecule, parameters)


def _unassigned_stereo(molecule: Mol) -> tuple[str, ...]:
    """Name the double bonds and stereocentres an input does not assign."""
    open_elements = [
        element.type
        for element in Chem.FindPotentialStereo(molecule)
        if element.specified == Chem.StereoSpecified.Unspecified
    ]
    counts = {
        "double bond": sum(kind == Chem.StereoType.Bond_Double for kind in open_elements),
        "stereocentre": sum(kind == Chem.StereoType.Atom_Tetrahedral for kind in open_elements),
    }
    return tuple(f"{n} {name}{'s' if n > 1 else ''}" for name, n in counts.items() if n)


def _reasons_to_refuse(molecule: Mol) -> list[str]:
    atoms = list(molecule.GetAtoms())
    if not atoms:
        return ["the input holds no atom"]
    reasons = []
    labelled = sorted(
        {f"{atom.GetIsotope()}{atom.GetSymbol()}" for atom in atoms if atom.GetIsotope()}
    )
    if labelled:
        reasons.append(
            f"isotope label(s) {', '.join(labelled)}: every atom is computed with the mass of "
            "its most abundant isotope, and no label is read"
        )
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
        smiles=Chem.MolToSmiles(without_hydrogens(molecule)),
        formula=CalcMolFormula(molecule),
        n_atoms=molecule.GetNumAtoms(),
        unassigned_stereo=_unassigned_stereo(molecule),
    )
