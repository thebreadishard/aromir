"""Which molecules the package computes, decided before anything is computed.

The network is trained on neutral, closed-shell aromatic molecules of a few elements. A
request outside that domain is refused with every reason found, not answered badly.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any, Final

from rdkit import Chem
from rdkit.Chem import rdDetermineBonds
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.rdMolDescriptors import CalcMolFormula

from aromir.constants import ELEMENTS
from aromir.errors import RefusedError

Mol = Any
"""An RDKit molecule with explicit hydrogens (RDKit ships no type information)."""

ATOM_MAPS_CLEARED: Final = (
    "the input's atom-map numbers were cleared: they label atoms for reaction software and "
    "change neither the molecule nor its spectrum"
)


@dataclass(frozen=True)
class Admission:
    """An admitted molecule, as the result names it.

    Attributes:
        smiles: RDKit's canonical SMILES of the molecule's standard form; every writing of
            the molecule, as a SMILES or as an xyz file, gives the same one.
        formula: The molecular formula.
        n_atoms: The number of atoms, hydrogens included.
        unassigned_stereo: What the input leaves open, such as "1 double bond": the
            computed geometry is then one isomer of several the SMILES stands for.
        input_changes: What reading the input changed without changing the molecule, such
            as cleared atom-map numbers.
    """

    smiles: str
    formula: str
    n_atoms: int
    unassigned_stereo: tuple[str, ...] = ()
    input_changes: tuple[str, ...] = ()


@dataclass(frozen=True)
class Reading:
    """A request read into the molecule the package works on.

    Attributes:
        molecule: The molecule in its standard form (``standardised``), with explicit
            hydrogens; read from an xyz file, it keeps the file's atoms and coordinates.
        input_changes: What reading changed in the input without changing the molecule.
    """

    molecule: Mol
    input_changes: tuple[str, ...] = ()


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


_NORMALIZER: Final = rdMolStandardize.Normalizer()
"""RDKit's Normalizer with its default (MolVS) transforms, built once."""


def _preference(form: Mol) -> tuple[int, int, int, str]:
    """Rank a resonance structure: the smallest key is the one the package works on.

    Fewest charged atoms first, then no negative charge on carbon, then negative charges
    on terminal atoms (the usual writing of an azide, a nitro group, an N-oxide); the
    canonical SMILES breaks a remaining tie, so the choice depends on the molecule alone,
    not on the order or the structure it was given in.
    """
    anions = [atom for atom in form.GetAtoms() if atom.GetFormalCharge() < 0]
    return (
        sum(atom.GetFormalCharge() != 0 for atom in form.GetAtoms()),
        sum(atom.GetAtomicNum() == 6 for atom in anions),
        sum(neighbour.GetAtomicNum() > 1 for atom in anions for neighbour in atom.GetNeighbors()),
        Chem.MolToSmiles(without_hydrogens(form)),
    )


def _one_resonance_structure(molecule: Mol) -> None:
    """Give a charge-separated molecule, in place, its preferred resonance structure.

    Bond perception from a geometry places the charges of an azide or a sulfoxide where it
    happens to (``N#[N+][N-]R`` or ``N#[N+]N=c1cccc[cH-]1`` for ``[N-]=[N+]=NR``), and
    RDKit's own ranking of resonance structures leaves such ties to the atom order. Every
    structure ``Chem.ResonanceMolSupplier`` enumerates is ranked by ``_preference``, and the
    first one's bonds and charges are set on the molecule's own atoms, so an xyz file's
    coordinates still belong to them. The supplier is called as in RDKit's
    ``Code/GraphMol/Wrap/rough_test.py``; the stereo is assigned again, from the
    coordinates where there are any, as in the same file's ``AssignStereochemistryFrom3D``
    tests.
    """
    forms = []
    for form in Chem.ResonanceMolSupplier(molecule):
        Chem.SanitizeMol(form)
        forms.append(form)
    chosen = min(forms, key=_preference)
    Chem.Kekulize(chosen, clearAromaticFlags=True)
    Chem.Kekulize(molecule, clearAromaticFlags=True)
    for atom in molecule.GetAtoms():
        atom.SetFormalCharge(chosen.GetAtomWithIdx(atom.GetIdx()).GetFormalCharge())
    for bond in molecule.GetBonds():
        bond.SetBondType(chosen.GetBondWithIdx(bond.GetIdx()).GetBondType())
    Chem.SanitizeMol(molecule)
    if molecule.GetNumConformers():
        Chem.AssignStereochemistryFrom3D(molecule)
    else:
        Chem.AssignStereochemistry(molecule, cleanIt=True, force=True)


def standardised(molecule: Mol) -> Mol:
    """Return a molecule in the package's standard form: one Lewis structure, no atom maps.

    The network reads the atoms in the order of the canonical SMILES, and the 'seen'
    look-up compares canonical SMILES, so one molecule must have one canonical SMILES
    however it was written. Three steps get there; atom indices and coordinates are kept.

    1. Atom-map numbers are cleared: RDKit's canonical ranking reads them, so a mapped
       SMILES would be numbered and named as another molecule.
    2. RDKit's Normalizer (the MolVS transforms) writes a hypervalent group the way bond
       perception from a geometry does: a sulfoxide ``S=O`` as ``[S+][O-]``, a nitro group
       ``N(=O)=O`` as ``[N+](=O)[O-]``. Called as in RDKit's
       ``Code/GraphMol/MolStandardize/Wrap/testMolStandardize.py``.
    3. A molecule that still carries formal charges gets its preferred resonance structure
       (``_one_resonance_structure``). A molecule without charges is left as it is: its
       Kekulé structures are one aromatic structure to RDKit already.
    """
    standard = Chem.Mol(molecule)
    for atom in standard.GetAtoms():
        atom.SetAtomMapNum(0)
    _NORMALIZER.normalizeInPlace(standard)
    if any(atom.GetFormalCharge() for atom in standard.GetAtoms()):
        _one_resonance_structure(standard)
    return standard


def read(source: str | Path) -> Reading:
    """Read a request, a SMILES string or an xyz file, into the molecule the package works on.

    Both routes end in the same standard form (``standardised``), so a molecule gets one
    canonical SMILES and one atom order whether it was given as a SMILES or as an xyz file.
    A mapped SMILES is read with its map numbers cleared and the clearing said, not
    refused: the numbers label atoms for reaction software and are no part of the molecule.

    Raises:
        RefusedError: The SMILES or the xyz file does not parse.
    """
    molecule = parse_xyz(source) if isinstance(source, Path) else parse_smiles(source)
    mapped = any(atom.GetAtomMapNum() for atom in molecule.GetAtoms())
    return Reading(standardised(molecule), (ATOM_MAPS_CLEARED,) if mapped else ())


@cache
def standard_smiles(smiles: str) -> str:
    """Return the canonical SMILES of a SMILES's standard form, as admission writes it.

    Raises:
        ValueError: The string does not parse as a SMILES.
    """
    molecule: Mol | None = Chem.MolFromSmiles(smiles)  # RDKit's stubs omit the None
    if molecule is None:
        msg = f"{smiles!r} does not parse as a SMILES"
        raise ValueError(msg)
    return str(Chem.MolToSmiles(without_hydrogens(standardised(Chem.AddHs(molecule)))))


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


def _as_computed(molecule: Mol, smiles: str) -> Mol:
    """Return the molecule whose stereo the computed geometry has.

    An xyz file's geometry is computed as it is. A SMILES is embedded from the admitted
    SMILES, which drops a stereo mark that rests on a hydrogen (the N–H of an imine):
    what that SMILES leaves open is open in the computation.
    """
    if molecule.GetNumConformers():
        return molecule
    return Chem.AddHs(Chem.MolFromSmiles(smiles))


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


def admit(reading: Reading) -> Admission:
    """Check a read molecule against the supported domain.

    Raises:
        RefusedError: With every reason the molecule is outside the domain.
    """
    molecule = reading.molecule
    reasons = _reasons_to_refuse(molecule)
    if reasons:
        raise RefusedError(reasons)
    smiles = Chem.MolToSmiles(without_hydrogens(molecule))
    return Admission(
        smiles=smiles,
        formula=CalcMolFormula(molecule),
        n_atoms=molecule.GetNumAtoms(),
        unassigned_stereo=_unassigned_stereo(_as_computed(molecule, smiles)),
        input_changes=reading.input_changes,
    )
