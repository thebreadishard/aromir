"""Which molecules the package computes, decided before anything is computed.

The network is trained on neutral, closed-shell aromatic molecules of a few elements. A
request outside that domain is refused with every reason found, not answered badly.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any, Final

from rdkit import Chem, rdBase
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

RESONANCE_MAX_STRUCTURES: Final = 256
"""The most resonance structures enumerated; a molecule with more keeps its own structure.
How many RDKit finds depends on the structure and atom order it starts from: 30 to 82 for
the writings and xyz listings of 3-phenylsydnone, so a cap of 64 left some unnamed."""

RESONANCE_FLAGS: Final = Chem.UNCONSTRAINED_ANIONS | Chem.UNCONSTRAINED_CATIONS
"""Let the enumeration move a charge onto any atom. By default RDKit keeps a negative charge
on the most electronegative atoms, so from the structure bond perception gives a pyridinium
methylide (``C=[N+]1C=CC=C[CH-]1``) or a diazo compound it finds no other, and from
``[N-]=[N+]=Cc1ccccc1`` none at all; ``_preference`` still puts carbanions last."""

RESONANCE_MAX_CHARGED_ATOMS: Final = 8
"""The most charged atoms a molecule may have to be enumerated; one with more keeps its own
structure. Twice the most in any entry of the shipped training lists (4, a dinitro
compound); octanitronaphthalene (16) took 6 s to reach the structure cap."""

NITRO_GROUP: Final = Chem.MolFromSmarts("[N+X3](=O)[O-]")
"""A nitro group as the Normalizer writes it. A molecule charged only in such groups is not
enumerated: no structure has fewer charged atoms (the nitrogen needs its positive charge for
its four bonds), one that moves a charge into the ring ranks worse (``_preference``), and the
group's two structures share one canonical SMILES, so the molecule's own structure is the
one chosen. Enumerated, nitro groups conjugated with a ring multiply the structures with
the writing deciding how many RDKit finds: 208 to 295 for 1,5-dinitronaphthalene over ten
writings, 1846 to 3041 for 1,8-dinitropyrene, 5924 to 9897 for 1,3,6-trinitropyrene."""

Composition = tuple[str, tuple[int, ...], tuple[int, ...]]
"""The formula and each atom's hydrogens and unpaired electrons: what no step may change."""


def _composition(molecule: Mol) -> Composition:
    atoms = list(molecule.GetAtoms())
    return (
        CalcMolFormula(molecule),
        tuple(atom.GetTotalNumHs(includeNeighbors=True) for atom in atoms),
        tuple(atom.GetNumRadicalElectrons() for atom in atoms),
    )


def _structure(molecule: Mol) -> tuple[tuple[int, ...], tuple[Any, ...]]:
    """Return each atom's formal charge and each bond's type: the Lewis structure."""
    return (
        tuple(atom.GetFormalCharge() for atom in molecule.GetAtoms()),
        tuple(bond.GetBondType() for bond in molecule.GetBonds()),
    )


def _n_aromatic(molecule: Mol) -> int:
    return sum(atom.GetIsAromatic() for atom in molecule.GetAtoms())


def _preference(form: Mol) -> tuple[int, int, int, int]:
    """Rank a resonance structure: the smallest key is preferred.

    Fewest charged atoms first, then no negative and then no positive charge on carbon
    (a pyridinium ylide is ``[CH2-][n+]1ccccc1``, not ``[CH2-]n1cccc[cH+]1``), then negative
    charges on terminal atoms (the usual writing of an azide, a nitro group, an N-oxide).
    """
    charged = [atom for atom in form.GetAtoms() if atom.GetFormalCharge()]
    anions = [atom for atom in charged if atom.GetFormalCharge() < 0]
    return (
        len(charged),
        sum(atom.GetAtomicNum() == 6 for atom in anions),
        sum(atom.GetAtomicNum() == 6 and atom.GetFormalCharge() > 0 for atom in charged),
        sum(neighbour.GetAtomicNum() > 1 for atom in anions for neighbour in atom.GetNeighbors()),
    )


def _charge_placement(form: Mol) -> tuple[tuple[int, int], ...]:
    """Return (atom index, formal charge) of the charged atoms: the last tie-break."""
    return tuple(
        (atom.GetIdx(), atom.GetFormalCharge())
        for atom in form.GetAtoms()
        if atom.GetFormalCharge()
    )


def _canonical(form: Mol) -> str:
    """Return a structure's canonical SMILES without stereo: what tells structures apart."""
    return str(Chem.MolToSmiles(without_hydrogens(form), isomericSmiles=False))


def _resonance_candidates(molecule: Mol) -> list[Mol]:
    """Return the resonance structures that may replace a molecule's own; none to keep it.

    A structure qualifies when it sanitises, keeps the formula and every atom's hydrogens
    and unpaired electrons, and keeps at least as many aromatic atoms as the molecule
    (a betaine is not swapped for its quinoid form). A molecule with more than
    ``RESONANCE_MAX_CHARGED_ATOMS`` charged atoms is not enumerated, nor one charged only in
    nitro groups (``NITRO_GROUP``), and an enumeration
    that finds nothing or reaches ``RESONANCE_MAX_STRUCTURES`` (the choice would depend on
    which ones it found) keeps the molecule as it is: both bounds are counts, so the
    result does not depend on the machine. The supplier is built as in RDKit's
    ``Code/GraphMol/Wrap/rough_test.py`` (its resonance supplier tests).
    """
    charged = {atom.GetIdx() for atom in molecule.GetAtoms() if atom.GetFormalCharge()}
    if len(charged) > RESONANCE_MAX_CHARGED_ATOMS:
        return []
    in_nitro = {index for match in molecule.GetSubstructMatches(NITRO_GROUP) for index in match}
    if charged <= in_nitro:
        return []
    supplier = Chem.ResonanceMolSupplier(molecule, RESONANCE_FLAGS, RESONANCE_MAX_STRUCTURES)
    supplier.Enumerate()
    if not 0 < len(supplier) < RESONANCE_MAX_STRUCTURES:
        return []
    composition, n_aromatic = _composition(molecule), _n_aromatic(molecule)
    candidates = []
    for form in supplier:
        if form is None or Chem.SanitizeMol(form, catchErrors=True) != Chem.SANITIZE_NONE:
            continue
        if _composition(form) == composition and _n_aromatic(form) >= n_aromatic:
            candidates.append(form)
    return candidates


def _set_structure(molecule: Mol, form: Mol) -> None:
    """Give a molecule, in place, the bonds and charges of one of its resonance structures.

    The molecule's own atoms are kept, so an xyz file's coordinates still belong to them;
    the stereo is assigned again, from the coordinates where there are any, as in RDKit's
    ``Code/GraphMol/Wrap/rough_test.py`` (``AssignStereochemistryFrom3D``).
    """
    chosen = Chem.Mol(form)
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


def _one_resonance_structure(molecule: Mol) -> None:
    """Give a charge-separated molecule, in place, its preferred resonance structure.

    Bond perception from a geometry places the charges of an azide where it happens to
    (``N#[N+][N-]R`` or ``N#[N+]N=c1cccc[cH-]1`` for ``[N-]=[N+]=NR``). Among the
    candidates (``_resonance_candidates``) and the molecule itself, those with the smallest
    ``_preference`` tie; of them the one with the smallest canonical SMILES is taken, so
    two writings of a sydnone get one name. The molecule keeps its own bonds and charges
    when it is one of those (a nitro group keeps its charge on the oxygen it was given);
    otherwise a remaining tie is broken by ``_charge_placement``.
    """
    candidates = _resonance_candidates(molecule)
    if not candidates:
        return
    own = _preference(molecule)
    best = min(*map(_preference, candidates), own)
    tied = [form for form in candidates if _preference(form) == best]
    chosen = min(
        [_canonical(form) for form in tied] + ([_canonical(molecule)] if own == best else [])
    )
    if own == best and _canonical(molecule) == chosen:
        return
    named = (form for form in tied if _canonical(form) == chosen)
    _set_structure(molecule, min(named, key=_charge_placement))


def standardised(molecule: Mol) -> Mol:
    """Return a molecule in the package's standard form: one Lewis structure, no atom maps.

    The network reads the atoms in the order of the canonical SMILES, and the 'seen'
    look-up compares canonical SMILES, so one molecule should have one canonical SMILES
    however it was written. Atom indices and coordinates are kept throughout.

    1. Atom-map numbers are cleared: RDKit's canonical ranking reads them, so a mapped
       SMILES would be numbered and named as another molecule.
    2. Every atom's hydrogen count is fixed (``SetNoImplicit``) while the bonds change, so
       a step that lowers a bond order cannot add a hydrogen.
    3. RDKit's Normalizer (the MolVS transforms) writes a hypervalent group the way bond
       perception from a geometry does: a sulfoxide ``S=O`` as ``[S+][O-]``, a nitro group
       ``N(=O)=O`` as ``[N+](=O)[O-]``. Called as in RDKit's
       ``Code/GraphMol/MolStandardize/Wrap/testMolStandardize.py``.
    4. A molecule that still carries formal charges gets its preferred resonance structure
       (``_one_resonance_structure``). A molecule without charges is left as it is: its
       Kekulé structures are one aromatic structure to RDKit already.

    A step whose result has another formula, or another hydrogen count or unpaired
    electron on any atom, is discarded: the molecule keeps its given structure.
    """
    given = Chem.Mol(molecule)
    for atom in given.GetAtoms():
        atom.SetAtomMapNum(0)
    composition = _composition(given)
    standard = Chem.Mol(given)
    for atom in standard.GetAtoms():
        atom.SetNoImplicit(True)  # noqa: FBT003  RDKit's positional flag
    fixed = Chem.Mol(standard)
    _NORMALIZER.normalizeInPlace(standard)
    # normalizeInPlace may leave a molecule that does not sanitise (RDKit's own
    # MolStandardize/Pipeline.cpp checks for it), as on a sydnone read from an xyz file.
    if (
        Chem.SanitizeMol(standard, catchErrors=True) != Chem.SANITIZE_NONE
        or _composition(standard) != composition
    ):
        standard = fixed
    if any(atom.GetFormalCharge() for atom in standard.GetAtoms()):
        _one_resonance_structure(standard)
    if _structure(standard) == _structure(given):
        return given  # untouched: the Normalizer may still have rewritten its stereo marks
    for atom in standard.GetAtoms():
        atom.SetNoImplicit(given.GetAtomWithIdx(atom.GetIdx()).GetNoImplicit())
    standard.UpdatePropertyCache(strict=False)
    return standard if _composition(standard) == composition else given


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


FIXED_H_OPTIONS: Final = "/FixedH"
"""InChI's fixed-hydrogen layer: the key keeps tautomers apart (2-pyridone is not
2-hydroxypyridine) while staying the same for every resonance structure."""


@cache
def fixed_h_key(smiles: str) -> str | None:
    """Return a SMILES's InChIKey with the fixed-hydrogen layer; None where InChI makes none.

    The second key of the 'seen' look-up: a molecule whose standard form still depends on
    how it was given (one past the resonance step's bounds) is found by it. RDKit
    hands ``options`` to InChI as its command-line options; ``/FixedH`` is used so in RDKit's
    ``rdkit/Chem/UnitTestInchi.py`` and, for the key, ``Code/PgSQL/rdkit/sql/inchi.sql``.
    """
    molecule: Mol | None = Chem.MolFromSmiles(smiles)  # RDKit's stubs omit the None
    if molecule is None:
        return None
    with rdBase.BlockLogs():  # InChI's warnings (undefined stereo) are no message to the user
        key = str(Chem.MolToInchiKey(molecule, options=FIXED_H_OPTIONS))  # type: ignore[no-untyped-call]
    return key or None


_LINEAR_WHEN_CUMULATED: Final = ("C", "N")
"""The elements whose atom with two double bonds is linear, so neither bond has an E/Z."""


def _can_be_cis_or_trans(molecule: Mol, bond_index: int) -> bool:
    """Tell whether a double bond can have an E/Z configuration at all.

    It cannot when one of its atoms is terminal (=O, =N⁻: nothing on it to lie on one side)
    or is a carbon or a nitrogen that carries a second double bond (the middle N of an
    azide, an allene's or a ketene's carbon: such an atom is linear). A sulfur with two
    double bonds is bent — a sulfine R–CH=S=O and a sulfur diimide have E/Z isomers — and
    excludes nothing. RDKit before 2026.9.2 offers the azide's N=N bonds as potential
    stereo bonds; ``Chem.FindPotentialStereo`` is documented as returning *possible* stereo
    elements, so its candidates are filtered here by what a configuration needs.
    """
    bond = molecule.GetBondWithIdx(bond_index)
    for atom in (bond.GetBeginAtom(), bond.GetEndAtom()):
        doubles = sum(other.GetBondType() == Chem.BondType.DOUBLE for other in atom.GetBonds())
        linear = doubles > 1 and atom.GetSymbol() in _LINEAR_WHEN_CUMULATED
        if atom.GetDegree() < 2 or linear:
            return False
    return True


def _unassigned_stereo(molecule: Mol) -> tuple[str, ...]:
    """Name the double bonds and stereocentres an input does not assign.

    ``Chem.FindPotentialStereo`` gives each candidate's ``type``, ``specified`` and
    ``centeredOn`` (the bond's index for a double bond): RDKit's own
    ``rdkit/Chem/EnumerateStereoisomers.py`` reads a ``Bond_Double`` candidate's bond with
    ``mol.GetBondWithIdx(si.centeredOn)``.
    """
    open_elements = [
        element.type
        for element in Chem.FindPotentialStereo(molecule)
        if element.specified == Chem.StereoSpecified.Unspecified
        and (
            element.type != Chem.StereoType.Bond_Double
            or _can_be_cis_or_trans(molecule, element.centeredOn)
        )
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
