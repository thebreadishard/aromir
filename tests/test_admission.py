"""Admission: every reason to refuse, and what an admitted molecule is called."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds

from aromir import admission, pipeline
from aromir.constants import ANGSTROM_PER_BOHR
from aromir.errors import RefusedError

from support import BENZENE_XYZ

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

    from aromir.molecule import Geometry


def test_an_aromatic_molecule_is_admitted_under_its_canonical_name() -> None:
    admitted = admission.admit(admission.read("C1=CC=CC=C1"))
    assert admitted == admission.Admission(smiles="c1ccccc1", formula="C6H6", n_atoms=12)


def test_two_writings_get_one_name() -> None:
    first = admission.admit(admission.read("c1ccc2ccccc2c1"))
    second = admission.admit(admission.read("c1cccc2c1cccc2"))
    assert first == second


@pytest.mark.parametrize(
    ("smiles", "reason"),
    [
        ("CCO", "no aromatic ring"),
        ("c1ccccc1[O-]", "net charge"),
        ("c1ccccc1[CH2]", "unpaired electrons"),
        ("c1ccccc1Br", "element(s) Br outside"),
        ("c1ccccc1.O", "2 separate molecules"),
    ],
)
def test_a_molecule_outside_the_domain_is_refused(smiles: str, reason: str) -> None:
    with pytest.raises(RefusedError, match=reason.replace("(", r"\(").replace(")", r"\)")):
        admission.admit(admission.read(smiles))


def test_every_reason_is_reported_at_once() -> None:
    with pytest.raises(RefusedError) as refusal:
        admission.admit(admission.read("[O-]CCBr"))
    assert len(refusal.value.reasons) == 3


def test_a_string_that_is_no_smiles_is_refused() -> None:
    with pytest.raises(RefusedError, match="does not parse as a SMILES"):
        admission.parse_smiles("c1ccccc")


def test_an_xyz_file_is_read_with_its_bonds(tmp_path: Path) -> None:
    path = tmp_path / "benzene.xyz"
    path.write_text(BENZENE_XYZ, encoding="utf-8")
    assert admission.admit(admission.read(path)).smiles == "c1ccccc1"


def test_an_unreadable_xyz_file_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "broken.xyz"
    path.write_text("this is not an xyz file\n", encoding="utf-8")
    with pytest.raises(RefusedError, match="does not parse as an xyz file"):
        admission.parse_xyz(path)


def test_an_xyz_without_a_neutral_bonding_pattern_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "ammonium.xyz"
    path.write_text(
        "5\nNH4+\nN 0 0 0\nH 0.59 0.59 0.59\nH -0.59 -0.59 0.59\n"
        "H -0.59 0.59 -0.59\nH 0.59 -0.59 -0.59\n",
        encoding="utf-8",
    )
    with pytest.raises(RefusedError, match="no neutral bonding pattern"):
        admission.parse_xyz(path)


MAPPED_PAIRS = [
    ("Cc1ccc(O)cc1Cl", "[CH3:5]c1ccc(O)cc1Cl"),
    ("c1ccc2ncccc2c1", "c1ccc2[n:1]cccc2c1"),
    ("N#Cc1ccc(F)cc1", "N#[C:2]c1ccc(F)cc1"),
    ("c1ccccc1", "[cH:1]1ccccc1"),
]


@pytest.mark.parametrize(("plain", "mapped"), MAPPED_PAIRS)
def test_atom_map_numbers_are_cleared_and_said(plain: str, mapped: str) -> None:
    unmapped = admission.admit(admission.read(plain))
    cleared = admission.admit(admission.read(mapped))
    assert cleared.smiles == unmapped.smiles
    assert unmapped.input_changes == ()
    assert cleared.input_changes == (admission.ATOM_MAPS_CLEARED,)


def test_a_mapped_smiles_starts_from_the_same_geometry_in_the_same_order() -> None:
    _, plain = pipeline.admit(pipeline.MoleculeInput(smiles="Cc1ccc(O)cc1Cl"))
    _, mapped = pipeline.admit(pipeline.MoleculeInput(smiles="[CH3:5]c1ccc(O)cc1Cl"))
    assert mapped.symbols == plain.symbols
    np.testing.assert_array_equal(mapped.coords_bohr, plain.coords_bohr)


AZIDE_WRITINGS = ("[N-]=[N+]=Nc1ccccc1", "N#[N+][N-]c1ccccc1", "N#[N+]N=c1cccc[cH-]1")


@pytest.mark.parametrize("writing", AZIDE_WRITINGS)
def test_every_resonance_structure_of_an_azide_gets_one_name(writing: str) -> None:
    assert admission.admit(admission.read(writing)).smiles == "[N-]=[N+]=Nc1ccccc1"


def test_the_resonance_structure_does_not_depend_on_the_atom_order() -> None:
    # RDKit ranks the two azide structures alike and leaves the tie to the atom order.
    for writing in AZIDE_WRITINGS:
        listings = Chem.MolToRandomSmilesVect(Chem.MolFromSmiles(writing), 20, randomSeed=7)
        names = {admission.admit(admission.read(listing)).smiles for listing in listings}
        assert names == {"[N-]=[N+]=Nc1ccccc1"}


def test_a_hypervalent_sulfoxide_is_written_charge_separated() -> None:
    admitted = admission.admit(admission.read("CS(=O)c1ccccc1"))
    assert admitted.smiles == "C[S+]([O-])c1ccccc1"
    assert admitted == admission.admit(admission.read("C[S+]([O-])c1ccccc1"))


@pytest.mark.parametrize(
    "smiles",
    [
        "c1ccccc1",
        "Cc1ccccc1",
        "c1ccc2ccccc2c1",
        "O=[N+]([O-])c1ccccc1",
        "[O-][n+]1ccccc1",
        "N#Cc1ccccc1",
        "O=c1cccc[nH]1",
        "CS(=O)(=O)c1ccccc1",
        "O=C(O)c1ccccc1",
        "c1ccc2[nH]ccc2c1",
        "Nc1ccc([N+](=O)[O-])cc1",
        "[C-]#[N+]c1ccccc1",
    ],
)
def test_a_molecule_already_in_standard_form_keeps_its_name(smiles: str) -> None:
    assert admission.admit(admission.read(smiles)).smiles == smiles
    assert admission.standard_smiles(smiles) == smiles


def test_a_nitro_group_written_hypervalent_gets_the_usual_name() -> None:
    assert admission.standard_smiles("O=N(=O)c1ccccc1") == "O=[N+]([O-])c1ccccc1"


def test_standard_smiles_refuses_a_string_that_is_no_smiles() -> None:
    with pytest.raises(ValueError, match="does not parse as a SMILES"):
        admission.standard_smiles("c1ccccc")


def _listed(start: Geometry, path: Path, seed: int) -> Path:
    """Write a geometry as an xyz file, its atoms listed in a random order."""
    angstrom = start.coords_bohr * ANGSTROM_PER_BOHR
    rows = [
        f"{s} {x:.8f} {y:.8f} {z:.8f}" for s, (x, y, z) in zip(start.symbols, angstrom, strict=True)
    ]
    order = np.random.default_rng(seed).permutation(len(rows))
    text = "\n".join([str(len(rows)), "listed", *(rows[k] for k in order)]) + "\n"
    path.write_text(text, encoding="utf-8")
    return path


def _without_stereo(smiles: str) -> str:
    return str(Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=False))


@pytest.mark.parametrize(
    "smiles",
    [
        "[N-]=[N+]=Nc1ccccc1",
        "CS(=O)c1ccccc1",
        "[O-][n+]1ccccc1",
        "CS(=O)(=O)c1ccccc1",
        "O=c1cccc[nH]1",
        "Cc1ccc(O)cc1Cl",
    ],
)
def test_an_xyz_file_gets_the_name_and_the_atom_order_of_its_smiles(
    smiles: str, tmp_path: Path
) -> None:
    # Bond perception from a geometry placed an azide's or a sulfoxide's charges otherwise
    # than the SMILES. The order is checked position by position through the bond graph:
    # atoms the graph cannot tell apart (a methyl's hydrogens) keep the file's order.
    from_smiles, made = pipeline.admit(pipeline.MoleculeInput(smiles=smiles))
    for seed in range(3):
        xyz = pipeline.MoleculeInput(xyz=_listed(made, tmp_path / f"{seed}.xyz", seed))
        from_xyz, ordered = pipeline.admit(xyz)
        assert _without_stereo(from_xyz.smiles) == from_smiles.smiles
        assert ordered.symbols == made.symbols
        np.testing.assert_array_equal(_bonded(ordered), _bonded(made))


def _bonded(start: Geometry) -> NDArray[np.bool_]:
    """Return which atoms lie within bonding distance: 1.2 times their covalent radii's sum."""
    table = Chem.GetPeriodicTable()
    radii = np.array([table.GetRcovalent(symbol) for symbol in start.symbols])
    distances = np.linalg.norm(start.coords_bohr[:, None] - start.coords_bohr[None], axis=2)
    limit = 1.2 * (radii[:, None] + radii[None]) / ANGSTROM_PER_BOHR
    return np.asarray((distances > 0) & (distances < limit))


def test_an_xyz_sulfoxide_keeps_the_configuration_of_its_geometry(tmp_path: Path) -> None:
    # A sulfoxide's sulfur is a stereocentre: a SMILES without the mark leaves it open, a
    # geometry has one configuration, and the name says which.
    from_smiles, made = pipeline.admit(pipeline.MoleculeInput(smiles="CS(=O)c1ccccc1"))
    assert from_smiles.unassigned_stereo == ("1 stereocentre",)
    from_xyz, _ = pipeline.admit(pipeline.MoleculeInput(xyz=_listed(made, tmp_path / "s.xyz", 0)))
    assert from_xyz.smiles in {"C[S@+]([O-])c1ccccc1", "C[S@@+]([O-])c1ccccc1"}
    assert from_xyz.unassigned_stereo == ()


@pytest.mark.parametrize(
    ("smiles", "can"),
    [
        ("[N-]=[N+]=Nc1ccccc1", [False, False]),
        ("C=C=Cc1ccccc1", [False, False]),
        ("O=C=Cc1ccccc1", [False, False]),
        # A sulfur with two double bonds is bent: a sulfine's C=S and a sulfur diimide's
        # N=S bonds have E/Z isomers, and only the terminal S=O is none.
        ("O=S=Cc1ccccc1", [False, True]),
        ("CN=S=Nc1ccccc1", [True, True]),
        ("C(=Cc1ccccc1)c1ccccc1", [True]),
        ("CC=NC", [True]),
    ],
)
def test_only_a_double_bond_that_can_be_cis_or_trans_is_left_open(
    smiles: str, can: list[bool]
) -> None:
    # Whatever RDKit offers as potential stereo, a terminal double bond or one cumulated on
    # a carbon or a nitrogen has no E/Z configuration (RDKit before 2026.9.2 offered the
    # azide's N=N bonds). The rule is asked of each double bond directly, in bond order.
    molecule = Chem.AddHs(Chem.MolFromSmiles(smiles))
    doubles = [
        bond.GetIdx() for bond in molecule.GetBonds() if bond.GetBondType() == Chem.BondType.DOUBLE
    ]
    found = [admission._can_be_cis_or_trans(molecule, index) for index in doubles]  # noqa: SLF001  the rule itself
    assert found == can


def test_an_azide_leaves_no_double_bond_open_in_any_listing(tmp_path: Path) -> None:
    # Before the resonance step, bond perception gave some listings of phenyl azide the
    # quinoid structure N#[N+]N=c1cccc[cH-]1, whose exocyclic N=C is a potential E/Z bond,
    # and the certificate said a double bond was left unassigned. The azide has none.
    from_smiles, made = pipeline.admit(pipeline.MoleculeInput(smiles="[N-]=[N+]=Nc1ccccc1"))
    assert from_smiles.unassigned_stereo == ()
    for seed in range(8):
        from_xyz = admission.admit(admission.read(_listed(made, tmp_path / f"{seed}.xyz", seed)))
        assert (from_xyz.smiles, from_xyz.unassigned_stereo) == ("[N-]=[N+]=Nc1ccccc1", ())


SYDNONE = "[O-]c1c[n+](-c2ccccc2)no1"


def test_a_step_that_would_change_the_hydrogens_is_discarded(tmp_path: Path) -> None:
    # On a sydnone RDKit's Normalizer and a resonance structure both rebuilt the molecule
    # with two more hydrogens (O=C1CN(c2ccccc2)NO1) when hydrogens could still be implied.
    admitted = admission.admit(admission.read(SYDNONE))
    assert admitted.smiles == SYDNONE
    assert admitted.formula == "C8H6N2O2"
    _, made = pipeline.admit(pipeline.MoleculeInput(smiles=SYDNONE))
    for seed in range(4):
        path = _listed(made, tmp_path / f"{seed}.xyz", seed)
        from_xyz = admission.admit(admission.read(path))
        assert from_xyz.formula == "C8H6N2O2"
        assert admission.fixed_h_key(from_xyz.smiles) == admission.fixed_h_key(SYDNONE)


def test_a_betaine_keeps_its_aromatic_form() -> None:
    admitted = admission.admit(admission.read("[O-]c1ccc(C=Cc2cc[n+](C)cc2)cc1"))
    assert admitted.smiles == "C[n+]1ccc(C=Cc2ccc([O-])cc2)cc1"


@pytest.mark.parametrize(
    ("smiles", "named"),
    [
        ("[CH2-][n+]1ccccc1", "[CH2-][n+]1ccccc1"),
        ("[N-]=[N+]=Cc1ccccc1", "[N-]=[N+]=Cc1ccccc1"),
        ("[O-]c1ccc([N+]#N)cc1", "N#[N+]c1ccc([O-])cc1"),
    ],
)
def test_an_ylide_or_a_diazo_compound_gets_one_name_in_every_listing(
    smiles: str, named: str, tmp_path: Path
) -> None:
    # Bond perception puts the ylide's charge in the ring (C=[N+]1C=CC=C[CH-]1); only an
    # enumeration that may move a charge onto carbon finds the usual structure from there.
    _, made = pipeline.admit(pipeline.MoleculeInput(smiles=smiles))
    assert admission.admit(admission.read(smiles)).smiles == named
    for seed in range(8):
        reading = admission.read(_listed(made, tmp_path / f"{seed}.xyz", seed))
        assert admission.admit(reading).smiles == named


class _NoStructures:
    """A resonance supplier that finds nothing, as RDKit's does for some molecules."""

    def __init__(self, *_arguments: object) -> None:
        pass

    def SetProgressCallback(self, _callback: object) -> None:  # noqa: N802  RDKit's name
        pass

    def Enumerate(self) -> None:  # noqa: N802  RDKit's name
        pass

    def WasCanceled(self) -> bool:  # noqa: N802  RDKit's name
        return False

    def __len__(self) -> int:
        return 0


def test_an_enumeration_that_finds_nothing_keeps_the_structure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Chem, "ResonanceMolSupplier", _NoStructures)
    assert admission.admit(admission.read("N#[N+][N-]c1ccccc1")).smiles == "N#[N+][N-]c1ccccc1"


def test_an_enumeration_that_reaches_its_cap_keeps_the_structure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(admission, "RESONANCE_MAX_STRUCTURES", 2)
    assert admission.admit(admission.read("N#[N+][N-]c1ccccc1")).smiles == "N#[N+][N-]c1ccccc1"


def test_a_molecule_with_too_many_charged_atoms_keeps_the_structure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(admission, "RESONANCE_MAX_CHARGED_ATOMS", 1)
    assert admission.admit(admission.read("N#[N+][N-]c1ccccc1")).smiles == "N#[N+][N-]c1ccccc1"


HANG_GUARD_S = 60.0
"""Only a guard against a hang: the bounds are counts, not time."""


@pytest.mark.parametrize(
    "smiles",
    [
        (
            "O=[N+]([O-])c1c([N+](=O)[O-])c([N+](=O)[O-])c2c([N+](=O)[O-])c([N+](=O)[O-])"
            "c([N+](=O)[O-])c([N+](=O)[O-])c2c1[N+](=O)[O-]"
        ),
        "[N-]=[N+]=Nc1c(N=[N+]=[N-])c(N=[N+]=[N-])c(N=[N+]=[N-])c(N=[N+]=[N-])c1N=[N+]=[N-]",
    ],
)
def test_a_heavily_charged_molecule_gets_one_name_and_finishes(smiles: str) -> None:
    started = time.monotonic()
    names = {admission.admit(admission.read(smiles)).smiles for _ in range(2)}
    assert len(names) == 1
    assert time.monotonic() - started < HANG_GUARD_S


def _refuse_to_enumerate(*_arguments: object) -> None:
    msg = "a molecule charged only in nitro groups was enumerated"
    raise AssertionError(msg)


@pytest.mark.parametrize(
    "smiles",
    [
        "O=[N+]([O-])c1cccc2c([N+](=O)[O-])cccc12",
        "O=[N+]([O-])c1cc([N+](=O)[O-])c2ccc3c([N+](=O)[O-])cc([N+](=O)[O-])c4ccc1c2c34",
    ],
    ids=["1,5-dinitronaphthalene", "1,3,6,8-tetranitropyrene"],
)
def test_a_molecule_charged_only_in_nitro_groups_is_not_enumerated(
    smiles: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Enumerated, 1,5-dinitronaphthalene gave 208 to 295 structures over ten writings,
    # either side of the cap: the writing decided whether it was named by the enumeration.
    monkeypatch.setattr(Chem, "ResonanceMolSupplier", _refuse_to_enumerate)
    writings = Chem.MolToRandomSmilesVect(Chem.MolFromSmiles(smiles), 10, randomSeed=7)
    names = {admission.admit(admission.read(writing)).smiles for writing in writings}
    assert names == {admission.standard_smiles(smiles)}


@pytest.mark.parametrize(
    "smiles",
    [
        "O=[N+]([O-])c1ccc(O)cc1",
        "Nc1ccc([N+](=O)[O-])cc1[N+](=O)[O-]",
        "O=Cc1ccc([N+](=O)[O-])cc1",
        "O=[N+]([O-])c1ccncc1",
        "O=[N+]([O-])c1ccc(C=O)o1",
    ],
)
def test_a_nitro_compound_is_named_as_the_full_enumeration_names_it(
    smiles: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    with_rule = admission.admit(admission.read(smiles)).smiles
    monkeypatch.setattr(admission, "NITRO_GROUP", Chem.MolFromSmarts("[#0]"))
    monkeypatch.setattr(admission, "RESONANCE_MAX_STRUCTURES", 20000)
    assert admission.admit(admission.read(smiles)).smiles == with_rule


SYDNONE_WRITINGS = (SYDNONE, "[O-]c1cn(-c2ccccc2)n[o+]1")


def test_two_tied_writings_of_a_sydnone_get_one_name(tmp_path: Path) -> None:
    # The two structures rank alike and are not the same writing: the smaller canonical
    # SMILES names the molecule, from any writing and any listing of an xyz file.
    names = set()
    for writing in SYDNONE_WRITINGS:
        for listing in Chem.MolToRandomSmilesVect(Chem.MolFromSmiles(writing), 10, randomSeed=3):
            names.add(admission.admit(admission.read(listing)).smiles)
    _, made = pipeline.admit(pipeline.MoleculeInput(smiles=SYDNONE_WRITINGS[1]))
    for seed in range(8):
        names.add(
            admission.admit(admission.read(_listed(made, tmp_path / f"{seed}.xyz", seed))).smiles
        )
    assert names == {SYDNONE}


@pytest.mark.parametrize(
    "smiles",
    [
        "O=[N+]([O-])c1ccccc1",
        "O=[N+]([O-])c1cccc([N+](=O)[O-])c1",
        "[O-][n+]1ccccc1",
        "C[S+]([O-])c1ccccc1",
        "[N-]=[N+]=Nc1ccccc1",
    ],
)
def test_a_structure_already_preferred_keeps_its_charges_on_its_atoms(smiles: str) -> None:
    # A nitro group's two structures tie: the one given is kept, so the charge stays on
    # the oxygen the canonical SMILES puts it on.
    given = admission.parse_smiles(smiles)
    read = admission.read(smiles).molecule
    assert [a.GetFormalCharge() for a in read.GetAtoms()] == [
        a.GetFormalCharge() for a in given.GetAtoms()
    ]


LISTING_FREE = [
    "O=[N+]([O-])c1ccccc1",
    "O=[N+]([O-])c1cccc([N+](=O)[O-])c1",
    "O=[N+]([O-])c1ccc2ccccc2c1",
    "Nc1ccc([N+](=O)[O-])cc1",
    "[N-]=[N+]=Nc1ccccc1",
    "[O-][n+]1ccccc1",
    "N#Cc1ccccc1",
]
"""Nitro compounds RDKit's DetermineBonds refused in some listings, and three that it did not."""


@pytest.mark.parametrize("smiles", LISTING_FREE)
def test_every_listing_of_an_xyz_file_is_admitted_under_one_name(
    smiles: str, tmp_path: Path
) -> None:
    named, made = pipeline.admit(pipeline.MoleculeInput(smiles=smiles))
    names = set()
    for seed in range(20):
        path = _listed(made, tmp_path / f"{seed}.xyz", seed)
        names.add(admission.admit(admission.read(path)).smiles)
    assert names == {named.smiles}


def test_the_listings_of_nitrobenzene_include_some_the_plain_call_refuses(tmp_path: Path) -> None:
    # What the test above holds: bond orders sought in the file's own order fail for
    # some listings of nitrobenzene (7 of these 20 in RDKit 2026.09).
    _, made = pipeline.admit(pipeline.MoleculeInput(smiles="O=[N+]([O-])c1ccccc1"))
    refused = 0
    for seed in range(20):
        molecule = Chem.MolFromXYZBlock(
            _listed(made, tmp_path / "n.xyz", seed).read_text(encoding="utf-8")
        )
        try:
            rdDetermineBonds.DetermineBonds(molecule, charge=0)
        except ValueError:
            refused += 1
    assert refused > 0


def test_a_four_membered_rings_diagonal_is_no_bond_of_an_xyz_file(tmp_path: Path) -> None:
    # The force-field geometry puts the diagonal at 1.970 Å, within the 1.97 Å of two
    # carbons: read as a bond, it gave COC1=C2C34C=C[CH+]C=C3C24C=C[CH-]1.
    named, made = pipeline.admit(pipeline.MoleculeInput(smiles="COc1cccc2c1-c1ccccc1-2"))
    read = admission.admit(admission.read(_listed(made, tmp_path / "b.xyz", 0)))
    assert read.smiles == named.smiles
