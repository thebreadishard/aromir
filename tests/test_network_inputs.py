"""Primitives, the pair pattern and the tensors built from them."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from aromir import geometry
from aromir.molecule import Geometry
from aromir.network import inputs, pattern, primitives
from aromir.network.primitives import PrimitiveSettings

from support import random_hessian

RAW = PrimitiveSettings()
ORDER_FREE = PrimitiveSettings(angle_choice="canonical", dihedral_encoding="cos")


def test_benzenes_primitives() -> None:
    built = primitives.build(geometry.start_from_smiles("c1ccccc1"), RAW)
    kinds = Counter(primitives.PRIMITIVE_KINDS[primitives.kind(p)] for p in built.internals)
    assert kinds == {"Distance": 12, "Angle": 12, "OutOfPlane": 6, "Dihedral": 24}
    assert built.wilson_b.shape == (54, 36)
    assert built.order.tolist() == list(range(12))


def test_a_nitrile_has_linear_angles() -> None:
    built = primitives.build(geometry.start_from_smiles("N#Cc1ccccc1"), RAW)
    kinds = [primitives.kind(p) for p in built.internals]
    assert kinds.count(primitives.LINEAR_ANGLE) == 2


def test_wilson_b_is_the_derivative_of_the_values() -> None:
    start = geometry.start_from_smiles("Oc1ccccc1")
    built = primitives.build(start, RAW)
    step = np.random.default_rng(0).normal(scale=1e-5, size=start.coords_bohr.shape)
    moved = primitives.build(Geometry(start.symbols, start.coords_bohr + step), RAW)
    bonds = [k for k, p in enumerate(built.internals) if primitives.kind(p) == primitives.BOND]
    change = np.array([moved.value(k) - built.value(k) for k in bonds])
    np.testing.assert_allclose(change, (built.wilson_b @ step.ravel())[bonds], atol=1e-9)


def test_the_canonical_choice_maps_b_back_to_the_input_order() -> None:
    start = geometry.start_from_smiles("Oc1ccc(Cl)cc1")
    permutation = np.random.default_rng(1).permutation(start.n_atoms)
    shuffled = start.reordered(permutation.tolist())
    columns = (3 * permutation[:, None] + np.arange(3)).ravel()
    here = primitives.build(start, ORDER_FREE)
    there = primitives.build(shuffled, ORDER_FREE)
    np.testing.assert_allclose(there.wilson_b, here.wilson_b[:, columns], atol=1e-12)
    np.testing.assert_array_equal(there.built_on.coords_bohr, here.built_on.coords_bohr)


def test_internal_force_constants_reproduce_the_hessian_in_the_span_of_b() -> None:
    start = geometry.start_from_smiles("c1ccccc1")
    built = primitives.build(start, RAW)
    internal = np.random.default_rng(2).normal(size=(54, 54))
    hessian = built.wilson_b.T @ (internal + internal.T) @ built.wilson_b
    recovered = primitives.internal_force_constants(hessian, built.wilson_b)
    np.testing.assert_allclose(built.wilson_b.T @ recovered @ built.wilson_b, hessian, atol=1e-9)


def test_parities_of_a_planar_and_a_twisted_molecule() -> None:
    benzene = primitives.build(geometry.start_from_smiles("c1ccccc1"), RAW)
    odd, sines = primitives.parities(benzene)
    kinds = np.array([primitives.kind(p) for p in benzene.internals])
    assert odd.sum() == 30
    assert set(kinds[odd]) == {primitives.DIHEDRAL, primitives.OUT_OF_PLANE}
    np.testing.assert_allclose(sines, 0.0, atol=1e-3)
    biphenyl = primitives.build(geometry.start_from_smiles("c1ccc(cc1)-c1ccccc1"), RAW)
    _, twisted_sines = primitives.parities(biphenyl)
    assert np.abs(twisted_sines).max() > 0.3


def test_a_primitive_that_is_neither_even_nor_odd_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    benzene = primitives.build(geometry.start_from_smiles("c1ccccc1"), RAW)
    monkeypatch.setattr(primitives, "_PARITY_TOLERANCE", -1.0)
    with pytest.raises(ValueError, match="neither even nor odd"):
        primitives.parities(benzene)


def test_an_unfamiliar_primitive_is_of_the_other_kind() -> None:
    assert primitives.kind(object()) == primitives.OTHER


def test_a_wider_pattern_holds_every_pair_of_a_narrower_one() -> None:
    built = primitives.build(geometry.start_from_smiles("Cc1ccccc1"), RAW)
    force = np.eye(len(built.internals))
    held = [
        {tuple(pair) for pair in pattern.build(built, force, name, RAW).pairs.tolist()}
        for name in ("c", "d", "e", "f")
    ]
    assert [len(pairs) for pairs in held] == [1401, 2030, 2325, 2409]
    assert held[0] < held[1] < held[2] < held[3]


def test_the_pattern_names_each_kind_of_pair() -> None:
    built = primitives.build(geometry.start_from_smiles("c1ccccc1"), RAW)
    force = np.arange(54.0 * 54).reshape(54, 54)
    pairs = pattern.build(built, force, "f", RAW)
    classes = Counter(pattern.PAIR_CLASSES[c] for c in pairs.classes.tolist())
    assert classes["diag_bond"] == 12
    assert classes["diag_angle"] == 12
    assert classes["diag_dihedral"] == 24
    assert classes["diag_other"] == 6
    assert set(classes) == set(pattern.PAIR_CLASSES)
    assert (pairs.pairs[:, 0] <= pairs.pairs[:, 1]).all()
    assert pairs.features.shape == (len(pairs.pairs), 66)
    diagonal = pairs.pairs[:, 0] == pairs.pairs[:, 1]
    np.testing.assert_array_equal(pairs.features[:, -1], diagonal.astype(np.float32))
    np.testing.assert_array_equal(
        pairs.features[:, -3], force[pairs.pairs[:, 0], pairs.pairs[:, 1]].astype(np.float32)
    )


@pytest.mark.parametrize(("encoding", "width"), [("raw", 66), ("cos", 66), ("cossin", 68)])
def test_the_feature_width_follows_the_encoding(encoding: str, width: int) -> None:
    settings = PrimitiveSettings.model_validate({"dihedral_encoding": encoding})
    assert pattern.n_pair_features(settings) == width
    built = primitives.build(geometry.start_from_smiles("c1ccccc1"), settings)
    assert pattern.build(built, np.eye(54), "c", settings).features.shape[1] == width


def test_a_dihedral_enters_as_its_cosine_under_the_cos_encoding() -> None:
    start = geometry.start_from_smiles("c1ccc(cc1)-c1ccccc1")
    built = primitives.build(start, RAW)
    force = np.eye(len(built.internals))
    raw = pattern.build(built, force, "c", RAW)
    cos = pattern.build(built, force, "c", PrimitiveSettings(dihedral_encoding="cos"))
    value_slot = 14
    diagonal = raw.pairs[:, 0] == raw.pairs[:, 1]
    dihedral = diagonal & (raw.classes == pattern.PAIR_CLASSES.index("diag_dihedral"))
    # A diagonal pair's sum slot holds twice the primitive's value.
    np.testing.assert_allclose(
        cos.features[dihedral, value_slot],
        2 * np.cos(raw.features[dihedral, value_slot] / 2),
        atol=1e-5,
    )
    np.testing.assert_array_equal(
        cos.features[~dihedral & diagonal], raw.features[~dihedral & diagonal]
    )


def test_an_unknown_setting_is_refused() -> None:
    with pytest.raises(ValueError, match="angle_choice"):
        PrimitiveSettings.model_validate({"angle_choice": "alphabetical"})
    with pytest.raises(ValueError, match="pattern"):
        inputs.InputSettings.model_validate({"pattern": "z"})


def test_the_tensors_have_the_shapes_the_network_reads() -> None:
    start = geometry.start_from_smiles("c1ccccc1")
    built = inputs.build(start, random_hessian(12, seed=0), inputs.InputSettings())
    assert built.atomic_numbers.tolist() == [6] * 6 + [1] * 6
    assert built.positions.shape == (12, 3)
    assert built.hessian.shape == (36, 36)
    assert built.wilson_b.shape == (54, 36)
    assert built.force_constants.shape == (54, 54)
    assert built.pool.shape == (54, 12)
    np.testing.assert_allclose(built.pool.sum(1).numpy(), 1.0, atol=1e-6)
    assert built.odd.numel() == 0
    with_parity = inputs.build(start, random_hessian(12, seed=0), inputs.InputSettings(parity=True))
    assert with_parity.odd.shape == (54,)
    assert with_parity.sines.shape == (54,)


def test_the_hessian_is_projected_before_the_network_reads_it() -> None:
    start = geometry.start_from_smiles("c1ccccc1")
    built = inputs.build(start, random_hessian(12, seed=0), inputs.InputSettings())
    translation = np.tile(np.eye(3), (12, 1)) * np.sqrt(np.repeat(start.masses_amu, 3))[:, None]
    weights = np.sqrt(np.repeat(start.masses_amu, 3))
    mass_weighted = built.hessian.numpy().astype(float) / np.outer(weights, weights)
    np.testing.assert_allclose(mass_weighted @ translation, 0.0, atol=1e-6)
