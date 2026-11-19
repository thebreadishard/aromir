"""Shared test molecules: small, synthetic where possible, so most tests need no chemistry."""

from __future__ import annotations

import numpy as np
import pytest

from aromir import own_spread
from aromir.molecule import FloatArray, Geometry

from support import WATER_BONDS, bond_force_field, unmoved


@pytest.fixture
def water() -> Geometry:
    """Water near its B3LYP/6-31G* minimum, in bohr."""
    return Geometry(
        ("O", "H", "H"),
        np.array([[0.0, 0.0, 0.2217], [0.0, 1.4309, -0.8867], [0.0, -1.4309, -0.8867]]),
    )


@pytest.fixture
def water_hessian(water: Geometry) -> FloatArray:
    """A synthetic, exactly rigid-motion-free Hessian for the water fixture."""
    return bond_force_field(water, WATER_BONDS)


@pytest.fixture(scope="module")
def water_for_module() -> Geometry:
    """The water geometry for fixtures that compute once per module."""
    return Geometry(
        ("O", "H", "H"),
        np.array([[0.0, 0.0, 0.2217], [0.0, 1.4309, -0.8867], [0.0, -1.4309, -0.8867]]),
    )


@pytest.fixture
def networks_that_move_no_band(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give every certificate the own spread of networks that move no band.

    The random networks of a made-up release are not invariant at all, so the own spread
    would withhold every band; a test of the other conditions of the licence sets it aside.
    ``tests/test_own_spread.py`` measures it on the shipped networks.
    """

    def unmoving(
        _release: object, _smiles: str, molecule: Geometry, *_: object
    ) -> own_spread.OwnSpread:
        return unmoved(3 * molecule.n_atoms - 6)

    monkeypatch.setattr(own_spread, "measure", unmoving)
