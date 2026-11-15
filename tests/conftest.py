"""Shared test molecules: small, synthetic where possible, so most tests need no chemistry."""

from __future__ import annotations

import numpy as np
import pytest

from aromir.molecule import FloatArray, Geometry

from support import WATER_BONDS, bond_force_field


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
