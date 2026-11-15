"""The network against the pins: the same ΔH as the code its weights are trained with."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from safetensors.torch import load_file

from aromir.molecule import Geometry
from aromir.network import convert, inputs, weights
from aromir.network.model import DeltaHessianNetwork

PINS = Path(__file__).parent / "data" / "network"
CONFIGURATIONS = ("carried_shape", "order_free", "parity_odd")
RELATIVE_TOLERANCE = 1e-5
"""Single-precision arithmetic in another order of summation; the ports agree to 3e-7."""


def _network(configuration: str) -> tuple[DeltaHessianNetwork, inputs.InputSettings]:
    record = json.loads((PINS / f"{configuration}.json").read_text(encoding="utf-8"))
    architecture, settings = convert.settings(record)
    network = DeltaHessianNetwork(architecture)
    trainer_state = load_file(PINS / f"{configuration}.safetensors")
    network.load_state_dict(convert.state(trainer_state), strict=True)
    return network, settings


def _pins() -> list[tuple[str, Path]]:
    return [
        (configuration, path)
        for configuration in CONFIGURATIONS
        for path in sorted(PINS.glob(f"{configuration}_*.npz"))
    ]


def test_every_configuration_is_pinned() -> None:
    pinned = {configuration for configuration, _ in _pins()}
    assert pinned == set(CONFIGURATIONS)
    assert len(_pins()) == 16


@pytest.mark.parametrize(("configuration", "path"), _pins(), ids=lambda value: Path(value).stem)
def test_the_port_reproduces_the_trainers_output(configuration: str, path: Path) -> None:
    network, settings = _network(configuration)
    with np.load(path) as arrays:
        pin = {key: arrays[key] for key in arrays.files}
    geometry = Geometry(tuple(str(symbol) for symbol in pin["symbols"]), pin["coords_bohr"])

    built = inputs.build(geometry, pin["hessian"], settings)

    np.testing.assert_array_equal(built.pairs.numpy(), pin["pairs"])
    np.testing.assert_array_equal(built.pair_classes.numpy(), pin["pair_classes"])
    np.testing.assert_array_equal(built.hinge_classes.numpy(), pin["hinge_classes"])
    if settings.parity:
        np.testing.assert_array_equal(built.odd.numpy(), pin["odd"])
    delta_h = weights.correction(network, built)
    scale = float(np.abs(pin["delta_h"]).max())
    assert float(np.abs(delta_h - pin["delta_h"]).max()) <= RELATIVE_TOLERANCE * scale
