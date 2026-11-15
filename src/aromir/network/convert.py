"""Turn a training checkpoint of the capstone's trainer into aromir's weight files.

    python -m aromir.network.convert <checkpoint.pt> <folder> <name> --network-version 1.3

The trainer saves a pickled record with its own parameter names and with the Cartesian
read-out head of the encoder, which the pair head never uses. This module renames the
parameters, drops that head, and refuses a checkpoint of a shape aromir does not run.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import torch

from aromir.network import weights
from aromir.network.inputs import InputSettings
from aromir.network.model import Architecture
from aromir.network.primitives import PrimitiveSettings

if TYPE_CHECKING:
    from torch import Tensor

_UNUSED: Final = re.compile(r"^body\.(head\.\d+\.(weight|bias)|proj\.weight)$")
"""The encoder's Cartesian read-out: in every checkpoint, used by no pair head."""

_RENAMES: Final = (
    (r"^body\.emb\.", "encoder.element_embedding."),
    (r"^body\.node_in\.", "encoder.atom_invariants."),
    (r"^body\.q_emb\.", "encoder.charge_embedding."),
    (r"^body\.h_emb\.", "encoder.hinge_embedding."),
    (r"^body\.rbf\.", "encoder.radial_basis."),
    (r"^body\.inv\.", "encoder.pair_invariants."),
    (r"^body\.blocks\.(\d+)\.phi\.", r"encoder.blocks.\1.message."),
    (r"^body\.blocks\.(\d+)\.W\.", r"encoder.blocks.\1.filter."),
    (r"^body\.blocks\.(\d+)\.U\.", r"encoder.blocks.\1.mix_u."),
    (r"^body\.blocks\.(\d+)\.V\.", r"encoder.blocks.\1.mix_v."),
    (r"^body\.blocks\.(\d+)\.a\.", r"encoder.blocks.\1.update."),
    (r"^cls_emb\.", "class_embedding."),
    (r"^flow_in\.", "force_constants."),
    (r"^pfeat_in\.", "pair_features."),
    (r"^pfeat_mu$", "pair_feature_mean"),
    (r"^pfeat_sd$", "pair_feature_std"),
    (r"^flow_scale$", "force_constant_scale"),
    (r"^alpha$", "low_level_share"),
    (r"^(head\.\d+\.(weight|bias)|class_scale)$", r"\1"),
)


def _renamed(key: str) -> str:
    for old, new in _RENAMES:
        renamed, count = re.subn(old, new, key)
        if count:
            return renamed
    msg = f"the checkpoint holds a parameter aromir does not know: {key}"
    raise ValueError(msg)


def state(trainer_state: dict[str, Tensor]) -> dict[str, Tensor]:
    """Return a trainer's state dict under aromir's parameter names, without the unused head."""
    return {_renamed(key): value for key, value in trainer_state.items() if not _UNUSED.match(key)}


def settings(checkpoint: dict[str, Any]) -> tuple[Architecture, InputSettings]:
    """Read the architecture and the input settings from a trainer's checkpoint record.

    Raises:
        ValueError: The checkpoint is not a pair-head network, or uses the rank-2 input,
            neither of which aromir runs.
    """
    constructor = dict(checkpoint["ctor"])
    if checkpoint["head"] != "hybrid":
        msg = f"head {checkpoint['head']!r}: only the pair ('hybrid') head is supported"
        raise ValueError(msg)
    if constructor.pop("tensor_input"):
        msg = "the rank-2 Hessian input (tensor_input) is not supported"
        raise ValueError(msg)
    n_scalar, n_vector = constructor.pop("n_s"), constructor.pop("n_v")
    if n_scalar != n_vector:
        msg = f"{n_scalar} scalar and {n_vector} vector channels: the network needs as many of each"
        raise ValueError(msg)
    architecture = Architecture(
        aggregation=constructor.pop("aggregation"),
        n_features=n_scalar,
        n_blocks=constructor.pop("n_blocks"),
        hidden=constructor.pop("hidden"),
        n_pair_features=constructor.pop("n_pair_features"),
        sqm_scale=constructor.pop("sqm_scale"),
        parity_odd=constructor.pop("parity_odd"),
    )
    if constructor:
        msg = (
            f"the checkpoint's constructor has settings aromir does not know: {sorted(constructor)}"
        )
        raise ValueError(msg)
    primitive_keys = {"angle_choice", "dihedral_encoding"} & checkpoint.keys()
    if len(primitive_keys) == 1:
        msg = f"the checkpoint names {sorted(primitive_keys)} without the other primitive setting"
        raise ValueError(msg)
    primitives = PrimitiveSettings(**{key: checkpoint[key] for key in primitive_keys})
    inputs = InputSettings(
        pattern=checkpoint["pattern"], primitives=primitives, parity=architecture.parity_odd
    )
    return architecture, inputs


def convert(checkpoint_path: Path, folder: Path, name: str, network_version: str) -> weights.Recipe:
    """Convert one checkpoint file; return the recipe written beside the tensors."""
    # The trainer's record is a pickle of tensors, settings and provenance; it is the
    # maintainer's own file, read once to produce the pickle-free files users load.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    architecture, inputs = settings(checkpoint)
    recipe = weights.Recipe(
        network_version=network_version,
        seed=int(checkpoint["seed"]),
        architecture=architecture,
        inputs=inputs,
    )
    weights.save(state(checkpoint["state"]), folder, name, recipe)
    return recipe


def main(argv: list[str] | None = None) -> None:
    """Run the conversion from the command line."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("folder", type=Path)
    parser.add_argument("name")
    parser.add_argument("--network-version", required=True)
    arguments = parser.parse_args(argv)
    recipe = convert(
        arguments.checkpoint, arguments.folder, arguments.name, arguments.network_version
    )
    print(f"{arguments.name}: network {recipe.network_version}, seed {recipe.seed}")  # noqa: T201  the tool's answer


if __name__ == "__main__":
    main()
