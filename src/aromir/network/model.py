"""The ΔH network: an equivariant encoder of the atoms and a head on pairs of primitives.

The encoder is a PaiNN-type message-passing network. It reads the atoms, their positions
and the low-level Hessian as rotation-invariant scalars per atom pair, and returns scalar
and vector features per atom. The head pools those features over the atoms of each
primitive and predicts one number per pair of the pattern: the entry of ΔF. The Cartesian
correction is ΔH = Bᵀ ΔF B.

The head sees only invariants, ΔF is invariant, and B turns with the frame, so ΔH turns
with the molecule by construction. The operations and their order follow the code the
weights were trained with; a regression pin holds the two to the same numbers.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Final, Literal, Self

import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator
from torch import Tensor, nn

from aromir.constants import ANGSTROM_PER_BOHR
from aromir.network.graph import HINGE_CLASSES
from aromir.network.pattern import PAIR_CLASSES

if TYPE_CHECKING:
    from aromir.network.inputs import NetworkInputs

CUTOFF_BOHR: Final = 5.0 / ANGSTROM_PER_BOHR
N_RADIAL: Final = 20
N_PAIR_INVARIANTS: Final = 16
N_ELEMENT_ROWS: Final = 20
N_CHARGE_STATES: Final = 4
NEUTRAL_SINGLET: Final = 0
N_CLASS_EMBEDDING: Final = 8
N_FORCE_CONSTANT_FEATURES: Final = 16
N_PAIR_FEATURE_HIDDEN: Final = 32
_NORM_FLOOR: Final = 1e-8


class Architecture(BaseModel):
    """The shape of a network; the defaults are the carried recipe's."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    aggregation: Literal["sum", "mean"] = "sum"
    n_features: int = Field(default=64, ge=1)
    """Scalar channels per atom, and as many vector channels: the update adds the two."""
    n_blocks: int = Field(default=3, ge=1)
    hidden: int = Field(default=256, ge=1)
    n_pair_features: int = Field(default=66, ge=0)
    sqm_scale: bool = True
    parity_odd: bool = False

    @model_validator(mode="after")
    def _parity_or_pair_features(self) -> Self:
        if self.parity_odd and self.n_pair_features:
            msg = (
                "parity_odd with pair features: the pair features carry signed primitive "
                "values, which the parity-odd factor does not combine with"
            )
            raise ValueError(msg)
        return self


class RadialBasis(nn.Module):
    """Gaussians of the interatomic distance under a cosine cutoff."""

    centres: Tensor

    def __init__(self) -> None:
        """Place the Gaussians evenly between zero and the cutoff."""
        super().__init__()
        self.register_buffer("centres", torch.linspace(0.0, CUTOFF_BOHR, N_RADIAL))
        self.gamma = 0.5 / (CUTOFF_BOHR / N_RADIAL) ** 2

    def forward(self, distance: Tensor) -> Tensor:
        """Return the basis values, shape (E, ``N_RADIAL``)."""
        envelope = (
            0.5 * (torch.cos(math.pi * distance / CUTOFF_BOHR) + 1.0) * (distance < CUTOFF_BOHR)
        )
        gaussians = torch.exp(-self.gamma * (distance[:, None] - self.centres[None, :]) ** 2)
        return gaussians * envelope[:, None]


def pair_invariants(hessian: Tensor, positions: Tensor) -> tuple[Tensor, Tensor]:
    """Return the Hessian as rotation-invariant scalars.

    Per atom pair, the three invariants of its 3×3 block: trace, r̂ᵀ B r̂ and Frobenius
    norm, shape (N, N, 3). Per atom, the trace and norm of its own block, shape (N, 2).
    """
    n_atoms = positions.shape[0]
    blocks = hessian.reshape(n_atoms, 3, n_atoms, 3).permute(0, 2, 1, 3)
    separation = positions[None, :, :] - positions[:, None, :]
    distance = torch.linalg.norm(separation, dim=-1)
    direction = (
        separation / torch.where(distance > 0, distance, torch.ones_like(distance))[..., None]
    )
    trace = blocks.diagonal(dim1=-2, dim2=-1).sum(-1)
    projection = torch.einsum("ija,ijab,ijb->ij", direction, blocks, direction)
    norm = torch.linalg.norm(blocks.reshape(n_atoms, n_atoms, 9), dim=-1)
    per_atom = torch.stack([trace.diagonal(), norm.diagonal()], -1)
    return torch.stack([trace, projection, norm], -1), per_atom


def _pool(
    messages: Tensor, receiver: Tensor, n_atoms: int, inverse_degree: Tensor | None
) -> Tensor:
    """Add each edge's message to its receiving atom; divide by the degree for a mean."""
    shape = (n_atoms, *messages.shape[1:])
    pooled = torch.zeros(shape, dtype=messages.dtype).index_add(0, receiver, messages)
    if inverse_degree is not None:
        pooled = pooled * inverse_degree.reshape((n_atoms,) + (1,) * (pooled.dim() - 1))
    return pooled


class Interaction(nn.Module):
    """One message-passing block: neighbours send scalars and vectors, then each atom mixes."""

    def __init__(self, n_features: int) -> None:
        """Build the message and update layers."""
        super().__init__()
        n_scalar = n_vector = n_features
        n_message = n_scalar + 2 * n_vector
        self.message = nn.Sequential(
            nn.Linear(n_scalar, n_scalar), nn.SiLU(), nn.Linear(n_scalar, n_message)
        )
        self.filter = nn.Linear(N_RADIAL + N_PAIR_INVARIANTS, n_message)
        self.mix_u = nn.Linear(n_vector, n_vector, bias=False)
        self.mix_v = nn.Linear(n_vector, n_vector, bias=False)
        self.update = nn.Sequential(
            nn.Linear(n_scalar + n_vector, n_scalar),
            nn.SiLU(),
            nn.Linear(n_scalar, n_scalar + 2 * n_vector),
        )
        self.n_scalar, self.n_vector = n_scalar, n_vector

    def forward(
        self,
        scalars: Tensor,
        vectors: Tensor,
        edges: tuple[Tensor, Tensor],
        edge_filter: Tensor,
        direction: Tensor,
        inverse_degree: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        """Return the updated scalar (N, n_scalar) and vector (N, 3, n_vector) features."""
        receiver, sender = edges
        message = self.message(scalars)[sender] * self.filter(edge_filter)
        d_scalar, gate_vector, gate_direction = torch.split(
            message, [self.n_scalar, self.n_vector, self.n_vector], dim=-1
        )
        d_vector = (
            vectors[sender] * gate_vector[:, None, :]
            + direction[:, :, None] * gate_direction[:, None, :]
        )
        n_atoms = scalars.shape[0]
        scalars = scalars + _pool(d_scalar, receiver, n_atoms, inverse_degree)
        vectors = vectors + _pool(d_vector, receiver, n_atoms, inverse_degree)
        mixed_u, mixed_v = self.mix_u(vectors), self.mix_v(vectors)
        norm = torch.sqrt((mixed_v**2).sum(1) + _NORM_FLOOR)
        update = self.update(torch.cat([scalars, norm], -1))
        scale_vector, scale_product, d_scalar = torch.split(
            update, [self.n_vector, self.n_vector, self.n_scalar], dim=-1
        )
        vectors = vectors + scale_vector[:, None, :] * mixed_u
        scalars = scalars + d_scalar + scale_product * (mixed_u * mixed_v).sum(1)
        return scalars, vectors


class Encoder(nn.Module):
    """Per-atom scalar and vector features from the atoms and the low-level Hessian."""

    def __init__(self, architecture: Architecture) -> None:
        """Build the embeddings and the interaction blocks."""
        super().__init__()
        n_scalar = n_vector = architecture.n_features
        self.element_embedding = nn.Embedding(N_ELEMENT_ROWS, n_scalar)
        self.atom_invariants = nn.Linear(2, n_scalar)
        self.charge_embedding = nn.Embedding(N_CHARGE_STATES, n_scalar)
        self.hinge_embedding = nn.Embedding(len(HINGE_CLASSES), n_scalar)
        self.radial_basis = RadialBasis()
        self.pair_invariants = nn.Sequential(
            nn.Linear(3, N_PAIR_INVARIANTS),
            nn.SiLU(),
            nn.Linear(N_PAIR_INVARIANTS, N_PAIR_INVARIANTS),
        )
        self.blocks = nn.ModuleList(
            Interaction(architecture.n_features) for _ in range(architecture.n_blocks)
        )
        self.n_vector = n_vector
        self.mean_aggregation = architecture.aggregation == "mean"

    def forward(self, inputs: NetworkInputs) -> tuple[Tensor, Tensor]:
        """Return scalars (N, n_scalar) and vectors (N, 3, n_vector)."""
        positions = inputs.positions
        n_atoms = positions.shape[0]
        pair_scalars, atom_scalars = pair_invariants(inputs.hessian, positions)
        within = (torch.cdist(positions, positions) < CUTOFF_BOHR) & ~torch.eye(
            n_atoms, dtype=torch.bool
        )
        receiver, sender = torch.nonzero(within, as_tuple=True)
        separation = positions[sender] - positions[receiver]
        distance = torch.linalg.norm(separation, dim=-1)
        direction = separation / distance[:, None]
        edge_filter = torch.cat(
            [self.radial_basis(distance), self.pair_invariants(pair_scalars[receiver, sender])], -1
        )
        scalars = self.element_embedding(inputs.atomic_numbers) + self.atom_invariants(atom_scalars)
        scalars = scalars + self.charge_embedding(torch.tensor(NEUTRAL_SINGLET))[None, :]
        scalars = scalars + self.hinge_embedding(inputs.hinge_classes)
        vectors = torch.zeros(n_atoms, 3, self.n_vector, dtype=positions.dtype)
        inverse_degree = None
        if self.mean_aggregation:
            degree = torch.bincount(receiver, minlength=n_atoms).clamp_min(1)
            inverse_degree = 1.0 / degree.to(positions.dtype)
        for block in self.blocks:
            scalars, vectors = block(
                scalars, vectors, (receiver, sender), edge_filter, direction, inverse_degree
            )
        return scalars, vectors


class DeltaHessianNetwork(nn.Module):
    """The encoder with the head that predicts ΔF on the pair pattern."""

    class_scale: Tensor
    force_constant_scale: Tensor
    pair_feature_mean: Tensor
    pair_feature_std: Tensor

    def __init__(self, architecture: Architecture) -> None:
        """Build the network; every parameter and buffer is then loaded from the weights."""
        super().__init__()
        self.architecture = architecture
        self.encoder = Encoder(architecture)
        self.class_embedding = nn.Embedding(len(PAIR_CLASSES), N_CLASS_EMBEDDING)
        self.force_constants = nn.Linear(3, N_FORCE_CONSTANT_FEATURES)
        n_features = architecture.n_pair_features
        self.pair_features = nn.Linear(n_features, N_PAIR_FEATURE_HIDDEN) if n_features else None
        n_inputs = (
            3 * architecture.n_features
            + N_CLASS_EMBEDDING
            + N_FORCE_CONSTANT_FEATURES
            + (N_PAIR_FEATURE_HIDDEN if n_features else 0)
        )
        hidden = architecture.hidden
        self.head = nn.Sequential(
            nn.Linear(n_inputs, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        self.low_level_share = (
            nn.Parameter(torch.zeros(len(PAIR_CLASSES))) if architecture.sqm_scale else None
        )
        self.register_buffer("class_scale", torch.ones(len(PAIR_CLASSES)))
        self.register_buffer("force_constant_scale", torch.ones(3))
        self.register_buffer("pair_feature_mean", torch.zeros(max(n_features, 1)))
        self.register_buffer("pair_feature_std", torch.ones(max(n_features, 1)))

    def delta_f(self, inputs: NetworkInputs) -> Tensor:
        """Return the predicted ΔF on the pattern's pairs, shape (P,), atomic units."""
        scalars, vectors = self.encoder(inputs)
        vector_norms = torch.sqrt((vectors**2).sum(1) + _NORM_FLOOR)
        pooled_scalars = inputs.pool @ scalars
        pooled_norms = inputs.pool @ vector_norms
        p, q = inputs.pairs[:, 0], inputs.pairs[:, 1]
        force = inputs.force_constants
        low = torch.stack([force[p, q], force[p, p], force[q, q]], -1)
        mixed = None
        if self.architecture.parity_odd:
            mixed = inputs.odd[p] ^ inputs.odd[q]
            low = torch.cat(
                [torch.where(mixed, low[:, 0].abs(), low[:, 0])[:, None], low[:, 1:]], -1
            )
        parts = [
            pooled_scalars[p] + pooled_scalars[q],
            pooled_scalars[p] * pooled_scalars[q],
            pooled_norms[p] + pooled_norms[q],
            self.class_embedding(inputs.pair_classes),
            self.force_constants(low / self.force_constant_scale),
        ]
        if self.pair_features is not None:
            standardised = (inputs.pair_features - self.pair_feature_mean) / self.pair_feature_std
            parts.append(self.pair_features(standardised))
        raw: Tensor = self.head(torch.cat(parts, -1)).squeeze(-1)
        delta = raw * self.class_scale[inputs.pair_classes]
        if mixed is not None:
            odd_sine = torch.where(inputs.odd[p], inputs.sines[p], inputs.sines[q])
            delta = torch.where(mixed, delta * odd_sine, delta)
        if self.low_level_share is not None:
            delta = delta + self.low_level_share[inputs.pair_classes] * force[p, q]
        return delta

    def forward(self, inputs: NetworkInputs) -> Tensor:
        """Return the Cartesian correction ΔH = Bᵀ ΔF B, shape (3N, 3N), hartree/bohr²."""
        delta = self.delta_f(inputs)
        n_primitives = inputs.wilson_b.shape[0]
        p, q = inputs.pairs[:, 0], inputs.pairs[:, 1]
        delta_f = torch.zeros(n_primitives, n_primitives, dtype=delta.dtype)
        delta_f = delta_f.index_put((p, q), delta).index_put((q, p), delta)
        return inputs.wilson_b.T @ delta_f @ inputs.wilson_b
