"""The package's exceptions: every failure a caller can act on has its own type."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


class AromirError(Exception):
    """Base class of every error this package raises on purpose."""


class RefusedError(AromirError):
    """A request is refused before any computation; ``reasons`` names every cause found."""

    def __init__(self, reasons: Sequence[str]) -> None:
        """Keep each reason separately, and join them for the message."""
        super().__init__("; ".join(reasons))
        self.reasons = tuple(reasons)


class ConvergenceError(AromirError):
    """A self-consistent field or a geometry optimisation did not converge."""


class NotAMinimumError(AromirError):
    """The optimised geometry has an imaginary mode, so it has no harmonic spectrum."""


class ResponseError(AromirError):
    """The dipole derivatives failed one of their own checks, so no intensity is trusted."""


class InternalCoordinatesError(AromirError):
    """The Wilson B matrix is not one of exactly 3N − 6 internal directions, or not finite."""
