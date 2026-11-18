"""The error of what a release delivers, measured on molecules with a reference Hessian.

Two quantities carry the name "the network's error", and they are not the same number.

The *band error of the seed mean* is the product's: the seeds' mean ΔH is added to the low
level's Hessian, the sum is diagonalised, each corrected band is paired with the reference
band that moves most like it, and the difference of the two positions is the error. A band
counts in the family of the low-level mode it is the partner of. This is what a user of
``aromir spectrum`` gets, and what a licence is read on.

The *first-order error per seed* is the trainer's: for each seed alone, the diagonal of
the ΔH error in the low level's mode basis, as a first-order shift of each mode. It needs
no diagonalisation, which is why a training loop reports it. The two differ for two
reasons, and neither is a law: averaging the seeds removes part of each seed's error, and
a first-order shift is not the shift of the diagonalised Hessian. For release v1.3 the
band error is the smaller number in every family on both hold-outs.

A hold-out is a folder of ``.npz`` files, one molecule each, with ``symbols``,
``coords_bohr``, ``hessian`` (the low level's, of the kind the release was trained on,
hartree/bohr²) and ``reference_hessian`` (the higher level's at the same geometry). The
atoms are in the order the network is to read them in. ``evaluate`` reads a folder of such
hold-out folders; the molecules of a hold-out are taken in the order of their file names.
"""

from __future__ import annotations

import zipfile
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

import numpy as np

from aromir import families, spectrum, vibrations
from aromir.constants import WAVENUMBERS_PER_HARTREE
from aromir.errors import RefusedError
from aromir.molecule import Geometry

if TYPE_CHECKING:
    from pathlib import Path

    from aromir.molecule import FloatArray
    from aromir.release import Release

BOOTSTRAP_DRAWS: Final = 4000
BOOTSTRAP_SEED: Final = 20261116
"""The resampling is seeded: the same hold-out gives the same interval."""
INTERVAL_PERCENT: Final = (2.5, 97.5)
DECIMALS: Final = 2
"""A record holds its errors to this many decimals, and a licence is read on the record."""
_FIELDS: Final = ("symbols", "coords_bohr", "hessian", "reference_hessian")


@dataclass(frozen=True, eq=False)
class Reference:
    """One molecule with what the network reads and what its answer is compared with."""

    name: str
    geometry: Geometry
    hessian: FloatArray
    reference_hessian: FloatArray


@dataclass(frozen=True, eq=False)
class MoleculeErrors:
    """A release's errors on one molecule, mode by mode, in cm⁻¹.

    Attributes:
        band: The corrected band position of the seeds' mean minus its reference partner's.
        band_family: The family each of those bands counts in.
        first_order: Per seed, the first-order shift the seed's ΔH error gives each
            low-level mode, shape (seeds, modes).
        first_order_family: The family of each low-level mode.
    """

    band: FloatArray
    band_family: tuple[families.Family, ...]
    first_order: FloatArray
    first_order_family: tuple[families.Family, ...]


def write(path: Path, reference: Reference) -> None:
    """Write a molecule in the format ``read`` takes."""
    np.savez_compressed(
        path,
        symbols=np.array(reference.geometry.symbols),
        coords_bohr=reference.geometry.coords_bohr,
        hessian=reference.hessian,
        reference_hessian=reference.reference_hessian,
    )


def read(path: Path) -> Reference:
    """Read one molecule of a hold-out.

    Raises:
        RefusedError: The file does not hold the four arrays, or they do not fit together.
    """
    try:
        with np.load(path) as arrays:
            missing = [name for name in _FIELDS if name not in arrays.files]
            if missing:
                raise RefusedError([f"{path}: no {', '.join(missing)}"])
            symbols = tuple(str(symbol) for symbol in arrays["symbols"])
            geometry = Geometry(symbols, np.array(arrays["coords_bohr"], float))
            hessian = np.array(arrays["hessian"], float)
            reference_hessian = np.array(arrays["reference_hessian"], float)
    except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile) as error:
        raise RefusedError([f"{path} cannot be read as a hold-out molecule: {error}"]) from error
    expected = (3 * geometry.n_atoms, 3 * geometry.n_atoms)
    if hessian.shape != expected or reference_hessian.shape != expected:
        reason = (
            f"{path}: Hessians of shape {hessian.shape} and {reference_hessian.shape} for "
            f"{geometry.n_atoms} atoms"
        )
        raise RefusedError([reason])
    if not (np.isfinite(hessian).all() and np.isfinite(reference_hessian).all()):
        raise RefusedError([f"{path}: a Hessian holds a value that is not finite"])
    return Reference(path.stem, geometry, hessian, reference_hessian)


def first_order_shifts(
    delta: FloatArray, low: vibrations.Modes, masses_amu: FloatArray
) -> FloatArray:
    """Return the first-order shift of every low-level mode under a Hessian change, cm⁻¹.

    A wavenumber is the signed root of a curvature, so its derivative to the curvature is
    1 / (2 |ν|) in these units: the same for a real and an imaginary mode.
    """
    weights = vibrations.mass_weights(masses_amu)
    weighted = 0.5 * (delta + delta.T) / np.outer(weights, weights)
    curvature_shift = np.einsum("ik,ij,jk->k", low.vectors, weighted, low.vectors)
    shifts = curvature_shift * WAVENUMBERS_PER_HARTREE**2 / (2 * np.abs(low.wavenumbers))
    return np.asarray(shifts, dtype=float)


def molecule_errors(reference: Reference, release: Release) -> MoleculeErrors:
    """Return a release's two kinds of error on one molecule.

    Raises:
        RefusedError: The low-level Hessian has an imaginary mode: the molecule is at no
            minimum of the low level, and a first-order shift of such a mode means nothing.
    """
    geometry, low_hessian = reference.geometry, reference.hessian
    low = vibrations.vibrational_modes(low_hessian, geometry)
    if low.n_imaginary:
        reason = (
            f"{reference.name}: the low-level Hessian has {low.n_imaginary} imaginary "
            "mode(s); a hold-out molecule is at a minimum of the low level"
        )
        raise RefusedError([reason])
    target = vibrations.vibrational_modes(reference.reference_hessian, geometry)
    family = families.classify(geometry, low)
    corrections = release.corrections(geometry, low_hessian)
    corrected = vibrations.vibrational_modes(low_hessian + corrections.mean, geometry)
    to_target = spectrum.partners(corrected, target)
    to_low = spectrum.partners(corrected, low)
    truth = first_order_shifts(reference.reference_hessian - low_hessian, low, geometry.masses_amu)
    first_order = np.array(
        [
            first_order_shifts(delta, low, geometry.masses_amu) - truth
            for delta in corrections.per_seed
        ]
    )
    return MoleculeErrors(
        band=np.asarray(corrected.wavenumbers - target.wavenumbers[to_target], dtype=float),
        band_family=tuple(family[index] for index in to_low),
        first_order=first_order,
        first_order_family=tuple(family),
    )


def _rms(parts: list[FloatArray]) -> float | None:
    """Return the rms over all parts; None for no value."""
    values = np.concatenate(parts)
    return float(np.sqrt(np.mean(values**2))) if values.size else None


def _recorded(value: float | None) -> float | None:
    return None if value is None else round(value, DECIMALS)


def interval(per_molecule: list[FloatArray], line_cm: float) -> tuple[float, float, float]:
    """Return a hold-out's rms under resampling of whole molecules.

    The modes of one molecule are not independent, so molecules are drawn, with
    replacement, as often as the hold-out has them. The answer is the 2.5th and the 97.5th
    percentile of the drawn sets' rms and the share of draws within the licence line (at
    or below it, as the licence itself counts): a statement about these molecules as a
    sample, no more. A draw of molecules that have no band of the family counts as zero.
    """
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    sums = np.array([float(np.sum(errors**2)) for errors in per_molecule])
    counts = np.array([len(errors) for errors in per_molecule], dtype=float)
    picks = rng.integers(0, len(sums), size=(BOOTSTRAP_DRAWS, len(sums)))
    drawn = np.sqrt(sums[picks].sum(axis=1) / np.maximum(counts[picks].sum(axis=1), 1.0))
    low, high = np.percentile(drawn, INTERVAL_PERCENT)
    return float(low), float(high), float(np.mean(drawn <= line_cm))


def _of_family(errors: FloatArray, family: tuple[families.Family, ...], wanted: str) -> FloatArray:
    return np.asarray(errors[..., np.array(family) == wanted], dtype=float)


def holdout_errors(molecules: list[MoleculeErrors], line_cm: float) -> dict[str, Any]:
    """Return a hold-out's two kinds of error per family, rounded as a record holds them."""
    n_seeds = molecules[0].first_order.shape[0]
    band: dict[str, Any] = {}
    first_order: dict[str, list[float | None]] = {}
    first_order_mean: dict[str, float | None] = {}
    for family in families.FAMILIES:
        bands = [_of_family(m.band, m.band_family, family) for m in molecules]
        n_modes = int(sum(len(part) for part in bands))
        band[family] = {"rms_cm": _recorded(_rms(bands)), "n_modes": n_modes}
        if n_modes:
            low, high, share = interval(bands, line_cm)
            band[family] |= {
                "interval_cm": [round(low, DECIMALS), round(high, DECIMALS)],
                "share_of_resamples_within_line": round(share, DECIMALS),
            }
        per_seed = [
            _rms([_of_family(m.first_order[seed], m.first_order_family, family) for m in molecules])
            for seed in range(n_seeds)
        ]
        first_order[family] = [_recorded(value) for value in per_seed]
        values = [value for value in per_seed if value is not None]
        first_order_mean[family] = _recorded(float(np.mean(values))) if values else None
    return {
        "n_molecules": len(molecules),
        "band_error_seed_mean": band,
        "first_order_error_per_seed": first_order,
        "first_order_error_per_seed_mean": first_order_mean,
    }


def evaluate(release: Release, folder: Path) -> dict[str, Any]:
    """Measure a release on every hold-out folder under a folder.

    Returns:
        Per hold-out (the sub-folder's name): the number of molecules, and per family the
        band error of the seed mean with its mode count and resampling interval, and the
        first-order error of each seed with their mean. A family without a mode in a
        hold-out has the count 0 and no error.

    Raises:
        RefusedError: The folder holds no hold-out, or a hold-out no molecule, or the
            release's seeds are release networks, which trained on the hold-outs.
    """
    if release.shipped is not None:
        release_file = "release.json"
        reason = (
            f"the folder ships release networks of v{release.network_version}, which trained "
            "on the hold-outs: their error there is no measurement. The folder's errors were "
            "measured on the version's measurement networks, which the package does not ship "
            f"({release_file} names their checkpoints)"
        )
        raise RefusedError([reason])
    holdouts = sorted(path for path in folder.iterdir() if path.is_dir()) if folder.is_dir() else []
    if not holdouts:
        raise RefusedError([f"{folder}: no hold-out folder in it"])
    line = release.evaluation.licence_line_cm
    measured = {}
    for holdout in holdouts:
        files = sorted(holdout.glob("*.npz"))
        if not files:
            raise RefusedError([f"{holdout}: no molecule (.npz) in it"])
        molecules = [molecule_errors(read(path), release) for path in files]
        measured[holdout.name] = holdout_errors(molecules, line)
    return measured


def record_tables(measured: dict[str, Any]) -> dict[str, Any]:
    """Return what ``evaluate`` measured as the tables an evaluation record holds.

    A record lists every family on every hold-out, so a hold-out without a band of some
    family cannot carry a record.

    Raises:
        RefusedError: A family has no band in a hold-out.
    """
    empty = [
        f"hold-out ({name}) has no {family} band"
        for name, holdout in measured.items()
        for family, entry in holdout["band_error_seed_mean"].items()
        if not entry["n_modes"]
    ]
    if empty:
        raise RefusedError(empty)
    names = {
        "band_error_seed_mean_cm": "band_error_seed_mean",
        "first_order_error_per_seed_cm": "first_order_error_per_seed",
        "first_order_error_per_seed_mean_cm": "first_order_error_per_seed_mean",
    }
    return {
        field: {
            family: {name: holdout[key][family] for name, holdout in measured.items()}
            for family in families.FAMILIES
        }
        for field, key in names.items()
    }
