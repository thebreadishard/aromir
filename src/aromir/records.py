"""Writing result files: complete or not at all, and each with where it came from."""

from __future__ import annotations

import json
import os
import platform
import sys
from datetime import UTC, datetime
from importlib import metadata
from typing import TYPE_CHECKING, Any, Final

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from aromir.molecule import FloatArray

_RECORDED_PACKAGES: Final = ("aromir", "pyscf", "geometric", "rdkit", "numpy", "scipy")


def provenance() -> dict[str, Any]:
    """Return what produced a result: time, command, platform and package versions."""
    return {
        "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {name: metadata.version(name) for name in _RECORDED_PACKAGES},
    }


def write_json(path: Path, record: dict[str, Any]) -> None:
    """Write a JSON record through a temporary file, so a reader never sees half of one."""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(record, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)  # noqa: PTH105  os.replace is the atomic move; Path.replace wraps it


def write_arrays(path: Path, arrays: Mapping[str, FloatArray]) -> None:
    """Write named arrays to a compressed .npz through a temporary file."""
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, allow_pickle=False, **arrays)
    os.replace(temporary, path)  # noqa: PTH105  as in write_json
