"""Where GIMaP's proposal network is and loading it once per process.

The network that comes with GIMaP is ``data/<DEFAULT>.npz`` beside this file, with its record ``.json`` (architecture,
training, SHA-256) and ``MODEL_CARD.md``; the environment variable ``GIMAP_ML1D_NETWORK`` (a path without suffix)
selects another one, as the tests do.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from .flow import Network

DEFAULT = "flow_v1"
FOLDER = Path(__file__).resolve().parent / "data"
VARIABLE = "GIMAP_ML1D_NETWORK"


def path() -> Path:
    chosen = os.environ.get(VARIABLE)
    return Path(chosen).with_suffix(".npz") if chosen else FOLDER / f"{DEFAULT}.npz"


def available() -> bool:
    where = path()
    return where.is_file() and where.with_suffix(".json").is_file()


@lru_cache(maxsize=2)
def _load(where: str) -> Network:
    return Network.load(where)


def load() -> Network:
    """The network (checked against its SHA-256 on first use); a clear error when it is not there."""
    where = path()
    if not available():
        raise FileNotFoundError(f"The proposal network is missing: {where.name} (and its .json) belong in {where.parent}")
    return _load(str(where))


__all__ = ["DEFAULT", "FOLDER", "VARIABLE", "available", "load", "path"]
