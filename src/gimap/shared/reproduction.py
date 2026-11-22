"""Versioned recipes identify their inputs and carry lossless reference columns for comparison."""

from __future__ import annotations

import base64
import hashlib
import platform
import zlib
from concurrent.futures import CancelledError
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path, PureWindowsPath

import numpy as np

from .file_digest import file_record
from .revision import source_commit
from .version import SOFTWARE


_verification = ContextVar("gimap_reproduction_verification", default=None)


@contextmanager
def verify_run(should_cancel=None):
    """Share verified identities across frames of one batch, never across reproduction requests."""
    relocated = {}
    token = _verification.set(({}, should_cancel, relocated))
    try:
        yield relocated
    finally:
        _verification.reset(token)


def file_reference(path: str | Path) -> dict:
    return file_record(Path(path).resolve())


def verified_path(reference: dict, record_dir: Path) -> Path:
    """Verify the recorded file, or its same-named copy beside the record when the original is absent."""
    path = record_dir / reference["path"]
    original = path
    if not path.is_file():
        # The exporting and receiving machines need not use the same path separator.
        name = PureWindowsPath(reference["path"]).name
        copy = record_dir / name
        if not name or not copy.is_file():
            raise ValueError(f"Missing referenced file: {path}")
        path = copy
    if not reference.get("sha256"):
        raise ValueError(f"The record has no SHA-256 for {path}")
    cache, cancelled, relocated = _verification.get() or ({}, None, {})
    stat = path.stat()
    key = (path.resolve(), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino)
    if key not in cache:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                if cancelled and cancelled():
                    raise CancelledError()
                digest.update(block)
        cache[key] = digest.hexdigest()
    if cache[key] != reference["sha256"]:
        raise ValueError(f"SHA-256 mismatch for {path}: expected {reference['sha256']}, got {cache[key]}")
    if path != original:
        relocated[str(reference["path"])] = {"path": str(path.resolve()), "sha256": cache[key]}
    return path


def _column(values) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or array.dtype.kind not in "fiub":
        raise ValueError("A reproduction column must be a one-dimensional numeric array.")
    return np.ascontiguousarray(array, dtype=array.dtype.newbyteorder("<"))


def pack_column(values) -> dict:
    array = _column(values)
    return {"dtype": array.dtype.str, "size": len(array),
            "zlib_base64": base64.b64encode(zlib.compress(array.tobytes())).decode("ascii")}


def unpack_column(column: dict) -> np.ndarray:
    dtype = np.dtype(column["dtype"])
    size = column["size"]
    if dtype.kind not in "fiub" or not isinstance(size, int) or size < 0:
        raise ValueError("Invalid reference column type.")
    length = size * dtype.itemsize
    inflater = zlib.decompressobj()
    raw = inflater.decompress(base64.b64decode(column["zlib_base64"], validate=True), length + 1)
    if len(raw) != length or not inflater.eof or inflater.unused_data:
        raise ValueError("Invalid reference column length.")
    return np.frombuffer(raw, dtype=dtype)


@lru_cache(maxsize=1)
def numerical_environment() -> dict:
    return {"python": platform.python_version(), "numpy": np.__version__, "scipy": version("scipy"),
            "system": platform.system(), "machine": platform.machine()}


def replay_header(kind: str, tables: dict) -> dict:
    reference = {table: {name: pack_column(values) for name, values in columns.items()}
                 for table, columns in tables.items()}
    return {"schema": 1, "kind": kind, "software_version": SOFTWARE, "commit": source_commit(),
            "runtime": numerical_environment(), "reference": reference}


def compare_column(reference: np.ndarray, values) -> dict:
    actual = _column(values)
    if len(reference) != len(actual):
        return {"status": "different", "max_relative_difference": None,
                "expected_rows": len(reference), "actual_rows": len(actual)}
    if reference.dtype == actual.dtype and reference.tobytes() == actual.tobytes():
        return {"status": "identical", "max_relative_difference": 0.0}
    # NaNs at the same position and matching infinities have zero numerical difference.
    # Signed zero and NaN payloads are distinguished by the byte comparison above.
    equal = (reference == actual) | (np.isnan(reference) & np.isnan(actual))
    if reference.dtype.kind in "uib" and actual.dtype.kind in "uib":
        pairs = [(int(r), int(a)) for r, a in zip(reference, actual) if int(r) != int(a)]
        maximum = ("infinity" if any(r == 0 for r, _ in pairs)
                   else max((abs(a - r) / abs(r) for r, a in pairs), default=0.0))
        return {"status": "different", "max_relative_difference": maximum}
    r, a = reference[~equal].astype(np.float64), actual[~equal].astype(np.float64)
    if np.any(r == 0) or not np.all(np.isfinite(r) & np.isfinite(a)):
        maximum = "infinity"
    else:
        with np.errstate(over="ignore"):
            # Opposite signs can overflow a-r before division even when the relative error is small.
            same_sign = np.signbit(a) == np.signbit(r)
            delta = np.empty_like(r)
            delta[same_sign] = np.abs((a[same_sign] - r[same_sign]) / r[same_sign])
            delta[~same_sign] = 1.0 + np.abs(a[~same_sign] / r[~same_sign])
            largest = float(np.max(delta, initial=0.0))
            maximum = largest if np.isfinite(largest) else "infinity"
    return {"status": "different", "max_relative_difference": maximum}
