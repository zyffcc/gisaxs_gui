"""Deterministic, checked, and atomically published V5 array artifacts.

The container is deliberately generic: the grouped-dataset contract owns the
table semantics, while this module guarantees exact manifest/array replay,
safe ``allow_pickle=False`` loading, a stable physical SHA-256, and exclusive
publication.  Fixed ZIP metadata makes identical inputs byte-identical.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import re
import tempfile
from types import MappingProxyType
from typing import Mapping
import zipfile

import numpy as np


V5_CHECKED_ARRAY_ARTIFACT_SCHEMA = "gisaxs.posterior_v8.checked_array_artifact/v1"
V5_CHECKED_ARRAY_ARTIFACT_VERSION = "deterministic_zip_npy_atomic_exclusive_v1"
_ARRAY_NAME = re.compile(r"[a-z][a-z0-9_]*(?:__[a-z][a-z0-9_]*)*\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)


def canonical_json(payload: object) -> str:
    """Encode one strict, finite JSON value for hashing."""

    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _strict_json_object(encoded: str) -> dict[str, object]:
    def reject_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON field {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(encoded, object_pairs_hook=reject_duplicates)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("artifact manifest is not valid strict JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("artifact manifest must contain one JSON object")
    return value


def _validated_name(name: object) -> str:
    if not isinstance(name, str) or _ARRAY_NAME.fullmatch(name) is None:
        raise ValueError(f"unsafe or unsupported array name {name!r}")
    return name


def _validate_container_identity(manifest: Mapping[str, object]) -> None:
    if manifest.get("container_schema") != V5_CHECKED_ARRAY_ARTIFACT_SCHEMA:
        raise ValueError("unsupported checked-array artifact schema")
    if manifest.get("container_version") != V5_CHECKED_ARRAY_ARTIFACT_VERSION:
        raise ValueError("unsupported checked-array artifact version")


def _canonical_array(value: object, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.hasobject:
        raise TypeError(f"{name} cannot use object/pickle dtype")
    if array.dtype.kind in "fc" and not np.all(np.isfinite(array)):
        raise ValueError(f"{name} cannot contain NaN or infinity")
    if array.dtype.byteorder == ">" or (
        array.dtype.byteorder == "=" and not np.little_endian and array.dtype.itemsize > 1
    ):
        array = array.astype(array.dtype.newbyteorder("<"), copy=False)
    result = np.ascontiguousarray(array)
    result.setflags(write=False)
    return result


def array_sha256(name: str, array: np.ndarray) -> str:
    """Hash an array together with its semantic name, dtype, and shape."""

    selected = _validated_name(name)
    value = _canonical_array(array, selected)
    digest = sha256()
    digest.update(selected.encode("utf-8"))
    digest.update(b"\0")
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(canonical_json(list(value.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def freeze_named_arrays(arrays: Mapping[str, object]) -> Mapping[str, np.ndarray]:
    """Return a sorted, immutable, pickle-free array mapping."""

    if not isinstance(arrays, Mapping) or not arrays:
        raise ValueError("arrays must be a non-empty mapping")
    result = {}
    for raw_name, value in arrays.items():
        name = _validated_name(raw_name)
        if name in result:  # pragma: no cover - mappings cannot normally expose this
            raise ValueError(f"duplicate array name {name!r}")
        result[name] = _canonical_array(value, name)
    return MappingProxyType(dict(sorted(result.items())))


def array_manifest(arrays: Mapping[str, np.ndarray]) -> dict[str, dict[str, object]]:
    values = freeze_named_arrays(arrays)
    return {
        name: {
            "dtype": value.dtype.str,
            "shape": list(value.shape),
            "sha256": array_sha256(name, value),
        }
        for name, value in values.items()
    }


def validate_array_manifest(
    arrays: Mapping[str, np.ndarray],
    expected: Mapping[str, object],
) -> Mapping[str, np.ndarray]:
    """Fail closed if names, metadata, or any array byte changed."""

    values = freeze_named_arrays(arrays)
    if not isinstance(expected, Mapping) or set(expected) != set(values):
        raise ValueError("artifact array inventory is incomplete or unsupported")
    observed = array_manifest(values)
    if dict(expected) != observed:
        raise ValueError("artifact array metadata or SHA-256 does not reproduce")
    return values


@dataclass(frozen=True)
class V5ArtifactReceipt:
    path: Path
    artifact_sha256: str
    byte_count: int
    manifest_sha256: str


def _zip_member(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=_ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = 0o100600 << 16
    return info


def _write_container(
    stream,
    manifest_json: str,
    arrays: Mapping[str, np.ndarray],
) -> None:
    """Stream one deterministic ZIP without duplicating a whole shard in RAM."""

    with zipfile.ZipFile(
        stream,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
        strict_timestamps=True,
    ) as archive:
        archive.writestr(_zip_member("manifest.json"), manifest_json.encode("utf-8"))
        for name, array in arrays.items():
            with archive.open(_zip_member(f"arrays/{name}.npy"), "w", force_zip64=True) as member:
                np.lib.format.write_array(member, array, allow_pickle=False)


def _file_sha256(path: Path) -> tuple[str, int]:
    digest = sha256()
    byte_count = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
            byte_count += len(chunk)
    return digest.hexdigest(), byte_count


def write_checked_array_artifact(
    path: str | os.PathLike[str],
    *,
    manifest: Mapping[str, object],
    arrays: Mapping[str, object],
) -> V5ArtifactReceipt:
    """Publish one deterministic container atomically without overwriting."""

    target = Path(path)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {target}")
    if not target.parent.is_dir():
        raise FileNotFoundError(f"artifact parent directory does not exist: {target.parent}")
    frozen = freeze_named_arrays(arrays)
    payload = dict(manifest)
    _validate_container_identity(payload)
    manifest_hash = payload.get("manifest_sha256")
    if not isinstance(manifest_hash, str) or _SHA256.fullmatch(manifest_hash) is None:
        raise ValueError("manifest must carry its lowercase manifest_sha256")
    core = dict(payload)
    core.pop("manifest_sha256")
    if sha256(canonical_json(core).encode("utf-8")).hexdigest() != manifest_hash:
        raise ValueError("manifest_sha256 does not reproduce")
    validate_array_manifest(frozen, payload.get("arrays", {}))
    encoded_manifest = canonical_json(payload) + "\n"

    descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w+b") as stream:
            _write_container(stream, encoded_manifest, frozen)
            stream.flush()
            os.fsync(stream.fileno())
        artifact_hash, byte_count = _file_sha256(temporary)
        try:
            os.link(temporary, target)
        except FileExistsError:
            raise FileExistsError(f"refusing to overwrite existing artifact: {target}") from None
        directory_fd = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
    return V5ArtifactReceipt(target, artifact_hash, byte_count, manifest_hash)


def read_checked_array_artifact(
    path: str | os.PathLike[str],
) -> tuple[dict[str, object], Mapping[str, np.ndarray], V5ArtifactReceipt]:
    """Read and verify a checked container without enabling pickle loading."""

    source = Path(path)
    artifact_hash, byte_count = _file_sha256(source)
    try:
        with zipfile.ZipFile(source, mode="r") as archive:
            names = archive.namelist()
            if len(names) != len(set(names)) or "manifest.json" not in names:
                raise ValueError("artifact has duplicate members or no manifest")
            encoded_manifest = archive.read("manifest.json").decode("utf-8")
            manifest = _strict_json_object(encoded_manifest)
            _validate_container_identity(manifest)
            expected_arrays = manifest.get("arrays")
            if not isinstance(expected_arrays, Mapping):
                raise ValueError("artifact manifest has no array inventory")
            expected_members = {"manifest.json"} | {
                f"arrays/{_validated_name(name)}.npy" for name in expected_arrays
            }
            if set(names) != expected_members:
                raise ValueError("artifact ZIP members are incomplete or unsupported")
            arrays = {
                name: np.load(BytesIO(archive.read(f"arrays/{name}.npy")), allow_pickle=False)
                for name in expected_arrays
            }
    except (OSError, UnicodeError, zipfile.BadZipFile) as exc:
        raise ValueError("artifact container is corrupt") from exc

    manifest_hash = manifest.get("manifest_sha256")
    if not isinstance(manifest_hash, str) or _SHA256.fullmatch(manifest_hash) is None:
        raise ValueError("artifact manifest SHA-256 is missing or malformed")
    core = dict(manifest)
    core.pop("manifest_sha256")
    if sha256(canonical_json(core).encode("utf-8")).hexdigest() != manifest_hash:
        raise ValueError("artifact manifest SHA-256 does not reproduce")
    frozen = validate_array_manifest(arrays, expected_arrays)
    return (
        manifest,
        frozen,
        V5ArtifactReceipt(source, artifact_hash, byte_count, manifest_hash),
    )


__all__ = [
    "V5_CHECKED_ARRAY_ARTIFACT_SCHEMA",
    "V5_CHECKED_ARRAY_ARTIFACT_VERSION",
    "V5ArtifactReceipt",
    "array_manifest",
    "array_sha256",
    "canonical_json",
    "freeze_named_arrays",
    "read_checked_array_artifact",
    "validate_array_manifest",
    "write_checked_array_artifact",
]
