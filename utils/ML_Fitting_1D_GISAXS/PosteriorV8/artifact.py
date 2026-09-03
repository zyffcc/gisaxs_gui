"""Fail-closed, TensorFlow-free artifact manifest for Posterior V8.

The manifest binds a serialized model to the exact scientific, topology, and
preprocessing contracts that produced it.  Validation happens before any
machine-learning runtime is imported.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
import json
from numbers import Real
import os
from pathlib import Path
import re
import tempfile
from types import MappingProxyType
from typing import Any, Mapping

from .contract import CODEC_VERSION, CONTRACT_VERSION, FORWARD_MODEL_VERSION, TOPOLOGIES
from .preprocessing import DEFAULT_CONTRACT, PREPROCESSING_VERSION


ARTIFACT_SCHEMA = "gisaxs.posterior_v8.artifact/v1"
MODEL_FAMILY = "gisaxs_posterior_v8"
TOPOLOGY_CATALOG_VERSION = "posterior_v8_topology_catalog_v1"
MODEL_FORMAT = "keras_v3"
DEFAULT_MODEL_FILE = "model.keras"
MANIFEST_FILE = "manifest.json"

_SHA256_PATTERN = re.compile(r"[0-9a-fA-F]{64}\Z")


class ArtifactValidationError(ValueError):
    """The artifact cannot safely be used with the current Posterior V8 code."""


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize a JSON value deterministically for hashing and persistence."""

    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("value is not canonical-JSON serializable") from exc
    return encoded.encode("utf-8")


def _topology_catalog_payload() -> dict[str, list[dict[str, object]]]:
    # List order is part of the contract because it defines every topology ID.
    return {
        "topologies": [
            {"id": topology_id, "shapes": list(shapes)}
            for topology_id, shapes in enumerate(TOPOLOGIES)
        ]
    }


TOPOLOGY_CATALOG_SHA256 = hashlib.sha256(
    canonical_json_bytes(_topology_catalog_payload())
).hexdigest()


def file_sha256(path: str | os.PathLike[str]) -> str:
    """Return the lowercase SHA-256 of one regular file without loading it all."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"model file does not exist: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _expect_current(name: str, actual: object, expected: object) -> None:
    if isinstance(expected, float):
        valid_type = isinstance(actual, Real) and not isinstance(actual, bool)
        matches = valid_type and float(actual) == expected
    elif isinstance(expected, int):
        matches = isinstance(actual, int) and not isinstance(actual, bool) and actual == expected
    else:
        matches = type(actual) is type(expected) and actual == expected
    if not matches:
        raise ArtifactValidationError(
            f"manifest field {name!r} is incompatible: expected {expected!r}, got {actual!r}"
        )


@dataclass(frozen=True, kw_only=True)
class ArtifactContract:
    """Immutable manifest payload for one serialized Posterior V8 model."""

    model_file_sha256: str
    model_file: str = DEFAULT_MODEL_FILE
    artifact_schema: str = ARTIFACT_SCHEMA
    model_family: str = MODEL_FAMILY
    contract_version: str = CONTRACT_VERSION
    codec_version: str = CODEC_VERSION
    forward_model_version: str = FORWARD_MODEL_VERSION
    preprocessing_version: str = PREPROCESSING_VERSION
    topology_catalog_version: str = TOPOLOGY_CATALOG_VERSION
    topology_catalog_sha256: str = TOPOLOGY_CATALOG_SHA256
    q_unit: str = DEFAULT_CONTRACT.q_unit
    q_min: float = DEFAULT_CONTRACT.q_min
    q_max: float = DEFAULT_CONTRACT.q_max
    max_points: int = DEFAULT_CONTRACT.max_points
    model_format: str = MODEL_FORMAT

    def __post_init__(self) -> None:
        expected_values = {
            "artifact_schema": ARTIFACT_SCHEMA,
            "model_family": MODEL_FAMILY,
            "contract_version": CONTRACT_VERSION,
            "codec_version": CODEC_VERSION,
            "forward_model_version": FORWARD_MODEL_VERSION,
            "preprocessing_version": PREPROCESSING_VERSION,
            "topology_catalog_version": TOPOLOGY_CATALOG_VERSION,
            "topology_catalog_sha256": TOPOLOGY_CATALOG_SHA256,
            "q_unit": DEFAULT_CONTRACT.q_unit,
            "q_min": DEFAULT_CONTRACT.q_min,
            "q_max": DEFAULT_CONTRACT.q_max,
            "max_points": DEFAULT_CONTRACT.max_points,
            "model_format": MODEL_FORMAT,
        }
        for name, expected in expected_values.items():
            _expect_current(name, getattr(self, name), expected)

        if (
            not isinstance(self.model_file, str)
            or not self.model_file
            or self.model_file in {".", ".."}
            or "/" in self.model_file
            or "\\" in self.model_file
            or "\x00" in self.model_file
        ):
            raise ArtifactValidationError("model_file must be one safe basename")
        if not isinstance(self.model_file_sha256, str) or not _SHA256_PATTERN.fullmatch(
            self.model_file_sha256
        ):
            raise ArtifactValidationError("model_file_sha256 must contain exactly 64 hex digits")
        object.__setattr__(self, "model_file_sha256", self.model_file_sha256.lower())

    @classmethod
    def from_model_file(
        cls,
        model_file: str | os.PathLike[str],
        *,
        model_format: str = MODEL_FORMAT,
    ) -> "ArtifactContract":
        """Construct the current contract and bind it to an existing model file."""

        path = Path(model_file)
        if not path.is_file():
            raise FileNotFoundError(f"model file does not exist: {path}")
        return cls(
            model_file=path.name,
            model_file_sha256=file_sha256(path),
            model_format=model_format,
        )

    @classmethod
    def from_manifest_payload(cls, payload: Mapping[str, object]) -> "ArtifactContract":
        """Parse all required fields and reject stale or incompatible contracts."""

        if not isinstance(payload, Mapping):
            raise ArtifactValidationError("artifact manifest must contain a JSON object")
        required = tuple(cls.__dataclass_fields__)
        missing = [name for name in required if name not in payload]
        if missing:
            raise ArtifactValidationError(
                "artifact manifest is missing required fields: " + ", ".join(missing)
            )
        unexpected = sorted(set(payload) - set(required))
        if unexpected:
            raise ArtifactValidationError(
                "artifact manifest contains unsupported fields: " + ", ".join(unexpected)
            )
        return cls(**{name: payload[name] for name in required})

    def manifest_payload(self) -> Mapping[str, object]:
        """Return a read-only, flat JSON payload in stable field order."""

        return MappingProxyType(
            {name: getattr(self, name) for name in self.__dataclass_fields__}
        )

    def to_dict(self) -> dict[str, object]:
        """Return an independent JSON-compatible dictionary."""

        return dict(self.manifest_payload())


def _coerce_contract(
    value: ArtifactContract | Mapping[str, object],
) -> ArtifactContract:
    if isinstance(value, ArtifactContract):
        return value
    if isinstance(value, Mapping):
        return ArtifactContract.from_manifest_payload(value)
    raise TypeError("manifest must be an ArtifactContract or mapping payload")


def write_manifest_atomic(
    path: str | os.PathLike[str],
    manifest: ArtifactContract | Mapping[str, object],
    *,
    overwrite: bool = False,
) -> Path:
    """Atomically write one manifest path, refusing replacement by default."""

    if not isinstance(overwrite, bool):
        raise TypeError("overwrite must be boolean")
    target = Path(path)
    parent = target.parent
    if not parent.is_dir():
        raise FileNotFoundError(f"manifest parent directory does not exist: {parent}")
    if not overwrite and (target.exists() or target.is_symlink()):
        raise FileExistsError(f"refusing to overwrite existing manifest: {target}")

    contract = _coerce_contract(manifest)
    serialized = canonical_json_bytes(contract.to_dict()) + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(
        dir=str(parent), prefix=f".{target.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temporary, target)
        else:
            try:
                os.link(temporary, target)
            except FileExistsError as exc:
                raise FileExistsError(
                    f"refusing to overwrite existing manifest: {target}"
                ) from exc
            temporary.unlink()
    finally:
        if temporary.exists():
            temporary.unlink()
    return target


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ArtifactValidationError(f"duplicate manifest field: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ArtifactValidationError(f"non-finite JSON value is forbidden: {value}")


def validate_manifest(model_dir: str | os.PathLike[str]) -> ArtifactContract:
    """Validate one complete artifact directory without importing TensorFlow."""

    root = Path(model_dir)
    if not root.is_dir():
        raise ArtifactValidationError(f"model directory does not exist: {root}")
    manifest_path = root / MANIFEST_FILE
    if manifest_path.is_symlink():
        raise ArtifactValidationError(
            f"artifact manifest must not be a symbolic link: {manifest_path}"
        )
    if not manifest_path.is_file():
        raise ArtifactValidationError(f"artifact manifest is missing: {manifest_path}")
    try:
        payload = json.loads(
            manifest_path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except ArtifactValidationError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ArtifactValidationError(f"invalid artifact manifest {manifest_path}: {exc}") from exc

    contract = ArtifactContract.from_manifest_payload(payload)
    model_path = root / contract.model_file
    if model_path.is_symlink():
        raise ArtifactValidationError(f"model file must not be a symbolic link: {model_path}")
    if not model_path.is_file():
        raise ArtifactValidationError(f"model file is missing: {model_path}")
    actual_sha256 = file_sha256(model_path)
    if not hmac.compare_digest(actual_sha256, contract.model_file_sha256):
        raise ArtifactValidationError(
            f"model checksum mismatch for {contract.model_file}: "
            f"expected {contract.model_file_sha256}, got {actual_sha256}"
        )
    return contract


__all__ = [
    "ARTIFACT_SCHEMA",
    "DEFAULT_MODEL_FILE",
    "MANIFEST_FILE",
    "MODEL_FAMILY",
    "MODEL_FORMAT",
    "TOPOLOGY_CATALOG_SHA256",
    "TOPOLOGY_CATALOG_VERSION",
    "ArtifactContract",
    "ArtifactValidationError",
    "canonical_json_bytes",
    "file_sha256",
    "validate_manifest",
    "write_manifest_atomic",
]
