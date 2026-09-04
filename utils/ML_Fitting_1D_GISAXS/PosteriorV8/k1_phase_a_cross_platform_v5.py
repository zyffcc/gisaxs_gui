"""Strict Phase-A binding for the V5 Sobol cross-platform PASS marker.

The scientific manifest/gate owns the numerical comparison.  This module
binds that result to the immutable source snapshot and to the exact external
reference file selected by the Phase-A launch plan.  It intentionally has no
Slurm or training responsibilities.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import stat
from typing import Mapping, Sequence

from .k1_staging_files_v5 import read_only_bytes_identity

from .run_sobol_cross_platform_gate_v5 import (
    validate_v5_sobol_cross_platform_pass_marker,
)
from .sobol_cross_platform_manifest_v5 import (
    V5_SOBOL_CROSS_PLATFORM_MANIFEST_SCHEMA,
    V5_SOBOL_CROSS_PLATFORM_MANIFEST_VERSION,
    V5SobolCrossPlatformManifest,
    compare_v5_sobol_cross_platform_manifests,
)


V5_K1_CROSS_PLATFORM_GATE_CLAIM_SCHEMA = (
    "gisaxs.posterior_v8.k1_phase_a_cross_platform_gate_claim/v1"
)
V5_K1_CROSS_PLATFORM_GATE_CLAIM_VERSION = (
    "posterior_v8_exact_external_reference_source_and_pass_marker_binding_v1"
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_WRITE_BITS = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
_SOURCE_FIELDS = (
    "source_archive_sha256",
    "source_manifest_sha256",
    "source_tree_sha256",
)


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _exact_nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _source_payload(value: Mapping[str, object]) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != set(_SOURCE_FIELDS):
        raise ValueError("source identity fields are incomplete or unsupported")
    return {name: _digest(value[name], name) for name in _SOURCE_FIELDS}


def _assert_regular_no_symlink(path: Path, name: str) -> Path:
    lexical = Path(os.path.abspath(path.expanduser()))
    current = Path(lexical.anchor)
    for component in lexical.parts[1:]:
        current /= component
        if current.is_symlink():
            raise ValueError(f"{name} must not traverse a symlink: {current}")
    resolved = lexical.resolve(strict=True)
    if not resolved.is_file() or resolved.is_symlink():
        raise ValueError(f"{name} must be a regular non-symlink file")
    return resolved


def file_identity(path: Path, *, name: str, require_read_only: bool) -> dict[str, object]:
    """Return an exact regular-file identity without following symlink components."""

    if require_read_only:
        _, strict = read_only_bytes_identity(path, name)
        return {
            "path": strict["path"],
            "sha256": strict["sha256"],
            "byte_count": strict["byte_count"],
            "mode": int(str(strict["mode_octal"]), 8),
            "device": strict["device"],
            "inode": strict["inode"],
            "mtime_ns": strict["mtime_ns"],
            "ctime_ns": strict["ctime_ns"],
            "nlink": strict["link_count"],
        }
    resolved = _assert_regular_no_symlink(path, name)
    if not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeError("strict file identity requires O_NOFOLLOW support")
    path_before = os.stat(resolved, follow_symlinks=False)
    if not stat.S_ISREG(path_before.st_mode):
        raise ValueError(f"{name} must remain a regular non-symlink file")
    flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(resolved, flags)
    except OSError as exc:
        raise ValueError(f"{name} changed or became a symlink before open") from exc
    try:
        status_before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(status_before.st_mode)
            or (status_before.st_dev, status_before.st_ino)
            != (path_before.st_dev, path_before.st_ino)
        ):
            raise RuntimeError(f"{name} was replaced while it was opened")
        if require_read_only and stat.S_IMODE(status_before.st_mode) & _WRITE_BITS:
            raise ValueError(f"{name} must be read-only")
        file_digest = sha256()
        byte_count = 0
        while chunk := os.read(descriptor, 1024 * 1024):
            file_digest.update(chunk)
            byte_count += len(chunk)
        status_after = os.fstat(descriptor)
        try:
            path_after = os.stat(resolved, follow_symlinks=False)
        except OSError as exc:
            raise RuntimeError(f"{name} path changed while it was hashed") from exc
        before = (
            status_before.st_dev,
            status_before.st_ino,
            status_before.st_mode,
            status_before.st_size,
            status_before.st_mtime_ns,
            status_before.st_ctime_ns,
            status_before.st_nlink,
        )
        after = (
            status_after.st_dev,
            status_after.st_ino,
            status_after.st_mode,
            status_after.st_size,
            status_after.st_mtime_ns,
            status_after.st_ctime_ns,
            status_after.st_nlink,
        )
        path_final = (
            path_after.st_dev,
            path_after.st_ino,
            path_after.st_mode,
            path_after.st_size,
            path_after.st_mtime_ns,
            path_after.st_ctime_ns,
            path_after.st_nlink,
        )
        if (
            before != after
            or after != path_final
            or not stat.S_ISREG(path_after.st_mode)
            or byte_count != status_after.st_size
        ):
            raise RuntimeError(f"{name} changed while it was hashed")
    finally:
        os.close(descriptor)
    return {
        "path": str(resolved),
        "sha256": file_digest.hexdigest(),
        "byte_count": byte_count,
        "mode": stat.S_IMODE(status_after.st_mode),
        "device": status_after.st_dev,
        "inode": status_after.st_ino,
        "mtime_ns": status_after.st_mtime_ns,
    }


def _legacy_identity(strict: Mapping[str, object]) -> dict[str, object]:
    return {
        "path": strict["path"],
        "sha256": strict["sha256"],
        "byte_count": strict["byte_count"],
        "mode": int(str(strict["mode_octal"]), 8),
        "device": strict["device"],
        "inode": strict["inode"],
        "mtime_ns": strict["mtime_ns"],
        "ctime_ns": strict["ctime_ns"],
        "nlink": strict["link_count"],
    }


def _gate_claim_core(
    *,
    source: Mapping[str, object],
    reference_file_sha256: str,
    reference_file_byte_count: int,
    reference_manifest_sha256: str,
    scientific_content_sha256: str,
    comparison_result_sha256: str,
) -> dict[str, object]:
    return {
        "schema": V5_K1_CROSS_PLATFORM_GATE_CLAIM_SCHEMA,
        "version": V5_K1_CROSS_PLATFORM_GATE_CLAIM_VERSION,
        "expected_status": "PASS",
        "source": _source_payload(source),
        "reference_manifest_schema": V5_SOBOL_CROSS_PLATFORM_MANIFEST_SCHEMA,
        "reference_manifest_version": V5_SOBOL_CROSS_PLATFORM_MANIFEST_VERSION,
        "reference_file_sha256": _digest(
            reference_file_sha256, "reference_file_sha256"
        ),
        "reference_file_byte_count": _exact_nonnegative_integer(
            reference_file_byte_count, "reference_file_byte_count"
        ),
        "reference_manifest_sha256": _digest(
            reference_manifest_sha256, "reference_manifest_sha256"
        ),
        "candidate_manifest_sha256": _digest(
            reference_manifest_sha256, "reference_manifest_sha256"
        ),
        "scientific_content_sha256": _digest(
            scientific_content_sha256, "scientific_content_sha256"
        ),
        "comparison_result_sha256": _digest(
            comparison_result_sha256, "comparison_result_sha256"
        ),
    }


def gate_claim_sha256(**values: object) -> str:
    """Hash the complete deterministic prerequisite claim."""

    return sha256(_canonical_json(_gate_claim_core(**values)).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class V5K1CrossPlatformReferenceBinding:
    path: str
    file_sha256: str
    file_byte_count: int
    manifest_sha256: str
    manifest_schema: str
    manifest_version: str
    scientific_content_sha256: str
    comparison_result_sha256: str
    source_archive_sha256: str
    source_manifest_sha256: str
    source_tree_sha256: str
    gate_claim_sha256: str

    @property
    def source(self) -> dict[str, str]:
        return {name: getattr(self, name) for name in _SOURCE_FIELDS}

    def audit_payload(self) -> dict[str, object]:
        return {
            "path": self.path,
            "file_sha256": self.file_sha256,
            "file_byte_count": self.file_byte_count,
            "manifest_sha256": self.manifest_sha256,
            "manifest_schema": self.manifest_schema,
            "manifest_version": self.manifest_version,
            "scientific_content_sha256": self.scientific_content_sha256,
            "comparison_result_sha256": self.comparison_result_sha256,
            "source": self.source,
            "gate_claim_schema": V5_K1_CROSS_PLATFORM_GATE_CLAIM_SCHEMA,
            "gate_claim_version": V5_K1_CROSS_PLATFORM_GATE_CLAIM_VERSION,
            "gate_claim_sha256": self.gate_claim_sha256,
        }


def inspect_v5_k1_cross_platform_reference(
    path: Path,
    *,
    expected_file_sha256: str,
    expected_source: Mapping[str, object],
    require_read_only: bool = True,
) -> tuple[V5K1CrossPlatformReferenceBinding, dict[str, object]]:
    """Strictly read one reference and derive its frozen Phase-A claim."""

    expected_digest = _digest(expected_file_sha256, "expected reference file SHA-256")
    raw, strict_identity = read_only_bytes_identity(path, "cross-platform reference")
    identity = _legacy_identity(strict_identity)
    if identity["sha256"] != expected_digest:
        raise ValueError("cross-platform reference SHA-256 does not match the frozen digest")
    encoded = raw.decode("utf-8")
    after = file_identity(
        Path(str(identity["path"])),
        name="cross-platform reference",
        require_read_only=require_read_only,
    )
    if after != identity:
        raise RuntimeError("cross-platform reference changed while it was parsed")
    manifest = V5SobolCrossPlatformManifest.from_json(encoded)
    source = _source_payload(expected_source)
    if manifest.payload["source"] != source:
        raise ValueError("cross-platform reference source identity does not match the snapshot")
    comparison = compare_v5_sobol_cross_platform_manifests(manifest, manifest)
    if not comparison.passed:
        raise RuntimeError("cross-platform reference did not compare equal to itself")
    comparison_sha = str(comparison.audit_payload()["result_sha256"])
    scientific_sha = str(manifest.layer_sha256["scientific_content_sha256"])
    claim = gate_claim_sha256(
        source=source,
        reference_file_sha256=expected_digest,
        reference_file_byte_count=int(identity["byte_count"]),
        reference_manifest_sha256=manifest.sha256,
        scientific_content_sha256=scientific_sha,
        comparison_result_sha256=comparison_sha,
    )
    binding = V5K1CrossPlatformReferenceBinding(
        path=str(identity["path"]),
        file_sha256=expected_digest,
        file_byte_count=int(identity["byte_count"]),
        manifest_sha256=manifest.sha256,
        manifest_schema=V5_SOBOL_CROSS_PLATFORM_MANIFEST_SCHEMA,
        manifest_version=V5_SOBOL_CROSS_PLATFORM_MANIFEST_VERSION,
        scientific_content_sha256=scientific_sha,
        comparison_result_sha256=comparison_sha,
        source_archive_sha256=source["source_archive_sha256"],
        source_manifest_sha256=source["source_manifest_sha256"],
        source_tree_sha256=source["source_tree_sha256"],
        gate_claim_sha256=claim,
    )
    return binding, identity


def validate_v5_k1_cross_platform_marker(
    encoded: str,
    *,
    expected_source: Mapping[str, object],
    reference_file_sha256: str,
    reference_file_byte_count: int,
    reference_manifest_sha256: str,
    scientific_content_sha256: str,
    comparison_result_sha256: str,
    expected_gate_claim_sha256: str,
) -> dict[str, object]:
    """Validate a PASS marker against every launch-plan-bound identity."""

    marker = validate_v5_sobol_cross_platform_pass_marker(encoded)
    source = _source_payload(expected_source)
    if marker["source"] != source:
        raise ValueError("cross-platform PASS marker source identity does not match")
    expected_manifest = _digest(reference_manifest_sha256, "reference_manifest_sha256")
    if (
        marker["reference_manifest_sha256"] != expected_manifest
        or marker["candidate_manifest_sha256"] != expected_manifest
    ):
        raise ValueError("cross-platform PASS marker manifest identity does not match")
    if marker["scientific_content_sha256"] != _digest(
        scientific_content_sha256, "scientific_content_sha256"
    ):
        raise ValueError("cross-platform PASS marker scientific identity does not match")
    if marker["comparison_result_sha256"] != _digest(
        comparison_result_sha256, "comparison_result_sha256"
    ):
        raise ValueError("cross-platform PASS marker comparison identity does not match")
    claim = gate_claim_sha256(
        source=source,
        reference_file_sha256=reference_file_sha256,
        reference_file_byte_count=reference_file_byte_count,
        reference_manifest_sha256=expected_manifest,
        scientific_content_sha256=scientific_content_sha256,
        comparison_result_sha256=comparison_result_sha256,
    )
    if claim != _digest(expected_gate_claim_sha256, "expected_gate_claim_sha256"):
        raise ValueError("cross-platform PASS marker gate claim does not match")
    return {**marker, "gate_claim_sha256": claim}


def validate_v5_k1_cross_platform_marker_file(path: Path, **expected: object) -> dict[str, object]:
    raw, strict_identity = read_only_bytes_identity(path, "cross-platform PASS marker")
    identity = _legacy_identity(strict_identity)
    encoded = raw.decode("utf-8")
    after = file_identity(
        Path(str(identity["path"])),
        name="cross-platform PASS marker",
        require_read_only=True,
    )
    if after != identity:
        raise RuntimeError("cross-platform PASS marker changed while it was parsed")
    return {
        "marker": validate_v5_k1_cross_platform_marker(encoded, **expected),
        "file": identity,
    }


def _expected_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source-archive-sha256", required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--source-tree-sha256", required=True)
    parser.add_argument("--reference-file-sha256", required=True)
    parser.add_argument("--reference-file-byte-count", required=True, type=int)
    parser.add_argument("--reference-manifest-sha256", required=True)
    parser.add_argument("--scientific-content-sha256", required=True)
    parser.add_argument("--comparison-result-sha256", required=True)
    parser.add_argument("--expected-gate-claim-sha256", required=True)


def _expected_from_args(args: argparse.Namespace) -> dict[str, object]:
    return {
        "expected_source": {name: getattr(args, name) for name in _SOURCE_FIELDS},
        "reference_file_sha256": args.reference_file_sha256,
        "reference_file_byte_count": args.reference_file_byte_count,
        "reference_manifest_sha256": args.reference_manifest_sha256,
        "scientific_content_sha256": args.scientific_content_sha256,
        "comparison_result_sha256": args.comparison_result_sha256,
        "expected_gate_claim_sha256": args.expected_gate_claim_sha256,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--marker", required=True, type=Path)
    _expected_arguments(parser)
    args = parser.parse_args(argv)
    result = validate_v5_k1_cross_platform_marker_file(
        args.marker, **_expected_from_args(args)
    )
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "V5K1CrossPlatformReferenceBinding",
    "V5_K1_CROSS_PLATFORM_GATE_CLAIM_SCHEMA",
    "V5_K1_CROSS_PLATFORM_GATE_CLAIM_VERSION",
    "file_identity",
    "gate_claim_sha256",
    "inspect_v5_k1_cross_platform_reference",
    "main",
    "validate_v5_k1_cross_platform_marker",
    "validate_v5_k1_cross_platform_marker_file",
]
