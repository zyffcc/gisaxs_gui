"""Build or compare the finite V5.2 Sobol cross-platform manifest.

This command performs no curve simulation or model training.  It is suitable
for a CPU worker and writes new files exclusively.  Slurm integration is kept
outside this minimal scientific gate.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Sequence

from .sobol_cross_platform_manifest_v5 import (
    V5SobolCrossPlatformManifest,
    build_v5_sobol_cross_platform_manifest,
    compare_v5_sobol_cross_platform_manifests,
)


_MAX_MANIFEST_BYTES = 16 * 1024 * 1024
_PASS_MARKER_SCHEMA = "gisaxs.posterior_v8.sobol_cross_platform_pass_marker/v1"
_PASS_MARKER_VERSION = "posterior_v8_atomic_exclusive_pass_after_exact_compare_v1"
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_SOURCE_FIELDS = {
    "source_archive_sha256",
    "source_manifest_sha256",
    "source_tree_sha256",
}


def _read_manifest(path: Path) -> V5SobolCrossPlatformManifest:
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"manifest path is not a regular file: {resolved}")
    size = resolved.stat().st_size
    if size <= 0 or size > _MAX_MANIFEST_BYTES:
        raise ValueError("manifest file size is empty or exceeds the 16 MiB gate limit")
    return V5SobolCrossPlatformManifest.from_json(
        resolved.read_text(encoding="utf-8")
    )


def _require_new_paths(*paths: Path) -> None:
    requested = [path.expanduser() for path in paths]
    existing = [str(path) for path in requested if path.exists() or path.is_symlink()]
    if existing:
        raise FileExistsError(f"output path already exists: {existing[0]}")
    resolved = [path.resolve() for path in requested]
    if len(set(resolved)) != len(resolved):
        raise ValueError("output paths must be distinct")


def _exclusive_atomic_write(path: Path, text: str) -> None:
    requested = path.expanduser()
    if requested.exists() or requested.is_symlink():
        raise FileExistsError(f"output path already exists: {requested}")
    target = requested.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"output path already exists: {target}")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        # A hard link publishes the complete inode atomically and, unlike
        # replace(), fails if another process won the destination race.
        os.link(temporary, target)
        directory_fd = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _source_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source-archive-sha256", required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--source-tree-sha256", required=True)


def _build_from_args(args: argparse.Namespace) -> V5SobolCrossPlatformManifest:
    return build_v5_sobol_cross_platform_manifest(
        source_archive_sha256=args.source_archive_sha256,
        source_manifest_sha256=args.source_manifest_sha256,
        source_tree_sha256=args.source_tree_sha256,
    )


def _result_text(result) -> str:
    encoded = result.to_json()
    if type(result).from_json(encoded) != result:
        raise RuntimeError("generated gate result did not pass strict canonical replay")
    return encoded


def _strict_json_object(encoded: str, name: str) -> dict[str, object]:
    def reject_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate {name} field {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str):
        raise ValueError(f"non-finite JSON constant {value!r} is forbidden")

    try:
        payload = json.loads(
            encoded,
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} is not strict JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must contain one JSON object")
    return payload


def validate_v5_sobol_cross_platform_pass_marker(encoded: str) -> dict[str, object]:
    payload = _strict_json_object(encoded, "cross-platform PASS marker")
    expected_fields = {
        "schema",
        "version",
        "status",
        "comparison_result_sha256",
        "reference_manifest_sha256",
        "candidate_manifest_sha256",
        "scientific_content_sha256",
        "source",
        "marker_sha256",
    }
    if set(payload) != expected_fields:
        raise ValueError("cross-platform PASS marker fields are incomplete or unsupported")
    if (
        payload["schema"] != _PASS_MARKER_SCHEMA
        or payload["version"] != _PASS_MARKER_VERSION
        or payload["status"] != "PASS"
    ):
        raise ValueError("unsupported cross-platform PASS marker identity")
    source = payload["source"]
    if not isinstance(source, dict) or set(source) != _SOURCE_FIELDS:
        raise ValueError("cross-platform PASS marker source identity is incomplete")
    for name, value in {
        "comparison_result_sha256": payload["comparison_result_sha256"],
        "reference_manifest_sha256": payload["reference_manifest_sha256"],
        "candidate_manifest_sha256": payload["candidate_manifest_sha256"],
        "scientific_content_sha256": payload["scientific_content_sha256"],
        **source,
    }.items():
        if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
            raise ValueError(f"PASS marker {name} must be a lowercase SHA-256")
    core = dict(payload)
    marker_digest = core.pop("marker_sha256")
    canonical = json.dumps(core, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if marker_digest != sha256(canonical.encode("utf-8")).hexdigest():
        raise ValueError("cross-platform PASS marker SHA-256 does not reproduce")
    expected = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if encoded != expected:
        raise ValueError("cross-platform PASS marker JSON is not canonical")
    return payload


def _pass_marker_text(result) -> str:
    result_payload = result.audit_payload()
    core = {
        "schema": _PASS_MARKER_SCHEMA,
        "version": _PASS_MARKER_VERSION,
        "status": "PASS",
        "comparison_result_sha256": result_payload["result_sha256"],
        "reference_manifest_sha256": result.reference_manifest_sha256,
        "candidate_manifest_sha256": result.candidate_manifest_sha256,
        "scientific_content_sha256": result.reference_scientific_content_sha256,
        "source": result.reference_source_payload,
    }
    canonical = json.dumps(
        core, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    payload = {
        **core,
        "marker_sha256": sha256(canonical.encode("utf-8")).hexdigest(),
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    validate_v5_sobol_cross_platform_pass_marker(encoded)
    return encoded


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    build = commands.add_parser("build", help="build one immutable scientific manifest")
    _source_arguments(build)
    build.add_argument("--output", required=True, type=Path)

    compare = commands.add_parser("compare", help="strictly compare two existing manifests")
    compare.add_argument("--reference", required=True, type=Path)
    compare.add_argument("--candidate", required=True, type=Path)
    compare.add_argument("--result-output", required=True, type=Path)
    compare.add_argument("--pass-marker", required=True, type=Path)

    gate = commands.add_parser(
        "build-and-compare",
        help="build a candidate from the live source and compare it to a reference",
    )
    _source_arguments(gate)
    gate.add_argument("--reference", required=True, type=Path)
    gate.add_argument("--candidate-output", required=True, type=Path)
    gate.add_argument("--result-output", required=True, type=Path)
    gate.add_argument("--pass-marker", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "build":
        _require_new_paths(args.output)
        manifest = _build_from_args(args)
        _exclusive_atomic_write(args.output, manifest.to_json())
        print(
            json.dumps(
                {
                    "status": "manifest_built",
                    "manifest_sha256": manifest.sha256,
                    "scientific_content_sha256": manifest.layer_sha256[
                        "scientific_content_sha256"
                    ],
                    "output": str(args.output.expanduser().resolve()),
                },
                sort_keys=True,
                allow_nan=False,
            )
        )
        return 0

    if args.command == "compare":
        _require_new_paths(args.result_output, args.pass_marker)
    else:
        _require_new_paths(
            args.candidate_output,
            args.result_output,
            args.pass_marker,
        )
    reference = _read_manifest(args.reference)
    if args.command == "compare":
        candidate = _read_manifest(args.candidate)
    else:
        candidate = _build_from_args(args)
        _exclusive_atomic_write(args.candidate_output, candidate.to_json())
    result = compare_v5_sobol_cross_platform_manifests(reference, candidate)
    encoded = _result_text(result)
    _exclusive_atomic_write(args.result_output, encoded)
    if result.passed:
        _exclusive_atomic_write(args.pass_marker, _pass_marker_text(result))
    print(encoded, end="")
    return 0 if result.passed else 1


if __name__ == "__main__":  # pragma: no cover - exercised through CLI
    raise SystemExit(main())


__all__ = ["main", "validate_v5_sobol_cross_platform_pass_marker"]
