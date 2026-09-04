"""Completion-last dataset binding for the K1 Phase-A Slurm chain."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
from typing import Mapping, Sequence

from .grouped_dataset_v5 import read_v5_grouped_dataset
from .k1_phase_a_capability_v7 import (
    V7PhaseAInputCapability,
    _consumed_capability_payload,
    _validate_phase_a_capability,
)
from .k1_phase_a_contract_v7 import validate_launch_binding_payload
from .k1_phase_a_cross_platform_v5 import (
    file_identity,
    validate_v5_k1_cross_platform_marker_file,
)
from .sobol_cross_platform_manifest_v5 import (
    V5_SOBOL_CROSS_PLATFORM_MANIFEST_SCHEMA,
    V5_SOBOL_CROSS_PLATFORM_MANIFEST_VERSION,
)
from .k1_staging_files_v5 import read_only_bytes_identity


V5_K1_PHASE_A_DATASET_BINDING_SCHEMA = (
    "gisaxs.posterior_v8.k1_phase_a_dataset_completion_binding/v2"
)
V5_K1_PHASE_A_DATASET_BINDING_VERSION = (
    "posterior_v8_dataset_source_gate_launch_and_job_capability_binding_v2"
)
_SOURCE_FIELDS = {
    "source_archive_sha256",
    "source_manifest_sha256",
    "source_tree_sha256",
}
_DATASET_FIELDS = {
    "original_path",
    "artifact_sha256",
    "byte_count",
    "dataset_id",
    "dataset_schema",
    "dataset_version",
    "manifest_sha256",
}
_GATE_FIELDS = {
    "reference_file_sha256",
    "reference_file_byte_count",
    "reference_manifest_schema",
    "reference_manifest_version",
    "reference_manifest_sha256",
    "scientific_content_sha256",
    "comparison_result_sha256",
    "gate_claim_sha256",
    "pass_marker_sha256",
    "pass_marker_file_sha256",
    "pass_marker_byte_count",
}
_TOP_LEVEL_FIELDS = {
    "schema",
    "version",
    "status",
    "dataset",
    "source",
    "cross_platform_gate",
    "launch_binding",
    "job_local_capability",
    "binding_sha256",
}


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _strict_json_object(encoded: str) -> dict[str, object]:
    def reject_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate dataset-binding field {key!r}")
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
        raise ValueError("dataset completion binding is not strict JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("dataset completion binding must contain one JSON object")
    return payload


def _exact_fields(value: object, expected: set[str], name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError(f"{name} fields are incomplete or unsupported")
    return value


def _exclusive_atomic_write(path: Path, encoded: str) -> None:
    target = Path(os.path.abspath(path.expanduser()))
    if not target.parent.is_dir() or target.parent.is_symlink():
        raise FileNotFoundError("dataset-binding output parent must be a real directory")
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"refusing to overwrite dataset completion binding: {target}")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent, prefix=f".{target.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(0o400)
        try:
            os.link(temporary, target)
        except FileExistsError:
            raise FileExistsError(
                f"refusing to overwrite dataset completion binding: {target}"
            ) from None
        directory_fd = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _marker_expected(expected: Mapping[str, object]) -> dict[str, object]:
    names = {
        "expected_source",
        "reference_file_sha256",
        "reference_file_byte_count",
        "reference_manifest_sha256",
        "scientific_content_sha256",
        "comparison_result_sha256",
        "expected_gate_claim_sha256",
    }
    if set(expected) != names:
        raise ValueError("marker expectation fields are incomplete or unsupported")
    return dict(expected)


def _dataset_payload(path: Path, *, original_path: str) -> dict[str, object]:
    identity = file_identity(path, name="K1 dataset", require_read_only=True)
    dataset, receipt = read_v5_grouped_dataset(Path(str(identity["path"])))
    after = file_identity(
        Path(str(identity["path"])), name="K1 dataset", require_read_only=True
    )
    if after != identity:
        raise RuntimeError("K1 dataset changed while its manifest was read")
    if (
        receipt.artifact_sha256 != identity["sha256"]
        or receipt.byte_count != identity["byte_count"]
        or receipt.manifest_sha256 != dataset.manifest["manifest_sha256"]
    ):
        raise RuntimeError("K1 dataset file and checked-artifact receipt disagree")
    return {
        "original_path": original_path,
        "artifact_sha256": receipt.artifact_sha256,
        "byte_count": receipt.byte_count,
        "dataset_id": dataset.manifest["dataset_id"],
        "dataset_schema": dataset.manifest["dataset_schema"],
        "dataset_version": dataset.manifest["dataset_version"],
        "manifest_sha256": receipt.manifest_sha256,
    }


def _gate_payload(marker_result: Mapping[str, object], expected: Mapping[str, object]) -> dict[str, object]:
    marker = marker_result["marker"]
    marker_file = marker_result["file"]
    return {
        "reference_file_sha256": expected["reference_file_sha256"],
        "reference_file_byte_count": expected["reference_file_byte_count"],
        "reference_manifest_schema": V5_SOBOL_CROSS_PLATFORM_MANIFEST_SCHEMA,
        "reference_manifest_version": V5_SOBOL_CROSS_PLATFORM_MANIFEST_VERSION,
        "reference_manifest_sha256": expected["reference_manifest_sha256"],
        "scientific_content_sha256": expected["scientific_content_sha256"],
        "comparison_result_sha256": expected["comparison_result_sha256"],
        "gate_claim_sha256": expected["expected_gate_claim_sha256"],
        "pass_marker_sha256": marker["marker_sha256"],
        "pass_marker_file_sha256": marker_file["sha256"],
        "pass_marker_byte_count": marker_file["byte_count"],
    }


def build_v5_k1_phase_a_dataset_binding(
    dataset_path: Path,
    marker_path: Path,
    *,
    original_dataset_path: str,
    marker_expected: Mapping[str, object],
    launch_binding: Mapping[str, object],
    capability: V7PhaseAInputCapability,
) -> dict[str, object]:
    if not isinstance(launch_binding, Mapping):
        raise ValueError("dataset binding requires a validated Phase-A v7 launch binding")
    launch_binding = validate_launch_binding_payload(launch_binding)
    if launch_binding["stage"] not in {"smoke_dataset", "full_dataset"}:
        raise ValueError("dataset binding requires a validated Phase-A v7 launch binding")
    _validate_phase_a_capability(
        capability, stage=str(launch_binding["stage"]), phase="pre_use"
    )
    expected = _marker_expected(marker_expected)
    marker = validate_v5_k1_cross_platform_marker_file(marker_path, **expected)
    source = dict(expected["expected_source"])
    core = {
        "schema": V5_K1_PHASE_A_DATASET_BINDING_SCHEMA,
        "version": V5_K1_PHASE_A_DATASET_BINDING_VERSION,
        "status": "COMPLETE",
        "dataset": _dataset_payload(dataset_path, original_path=original_dataset_path),
        "source": source,
        "cross_platform_gate": _gate_payload(marker, expected),
        "launch_binding": dict(launch_binding),
    }
    _validate_phase_a_capability(
        capability, stage=str(launch_binding["stage"]), phase="post_use"
    )
    core["job_local_capability"] = _consumed_capability_payload(capability)
    return {**core, "binding_sha256": sha256(_canonical_json(core).encode()).hexdigest()}


def publish_v5_k1_phase_a_dataset_binding(
    dataset_path: Path,
    marker_path: Path,
    binding_path: Path,
    *,
    original_dataset_path: str,
    marker_expected: Mapping[str, object],
    launch_binding: Mapping[str, object],
    capability: V7PhaseAInputCapability,
) -> dict[str, object]:
    if binding_path.exists() or binding_path.is_symlink():
        raise FileExistsError("refusing to overwrite dataset completion binding")
    payload = build_v5_k1_phase_a_dataset_binding(
        dataset_path,
        marker_path,
        original_dataset_path=original_dataset_path,
        marker_expected=marker_expected,
        launch_binding=launch_binding,
        capability=capability,
    )
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    _exclusive_atomic_write(binding_path, encoded)
    validate_v5_k1_phase_a_dataset_binding_file(
        binding_path,
        dataset_path=dataset_path,
        marker_path=marker_path,
        expected_original_dataset_path=original_dataset_path,
        marker_expected=marker_expected,
        expected_launch_binding=launch_binding,
    )
    return payload


def validate_v5_k1_phase_a_dataset_binding(
    encoded: str,
    *,
    dataset_path: Path,
    marker_path: Path,
    expected_original_dataset_path: str,
    marker_expected: Mapping[str, object],
    expected_launch_binding: Mapping[str, object] | None = None,
) -> dict[str, object]:
    payload = _strict_json_object(encoded)
    _exact_fields(payload, _TOP_LEVEL_FIELDS, "dataset completion binding")
    if (
        payload["schema"] != V5_K1_PHASE_A_DATASET_BINDING_SCHEMA
        or payload["version"] != V5_K1_PHASE_A_DATASET_BINDING_VERSION
        or payload["status"] != "COMPLETE"
    ):
        raise ValueError("unsupported dataset completion binding identity")
    dataset = _exact_fields(payload["dataset"], _DATASET_FIELDS, "dataset binding")
    source = _exact_fields(payload["source"], _SOURCE_FIELDS, "dataset source binding")
    gate = _exact_fields(payload["cross_platform_gate"], _GATE_FIELDS, "gate binding")
    core = dict(payload)
    supplied_sha = core.pop("binding_sha256")
    if supplied_sha != sha256(_canonical_json(core).encode()).hexdigest():
        raise ValueError("dataset completion binding SHA-256 does not reproduce")
    expected = _marker_expected(marker_expected)
    marker = validate_v5_k1_cross_platform_marker_file(marker_path, **expected)
    if source != expected["expected_source"]:
        raise ValueError("dataset completion binding source identity does not match")
    if gate != _gate_payload(marker, expected):
        raise ValueError("dataset completion binding gate identity does not match")
    if dataset["original_path"] != expected_original_dataset_path:
        raise ValueError("dataset completion binding original path does not match")
    dataset_launch = validate_launch_binding_payload(payload["launch_binding"])
    if dataset_launch["stage"] not in {"smoke_dataset", "full_dataset"}:
        raise ValueError("dataset completion binding has the wrong producer stage")
    if expected_launch_binding is not None and dataset_launch != expected_launch_binding:
        raise ValueError("dataset completion binding launch identity does not match")
    capability_payload = payload["job_local_capability"]
    if (
        not isinstance(capability_payload, Mapping)
        or capability_payload.get("pre_post_equal") is not True
        or capability_payload.get("capability", {}).get("launch_binding_sha256")
        != dataset_launch.get("binding_sha256")
    ):
        raise ValueError("dataset completion binding capability identity does not match")
    actual_dataset = _dataset_payload(
        dataset_path, original_path=expected_original_dataset_path
    )
    if dataset != actual_dataset:
        raise ValueError("dataset completion binding does not match the consumed dataset")
    canonical = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if encoded != canonical:
        raise ValueError("dataset completion binding JSON is not canonical")
    return payload


def validate_v5_k1_phase_a_dataset_binding_file(
    binding_path: Path, **expected: object
) -> dict[str, object]:
    raw, _ = read_only_bytes_identity(binding_path, "K1 dataset completion binding")
    encoded = raw.decode("utf-8")
    return validate_v5_k1_phase_a_dataset_binding(encoded, **expected)


def _marker_arguments(parser: argparse.ArgumentParser) -> None:
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
        "expected_source": {
            "source_archive_sha256": args.source_archive_sha256,
            "source_manifest_sha256": args.source_manifest_sha256,
            "source_tree_sha256": args.source_tree_sha256,
        },
        "reference_file_sha256": args.reference_file_sha256,
        "reference_file_byte_count": args.reference_file_byte_count,
        "reference_manifest_sha256": args.reference_manifest_sha256,
        "scientific_content_sha256": args.scientific_content_sha256,
        "comparison_result_sha256": args.comparison_result_sha256,
        "expected_gate_claim_sha256": args.expected_gate_claim_sha256,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    publish = commands.add_parser("publish")
    validate = commands.add_parser("validate")
    for command in (publish, validate):
        command.add_argument("--dataset", required=True, type=Path)
        command.add_argument("--marker", required=True, type=Path)
        command.add_argument("--binding", required=True, type=Path)
        command.add_argument("--original-dataset-path", required=True)
        _marker_arguments(command)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "validate":
        validate_v5_k1_phase_a_dataset_binding_file(
            args.binding,
            dataset_path=args.dataset,
            marker_path=args.marker,
            expected_original_dataset_path=args.original_dataset_path,
            marker_expected=_expected_from_args(args),
        )
        return 0
    raise RuntimeError(
        "Phase-A v7 dataset publication requires the process-live pinned-wrapper runtime"
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "V5_K1_PHASE_A_DATASET_BINDING_SCHEMA",
    "V5_K1_PHASE_A_DATASET_BINDING_VERSION",
    "build_v5_k1_phase_a_dataset_binding",
    "main",
    "publish_v5_k1_phase_a_dataset_binding",
    "validate_v5_k1_phase_a_dataset_binding",
    "validate_v5_k1_phase_a_dataset_binding_file",
]
