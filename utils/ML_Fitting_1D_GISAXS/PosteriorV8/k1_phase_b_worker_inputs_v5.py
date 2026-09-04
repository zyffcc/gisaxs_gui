"""Strict original/local input identities for the audited K1 Phase-B worker."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from numbers import Integral
import os
from pathlib import Path
from typing import Mapping

from .k1_phase_a_cross_platform_v5 import file_identity as strict_file_identity
from .k1_phase_b_contract_v5 import digest
from .package_source_snapshot_v5 import verify_extracted_source_snapshot
from .run_k1_memorization_gate_v5 import (
    MODEL_FILENAME as PHASE_A_MODEL_FILENAME,
    MODEL_PROVENANCE_FILENAME as PHASE_A_MODEL_PROVENANCE_FILENAME,
    RESULT_FILENAME as PHASE_A_RESULT_FILENAME,
)


AUTHORITATIVE_INPUT_KEYS = (
    "source_archive",
    "dataset",
    "dataset_binding",
    "cross_platform_pass_marker",
    "phase_a_result",
    "phase_a_model",
    "phase_a_model_provenance",
    "phase_a_completion",
)
_SOURCE_SNAPSHOT_LOCATION_FIELDS = {"archive_path", "source_root"}


@dataclass(frozen=True, kw_only=True)
class V5K1PhaseBWorkerInputSpec:
    allowed_root: Path
    input_root: Path
    original_source_root: Path
    job_source_root: Path
    phase_a_output_dir: Path
    authoritative_paths: Mapping[str, Path]
    local_paths: Mapping[str, Path]
    expected: Mapping[str, Mapping[str, object]]
    source_manifest_sha256: str
    source_tree_sha256: str


def _below_root(path: Path, root: Path, name: str) -> Path:
    resolved_root = root.expanduser().resolve(strict=True)
    lexical = Path(os.path.abspath(path.expanduser()))
    if lexical == resolved_root or not lexical.is_relative_to(resolved_root):
        raise ValueError(f"{name} must be below {resolved_root}")
    return lexical


def _expected_identity(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != {"sha256", "byte_count"}:
        raise ValueError(f"{name} expected identity is incomplete or unsupported")
    byte_count = value["byte_count"]
    if isinstance(byte_count, bool) or not isinstance(byte_count, Integral) or int(byte_count) < 1:
        raise ValueError(f"{name} expected byte count must be positive")
    return {
        "sha256": digest(value["sha256"], f"{name} expected SHA-256"),
        "byte_count": int(byte_count),
    }


def capture_v5_k1_phase_b_authoritative_inputs(
    *,
    paths: Mapping[str, Path],
    expected: Mapping[str, Mapping[str, object]],
    allowed_root: Path,
    phase_a_output_dir: Path,
) -> dict[str, dict[str, object]]:
    """Fingerprint every authoritative immutable original without trusting a copy."""

    expected_keys = set(AUTHORITATIVE_INPUT_KEYS)
    if not isinstance(paths, Mapping) or set(paths) != expected_keys:
        raise ValueError("authoritative input path inventory is incomplete or unsupported")
    if not isinstance(expected, Mapping) or set(expected) != expected_keys:
        raise ValueError("authoritative expected identity inventory is incomplete or unsupported")

    output = Path(os.path.abspath(phase_a_output_dir.expanduser()))
    root = allowed_root.expanduser().resolve(strict=True)
    if output == root or not output.is_relative_to(root):
        raise ValueError("authoritative Phase-A output must be below the allowed root")
    if not output.is_dir() or output.is_symlink():
        raise ValueError("authoritative Phase-A output must be a real directory")

    identities: dict[str, dict[str, object]] = {}
    for key in AUTHORITATIVE_INPUT_KEYS:
        lexical = _below_root(paths[key], root, f"authoritative {key}")
        identity = strict_file_identity(
            lexical,
            name=f"authoritative {key}",
            require_read_only=True,
        )
        expected_identity = _expected_identity(expected[key], f"authoritative {key}")
        if {
            "sha256": identity["sha256"],
            "byte_count": identity["byte_count"],
        } != expected_identity:
            raise ValueError(f"authoritative {key} does not match its exported identity")
        identities[key] = identity

    exact_phase_a_paths = {
        "dataset_binding": Path(str(identities["dataset"]["path"]) + ".binding-v1.json"),
        "cross_platform_pass_marker": (
            output.parents[1] / "audit/sobol-cross-platform-PASS-v1.json"
        ),
        "phase_a_result": output / PHASE_A_RESULT_FILENAME,
        "phase_a_model": output / PHASE_A_MODEL_FILENAME,
        "phase_a_model_provenance": output / PHASE_A_MODEL_PROVENANCE_FILENAME,
        "phase_a_completion": output.parents[1] / "audit/full-gate-completion-v1.json",
    }
    for key, required_path in exact_phase_a_paths.items():
        if Path(str(identities[key]["path"])) != required_path:
            raise ValueError(f"authoritative {key} escaped the exact Phase-A output")
    return identities


def assert_v5_k1_phase_b_local_copies_match(
    *,
    paths: Mapping[str, Path],
    authoritative: Mapping[str, Mapping[str, object]],
    input_root: Path,
) -> dict[str, dict[str, object]]:
    """Require each read-only job-local copy to match its authoritative original."""

    expected_keys = set(AUTHORITATIVE_INPUT_KEYS)
    if not isinstance(paths, Mapping) or set(paths) != expected_keys:
        raise ValueError("job-local input path inventory is incomplete or unsupported")
    if not isinstance(authoritative, Mapping) or set(authoritative) != expected_keys:
        raise ValueError("authoritative input identity inventory is incomplete or unsupported")
    local: dict[str, dict[str, object]] = {}
    for key in AUTHORITATIVE_INPUT_KEYS:
        lexical = _below_root(paths[key], input_root, f"job-local {key}")
        identity = strict_file_identity(
            lexical,
            name=f"job-local {key}",
            require_read_only=True,
        )
        portable = {
            "sha256": identity["sha256"],
            "byte_count": identity["byte_count"],
        }
        original = authoritative[key]
        if portable != {
            "sha256": original.get("sha256"),
            "byte_count": original.get("byte_count"),
        }:
            raise ValueError(f"job-local {key} does not match its authoritative original")
        local[key] = identity
    return local


def assert_v5_k1_phase_b_authoritative_inputs_unchanged(
    before: Mapping[str, Mapping[str, object]],
    *,
    paths: Mapping[str, Path],
    expected: Mapping[str, Mapping[str, object]],
    allowed_root: Path,
    phase_a_output_dir: Path,
) -> dict[str, dict[str, object]]:
    """Re-fingerprint originals and also reject same-content file replacement."""

    after = capture_v5_k1_phase_b_authoritative_inputs(
        paths=paths,
        expected=expected,
        allowed_root=allowed_root,
        phase_a_output_dir=phase_a_output_dir,
    )
    if after != before:
        raise RuntimeError("authoritative Phase-A/source inputs changed during Phase-B")
    return after


def bind_v5_k1_phase_b_worker_inputs(
    spec: V5K1PhaseBWorkerInputSpec,
) -> dict[str, object]:
    """Bind originals, copies, and both source trees into one exact snapshot."""

    if not isinstance(spec, V5K1PhaseBWorkerInputSpec):
        raise TypeError("spec must be a V5K1PhaseBWorkerInputSpec")
    authoritative = capture_v5_k1_phase_b_authoritative_inputs(
        paths=spec.authoritative_paths,
        expected=spec.expected,
        allowed_root=spec.allowed_root,
        phase_a_output_dir=spec.phase_a_output_dir,
    )
    local = assert_v5_k1_phase_b_local_copies_match(
        paths=spec.local_paths,
        authoritative=authoritative,
        input_root=spec.input_root,
    )
    archive_sha256 = str(authoritative["source_archive"]["sha256"])
    original_snapshot = verify_extracted_source_snapshot(
        _below_root(
            spec.authoritative_paths["source_archive"],
            spec.allowed_root,
            "authoritative source archive",
        ),
        _below_root(
            spec.original_source_root,
            spec.allowed_root,
            "authoritative source root",
        ),
        expected_archive_sha256=archive_sha256,
        expected_manifest_sha256=spec.source_manifest_sha256,
        expected_source_tree_sha256=spec.source_tree_sha256,
    )
    local_snapshot = verify_extracted_source_snapshot(
        _below_root(
            spec.local_paths["source_archive"],
            spec.input_root,
            "job-local source archive",
        ),
        _below_root(
            spec.job_source_root,
            spec.input_root,
            "job-local source root",
        ),
        expected_archive_sha256=archive_sha256,
        expected_manifest_sha256=spec.source_manifest_sha256,
        expected_source_tree_sha256=spec.source_tree_sha256,
    )
    if {
        name: value
        for name, value in local_snapshot.items()
        if name not in _SOURCE_SNAPSHOT_LOCATION_FIELDS
    } != {
        name: value
        for name, value in original_snapshot.items()
        if name not in _SOURCE_SNAPSHOT_LOCATION_FIELDS
    }:
        raise ValueError("job-local source snapshot does not match its authoritative original")
    return {
        "authoritative": authoritative,
        "local": local,
        "source_snapshot": original_snapshot,
    }


def assert_v5_k1_phase_b_worker_inputs_unchanged(
    before: Mapping[str, object], spec: V5K1PhaseBWorkerInputSpec
) -> dict[str, object]:
    """Repeat every original/copy/source check and reject exact-identity drift."""

    after = bind_v5_k1_phase_b_worker_inputs(spec)
    if after != before:
        raise RuntimeError("bound authoritative or job-local inputs changed during Phase-B")
    return after


def add_v5_k1_phase_b_worker_input_arguments(
    parser: argparse.ArgumentParser,
) -> None:
    """Add the mandatory original/copy/source binding CLI arguments."""

    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--source-archive", required=True, type=Path)
    parser.add_argument("--source-archive-sha256", required=True)
    parser.add_argument("--source-archive-byte-count", required=True, type=int)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--source-tree-sha256", required=True)
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--original-source-root", required=True, type=Path)
    parser.add_argument("--original-source-archive", required=True, type=Path)
    parser.add_argument("--original-dataset-path", required=True, type=Path)
    parser.add_argument("--dataset-sha256", required=True)
    parser.add_argument("--dataset-byte-count", required=True, type=int)
    parser.add_argument("--dataset-binding", required=True, type=Path)
    parser.add_argument("--original-dataset-binding", required=True, type=Path)
    parser.add_argument("--dataset-binding-sha256", required=True)
    parser.add_argument("--dataset-binding-byte-count", required=True, type=int)
    parser.add_argument("--cross-platform-pass-marker", required=True, type=Path)
    parser.add_argument("--original-cross-platform-pass-marker", required=True, type=Path)
    parser.add_argument("--cross-platform-pass-marker-sha256", required=True)
    parser.add_argument("--cross-platform-pass-marker-byte-count", required=True, type=int)
    parser.add_argument("--original-phase-a-output-dir", required=True, type=Path)
    parser.add_argument("--original-phase-a-result", required=True, type=Path)
    parser.add_argument("--phase-a-result-sha256", required=True)
    parser.add_argument("--phase-a-result-byte-count", required=True, type=int)
    parser.add_argument("--original-phase-a-model", required=True, type=Path)
    parser.add_argument("--phase-a-model-sha256", required=True)
    parser.add_argument("--phase-a-model-byte-count", required=True, type=int)
    parser.add_argument("--original-phase-a-model-provenance", required=True, type=Path)
    parser.add_argument("--phase-a-model-provenance-sha256", required=True)
    parser.add_argument("--phase-a-model-provenance-byte-count", required=True, type=int)
    parser.add_argument("--phase-a-completion", required=True, type=Path)
    parser.add_argument("--original-phase-a-completion", required=True, type=Path)
    parser.add_argument("--phase-a-completion-sha256", required=True)
    parser.add_argument("--phase-a-completion-byte-count", required=True, type=int)


def v5_k1_phase_b_worker_input_kwargs(args: argparse.Namespace) -> dict[str, object]:
    """Translate parsed binding arguments into the worker's keyword interface."""

    return {
        "source_root": args.source_root,
        "source_archive_path": args.source_archive,
        "source_archive_sha256": args.source_archive_sha256,
        "source_archive_byte_count": args.source_archive_byte_count,
        "source_manifest_sha256": args.source_manifest_sha256,
        "source_tree_sha256": args.source_tree_sha256,
        "original_source_root": args.original_source_root,
        "original_source_archive_path": args.original_source_archive,
        "original_dataset_path": args.original_dataset_path,
        "dataset_sha256": args.dataset_sha256,
        "dataset_byte_count": args.dataset_byte_count,
        "dataset_binding_path": args.dataset_binding,
        "original_dataset_binding_path": args.original_dataset_binding,
        "dataset_binding_sha256": args.dataset_binding_sha256,
        "dataset_binding_byte_count": args.dataset_binding_byte_count,
        "original_cross_platform_pass_marker_path": (
            args.original_cross_platform_pass_marker
        ),
        "cross_platform_pass_marker_sha256": (
            args.cross_platform_pass_marker_sha256
        ),
        "cross_platform_pass_marker_byte_count": (
            args.cross_platform_pass_marker_byte_count
        ),
        "cross_platform_pass_marker_path": args.cross_platform_pass_marker,
        "original_phase_a_output_dir": args.original_phase_a_output_dir,
        "original_phase_a_result_path": args.original_phase_a_result,
        "phase_a_result_sha256": args.phase_a_result_sha256,
        "phase_a_result_byte_count": args.phase_a_result_byte_count,
        "original_phase_a_model_path": args.original_phase_a_model,
        "phase_a_model_sha256": args.phase_a_model_sha256,
        "phase_a_model_byte_count": args.phase_a_model_byte_count,
        "original_phase_a_model_provenance_path": (args.original_phase_a_model_provenance),
        "phase_a_model_provenance_sha256": args.phase_a_model_provenance_sha256,
        "phase_a_model_provenance_byte_count": (args.phase_a_model_provenance_byte_count),
        "phase_a_completion_path": args.phase_a_completion,
        "original_phase_a_completion_path": args.original_phase_a_completion,
        "phase_a_completion_sha256": args.phase_a_completion_sha256,
        "phase_a_completion_byte_count": args.phase_a_completion_byte_count,
        "input_allowed_root": args.input_root,
    }


__all__ = [
    "AUTHORITATIVE_INPUT_KEYS",
    "V5K1PhaseBWorkerInputSpec",
    "add_v5_k1_phase_b_worker_input_arguments",
    "assert_v5_k1_phase_b_authoritative_inputs_unchanged",
    "assert_v5_k1_phase_b_local_copies_match",
    "assert_v5_k1_phase_b_worker_inputs_unchanged",
    "bind_v5_k1_phase_b_worker_inputs",
    "capture_v5_k1_phase_b_authoritative_inputs",
    "v5_k1_phase_b_worker_input_kwargs",
]
