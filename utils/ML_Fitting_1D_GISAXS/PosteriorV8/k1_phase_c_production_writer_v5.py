"""Completion-last writer for lossless K1 Phase-C replay snapshots."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import os
from pathlib import Path
import stat

import numpy as np

from .evaluation import CandidateInput
from .grouped_artifact_v5 import canonical_json
from .k1_phase_c_filesystem_replay_v5 import _load_filesystem_snapshot
from .k1_phase_c_raw_codecs_v5 import (
    V5_K1_PHASE_C_RAW_ARTIFACT_SHA_SEMANTICS,
    V5_K1_PHASE_C_RAW_MANIFEST_SCHEMA,
    V5_K1_PHASE_C_RAW_MANIFEST_VERSION,
)
from .k1_phase_c_replay_runner_v5 import derive_v5_k1_phase_c_parent_record
from .k1_phase_c_writer_contract_v5 import (
    V5K1PhaseCWriterSnapshot,
    revalidate_v5_k1_phase_c_writer_sources,
)
from .k1_phase_c_writer_payloads_v5 import write_v5_k1_phase_c_raw_files
from .read_only_json_publication_v5 import publish_read_only_canonical_json


V5_K1_PHASE_C_WRITER_RECEIPT_SCHEMA = "gisaxs.posterior_v8.k1_phase_c_lossless_writer_receipt/v1"
V5_K1_PHASE_C_WRITER_RECEIPT_VERSION = "live_typed_sources_atomic_read_only_completion_last_v1"
V5_K1_PHASE_C_WRITER_RECEIPT_FILENAME = "writer-completion.json"
V5_K1_PHASE_C_MANIFEST_FILENAME = "manifest.json"


@dataclass(frozen=True, kw_only=True)
class V5K1PhaseCWriterResult:
    output_root: Path
    manifest_path: Path
    manifest_file_sha256: str
    manifest_sha256: str
    bundle_sha256: str
    receipt_path: Path
    receipt_file_sha256: str
    receipt_sha256: str
    production_eligible: bool


def _validate_representative_round_trip(source, restored) -> None:
    rebound = replace(source, source_artifact_sha256=restored.source_artifact_sha256)
    if rebound.audit_payload() != restored.audit_payload() or rebound.sha256 != restored.sha256:
        raise RuntimeError("writer changed a typed representative payload")
    if isinstance(source.parameter, CandidateInput) and not np.array_equal(
        source.parameter.exact_intensity,
        restored.parameter.exact_intensity,
    ):
        raise RuntimeError("writer changed an exact intensity vector")


def _validate_method_trace_round_trip(source, restored) -> None:
    scalar_fields = (
        "query_id",
        "pairing_unit_id",
        "method_id",
        "method_protocol_id",
        "method_protocol_sha256",
        "trace_id",
        "reference_set_id",
        "comparison_protocol_id",
        "comparison_protocol_sha256",
        "exact_forward_call_budget",
    )
    if any(getattr(source, name) != getattr(restored, name) for name in scalar_fields):
        raise RuntimeError("writer changed an exact method trace identity")
    if [asdict(value) for value in source.exact_forward_calls] != [
        asdict(value) for value in restored.exact_forward_calls
    ]:
        raise RuntimeError("writer changed an exact-forward call ledger")
    if len(source.candidate_emissions) != len(restored.candidate_emissions):
        raise RuntimeError("writer lost an emitted candidate")
    emission_fields = (
        "available_after_call",
        "output_rank",
        "candidate_id",
        "compatibility_status",
        "elapsed_seconds",
    )
    for original, replayed in zip(source.candidate_emissions, restored.candidate_emissions):
        if any(getattr(original, name) != getattr(replayed, name) for name in emission_fields):
            raise RuntimeError("writer changed an emitted-candidate trace row")
        _validate_representative_round_trip(original.payload, replayed.payload)


def _matcher_query_contracts(matcher) -> dict[str, tuple[tuple[object, ...], ...]]:
    contexts: dict[str, set[tuple[object, ...]]] = {}
    for (context_sha, _), task in matcher._tasks.items():
        query = task.universal_context.topology_queries[0]
        contexts.setdefault(context_sha, set()).add(
            (
                query.topology_id,
                query.sha256,
                query.geometry.canonical_json,
                query.amplitude.canonical_json,
            )
        )
    return {context_sha: tuple(sorted(values)) for context_sha, values in contexts.items()}


def _validate_lossless_typed_replay(snapshot, loaded) -> None:
    source_bundle = snapshot.bundle
    restored_bundle = loaded.bundle
    if (
        source_bundle.artifact_binding != restored_bundle.artifact_binding
        or source_bundle.evaluator_config != restored_bundle.evaluator_config
        or replace(
            source_bundle.split_receipt,
            artifact_sha256=restored_bundle.split_receipt.artifact_sha256,
        )
        != restored_bundle.split_receipt
        or len(source_bundle.parents) != len(restored_bundle.parents)
    ):
        raise RuntimeError("writer changed shared typed replay evidence")

    expected_queries: dict[str, tuple[tuple[object, ...], ...]] = {}
    for parent_input, source, restored in zip(
        snapshot.parents, source_bundle.parents, restored_bundle.parents
    ):
        if (
            source.provenance != restored.provenance
            or source.product_candidate_judgements != restored.product_candidate_judgements
            or len(source.branch_searches) != len(restored.branch_searches)
            or len(source.methods) != len(restored.methods)
        ):
            raise RuntimeError("writer changed parent replay evidence")
        for original, replayed in zip(source.branch_searches, restored.branch_searches):
            if (
                replace(
                    original,
                    search_sidecar_sha256=replayed.search_sidecar_sha256,
                )
                != replayed
            ):
                raise RuntimeError("writer changed a branch-search sidecar")

        normalized_bank = replace(
            source.reference_bank,
            bank_artifact_sha256=restored.reference_bank.bank_artifact_sha256,
            search_trace_artifact_sha256=(restored.reference_bank.search_trace_artifact_sha256),
            representative_payload_sha256s=(restored.reference_bank.representative_payload_sha256s),
        )
        if normalized_bank.audit_payload() != restored.reference_bank.audit_payload() or [
            asdict(value) for value in source.reference_bank.exact_forward_calls
        ] != [asdict(value) for value in restored.reference_bank.exact_forward_calls]:
            raise RuntimeError("writer changed contextual reference-bank evidence")

        source_reference = source.reference_set
        restored_reference = restored.reference_set
        reference_fields = (
            "query_id",
            "pairing_unit_id",
            "reference_set_id",
            "comparison_protocol_id",
            "comparison_protocol_sha256",
        )
        if (
            any(
                getattr(source_reference, name) != getattr(restored_reference, name)
                for name in reference_fields
            )
            or len(source_reference.representatives) != len(restored_reference.representatives)
            or source_reference.reference_set_sha256 != source.reference_bank.bank_artifact_sha256
            or restored_reference.reference_set_sha256
            != restored.reference_bank.bank_artifact_sha256
        ):
            raise RuntimeError("writer changed a frozen reference set")
        for original, replayed in zip(
            source_reference.representatives,
            restored_reference.representatives,
        ):
            if original.representative_id != replayed.representative_id:
                raise RuntimeError("writer changed a reference representative ID")
            _validate_representative_round_trip(original.payload, replayed.payload)

        method_fields = (
            "status",
            "source_bundle_sha256_used",
            "model_artifact_sha256_used",
            "proposal_execution_policy_sha256_used",
        )
        for original, replayed in zip(source.methods, restored.methods):
            if any(
                getattr(original, name) != getattr(replayed, name) for name in method_fields
            ) or (
                original.trace.reference_set_sha256 != source_reference.reference_set_sha256
                or replayed.trace.reference_set_sha256 != restored_reference.reference_set_sha256
            ):
                raise RuntimeError("writer changed method provenance")
            _validate_method_trace_round_trip(original.trace, replayed.trace)

        context_sha = source.provenance.evaluation_query_context_sha256
        query_contracts = tuple(
            sorted(
                (
                    query.topology_id,
                    query.sha256,
                    query.geometry.canonical_json,
                    query.amplitude.canonical_json,
                )
                for query in parent_input.topology_queries
            )
        )
        previous = expected_queries.setdefault(context_sha, query_contracts)
        if previous != query_contracts:
            raise RuntimeError("one query context was assigned conflicting bounds")
    if _matcher_query_contracts(loaded.matcher) != expected_queries:
        raise RuntimeError("writer changed a query/bounds distance context")


def _upstream_inventory(snapshot: V5K1PhaseCWriterSnapshot) -> list[dict[str, object]]:
    rows = [
        {
            "scope": "global",
            "parent_sha256": None,
            "role": value.role,
            "file_sha256": value.file_sha256,
        }
        for value in snapshot.global_upstream_files
    ]
    for parent in snapshot.parents:
        rows.extend(
            {
                "scope": "parent",
                "parent_sha256": parent.evidence.provenance.clean_parent_sha256,
                "role": value.role,
                "file_sha256": value.file_sha256,
            }
            for value in parent.upstream_files
        )
    rows.sort(
        key=lambda value: (
            str(value["scope"]),
            str(value["parent_sha256"]),
            str(value["role"]),
            str(value["file_sha256"]),
        )
    )
    if len(rows) != len(
        {
            (
                value["scope"],
                value["parent_sha256"],
                value["role"],
                value["file_sha256"],
            )
            for value in rows
        }
    ):
        raise ValueError("upstream provenance inventory contains duplicate bindings")
    return rows


def _inventory_sha256(value: object) -> str:
    return sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _make_directories_read_only(root: Path) -> None:
    directories = [value for value in root.rglob("*") if value.is_dir()]
    for directory in sorted(directories, key=lambda value: len(value.parts), reverse=True):
        os.chmod(directory, 0o500, follow_symlinks=False)
    os.chmod(root, 0o500, follow_symlinks=False)


def _validate_publication_modes(root: Path) -> None:
    for path in root.rglob("*"):
        metadata = path.stat(follow_symlinks=False)
        if path.is_symlink():
            raise RuntimeError("writer published a symbolic link")
        if stat.S_ISREG(metadata.st_mode):
            if metadata.st_nlink != 1 or stat.S_IMODE(metadata.st_mode) != 0o400:
                raise RuntimeError("writer file is writable or multiply linked")
        elif stat.S_ISDIR(metadata.st_mode):
            if stat.S_IMODE(metadata.st_mode) != 0o500:
                raise RuntimeError("writer directory is not read-only")
        else:
            raise RuntimeError("writer output contains an unsupported filesystem object")


def _write_snapshot(
    snapshot: V5K1PhaseCWriterSnapshot,
    output_root: str | os.PathLike[str],
) -> V5K1PhaseCWriterResult:
    if type(snapshot) is not V5K1PhaseCWriterSnapshot:
        raise TypeError("snapshot must use the exact K1 Phase-C writer type")
    root = Path(output_root)
    if root.exists() or root.is_symlink():
        raise FileExistsError("refusing to reuse a K1 Phase-C writer output root")
    if not root.parent.is_dir() or root.parent.is_symlink():
        raise ValueError("writer output parent must be an existing non-symlink directory")
    revalidate_v5_k1_phase_c_writer_sources(snapshot)
    root.mkdir(mode=0o700)
    written = write_v5_k1_phase_c_raw_files(snapshot, root)
    manifest_core = {
        "schema": V5_K1_PHASE_C_RAW_MANIFEST_SCHEMA,
        "version": V5_K1_PHASE_C_RAW_MANIFEST_VERSION,
        "status": "complete",
        "formal": snapshot.plan.formal,
        "artifact_sha256_semantics": V5_K1_PHASE_C_RAW_ARTIFACT_SHA_SEMANTICS,
        "plan_sha256": snapshot.plan.sha256,
        "contract_sha256": snapshot.contract["contract_sha256"],
        "artifact_binding_file_id": "artifact-binding",
        "proposal_execution_policy_file_id": "proposal-policy",
        "split_receipt_file_id": "split-receipt",
        "evaluator_config_file_id": "evaluator-config",
        "files": list(written.file_rows),
        "parents": list(written.parent_rows),
    }
    manifest_sha = _inventory_sha256(manifest_core)
    manifest = {**manifest_core, "manifest_sha256": manifest_sha}
    revalidate_v5_k1_phase_c_writer_sources(snapshot)
    manifest_path = root / V5_K1_PHASE_C_MANIFEST_FILENAME
    manifest_file_sha = publish_read_only_canonical_json(manifest_path, manifest)

    loaded = _load_filesystem_snapshot(
        manifest_path,
        expected_manifest_file_sha256=manifest_file_sha,
        plan=snapshot.plan,
        contract=snapshot.contract,
    )
    _validate_lossless_typed_replay(snapshot, loaded)
    for parent in loaded.bundle.parents:
        derive_v5_k1_phase_c_parent_record(
            parent,
            split_id=loaded.bundle.split_receipt.split_id,
            split_receipt_sha256=loaded.bundle.split_receipt.artifact_sha256,
            evaluator_config=loaded.bundle.evaluator_config,
            equivalence_distance_matcher=loaded.matcher,
        )
    raw_inventory = [dict(value) for value in written.file_rows]
    upstream = _upstream_inventory(snapshot)
    receipt_core = {
        "schema": V5_K1_PHASE_C_WRITER_RECEIPT_SCHEMA,
        "version": V5_K1_PHASE_C_WRITER_RECEIPT_VERSION,
        "status": "complete",
        "production_eligible": snapshot.production_verified,
        "formal": snapshot.plan.formal,
        "plan_sha256": snapshot.plan.sha256,
        "contract_sha256": snapshot.contract["contract_sha256"],
        "manifest_relative_path": V5_K1_PHASE_C_MANIFEST_FILENAME,
        "manifest_file_sha256": manifest_file_sha,
        "manifest_sha256": manifest_sha,
        "raw_file_count": len(raw_inventory),
        "raw_file_inventory_sha256": _inventory_sha256(raw_inventory),
        "parent_count": len(snapshot.parents),
        "output_bundle_sha256": loaded.bundle.sha256,
        "matcher_identity_sha256": loaded.matcher.identity_sha256,
        "upstream_file_count": len(upstream),
        "upstream_provenance": upstream,
        "upstream_provenance_sha256": _inventory_sha256(upstream),
        "writer_policy": {
            "typed_live_inputs_only_for_production": True,
            "audit_summaries_are_not_lossless_inputs": True,
            "canonical_json_without_trailing_bytes": True,
            "exclusive_non_overwrite_atomic_publication": True,
            "regular_files_mode": "0400",
            "directories_mode_after_completion": "0500",
            "unique_link_count_required": 1,
            "exact_emitted_intensity_dtype": "<f8",
            "exact_emitted_intensity_finite_and_positive": True,
            "receipt_written_after_manifest_and_full_typed_replay": True,
        },
    }
    receipt_sha = _inventory_sha256(receipt_core)
    revalidate_v5_k1_phase_c_writer_sources(snapshot)
    receipt_path = root / V5_K1_PHASE_C_WRITER_RECEIPT_FILENAME
    receipt_file_sha = publish_read_only_canonical_json(
        receipt_path, {**receipt_core, "receipt_sha256": receipt_sha}
    )
    _make_directories_read_only(root)
    _validate_publication_modes(root)
    return V5K1PhaseCWriterResult(
        output_root=root.resolve(strict=True),
        manifest_path=manifest_path.resolve(strict=True),
        manifest_file_sha256=manifest_file_sha,
        manifest_sha256=manifest_sha,
        bundle_sha256=loaded.bundle.sha256,
        receipt_path=receipt_path.resolve(strict=True),
        receipt_file_sha256=receipt_file_sha,
        receipt_sha256=receipt_sha,
        production_eligible=snapshot.production_verified,
    )


def write_v5_k1_phase_c_lossless_snapshot(
    snapshot: V5K1PhaseCWriterSnapshot,
    output_root: str | os.PathLike[str],
) -> V5K1PhaseCWriterResult:
    """Write a snapshot; non-production captures remain permanently non-claiming."""

    return _write_snapshot(snapshot, output_root)


def write_v5_k1_phase_c_production_snapshot(
    snapshot: V5K1PhaseCWriterSnapshot,
    output_root: str | os.PathLike[str],
) -> V5K1PhaseCWriterResult:
    """Write only a complete, live-source-verified frozen formal population."""

    if type(snapshot) is not V5K1PhaseCWriterSnapshot or not snapshot.production_verified:
        raise ValueError("production writer refuses partial, fixture, or aggregate-only input")
    return _write_snapshot(snapshot, output_root)


__all__ = [
    "V5K1PhaseCWriterResult",
    "V5_K1_PHASE_C_MANIFEST_FILENAME",
    "V5_K1_PHASE_C_WRITER_RECEIPT_FILENAME",
    "V5_K1_PHASE_C_WRITER_RECEIPT_SCHEMA",
    "V5_K1_PHASE_C_WRITER_RECEIPT_VERSION",
    "write_v5_k1_phase_c_lossless_snapshot",
    "write_v5_k1_phase_c_production_snapshot",
]
