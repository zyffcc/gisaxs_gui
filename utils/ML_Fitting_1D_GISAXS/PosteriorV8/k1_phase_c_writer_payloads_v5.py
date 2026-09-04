"""Encode typed K1 Phase-C inputs into canonical per-file raw payloads."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path

from .k1_phase_c_raw_codecs_v5 import (
    RAW_ARTIFACT_BINDING_SCHEMA,
    RAW_BRANCH_SIDECAR_SCHEMA,
    RAW_DISTANCE_CONTEXT_SCHEMA,
    RAW_EVALUATOR_CONFIG_SCHEMA,
    RAW_METHOD_TRACE_SCHEMA,
    RAW_PARENT_PROVENANCE_SCHEMA,
    RAW_REFERENCE_BANK_SCHEMA,
    RAW_REFERENCE_TRACE_SCHEMA,
    RAW_SPLIT_RECEIPT_SCHEMA,
    V5_K1_PHASE_C_RAW_ARTIFACT_VERSION,
)
from .k1_phase_c_writer_contract_v5 import V5K1PhaseCWriterSnapshot
from .lossless_representative_artifact_v5 import (
    encode_v5_lossless_representative_payload,
)
from .proposal_execution_policy_v5 import V5_PROPOSAL_EXECUTION_POLICY
from .read_only_json_publication_v5 import publish_read_only_canonical_json
from .search_supervision_sidecar_v5 import (
    V5_SEARCH_SIDECAR_SCHEMA,
    V5_SEARCH_SIDECAR_VERSION,
)


@dataclass(frozen=True, kw_only=True)
class V5K1PhaseCWrittenRawFiles:
    file_rows: tuple[dict[str, object], ...]
    parent_rows: tuple[dict[str, object], ...]
    paths: tuple[Path, ...]


@dataclass(frozen=True, kw_only=True)
class _PayloadBinding:
    file_id: str
    payload_sha256: str


class _RawPublisher:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.rows: list[dict[str, object]] = []
        self.paths: list[Path] = []
        self._ids: set[str] = set()
        self._relative_paths: set[str] = set()

    def add(
        self,
        *,
        file_id: str,
        role: str,
        relative_path: str,
        payload: dict[str, object],
    ) -> str:
        if file_id in self._ids or relative_path in self._relative_paths:
            raise ValueError("writer raw file IDs and relative paths must be unique")
        path = self.root.joinpath(*relative_path.split("/"))
        path.parent.mkdir(parents=True, exist_ok=True)
        file_sha = publish_read_only_canonical_json(path, payload)
        self._ids.add(file_id)
        self._relative_paths.add(relative_path)
        self.paths.append(path)
        self.rows.append(
            {
                "file_id": file_id,
                "role": role,
                "relative_path": relative_path,
                "sha256": file_sha,
            }
        )
        return file_sha


def _raw(schema: str, **values: object) -> dict[str, object]:
    return {
        "schema": schema,
        "version": V5_K1_PHASE_C_RAW_ARTIFACT_VERSION,
        **values,
    }


def _publish_representative(
    publisher: _RawPublisher,
    *,
    file_id: str,
    relative_path: str,
    payload,
) -> _PayloadBinding:
    raw = encode_v5_lossless_representative_payload(payload)
    file_sha = publisher.add(
        file_id=file_id,
        role="representative_payload",
        relative_path=relative_path,
        payload=raw,
    )
    rebound = replace(payload, source_artifact_sha256=file_sha)
    return _PayloadBinding(file_id=file_id, payload_sha256=rebound.sha256)


def _distance_payload(parent) -> dict[str, object]:
    evidence = parent.evidence
    return _raw(
        RAW_DISTANCE_CONTEXT_SCHEMA,
        query_context_sha256=evidence.provenance.evaluation_query_context_sha256,
        universal_query_sha256=evidence.provenance.universal_query_sha256,
        topology_queries=[
            {
                "topology_id": query.topology_id,
                "topology_query_sha256": query.sha256,
                "geometry_query_canonical_json": query.geometry.canonical_json,
                "geometry_query_sha256": query.geometry.sha256,
                "amplitude_query_canonical_json": query.amplitude.canonical_json,
                "amplitude_query_sha256": query.amplitude.sha256,
            }
            for query in parent.topology_queries
        ],
    )


def _publish_parent(
    publisher: _RawPublisher,
    *,
    parent_index: int,
    parent,
) -> dict[str, object]:
    evidence = parent.evidence
    provenance = evidence.provenance
    prefix = f"parent-{parent_index:04d}-{provenance.clean_parent_sha256[:16]}"
    file_prefix = f"p{parent_index:04d}"
    provenance_id = f"{file_prefix}-provenance"
    publisher.add(
        file_id=provenance_id,
        role="parent_provenance",
        relative_path=f"parents/{prefix}/provenance.json",
        payload=_raw(RAW_PARENT_PROVENANCE_SCHEMA, provenance=asdict(provenance)),
    )
    distance_id = f"{file_prefix}-distance-context"
    publisher.add(
        file_id=distance_id,
        role="distance_context",
        relative_path=f"parents/{prefix}/distance-context.json",
        payload=_distance_payload(parent),
    )

    branch_ids = []
    for index, branch in enumerate(evidence.branch_searches):
        file_id = f"{file_prefix}-branch-{index:02d}"
        publisher.add(
            file_id=file_id,
            role="branch_search_sidecar",
            relative_path=f"parents/{prefix}/branches/{index:02d}.json",
            payload=_raw(
                RAW_BRANCH_SIDECAR_SCHEMA,
                search_evidence_schema=V5_SEARCH_SIDECAR_SCHEMA,
                search_evidence_version=V5_SEARCH_SIDECAR_VERSION,
                branch_id=branch.branch_id,
                frozen_search_yield_rank=branch.frozen_search_yield_rank,
                completed=branch.completed,
                candidates=[asdict(value) for value in branch.candidates],
            ),
        )
        branch_ids.append(file_id)

    reference_payloads = []
    for index, representative in enumerate(evidence.reference_set.representatives):
        file_id = f"{file_prefix}-reference-payload-{index:04d}"
        binding = _publish_representative(
            publisher,
            file_id=file_id,
            relative_path=f"parents/{prefix}/reference/payload-{index:04d}.json",
            payload=representative.payload,
        )
        reference_payloads.append(
            {
                "representative_id": representative.representative_id,
                "payload_file_id": binding.file_id,
                "payload_sha256": binding.payload_sha256,
            }
        )

    bank = evidence.reference_bank
    reference_bank_id = f"{file_prefix}-reference-bank"
    reference_bank_sha = publisher.add(
        file_id=reference_bank_id,
        role="reference_bank",
        relative_path=f"parents/{prefix}/reference/bank.json",
        payload=_raw(
            RAW_REFERENCE_BANK_SCHEMA,
            reference_bank_schema=bank.schema,
            reference_bank_version=bank.version,
            calibration_identity_sha256=bank.calibration_identity_sha256,
            calibrated_threshold_sha256=bank.calibrated_threshold_sha256,
            exact_judge_sha256=bank.exact_judge_sha256,
            source_bundle_sha256=bank.source_bundle_sha256,
            query_id=bank.query_id,
            pairing_unit_id=bank.pairing_unit_id,
            query_context_sha256=bank.query_context_sha256,
            reference_set_id=evidence.reference_set.reference_set_id,
            comparison_protocol_id=evidence.reference_set.comparison_protocol_id,
            comparison_protocol_sha256=(evidence.reference_set.comparison_protocol_sha256),
            candidate_ids=list(bank.candidate_ids),
            exact_compatible_candidate_ids=list(bank.exact_compatible_candidate_ids),
            representative_clusters=[asdict(value) for value in bank.representative_clusters],
            representative_payloads=reference_payloads,
            distance_schema=bank.distance_schema,
            distance_version=bank.distance_version,
            distance_sha256=bank.distance_sha256,
        ),
    )
    reference_trace_id = f"{file_prefix}-reference-trace"
    publisher.add(
        file_id=reference_trace_id,
        role="reference_search_trace",
        relative_path=f"parents/{prefix}/reference/exact-trace.json",
        payload=_raw(
            RAW_REFERENCE_TRACE_SCHEMA,
            query_id=bank.query_id,
            pairing_unit_id=bank.pairing_unit_id,
            calibration_identity_sha256=bank.calibration_identity_sha256,
            calibrated_threshold_sha256=bank.calibrated_threshold_sha256,
            exact_judge_sha256=bank.exact_judge_sha256,
            source_bundle_sha256=bank.source_bundle_sha256,
            configured_exact_call_budget=bank.configured_exact_call_budget,
            consumed_exact_calls=bank.consumed_exact_calls,
            exact_forward_calls=[asdict(value) for value in bank.exact_forward_calls],
            candidate_judgements=[asdict(value) for value in bank.candidate_judgements],
            enumeration_complete=bank.enumeration_complete,
            network_free=bank.network_free,
        ),
    )

    method_ids = []
    for method_index, method in enumerate(evidence.methods):
        trace = method.trace
        emission_rows = []
        for emission_index, emission in enumerate(trace.candidate_emissions):
            payload_id = f"{file_prefix}-method-{method_index:02d}-payload-{emission_index:04d}"
            binding = _publish_representative(
                publisher,
                file_id=payload_id,
                relative_path=(
                    f"parents/{prefix}/methods/{method_index:02d}/payload-{emission_index:04d}.json"
                ),
                payload=emission.payload,
            )
            emission_rows.append(
                {
                    "available_after_call": emission.available_after_call,
                    "output_rank": emission.output_rank,
                    "candidate_id": emission.candidate_id,
                    "compatibility_status": emission.compatibility_status,
                    "elapsed_seconds": emission.elapsed_seconds,
                    "payload_file_id": binding.file_id,
                    "payload_sha256": binding.payload_sha256,
                }
            )
        method_id = f"{file_prefix}-method-{method_index:02d}-trace"
        publisher.add(
            file_id=method_id,
            role="method_exact_call_trace",
            relative_path=f"parents/{prefix}/methods/{method_index:02d}/trace.json",
            payload=_raw(
                RAW_METHOD_TRACE_SCHEMA,
                status=method.status,
                query_id=trace.query_id,
                pairing_unit_id=trace.pairing_unit_id,
                method_id=trace.method_id,
                method_protocol_id=trace.method_protocol_id,
                method_protocol_sha256=trace.method_protocol_sha256,
                trace_id=trace.trace_id,
                reference_set_id=trace.reference_set_id,
                reference_set_sha256=reference_bank_sha,
                comparison_protocol_id=trace.comparison_protocol_id,
                comparison_protocol_sha256=trace.comparison_protocol_sha256,
                exact_forward_call_budget=trace.exact_forward_call_budget,
                exact_forward_calls=[asdict(value) for value in trace.exact_forward_calls],
                candidate_emissions=emission_rows,
                source_bundle_sha256_used=method.source_bundle_sha256_used,
                model_artifact_sha256_used=method.model_artifact_sha256_used,
                proposal_execution_policy_sha256_used=(
                    method.proposal_execution_policy_sha256_used
                ),
            ),
        )
        method_ids.append(method_id)
    return {
        "clean_parent_sha256": provenance.clean_parent_sha256,
        "provenance_file_id": provenance_id,
        "distance_context_file_id": distance_id,
        "branch_sidecar_file_ids": branch_ids,
        "reference_bank_file_id": reference_bank_id,
        "reference_trace_file_id": reference_trace_id,
        "method_trace_file_ids": method_ids,
    }


def write_v5_k1_phase_c_raw_files(
    snapshot: V5K1PhaseCWriterSnapshot,
    root: Path,
) -> V5K1PhaseCWrittenRawFiles:
    """Write every manifest-addressed file except the manifest itself."""

    publisher = _RawPublisher(root)
    binding = snapshot.bundle.artifact_binding
    source_names = (
        "source_archive_sha256",
        "source_manifest_sha256",
        "source_tree_sha256",
        "source_bundle_sha256",
    )
    model_names = (
        "cross_platform_gate_claim_sha256",
        "phase_a_launch_receipt_sha256",
        "model_artifact_sha256",
        "model_weights_sha256",
        "model_training_result_sha256",
        "proposal_execution_policy_sha256",
    )
    publisher.add(
        file_id="artifact-binding",
        role="artifact_binding",
        relative_path="shared/artifact-binding.json",
        payload=_raw(
            RAW_ARTIFACT_BINDING_SCHEMA,
            artifact_binding=binding.audit_payload(),
            source_provenance={name: getattr(binding, name) for name in source_names},
            model_provenance={name: getattr(binding, name) for name in model_names},
        ),
    )
    publisher.add(
        file_id="proposal-policy",
        role="proposal_execution_policy",
        relative_path="shared/proposal-policy.json",
        payload=V5_PROPOSAL_EXECUTION_POLICY.audit_payload(),
    )
    split = snapshot.bundle.split_receipt
    publisher.add(
        file_id="split-receipt",
        role="split_receipt",
        relative_path="shared/split-receipt.json",
        payload=_raw(
            RAW_SPLIT_RECEIPT_SCHEMA,
            split_id=split.split_id,
            plan_sha256=split.plan_sha256,
            included_clean_parent_sha256s=list(split.included_clean_parent_sha256s),
            excluded_population_sha256s=list(split.excluded_population_sha256s),
            disjointness_verified=split.disjointness_verified,
        ),
    )
    publisher.add(
        file_id="evaluator-config",
        role="evaluator_config",
        relative_path="shared/evaluator-config.json",
        payload=_raw(
            RAW_EVALUATOR_CONFIG_SCHEMA,
            config=asdict(snapshot.bundle.evaluator_config),
        ),
    )
    parent_rows = tuple(
        _publish_parent(publisher, parent_index=index, parent=parent)
        for index, parent in enumerate(snapshot.parents)
    )
    return V5K1PhaseCWrittenRawFiles(
        file_rows=tuple(publisher.rows),
        parent_rows=parent_rows,
        paths=tuple(publisher.paths),
    )


__all__ = ["V5K1PhaseCWrittenRawFiles", "write_v5_k1_phase_c_raw_files"]
