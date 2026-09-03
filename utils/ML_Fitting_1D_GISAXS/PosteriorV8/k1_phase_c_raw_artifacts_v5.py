"""Assemble typed K1 Phase-C replay bundles from lossless raw files.

Audit payloads omit numerical arrays and cannot be production replay inputs.
This consumer-only module joins separately hash-checked raw files and provides
no production artifact writer.
"""

from __future__ import annotations

from hashlib import sha256
from typing import Mapping

import numpy as np

from .contextual_reference_bank_v5 import (
    V5_CONTEXTUAL_REFERENCE_BANK_SCHEMA,
    V5_CONTEXTUAL_REFERENCE_BANK_VERSION,
)
from .grouped_artifact_v5 import canonical_json
from .k1_phase_c_contract_v5 import K1_PHASE_C_METHOD_IDS, digest
from .k1_phase_c_plan_v5 import V5K1PhaseCPlan
from .k1_phase_c_raw_codecs_v5 import (
    RAW_ARTIFACT_BINDING_SCHEMA,
    RAW_BRANCH_SIDECAR_SCHEMA,
    RAW_DISTANCE_CONTEXT_SCHEMA,
    RAW_EVALUATOR_CONFIG_SCHEMA,
    RAW_METHOD_TRACE_SCHEMA,
    RAW_PARENT_PROVENANCE_SCHEMA,
    RAW_REFERENCE_BANK_SCHEMA,
    RAW_REFERENCE_TRACE_SCHEMA,
    RAW_REPRESENTATIVE_PAYLOAD_SCHEMA,
    RAW_SPLIT_RECEIPT_SCHEMA,
    V5_K1_PHASE_C_RAW_ARTIFACT_SHA_SEMANTICS,
    V5_K1_PHASE_C_RAW_ARTIFACT_VERSION,
    V5_K1_PHASE_C_RAW_MANIFEST_SCHEMA,
    V5_K1_PHASE_C_RAW_MANIFEST_VERSION,
    V5K1PhaseCQueryDistanceMatcher,
    V5K1PhaseCRawFile,
    _artifact_binding,
    _evaluator_config,
    _exact_calls,
    _field_names,
    _judgement,
    _object,
    _parse_branch_sidecar,
    _parse_distance_context,
    _parse_parent_provenance,
    _parse_representative,
    _raw_envelope,
    _sequence,
    _split_receipt,
    _text,
)
from .k1_phase_c_replay_contract_v5 import (
    V5K1PhaseCMethodReplay,
    V5K1PhaseCParentReplayEvidence,
    V5K1PhaseCReferenceBankReplay,
    V5K1PhaseCReferenceCluster,
    V5K1PhaseCReplayBundle,
)
from .paper_budget_evaluator_v5 import (
    V5CandidateEmission,
    V5FrozenReferenceRepresentative,
    V5FrozenReferenceSet,
    V5MethodExactCallTrace,
)
from .paper_representative_payload_v5 import (
    V5_EMITTED_REPRESENTATIVE_ROLE,
    V5_REFERENCE_REPRESENTATIVE_ROLE,
)
from .proposal_execution_policy_v5 import V5_PROPOSAL_EXECUTION_POLICY
from .query_parameter_distance_v5 import (
    V5_QUERY_PARAMETER_DISTANCE_SCHEMA,
    V5_QUERY_PARAMETER_DISTANCE_SHA256,
    V5_QUERY_PARAMETER_DISTANCE_VERSION,
    query_local_parameter_distance,
)


def _parse_method_trace(
    raw: V5K1PhaseCRawFile,
    files: Mapping[str, V5K1PhaseCRawFile],
    used: set[str],
) -> V5K1PhaseCMethodReplay:
    row = _raw_envelope(
        raw.payload,
        RAW_METHOD_TRACE_SCHEMA,
        {
            "status",
            "query_id",
            "pairing_unit_id",
            "method_id",
            "method_protocol_id",
            "method_protocol_sha256",
            "trace_id",
            "reference_set_id",
            "reference_set_sha256",
            "comparison_protocol_id",
            "comparison_protocol_sha256",
            "exact_forward_call_budget",
            "exact_forward_calls",
            "candidate_emissions",
            "source_bundle_sha256_used",
            "model_artifact_sha256_used",
            "proposal_execution_policy_sha256_used",
        },
        "method exact-call trace",
    )
    emissions = []
    emission_fields = {
        "available_after_call",
        "output_rank",
        "candidate_id",
        "compatibility_status",
        "elapsed_seconds",
        "payload_file_id",
        "payload_sha256",
    }
    for index, value in enumerate(_sequence(row["candidate_emissions"], "candidate_emissions")):
        item = _object(value, emission_fields, f"candidate_emissions[{index}]")
        payload_raw = _required_file(files, item["payload_file_id"], "representative_payload", used)
        payload = _parse_representative(payload_raw)
        if (
            payload.role != V5_EMITTED_REPRESENTATIVE_ROLE
            or payload.representative_id != item["candidate_id"]
            or payload.sha256 != digest(item["payload_sha256"], "payload_sha256")
        ):
            raise ValueError("method emission payload identity does not reproduce")
        emissions.append(
            V5CandidateEmission(
                available_after_call=item["available_after_call"],
                output_rank=item["output_rank"],
                candidate_id=item["candidate_id"],
                compatibility_status=item["compatibility_status"],
                elapsed_seconds=item["elapsed_seconds"],
                payload=payload,
            )
        )
    trace = V5MethodExactCallTrace(
        query_id=row["query_id"],
        pairing_unit_id=row["pairing_unit_id"],
        method_id=row["method_id"],
        method_protocol_id=row["method_protocol_id"],
        method_protocol_sha256=row["method_protocol_sha256"],
        trace_id=row["trace_id"],
        trace_artifact_sha256=raw.file_sha256,
        reference_set_id=row["reference_set_id"],
        reference_set_sha256=row["reference_set_sha256"],
        comparison_protocol_id=row["comparison_protocol_id"],
        comparison_protocol_sha256=row["comparison_protocol_sha256"],
        exact_forward_call_budget=row["exact_forward_call_budget"],
        exact_forward_calls=_exact_calls(row["exact_forward_calls"], "exact_forward_calls"),
        candidate_emissions=tuple(emissions),
    )
    return V5K1PhaseCMethodReplay(
        status=row["status"],
        trace=trace,
        source_bundle_sha256_used=row["source_bundle_sha256_used"],
        model_artifact_sha256_used=row["model_artifact_sha256_used"],
        proposal_execution_policy_sha256_used=row["proposal_execution_policy_sha256_used"],
    )


def _required_file(
    files: Mapping[str, V5K1PhaseCRawFile],
    file_id: object,
    role: str,
    used: set[str],
) -> V5K1PhaseCRawFile:
    identifier = _text(file_id, "file_id")
    try:
        value = files[identifier]
    except KeyError as exc:
        raise ValueError(f"manifest references unknown raw file {identifier!r}") from exc
    if value.role != role:
        raise ValueError(f"raw file {identifier!r} has role {value.role!r}, expected {role!r}")
    used.add(identifier)
    return value


def _parse_reference(
    bank_raw: V5K1PhaseCRawFile,
    trace_raw: V5K1PhaseCRawFile,
    files: Mapping[str, V5K1PhaseCRawFile],
    used: set[str],
) -> tuple[V5K1PhaseCReferenceBankReplay, V5FrozenReferenceSet]:
    bank = _raw_envelope(
        bank_raw.payload,
        RAW_REFERENCE_BANK_SCHEMA,
        {
            "reference_bank_schema",
            "reference_bank_version",
            "calibration_identity_sha256",
            "calibrated_threshold_sha256",
            "exact_judge_sha256",
            "source_bundle_sha256",
            "query_id",
            "pairing_unit_id",
            "query_context_sha256",
            "reference_set_id",
            "comparison_protocol_id",
            "comparison_protocol_sha256",
            "candidate_ids",
            "exact_compatible_candidate_ids",
            "representative_clusters",
            "representative_payloads",
            "distance_schema",
            "distance_version",
            "distance_sha256",
        },
        "reference-bank artifact",
    )
    if (bank["reference_bank_schema"], bank["reference_bank_version"]) != (
        V5_CONTEXTUAL_REFERENCE_BANK_SCHEMA,
        V5_CONTEXTUAL_REFERENCE_BANK_VERSION,
    ):
        raise ValueError("reference bank uses an unsupported contextual contract")
    if (bank["distance_schema"], bank["distance_version"], bank["distance_sha256"]) != (
        V5_QUERY_PARAMETER_DISTANCE_SCHEMA,
        V5_QUERY_PARAMETER_DISTANCE_VERSION,
        V5_QUERY_PARAMETER_DISTANCE_SHA256,
    ):
        raise ValueError("reference bank uses an unsupported distance contract")
    representatives = []
    payload_pairs = []
    payload_ref_fields = {"representative_id", "payload_file_id", "payload_sha256"}
    for index, value in enumerate(
        _sequence(bank["representative_payloads"], "representative_payloads")
    ):
        item = _object(value, payload_ref_fields, f"representative_payloads[{index}]")
        raw = _required_file(files, item["payload_file_id"], "representative_payload", used)
        payload = _parse_representative(raw)
        if (
            payload.role != V5_REFERENCE_REPRESENTATIVE_ROLE
            or payload.representative_id != item["representative_id"]
            or payload.sha256 != digest(item["payload_sha256"], "payload_sha256")
        ):
            raise ValueError("reference representative payload identity does not reproduce")
        representatives.append(
            V5FrozenReferenceRepresentative(
                representative_id=item["representative_id"], payload=payload
            )
        )
        payload_pairs.append((item["representative_id"], payload.sha256))
    reference_set = V5FrozenReferenceSet(
        query_id=bank["query_id"],
        pairing_unit_id=bank["pairing_unit_id"],
        reference_set_id=bank["reference_set_id"],
        reference_set_sha256=bank_raw.file_sha256,
        comparison_protocol_id=bank["comparison_protocol_id"],
        comparison_protocol_sha256=bank["comparison_protocol_sha256"],
        representatives=tuple(representatives),
    )

    trace = _raw_envelope(
        trace_raw.payload,
        RAW_REFERENCE_TRACE_SCHEMA,
        {
            "query_id",
            "pairing_unit_id",
            "calibration_identity_sha256",
            "calibrated_threshold_sha256",
            "exact_judge_sha256",
            "source_bundle_sha256",
            "configured_exact_call_budget",
            "consumed_exact_calls",
            "exact_forward_calls",
            "candidate_judgements",
            "enumeration_complete",
            "network_free",
        },
        "reference-search trace",
    )
    identity_names = (
        "query_id",
        "pairing_unit_id",
        "calibration_identity_sha256",
        "calibrated_threshold_sha256",
        "exact_judge_sha256",
        "source_bundle_sha256",
    )
    if any(trace[name] != bank[name] for name in identity_names):
        raise ValueError("reference bank and exact-search trace identities disagree")
    judgements = tuple(
        _judgement(value, f"reference candidate_judgements[{index}]")
        for index, value in enumerate(
            _sequence(trace["candidate_judgements"], "candidate_judgements")
        )
    )
    clusters = tuple(
        V5K1PhaseCReferenceCluster(
            **{
                **_object(
                    value,
                    _field_names(V5K1PhaseCReferenceCluster),
                    f"representative_clusters[{index}]",
                ),
                "member_candidate_ids": tuple(value["member_candidate_ids"]),
            }
        )
        for index, value in enumerate(
            _sequence(bank["representative_clusters"], "representative_clusters")
        )
    )
    replay = V5K1PhaseCReferenceBankReplay(
        bank_artifact_sha256=bank_raw.file_sha256,
        search_trace_artifact_sha256=trace_raw.file_sha256,
        calibration_identity_sha256=bank["calibration_identity_sha256"],
        calibrated_threshold_sha256=bank["calibrated_threshold_sha256"],
        exact_judge_sha256=bank["exact_judge_sha256"],
        source_bundle_sha256=bank["source_bundle_sha256"],
        query_id=bank["query_id"],
        pairing_unit_id=bank["pairing_unit_id"],
        query_context_sha256=bank["query_context_sha256"],
        candidate_ids=tuple(_sequence(bank["candidate_ids"], "candidate_ids")),
        exact_compatible_candidate_ids=tuple(
            _sequence(bank["exact_compatible_candidate_ids"], "exact_compatible_candidate_ids")
        ),
        candidate_judgements=judgements,
        representative_clusters=clusters,
        representative_payload_sha256s=tuple(payload_pairs),
        configured_exact_call_budget=trace["configured_exact_call_budget"],
        consumed_exact_calls=trace["consumed_exact_calls"],
        exact_forward_calls=_exact_calls(trace["exact_forward_calls"], "reference exact calls"),
        enumeration_complete=trace["enumeration_complete"],
        network_free=trace["network_free"],
    )
    return replay, reference_set


def build_v5_k1_phase_c_bundle_from_raw_files(
    *,
    manifest: Mapping[str, object],
    files: Mapping[str, V5K1PhaseCRawFile],
    plan: V5K1PhaseCPlan,
    contract: Mapping[str, object],
) -> tuple[V5K1PhaseCReplayBundle, V5K1PhaseCQueryDistanceMatcher]:
    """Reconstruct a typed bundle only from complete, hash-checked raw files."""

    root_fields = {
        "schema",
        "version",
        "status",
        "formal",
        "artifact_sha256_semantics",
        "plan_sha256",
        "contract_sha256",
        "artifact_binding_file_id",
        "proposal_execution_policy_file_id",
        "split_receipt_file_id",
        "evaluator_config_file_id",
        "files",
        "parents",
        "manifest_sha256",
    }
    value = _object(manifest, root_fields, "raw manifest")
    if (value["schema"], value["version"], value["status"]) != (
        V5_K1_PHASE_C_RAW_MANIFEST_SCHEMA,
        V5_K1_PHASE_C_RAW_MANIFEST_VERSION,
        "complete",
    ):
        raise ValueError("unsupported or incomplete K1 Phase-C raw manifest")
    if value["artifact_sha256_semantics"] != V5_K1_PHASE_C_RAW_ARTIFACT_SHA_SEMANTICS:
        raise ValueError("raw manifest artifact SHA-256 semantics are unsupported")
    if type(value["formal"]) is not bool or value["formal"] is not plan.formal:
        raise ValueError("raw manifest formal role disagrees with the plan")
    if value["plan_sha256"] != plan.sha256 or value["contract_sha256"] != contract.get(
        "contract_sha256"
    ):
        raise ValueError("raw manifest escaped the requested plan or contract")
    core = {name: item for name, item in value.items() if name != "manifest_sha256"}
    if digest(value["manifest_sha256"], "manifest_sha256") != sha256(
        canonical_json(core).encode("utf-8")
    ).hexdigest():
        raise ValueError("raw manifest semantic SHA-256 does not reproduce")

    used: set[str] = set()
    binding = _artifact_binding(
        _required_file(files, value["artifact_binding_file_id"], "artifact_binding", used)
    )
    policy = _required_file(
        files,
        value["proposal_execution_policy_file_id"],
        "proposal_execution_policy",
        used,
    )
    if dict(policy.payload) != V5_PROPOSAL_EXECUTION_POLICY.audit_payload():
        raise ValueError("proposal-execution policy file is not the frozen live policy")
    split = _split_receipt(
        _required_file(files, value["split_receipt_file_id"], "split_receipt", used)
    )
    config = _evaluator_config(
        _required_file(files, value["evaluator_config_file_id"], "evaluator_config", used)
    )

    parent_fields = {
        "clean_parent_sha256",
        "provenance_file_id",
        "distance_context_file_id",
        "branch_sidecar_file_ids",
        "reference_bank_file_id",
        "reference_trace_file_id",
        "method_trace_file_ids",
    }
    parents = []
    distance_tasks: dict[tuple[str, str], object] = {}
    for index, raw_parent in enumerate(_sequence(value["parents"], "parents")):
        parent = _object(raw_parent, parent_fields, f"parents[{index}]")
        parent_sha = digest(parent["clean_parent_sha256"], "clean_parent_sha256")
        provenance = _parse_parent_provenance(
            _required_file(files, parent["provenance_file_id"], "parent_provenance", used)
        )
        if provenance.clean_parent_sha256 != parent_sha:
            raise ValueError("parent manifest key disagrees with raw provenance")
        context_raw = _required_file(
            files, parent["distance_context_file_id"], "distance_context", used
        )
        for key, task in _parse_distance_context(context_raw, provenance).items():
            if key in distance_tasks:
                raise ValueError("distance-context identity is duplicated across parents")
            distance_tasks[key] = task
        branches = tuple(
            _parse_branch_sidecar(
                _required_file(files, file_id, "branch_search_sidecar", used)
            )
            for file_id in _sequence(
                parent["branch_sidecar_file_ids"], "branch_sidecar_file_ids"
            )
        )
        bank_raw = _required_file(
            files, parent["reference_bank_file_id"], "reference_bank", used
        )
        reference_trace_raw = _required_file(
            files, parent["reference_trace_file_id"], "reference_search_trace", used
        )
        bank, reference_set = _parse_reference(bank_raw, reference_trace_raw, files, used)
        methods = tuple(
            _parse_method_trace(
                _required_file(files, file_id, "method_exact_call_trace", used),
                files,
                used,
            )
            for file_id in _sequence(
                parent["method_trace_file_ids"], "method_trace_file_ids"
            )
        )
        if tuple(sorted(value.trace.method_id for value in methods)) != tuple(
            sorted(K1_PHASE_C_METHOD_IDS)
        ):
            raise ValueError("parent raw files require exactly the three frozen method traces")
        parents.append(
            V5K1PhaseCParentReplayEvidence(
                provenance=provenance,
                branch_searches=branches,
                product_candidate_judgements=tuple(
                    candidate for branch in branches for candidate in branch.candidates
                ),
                reference_bank=bank,
                reference_set=reference_set,
                methods=methods,
            )
        )

    if set(files) != used:
        unused = sorted(set(files) - used)
        raise ValueError(f"raw manifest contains unreferenced files: {unused}")
    matcher = V5K1PhaseCQueryDistanceMatcher(distance_tasks)
    payloads = [
        value.payload
        for parent in parents
        for value in parent.reference_set.representatives
    ] + [
        emission.payload
        for parent in parents
        for method in parent.methods
        for emission in method.trace.candidate_emissions
    ]
    for payload in payloads:
        key = (payload.query_context_sha256, payload.global_branch_key)
        if key not in distance_tasks:
            raise ValueError("typed representative escaped the raw distance context")
        distance = query_local_parameter_distance(
            distance_tasks[key], payload.parameter, payload.parameter
        )
        if not np.isfinite(distance) or distance != 0.0:
            raise ValueError("typed representative does not replay inside its GUI query")

    bundle = V5K1PhaseCReplayBundle(
        plan_sha256=plan.sha256,
        contract_sha256=str(contract["contract_sha256"]),
        artifact_binding=binding,
        split_receipt=split,
        evaluator_config=config,
        parents=tuple(parents),
    )
    return bundle, matcher


__all__ = [
    "RAW_ARTIFACT_BINDING_SCHEMA",
    "RAW_BRANCH_SIDECAR_SCHEMA",
    "RAW_DISTANCE_CONTEXT_SCHEMA",
    "RAW_EVALUATOR_CONFIG_SCHEMA",
    "RAW_METHOD_TRACE_SCHEMA",
    "RAW_PARENT_PROVENANCE_SCHEMA",
    "RAW_REFERENCE_BANK_SCHEMA",
    "RAW_REFERENCE_TRACE_SCHEMA",
    "RAW_REPRESENTATIVE_PAYLOAD_SCHEMA",
    "RAW_SPLIT_RECEIPT_SCHEMA",
    "V5_K1_PHASE_C_RAW_ARTIFACT_VERSION",
    "V5_K1_PHASE_C_RAW_ARTIFACT_SHA_SEMANTICS",
    "V5_K1_PHASE_C_RAW_MANIFEST_SCHEMA",
    "V5_K1_PHASE_C_RAW_MANIFEST_VERSION",
    "V5K1PhaseCQueryDistanceMatcher",
    "V5K1PhaseCRawFile",
    "build_v5_k1_phase_c_bundle_from_raw_files",
]
