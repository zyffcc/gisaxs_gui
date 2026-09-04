"""Typed live-input contract for the K1 Phase-C lossless writer."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from hashlib import sha256
import json
import os
from pathlib import Path
import stat
from typing import Mapping

import numpy as np

from .contextual_reference_bank_v5 import V5ContextualReferenceBank
from .grouped_artifact_v5 import V5ArtifactReceipt, canonical_json
from .k1_phase_c_contract_v5 import (
    K1_PHASE_C_BRANCHES,
    digest,
    validate_v5_k1_phase_c_contract,
)
from .k1_phase_c_plan_v5 import V5K1PhaseCPlan, validate_v5_k1_phase_c_plan
from .k1_phase_c_replay_contract_v5 import (
    V5K1PhaseCParentReplayEvidence,
    V5K1PhaseCReplayBundle,
)
from .lossless_representative_artifact_v5 import (
    decode_v5_lossless_representative_payload,
)
from .search_evidence_receipt_v5 import V5SearchEvidenceReceipt
from .search_supervision_contract_v5 import V5FrozenSearchTask
from .search_supervision_sidecar_v5 import (
    V5SearchSupervisionSidecar,
    branch_array,
    query_array,
)
from .universal_query_contract_v5 import V5TopologyQuery


_PRODUCTION_CAPTURE_SEAL = object()


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _file_sha256(path: Path) -> str:
    value = sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


@dataclass(frozen=True, kw_only=True)
class V5K1PhaseCUpstreamFile:
    role: str
    path: Path
    file_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.role, str) or not self.role.strip():
            raise ValueError("upstream file role must be non-empty text")
        selected = Path(self.path)
        lexical = selected if selected.is_absolute() else Path.cwd() / selected
        probe = Path(lexical.anchor)
        for part in lexical.parts[1:]:
            probe /= part
            if probe.is_symlink():
                raise ValueError("upstream source path must not contain symbolic links")
        if selected.is_symlink() or not selected.is_file():
            raise ValueError("upstream source must be a regular non-symlink file")
        metadata = selected.stat(follow_symlinks=False)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("upstream source must be a regular file")
        expected = digest(self.file_sha256, "upstream file SHA-256")
        resolved = selected.resolve(strict=True)
        if _file_sha256(resolved) != expected:
            raise ValueError("upstream source file SHA-256 does not reproduce")
        object.__setattr__(self, "role", self.role.strip())
        object.__setattr__(self, "path", resolved)
        object.__setattr__(self, "file_sha256", expected)


@dataclass(frozen=True, kw_only=True)
class V5K1PhaseCParentWriterInput:
    evidence: V5K1PhaseCParentReplayEvidence
    topology_queries: tuple[V5TopologyQuery, ...]
    upstream_files: tuple[V5K1PhaseCUpstreamFile, ...] = ()
    _production_seal: object | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if type(self.evidence) is not V5K1PhaseCParentReplayEvidence:
            raise TypeError("writer parent evidence must use the exact typed replay value")
        queries = tuple(self.topology_queries)
        if not queries or not all(type(value) is V5TopologyQuery for value in queries):
            raise ValueError("writer parent requires genuine typed topology queries")
        if tuple(value.topology_id for value in queries) != tuple(
            sorted({value.topology_id for value in queries})
        ):
            raise ValueError("writer topology queries must be unique and sorted")
        files = tuple(self.upstream_files)
        if not all(type(value) is V5K1PhaseCUpstreamFile for value in files):
            raise TypeError("upstream_files contain an unsupported value")
        identities = [(value.role, value.file_sha256) for value in files]
        if len(identities) != len(set(identities)):
            raise ValueError("upstream file role/SHA bindings must be unique")
        object.__setattr__(self, "topology_queries", queries)
        object.__setattr__(self, "upstream_files", files)

    @property
    def production_verified(self) -> bool:
        return self._production_seal is _PRODUCTION_CAPTURE_SEAL


@dataclass(frozen=True, kw_only=True)
class V5K1PhaseCWriterSnapshot:
    plan: V5K1PhaseCPlan
    contract: Mapping[str, object]
    bundle: V5K1PhaseCReplayBundle
    parents: tuple[V5K1PhaseCParentWriterInput, ...]
    global_upstream_files: tuple[V5K1PhaseCUpstreamFile, ...] = ()
    _production_seal: object | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        frozen = validate_v5_k1_phase_c_contract(self.contract)
        validate_v5_k1_phase_c_plan(self.plan, contract=frozen)
        if type(self.bundle) is not V5K1PhaseCReplayBundle:
            raise TypeError("writer snapshot requires a typed replay bundle")
        if (self.bundle.plan_sha256, self.bundle.contract_sha256) != (
            self.plan.sha256,
            frozen["contract_sha256"],
        ):
            raise ValueError("writer bundle escaped its plan or contract")
        parents = tuple(self.parents)
        if not all(type(value) is V5K1PhaseCParentWriterInput for value in parents):
            raise TypeError("writer snapshot parents contain an unsupported value")
        if tuple(value.evidence for value in parents) != self.bundle.parents:
            raise ValueError("writer parent inputs do not exactly cover the typed bundle")
        files = tuple(self.global_upstream_files)
        if not all(type(value) is V5K1PhaseCUpstreamFile for value in files):
            raise TypeError("global upstream files contain an unsupported value")
        identities = [(value.role, value.file_sha256) for value in files]
        if len(identities) != len(set(identities)):
            raise ValueError("global upstream role/SHA bindings must be unique")
        object.__setattr__(self, "contract", frozen)
        object.__setattr__(self, "parents", parents)
        object.__setattr__(self, "global_upstream_files", files)

    @property
    def production_verified(self) -> bool:
        return (
            self._production_seal is _PRODUCTION_CAPTURE_SEAL
            and self.plan.formal is True
            and all(value.production_verified for value in self.parents)
        )


def _verify_upstream_files(
    files: tuple[V5K1PhaseCUpstreamFile, ...],
    *,
    required_sha256s: set[str],
) -> None:
    observed = {value.file_sha256 for value in files}
    missing = required_sha256s - observed
    if missing:
        raise ValueError(f"production capture is missing upstream files: {sorted(missing)}")
    for value in files:
        if value.path.is_symlink() or _file_sha256(value.path) != value.file_sha256:
            raise RuntimeError("upstream source changed during production capture")


def _verify_named_upstream_files(
    files: tuple[V5K1PhaseCUpstreamFile, ...],
    required: Mapping[str, set[str]],
) -> None:
    observed: dict[str, set[str]] = {}
    for value in files:
        observed.setdefault(value.role, set()).add(value.file_sha256)
    if observed != {name: set(values) for name, values in required.items()} or len(files) != sum(
        len(values) for values in observed.values()
    ):
        raise ValueError("production upstream role/SHA inventory is incomplete or has extras")
    _verify_upstream_files(
        files,
        required_sha256s={hash_value for values in required.values() for hash_value in values},
    )


def _verify_lossless_source_payloads(
    evidence: V5K1PhaseCParentReplayEvidence,
    files: tuple[V5K1PhaseCUpstreamFile, ...],
) -> None:
    by_sha = {value.file_sha256: value for value in files}
    payloads = [
        emission.payload
        for method in evidence.methods
        for emission in method.trace.candidate_emissions
    ]
    for payload in payloads:
        try:
            source = by_sha[payload.source_artifact_sha256]
        except KeyError as exc:
            raise ValueError("typed representative has no exact upstream source file") from exc
        raw = source.path.read_bytes()
        try:
            value = json.loads(
                raw,
                object_pairs_hook=_strict_object,
                parse_constant=_reject_constant,
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("upstream representative source is not JSON") from exc
        if not isinstance(value, dict) or canonical_json(value).encode("utf-8") != raw:
            raise ValueError("upstream representative source must be canonical JSON")
        decoded = decode_v5_lossless_representative_payload(
            value, source_artifact_sha256=source.file_sha256
        )
        if decoded.sha256 != payload.sha256 or decoded.audit_payload() != payload.audit_payload():
            raise ValueError("upstream lossless representative disagrees with typed payload")


def _verify_search_sidecar_context(
    *,
    evidence: V5K1PhaseCParentReplayEvidence,
    search_task: V5FrozenSearchTask,
    search_sidecar: V5SearchSupervisionSidecar,
    search_sidecar_receipt: V5ArtifactReceipt,
    search_evidence_receipt: V5SearchEvidenceReceipt,
) -> None:
    manifest = search_sidecar.manifest
    receipt_sidecar = search_evidence_receipt.manifest.get("sidecar")
    if (
        search_sidecar_receipt.manifest_sha256 != manifest.get("manifest_sha256")
        or not isinstance(receipt_sidecar, Mapping)
        or receipt_sidecar.get("artifact_sha256") != search_sidecar_receipt.artifact_sha256
        or receipt_sidecar.get("manifest_sha256") != search_sidecar_receipt.manifest_sha256
        or manifest.get("protocol_sha256") != search_task.protocol.sha256
    ):
        raise ValueError("search task, sidecar, and evidence receipts are not one artifact")

    query_index = search_task.query_index
    if query_index >= search_sidecar.query_count:
        raise ValueError("search task query index is absent from its checked sidecar")
    arrays = search_sidecar.arrays
    expected_query_values = {
        "clean_group_id": search_task.clean_group_id,
        "recipe_id": search_task.recipe_id,
        "observation_id": search_task.observation_id,
        "universal_query_sha256": search_task.universal_context.audit_sha256,
        "exact_curve_sha256": search_task.exact_curve_sha256,
        "exact_observation_audit_sha256": search_task.exact_observation.audit_sha256,
        "query_catalog_artifact_sha256": search_task.query_catalog_artifact_sha256,
    }
    if any(
        str(arrays[query_array(name)][query_index]) != expected
        for name, expected in expected_query_values.items()
    ) or int(arrays[query_array("branch_count")][query_index]) != len(
        search_task.universal_context.branches
    ):
        raise ValueError("search task does not reproduce its checked sidecar query row")

    rows = np.flatnonzero(arrays[branch_array("query_index")] == query_index).tolist()
    context_branches = tuple(search_task.universal_context.branches)
    if len(rows) != len(context_branches) or not all(
        bool(arrays[branch_array("runner_completed")][row]) for row in rows
    ):
        raise ValueError("search sidecar lacks the complete query branch population")
    actual_by_key = {str(arrays[branch_array("global_branch_key")][row]): row for row in rows}
    expected_by_key = {
        branch.global_key.wire_key: (index, branch) for index, branch in enumerate(context_branches)
    }
    if set(actual_by_key) != set(expected_by_key):
        raise ValueError("search sidecar branch keys escaped the live query context")
    for wire_key, (branch_index, branch) in expected_by_key.items():
        row = actual_by_key[wire_key]
        task = replace(search_task, branch_index=branch_index)
        if (
            str(arrays[branch_array("task_audit_sha256")][row]) != task.audit_sha256
            or int(arrays[branch_array("topology_id")][row]) != branch.global_key.topology_id
            or int(arrays[branch_array("pattern_id")][row]) != branch.global_key.pattern_id
            or str(arrays[branch_array("context_sha256")][row]) != branch.context_sha256
            or int(arrays[branch_array("frozen_exact_forward_call_budget")][row])
            != search_task.protocol.exact_forward_call_budget
        ):
            raise ValueError("search sidecar branch row does not reproduce its live task")
    k1_keys = {
        (value.topology_id, value.pattern_id): value.branch_id for value in K1_PHASE_C_BRANCHES
    }
    evidence_keys = {value.branch_id for value in evidence.branch_searches}
    if set(k1_keys.values()) != evidence_keys or set(k1_keys) != {
        (branch.global_key.topology_id, branch.global_key.pattern_id) for branch in context_branches
    }:
        raise ValueError("K1 branch evidence does not cover the checked sidecar query")


def capture_v5_k1_phase_c_production_parent(
    *,
    evidence: V5K1PhaseCParentReplayEvidence,
    reference_bank: V5ContextualReferenceBank,
    search_task: V5FrozenSearchTask,
    search_sidecar: V5SearchSupervisionSidecar,
    search_sidecar_receipt: V5ArtifactReceipt,
    search_evidence_receipt: V5SearchEvidenceReceipt,
    upstream_files: tuple[V5K1PhaseCUpstreamFile, ...],
) -> V5K1PhaseCParentWriterInput:
    """Capture one parent only from live production objects and exact source files."""

    exact_types = (
        (evidence, V5K1PhaseCParentReplayEvidence),
        (reference_bank, V5ContextualReferenceBank),
        (search_task, V5FrozenSearchTask),
        (search_sidecar, V5SearchSupervisionSidecar),
        (search_sidecar_receipt, V5ArtifactReceipt),
        (search_evidence_receipt, V5SearchEvidenceReceipt),
    )
    if any(type(value) is not expected for value, expected in exact_types):
        raise TypeError("production parent capture requires exact live production types")
    provenance = evidence.provenance
    scope = reference_bank.scope
    if search_task.audit_sha256 != scope.task_template.audit_sha256:
        raise ValueError("reference bank and search task do not share one exact task")
    threshold = search_task.calibrated_threshold
    if threshold is None or (
        provenance.universal_query_sha256 != search_task.universal_context.audit_sha256
        or provenance.protocol_sha256 != search_task.protocol.sha256
        or provenance.calibration_identity_sha256 != threshold.calibration_identity.sha256
        or provenance.calibrated_threshold_sha256 != threshold.sha256
        or provenance.evaluation_query_context_sha256 != scope.sha256
        or provenance.source_bundle_sha256 != scope.source_bundle_sha256
    ):
        raise ValueError("production parent escaped task/query/calibration/source context")
    if (
        search_evidence_receipt.sidecar_artifact_sha256 != search_sidecar_receipt.artifact_sha256
        or search_sidecar.query_count < 1
        or search_sidecar.branch_count < len(evidence.branch_searches)
        or {value.search_sidecar_sha256 for value in evidence.branch_searches}
        != {search_sidecar_receipt.artifact_sha256}
    ):
        raise ValueError("typed branch evidence does not bind the audited search sidecar")
    _verify_search_sidecar_context(
        evidence=evidence,
        search_task=search_task,
        search_sidecar=search_sidecar,
        search_sidecar_receipt=search_sidecar_receipt,
        search_evidence_receipt=search_evidence_receipt,
    )
    bank_candidates = {value.candidate.candidate_id for value in reference_bank.candidates}
    if bank_candidates != set(evidence.reference_bank.candidate_ids):
        raise ValueError("typed reference replay does not bind the live contextual bank")
    by_candidate = {value.candidate.candidate_id: value for value in reference_bank.candidates}
    representative_ids = {
        value.representative_candidate_id for value in reference_bank.representatives
    }
    replay_representatives = {
        value.representative_id for value in evidence.reference_set.representatives
    }
    if representative_ids != replay_representatives or representative_ids != {
        value.representative_id for value in evidence.reference_bank.representative_clusters
    }:
        raise ValueError("reference replay does not expose the live bank representatives")
    for value in evidence.reference_set.representatives:
        reference = value.payload
        candidate = by_candidate[value.representative_id]
        if (
            reference.query_context_sha256 != scope.sha256
            or reference.global_branch_key != candidate.evidence.value("global_branch_key")
            or reference.parameter.topology_id != candidate.candidate.topology_id
            or reference.parameter.components != candidate.candidate.components
            or reference.parameter.resolution != candidate.candidate.resolution
            or reference.parameter.linear_solution != candidate.candidate.linear_solution
        ):
            raise ValueError("reference payload does not reproduce its live bank candidate")
    required = {
        "search_sidecar": {search_sidecar_receipt.artifact_sha256},
        "search_evidence_receipt": {search_evidence_receipt.file_sha256},
        "reference_bank": {evidence.reference_bank.bank_artifact_sha256},
        "reference_trace": {evidence.reference_bank.search_trace_artifact_sha256},
    }
    required.update(
        {
            f"method_trace:{value.trace.method_id}": {value.trace.trace_artifact_sha256}
            for value in evidence.methods
        }
    )
    for method in evidence.methods:
        for emission in method.trace.candidate_emissions:
            required.setdefault(f"representative_payload:{emission.candidate_id}", set()).add(
                emission.payload.source_artifact_sha256
            )
    _verify_named_upstream_files(upstream_files, required)
    _verify_lossless_source_payloads(evidence, upstream_files)
    return V5K1PhaseCParentWriterInput(
        evidence=evidence,
        topology_queries=tuple(search_task.universal_context.topology_queries),
        upstream_files=upstream_files,
        _production_seal=_PRODUCTION_CAPTURE_SEAL,
    )


def capture_v5_k1_phase_c_production_snapshot(
    *,
    plan: V5K1PhaseCPlan,
    contract: Mapping[str, object],
    bundle: V5K1PhaseCReplayBundle,
    parents: tuple[V5K1PhaseCParentWriterInput, ...],
    global_upstream_files: tuple[V5K1PhaseCUpstreamFile, ...],
) -> V5K1PhaseCWriterSnapshot:
    """Mint a formal snapshot only after every live source binding is present."""

    frozen = validate_v5_k1_phase_c_contract(contract)
    validate_v5_k1_phase_c_plan(plan, contract=frozen)
    if not plan.formal or len(parents) != plan.total_parent_count:
        raise ValueError("production writer requires the complete frozen formal population")
    if not all(value.production_verified for value in parents):
        raise ValueError("production writer refuses fixture or aggregate-only parent inputs")
    binding = bundle.artifact_binding
    required = {
        "source_archive": {binding.source_archive_sha256},
        "source_manifest": {binding.source_manifest_sha256},
        "cross_platform_gate_claim": {binding.cross_platform_gate_claim_sha256},
        "phase_a_launch_receipt": {binding.phase_a_launch_receipt_sha256},
        "model_artifact": {binding.model_artifact_sha256},
        "model_weights": {binding.model_weights_sha256},
        "model_training_result": {binding.model_training_result_sha256},
        "split_receipt": {bundle.split_receipt.artifact_sha256},
    }
    _verify_named_upstream_files(global_upstream_files, required)
    return V5K1PhaseCWriterSnapshot(
        plan=plan,
        contract=frozen,
        bundle=bundle,
        parents=parents,
        global_upstream_files=global_upstream_files,
        _production_seal=_PRODUCTION_CAPTURE_SEAL,
    )


def revalidate_v5_k1_phase_c_writer_sources(snapshot: V5K1PhaseCWriterSnapshot) -> None:
    if type(snapshot) is not V5K1PhaseCWriterSnapshot:
        raise TypeError("snapshot must use the exact writer snapshot type")
    for value in snapshot.global_upstream_files:
        if _file_sha256(value.path) != value.file_sha256:
            raise RuntimeError("global upstream source changed during writing")
    for parent in snapshot.parents:
        for value in parent.upstream_files:
            if _file_sha256(value.path) != value.file_sha256:
                raise RuntimeError("parent upstream source changed during writing")


__all__ = [
    "V5K1PhaseCParentWriterInput",
    "V5K1PhaseCUpstreamFile",
    "V5K1PhaseCWriterSnapshot",
    "capture_v5_k1_phase_c_production_parent",
    "capture_v5_k1_phase_c_production_snapshot",
    "revalidate_v5_k1_phase_c_writer_sources",
]
