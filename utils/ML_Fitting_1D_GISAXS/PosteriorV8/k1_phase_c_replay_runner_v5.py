"""Replay raw K1 Phase-C evidence and issue completion-last checked receipts."""

from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import stat
import tempfile
from typing import Mapping

from .candidate_refinement_contract_v5 import V5_EXACT_FORWARD_BUDGET_UNIT
from .grouped_artifact_v5 import canonical_json
from .k1_phase_c_contract_v5 import (
    K1_PHASE_C_BRANCHES,
    K1_PHASE_C_METHOD_IDS,
    K1_PHASE_C_PRODUCT_METHOD_ID,
    K1_PHASE_C_REFERENCE_EXACT_BUDGET,
    K1_PHASE_C_REFERENCE_OUTPUT_CAP,
    K1_PHASE_C_SPLIT_ID,
    K1_PHASE_C_TOP_K,
    validate_v5_k1_phase_c_contract,
    v5_k1_phase_c_contract_payload,
)
from .k1_phase_c_evaluation_v5 import _assess_v5_k1_phase_c_records
from .k1_phase_c_plan_v5 import V5K1PhaseCPlan, validate_v5_k1_phase_c_plan
from .k1_phase_c_records_v5 import V5K1PhaseCMethodAudit, V5K1PhaseCParentRecord
from .k1_phase_c_replay_contract_v5 import (
    V5_K1_PHASE_C_METHOD_CRASHED_ZERO,
    V5K1PhaseCParentReplayEvidence,
    V5K1PhaseCReplayBundle,
    V5K1PhaseCReplayPort,
)
from .k1_phase_c_replay_receipt_v5 import (
    V5K1PhaseCCheckedReplayReceipt,
    _mint_v5_k1_phase_c_checked_receipt,
)
from .paper_budget_evaluator_v5 import (
    V5_EXACT_COMPATIBLE,
    V5_EXACT_UNVERIFIED,
    V5_PAPER_BUDGET_EVALUATOR_SCHEMA,
    V5_PAPER_BUDGET_EVALUATOR_VERSION,
    EquivalenceDistanceMatcher,
    V5PairedPaperBudgetQueryRecord,
    V5PaperBudgetEvaluationConfig,
    evaluate_v5_paired_paper_budget_query,
)
from .paper_endpoint_metrics import EXACT_FORWARD_BUDGETS
from .proposal_execution_policy_v5 import V5_PROPOSAL_EXECUTION_POLICY_SHA256
from .query_parameter_distance_v5 import V5_QUERY_PARAMETER_DISTANCE_SHA256


V5_K1_PHASE_C_REPLAY_RECEIPT_SCHEMA = "gisaxs.posterior_v8.k1_phase_c_replay_receipt/v3"
V5_K1_PHASE_C_REPLAY_RECEIPT_VERSION = (
    "policy_artifact_revalidated_raw_typed_completion_last_non_overwrite_v3"
)
V5_K1_PHASE_C_PRODUCTION_ADAPTER_BLOCKER = (
    "formal_phase_c_requires_verified_lossless_writer_receipt_capability"
)
_MAX_RECEIPT_BYTES = 256 * 1024 * 1024
_ROOT_FIELDS = {
    "schema",
    "version",
    "status",
    "formal",
    "claim_eligible",
    "runner_role",
    "production_adapter_blocker",
    "plan_sha256",
    "contract_sha256",
    "artifact_binding",
    "adapter",
    "evidence_bundle_sha256",
    "evidence_manifest",
    "parent_replays",
    "assessment",
    "receipt_sha256",
}


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _adapter_identity(port: V5K1PhaseCReplayPort) -> dict[str, str]:
    values = {}
    for name in ("adapter_id", "adapter_version"):
        value = getattr(port, name, None)
        if not isinstance(value, str) or not value.strip() or value != value.strip():
            raise ValueError(f"replay port {name} must be a non-empty stripped string")
        values[name] = value
    return values


def _method_audits(
    record: V5PairedPaperBudgetQueryRecord,
    evidence: V5K1PhaseCParentReplayEvidence,
) -> tuple[V5K1PhaseCMethodAudit, ...]:
    trace_by_method = {value.trace.method_id: value for value in evidence.methods}
    audits = []
    for result in record.method_results:
        replay = trace_by_method[result.method_id]
        auc = dict(result.auc_by_output_cap)[K1_PHASE_C_REFERENCE_OUTPUT_CAP]
        if replay.status == V5_K1_PHASE_C_METHOD_CRASHED_ZERO:
            if result.candidate_emissions_evaluated != 0 or auc != 0.0:
                raise ValueError("crashed baseline must remain present and score exactly zero")
        audits.append(
            V5K1PhaseCMethodAudit(
                method_id=result.method_id,
                evaluator_schema=V5_PAPER_BUDGET_EVALUATOR_SCHEMA,
                evaluator_version=V5_PAPER_BUDGET_EVALUATOR_VERSION,
                evaluator_config_sha256=record.evaluator_config_sha256,
                comparison_protocol_sha256=record.comparison_protocol_sha256,
                exact_judge_sha256=evidence.provenance.exact_judge_sha256,
                reference_set_sha256=record.reference_set_sha256,
                exact_forward_budget_unit=V5_EXACT_FORWARD_BUDGET_UNIT,
                exact_forward_budgets=record.exact_forward_budgets,
                output_cap=K1_PHASE_C_REFERENCE_OUTPUT_CAP,
                configured_exact_call_budget=K1_PHASE_C_REFERENCE_EXACT_BUDGET,
                complete_contiguous_trace=True,
                trace_artifact_sha256=result.trace_artifact_sha256,
                trace_ledger_sha256=result.trace_ledger_sha256,
                primary_log2_budget_auc=auc,
            )
        )
    return tuple(sorted(audits, key=lambda value: value.method_id))


def _validate_parent_evidence(
    evidence: V5K1PhaseCParentReplayEvidence,
    *,
    config: V5PaperBudgetEvaluationConfig,
) -> None:
    if not isinstance(evidence, V5K1PhaseCParentReplayEvidence):
        raise TypeError("parent evidence must be typed")
    provenance = evidence.provenance
    branches = tuple(evidence.branch_searches)
    expected_branches = {value.branch_id for value in K1_PHASE_C_BRANCHES}
    if len(branches) != len(expected_branches) or {
        value.branch_id for value in branches
    } != expected_branches:
        raise ValueError("parent replay requires exactly one completed sidecar for all 12 branches")
    if {value.frozen_search_yield_rank for value in branches} != set(
        range(1, len(expected_branches) + 1)
    ):
        raise ValueError("frozen search-yield branch ranks must be a complete permutation")
    branch_candidates = [candidate for branch in branches for candidate in branch.candidates]
    product_candidates = tuple(evidence.product_candidate_judgements)
    candidate_ids = [value.candidate_id for value in product_candidates]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("product candidate judgement IDs must be unique")
    branch_candidate_ids = [value.candidate_id for value in branch_candidates]
    if len(branch_candidate_ids) != len(set(branch_candidate_ids)):
        raise ValueError("candidate IDs must be unique across branch sidecars")
    if {value.candidate_id: value for value in branch_candidates} != {
        value.candidate_id: value for value in product_candidates
    }:
        raise ValueError("product candidate judgements must exactly equal all branch sidecars")
    if any(
        value.exact_judge_sha256 != provenance.exact_judge_sha256
        for value in (*branch_candidates, *product_candidates)
    ):
        raise ValueError("candidate judgement escaped the frozen exact judge")
    if any(
        value.exact_call_index > K1_PHASE_C_REFERENCE_EXACT_BUDGET
        for value in (*branch_candidates, *product_candidates)
    ):
        raise ValueError("candidate judgement escaped the frozen exact-call budget")

    references = evidence.reference_set
    bank = evidence.reference_bank
    if (
        bank.source_bundle_sha256 != provenance.source_bundle_sha256
        or bank.query_id != provenance.universal_query_sha256
        or bank.pairing_unit_id != provenance.clean_parent_sha256
        or bank.query_context_sha256 != provenance.evaluation_query_context_sha256
    ):
        raise ValueError("reference bank escaped the frozen source, query, parent, or context")
    if references.query_id != provenance.universal_query_sha256:
        raise ValueError("reference set escaped the frozen universal query")
    if references.pairing_unit_id != provenance.clean_parent_sha256:
        raise ValueError("reference set escaped the clean parent pairing unit")
    if references.query_context_sha256 != provenance.evaluation_query_context_sha256:
        raise ValueError("reference set escaped the frozen observation/query context")
    if references.reference_set_sha256 != bank.bank_artifact_sha256:
        raise ValueError("reference set SHA disagrees with the contextual bank")
    reference_ids = {value.representative_id for value in references.representatives}
    cluster_ids = {value.representative_id for value in bank.representative_clusters}
    if reference_ids != cluster_ids:
        raise ValueError("reference payloads do not equal the visible bank representatives")
    if dict(bank.representative_payload_sha256s) != {
        value.representative_id: value.payload.sha256 for value in references.representatives
    }:
        raise ValueError("reference bank does not bind the exact typed representative payloads")
    if bank.configured_exact_call_budget != K1_PHASE_C_REFERENCE_EXACT_BUDGET:
        raise ValueError("reference bank used the wrong exact-forward budget")
    if (
        bank.calibration_identity_sha256 != provenance.calibration_identity_sha256
        or bank.calibrated_threshold_sha256 != provenance.calibrated_threshold_sha256
        or bank.exact_judge_sha256 != provenance.exact_judge_sha256
    ):
        raise ValueError("reference bank escaped the frozen calibration or exact judge")

    methods = tuple(evidence.methods)
    if len(methods) != len(K1_PHASE_C_METHOD_IDS) or {
        value.trace.method_id for value in methods
    } != set(K1_PHASE_C_METHOD_IDS):
        raise ValueError("product, Sobol-only, and retrieval-only traces are all required")
    if {value.trace.exact_forward_call_budget for value in methods} != {
        K1_PHASE_C_REFERENCE_EXACT_BUDGET
    }:
        raise ValueError("paired method traces must use the identical frozen budget")
    if any(value.source_bundle_sha256_used != provenance.source_bundle_sha256 for value in methods):
        raise ValueError("method replay escaped the frozen source bundle")
    for value in methods:
        expected_model = (
            provenance.model_artifact_sha256
            if value.trace.method_id == K1_PHASE_C_PRODUCT_METHOD_ID
            else None
        )
        if value.model_artifact_sha256_used != expected_model:
            raise ValueError("method replay escaped the frozen product-model binding")
        expected_policy = (
            V5_PROPOSAL_EXECUTION_POLICY_SHA256
            if value.trace.method_id == K1_PHASE_C_PRODUCT_METHOD_ID
            else None
        )
        if value.proposal_execution_policy_sha256_used != expected_policy:
            raise ValueError("method replay escaped the frozen proposal-execution policy")
    product_trace = next(
        value.trace for value in methods if value.trace.method_id == K1_PHASE_C_PRODUCT_METHOD_ID
    )
    by_candidate = {value.candidate_id: value for value in product_candidates}
    emitted_clusters = []
    for emission in product_trace.candidate_emissions:
        judgement = by_candidate.get(emission.candidate_id)
        if judgement is None:
            raise ValueError("product emitted a candidate absent from all branch sidecars")
        parameter = emission.payload.parameter
        if (
            judgement.compatibility_status != emission.compatibility_status
            or judgement.exact_call_index != emission.available_after_call
            or judgement.bounds_compliant is not parameter.bounds_pass
            or judgement.physics_compliant is not parameter.physics_pass
        ):
            raise ValueError("product emission disagrees with its exact sidecar judgement")
        emitted_clusters.append(judgement.dedup_cluster_id)
    if len(emitted_clusters) != len(set(emitted_clusters)):
        raise ValueError("product trace exposes more than one representative from a dedup cluster")
    if config.output_caps[-1] != K1_PHASE_C_REFERENCE_OUTPUT_CAP or (
        config.exact_forward_budgets != EXACT_FORWARD_BUDGETS
    ):
        raise ValueError("paper evaluator did not use the frozen N/B grid")
    if (
        config.equivalence_threshold_sha256 != provenance.calibrated_threshold_sha256
        or config.equivalence_matcher_sha256 != V5_QUERY_PARAMETER_DISTANCE_SHA256
    ):
        raise ValueError("paper evaluator escaped the calibration or query-local distance")


def derive_v5_k1_phase_c_parent_record(
    evidence: V5K1PhaseCParentReplayEvidence,
    *,
    split_id: str,
    split_receipt_sha256: str,
    evaluator_config: V5PaperBudgetEvaluationConfig,
    equivalence_distance_matcher: EquivalenceDistanceMatcher,
) -> tuple[V5K1PhaseCParentRecord, V5PairedPaperBudgetQueryRecord]:
    """Recompute one parent record from raw sidecars, bank, and typed traces."""

    _validate_parent_evidence(evidence, config=evaluator_config)
    budget_record = evaluate_v5_paired_paper_budget_query(
        evidence.reference_set,
        tuple(value.trace for value in evidence.methods),
        config=evaluator_config,
        equivalence_distance_matcher=equivalence_distance_matcher,
    )
    product_result = next(
        value for value in budget_record.method_results if value.method_id == K1_PHASE_C_PRODUCT_METHOD_ID
    )
    cap_index = budget_record.output_caps.index(K1_PHASE_C_REFERENCE_OUTPUT_CAP)
    budget_index = budget_record.exact_forward_budgets.index(K1_PHASE_C_REFERENCE_EXACT_BUDGET)
    reference_hits = product_result.hit_count_matrix[cap_index][budget_index]

    branches = tuple(evidence.branch_searches)
    candidates = {value.candidate_id: value for value in evidence.product_candidate_judgements}
    product_trace = next(
        value.trace
        for value in evidence.methods
        if value.trace.method_id == K1_PHASE_C_PRODUCT_METHOD_ID
    )
    visible = tuple(
        candidates[value.candidate_id]
        for value in product_trace.candidate_emissions
        if value.output_rank <= K1_PHASE_C_REFERENCE_OUTPUT_CAP
        and value.available_after_call <= K1_PHASE_C_REFERENCE_EXACT_BUDGET
    )
    verified = tuple(
        value for value in visible if value.compatibility_status != V5_EXACT_UNVERIFIED
    )
    compatible = tuple(
        value for value in visible if value.compatibility_status == V5_EXACT_COMPATIBLE
    )
    positive_branches = {
        branch.branch_id
        for branch in branches
        if any(
            value.compatibility_status == V5_EXACT_COMPATIBLE for value in branch.candidates
        )
    }
    top_four_branches = {
        value.branch_id for value in branches if value.frozen_search_yield_rank <= K1_PHASE_C_TOP_K
    }
    cluster_ids = {value.dedup_cluster_id for value in compatible}
    payload = evidence.provenance.record_payload()
    payload.update(
        {
            "reference_set_sha256": evidence.reference_set.reference_set_sha256,
            "split_id": split_id,
            "split_disjointness_receipt_sha256": split_receipt_sha256,
            "split_disjointness_verified": True,
            "branch_catalog_size": len(K1_PHASE_C_BRANCHES),
            "completed_branch_outcomes": len(branches),
            "unverified_branch_outcomes": 0,
            "positive_branch_count": len(positive_branches),
            "matched_positive_branches_at_4": len(positive_branches & top_four_branches),
            "any_exact_compatible": bool(compatible),
            "reference_bank_qualified": True,
            "reference_bank_saturated": evidence.reference_bank.saturated,
            "reference_representative_count": len(evidence.reference_set.representatives),
            "matched_reference_representative_count_at_n16_b4096": reference_hits,
            "verified_candidate_count": len(verified),
            "bounds_compliant_candidate_count": sum(value.bounds_compliant for value in verified),
            "physics_compliant_candidate_count": sum(value.physics_compliant for value in verified),
            "amplitude_compliant_candidate_count": sum(
                value.amplitude_compliant for value in verified
            ),
            "compatible_candidate_count_before_dedup_at_n16_b4096": len(compatible),
            "duplicate_candidate_count_at_n16_b4096": len(compatible) - len(cluster_ids),
            "method_audits": _method_audits(budget_record, evidence),
        }
    )
    return V5K1PhaseCParentRecord(**payload), budget_record


def _replay_core(
    *, plan: V5K1PhaseCPlan, port: V5K1PhaseCReplayPort, contract: Mapping[str, object]
) -> tuple[dict[str, object], tuple[V5K1PhaseCParentRecord, ...]]:
    _require_formal_production_adapter(plan, port, contract)
    identity = _adapter_identity(port)
    frozen = validate_v5_k1_phase_c_contract(contract)
    validate_v5_k1_phase_c_plan(plan, contract=frozen)
    bundle = port.load_bundle(plan=plan, contract=frozen)
    if not isinstance(bundle, V5K1PhaseCReplayBundle):
        raise TypeError("replay port returned an untyped evidence bundle")
    revalidate = getattr(port, "revalidate_bundle", None)
    if not callable(revalidate):
        raise TypeError("replay port must revalidate immutable artifacts before and after replay")
    revalidate(bundle=bundle, plan=plan, contract=frozen)
    expected = (plan.sha256, frozen["contract_sha256"])
    if (bundle.plan_sha256, bundle.contract_sha256) != expected:
        raise ValueError("evidence bundle escaped the frozen plan or contract")
    if bundle.split_receipt.plan_sha256 != plan.sha256:
        raise ValueError("split receipt escaped the frozen plan")
    if bundle.split_receipt.split_id != K1_PHASE_C_SPLIT_ID:
        raise ValueError("split receipt uses the wrong independent split")
    if len(bundle.parents) != plan.total_parent_count:
        raise ValueError("evidence bundle parent count does not match the plan")
    parent_ids = tuple(value.provenance.clean_parent_sha256 for value in bundle.parents)
    if len(parent_ids) != len(set(parent_ids)):
        raise ValueError("evidence bundle repeats a clean parent")
    if set(parent_ids) != set(bundle.split_receipt.included_clean_parent_sha256s):
        raise ValueError("split receipt does not bind the exact replay parent cohort")
    if any(
        value.provenance.source_bundle_sha256 != bundle.artifact_binding.source_bundle_sha256
        or value.provenance.model_artifact_sha256
        != bundle.artifact_binding.model_artifact_sha256
        for value in bundle.parents
    ):
        raise ValueError("parent evidence escaped the frozen source or model")

    derived = tuple(
        derive_v5_k1_phase_c_parent_record(
            value,
            split_id=bundle.split_receipt.split_id,
            split_receipt_sha256=bundle.split_receipt.artifact_sha256,
            evaluator_config=bundle.evaluator_config,
            equivalence_distance_matcher=port.equivalence_distance_matcher,
        )
        for value in bundle.parents
    )
    records = tuple(value[0] for value in derived)
    budget_records = tuple(value[1] for value in derived)
    assessment = _assess_v5_k1_phase_c_records(records, plan=plan, contract=frozen)
    revalidate(bundle=bundle, plan=plan, contract=frozen)
    if plan.formal:
        from .k1_phase_c_filesystem_replay_v5 import (
            _authorize_v5_k1_phase_c_formal_filesystem_replay,
        )

        _authorize_v5_k1_phase_c_formal_filesystem_replay(
            port,
            bundle=bundle,
            plan=plan,
            contract=frozen,
        )
    claim_eligible = bool(
        plan.formal and assessment["full_k1_phase_c_generalization_gate_passed"]
    )
    core = {
        "schema": V5_K1_PHASE_C_REPLAY_RECEIPT_SCHEMA,
        "version": V5_K1_PHASE_C_REPLAY_RECEIPT_VERSION,
        "status": "complete",
        "formal": plan.formal,
        "claim_eligible": claim_eligible,
        "runner_role": (
            "formal_production_filesystem_replay"
            if plan.formal
            else "non_claiming_fixture_replay_only"
        ),
        "production_adapter_blocker": (
            None if plan.formal else V5_K1_PHASE_C_PRODUCTION_ADAPTER_BLOCKER
        ),
        "plan_sha256": plan.sha256,
        "contract_sha256": frozen["contract_sha256"],
        "artifact_binding": bundle.artifact_binding.audit_payload(),
        "adapter": identity,
        "evidence_bundle_sha256": bundle.sha256,
        "evidence_manifest": bundle.audit_payload(),
        "parent_replays": [
            {
                "parent_record": record.audit_payload(),
                "paper_budget_record": budget.audit_payload(),
                "paper_budget_record_sha256": budget.sha256,
            }
            for record, budget in zip(records, budget_records)
        ],
        "assessment": assessment,
    }
    return core, records


def _require_formal_production_adapter(
    plan: V5K1PhaseCPlan,
    port: V5K1PhaseCReplayPort,
    contract: Mapping[str, object],
) -> None:
    if not plan.formal:
        return
    from .k1_phase_c_filesystem_replay_v5 import (
        _is_v5_k1_phase_c_production_filesystem_adapter,
        _require_audited_v5_k1_phase_c_production_writer,
    )

    if not _is_v5_k1_phase_c_production_filesystem_adapter(port):
        raise RuntimeError(V5_K1_PHASE_C_PRODUCTION_ADAPTER_BLOCKER)
    _require_audited_v5_k1_phase_c_production_writer(
        port, plan=plan, contract=contract
    )


def _exclusive_completion_last_write(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"replay receipt already exists: {path}")
    encoded = (canonical_json(payload) + "\n").encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _read_bounded_regular_file(path: Path) -> str:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError("replay receipt must be a regular non-symlink file") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > _MAX_RECEIPT_BYTES:
            raise ValueError("replay receipt must be a bounded regular non-symlink file")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            encoded = stream.read(_MAX_RECEIPT_BYTES + 1)
        if len(encoded) > _MAX_RECEIPT_BYTES:
            raise ValueError("replay receipt exceeds the bounded size")
        return encoded.decode("utf-8")
    finally:
        os.close(descriptor)


def run_v5_k1_phase_c_replay(
    *,
    plan: V5K1PhaseCPlan,
    port: V5K1PhaseCReplayPort,
    receipt_path: str | Path,
    contract: Mapping[str, object] | None = None,
) -> Path:
    """Run fixture replay or a capability-authorized production formal replay."""

    frozen = v5_k1_phase_c_contract_payload() if contract is None else contract
    core, _ = _replay_core(plan=plan, port=port, contract=frozen)
    receipt = {**core, "receipt_sha256": sha256(canonical_json(core).encode()).hexdigest()}
    target = Path(receipt_path)
    _exclusive_completion_last_write(target, receipt)
    return target


def read_and_replay_v5_k1_phase_c_receipt(
    path: str | Path,
    *,
    plan: V5K1PhaseCPlan,
    port: V5K1PhaseCReplayPort,
    contract: Mapping[str, object] | None = None,
) -> V5K1PhaseCCheckedReplayReceipt:
    """Strictly read a receipt, reload all evidence through the port, and compare."""

    target = Path(path)
    loaded = json.loads(_read_bounded_regular_file(target), object_pairs_hook=_strict_object)
    if not isinstance(loaded, dict) or set(loaded) != _ROOT_FIELDS:
        raise ValueError("replay receipt root fields are incomplete or unsupported")
    supplied_sha = loaded.pop("receipt_sha256")
    expected_sha = sha256(canonical_json(loaded).encode()).hexdigest()
    if supplied_sha != expected_sha:
        raise ValueError("replay receipt SHA-256 does not reproduce")
    frozen = v5_k1_phase_c_contract_payload() if contract is None else contract
    replayed, records = _replay_core(plan=plan, port=port, contract=frozen)
    if loaded != replayed:
        raise ValueError("stored receipt does not exactly equal fresh evidence replay")
    return _mint_v5_k1_phase_c_checked_receipt(
        plan_sha256=plan.sha256,
        contract_sha256=str(replayed["contract_sha256"]),
        evidence_bundle_sha256=str(replayed["evidence_bundle_sha256"]),
        receipt_sha256=expected_sha,
        formal=plan.formal,
        records=records,
    )


__all__ = [
    "V5_K1_PHASE_C_PRODUCTION_ADAPTER_BLOCKER",
    "V5_K1_PHASE_C_REPLAY_RECEIPT_SCHEMA",
    "V5_K1_PHASE_C_REPLAY_RECEIPT_VERSION",
    "derive_v5_k1_phase_c_parent_record",
    "read_and_replay_v5_k1_phase_c_receipt",
    "run_v5_k1_phase_c_replay",
]
