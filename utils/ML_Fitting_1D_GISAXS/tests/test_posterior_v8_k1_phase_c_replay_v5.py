from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json

import numpy as np
import pytest

from utils.ML_Fitting_1D_GISAXS.PosteriorV8.contract import LatentComponentParameters
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.evaluation import (
    CandidateInput,
    LinearSolutionSnapshot,
    ReferenceMode,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.k1_phase_c_contract_v5 import (
    K1_PHASE_C_BRANCHES,
    K1_PHASE_C_PRODUCT_METHOD_ID,
    K1_PHASE_C_REFERENCE_EXACT_BUDGET,
    K1_PHASE_C_RETRIEVAL_BASELINE_ID,
    K1_PHASE_C_SOBOL_BASELINE_ID,
    K1_PHASE_C_SPLIT_ID,
    v5_k1_phase_c_contract_payload,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.k1_phase_c_evaluation_v5 import (
    assess_v5_k1_phase_c_records,
    assess_v5_k1_phase_c_replay_receipt,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.k1_phase_c_fixture_v5 import (
    build_v5_k1_phase_c_fixture_records,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.k1_phase_c_replay_contract_v5 import (
    V5_K1_PHASE_C_METHOD_COMPLETED,
    V5_K1_PHASE_C_METHOD_CRASHED_ZERO,
    V5K1PhaseCArtifactBinding,
    V5K1PhaseCBranchSearchReplay,
    V5K1PhaseCCandidateJudgement,
    V5K1PhaseCMethodReplay,
    V5K1PhaseCParentProvenance,
    V5K1PhaseCParentReplayEvidence,
    V5K1PhaseCReferenceBankReplay,
    V5K1PhaseCReferenceCluster,
    V5K1PhaseCReplayBundle,
    V5K1PhaseCSplitReplayReceipt,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.k1_phase_c_replay_receipt_v5 import (
    V5K1PhaseCCheckedReplayReceipt,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.k1_phase_c_replay_runner_v5 import (
    V5_K1_PHASE_C_PRODUCTION_ADAPTER_BLOCKER,
    V5_K1_PHASE_C_REPLAY_RECEIPT_SCHEMA,
    V5_K1_PHASE_C_REPLAY_RECEIPT_VERSION,
    _replay_core,
    derive_v5_k1_phase_c_parent_record,
    read_and_replay_v5_k1_phase_c_receipt,
    run_v5_k1_phase_c_replay,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.paper_budget_evaluator_v5 import (
    V5CandidateEmission,
    V5ExactForwardCall,
    V5FrozenReferenceRepresentative,
    V5FrozenReferenceSet,
    V5MethodExactCallTrace,
    V5PaperBudgetEvaluationConfig,
    V5PaperParameterRepresentativePayload,
    V5_EMITTED_REPRESENTATIVE_ROLE,
    V5_EXACT_COMPATIBLE,
    V5_REFERENCE_REPRESENTATIVE_ROLE,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.query_parameter_distance_v5 import (
    V5_QUERY_PARAMETER_DISTANCE_SHA256,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.proposal_execution_policy_v5 import (
    V5_PROPOSAL_EXECUTION_POLICY_SHA256,
)


def _sha(label: str) -> str:
    return sha256(label.encode()).hexdigest()


def _artifact_binding() -> V5K1PhaseCArtifactBinding:
    return V5K1PhaseCArtifactBinding(
        **{
            name: _sha(name)
            for name in (
                "source_archive_sha256",
                "source_manifest_sha256",
                "source_tree_sha256",
                "source_bundle_sha256",
                "cross_platform_gate_claim_sha256",
                "phase_a_launch_receipt_sha256",
                "model_artifact_sha256",
                "model_weights_sha256",
                "model_training_result_sha256",
            )
        }
    )


def _component() -> LatentComponentParameters:
    return LatentComponentParameters(shape="sphere", log_R=float(np.log(10.0)), sigma_R_fraction=0.1)


def _linear() -> LinearSolutionSnapshot:
    return LinearSolutionSnapshot(
        background=0.1, particle_amplitudes=(1.0,), resolution_amplitude=0.0, k=1.0
    )


def _reference_payload(reference_id: str, query_sha: str):
    return V5PaperParameterRepresentativePayload(
        representative_id=reference_id,
        role=V5_REFERENCE_REPRESENTATIVE_ROLE,
        parameter=ReferenceMode(
            reference_id=reference_id,
            topology_id=0,
            components=(_component(),),
            resolution=None,
            linear_solution=_linear(),
        ),
        global_branch_key="topology-00:wire-00",
        query_context_sha256=query_sha,
        source_artifact_sha256=_sha(f"reference:{reference_id}"),
    )


def _candidate_payload(candidate_id: str, query_sha: str):
    return V5PaperParameterRepresentativePayload(
        representative_id=candidate_id,
        role=V5_EMITTED_REPRESENTATIVE_ROLE,
        parameter=CandidateInput(
            candidate_id=candidate_id,
            proposal_rank=1,
            topology_id=0,
            components=(_component(),),
            resolution=None,
            linear_solution=_linear(),
            exact_intensity=np.asarray((1.0,), dtype=np.float64),
            bounds_pass=True,
            physics_pass=True,
        ),
        global_branch_key="topology-00:wire-00",
        query_context_sha256=query_sha,
        source_artifact_sha256=_sha(f"candidate:{candidate_id}"),
    )


@pytest.fixture(scope="module")
def exact_calls():
    return tuple(
        V5ExactForwardCall(exact_call_index=index, elapsed_seconds=float(index))
        for index in range(1, K1_PHASE_C_REFERENCE_EXACT_BUDGET + 1)
    )


def _replay_evidence(exact_calls):
    plan, fixture_records = build_v5_k1_phase_c_fixture_records()
    source = fixture_records[0]
    query_sha = source.universal_query_sha256
    clean_sha = source.clean_parent_sha256
    reference_sha = _sha("reference-bank")
    exact_judge_sha = _sha("exact-judge")
    config = V5PaperBudgetEvaluationConfig(
        comparison_protocol_id="phase-c-comparison",
        comparison_protocol_sha256=_sha("phase-c-comparison"),
        equivalence_matcher_id="query-local-distance",
        equivalence_matcher_sha256=V5_QUERY_PARAMETER_DISTANCE_SHA256,
        equivalence_threshold_id="calibrated-threshold",
        equivalence_threshold_sha256=_sha("calibrated-threshold"),
        maximum_normalized_distance=1.0,
    )
    references = V5FrozenReferenceSet(
        query_id=query_sha,
        pairing_unit_id=clean_sha,
        reference_set_id="phase-c-reference",
        reference_set_sha256=reference_sha,
        comparison_protocol_id=config.comparison_protocol_id,
        comparison_protocol_sha256=config.comparison_protocol_sha256,
        representatives=(
            V5FrozenReferenceRepresentative(
                representative_id="reference-1",
                payload=_reference_payload("reference-1", query_sha),
            ),
        ),
    )
    emission = V5CandidateEmission(
        available_after_call=1,
        output_rank=1,
        candidate_id="candidate-1",
        compatibility_status=V5_EXACT_COMPATIBLE,
        elapsed_seconds=1.0,
        payload=_candidate_payload("candidate-1", query_sha),
    )

    def trace(method_id, emissions=()):
        return V5MethodExactCallTrace(
            query_id=query_sha,
            pairing_unit_id=clean_sha,
            method_id=method_id,
            method_protocol_id=f"{method_id}-protocol",
            method_protocol_sha256=_sha(f"{method_id}-protocol"),
            trace_id=f"{method_id}-trace",
            trace_artifact_sha256=_sha(f"{method_id}-trace"),
            reference_set_id=references.reference_set_id,
            reference_set_sha256=reference_sha,
            comparison_protocol_id=config.comparison_protocol_id,
            comparison_protocol_sha256=config.comparison_protocol_sha256,
            exact_forward_call_budget=K1_PHASE_C_REFERENCE_EXACT_BUDGET,
            exact_forward_calls=exact_calls,
            candidate_emissions=tuple(emissions),
        )

    judgement = V5K1PhaseCCandidateJudgement(
        candidate_id="candidate-1",
        exact_call_index=1,
        compatibility_status=V5_EXACT_COMPATIBLE,
        bounds_compliant=True,
        physics_compliant=True,
        amplitude_compliant=True,
        dedup_cluster_id="product-cluster-1",
        exact_judge_sha256=exact_judge_sha,
    )
    reference_judgement = replace(
        judgement,
        candidate_id="reference-1",
        dedup_cluster_id="reference-1",
    )
    searches = tuple(
        V5K1PhaseCBranchSearchReplay(
            branch_id=branch.branch_id,
            search_sidecar_sha256=_sha(f"sidecar:{branch.branch_id}"),
            frozen_search_yield_rank=index + 1,
            completed=True,
            candidates=(judgement,) if index == 0 else (),
        )
        for index, branch in enumerate(K1_PHASE_C_BRANCHES)
    )
    provenance_payload = {
        name: getattr(source, name)
        for name in V5K1PhaseCParentProvenance.__dataclass_fields__
        if hasattr(source, name)
    }
    provenance_payload.update(
        clean_recipe_artifact_sha256=_sha("clean-recipe"),
        query_ranges_artifact_sha256=_sha("query-ranges"),
        observation_artifact_sha256=_sha("observation"),
        evaluation_query_context_sha256=query_sha,
        calibration_identity_sha256=_sha("calibration"),
        calibrated_threshold_sha256=_sha("calibrated-threshold"),
        exact_judge_sha256=exact_judge_sha,
    )
    evidence = V5K1PhaseCParentReplayEvidence(
        provenance=V5K1PhaseCParentProvenance(**provenance_payload),
        branch_searches=searches,
        product_candidate_judgements=(judgement,),
        reference_bank=V5K1PhaseCReferenceBankReplay(
            bank_artifact_sha256=reference_sha,
            search_trace_artifact_sha256=_sha("reference-search-trace"),
            calibration_identity_sha256=_sha("calibration"),
            calibrated_threshold_sha256=_sha("calibrated-threshold"),
            exact_judge_sha256=exact_judge_sha,
            source_bundle_sha256=source.source_bundle_sha256,
            query_id=query_sha,
            pairing_unit_id=clean_sha,
            query_context_sha256=query_sha,
            candidate_ids=("reference-1",),
            exact_compatible_candidate_ids=("reference-1",),
            candidate_judgements=(reference_judgement,),
            representative_clusters=(
                V5K1PhaseCReferenceCluster(
                    representative_id="reference-1", member_candidate_ids=("reference-1",)
                ),
            ),
            representative_payload_sha256s=(
                ("reference-1", references.representatives[0].payload.sha256),
            ),
            configured_exact_call_budget=K1_PHASE_C_REFERENCE_EXACT_BUDGET,
            consumed_exact_calls=K1_PHASE_C_REFERENCE_EXACT_BUDGET,
            exact_forward_calls=exact_calls,
            enumeration_complete=False,
            network_free=True,
        ),
        reference_set=references,
        methods=(
            V5K1PhaseCMethodReplay(
                status=V5_K1_PHASE_C_METHOD_COMPLETED,
                trace=trace(K1_PHASE_C_PRODUCT_METHOD_ID, (emission,)),
                source_bundle_sha256_used=source.source_bundle_sha256,
                model_artifact_sha256_used=source.model_artifact_sha256,
                proposal_execution_policy_sha256_used=(
                    V5_PROPOSAL_EXECUTION_POLICY_SHA256
                ),
            ),
            V5K1PhaseCMethodReplay(
                status=V5_K1_PHASE_C_METHOD_CRASHED_ZERO,
                trace=trace(K1_PHASE_C_SOBOL_BASELINE_ID),
                source_bundle_sha256_used=source.source_bundle_sha256,
                model_artifact_sha256_used=None,
                proposal_execution_policy_sha256_used=None,
            ),
            V5K1PhaseCMethodReplay(
                status=V5_K1_PHASE_C_METHOD_CRASHED_ZERO,
                trace=trace(K1_PHASE_C_RETRIEVAL_BASELINE_ID),
                source_bundle_sha256_used=source.source_bundle_sha256,
                model_artifact_sha256_used=None,
                proposal_execution_policy_sha256_used=None,
            ),
        ),
    )
    return plan, source, config, evidence


def test_parent_counts_and_typed_budget_record_are_recomputed(exact_calls) -> None:
    _, source, config, evidence = _replay_evidence(exact_calls)
    record, budget = derive_v5_k1_phase_c_parent_record(
        evidence,
        split_id=source.split_id,
        split_receipt_sha256=source.split_disjointness_receipt_sha256,
        evaluator_config=config,
        equivalence_distance_matcher=lambda _reference, _candidate: 0.0,
    )

    assert record.completed_branch_outcomes == 12
    assert record.positive_branch_count == record.matched_positive_branches_at_4 == 1
    assert record.verified_candidate_count == record.compatible_candidate_count_before_dedup_at_n16_b4096 == 1
    assert record.matched_reference_representative_count_at_n16_b4096 == 1
    assert budget.method_results[0].exact_forward_calls_evaluated == 4096
    baseline_auc = {
        audit.method_id: audit.primary_log2_budget_auc for audit in record.method_audits
    }
    assert baseline_auc[K1_PHASE_C_SOBOL_BASELINE_ID] == 0.0
    assert baseline_auc[K1_PHASE_C_RETRIEVAL_BASELINE_ID] == 0.0


def test_missing_branch_and_unequal_budget_fail_closed(exact_calls) -> None:
    _, source, config, evidence = _replay_evidence(exact_calls)
    kwargs = {
        "split_id": source.split_id,
        "split_receipt_sha256": source.split_disjointness_receipt_sha256,
        "evaluator_config": config,
        "equivalence_distance_matcher": lambda _reference, _candidate: 0.0,
    }
    with pytest.raises(ValueError, match="all 12 branches"):
        derive_v5_k1_phase_c_parent_record(
            replace(evidence, branch_searches=evidence.branch_searches[:-1]), **kwargs
        )

    changed_trace = replace(
        evidence.methods[-1].trace,
        exact_forward_call_budget=4095,
        exact_forward_calls=exact_calls[:-1],
    )
    changed_methods = (*evidence.methods[:-1], replace(evidence.methods[-1], trace=changed_trace))
    with pytest.raises(ValueError, match="identical frozen budget"):
        derive_v5_k1_phase_c_parent_record(replace(evidence, methods=changed_methods), **kwargs)
    with pytest.raises(ValueError, match="complete, ordered, and contiguous"):
        replace(evidence.methods[0].trace, exact_forward_calls=exact_calls[1:])
    assert evidence.branch_searches[0].sidecar_schema.endswith("/v5")
    with pytest.raises(ValueError, match="current search-sidecar"):
        replace(evidence.branch_searches[0], sidecar_schema="legacy/v4")


def test_reference_bank_rejects_hidden_members_and_legacy_distance() -> None:
    values = {
        "bank_artifact_sha256": _sha("bank"),
        "search_trace_artifact_sha256": _sha("reference-search-trace"),
        "calibration_identity_sha256": _sha("calibration"),
        "calibrated_threshold_sha256": _sha("threshold"),
        "exact_judge_sha256": _sha("judge"),
        "source_bundle_sha256": _sha("source"),
        "query_id": _sha("query"),
        "pairing_unit_id": _sha("parent"),
        "query_context_sha256": _sha("context"),
        "candidate_ids": ("visible",),
        "exact_compatible_candidate_ids": ("visible",),
        "candidate_judgements": (
            V5K1PhaseCCandidateJudgement(
                candidate_id="visible",
                exact_call_index=1,
                compatibility_status=V5_EXACT_COMPATIBLE,
                bounds_compliant=True,
                physics_compliant=True,
                amplitude_compliant=True,
                dedup_cluster_id="visible",
                exact_judge_sha256=_sha("judge"),
            ),
        ),
        "representative_payload_sha256s": (("visible", _sha("payload")),),
        "configured_exact_call_budget": 4,
        "consumed_exact_calls": 4,
        "exact_forward_calls": tuple(
            V5ExactForwardCall(exact_call_index=index, elapsed_seconds=float(index))
            for index in range(1, 5)
        ),
        "enumeration_complete": False,
        "network_free": True,
    }
    with pytest.raises(ValueError, match="exact candidate partition"):
        V5K1PhaseCReferenceBankReplay(
            representative_clusters=(
                V5K1PhaseCReferenceCluster(
                    representative_id="visible", member_candidate_ids=("visible", "hidden")
                ),
            ),
            **values,
        )
    with pytest.raises(ValueError, match="legacy global"):
        V5K1PhaseCReferenceBankReplay(
            representative_clusters=(
                V5K1PhaseCReferenceCluster(
                    representative_id="visible", member_candidate_ids=("visible",)
                ),
            ),
            distance_version="legacy_global_parameter_distance/v1",
            **values,
        )
    with pytest.raises(ValueError, match="bind every candidate"):
        V5K1PhaseCReferenceBankReplay(
            representative_clusters=(
                V5K1PhaseCReferenceCluster(
                    representative_id="visible", member_candidate_ids=("visible",)
                ),
            ),
            **{**values, "candidate_judgements": ()},
        )


def test_raw_emissions_are_persisted_and_non_emitted_candidates_do_not_inflate_counts(
    exact_calls,
) -> None:
    _, source, config, evidence = _replay_evidence(exact_calls)
    hidden = replace(
        evidence.product_candidate_judgements[0],
        candidate_id="candidate-never-emitted",
        dedup_cluster_id="hidden-cluster",
    )
    branch = evidence.branch_searches[0]
    expanded = replace(
        evidence,
        branch_searches=(replace(branch, candidates=(*branch.candidates, hidden)), *evidence.branch_searches[1:]),
        product_candidate_judgements=(*evidence.product_candidate_judgements, hidden),
    )
    record, _ = derive_v5_k1_phase_c_parent_record(
        expanded,
        split_id=source.split_id,
        split_receipt_sha256=source.split_disjointness_receipt_sha256,
        evaluator_config=config,
        equivalence_distance_matcher=lambda _reference, _candidate: 0.0,
    )
    assert record.compatible_candidate_count_before_dedup_at_n16_b4096 == 1
    method_payload = expanded.methods[0].audit_payload()
    emission = method_payload["raw_typed_candidate_emissions"][0]
    assert emission["candidate_id"] == "candidate-1"
    assert emission["payload"]["canonical_parameter"]["topology_id"] == 0


def test_branch_union_reference_payload_and_model_bindings_fail_closed(exact_calls) -> None:
    _, source, config, evidence = _replay_evidence(exact_calls)
    kwargs = {
        "split_id": source.split_id,
        "split_receipt_sha256": source.split_disjointness_receipt_sha256,
        "evaluator_config": config,
        "equivalence_distance_matcher": lambda _reference, _candidate: 0.0,
    }
    with pytest.raises(ValueError, match="exactly equal all branch sidecars"):
        derive_v5_k1_phase_c_parent_record(
            replace(evidence, branch_searches=tuple(replace(value, candidates=()) for value in evidence.branch_searches)),
            **kwargs,
        )
    with pytest.raises(ValueError, match="typed representative payloads"):
        derive_v5_k1_phase_c_parent_record(
            replace(
                evidence,
                reference_bank=replace(
                    evidence.reference_bank,
                    representative_payload_sha256s=(("reference-1", _sha("wrong-payload")),),
                ),
            ),
            **kwargs,
        )
    with pytest.raises(ValueError, match="product-model binding"):
        derive_v5_k1_phase_c_parent_record(
            replace(
                evidence,
                methods=(
                    replace(evidence.methods[0], model_artifact_sha256_used=_sha("wrong-model")),
                    *evidence.methods[1:],
                ),
            ),
            **kwargs,
        )
    with pytest.raises(ValueError, match="proposal-execution policy"):
        derive_v5_k1_phase_c_parent_record(
            replace(
                evidence,
                methods=(
                    replace(
                        evidence.methods[0],
                        proposal_execution_policy_sha256_used=_sha("wrong-policy"),
                    ),
                    *evidence.methods[1:],
                ),
            ),
            **kwargs,
        )


def test_artifact_binding_rejects_stale_model_contract() -> None:
    values = {
        name: _sha(name)
        for name in (
            "source_archive_sha256",
            "source_manifest_sha256",
            "source_tree_sha256",
            "source_bundle_sha256",
            "cross_platform_gate_claim_sha256",
            "phase_a_launch_receipt_sha256",
            "model_artifact_sha256",
            "model_weights_sha256",
            "model_training_result_sha256",
        )
    }
    binding = V5K1PhaseCArtifactBinding(**values)
    assert binding.model_version.endswith("v5_2_r2")
    assert binding.cross_platform_manifest_schema.endswith("/v2")
    assert binding.phase_a_launch_schema.endswith("/v7")
    assert binding.proposal_execution_policy_sha256 == V5_PROPOSAL_EXECUTION_POLICY_SHA256
    assert binding.universal_inference_schema.endswith("/v3")
    with pytest.raises(ValueError, match="stale source/model contract"):
        V5K1PhaseCArtifactBinding(**values, model_contract_sha256=_sha("stale"))


def test_replay_port_must_revalidate_immutable_artifacts(exact_calls) -> None:
    plan, _, config, evidence = _replay_evidence(exact_calls)
    contract = v5_k1_phase_c_contract_payload()
    binding = _artifact_binding()
    provenance = replace(
        evidence.provenance,
        source_bundle_sha256=binding.source_bundle_sha256,
        model_artifact_sha256=binding.model_artifact_sha256,
    )
    methods = tuple(
        replace(
            value,
            source_bundle_sha256_used=binding.source_bundle_sha256,
            model_artifact_sha256_used=(
                binding.model_artifact_sha256
                if value.trace.method_id == K1_PHASE_C_PRODUCT_METHOD_ID
                else None
            ),
            proposal_execution_policy_sha256_used=(
                V5_PROPOSAL_EXECUTION_POLICY_SHA256
                if value.trace.method_id == K1_PHASE_C_PRODUCT_METHOD_ID
                else None
            ),
        )
        for value in evidence.methods
    )
    parent = replace(
        evidence,
        provenance=provenance,
        reference_bank=replace(
            evidence.reference_bank,
            source_bundle_sha256=binding.source_bundle_sha256,
        ),
        methods=methods,
    )
    bundle = V5K1PhaseCReplayBundle(
        plan_sha256=plan.sha256,
        contract_sha256=contract["contract_sha256"],
        artifact_binding=binding,
        split_receipt=V5K1PhaseCSplitReplayReceipt(
            split_id=K1_PHASE_C_SPLIT_ID,
            artifact_sha256=_sha("split"),
            plan_sha256=plan.sha256,
            included_clean_parent_sha256s=(parent.provenance.clean_parent_sha256,),
            excluded_population_sha256s=(_sha("excluded"),),
            disjointness_verified=True,
        ),
        evaluator_config=config,
        parents=(parent,),
    )
    assert len(bundle.sha256) == 64

    class NoRevalidationPort:
        adapter_id = "no-revalidation"
        adapter_version = "v1"
        equivalence_distance_matcher = staticmethod(lambda _reference, _candidate: 0.0)

        def load_bundle(self, **_kwargs):
            return bundle

    with pytest.raises(TypeError, match="must revalidate"):
        _replay_core(plan=plan, port=NoRevalidationPort(), contract=contract)


def test_formal_raw_records_and_unimplemented_production_runner_are_blocked(tmp_path) -> None:
    _, records = build_v5_k1_phase_c_fixture_records()
    from utils.ML_Fitting_1D_GISAXS.PosteriorV8.k1_phase_c_plan_v5 import build_v5_k1_phase_c_plan

    formal = build_v5_k1_phase_c_plan(formal=True)
    with pytest.raises(ValueError, match="manually assembled"):
        assess_v5_k1_phase_c_records(records, plan=formal)
    forged = V5K1PhaseCCheckedReplayReceipt(
        plan_sha256=formal.sha256,
        contract_sha256=formal.contract_sha256,
        evidence_bundle_sha256=_sha("forged-bundle"),
        receipt_sha256=_sha("forged-receipt"),
        formal=True,
        records=records,
        _seal=object(),
    )
    with pytest.raises(TypeError, match="runner-checked"):
        assess_v5_k1_phase_c_replay_receipt(forged, plan=formal)

    class MustNotLoad:
        adapter_id = "must-not-load"
        adapter_version = "v1"

        def load_bundle(self, **_kwargs):
            raise AssertionError("formal blocker must run before artifact loading")

    target = tmp_path / "formal-receipt.json"
    with pytest.raises(RuntimeError, match=V5_K1_PHASE_C_PRODUCTION_ADAPTER_BLOCKER):
        run_v5_k1_phase_c_replay(plan=formal, port=MustNotLoad(), receipt_path=target)
    assert not target.exists()


def test_receipt_is_completion_last_non_overwrite_and_replayed(monkeypatch, tmp_path) -> None:
    plan, records = build_v5_k1_phase_c_fixture_records()
    contract = v5_k1_phase_c_contract_payload()
    core = {
        "schema": V5_K1_PHASE_C_REPLAY_RECEIPT_SCHEMA,
        "version": V5_K1_PHASE_C_REPLAY_RECEIPT_VERSION,
        "status": "complete",
        "formal": False,
        "claim_eligible": False,
        "runner_role": "non_claiming_fixture_replay_only",
        "production_adapter_blocker": V5_K1_PHASE_C_PRODUCTION_ADAPTER_BLOCKER,
        "plan_sha256": plan.sha256,
        "contract_sha256": contract["contract_sha256"],
        "artifact_binding": {"fixture": True},
        "adapter": {"adapter_id": "fake", "adapter_version": "v1"},
        "evidence_bundle_sha256": _sha("bundle"),
        "evidence_manifest": {},
        "parent_replays": [],
        "assessment": {},
    }

    class FakePort:
        adapter_id = "fake"
        adapter_version = "v1"

    monkeypatch.setattr(
        "utils.ML_Fitting_1D_GISAXS.PosteriorV8.k1_phase_c_replay_runner_v5._replay_core",
        lambda **_kwargs: (core, records),
    )
    target = tmp_path / "fixture-receipt.json"
    run_v5_k1_phase_c_replay(plan=plan, port=FakePort(), receipt_path=target)
    checked = read_and_replay_v5_k1_phase_c_receipt(target, plan=plan, port=FakePort())
    assert checked.records == records
    assert assess_v5_k1_phase_c_replay_receipt(checked, plan=plan)["fixture_gate_passed"]
    alias = tmp_path / "receipt-alias.json"
    alias.symlink_to(target)
    with pytest.raises(ValueError, match="non-symlink"):
        read_and_replay_v5_k1_phase_c_receipt(alias, plan=plan, port=FakePort())
    with pytest.raises(FileExistsError):
        run_v5_k1_phase_c_replay(plan=plan, port=FakePort(), receipt_path=target)

    payload = json.loads(target.read_text())
    payload["assessment"] = {"manually_forged_count": 999}
    payload.pop("receipt_sha256")
    from utils.ML_Fitting_1D_GISAXS.PosteriorV8.grouped_artifact_v5 import canonical_json

    payload["receipt_sha256"] = sha256(canonical_json(payload).encode()).hexdigest()
    target.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="fresh evidence replay"):
        read_and_replay_v5_k1_phase_c_receipt(target, plan=plan, port=FakePort())

    payload["status"] = "tampered"
    target.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="SHA-256"):
        read_and_replay_v5_k1_phase_c_receipt(target, plan=plan, port=FakePort())
