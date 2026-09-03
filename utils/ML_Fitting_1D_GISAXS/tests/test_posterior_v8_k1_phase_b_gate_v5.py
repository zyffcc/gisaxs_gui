from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import pytest

from utils.ML_Fitting_1D_GISAXS.PosteriorV8.grouped_artifact_v5 import canonical_json
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.k1_phase_b_contract_v5 import (
    PHASE_A_SOURCE_PATHS,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.model_v5_contract import (
    MODEL_V5_NAME,
    MODEL_V5_SCHEMA,
    MODEL_V5_VERSION,
    model_v5_contract_payload,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.run_k1_memorization_gate_v5 import (
    V5_K1_MODEL_PROVENANCE_SCHEMA,
    V5_K1_MODEL_PROVENANCE_VERSION,
    V5_K1_TRAINING_EVIDENCE_SCHEMA,
    V5_K1_TRAINING_EVIDENCE_VERSION,
    V5_K1_DATASET_GATE_ROLE,
    V5_K1_DATASET_GATE_SCHEMA,
    V5_K1_DATASET_GATE_VERSION,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.run_k1_phase_b_gate_v5 import (
    V5_K1_PHASE_B_ROLE,
    V5_K1_PHASE_B_VERSION,
    V5K1PhaseBGateConfig,
    V5K1PhaseBParentRecord,
    assess_v5_k1_phase_b_records,
    validate_v5_k1_phase_a_bindings,
)


SLURM_WRAPPER = (
    Path(__file__).resolve().parents[1]
    / "PosteriorV8/slurm/v5_k1_phase_b_gate_cpu.sbatch"
)
GATES = {
    "branch_conditioned_local_mdn_single_draw_local_rms_median_lt": 0.05,
    "branch_conditioned_local_mdn_best_of_32_local_rms_median_lt": 0.01,
    "branch_conditioned_local_mdn_best_of_32_local_rms_p90_lt": 0.03,
    "exact_post_refine_raw_log_rmse_p90_lt": 1.0e-3,
    "exact_post_refine_compatible_rate_gte": 0.99,
}


def _contract(recipes: int = 2) -> dict[str, object]:
    return {
        "schema_version": "protocol-schema",
        "protocol_version": "protocol-version",
        "protocol_sha256": "f" * 64,
        "json_pointer": "/stages/k1_memorization/gates",
        "recipes": recipes,
        "training_augmentation_views": 1,
        "topology_schedule": "single_branch_sphere_pattern0",
        "authoritative_forward_version": "forward-version",
        "gates": dict(GATES),
    }


def _record(index: int = 0, **changes) -> V5K1PhaseBParentRecord:
    base = V5K1PhaseBParentRecord(
        clean_parent_sha256=f"{index + 1:064x}",
        candidate_query_sha256="2" * 64,
        geometry_query_sha256="3" * 64,
        amplitude_query_sha256="4" * 64,
        topology_id=0,
        pattern_id=0,
        frozen_seed=100 + index,
        varying_dimension_count=2,
        proposal_count=32,
        single_draw_index=1,
        single_draw_local_rms=0.02,
        best_of_32_local_rms=0.005,
        exact_best_raw_log_rmse=2.0e-4,
        exact_compatible=True,
        refinement_status="exact_candidates_ready_for_verification",
        all_input_seeds_processed=True,
        configured_total_limit=4096,
        configured_per_candidate_limit=128,
        exact_calls_used=64,
        exact_calls_remaining=4032,
        exact_calls_by_phase=(
            ("seed_profile_verification", 32),
            ("initial_profile_verification", 0),
            ("optimizer_residual", 0),
            ("optimizer_terminal_verification", 32),
        ),
        attempts_recorded=32,
        refinement_successes=32,
        validation_failures=0,
        refinement_failures=0,
        per_candidate_budget_exhausted_attempts=0,
        total_budget_exhausted_before_seed_attempts=0,
        bounds_compliant_attempts=32,
        physics_compliant_attempts=32,
        amplitude_compliant_attempts=32,
        model_output_sha256="5" * 64,
    )
    return replace(base, **changes)


def test_single_branch_phase_b_success_evaluates_all_dynamic_gates_but_not_full_k1():
    result = assess_v5_k1_phase_b_records(
        (_record(0), _record(1)), gate_contract=_contract()
    )

    assert result["single_branch_phase_b_gate_passed"] is True
    assert result["complete_k1_proposal_exact_gate_passed"] is False
    assert result["full_k1_all_legal_branches_gate_status"].startswith("pending_fail_closed")
    assert set(result["gate_decisions"]) == set(GATES)
    assert all(value["passed"] for value in result["gate_decisions"].values())
    assert result["metrics"][
        "branch_conditioned_local_mdn_single_draw_local_rms_median"
    ] == pytest.approx(0.02)
    assert result["exact_forward_ledger"]["input_seed_count"] == 64
    assert result["bounds_and_physics_compliance"]["bounds_compliance_fraction"] == 1.0


def test_partial_or_unverified_parent_fails_every_gate_closed():
    partial = _record(
        1,
        refinement_status="exact_candidates_partial_budget_exhausted",
        all_input_seeds_processed=False,
        attempts_recorded=31,
        refinement_successes=31,
        total_budget_exhausted_before_seed_attempts=0,
        bounds_compliant_attempts=31,
        physics_compliant_attempts=31,
        amplitude_compliant_attempts=31,
    )
    result = assess_v5_k1_phase_b_records(
        (_record(0), partial), gate_contract=_contract()
    )

    assert result["integrity_passed"] is False
    assert result["single_branch_phase_b_gate_passed"] is False
    assert not any(value["passed"] for value in result["gate_decisions"].values())
    assert result["per_parent_failure_reasons"][partial.clean_parent_sha256]


def test_duplicate_exact_phase_entries_do_not_hide_a_broken_ledger():
    duplicate = _record(
        1,
        exact_calls_by_phase=(
            ("seed_profile_verification", 16),
            ("seed_profile_verification", 16),
            ("optimizer_residual", 0),
            ("optimizer_terminal_verification", 32),
        ),
    )
    result = assess_v5_k1_phase_b_records(
        (_record(0), duplicate), gate_contract=_contract()
    )

    assert result["single_branch_phase_b_gate_passed"] is False
    assert "exact_call_phase_ledger_does_not_reconcile" in result[
        "per_parent_failure_reasons"
    ][duplicate.clean_parent_sha256]


def test_engineering_parent_subset_never_becomes_a_gate_pass():
    result = assess_v5_k1_phase_b_records(
        (_record(0), _record(1)),
        gate_contract=_contract(),
        engineering_subset=True,
    )
    assert result["single_branch_phase_b_gate_passed"] is False
    assert "engineering_parent_subset_cannot_pass_the_formal_gate" in result[
        "integrity_failure_reasons"
    ]


def test_budget_starvation_and_nan_are_fail_closed():
    with pytest.raises(ValueError, match="full configured limit"):
        V5K1PhaseBGateConfig(
            per_candidate_forward_evaluation_limit=128,
            per_parent_forward_evaluation_limit=4095,
        )

    starved = _record(
        1,
        configured_total_limit=64,
        exact_calls_used=64,
        exact_calls_remaining=0,
        per_candidate_budget_exhausted_attempts=1,
        refinement_successes=31,
        bounds_compliant_attempts=31,
        physics_compliant_attempts=31,
        amplitude_compliant_attempts=31,
    )
    result = assess_v5_k1_phase_b_records(
        (_record(0), starved), gate_contract=_contract()
    )
    assert result["single_branch_phase_b_gate_passed"] is False
    assert "one_or_more_clean_parent_records_are_partial_or_invalid" in result[
        "integrity_failure_reasons"
    ]

    nan_record = _record(1, exact_best_raw_log_rmse=float("nan"), exact_compatible=False)
    nan_result = assess_v5_k1_phase_b_records(
        (_record(0), nan_record), gate_contract=_contract()
    )
    assert nan_result["metrics"]["exact_post_refine_raw_log_rmse_p90"] is None
    assert nan_result["single_branch_phase_b_gate_passed"] is False


def _phase_a_fixture():
    contract = _contract()
    dataset_identity = {
        "path": "/data/dust/user/zhaiyufe/dataset.gvd5",
        "dataset_id": "dataset-id",
        "dataset_schema": "dataset-schema",
        "dataset_version": "dataset-version",
        "dataset_manifest_sha256": "6" * 64,
        "dataset_file_sha256": "7" * 64,
        "dataset_file_byte_count": 123,
        "dataset_source_sha256": {"source.py": "8" * 64},
    }
    source_files = {
        name: {"sha256": "9" * 64, "byte_count": 10}
        for name in PHASE_A_SOURCE_PATHS
    }
    source = {
        "source_root": "/data/dust/user/zhaiyufe/source",
        "files": source_files,
        "bundle_sha256": sha256(canonical_json(source_files).encode()).hexdigest(),
    }
    model_identity = {"sha256": "a" * 64, "byte_count": 456}
    training_evidence_core = {
        "schema": V5_K1_TRAINING_EVIDENCE_SCHEMA,
        "version": V5_K1_TRAINING_EVIDENCE_VERSION,
        "status": "VALIDATED",
        "source_snapshot": {
            "source_archive_sha256": "1" * 64,
            "source_manifest_sha256": "2" * 64,
            "source_tree_sha256": "3" * 64,
        },
        "dataset_completion_binding": {
            "binding_sha256": "4" * 64,
            "file": {"sha256": "5" * 64, "byte_count": 12, "mode": 256},
            "dataset": {
                "original_path": "/data/dust/user/zhaiyufe/dataset.gvd5",
                "artifact_sha256": dataset_identity["dataset_file_sha256"],
                "byte_count": dataset_identity["dataset_file_byte_count"],
                "dataset_id": dataset_identity["dataset_id"],
                "dataset_schema": dataset_identity["dataset_schema"],
                "dataset_version": dataset_identity["dataset_version"],
                "manifest_sha256": dataset_identity["dataset_manifest_sha256"],
            },
        },
        "cross_platform_gate": {
            "gate_claim_sha256": "6" * 64,
            "pass_marker_sha256": "7" * 64,
            "pass_marker_file": {"sha256": "8" * 64, "byte_count": 13, "mode": 256},
            "reference_file_sha256": "9" * 64,
            "reference_file_byte_count": 14,
            "reference_manifest_sha256": "c" * 64,
            "scientific_content_sha256": "d" * 64,
            "comparison_result_sha256": "e" * 64,
        },
        "worker_validation": (
            "strict_local_dataset_binding_and_pass_marker_replayed_by_training_process"
        ),
    }
    training_evidence = {
        **training_evidence_core,
        "evidence_sha256": sha256(
            canonical_json(training_evidence_core).encode()
        ).hexdigest(),
    }
    model_provenance_core = {
        "schema": V5_K1_MODEL_PROVENANCE_SCHEMA,
        "version": V5_K1_MODEL_PROVENANCE_VERSION,
        "status": "BOUND",
        "model": {
            "filename": "model.keras",
            **model_identity,
            "reload_graph_contract_passed": True,
        },
        "training_evidence": training_evidence,
    }
    model_provenance = {
        **model_provenance_core,
        "binding_sha256": sha256(
            canonical_json(model_provenance_core).encode()
        ).hexdigest(),
    }
    model_provenance_identity = {"sha256": "f" * 64, "byte_count": 789}
    weights_sha = "b" * 64
    nested_core = {
        "schema_version": "gisaxs.posterior_v8.memorization_gate/v1",
        "version": "single_recipe_single_mdn_local_target_diagnostic_v1",
        "scientific_role": "wiring_and_memorization_diagnostic_not_model_acceptance",
        "initial_loss": 2.0,
        "final_loss": 0.1,
        "initial_target_median_rms": 0.4,
        "final_target_median_rms": 0.01,
        "target_count": 2,
        "varying_coordinate_count": 4,
        "model_input_keys": [],
        "objective_audit_sha256": "c" * 64,
        "trajectory_sha256": "d" * 64,
        "final_weights_sha256": weights_sha,
        "passed": True,
        "config": {
            "max_final_target_median_rms": 0.02,
            "minimum_loss_reduction": 0.5,
        },
    }
    nested = {
        **nested_core,
        "result_sha256": sha256(canonical_json(nested_core).encode()).hexdigest(),
    }
    payload_core = {
        "schema_version": V5_K1_DATASET_GATE_SCHEMA,
        "version": V5_K1_DATASET_GATE_VERSION,
        "scientific_role": V5_K1_DATASET_GATE_ROLE,
        "model_acceptance_evidence": False,
        "requested_config": {"smoke": False},
        "resolved_gate_config": {},
        "resolved_model_config": {"mixture_components": 1},
        "live_model_identity": {
            "schema_version": MODEL_V5_SCHEMA,
            "model_version": MODEL_V5_VERSION,
            "model_name": MODEL_V5_NAME,
        },
        "model_contract": model_v5_contract_payload(),
        "single_branch_k1_gate_contract": {
            "criteria_source": {
                name: contract[name]
                for name in (
                    "schema_version",
                    "protocol_version",
                    "protocol_sha256",
                    "json_pointer",
                )
            },
            "phase_b_frozen_criteria": dict(GATES),
            "topology_schedule": "single_branch_sphere_pattern0",
            "authoritative_forward_version": "forward-version",
            "stage_a_evaluated_metrics": [
                "configured_training_objective_reduction",
                "deterministic_mixture_median_target_local_rms_median",
            ],
            "stage_a_metric_semantics": (
                "sigmoid_mixture_loc_not_a_stochastic_single_draw_and_not_a_phase_b_metric"
            ),
            "stage_b_pending_metrics": [
                name.removesuffix("_lt").removesuffix("_gte") for name in GATES
            ],
            "stage_b_status": "pending_not_executed_fail_closed",
            "single_branch_phase_b_gate_passed": False,
            "complete_k1_proposal_exact_gate_passed": False,
            "full_k1_all_legal_branches_gate_status": (
                "pending_fail_closed_requires_balanced_12_branch_cohort"
            ),
        },
        "dataset": dataset_identity,
        "dataset_validation": {
            "clean_parent_count": 2,
            "observation_view_count": 2,
            "known_truth_target_count": 2,
        },
        "source": source,
        "training_evidence": training_evidence,
        "output_dir": "/data/dust/user/zhaiyufe/model",
        "created_at_utc": "2026-09-03T00:00:00+00:00",
        "status": "stage_a_passed",
        "writes_performed": True,
        "gate_executed": True,
        "stage_a_pass_enforced": True,
        "stage_a_configured_wiring_gate_passed": True,
        "stage_a_passed": True,
        "stage_a_memorization_result": nested,
        "single_branch_phase_b_gate_passed": False,
        "complete_k1_proposal_exact_gate_passed": False,
        "full_k1_all_legal_branches_gate_status": (
            "pending_fail_closed_requires_balanced_12_branch_cohort"
        ),
        "model_artifact": {
            "filename": "model.keras",
            **model_identity,
            "reload_graph_contract_passed": True,
            "provenance": {
                "filename": "model.provenance.json",
                **model_provenance_identity,
                "binding_sha256": model_provenance["binding_sha256"],
            },
        },
        "execution": {},
        "publication": "exclusive",
    }
    payload = {
        **payload_core,
        "result_payload_sha256": sha256(canonical_json(payload_core).encode()).hexdigest(),
    }
    return (
        payload,
        dataset_identity,
        model_identity,
        model_provenance,
        model_provenance_identity,
        source,
        weights_sha,
        contract,
    )


def test_phase_a_model_drift_is_rejected_even_if_result_is_rehashed():
    payload, dataset, model, provenance, provenance_file, source, weights, contract = (
        _phase_a_fixture()
    )
    validated = validate_v5_k1_phase_a_bindings(
        payload,
        dataset_identity=dataset,
        model_identity=model,
        model_provenance_payload=provenance,
        model_provenance_identity=provenance_file,
        phase_a_source_identity=source,
        source_snapshot_identity=payload["training_evidence"]["source_snapshot"],
        model_weights_sha256_value=weights,
        live_gate_contract_value=contract,
    )
    assert validated["phase_a_full_pass_reverified"] is True

    drifted = dict(payload)
    drifted["model_artifact"] = {
        **payload["model_artifact"],
        "sha256": "e" * 64,
    }
    core = dict(drifted)
    core.pop("result_payload_sha256")
    drifted["result_payload_sha256"] = sha256(canonical_json(core).encode()).hexdigest()
    with pytest.raises(ValueError, match="model artifact drift"):
        validate_v5_k1_phase_a_bindings(
            drifted,
            dataset_identity=dataset,
            model_identity=model,
            model_provenance_payload=provenance,
            model_provenance_identity=provenance_file,
            phase_a_source_identity=source,
            source_snapshot_identity=payload["training_evidence"]["source_snapshot"],
            model_weights_sha256_value=weights,
            live_gate_contract_value=contract,
        )


def _rehash_phase_a_result(payload, *, model_provenance=None):
    updated = dict(payload)
    if model_provenance is not None:
        artifact = dict(updated["model_artifact"])
        artifact["provenance"] = {
            **artifact["provenance"],
            "binding_sha256": model_provenance["binding_sha256"],
        }
        updated["model_artifact"] = artifact
    core = dict(updated)
    core.pop("result_payload_sha256")
    updated["result_payload_sha256"] = sha256(canonical_json(core).encode()).hexdigest()
    return updated


def test_phase_a_deleted_job_local_source_and_dataset_paths_are_historical(tmp_path):
    payload, dataset, model, provenance, provenance_file, source, weights, contract = (
        _phase_a_fixture()
    )
    source_scratch = tmp_path / "slurm-job/source"
    dataset_scratch = tmp_path / "slurm-job/input.gvd5"
    source_scratch.mkdir(parents=True)
    dataset_scratch.write_bytes(b"transient")

    replayed_source = {**source, "source_root": "/immutable/source-snapshot"}
    recorded_source = {**source, "source_root": str(source_scratch)}
    recorded_dataset = {**payload["dataset"], "path": str(dataset_scratch)}
    drifted = {
        **payload,
        "source": recorded_source,
        "dataset": recorded_dataset,
    }
    drifted = _rehash_phase_a_result(drifted)
    dataset_scratch.unlink()
    source_scratch.rmdir()
    source_scratch.parent.rmdir()

    validated = validate_v5_k1_phase_a_bindings(
        drifted,
        dataset_identity=dataset,
        model_identity=model,
        model_provenance_payload=provenance,
        model_provenance_identity=provenance_file,
        phase_a_source_identity=replayed_source,
        source_snapshot_identity=payload["training_evidence"]["source_snapshot"],
        model_weights_sha256_value=weights,
        live_gate_contract_value=contract,
    )

    assert not source_scratch.exists()
    assert not dataset_scratch.exists()
    assert validated["phase_a_recorded_execution_source_root"] == str(source_scratch)
    assert validated["reverified_immutable_source_root"] == "/immutable/source-snapshot"
    assert validated["phase_a_recorded_execution_dataset_path"] == str(dataset_scratch)
    assert validated["reverified_immutable_dataset_path"] == dataset["path"]


def test_phase_a_portable_source_dataset_and_snapshot_drift_fail_closed():
    payload, dataset, model, provenance, provenance_file, source, weights, contract = (
        _phase_a_fixture()
    )
    common = {
        "model_identity": model,
        "model_provenance_payload": provenance,
        "model_provenance_identity": provenance_file,
        "phase_a_source_identity": source,
        "source_snapshot_identity": payload["training_evidence"]["source_snapshot"],
        "model_weights_sha256_value": weights,
        "live_gate_contract_value": contract,
    }

    changed_dataset = {**dataset, "dataset_file_sha256": "0" * 64}
    with pytest.raises(ValueError, match="portable dataset content"):
        validate_v5_k1_phase_a_bindings(
            payload, dataset_identity=changed_dataset, **common
        )

    changed_source = {
        **source,
        "files": {
            **source["files"],
            next(iter(source["files"])): {"sha256": "0" * 64, "byte_count": 10},
        },
    }
    changed_source["bundle_sha256"] = sha256(
        canonical_json(changed_source["files"]).encode()
    ).hexdigest()
    with pytest.raises(ValueError, match="portable source content"):
        validate_v5_k1_phase_a_bindings(
            payload,
            dataset_identity=dataset,
            **{**common, "phase_a_source_identity": changed_source},
        )

    wrong_snapshot = {
        **payload["training_evidence"]["source_snapshot"],
        "source_tree_sha256": "0" * 64,
    }
    with pytest.raises(ValueError, match="archive, manifest, and tree"):
        validate_v5_k1_phase_a_bindings(
            payload,
            dataset_identity=dataset,
            **{**common, "source_snapshot_identity": wrong_snapshot},
        )

    wrong_original = dict(payload)
    evidence = dict(payload["training_evidence"])
    dataset_binding = dict(evidence["dataset_completion_binding"])
    dataset_binding["dataset"] = {
        **dataset_binding["dataset"],
        "original_path": "/data/dust/user/zhaiyufe/wrong.gvd5",
    }
    evidence_core = dict(evidence)
    evidence_core["dataset_completion_binding"] = dataset_binding
    evidence_core.pop("evidence_sha256")
    evidence = {
        **evidence_core,
        "evidence_sha256": sha256(canonical_json(evidence_core).encode()).hexdigest(),
    }
    provenance_core = dict(provenance)
    provenance_core["training_evidence"] = evidence
    provenance_core.pop("binding_sha256")
    changed_provenance = {
        **provenance_core,
        "binding_sha256": sha256(canonical_json(provenance_core).encode()).hexdigest(),
    }
    wrong_original["training_evidence"] = evidence
    wrong_original = _rehash_phase_a_result(
        wrong_original, model_provenance=changed_provenance
    )
    with pytest.raises(ValueError, match="original path"):
        validate_v5_k1_phase_a_bindings(
            wrong_original,
            dataset_identity=dataset,
            **{
                **common,
                "model_provenance_payload": changed_provenance,
                "source_snapshot_identity": evidence["source_snapshot"],
            },
        )


@pytest.mark.parametrize(
    "unsafe_path",
    ("", ".", "../escape.py", "/absolute.py", "a/../b.py", "a//b.py", "a\\b.py", 7),
)
def test_phase_a_dataset_source_inventory_requires_exact_safe_posix_paths(unsafe_path):
    payload, dataset, model, provenance, provenance_file, source, weights, contract = (
        _phase_a_fixture()
    )
    replayed = {
        **dataset,
        "dataset_source_sha256": {unsafe_path: "8" * 64},
    }
    with pytest.raises(ValueError, match="source inventory path"):
        validate_v5_k1_phase_a_bindings(
            payload,
            dataset_identity=replayed,
            model_identity=model,
            model_provenance_payload=provenance,
            model_provenance_identity=provenance_file,
            phase_a_source_identity=source,
            source_snapshot_identity=payload["training_evidence"]["source_snapshot"],
            model_weights_sha256_value=weights,
            live_gate_contract_value=contract,
        )


def test_phase_a_dataset_source_inventory_rejects_duplicate_mapping_items():
    class DuplicateInventory(dict):
        def items(self):
            return (("source.py", "8" * 64), ("source.py", "8" * 64))

    payload, dataset, model, provenance, provenance_file, source, weights, contract = (
        _phase_a_fixture()
    )
    replayed = {
        **dataset,
        "dataset_source_sha256": DuplicateInventory({"source.py": "8" * 64}),
    }
    with pytest.raises(ValueError, match="duplicate path"):
        validate_v5_k1_phase_a_bindings(
            payload,
            dataset_identity=replayed,
            model_identity=model,
            model_provenance_payload=provenance,
            model_provenance_identity=provenance_file,
            phase_a_source_identity=source,
            source_snapshot_identity=payload["training_evidence"]["source_snapshot"],
            model_weights_sha256_value=weights,
            live_gate_contract_value=contract,
        )


def test_phase_a_missing_metric_semantics_is_rejected_even_if_rehashed():
    payload, dataset, model, provenance, provenance_file, source, weights, contract = (
        _phase_a_fixture()
    )
    drifted = dict(payload)
    phase_a_contract = dict(payload["single_branch_k1_gate_contract"])
    phase_a_contract.pop("stage_a_metric_semantics")
    drifted["single_branch_k1_gate_contract"] = phase_a_contract
    core = dict(drifted)
    core.pop("result_payload_sha256")
    drifted["result_payload_sha256"] = sha256(canonical_json(core).encode()).hexdigest()

    with pytest.raises(ValueError, match="contract is incomplete"):
        validate_v5_k1_phase_a_bindings(
            drifted,
            dataset_identity=dataset,
            model_identity=model,
            model_provenance_payload=provenance,
            model_provenance_identity=provenance_file,
            phase_a_source_identity=source,
            source_snapshot_identity=payload["training_evidence"]["source_snapshot"],
            model_weights_sha256_value=weights,
            live_gate_contract_value=contract,
        )


def test_phase_a_model_provenance_cannot_be_replaced_by_rehashing_result():
    payload, dataset, model, provenance, provenance_file, source, weights, contract = (
        _phase_a_fixture()
    )
    drifted_provenance = dict(provenance)
    drifted_evidence = dict(provenance["training_evidence"])
    drifted_evidence["source_snapshot"] = {
        **drifted_evidence["source_snapshot"],
        "source_tree_sha256": "0" * 64,
    }
    drifted_provenance["training_evidence"] = drifted_evidence
    provenance_core = dict(drifted_provenance)
    provenance_core.pop("binding_sha256")
    drifted_provenance["binding_sha256"] = sha256(
        canonical_json(provenance_core).encode()
    ).hexdigest()

    with pytest.raises(ValueError, match="training evidence drift"):
        validate_v5_k1_phase_a_bindings(
            payload,
            dataset_identity=dataset,
            model_identity=model,
            model_provenance_payload=drifted_provenance,
            model_provenance_identity=provenance_file,
            phase_a_source_identity=source,
            source_snapshot_identity=payload["training_evidence"]["source_snapshot"],
            model_weights_sha256_value=weights,
            live_gate_contract_value=contract,
        )


def test_phase_b_identity_and_slurm_wrapper_are_single_branch_fail_closed():
    assert "single_branch" in V5_K1_PHASE_B_VERSION
    assert "not_full_k1" in V5_K1_PHASE_B_ROLE
    wrapper = SLURM_WRAPPER.read_text(encoding="utf-8")
    for value in (
        "--partition=allcpu",
        "SLURM_JOB_ID",
        "max-wgs*",
        "run_k1_phase_b_gate_v5",
        "POSTERIOR_V8_V5_K1_PHASE_B_DATASET",
        "POSTERIOR_V8_V5_K1_PHASE_B_RESULT",
        "POSTERIOR_V8_V5_K1_PHASE_B_MODEL",
        "POSTERIOR_V8_V5_K1_PHASE_B_OUTPUT",
        "POSTERIOR_V8_V5_K1_PHASE_B_MODE",
        "POSTERIOR_V8_V5_K1_PHASE_B_FORMAL_ACK",
        "--parent-limit",
        "PYTHONDONTWRITEBYTECODE=1",
        "XDG_CACHE_HOME",
        "KERAS_HOME",
        "CUDA_CACHE_PATH",
        "MPLCONFIGDIR",
        '[[ ! -d "$POSTERIOR_V8_SOURCE_ROOT" || -L "$POSTERIOR_V8_SOURCE_ROOT" ]]',
        "source root must be a real directory",
        "/data/dust/user/zhaiyufe/MaxwellRuns/"
        "GISAXS_ONE_CLICK_PAPER_V5_20260903_V5_2_R2",
    ):
        assert value in wrapper
    assert (
        "/data/dust/user/zhaiyufe/MaxwellRuns/"
        "GISAXS_ONE_CLICK_PAPER_V5_20260903_V5_2/"
    ) not in wrapper
