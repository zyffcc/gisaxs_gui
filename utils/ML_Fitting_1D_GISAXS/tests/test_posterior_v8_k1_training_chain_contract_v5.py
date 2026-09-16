from __future__ import annotations

from copy import deepcopy
from hashlib import sha256

import pytest

from utils.ML_Fitting_1D_GISAXS.PosteriorV8.amplitude_query_sampling_v5 import (
    V5_AMPLITUDE_RANGE_ASSIGNMENT_SCHEMA,
    V5_AMPLITUDE_RANGE_ASSIGNMENT_VERSION,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.k1_phase_c_contract_v5 import (
    K1_PHASE_C_BRANCHES,
    K1_PHASE_C_SPLIT_ID,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.k1_training_chain_contract_v5 import (
    V5K1TrainingArtifact,
    build_v5_k1_training_inventory,
    canonical_json,
    v5_k1_training_runtime_capabilities,
    validate_v5_k1_training_inventory,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.k1_training_chain_dataset_audit_v5 import (
    V5_K1_PARENT_SET_HASH_SEMANTICS,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.k1_training_identity_contract_v5 import (
    V5K1TrainingIdentityAuthorization,
    V5K1TrainingPopulationIdentity,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.sobol_recipe_coordinates_v5 import (
    V5_SOBOL_RECIPE_COORDINATE_SHA256,
    V5_SOBOL_RECIPE_DIM,
)


def _artifact(role: str, marker: str, *, counts: tuple[int, ...] | None = None):
    branch_counts = tuple(
        (branch.branch_id, 1 if counts is None else counts[index])
        for index, branch in enumerate(K1_PHASE_C_BRANCHES)
    )
    return V5K1TrainingArtifact(
        path=f"/data/dust/user/zhaiyufe/fixture/{marker}.gvd5",
        role=role,
        split_id="train" if role == "train" else "tuning_validation",
        artifact_sha256=marker[0] * 64,
        manifest_sha256=marker[-1] * 64,
        clean_parent_count=sum(value for _, value in branch_counts),
        branch_counts=branch_counts,
    )


def _identity_authorization():
    populations = (
        V5K1TrainingPopulationIdentity(
            role="train",
            split_id="train",
            plan_sha256="8" * 64,
            artifact_sha256s=("c" * 64,),
            manifest_sha256s=("d" * 64,),
            clean_parent_count=12,
            recipe_set_sha256="0" * 64,
            clean_group_set_sha256="1" * 64,
        ),
        V5K1TrainingPopulationIdentity(
            role="tuning_validation",
            split_id="tuning_validation",
            plan_sha256="8" * 64,
            artifact_sha256s=("e" * 64,),
            manifest_sha256s=("f" * 64,),
            clean_parent_count=12,
            recipe_set_sha256="6" * 64,
            clean_group_set_sha256="2" * 64,
        ),
        V5K1TrainingPopulationIdentity(
            role="phase_c_holdout",
            split_id=K1_PHASE_C_SPLIT_ID,
            plan_sha256="9" * 64,
            artifact_sha256s=("7" * 64,),
            manifest_sha256s=("8" * 64,),
            clean_parent_count=12,
            recipe_set_sha256="a" * 64,
            clean_group_set_sha256="b" * 64,
        ),
    )
    return V5K1TrainingIdentityAuthorization(
        source_archive_sha256="a" * 64,
        source_bundle_sha256="b" * 64,
        balanced_dataset_completion_file_sha256="0" * 64,
        balanced_dataset_completion_sha256="1" * 64,
        train_tuning_receipt_file_sha256="2" * 64,
        train_tuning_receipt_sha256="3" * 64,
        train_tuning_claim_sha256="4" * 64,
        phase_c_completion_file_sha256="5" * 64,
        phase_c_completion_sha256="6" * 64,
        three_way_receipt_file_sha256="7" * 64,
        three_way_receipt_sha256="4" * 64,
        phase_c_exclusion_claim_sha256="8" * 64,
        populations=populations,
    )


def _inventory(*, identity_authorization=None):
    return build_v5_k1_training_inventory(
        source_archive_sha256="a" * 64,
        source_bundle_sha256="b" * 64,
        train_artifacts=(_artifact("train", "cd"),),
        tuning_artifacts=(_artifact("tuning_validation", "ef"),),
        train_parent_set_sha256="1" * 64,
        tuning_parent_set_sha256="2" * 64,
        train_tuning_disjointness_receipt_sha256="3" * 64,
        k1_phase_c_disjointness_receipt_sha256="4" * 64,
        phase_a_result_payload_sha256="5" * 64,
        identity_authorization=identity_authorization,
    )


def test_inventory_freezes_all_twelve_balanced_branches_and_split_separation():
    value = validate_v5_k1_training_inventory(_inventory())

    assert set(value["splits"]["train"]["branch_counts"]) == {
        branch.branch_id for branch in K1_PHASE_C_BRANCHES
    }
    assert value["splits"]["train"]["clean_parent_count"] == 12
    assert value["splits"]["tuning_validation"]["clean_parent_count"] == 12
    assert value["splits"]["train_tuning_disjoint"] is True
    assert value["splits"]["parent_set_hash_semantics"] == V5_K1_PARENT_SET_HASH_SEMANTICS
    assert value["splits"]["k1_phase_c_split_id"] == K1_PHASE_C_SPLIT_ID
    assert value["splits"]["train_and_tuning_are_disjoint_from_k1_phase_c"] is False
    assert len(value["splits"]["k1_phase_c_disjointness_receipt_sha256"]) == 64


@pytest.mark.parametrize("drift", [None, "original_hash", "source_file", "source_manifest",
                                  "projected_file", "projected_manifest", "projected_path",
                                  "count", "groups", "missing_authorization"])
def test_projected_inventory_preserves_original_identity_without_promoting(drift):
    from utils.ML_Fitting_1D_GISAXS.PosteriorV8.k1_search_projection_evidence_v5 import (
        V5K1SearchProjectionEvidence,
    )

    authorization = _identity_authorization()
    train, tuning = _artifact("train", "01"), _artifact("tuning_validation", "23")
    rows = []
    for index, (item, original) in enumerate(((train, "cd"), (tuning, "ef"))):
        rows.append({
            "array_task_id": index, "role": item.role,
            "source_path": f"/data/dust/user/zhaiyufe/source/{original}.gvd5",
            "source_artifact_sha256": original[0] * 64,
            "source_manifest_sha256": original[1] * 64,
            "projected_path": item.path, "projected_artifact_sha256": item.artifact_sha256,
            "projected_manifest_sha256": item.manifest_sha256,
            "clean_parent_count": 12, "ordered_recipe_branch_split_group_arrays_equal": True,
        })
    field = {"source_file": "source_artifact_sha256", "source_manifest": "source_manifest_sha256",
             "projected_file": "projected_artifact_sha256", "projected_manifest": "projected_manifest_sha256"}
    if drift in field:
        rows[0][field[drift]] = "9" * 64
    if drift == "projected_path":
        rows[0]["projected_path"] += ".different"
    if drift == "count":
        rows[0]["clean_parent_count"] = 13
    evidence = V5K1SearchProjectionEvidence(
        plan_sha256="4" * 64, inventory_file_sha256="5" * 64,
        completion_file_sha256="6" * 64,
        original_identity_authorization_sha256=("9" * 64 if drift == "original_hash" else authorization.sha256),
        mappings=tuple(rows),
    )
    arguments = dict(
        source_archive_sha256="7" * 64, source_bundle_sha256="8" * 64,
        train_artifacts=(train,), tuning_artifacts=(tuning,),
        train_parent_set_sha256=("9" * 64 if drift == "groups" else "1" * 64),
        tuning_parent_set_sha256="2" * 64,
        train_tuning_disjointness_receipt_sha256="3" * 64,
        k1_phase_c_disjointness_receipt_sha256="4" * 64,
        identity_authorization=(None if drift == "missing_authorization" else authorization),
        projection_evidence=evidence,
    )
    if drift is not None:
        with pytest.raises(ValueError, match="projection"):
            build_v5_k1_training_inventory(**arguments)
        return
    value = validate_v5_k1_training_inventory(build_v5_k1_training_inventory(**arguments))
    assert value["splits"]["identity_authorization"] == authorization.to_payload()
    assert value["splits"]["projection_evidence"] == evidence.to_payload()
    assert value["splits"]["projection_requires_live_collection_replay"] is True
    assert value["splits"]["train_and_tuning_are_disjoint_from_k1_phase_c"] is False
    assert value["source_archive_sha256"] != authorization.source_archive_sha256
    from utils.ML_Fitting_1D_GISAXS.PosteriorV8.k1_training_chain_plan_v5 import _formal_blockers

    assert any("identity authorization" in reason for reason in _formal_blockers(value, "formal_multiseed"))
    tampered = deepcopy(value)
    tampered["splits"]["projection_requires_live_collection_replay"] = False
    tampered.pop("inventory_sha256")
    tampered["inventory_sha256"] = sha256(canonical_json(tampered).encode()).hexdigest()
    with pytest.raises(ValueError, match="drifted"):
        validate_v5_k1_training_inventory(tampered)


def test_inventory_accepts_only_an_identity_gate_bound_to_its_exact_populations():
    value = validate_v5_k1_training_inventory(
        _inventory(identity_authorization=_identity_authorization())
    )

    assert value["splits"]["train_and_tuning_are_disjoint_from_k1_phase_c"] is True
    assert value["splits"]["identity_authorization_sha256"] == (
        _identity_authorization().sha256
    )
    assert value["claim_limits"]["identity_authorization_grants_gradients"] is False


def test_inventory_binds_live_coordinate_amplitude_and_model_contracts():
    value = _inventory()

    assert value["coordinate_contract"]["sha256"] == V5_SOBOL_RECIPE_COORDINATE_SHA256
    assert value["coordinate_contract"]["dimension"] == V5_SOBOL_RECIPE_DIM
    assert value["amplitude_range_assignment_contract"] == {
        "schema": V5_AMPLITUDE_RANGE_ASSIGNMENT_SCHEMA,
        "version": V5_AMPLITUDE_RANGE_ASSIGNMENT_VERSION,
    }
    assert len(value["model_contract_sha256"]) == 64
    assert len(value["dataset_contract"]["dataset_manifest_bundle_sha256"]) == 64


def test_phase_a_is_only_optional_diagnostic_and_cannot_claim_k1_coverage():
    value = _inventory()

    assert value["phase_a"]["result_payload_sha256"] == "5" * 64
    assert value["phase_a"]["role"] == "single_branch_capacity_diagnostic_only"
    assert value["phase_a"]["counts_as_full_k1_coverage"] is False
    assert (
        value["phase_a"]["weights_accepted_as_k1_training_initialization_by_this_contract"]
        is False
    )


def test_current_full_training_and_exact_budget_tuning_blockers_are_explicit():
    capabilities = v5_k1_training_runtime_capabilities()

    assert capabilities["full_search_supervision_consumable_by_grouped_trainer"] is True
    assert capabilities["tuning_exact_budget_summary_runtime_available"] is False
    assert capabilities["formal_training_authorization_adapter_available"] is False
    assert capabilities["formal_chain_submission_ready"] is False
    assert any(
        "formal training authorization" in value
        for value in capabilities["blocked_interfaces"]
    )


def test_missing_or_imbalanced_branch_fails_before_inventory_exists():
    missing = (0, *(1 for _ in range(11)))
    with pytest.raises(ValueError, match="does not cover every legal K1 branch"):
        build_v5_k1_training_inventory(
            source_archive_sha256="a" * 64,
            source_bundle_sha256="b" * 64,
            train_artifacts=(_artifact("train", "cd", counts=missing),),
            tuning_artifacts=(_artifact("tuning_validation", "ef"),),
            train_parent_set_sha256="1" * 64,
            tuning_parent_set_sha256="2" * 64,
            train_tuning_disjointness_receipt_sha256="3" * 64,
            k1_phase_c_disjointness_receipt_sha256="4" * 64,
        )

    imbalanced = (3, *(1 for _ in range(11)))
    with pytest.raises(ValueError, match="not balanced"):
        build_v5_k1_training_inventory(
            source_archive_sha256="a" * 64,
            source_bundle_sha256="b" * 64,
            train_artifacts=(_artifact("train", "cd", counts=imbalanced),),
            tuning_artifacts=(_artifact("tuning_validation", "ef"),),
            train_parent_set_sha256="1" * 64,
            tuning_parent_set_sha256="2" * 64,
            train_tuning_disjointness_receipt_sha256="3" * 64,
            k1_phase_c_disjointness_receipt_sha256="4" * 64,
        )


def test_rehashed_identity_tampering_still_fails_against_live_owner_contracts():
    value = deepcopy(_inventory())
    value["coordinate_contract"]["sha256"] = "0" * 64
    core = dict(value)
    core.pop("inventory_sha256")
    value["inventory_sha256"] = sha256(canonical_json(core).encode()).hexdigest()

    with pytest.raises(ValueError, match="drifted from live contracts"):
        validate_v5_k1_training_inventory(value)
