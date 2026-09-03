from __future__ import annotations

from dataclasses import replace
from hashlib import sha256

import pytest

from utils.ML_Fitting_1D_GISAXS.PosteriorV8 import (
    formal_production_training_promotion_v5 as promotion_module,
)

from utils.ML_Fitting_1D_GISAXS.PosteriorV8.calibrated_search_threshold_v5 import (
    V5CalibrationArtifactIdentity,
    V5CheckedCompatibilityCalibration,
    compatibility_stratum_from_v5_observation,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.compatibility_calibration import (
    CompatibilityCalibrationSample,
    fit_compatibility_calibration,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.contract import (
    NUM_TOPOLOGIES,
    topology_from_id,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.exact_search_executor_v5 import (
    build_v5_frozen_exact_search_protocol,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.exact_search_schedule_v5 import (
    V5FrozenExactOptimizerSchedule,
    V5FrozenLocalSobolSchedule,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.formal_production_search_contract_v5 import (
    V5FormalProductionSearchStage,
    V5FormalProductionSourceIdentity,
    V5_FORMAL_PRODUCTION_STAGE_IDS,
    plan_v5_formal_production_search_shard,
    topology_ids_for_v5_formal_production_stage,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.formal_production_search_membership_v5 import (
    verify_v5_formal_production_receipt_membership,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.formal_production_search_plan_v5 import (
    V5FormalProductionSearchAuthorization,
    V5_FORMAL_PRODUCTION_TRAIN_CONSUMER_ROLE,
    V5_FORMAL_PRODUCTION_TRAINING_PROMOTION_ENABLED,
    V5_FORMAL_PRODUCTION_TUNING_CONSUMER_ROLE,
    authorize_v5_formal_production_search_shard,
    build_v5_formal_production_search_plan,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.formal_production_training_promotion_v5 import (
    materialize_v5_formal_production_input_snapshot,
    promote_v5_formal_production_search_evidence,
    reverify_v5_formal_production_evidence_promotion,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.frozen_search_pipeline_contract_v5 import (
    V5SelectedTopologySearchSchedule,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.grouped_artifact_v5 import canonical_json
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.observation_v5 import (
    build_v5_observation_data_view,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.search_evidence_receipt_v5 import (
    V5SearchEvidenceReceipt,
    V5_SEARCH_EVIDENCE_AUDIT_POLICY,
    V5_SEARCH_EVIDENCE_RECEIPT_SCHEMA,
    V5_SEARCH_EVIDENCE_RECEIPT_VERSION,
    V5_SEARCH_LABEL_BINDING_SCHEMA,
    V5_SEARCH_LABEL_PURPOSE_FORMAL_CONTRACT_SMOKE,
    V5_SEARCH_LABEL_PURPOSE_TRAINING,
    build_v5_search_label_binding,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.search_supervision_contract_v5 import (
    V5_SEARCH_PROTOCOL_TIER_PAPER_FULL_CALIBRATED,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.sobol_design_v5 import (
    materialize_v5_design_points_for_indices,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.sobol_recipe_coordinates_v5 import (
    V5_SOBOL_RECIPE_COORDINATE_INDEX,
    v5_sobol_recipe_design,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.split_design_v5 import (
    V5SplitCounts,
    V5SplitPlan,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.synthetic_recipe_v5 import (
    sample_v5_clean_recipe,
)


def _checked_calibration() -> V5CheckedCompatibilityCalibration:
    recipe = sample_v5_clean_recipe(("sphere",), recipe_seed=77)
    view = next(
        value
        for value in (
            build_v5_observation_data_view(recipe, index, split_id="calibration")
            for index in range(4)
        )
        if value.acceptance_sigma_log is not None
    )
    stratum = compatibility_stratum_from_v5_observation(view)
    samples = tuple(
        CompatibilityCalibrationSample(
            sample_id=f"formal-calibration-{index:03d}",
            independent_group_id=f"formal-recipe-{index:03d}",
            stratum=stratum,
            score=float(index + 1),
            effective_valid_point_count=view.effective_valid_point_count,
            acquisition_policy_id=view.acquisition_policy_id,
            measurement_sigma_available=True,
        )
        for index in range(9)
    )
    artifact = fit_compatibility_calibration(
        samples,
        dataset_manifest_sha256="1" * 64,
        calibration_split_sha256="2" * 64,
        target_coverage=0.8,
        minimum_samples_per_stratum=5,
    )
    identity = V5CalibrationArtifactIdentity(
        artifact_sha256=artifact.sha256,
        file_sha256="3" * 64,
        input_sha256=artifact.input_sha256,
        dataset_manifest_sha256=artifact.dataset_manifest_sha256,
        calibration_split_id=artifact.calibration_split_id,
        calibration_split_sha256=artifact.calibration_split_sha256,
        calibration_schema=artifact.schema,
        calibration_version=artifact.calibration_version,
        target_coverage=artifact.target_coverage,
        compatibility_stratum_version=artifact.compatibility_stratum_version,
        compatibility_stratum_fields=artifact.compatibility_stratum_fields,
        stratification_semantics=artifact.stratification_semantics,
        score_semantics=artifact.score_semantics,
        measurement_sigma_policy=artifact.measurement_sigma_policy,
    )
    return V5CheckedCompatibilityCalibration(artifact=artifact, identity=identity)


def _stage(stage_id, budget, calibration_identity, *, formal=True, topology_ids=None):
    seeds = V5FrozenLocalSobolSchedule.generate(
        schedule_id=f"formal-{stage_id.lower()}-seeds-v1",
        point_count=budget,
        base_seed=100 + budget,
    )
    optimizer = V5FrozenExactOptimizerSchedule(
        schedule_id=f"formal-{stage_id.lower()}-optimizer-v1",
        direct_scout_seed_count=1,
        per_seed_forward_evaluation_limit=1,
    )
    if formal:
        protocol = build_v5_frozen_exact_search_protocol(
            protocol_id=f"formal-{stage_id.lower()}-protocol-v1",
            seed_schedule=seeds,
            optimizer_schedule=optimizer,
            protocol_tier=V5_SEARCH_PROTOCOL_TIER_PAPER_FULL_CALIBRATED,
            calibration_identity=calibration_identity,
        )
    else:
        protocol = build_v5_frozen_exact_search_protocol(
            protocol_id="engineering-pilot",
            seed_schedule=seeds,
            optimizer_schedule=optimizer,
            standardized_threshold_name="pilot-standardized",
            standardized_threshold_value=3.0,
            raw_threshold_name="pilot-raw",
            raw_threshold_value=0.2,
            threshold_source_id="pilot-only",
        )
    selected = (
        topology_ids_for_v5_formal_production_stage(stage_id)
        if topology_ids is None
        else topology_ids
    )
    return V5FormalProductionSearchStage(
        stage_id=stage_id,
        topology_schedule=V5SelectedTopologySearchSchedule(
            schedule_id=f"formal-{stage_id.lower()}-topologies-v1",
            selected_topology_ids=selected,
        ),
        seed_schedule=seeds,
        optimizer_schedule=optimizer,
        protocol=protocol,
    )


def _generating_topology_id(point):
    value = point.unit_coordinates[
        V5_SOBOL_RECIPE_COORDINATE_INDEX["discrete.topology"]
    ]
    return min(int(value * NUM_TOPOLOGIES), NUM_TOPOLOGIES - 1)


def _stage_indices(split_plan, design, split):
    block = next(value for value in split_plan.blocks if value.name == split)
    indices = tuple(range(block.start, block.start + min(block.count, 256)))
    points = materialize_v5_design_points_for_indices(split_plan, design, indices)
    result = {}
    for point in points:
        component_count = len(topology_from_id(_generating_topology_id(point)))
        if component_count == 1 and "K1" not in result:
            result["K1"] = point.sobol_index
        elif component_count == 2 and "K2" not in result:
            result["K2"] = point.sobol_index
        elif component_count >= 3 and "ALL34" not in result:
            result["ALL34"] = point.sobol_index
        if set(result) == set(V5_FORMAL_PRODUCTION_STAGE_IDS):
            return result
    raise AssertionError("test design did not expose all staged topology sizes")


@pytest.fixture(scope="module")
def formal_plan_fixture():
    calibration = _checked_calibration()
    split_plan = V5SplitPlan.create(
        V5SplitCounts(
            train=256,
            tuning_validation=256,
            calibration=8,
            test=8,
            reference=8,
            ood_topology=2,
            ood_range_width=2,
            ood_weak_component=2,
            ood_acquisition_policy=2,
        ),
        guard_band=4,
    )
    design = v5_sobol_recipe_design(scramble_seed=20260903)
    stages = tuple(
        _stage(stage_id, budget, calibration.identity)
        for stage_id, budget in zip(V5_FORMAL_PRODUCTION_STAGE_IDS, (2, 4, 8))
    )
    by_split = {
        split: _stage_indices(split_plan, design, split)
        for split in ("train", "tuning_validation")
    }
    shards = tuple(
        plan_v5_formal_production_search_shard(
            split_plan=split_plan,
            sobol_design=design,
            stage=stage,
            target_split=split,
            sobol_indices=(by_split[split][stage.stage_id],),
            output_relative_path=(
                f"labels/{stage.stage_id.lower()}/{split}/shard-000000"
            ),
        )
        for stage in stages
        for split in ("train", "tuning_validation")
    )
    source = V5FormalProductionSourceIdentity.from_fingerprint(
        {
            "bundle_sha256": "4" * 64,
            "bundle_file_count": 7,
            "required_file_sha256": {
                "PosteriorV8/executor.py": "5" * 64,
                "fitting/domain/scattering_model.py": "6" * 64,
            },
        }
    )
    plan = build_v5_formal_production_search_plan(
        study_id="paper-v5-formal-search-20260903",
        source=source,
        split_plan=split_plan,
        sobol_design=design,
        calibration=calibration,
        candidate_view_indices=(0, 1),
        stages=stages,
        shards=reversed(shards),
    )
    return plan


def test_global_plan_freezes_three_stages_queries_views_and_disjoint_membership(
    formal_plan_fixture,
):
    plan = formal_plan_fixture
    assert tuple(value.stage_id for value in plan.stages) == ("K1", "K2", "ALL34")
    assert tuple(value.exact_forward_call_budget for value in plan.stages) == (2, 4, 8)
    assert tuple(len(value.selected_topology_ids) for value in plan.stages) == (3, 9, 34)
    assert len(plan.shards) == 6
    assert len(plan.sha256) == 64
    assert plan.sha256 == sha256(plan.canonical_json.encode("utf-8")).hexdigest()
    assert plan.to_payload()["plan_sha256"] == plan.sha256
    promotion = plan.to_payload()["promotion_boundary"]
    assert promotion["formal_membership_verifier_available"] is True
    assert promotion["training_promotion_enabled_by_this_module"] is True
    assert promotion["consumer_role_is_derived_from_frozen_split"] is True
    assert promotion["tuning_validation_can_supply_gradients"] is False
    assert promotion["membership_proof_alone_authorizes_training"] is False
    assert V5_FORMAL_PRODUCTION_TRAINING_PROMOTION_ENABLED is True
    indices = []
    for shard in plan.shards:
        assert shard.expected_query_count == 1
        assert shard.expected_branch_count > 0
        assert shard.expected_exact_forward_calls == (
            shard.expected_branch_count * shard.stage.exact_forward_call_budget
        )
        selection = shard.recipes[0].observation_selection.audit_payload()
        assert selection["decision_was_curve_blind"] is True
        assert selection["selection_count"] == 1
        selected = next(
            value
            for value in selection["candidate_uncertainty"]
            if value["view_index"] == selection["selected_view_index"]
        )
        assert selected["measurement_sigma_available"] is True
        indices.extend(value.sobol_index for value in shard.recipes)
    assert len(indices) == len(set(indices))


def test_stage_rejects_pilot_protocol_or_incomplete_topology_scope():
    calibration = _checked_calibration()
    with pytest.raises(ValueError, match="paper_full_calibrated"):
        _stage("K1", 2, calibration.identity, formal=False)
    with pytest.raises(ValueError, match="incomplete or over-broad"):
        _stage("K1", 2, calibration.identity, topology_ids=(0,))


def test_shard_rejects_wrong_split_unsafe_output_and_non_sigma_pool(
    formal_plan_fixture,
):
    plan = formal_plan_fixture
    stage = plan.stages[0]
    train_index = next(
        value.sobol_index
        for shard in plan.shards
        if shard.stage.stage_id == "K1" and shard.target_split == "train"
        for value in shard.recipes
    )
    with pytest.raises(ValueError, match="escaped its declared"):
        plan_v5_formal_production_search_shard(
            split_plan=plan.split_plan,
            sobol_design=plan.sobol_design,
            stage=stage,
            target_split="tuning_validation",
            sobol_indices=(train_index,),
            output_relative_path="labels/k1/tuning_validation/bad-split",
        )
    with pytest.raises(ValueError, match="must be below"):
        plan_v5_formal_production_search_shard(
            split_plan=plan.split_plan,
            sobol_design=plan.sobol_design,
            stage=stage,
            target_split="train",
            sobol_indices=(train_index,),
            output_relative_path="elsewhere/k1/train/bad-path",
        )
    with pytest.raises(ValueError, match="exactly one"):
        plan_v5_formal_production_search_shard(
            split_plan=plan.split_plan,
            sobol_design=plan.sobol_design,
            stage=stage,
            target_split="train",
            sobol_indices=(train_index,),
            output_relative_path="labels/k1/train/bad-view-pool",
            candidate_view_indices=(0, 2),
        )


def test_global_plan_rejects_overlapping_clean_parents(formal_plan_fixture):
    plan = formal_plan_fixture
    original = next(
        value
        for value in plan.shards
        if value.stage.stage_id == "K1" and value.target_split == "train"
    )
    duplicate = plan_v5_formal_production_search_shard(
        split_plan=plan.split_plan,
        sobol_design=plan.sobol_design,
        stage=original.stage,
        target_split="train",
        sobol_indices=tuple(value.sobol_index for value in original.recipes),
        output_relative_path="labels/k1/train/shard-duplicate",
    )
    with pytest.raises(ValueError, match="overlap clean parents"):
        build_v5_formal_production_search_plan(
            study_id=plan.study_id,
            source=plan.source,
            split_plan=plan.split_plan,
            sobol_design=plan.sobol_design,
            calibration=plan.calibration,
            candidate_view_indices=plan.candidate_view_indices,
            stages=plan.stages,
            shards=(*plan.shards, duplicate),
        )


def test_global_plan_rejects_stage_with_another_calibration_identity(
    formal_plan_fixture,
):
    plan = formal_plan_fixture
    other_identity = replace(plan.calibration.identity, file_sha256="9" * 64)
    changed = _stage("K1", 2, other_identity)
    with pytest.raises(ValueError, match="escaped the checked global calibration"):
        build_v5_formal_production_search_plan(
            study_id=plan.study_id,
            source=plan.source,
            split_plan=plan.split_plan,
            sobol_design=plan.sobol_design,
            calibration=plan.calibration,
            candidate_view_indices=plan.candidate_view_indices,
            stages=(changed, *plan.stages[1:]),
            shards=plan.shards,
        )


def _branch_entry(*, row, query_index, budget):
    return {
        "branch_row": row,
        "query_index": query_index,
        "global_branch_key": f"topology-00:wire-{row:02d}",
        "exact_curve_sha256": "a" * 64,
        "relative_path": f"executor-evidence/branch-{row:06d}.gvd5",
        "artifact_id": f"executor-{row:06d}",
        "artifact_sha256": "b" * 64,
        "artifact_schema": "gisaxs.posterior_v8.exact_search_executor/v-test",
        "artifact_version": "test",
        "task_audit_sha256": "c" * 64,
        "executor_task_payload_sha256": "d" * 64,
        "outcome": "no_compatible_found_within_frozen_search_budget",
        "exact_forward_calls_used": budget,
    }


def _training_receipt(plan, shard):
    stage = shard.stage
    authorization = authorize_v5_formal_production_search_shard(
        plan, shard_plan_sha256=shard.sha256
    )
    binding_core = {
        "schema": V5_SEARCH_LABEL_BINDING_SCHEMA,
        "launch_source_bundle_sha256": plan.source.bundle_sha256,
        "launch_plan_sha256": plan.sha256,
        "shard_plan_sha256": shard.sha256,
        "protocol_sha256": stage.protocol.sha256,
        "protocol_tier": V5_SEARCH_PROTOCOL_TIER_PAPER_FULL_CALIBRATED,
        "seed_schedule_sha256": stage.seed_schedule.sha256,
        "optimizer_schedule_sha256": stage.optimizer_schedule.sha256,
        "calibration_artifact_sha256": plan.calibration.identity.artifact_sha256,
        "formal_production_authorization_sha256": authorization.sha256,
        "authorized_consumer_role": authorization.consumer_role,
        "label_purpose": V5_SEARCH_LABEL_PURPOSE_TRAINING,
        "full_training_eligible": True,
    }
    binding = {
        **binding_core,
        "label_binding_sha256": sha256(
            canonical_json(binding_core).encode("utf-8")
        ).hexdigest(),
    }
    query_indices = tuple(
        query_index
        for query_index, recipe in enumerate(shard.recipes)
        for _ in range(recipe.branch_count)
    )
    executor_sources = {"exact_search_executor_v5.py": "e" * 64}
    core = {
        "schema": V5_SEARCH_EVIDENCE_RECEIPT_SCHEMA,
        "version": V5_SEARCH_EVIDENCE_RECEIPT_VERSION,
        "receipt_id": "formal-training-receipt-test",
        "label_purpose": V5_SEARCH_LABEL_PURPOSE_TRAINING,
        "full_training_eligible": True,
        "audit_policy": V5_SEARCH_EVIDENCE_AUDIT_POLICY,
        "launch_source_bundle_sha256": plan.source.bundle_sha256,
        "launch_plan_sha256": plan.sha256,
        "shard_plan_sha256": shard.sha256,
        "label_binding": binding,
        "label_binding_sha256": binding["label_binding_sha256"],
        "formal_production_authorization": authorization.to_payload(),
        "formal_production_authorization_sha256": authorization.sha256,
        "executor_root_relative_path": "executor-evidence",
        "parent": {
            "dataset_id": "parent-test",
            "artifact_sha256": "f" * 64,
            "manifest_sha256": "0" * 64,
        },
        "sidecar": {
            "sidecar_id": "formal-training-sidecar-test",
            "artifact_sha256": "1" * 64,
            "manifest_sha256": "2" * 64,
            "protocol_sha256": stage.protocol.sha256,
            "split_id": shard.target_split,
        },
        "executor_source_sha256": executor_sources,
        "executor_source_bundle_sha256": sha256(
            canonical_json(executor_sources).encode("utf-8")
        ).hexdigest(),
        "branch_evidence": [
            _branch_entry(
                row=row,
                query_index=query_index,
                budget=stage.exact_forward_call_budget,
            )
            for row, query_index in enumerate(query_indices)
        ],
        "counts": {
            "branches": shard.expected_branch_count,
            "queries": shard.expected_query_count,
            "exact_forward_calls_used": shard.expected_exact_forward_calls,
        },
    }
    return {
        **core,
        "receipt_sha256": sha256(canonical_json(core).encode("utf-8")).hexdigest(),
    }


def _rehash_receipt(receipt):
    core = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    receipt["receipt_sha256"] = sha256(
        canonical_json(core).encode("utf-8")
    ).hexdigest()
    return receipt


def test_pure_receipt_membership_proves_exact_shard_without_authorizing_training(
    formal_plan_fixture,
):
    plan = formal_plan_fixture
    shard = plan.shards[-1]
    receipt = _training_receipt(plan, shard)

    proof = verify_v5_formal_production_receipt_membership(
        plan,
        output_relative_path=shard.output_relative_path,
        receipt=receipt,
    )

    assert proof.launch_plan_sha256 == plan.sha256
    assert proof.shard_plan_sha256 == shard.sha256
    assert proof.stage_id == "ALL34"
    assert proof.split == "tuning_validation"
    assert proof.audit_payload()["contract_membership_verified"] is True
    assert proof.audit_payload()["training_authorized_by_this_proof"] is False


def test_receipt_membership_fails_closed_for_global_shard_path_and_budget_drift(
    formal_plan_fixture,
):
    plan = formal_plan_fixture
    shard = plan.shards[0]
    receipt = _training_receipt(plan, shard)

    wrong_global = dict(receipt)
    wrong_global["launch_plan_sha256"] = "9" * 64
    _rehash_receipt(wrong_global)
    with pytest.raises(ValueError, match="canonical global plan"):
        verify_v5_formal_production_receipt_membership(
            plan,
            output_relative_path=shard.output_relative_path,
            receipt=wrong_global,
        )

    with pytest.raises(ValueError, match="planned shard location"):
        verify_v5_formal_production_receipt_membership(
            plan,
            output_relative_path="labels/k1/train/not-the-member",
            receipt=receipt,
        )

    wrong_budget = {**receipt, "branch_evidence": [dict(value) for value in receipt["branch_evidence"]]}
    wrong_budget["branch_evidence"][0]["exact_forward_calls_used"] -= 1
    _rehash_receipt(wrong_budget)
    with pytest.raises(ValueError, match="ordering or full budget"):
        verify_v5_formal_production_receipt_membership(
            plan,
            output_relative_path=shard.output_relative_path,
            receipt=wrong_budget,
        )


def test_receipt_membership_rejects_smoke_purpose_even_with_valid_outer_hash(
    formal_plan_fixture,
):
    plan = formal_plan_fixture
    shard = plan.shards[0]
    receipt = _training_receipt(plan, shard)
    receipt["label_purpose"] = "formal_calibrated_contract_smoke_not_training_eligible"
    receipt["full_training_eligible"] = False
    _rehash_receipt(receipt)

    with pytest.raises(ValueError, match="requires a TRAINING receipt"):
        verify_v5_formal_production_receipt_membership(
            plan,
            output_relative_path=shard.output_relative_path,
            receipt=receipt,
        )


def test_plan_derived_authorization_strictly_separates_gradient_and_tuning_roles(
    formal_plan_fixture,
):
    plan = formal_plan_fixture
    train_shard = next(value for value in plan.shards if value.target_split == "train")
    tuning_shard = next(
        value for value in plan.shards if value.target_split == "tuning_validation"
    )

    train = authorize_v5_formal_production_search_shard(
        plan, shard_plan_sha256=train_shard.sha256
    )
    tuning = authorize_v5_formal_production_search_shard(
        plan, shard_plan_sha256=tuning_shard.sha256
    )

    assert train.consumer_role == V5_FORMAL_PRODUCTION_TRAIN_CONSUMER_ROLE
    assert train.gradient_training_eligible is True
    assert tuning.consumer_role == V5_FORMAL_PRODUCTION_TUNING_CONSUMER_ROLE
    assert tuning.gradient_training_eligible is False
    assert V5FormalProductionSearchAuthorization.from_payload(
        train.to_payload()
    ).to_payload() == train.to_payload()


def test_authorization_rejects_split_role_or_membership_tampering(
    formal_plan_fixture,
):
    plan = formal_plan_fixture
    train_shard = next(value for value in plan.shards if value.target_split == "train")
    tuning_shard = next(
        value for value in plan.shards if value.target_split == "tuning_validation"
    )
    train = authorize_v5_formal_production_search_shard(
        plan, shard_plan_sha256=train_shard.sha256
    )

    wrong_role = train.to_payload()
    wrong_role["consumer_role"] = V5_FORMAL_PRODUCTION_TUNING_CONSUMER_ROLE
    with pytest.raises(ValueError, match="consumer role"):
        V5FormalProductionSearchAuthorization.from_payload(wrong_role)

    receipt = _training_receipt(plan, train_shard)
    tuning_authorization = authorize_v5_formal_production_search_shard(
        plan, shard_plan_sha256=tuning_shard.sha256
    )
    receipt["formal_production_authorization"] = tuning_authorization.to_payload()
    receipt["formal_production_authorization_sha256"] = tuning_authorization.sha256
    _rehash_receipt(receipt)
    with pytest.raises(ValueError, match="plan-derived shard authorization"):
        verify_v5_formal_production_receipt_membership(
            plan,
            output_relative_path=train_shard.output_relative_path,
            receipt=receipt,
        )


def test_pilot_or_smoke_label_can_never_carry_formal_training_authorization(
    formal_plan_fixture,
):
    plan = formal_plan_fixture
    shard = plan.shards[0]
    authorization = authorize_v5_formal_production_search_shard(
        plan, shard_plan_sha256=shard.sha256
    )

    with pytest.raises(ValueError, match="cannot carry"):
        build_v5_search_label_binding(
            protocol=shard.stage.protocol,
            seed_schedule_sha256=shard.stage.seed_schedule.sha256,
            optimizer_schedule_sha256=shard.stage.optimizer_schedule.sha256,
            launch_source_bundle_sha256=plan.source.bundle_sha256,
            launch_plan_sha256=plan.sha256,
            shard_plan_sha256=shard.sha256,
            label_purpose=V5_SEARCH_LABEL_PURPOSE_FORMAL_CONTRACT_SMOKE,
            formal_production_authorization=authorization,
        )


def _patch_file_backed_receipt_reader(monkeypatch, receipt_path, manifest):
    checked = V5SearchEvidenceReceipt(
        path=receipt_path,
        manifest=manifest,
        file_sha256=sha256(receipt_path.read_bytes()).hexdigest(),
    )

    def fake_reader(
        path,
        *,
        parent_dataset_path,
        sidecar_path,
        require_training_eligible,
        expected_consumer_role,
    ):
        assert path == receipt_path.resolve()
        assert parent_dataset_path.is_file()
        assert sidecar_path.is_file()
        assert require_training_eligible is True
        assert expected_consumer_role == manifest["label_binding"][
            "authorized_consumer_role"
        ]
        return checked

    monkeypatch.setattr(
        promotion_module,
        "read_v5_search_evidence_receipt",
        fake_reader,
    )


def test_file_backed_promotion_detects_mutation_and_private_snapshot_isolated(
    formal_plan_fixture,
    monkeypatch,
    tmp_path,
):
    plan = formal_plan_fixture
    shard = next(value for value in plan.shards if value.target_split == "train")
    receipt_path = tmp_path / "receipt.json"
    parent_path = tmp_path / "parent.gvd5"
    sidecar_path = tmp_path / "sidecar.gvd5"
    receipt_path.write_bytes(b"receipt-bytes")
    parent_path.write_bytes(b"parent-bytes")
    sidecar_path.write_bytes(b"sidecar-bytes")
    _patch_file_backed_receipt_reader(
        monkeypatch,
        receipt_path,
        _training_receipt(plan, shard),
    )
    guard = lambda: plan.source.bundle_sha256
    promotion = promote_v5_formal_production_search_evidence(
        plan,
        output_relative_path=shard.output_relative_path,
        receipt_path=receipt_path,
        parent_dataset_path=parent_path,
        sidecar_path=sidecar_path,
        consumer_role=V5_FORMAL_PRODUCTION_TRAIN_CONSUMER_ROLE,
        pre_consume_source_guard=guard,
    )
    assert promotion.gradient_training_permitted is True

    snapshot = materialize_v5_formal_production_input_snapshot(
        plan,
        promotion,
        snapshot_root=tmp_path / "private-snapshot",
        pre_consume_source_guard=guard,
    )
    parent_path.write_bytes(b"mutated-source-parent")
    assert snapshot.parent_dataset_path.read_bytes() == b"parent-bytes"
    assert snapshot.completion_path.is_file()
    with pytest.raises(RuntimeError, match="changed after promotion"):
        reverify_v5_formal_production_evidence_promotion(
            plan,
            promotion,
            pre_consume_source_guard=guard,
        )

    # Restore write permission so the pytest temporary directory can be removed.
    snapshot.snapshot_root.chmod(0o700)


@pytest.mark.parametrize("symlink_parent_component", (False, True))
def test_file_backed_promotion_rejects_leaf_or_parent_component_symlink(
    formal_plan_fixture,
    monkeypatch,
    tmp_path,
    symlink_parent_component,
):
    plan = formal_plan_fixture
    shard = next(value for value in plan.shards if value.target_split == "train")
    real = tmp_path / "real"
    real.mkdir()
    receipt_path = real / "receipt.json"
    parent_path = real / "parent.gvd5"
    sidecar_path = real / "sidecar.gvd5"
    receipt_path.write_bytes(b"receipt-bytes")
    parent_path.write_bytes(b"parent-bytes")
    sidecar_path.write_bytes(b"sidecar-bytes")
    _patch_file_backed_receipt_reader(
        monkeypatch,
        receipt_path,
        _training_receipt(plan, shard),
    )
    if symlink_parent_component:
        linked_root = tmp_path / "linked-root"
        linked_root.symlink_to(real, target_is_directory=True)
        selected_receipt = linked_root / receipt_path.name
    else:
        selected_receipt = tmp_path / "linked-receipt.json"
        selected_receipt.symlink_to(receipt_path)

    with pytest.raises(ValueError, match="symbolic-link components"):
        promote_v5_formal_production_search_evidence(
            plan,
            output_relative_path=shard.output_relative_path,
            receipt_path=selected_receipt,
            parent_dataset_path=parent_path,
            sidecar_path=sidecar_path,
            consumer_role=V5_FORMAL_PRODUCTION_TRAIN_CONSUMER_ROLE,
            pre_consume_source_guard=lambda: plan.source.bundle_sha256,
        )


def test_source_identity_requires_concrete_safe_file_hashes():
    with pytest.raises(ValueError, match="source-file hashes"):
        V5FormalProductionSourceIdentity.from_fingerprint(
            {
                "bundle_sha256": "1" * 64,
                "bundle_file_count": 1,
                "required_file_sha256": {},
            }
        )
    with pytest.raises(ValueError, match="safe and relative"):
        V5FormalProductionSourceIdentity(
            bundle_sha256="1" * 64,
            bundle_file_count=1,
            required_file_sha256=(("../escaped.py", "2" * 64),),
        )
