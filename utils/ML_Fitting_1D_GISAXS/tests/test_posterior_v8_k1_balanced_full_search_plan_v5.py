from copy import deepcopy
from hashlib import sha256

import pytest

from utils.ML_Fitting_1D_GISAXS.PosteriorV8.contract import topology_from_id
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.formal_production_search_contract_v5 import (
    V5FormalProductionSearchStage,
    topology_ids_for_v5_formal_production_stage,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.grouped_artifact_v5 import canonical_json
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.k1_balanced_full_search_plan_v5 import (
    V5K1BalancedFullSearchParentBinding,
    build_v5_k1_balanced_full_search_plan,
    validate_v5_k1_balanced_full_search_plan,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.k1_balanced_full_search_authorization_v5 import (
    V5K1BalancedFullSearchRecipeMember,
    V5K1BalancedFullSearchTaskAuthorization,
    V5K1BalancedFullSearchTaskMembership,
    authorize_v5_k1_balanced_full_search_task,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.k1_balanced_dataset_plan_v5 import (
    K1_BALANCED_FORMAL_VIEW_INDICES,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.k1_phase_c_contract_v5 import (
    K1_PHASE_C_BRANCHES,
    K1_PHASE_C_SPLIT_ID,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.k1_training_identity_contract_v5 import (
    V5K1TrainingIdentityAuthorization,
    V5K1TrainingPopulationIdentity,
)


def _sha(value: str) -> str:
    return sha256(value.encode("ascii")).hexdigest()


class _K1StageFixture(V5FormalProductionSearchStage):
    def __init__(self):
        pass

    def audit_payload(self):
        selected = topology_ids_for_v5_formal_production_stage("K1")
        topology_schedule = {
            "schedule_id": "all-k1",
            "selected_topology_ids": list(selected),
        }
        local = {"schedule_id": "local", "point_count": 4096}
        optimizer = {"schedule_id": "optimizer"}
        protocol = {
            "protocol_tier": "paper_full_calibrated",
            "exact_forward_call_budget": 4096,
            "same_budget_for_every_branch": True,
            "uses_neural_proposal_scores": False,
            "calibration_identity": {"artifact_sha256": _sha("calibration")},
        }
        return {
            "schema": "gisaxs.posterior_v8.formal_production_search_stage/v1",
            "version": "posterior_v8_k1_then_k1_through_k2_then_all34_explicit_budget_v1",
            "stage_id": "K1",
            "stage_order": 0,
            "topology_scope": "all_and_only_K1_topologies",
            "selected_topology_ids": list(selected),
            "selected_topologies": [
                list(topology_from_id(value)) for value in selected
            ],
            "exact_forward_call_budget_per_branch": 4096,
            "topology_schedule": topology_schedule,
            "topology_schedule_sha256": _sha_payload(topology_schedule),
            "local_sobol_schedule": local,
            "local_sobol_schedule_sha256": _sha_payload(local),
            "optimizer_schedule": optimizer,
            "optimizer_schedule_sha256": _sha_payload(optimizer),
            "protocol": protocol,
            "protocol_sha256": _sha_payload(protocol),
        }


def _sha_payload(value: object) -> str:
    return sha256(canonical_json(value).encode()).hexdigest()


def _selected_indices(role, branch_ordinal, shard_index):
    role_offset = 0 if role == "train" else 10_000_000
    start = role_offset + branch_ordinal * 100_000 + shard_index * 288
    return tuple(range(start, start + 288))


def _selection_sha(role, branch_ordinal, shard_index, block_sha):
    indices = _selected_indices(role, branch_ordinal, shard_index)
    return _sha_payload(
        {
            "balanced_dataset_plan_sha256": _sha("balanced-plan"),
            "balanced_sobol_block_sha256": block_sha,
            "split_offset": shard_index * 288,
            "requested_count": 288,
            "actual_count": 288,
            "selected_sobol_indices": list(indices),
            "view_indices": list(K1_BALANCED_FORMAL_VIEW_INDICES),
            "selection_mode": "shard_index",
            "shard_index": shard_index,
            "generating_candidate_only": True,
        }
    )


def _parents():
    rows = []
    for role, shard_count in (("train", 4), ("tuning_validation", 1)):
        for branch_ordinal, branch in enumerate(K1_PHASE_C_BRANCHES):
            for shard_index in range(shard_count):
                task_id = len(rows)
                block_sha = _sha(f"block:{role}:{branch.branch_id}")
                rows.append(
                    V5K1BalancedFullSearchParentBinding(
                        array_task_id=task_id,
                        role=role,
                        split_id=role,
                        branch_id=branch.branch_id,
                        branch_ordinal=branch_ordinal,
                        balanced_sobol_block_sha256=block_sha,
                        shard_index=shard_index,
                        split_offset=shard_index * 288,
                        recipe_count=288,
                        selection_sha256=_selection_sha(
                            role,
                            branch_ordinal,
                            shard_index,
                            block_sha,
                        ),
                        parent_path=(
                            f"/data/dust/user/zhaiyufe/data/parent-{task_id}.gvd5"
                        ),
                        artifact_sha256=_sha(f"artifact:{task_id}"),
                        manifest_sha256=_sha(f"manifest:{task_id}"),
                        byte_count=1000 + task_id,
                        task_completion_path=(
                            f"/data/dust/user/zhaiyufe/completion/task-{task_id}.json"
                        ),
                        task_completion_file_sha256=_sha(
                            f"completion-file:{task_id}"
                        ),
                        task_completion_sha256=_sha(f"completion:{task_id}"),
                    )
                )
    return tuple(rows)


def _identity(parents):
    populations = []
    for role in ("train", "tuning_validation"):
        selected = tuple(value for value in parents if value.role == role)
        populations.append(
            V5K1TrainingPopulationIdentity(
                role=role,
                split_id=role,
                plan_sha256=_sha("balanced-plan"),
                artifact_sha256s=tuple(value.artifact_sha256 for value in selected),
                manifest_sha256s=tuple(value.manifest_sha256 for value in selected),
                clean_parent_count=sum(value.recipe_count for value in selected),
                recipe_set_sha256=_sha(f"recipes:{role}"),
                clean_group_set_sha256=_sha(f"groups:{role}"),
            )
        )
    populations.append(
        V5K1TrainingPopulationIdentity(
            role="phase_c_holdout",
            split_id=K1_PHASE_C_SPLIT_ID,
            plan_sha256=_sha("phase-c-plan"),
            artifact_sha256s=(_sha("phase-c-artifact"),),
            manifest_sha256s=(_sha("phase-c-manifest"),),
            clean_parent_count=13824,
            recipe_set_sha256=_sha("phase-c-recipes"),
            clean_group_set_sha256=_sha("phase-c-groups"),
        )
    )
    return V5K1TrainingIdentityAuthorization(
        source_archive_sha256=_sha("source-archive"),
        source_bundle_sha256=_sha("source-bundle"),
        balanced_dataset_completion_file_sha256=_sha("balanced-file"),
        balanced_dataset_completion_sha256=_sha("balanced-completion"),
        train_tuning_receipt_file_sha256=_sha("train-tune-file"),
        train_tuning_receipt_sha256=_sha("train-tune"),
        train_tuning_claim_sha256=_sha("train-tune-claim"),
        phase_c_completion_file_sha256=_sha("phase-c-file"),
        phase_c_completion_sha256=_sha("phase-c-completion"),
        three_way_receipt_file_sha256=_sha("three-way-file"),
        three_way_receipt_sha256=_sha("three-way"),
        phase_c_exclusion_claim_sha256=_sha("phase-c-claim"),
        populations=tuple(populations),
    )


def _plan():
    parents = _parents()
    identity = _identity(parents)
    return build_v5_k1_balanced_full_search_plan(
        source_archive_sha256=identity.source_archive_sha256,
        source_bundle_sha256=identity.source_bundle_sha256,
        balanced_dataset_launch_plan_file_sha256=_sha("balanced-launch-file"),
        balanced_dataset_launch_plan_sha256=_sha("balanced-launch"),
        balanced_dataset_completion_file_sha256=(
            identity.balanced_dataset_completion_file_sha256
        ),
        balanced_dataset_completion_sha256=(
            identity.balanced_dataset_completion_sha256
        ),
        identity_authorization=identity,
        k1_stage=_K1StageFixture(),
        search_run_root="/data/dust/user/zhaiyufe/runs/full-search-v1",
        parents=parents,
    )


def _membership(plan, task_id):
    parent = plan["parents"][task_id]
    indices = _selected_indices(
        parent["role"], parent["branch_ordinal"], parent["shard_index"]
    )
    members = tuple(
        V5K1BalancedFullSearchRecipeMember(
            sobol_index=index,
            clean_group_id=_sha(f"group:{task_id}:{index}"),
            recipe_sha256=_sha(f"recipe:{task_id}:{index}"),
            sobol_design_sha256=_sha(f"design:{task_id}"),
            query_set_sha256=_sha(f"query:{task_id}:{index}"),
            observation_selection_sha256=_sha(f"observation:{task_id}:{index}"),
            selected_view_index=index % 2,
            generating_topology_id=0,
            branch_count=13,
        )
        for index in indices
    )
    return V5K1BalancedFullSearchTaskMembership(
        array_task_id=task_id,
        source_parent_artifact_sha256=parent["parent"]["artifact_sha256"],
        balanced_dataset_plan_sha256=_sha("balanced-plan"),
        balanced_sobol_block_sha256=parent["balanced_sobol_block_sha256"],
        role=parent["role"],
        split_id=parent["split_id"],
        shard_index=parent["shard_index"],
        split_offset=parent["split_offset"],
        source_selection_sha256=parent["selection_sha256"],
        projected_parent_sha256=_sha(f"projection:{task_id}"),
        candidate_view_indices=(0, 1),
        members=members,
    )


def test_full_search_plan_binds_all_60_balanced_shards_without_granting_gradients():
    plan = validate_v5_k1_balanced_full_search_plan(_plan())

    assert plan["array"]["task_count"] == 60
    assert plan["array"]["clean_parent_counts"] == {
        "train": 13824,
        "tuning_validation": 3456,
    }
    assert plan["claim_limits"]["search_execution_authorized"] is True
    assert plan["claim_limits"]["full_search_supervision_complete"] is False
    assert plan["claim_limits"]["gradient_training_authorized"] is False


def test_full_search_plan_rejects_population_substitution_even_after_rehash():
    plan = _plan()
    drifted = deepcopy(plan)
    drifted.pop("plan_sha256")
    drifted["parents"][0]["parent"]["artifact_sha256"] = _sha("substitution")
    drifted["plan_sha256"] = _sha_payload(drifted)

    with pytest.raises(ValueError, match="authorized population"):
        validate_v5_k1_balanced_full_search_plan(drifted)


def test_full_search_plan_rejects_a_relaxed_gradient_claim_even_after_rehash():
    plan = _plan()
    drifted = deepcopy(plan)
    drifted.pop("plan_sha256")
    drifted["claim_limits"]["gradient_training_authorized"] = True
    drifted["plan_sha256"] = _sha_payload(drifted)

    with pytest.raises(ValueError, match="claim limits drifted"):
        validate_v5_k1_balanced_full_search_plan(drifted)


@pytest.mark.parametrize(
    ("task_id", "role", "consumer_role"),
    ((0, "train", "gradient_training"), (48, "tuning_validation", "tuning_validation_only")),
)
def test_full_search_task_authorization_binds_projection_and_existing_formal_receipt(
    task_id,
    role,
    consumer_role,
):
    plan = _plan()
    membership = _membership(plan, task_id)

    authorization = authorize_v5_k1_balanced_full_search_task(plan, membership)

    formal = authorization.formal_authorization
    assert formal.consumer_role == consumer_role
    assert formal.target_split == role
    assert formal.expected_query_count == 288
    assert formal.expected_branch_count == 288 * 13
    assert formal.expected_exact_forward_calls == 288 * 13 * 4096
    assert formal.shard_plan_sha256 == authorization.task_plan_sha256
    assert formal.recipe_membership_sha256 == membership.recipe_membership_sha256
    assert authorization.audit_payload()["gradient_training_authorized"] is False
    replay = V5K1BalancedFullSearchTaskAuthorization.from_payload(
        authorization.to_payload()
    )
    assert replay.sha256 == authorization.sha256


def test_full_search_task_authorization_rejects_parent_artifact_substitution():
    plan = _plan()
    original = _membership(plan, 0)
    values = {
        **original.__dict__,
        "source_parent_artifact_sha256": _sha("different-parent"),
    }
    substituted = V5K1BalancedFullSearchTaskMembership(**values)

    with pytest.raises(ValueError, match="global parent binding"):
        authorize_v5_k1_balanced_full_search_task(plan, substituted)


def test_full_search_task_membership_rejects_original_selection_drift():
    plan = _plan()
    original = _membership(plan, 0)
    values = {**original.__dict__, "source_selection_sha256": _sha("drift")}

    with pytest.raises(ValueError, match="source selection"):
        V5K1BalancedFullSearchTaskMembership(**values)
