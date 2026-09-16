from hashlib import sha256

import pytest

from PosteriorV8.build_grouped_dataset_v5 import build_v5_grouped_solution_dataset
from PosteriorV8.build_k1_balanced_grouped_shard_v5 import (
    materialize_v5_k1_grouped_recipe_specs,
    plan_v5_k1_balanced_grouped_shard,
)
from PosteriorV8.grouped_dataset_v5 import observation_array
from PosteriorV8.formal_production_search_contract_v5 import (
    topology_ids_for_v5_formal_production_stage,
)
from PosteriorV8.formal_production_search_plan_v5 import (
    V5FormalProductionSearchAuthorization,
)
from PosteriorV8.frozen_search_pipeline_contract_v5 import (
    V5SelectedTopologySearchSchedule,
)
from PosteriorV8.frozen_search_pipeline_v5 import (
    materialize_or_verify_v5_frozen_search_parent,
)
from PosteriorV8.grouped_artifact_v5 import canonical_json
from PosteriorV8.k1_balanced_dataset_plan_v5 import (
    build_v5_k1_balanced_dataset_plan,
)
from PosteriorV8.k1_balanced_full_search_parent_v5 import (
    project_v5_k1_balanced_full_search_parent,
)
from PosteriorV8.k1_balanced_full_search_authorization_v5 import (
    V5K1BalancedFullSearchTaskAuthorization,
    V5_K1_BALANCED_FULL_SEARCH_TASK_PLAN_SCHEMA,
    build_v5_k1_balanced_full_search_task_membership,
)
from PosteriorV8.k1_balanced_full_search_runtime_v5 import (
    V5K1BalancedFullSearchExecutableTask,
)


def _sha(value):
    return sha256(value.encode()).hexdigest()


def _source_parent():
    plan = build_v5_k1_balanced_dataset_plan(
        train_master_scramble_seed=11,
        tuning_master_scramble_seed=29,
        train_parents_per_branch=2,
        tuning_parents_per_branch=1,
    )
    block = plan.blocks[0]
    shard = plan_v5_k1_balanced_grouped_shard(
        dataset_plan=plan,
        block_sha256=block.block_sha256,
        shard_index=0,
        count=2,
        view_indices=(0, 1, 2),
    )
    source = build_v5_grouped_solution_dataset(
        materialize_v5_k1_grouped_recipe_specs(shard),
        dataset_id="balanced-source-fixture",
        generating_only=True,
    )
    return plan, block, shard, source


def test_projection_selects_one_curve_blind_view_and_preserves_source_rows():
    plan, block, _, source = _source_parent()

    projected = project_v5_k1_balanced_full_search_parent(
        source,
        source_parent_artifact_sha256="1" * 64,
        expected_balanced_dataset_plan_sha256=plan.sha256,
        expected_balanced_sobol_block_sha256=block.block_sha256,
        expected_role=block.role,
        expected_split_id=block.split_id,
    )

    assert projected.parent.recipe_count == 2
    assert projected.parent.observation_count == 2
    assert projected.parent.manifest["build_policy"]["generating_candidate_only"] is False
    assert all(value.selected_view_index in (0, 1) for value in projected.observation_selections)
    assert all(
        value.selected_topology_ids
        == topology_ids_for_v5_formal_production_stage("K1")
        for value in projected.query_sets
    )
    source_ids = set(source.arrays[observation_array("observation_id")].tolist())
    assert set(projected.parent.arrays[observation_array("observation_id")].tolist()) <= source_ids
    assert projected.audit["training_authorization_granted"] is False


def test_projection_rejects_historical_noise_policy_before_reconstruction(monkeypatch):
    from PosteriorV8 import observation_v5, k1_balanced_full_search_parent_v5 as projection
    with monkeypatch.context() as historical:
        historical.setattr(observation_v5, "NOISE_APPLICATION_VERSION",
                           "posterior_v8_observation_noise_poisson_lognormal_v1")
        plan, block, _, source = _source_parent()
    monkeypatch.setattr(projection, "build_v5_grouped_solution_dataset",
                        lambda *args, **kwargs: pytest.fail("must reject before reconstruction"))
    with pytest.raises(ValueError, match="acquisition policy is incompatible"):
        project_v5_k1_balanced_full_search_parent(
            source, source_parent_artifact_sha256="1" * 64,
            expected_balanced_dataset_plan_sha256=plan.sha256,
            expected_balanced_sobol_block_sha256=block.block_sha256,
            expected_role=block.role, expected_split_id=block.split_id,
        )


def test_projection_rejects_a_parent_from_another_balanced_block():
    plan, block, _, source = _source_parent()

    with pytest.raises(ValueError, match="escaped its balanced shard"):
        project_v5_k1_balanced_full_search_parent(
            source,
            source_parent_artifact_sha256="1" * 64,
            expected_balanced_dataset_plan_sha256=plan.sha256,
            expected_balanced_sobol_block_sha256="2" * 64,
            expected_role=block.role,
            expected_split_id=block.split_id,
        )


def test_projection_rejects_an_already_expanded_source_parent():
    plan, block, _, source = _source_parent()
    projected = project_v5_k1_balanced_full_search_parent(
        source,
        source_parent_artifact_sha256="1" * 64,
        expected_balanced_dataset_plan_sha256=plan.sha256,
        expected_balanced_sobol_block_sha256=block.block_sha256,
        expected_role=block.role,
        expected_split_id=block.split_id,
    )

    with pytest.raises(ValueError, match="generating-only"):
        project_v5_k1_balanced_full_search_parent(
            projected.parent,
            source_parent_artifact_sha256="3" * 64,
            expected_balanced_dataset_plan_sha256=plan.sha256,
            expected_balanced_sobol_block_sha256=block.block_sha256,
            expected_role=block.role,
            expected_split_id=block.split_id,
        )


def test_executable_task_replays_projection_membership_and_query_catalogs(tmp_path):
    plan, block, shard, source = _source_parent()
    projection = project_v5_k1_balanced_full_search_parent(
        source,
        source_parent_artifact_sha256="1" * 64,
        expected_balanced_dataset_plan_sha256=plan.sha256,
        expected_balanced_sobol_block_sha256=block.block_sha256,
        expected_role=block.role,
        expected_split_id=block.split_id,
    )
    membership = build_v5_k1_balanced_full_search_task_membership(
        projection,
        array_task_id=0,
        shard_index=0,
        split_offset=0,
        source_selection_sha256=shard.selection_sha256,
    )
    task_plan = {
        "schema": V5_K1_BALANCED_FULL_SEARCH_TASK_PLAN_SCHEMA,
        "global_launch_plan_sha256": _sha("global-plan"),
        "source": {
            "archive_sha256": _sha("archive"),
            "bundle_sha256": _sha("bundle"),
        },
        "identity_authorization_sha256": _sha("identity"),
        "array_task_id": 0,
        "stage_sha256": _sha("stage"),
        "parent_binding": {"artifact_sha256": "1" * 64},
        "membership": membership.to_payload(),
        "membership_sha256": membership.sha256,
        "output_relative_path": "shards/train/branch-00-shard-000000",
    }
    task_sha = sha256(canonical_json(task_plan).encode()).hexdigest()
    branch_count = sum(value.branch_count for value in membership.members)
    formal = V5FormalProductionSearchAuthorization(
        study_id="balanced-runtime-fixture",
        consumer_role="gradient_training",
        launch_plan_sha256=task_plan["global_launch_plan_sha256"],
        launch_source_bundle_sha256=task_plan["source"]["bundle_sha256"],
        source_identity_sha256=task_plan["identity_authorization_sha256"],
        shard_plan_sha256=task_sha,
        stage_id="K1",
        stage_sha256=task_plan["stage_sha256"],
        target_split="train",
        output_relative_path=task_plan["output_relative_path"],
        split_plan_sha256=plan.sha256,
        sobol_design_sha256=membership.members[0].sobol_design_sha256,
        protocol_sha256=_sha("protocol"),
        seed_schedule_sha256=_sha("seed-schedule"),
        optimizer_schedule_sha256=_sha("optimizer"),
        calibration_artifact_sha256=_sha("calibration"),
        recipe_sobol_indices=tuple(value.sobol_index for value in membership.members),
        clean_group_ids=tuple(value.clean_group_id for value in membership.members),
        recipe_membership_sha256=membership.recipe_membership_sha256,
        expected_query_count=2,
        expected_branch_count=branch_count,
        expected_exact_forward_calls=branch_count * 4096,
    )
    authorization = V5K1BalancedFullSearchTaskAuthorization(
        task_plan=task_plan,
        formal_authorization=formal,
    )
    schedule = V5SelectedTopologySearchSchedule(
        schedule_id="all-k1-runtime-fixture",
        selected_topology_ids=topology_ids_for_v5_formal_production_stage("K1"),
    )

    runtime = V5K1BalancedFullSearchExecutableTask(
        projection=projection,
        task_authorization=authorization,
        topology_schedule=schedule,
    )

    assert runtime.parent_dataset is projection.parent
    assert runtime.sha256 == task_sha
    assert runtime.view_indices_for_recipe(0) == (
        projection.observation_selections[0].selected_view_index,
    )
    assert tuple(value.sha256 for value in runtime.query_designs) == tuple(
        value.sha256 for value in projection.query_sets
    )
    parent_path = tmp_path / "projected-parent.gvd5"
    parent, receipt, recipes, reused = materialize_or_verify_v5_frozen_search_parent(
        runtime,
        parent_path,
    )
    assert parent.manifest == projection.parent.manifest
    assert receipt.path == parent_path
    assert recipes == projection.recipes
    assert reused is False
    _, replay_receipt, _, replay_reused = materialize_or_verify_v5_frozen_search_parent(
        runtime,
        parent_path,
    )
    assert replay_receipt.artifact_sha256 == receipt.artifact_sha256
    assert replay_reused is True
