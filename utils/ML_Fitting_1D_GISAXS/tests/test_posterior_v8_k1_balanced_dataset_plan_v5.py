from dataclasses import replace
import json
from pathlib import Path
import stat

import pytest

from PosteriorV8.k1_balanced_dataset_plan_v5 import (
    K1_BALANCED_FORMAL_RECIPES_PER_SHARD,
    K1_BALANCED_FORMAL_TRAIN_MASTER_SCRAMBLE_SEED,
    K1_BALANCED_FORMAL_TRAIN_PARENTS_PER_BRANCH,
    K1_BALANCED_FORMAL_TUNING_MASTER_SCRAMBLE_SEED,
    K1_BALANCED_FORMAL_TUNING_PARENTS_PER_BRANCH,
    K1_BALANCED_FORMAL_VIEW_INDICES,
    V5_K1_BALANCED_DATASET_ROLES,
    build_frozen_v5_k1_balanced_dataset_plan,
    build_v5_k1_balanced_dataset_plan,
    is_frozen_v5_k1_balanced_dataset_plan,
    main,
    v5_k1_balanced_formal_configuration_payload,
    v5_k1_balanced_dataset_authoring_payload,
    v5_k1_balanced_dataset_plan_from_payload,
    validate_v5_k1_balanced_dataset_plan,
    write_v5_k1_balanced_dataset_authoring_plan,
)
from PosteriorV8.k1_phase_c_contract_v5 import K1_PHASE_C_BRANCHES
from PosteriorV8.k1_phase_c_plan_v5 import build_v5_k1_phase_c_plan


def _plan():
    return build_v5_k1_balanced_dataset_plan(
        train_master_scramble_seed=11,
        tuning_master_scramble_seed=29,
        train_parents_per_branch=8,
        tuning_parents_per_branch=4,
    )


def test_plan_builds_two_independent_exactly_balanced_all12_schedules():
    plan = validate_v5_k1_balanced_dataset_plan(_plan())
    payload = plan.audit_payload()

    assert len(plan.blocks) == 2 * len(K1_PHASE_C_BRANCHES)
    assert payload["clean_parent_counts"] == {
        "train": 96,
        "tuning_validation": 48,
    }
    assert payload["branch_balance_is_exact"] is True
    for role in V5_K1_BALANCED_DATASET_ROLES:
        blocks = [value for value in plan.blocks if value.role == role]
        assert tuple(value.branch_id for value in blocks) == tuple(
            value.branch_id for value in K1_PHASE_C_BRANCHES
        )
        assert len({value.parent_count for value in blocks}) == 1


def test_train_tuning_and_phase_c_scramble_seeds_are_disjoint():
    plan = _plan()
    phase_c = build_v5_k1_phase_c_plan(formal=True)

    seeds = {value.scramble_seed for value in plan.blocks}
    assert len(seeds) == 24
    assert not seeds.intersection(value.scramble_seed for value in phase_c.sobol_blocks)
    assert plan.phase_c_plan_sha256 == phase_c.sha256
    assert plan.audit_payload()[
        "actual_recipe_hash_disjointness_receipt_required_before_inventory"
    ] is True


def test_minimal_authoring_payload_replays_but_unknown_or_changed_identity_fails():
    plan = _plan()
    authoring = {
        "train_master_scramble_seed": 11,
        "tuning_master_scramble_seed": 29,
        "train_parents_per_branch": 8,
        "tuning_parents_per_branch": 4,
        "plan_sha256": plan.sha256,
    }

    assert v5_k1_balanced_dataset_plan_from_payload(authoring) == plan
    with pytest.raises(ValueError, match="SHA-256"):
        v5_k1_balanced_dataset_plan_from_payload({**authoring, "plan_sha256": "0" * 64})
    with pytest.raises(ValueError, match="fields"):
        v5_k1_balanced_dataset_plan_from_payload({**authoring, "extra": True})
    with pytest.raises(ValueError, match="identity"):
        validate_v5_k1_balanced_dataset_plan(replace(plan, sha256="0" * 64))


def test_master_seeds_and_parent_counts_fail_closed():
    with pytest.raises(ValueError, match="must differ"):
        build_v5_k1_balanced_dataset_plan(
            train_master_scramble_seed=11,
            tuning_master_scramble_seed=11,
            train_parents_per_branch=8,
            tuning_parents_per_branch=4,
        )
    with pytest.raises(ValueError, match="must be positive"):
        build_v5_k1_balanced_dataset_plan(
            train_master_scramble_seed=11,
            tuning_master_scramble_seed=29,
            train_parents_per_branch=0,
            tuning_parents_per_branch=4,
        )


def test_authoring_plan_is_minimal_exclusive_and_read_only(tmp_path: Path):
    plan = _plan()
    target = tmp_path / "balanced-k1-authoring-plan.json"

    assert write_v5_k1_balanced_dataset_authoring_plan(target, plan) == target
    assert json.loads(target.read_text(encoding="utf-8")) == (
        v5_k1_balanced_dataset_authoring_payload(plan)
    )
    metadata = target.stat()
    assert stat.S_IMODE(metadata.st_mode) == 0o400
    assert metadata.st_nlink == 1
    assert v5_k1_balanced_dataset_plan_from_payload(
        json.loads(target.read_text(encoding="utf-8"))
    ) == plan
    with pytest.raises(FileExistsError, match="overwrite"):
        write_v5_k1_balanced_dataset_authoring_plan(target, plan)


def test_formal_configuration_freezes_e1_capacity_seeds_views_and_shards():
    config = v5_k1_balanced_formal_configuration_payload()
    plan = build_frozen_v5_k1_balanced_dataset_plan()

    assert config["parents_per_branch"] == {
        "train": 1152,
        "tuning_validation": 288,
    }
    assert config["clean_parent_counts"] == {
        "train": 13_824,
        "tuning_validation": 3_456,
    }
    assert config["view_indices"] == [0, 1, 2]
    assert config["recipes_per_shard"] == 288
    assert plan.train_master_scramble_seed == (
        K1_BALANCED_FORMAL_TRAIN_MASTER_SCRAMBLE_SEED
    )
    assert plan.tuning_master_scramble_seed == (
        K1_BALANCED_FORMAL_TUNING_MASTER_SCRAMBLE_SEED
    )
    assert plan.train_parents_per_branch == (
        K1_BALANCED_FORMAL_TRAIN_PARENTS_PER_BRANCH
    )
    assert plan.tuning_parents_per_branch == (
        K1_BALANCED_FORMAL_TUNING_PARENTS_PER_BRANCH
    )
    assert K1_BALANCED_FORMAL_VIEW_INDICES == (0, 1, 2)
    assert K1_BALANCED_FORMAL_RECIPES_PER_SHARD == 288
    assert is_frozen_v5_k1_balanced_dataset_plan(plan) is True
    assert is_frozen_v5_k1_balanced_dataset_plan(_plan()) is False


def test_formal_cli_writes_only_the_frozen_plan(tmp_path: Path):
    target = tmp_path / "formal-plan.json"

    assert main(("--output", str(target), "--formal")) == 0
    assert v5_k1_balanced_dataset_plan_from_payload(
        json.loads(target.read_text(encoding="utf-8"))
    ) == build_frozen_v5_k1_balanced_dataset_plan()
    with pytest.raises(ValueError, match="--formal"):
        main(("--output", str(tmp_path / "not-formal.json"),))
