from __future__ import annotations

from hashlib import sha256
import json

import numpy as np
import pytest

from PosteriorV8.grouped_artifact_v5 import (
    V5_CHECKED_ARRAY_ARTIFACT_SCHEMA,
    V5_CHECKED_ARRAY_ARTIFACT_VERSION,
    array_manifest,
    canonical_json,
    write_checked_array_artifact,
)
from PosteriorV8.grouped_dataset_v5 import (
    V5_GROUPED_DATASET_SCHEMA,
    V5_GROUPED_DATASET_VERSION,
    clean_array,
)
from PosteriorV8.k1_balanced_dataset_plan_v5 import build_v5_k1_balanced_dataset_plan
from PosteriorV8.k1_forced_sobol_recipe_v5 import V5K1ForcedSobolCleanRecipe
from PosteriorV8.k1_phase_c_contract_v5 import K1_PHASE_C_BRANCHES
from PosteriorV8.k1_training_chain_dataset_audit_v5 import (
    audit_v5_k1_training_datasets,
    k1_parent_set_sha256,
)
from PosteriorV8.sobol_design_v5 import materialize_v5_unit_coordinates_for_indices
from PosteriorV8.sobol_recipe_coordinates_v5 import v5_sobol_recipe_design


def _write_role_artifact(tmp_path, plan, role, *, tamper=False):
    blocks = tuple(value for value in plan.blocks if value.role == role)
    recipes = []
    for block in blocks:
        design = v5_sobol_recipe_design(scramble_seed=block.scramble_seed)
        coordinates = materialize_v5_unit_coordinates_for_indices(design, (0,))[0]
        recipes.append(
            V5K1ForcedSobolCleanRecipe.create(
                plan=plan,
                block=block,
                sobol_index=0,
                original_unit_coordinates=coordinates,
            )
        )
    encoded = [value.canonical_json for value in recipes]
    if tamper:
        payload = json.loads(encoded[0])
        payload["source"]["forced_unit_coordinates"][0] = 0.125
        encoded[0] = canonical_json(payload)
    split_id = blocks[0].split_id
    arrays = {
        clean_array("recipe_canonical_json"): np.asarray(encoded),
        clean_array("target_pattern_id"): np.asarray(
            [value.target.pattern_id for value in recipes], dtype=np.int32
        ),
        clean_array("split_id"): np.asarray([split_id] * len(recipes)),
        clean_array("clean_group_id"): np.asarray(
            [value.clean_group_id for value in recipes]
        ),
    }
    core = {
        "container_schema": V5_CHECKED_ARRAY_ARTIFACT_SCHEMA,
        "container_version": V5_CHECKED_ARRAY_ARTIFACT_VERSION,
        "dataset_schema": V5_GROUPED_DATASET_SCHEMA,
        "dataset_version": V5_GROUPED_DATASET_VERSION,
        "counts": {"clean_recipes": len(recipes)},
        "arrays": array_manifest(arrays),
    }
    manifest = {
        **core,
        "manifest_sha256": sha256(canonical_json(core).encode()).hexdigest(),
    }
    path = tmp_path / f"{role}.gvd5"
    write_checked_array_artifact(path, manifest=manifest, arrays=arrays)
    group_ids = tuple(value.clean_group_id for value in recipes)
    artifact = {
        "path": str(path),
        "split_id": split_id,
        "clean_parent_count": len(recipes),
        "branch_counts": {value.branch_id: 1 for value in K1_PHASE_C_BRANCHES},
    }
    return artifact, group_ids


def test_training_dataset_audit_replays_forced_recipe_envelope(tmp_path):
    plan = build_v5_k1_balanced_dataset_plan(
        train_master_scramble_seed=101,
        tuning_master_scramble_seed=303,
        train_parents_per_branch=1,
        tuning_parents_per_branch=1,
    )
    train, train_groups = _write_role_artifact(tmp_path, plan, "train")
    tuning, tuning_groups = _write_role_artifact(tmp_path, plan, "tuning_validation")
    inventory = {
        "artifacts": {"train": [train], "tuning_validation": [tuning]},
        "splits": {
            "train": {"clean_parent_count": 12, "branch_counts": train["branch_counts"]},
            "tuning_validation": {
                "clean_parent_count": 12,
                "branch_counts": tuning["branch_counts"],
            },
            "train_parent_set_sha256": k1_parent_set_sha256(train_groups),
            "tuning_parent_set_sha256": k1_parent_set_sha256(tuning_groups),
        },
    }

    result = audit_v5_k1_training_datasets(inventory)

    assert result["train_tuning_disjoint"] is True
    assert result["branch_counts"]["train"] == train["branch_counts"]


def test_training_dataset_audit_rejects_rehashed_nested_forcing_drift(tmp_path):
    plan = build_v5_k1_balanced_dataset_plan(
        train_master_scramble_seed=101,
        tuning_master_scramble_seed=303,
        train_parents_per_branch=1,
        tuning_parents_per_branch=1,
    )
    train, _ = _write_role_artifact(tmp_path, plan, "train", tamper=True)

    with pytest.raises(ValueError, match="branch transform"):
        from PosteriorV8.k1_training_chain_dataset_audit_v5 import _artifact_audit

        _artifact_audit(train)
