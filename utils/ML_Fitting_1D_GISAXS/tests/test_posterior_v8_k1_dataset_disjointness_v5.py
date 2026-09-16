from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
import stat

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
from PosteriorV8.k1_dataset_disjointness_v5 import (
    V5K1GroupedArtifactBinding,
    V5K1RecipePopulation,
    build_v5_k1_dataset_disjointness_receipt,
    build_v5_k1_train_tuning_disjointness_receipt,
    population_from_v5_k1_grouped_artifacts,
    validate_v5_k1_dataset_disjointness_receipt,
    validate_v5_k1_train_tuning_disjointness_receipt,
    write_v5_k1_dataset_disjointness_receipt,
    write_v5_k1_train_tuning_disjointness_receipt,
)
from PosteriorV8.k1_phase_c_contract_v5 import K1_PHASE_C_SPLIT_ID


def _digest(label: str) -> str:
    return sha256(label.encode()).hexdigest()


def _population(role: str, labels: tuple[str, ...]) -> V5K1RecipePopulation:
    split = {
        "train": "train",
        "tuning_validation": "tuning_validation",
        "phase_c_holdout": K1_PHASE_C_SPLIT_ID,
    }[role]
    plan = "balanced-plan" if role != "phase_c_holdout" else "phase-c-plan"
    return V5K1RecipePopulation(
        role=role,
        split_id=split,
        plan_sha256=_digest(plan),
        artifact_sha256s=(_digest(f"{role}-artifact"),),
        manifest_sha256s=(_digest(f"{role}-manifest"),),
        recipe_sha256s=tuple(_digest(f"recipe-{value}") for value in labels),
        clean_group_ids=tuple(_digest(f"group-{value}") for value in labels),
    )


def _receipt():
    return build_v5_k1_dataset_disjointness_receipt(
        (
            _population("train", ("t0", "t1")),
            _population("tuning_validation", ("v0",)),
            _population("phase_c_holdout", ("h0", "h1")),
        )
    )


def test_actual_recipe_and_group_populations_are_pairwise_disjoint_and_replayable():
    receipt = validate_v5_k1_dataset_disjointness_receipt(_receipt())

    assert receipt["all_recipe_sets_pairwise_disjoint"] is True
    assert receipt["all_clean_group_sets_pairwise_disjoint"] is True
    assert set(receipt["populations"]) == {
        "train",
        "tuning_validation",
        "phase_c_holdout",
    }
    assert all(
        values == {"recipe_sha256_count": 0, "clean_group_id_count": 0}
        for values in receipt["pairwise_intersection_counts"].values()
    )
    assert len(receipt["train_tuning_claim_sha256"]) == 64
    assert len(receipt["phase_c_exclusion_claim_sha256"]) == 64


def test_train_tuning_only_receipt_is_explicitly_not_a_phase_c_claim(tmp_path):
    receipt = build_v5_k1_train_tuning_disjointness_receipt(
        (
            _population("train", ("t0", "t1")),
            _population("tuning_validation", ("v0",)),
        )
    )

    assert validate_v5_k1_train_tuning_disjointness_receipt(receipt) == receipt
    assert receipt["intersection_counts"] == {
        "recipe_sha256_count": 0,
        "clean_group_id_count": 0,
    }
    assert receipt["claim_limits"]["phase_c_exclusion_proven"] is False
    target = tmp_path / "train-tuning-disjointness.json"
    write_v5_k1_train_tuning_disjointness_receipt(target, receipt)
    assert stat.S_IMODE(target.stat().st_mode) == 0o400
    assert target.stat().st_nlink == 1
    with pytest.raises(FileExistsError, match="overwrite"):
        write_v5_k1_train_tuning_disjointness_receipt(target, receipt)


@pytest.mark.parametrize("identity_kind", ("recipe", "group"))
def test_any_actual_cross_population_overlap_fails_closed(identity_kind):
    train = _population("train", ("t0", "t1"))
    tuning = _population("tuning_validation", ("v0",))
    holdout = _population("phase_c_holdout", ("h0",))
    if identity_kind == "recipe":
        tuning = V5K1RecipePopulation(
            **{
                **tuning.__dict__,
                "recipe_sha256s": (train.recipe_sha256s[0],),
            }
        )
    else:
        tuning = V5K1RecipePopulation(
            **{
                **tuning.__dict__,
                "clean_group_ids": (train.clean_group_ids[0],),
            }
        )

    with pytest.raises(ValueError, match="overlap"):
        build_v5_k1_dataset_disjointness_receipt((train, tuning, holdout))
    with pytest.raises(ValueError, match="overlap"):
        build_v5_k1_train_tuning_disjointness_receipt((train, tuning))


def test_receipt_tampering_fails_even_after_rehashing_outer_payload():
    receipt = deepcopy(_receipt())
    receipt["populations"]["train"]["recipe_sha256s"][0] = _digest("tampered")
    core = dict(receipt)
    core.pop("receipt_sha256")
    receipt["receipt_sha256"] = sha256(
        json.dumps(core, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    with pytest.raises(ValueError, match="derived identities"):
        validate_v5_k1_dataset_disjointness_receipt(receipt)


def test_receipt_writer_is_exclusive_and_seals_0400_nlink1(tmp_path):
    target = tmp_path / "disjointness.json"

    write_v5_k1_dataset_disjointness_receipt(target, _receipt())

    metadata = target.stat()
    assert stat.S_IMODE(metadata.st_mode) == 0o400
    assert metadata.st_nlink == 1
    with pytest.raises(FileExistsError):
        write_v5_k1_dataset_disjointness_receipt(target, _receipt())


def test_population_is_derived_from_checked_immutable_grouped_identity_arrays(tmp_path):
    plan_sha = _digest("balanced-plan")
    arrays = {
        clean_array("recipe_sha256"): np.asarray(
            [_digest("recipe-0"), _digest("recipe-1")]
        ),
        clean_array("clean_group_id"): np.asarray(
            [_digest("group-0"), _digest("group-1")]
        ),
        clean_array("split_id"): np.asarray(["train", "train"]),
        clean_array("split_plan_sha256"): np.asarray([plan_sha, plan_sha]),
    }
    core = {
        "container_schema": V5_CHECKED_ARRAY_ARTIFACT_SCHEMA,
        "container_version": V5_CHECKED_ARRAY_ARTIFACT_VERSION,
        "dataset_schema": V5_GROUPED_DATASET_SCHEMA,
        "dataset_version": V5_GROUPED_DATASET_VERSION,
        "arrays": array_manifest(arrays),
    }
    manifest = {
        **core,
        "manifest_sha256": sha256(canonical_json(core).encode()).hexdigest(),
    }
    path = tmp_path / "train.gvd5"
    receipt = write_checked_array_artifact(path, manifest=manifest, arrays=arrays)
    path.chmod(0o400)

    population = population_from_v5_k1_grouped_artifacts(
        role="train",
        split_id="train",
        plan_sha256=plan_sha,
        artifacts=(
            V5K1GroupedArtifactBinding(
                path=path,
                artifact_sha256=receipt.artifact_sha256,
                manifest_sha256=receipt.manifest_sha256,
            ),
        ),
        allowed_root=tmp_path,
    )

    assert population.clean_parent_count == 2
    assert population.recipe_sha256s == tuple(sorted(arrays[clean_array("recipe_sha256")]))
    assert population.clean_group_ids == tuple(
        sorted(arrays[clean_array("clean_group_id")])
    )
