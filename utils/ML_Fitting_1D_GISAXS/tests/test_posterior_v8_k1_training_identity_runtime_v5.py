from __future__ import annotations

from copy import deepcopy
from hashlib import sha256

import pytest

from utils.ML_Fitting_1D_GISAXS.PosteriorV8.grouped_artifact_v5 import canonical_json
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.k1_balanced_dataset_collector_v5 import (
    V5_K1_BALANCED_DATASET_COLLECTION_SCHEMA,
    V5_K1_BALANCED_DATASET_COLLECTION_VERSION,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.k1_dataset_disjointness_v5 import (
    V5K1RecipePopulation,
    build_v5_k1_dataset_disjointness_receipt,
    build_v5_k1_train_tuning_disjointness_receipt,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.k1_phase_c_contract_v5 import (
    K1_PHASE_C_SPLIT_ID,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.k1_phase_c_holdout_collector_v5 import (
    V5_K1_PHASE_C_HOLDOUT_COLLECTION_SCHEMA,
    V5_K1_PHASE_C_HOLDOUT_COLLECTION_VERSION,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.k1_training_identity_contract_v5 import (
    V5K1TrainingIdentityAuthorization,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.k1_training_identity_runtime_v5 import (
    issue_v5_k1_training_identity_authorization,
    issue_v5_k1_training_identity_authorization_from_files,
)


def _population(role: str, marker: str, plan: str) -> V5K1RecipePopulation:
    split = K1_PHASE_C_SPLIT_ID if role == "phase_c_holdout" else role
    return V5K1RecipePopulation(
        role=role,
        split_id=split,
        plan_sha256=plan * 64,
        artifact_sha256s=(marker * 64,),
        manifest_sha256s=(("f" if marker != "f" else "e") * 64,),
        recipe_sha256s=(("1" if role == "train" else "2" if role == "tuning_validation" else "3") * 64,),
        clean_group_ids=(("4" if role == "train" else "5" if role == "tuning_validation" else "6") * 64,),
    )


def _self_hashed(core: dict[str, object]) -> dict[str, object]:
    return {
        **core,
        "completion_sha256": sha256(canonical_json(core).encode("utf-8")).hexdigest(),
    }


def _fixture():
    populations = (
        _population("train", "a", "7"),
        _population("tuning_validation", "b", "7"),
        _population("phase_c_holdout", "c", "8"),
    )
    train_tuning = build_v5_k1_train_tuning_disjointness_receipt(populations[:2])
    three_way = build_v5_k1_dataset_disjointness_receipt(populations)
    train_tuning_file_sha = "9" * 64
    three_way_file_sha = "0" * 64
    balanced = _self_hashed(
        {
            "schema": V5_K1_BALANCED_DATASET_COLLECTION_SCHEMA,
            "version": V5_K1_BALANCED_DATASET_COLLECTION_VERSION,
            "status": "PASS",
            "scientific_acceptance_evidence": False,
            "training_authorization_granted": False,
            "phase_c_exclusion_proven": False,
            "plan_sha256": "d" * 64,
            "populations": {
                value.role: {
                    "recipe_set_sha256": value.recipe_set_sha256,
                    "clean_group_set_sha256": value.clean_group_set_sha256,
                    "clean_parent_count": value.clean_parent_count,
                }
                for value in populations[:2]
            },
            "train_tuning_disjointness_receipt": {
                "path": "/data/dust/user/zhaiyufe/train-tuning.json",
                "file_sha256": train_tuning_file_sha,
                "receipt_sha256": train_tuning["receipt_sha256"],
                "train_tuning_claim_sha256": train_tuning[
                    "train_tuning_claim_sha256"
                ],
                "mode_octal": "0400",
                "nlink": 1,
            },
            "immutable_input_identity_pre": {"source": "a" * 64},
            "immutable_input_identity_post": {"source": "a" * 64},
            "completion_written_after_receipt_seal": True,
        }
    )
    holdout = populations[2]
    phase_c = _self_hashed(
        {
            "schema": V5_K1_PHASE_C_HOLDOUT_COLLECTION_SCHEMA,
            "version": V5_K1_PHASE_C_HOLDOUT_COLLECTION_VERSION,
            "status": "PASS",
            "scientific_acceptance_evidence": False,
            "training_authorization_granted": False,
            "phase_c_exclusion_proven": True,
            "plan_sha256": "e" * 64,
            "holdout_population": {
                "recipe_set_sha256": holdout.recipe_set_sha256,
                "clean_group_set_sha256": holdout.clean_group_set_sha256,
                "clean_parent_count": holdout.clean_parent_count,
                "branch_counts": {},
                "all_25_stress_cells_per_branch": True,
                "stress_cell_max_minus_min_lte": 1,
            },
            "three_way_disjointness_receipt": {
                "path": "/data/dust/user/zhaiyufe/three-way.json",
                "file_sha256": three_way_file_sha,
                "receipt_sha256": three_way["receipt_sha256"],
                "phase_c_exclusion_claim_sha256": three_way[
                    "phase_c_exclusion_claim_sha256"
                ],
                "mode_octal": "0400",
                "nlink": 1,
            },
            "immutable_input_identity_pre": {"source": "b" * 64},
            "immutable_input_identity_post": {"source": "b" * 64},
            "completion_written_after_receipt_seal": True,
        }
    )
    return balanced, train_tuning, phase_c, three_way


def _issue():
    balanced, train_tuning, phase_c, three_way = _fixture()
    return issue_v5_k1_training_identity_authorization(
        source_archive_sha256="a" * 64,
        source_bundle_sha256="b" * 64,
        balanced_dataset_completion=balanced,
        balanced_dataset_completion_file_sha256="c" * 64,
        train_tuning_receipt=train_tuning,
        train_tuning_receipt_file_sha256="9" * 64,
        phase_c_completion=phase_c,
        phase_c_completion_file_sha256="d" * 64,
        three_way_receipt=three_way,
        three_way_receipt_file_sha256="0" * 64,
    )


def _write_publications(tmp_path):
    balanced, train_tuning, phase_c, three_way = _fixture()
    train_tuning_path = tmp_path / "train-tuning.json"
    three_way_path = tmp_path / "three-way.json"
    train_tuning_path.write_text(canonical_json(train_tuning), encoding="utf-8")
    three_way_path.write_text(canonical_json(three_way), encoding="utf-8")

    balanced_core = deepcopy(balanced)
    balanced_core.pop("completion_sha256")
    balanced_core["train_tuning_disjointness_receipt"].update(
        {
            "path": str(train_tuning_path),
            "file_sha256": sha256(train_tuning_path.read_bytes()).hexdigest(),
        }
    )
    balanced = _self_hashed(balanced_core)
    phase_c_core = deepcopy(phase_c)
    phase_c_core.pop("completion_sha256")
    phase_c_core["three_way_disjointness_receipt"].update(
        {
            "path": str(three_way_path),
            "file_sha256": sha256(three_way_path.read_bytes()).hexdigest(),
        }
    )
    phase_c = _self_hashed(phase_c_core)
    balanced_path = tmp_path / "balanced-completion.json"
    phase_c_path = tmp_path / "phase-c-completion.json"
    balanced_path.write_text(canonical_json(balanced), encoding="utf-8")
    phase_c_path.write_text(canonical_json(phase_c), encoding="utf-8")
    for path in (train_tuning_path, three_way_path, balanced_path, phase_c_path):
        path.chmod(0o400)
    return balanced_path, train_tuning_path, phase_c_path, three_way_path


def test_identity_runtime_replays_the_complete_exclusion_chain_without_granting_gradients():
    authorization = _issue()
    payload = authorization.to_payload()

    assert V5K1TrainingIdentityAuthorization.from_payload(payload).sha256 == (
        authorization.sha256
    )
    assert payload["phase_c_exclusion_proven"] is True
    assert payload["training_authorization_granted"] is False
    assert payload["full_search_supervision_complete"] is False
    assert tuple(payload["populations"]) == (
        "train",
        "tuning_validation",
        "phase_c_holdout",
    )


def test_identity_runtime_rejects_a_rehashed_completion_with_receipt_drift():
    balanced, train_tuning, phase_c, three_way = _fixture()
    core = deepcopy(balanced)
    core.pop("completion_sha256")
    core["train_tuning_disjointness_receipt"]["receipt_sha256"] = "f" * 64
    balanced = _self_hashed(core)

    with pytest.raises(ValueError, match="receipt binding drifted"):
        issue_v5_k1_training_identity_authorization(
            source_archive_sha256="a" * 64,
            source_bundle_sha256="b" * 64,
            balanced_dataset_completion=balanced,
            balanced_dataset_completion_file_sha256="c" * 64,
            train_tuning_receipt=train_tuning,
            train_tuning_receipt_file_sha256="9" * 64,
            phase_c_completion=phase_c,
            phase_c_completion_file_sha256="d" * 64,
            three_way_receipt=three_way,
            three_way_receipt_file_sha256="0" * 64,
        )


def test_identity_runtime_rejects_a_rehashed_completion_with_post_input_drift():
    balanced, train_tuning, phase_c, three_way = _fixture()
    core = deepcopy(phase_c)
    core.pop("completion_sha256")
    core["immutable_input_identity_post"] = {"source": "f" * 64}
    phase_c = _self_hashed(core)

    with pytest.raises(ValueError, match="completion contract is incomplete"):
        issue_v5_k1_training_identity_authorization(
            source_archive_sha256="a" * 64,
            source_bundle_sha256="b" * 64,
            balanced_dataset_completion=balanced,
            balanced_dataset_completion_file_sha256="c" * 64,
            train_tuning_receipt=train_tuning,
            train_tuning_receipt_file_sha256="9" * 64,
            phase_c_completion=phase_c,
            phase_c_completion_file_sha256="d" * 64,
            three_way_receipt=three_way,
            three_way_receipt_file_sha256="0" * 64,
        )


def test_identity_file_runtime_binds_the_exact_receipt_paths(tmp_path):
    balanced_path, train_tuning_path, phase_c_path, three_way_path = (
        _write_publications(tmp_path)
    )

    authorization = issue_v5_k1_training_identity_authorization_from_files(
        source_archive_sha256="a" * 64,
        source_bundle_sha256="b" * 64,
        balanced_dataset_completion_path=balanced_path,
        train_tuning_receipt_path=train_tuning_path,
        phase_c_completion_path=phase_c_path,
        three_way_receipt_path=three_way_path,
    )

    assert authorization.to_payload()["training_authorization_granted"] is False


def test_identity_file_runtime_rejects_an_identical_receipt_at_another_path(tmp_path):
    balanced_path, train_tuning_path, phase_c_path, three_way_path = (
        _write_publications(tmp_path)
    )
    duplicate = tmp_path / "duplicate-train-tuning.json"
    duplicate.write_bytes(train_tuning_path.read_bytes())
    duplicate.chmod(0o400)

    with pytest.raises(ValueError, match="points at another train/tuning receipt"):
        issue_v5_k1_training_identity_authorization_from_files(
            source_archive_sha256="a" * 64,
            source_bundle_sha256="b" * 64,
            balanced_dataset_completion_path=balanced_path,
            train_tuning_receipt_path=duplicate,
            phase_c_completion_path=phase_c_path,
            three_way_receipt_path=three_way_path,
        )
