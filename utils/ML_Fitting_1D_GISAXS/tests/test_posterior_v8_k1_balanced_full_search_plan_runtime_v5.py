from __future__ import annotations

from copy import deepcopy
from pathlib import Path, PurePosixPath

import pytest

from utils.ML_Fitting_1D_GISAXS.PosteriorV8 import (
    k1_balanced_full_search_plan_runtime_v5 as runtime,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.k1_balanced_dataset_collector_v5 import (
    V5_K1_BALANCED_DATASET_COLLECTION_SCHEMA,
    V5_K1_BALANCED_DATASET_COLLECTION_VERSION,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.k1_balanced_dataset_worker_v5 import (
    V5_K1_BALANCED_DATASET_COMPLETION_SCHEMA,
    V5_K1_BALANCED_DATASET_COMPLETION_VERSION,
)
from utils.ML_Fitting_1D_GISAXS.tests.test_posterior_v8_k1_balanced_full_search_plan_v5 import (
    _K1StageFixture,
    _identity,
    _parents,
    _plan,
    _sha,
)


@pytest.mark.parametrize("root_type", [Path, PurePosixPath])
def test_publication_path_accepts_lexical_root_without_weakening_containment(
    tmp_path, root_type
):
    child = tmp_path / "publication"
    child.mkdir()
    assert runtime._under_root(child, root_type(tmp_path), "test") == child  # noqa: SLF001
    with pytest.raises(ValueError):
        runtime._under_root(tmp_path, root_type(child), "test")  # noqa: SLF001


def _task(parent):
    return {
        "array_task_id": parent.array_task_id,
        "role": parent.role,
        "split_id": parent.split_id,
        "branch_id": parent.branch_id,
        "branch_ordinal": parent.branch_ordinal,
        "balanced_sobol_block_sha256": parent.balanced_sobol_block_sha256,
        "shard_index": parent.shard_index,
        "split_offset": parent.split_offset,
        "recipe_count": parent.recipe_count,
        "output": parent.parent_path,
        "completion": parent.task_completion_path,
    }


def _task_completion(parent, task, launch_inputs):
    core = {
        "schema": V5_K1_BALANCED_DATASET_COMPLETION_SCHEMA,
        "version": V5_K1_BALANCED_DATASET_COMPLETION_VERSION,
        "status": "PASS",
        "scientific_acceptance_evidence": False,
        "training_authorization_granted": False,
        "plan_sha256": _sha("balanced-launch"),
        "array_task_id": parent.array_task_id,
        "slurm_job_id": str(1000 + parent.array_task_id),
        "hostname": "max-wn001",
        "task": task,
        "selection_sha256": parent.selection_sha256,
        "artifact": {
            "path": parent.parent_path,
            "artifact_sha256": parent.artifact_sha256,
            "manifest_sha256": parent.manifest_sha256,
            "byte_count": parent.byte_count,
            "mode_octal": "0400",
            "nlink": 1,
        },
        "immutable_input_identity_pre": launch_inputs,
        "immutable_input_identity_post": launch_inputs,
        "completion_written_after_artifact_seal": True,
    }
    return {**core, "completion_sha256": parent.task_completion_sha256}


def test_task_binding_replays_completion_and_artifact_identity(monkeypatch):
    parent = _parents()[0]
    task = _task(parent)
    launch_inputs = {"balanced_plan_sha256": _sha("balanced-plan")}
    completion = _task_completion(parent, task, launch_inputs)
    monkeypatch.setattr(
        runtime,
        "_self_hash",
        lambda payload, **kwargs: parent.task_completion_sha256,
    )
    monkeypatch.setattr(
        runtime,
        "_sealed_json",
        lambda *args, **kwargs: (
            completion,
            Path(parent.task_completion_path),
            parent.task_completion_file_sha256,
        ),
    )
    monkeypatch.setattr(runtime, "_under_root", lambda path, root, name: path)
    monkeypatch.setattr(
        runtime,
        "_sealed_regular",
        lambda path, name: (path, parent.artifact_sha256, parent.byte_count),
    )
    verified_parents = []
    monkeypatch.setattr(
        runtime,
        "_verify_parent_noise_policy",
        lambda path, artifact: verified_parents.append((path, artifact)),
    )

    binding, completion_sha = runtime._task_binding(
        task,
        plan_sha256=_sha("balanced-launch"),
        launch_input_identity=launch_inputs,
        allowed_root=Path("/data/dust/user/zhaiyufe"),
    )

    assert binding == parent
    assert completion_sha == parent.task_completion_sha256
    assert verified_parents == [(Path(parent.parent_path), completion["artifact"])]


def test_task_binding_rejects_post_completion_input_drift(monkeypatch):
    parent = _parents()[0]
    task = _task(parent)
    launch_inputs = {"balanced_plan_sha256": _sha("balanced-plan")}
    completion = _task_completion(parent, task, launch_inputs)
    completion["immutable_input_identity_post"] = {"drift": _sha("drift")}
    monkeypatch.setattr(
        runtime,
        "_self_hash",
        lambda payload, **kwargs: parent.task_completion_sha256,
    )
    monkeypatch.setattr(
        runtime,
        "_sealed_json",
        lambda *args, **kwargs: (
            completion,
            Path(parent.task_completion_path),
            parent.task_completion_file_sha256,
        ),
    )

    with pytest.raises(ValueError, match="completion contract drifted"):
        runtime._task_binding(
            task,
            plan_sha256=_sha("balanced-launch"),
            launch_input_identity=launch_inputs,
            allowed_root=Path("/data/dust/user/zhaiyufe"),
        )


def test_noise_preflight_reads_real_grouped_artifact(tmp_path, monkeypatch):
    from PosteriorV8.grouped_dataset_v5 import write_v5_grouped_dataset
    from utils.ML_Fitting_1D_GISAXS.tests.test_posterior_v8_k1_balanced_full_search_parent_v5 import (
        _source_parent,
    )

    _, _, _, dataset = _source_parent()
    path = tmp_path / "parent.gvd5"
    receipt = write_v5_grouped_dataset(dataset, path)
    path.chmod(0o400)
    artifact = {
        "artifact_sha256": receipt.artifact_sha256,
        "manifest_sha256": receipt.manifest_sha256,
        "byte_count": receipt.byte_count,
    }
    runtime._verify_parent_noise_policy(path, artifact)
    for field in artifact:
        changed = {**artifact, field: 0 if field == "byte_count" else "0" * 64}
        with pytest.raises(ValueError, match="semantic receipt drifted"):
            runtime._verify_parent_noise_policy(path, changed)

    monkeypatch.setattr(runtime, "NOISE_APPLICATION_VERSION", "different-noise-version")
    with pytest.raises(ValueError, match="noise policy is incompatible"):
        runtime._verify_parent_noise_policy(path, artifact)
    assert runtime.file_sha256(path, "preserved fixture") == receipt.artifact_sha256


def _config():
    identity = _identity(_parents())
    return runtime.V5K1BalancedFullSearchPlanConfig(
        source_archive_sha256=identity.source_archive_sha256,
        source_bundle_sha256=identity.source_bundle_sha256,
        balanced_dataset_launch_plan_path=Path(
            "/data/dust/user/zhaiyufe/dataset/launch.json"
        ),
        balanced_dataset_launch_plan_file_sha256=_sha("balanced-launch-file"),
        balanced_dataset_launch_plan_sha256=_sha("balanced-launch"),
        balanced_dataset_completion_path=Path(
            "/data/dust/user/zhaiyufe/dataset/completion.json"
        ),
        balanced_dataset_completion_file_sha256=(
            identity.balanced_dataset_completion_file_sha256
        ),
        balanced_dataset_completion_sha256=(
            identity.balanced_dataset_completion_sha256
        ),
        train_tuning_receipt_path=Path(
            "/data/dust/user/zhaiyufe/dataset/train-tune.json"
        ),
        train_tuning_receipt_file_sha256=(
            identity.train_tuning_receipt_file_sha256
        ),
        train_tuning_receipt_sha256=identity.train_tuning_receipt_sha256,
        train_tuning_claim_sha256=identity.train_tuning_claim_sha256,
        phase_c_completion_path=Path(
            "/data/dust/user/zhaiyufe/phase-c/completion.json"
        ),
        phase_c_completion_file_sha256=identity.phase_c_completion_file_sha256,
        phase_c_completion_sha256=identity.phase_c_completion_sha256,
        three_way_receipt_path=Path(
            "/data/dust/user/zhaiyufe/phase-c/three-way.json"
        ),
        three_way_receipt_file_sha256=identity.three_way_receipt_file_sha256,
        three_way_receipt_sha256=identity.three_way_receipt_sha256,
        phase_c_exclusion_claim_sha256=identity.phase_c_exclusion_claim_sha256,
        k1_stage=_K1StageFixture(),
        search_run_root="/data/dust/user/zhaiyufe/runs/full-search-v1",
    )


def test_plan_runtime_derives_all_60_bindings_from_published_completions(
    monkeypatch,
):
    parents = _parents()
    identity = _identity(parents)
    launch_inputs = {"balanced_plan_sha256": _sha("balanced-plan")}
    launch = {
        "plan_sha256": _sha("balanced-launch"),
        "layout": {
            "plan": "/data/dust/user/zhaiyufe/dataset/launch.json",
            "dataset_completion": (
                "/data/dust/user/zhaiyufe/dataset/completion.json"
            ),
        },
        "array": {
            "expected_clean_parent_counts": {
                "train": 13824,
                "tuning_validation": 3456,
            }
        },
        "tasks": [_task(value) for value in parents],
    }
    balanced = {
        "schema": V5_K1_BALANCED_DATASET_COLLECTION_SCHEMA,
        "version": V5_K1_BALANCED_DATASET_COLLECTION_VERSION,
        "status": "PASS",
        "plan_sha256": launch["plan_sha256"],
        "task_count": 60,
        "task_completion_sha256s": [
            value.task_completion_sha256 for value in parents
        ],
        "observed_clean_parent_counts": launch["array"][
            "expected_clean_parent_counts"
        ],
        "immutable_input_identity_pre": launch_inputs,
        "immutable_input_identity_post": launch_inputs,
        "completion_written_after_receipt_seal": True,
        "completion_sha256": identity.balanced_dataset_completion_sha256,
    }
    config = _config()

    def fake_sealed(path, name, **kwargs):
        # Exercise the default PurePosixPath contract root through the public API.
        assert isinstance(kwargs["allowed_root"], Path)
        if name == "balanced dataset launch plan":
            return launch, path, config.balanced_dataset_launch_plan_file_sha256
        return balanced, path, config.balanced_dataset_completion_file_sha256

    monkeypatch.setattr(runtime, "_sealed_json", fake_sealed)
    monkeypatch.setattr(
        runtime, "validate_v5_k1_balanced_dataset_launch_plan", lambda value: value
    )
    monkeypatch.setattr(
        runtime,
        "replay_v5_k1_balanced_dataset_launch_inputs",
        lambda value, **kwargs: launch_inputs,
    )
    monkeypatch.setattr(
        runtime,
        "_self_hash",
        lambda payload, **kwargs: identity.balanced_dataset_completion_sha256,
    )
    monkeypatch.setattr(
        runtime,
        "issue_v5_k1_training_identity_authorization_from_files",
        lambda **kwargs: identity,
    )
    iterator = iter(parents)
    monkeypatch.setattr(
        runtime,
        "_task_binding",
        lambda *args, **kwargs: (
            (value := next(iterator)),
            value.task_completion_sha256,
        ),
    )

    assert runtime.build_v5_k1_balanced_full_search_plan_from_publications(
        config
    ) == _plan()


def test_plan_runtime_rejects_an_unexpected_receipt_chain(monkeypatch):
    config = _config()
    monkeypatch.setattr(
        runtime,
        "_sealed_json",
        lambda *args, **kwargs: ({}, Path(args[0]), _sha("wrong")),
    )
    monkeypatch.setattr(
        runtime, "validate_v5_k1_balanced_dataset_launch_plan", lambda value: {}
    )

    with pytest.raises((KeyError, ValueError)):
        runtime.build_v5_k1_balanced_full_search_plan_from_publications(
            deepcopy(config)
        )
