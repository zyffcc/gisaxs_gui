from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import stat

import pytest

from PosteriorV8.grouped_artifact_v5 import canonical_json
from PosteriorV8 import k1_phase_c_holdout_collector_v5
from PosteriorV8 import k1_phase_c_holdout_worker_v5
from PosteriorV8.k1_dataset_disjointness_v5 import (
    V5K1RecipePopulation,
    build_v5_k1_train_tuning_disjointness_receipt,
    write_v5_k1_train_tuning_disjointness_receipt,
)
from PosteriorV8.k1_phase_c_contract_v5 import (
    K1_PHASE_C_OBSERVATION_STRESS_STRATA,
    K1_PHASE_C_RANGE_STRESS_STRATA,
)
from PosteriorV8.k1_phase_c_holdout_collector_v5 import (
    collect_v5_k1_phase_c_holdout,
)
from PosteriorV8.k1_phase_c_holdout_launch_plan_v5 import (
    K1_PHASE_C_HOLDOUT_FORMAL_RECIPES_PER_SHARD,
    V5K1PhaseCHoldoutLaunchConfig,
    build_v5_k1_phase_c_holdout_launch_plan,
    replay_v5_k1_phase_c_holdout_launch_inputs,
    validate_v5_k1_phase_c_holdout_launch_plan,
    write_v5_k1_phase_c_holdout_launch_plan,
)
from PosteriorV8.k1_phase_c_holdout_worker_v5 import (
    V5_K1_PHASE_C_HOLDOUT_TASK_COMPLETION_SCHEMA,
    V5_K1_PHASE_C_HOLDOUT_TASK_COMPLETION_VERSION,
    run_v5_k1_phase_c_holdout_task,
)
from PosteriorV8.k1_phase_c_plan_v5 import (
    build_v5_k1_phase_c_plan,
    planned_v5_k1_phase_c_stress_cell,
    write_v5_k1_phase_c_authoring_plan,
)
from PosteriorV8.k1_training_chain_contract_v5 import (
    V5_K1_TRAIN_SPLIT_ID,
    V5_K1_TUNING_SPLIT_ID,
)
from PosteriorV8.k1_training_chain_plan_v5 import K1_TRAINING_REQUIRED_SOURCE_FILES
from PosteriorV8.launch_k1_phase_c_holdout_dag_v5 import (
    V5CommandResult,
    launch_v5_k1_phase_c_holdout_dag,
)
from PosteriorV8.package_source_snapshot_v5 import (
    build_source_snapshot,
    extract_source_snapshot,
)


def _digest(label: str) -> str:
    return sha256(label.encode()).hexdigest()


def _minimal_source(root: Path) -> Path:
    required = {
        Path("AGENTS.md"),
        Path("pyproject.toml"),
        Path("requirements.txt"),
        Path("requirements-dev.txt"),
        Path("utils/__init__.py"),
        Path("src/gimap/example.py"),
        Path("docs/architecture/placeholder.md"),
        Path("docs/research/placeholder.md"),
        Path("utils/ML_Fitting_1D_GISAXS/PosteriorV8/package_source_snapshot_v5.py"),
        Path("utils/ML_Fitting_1D_GISAXS/PosteriorV8/study_protocol.py"),
        Path("utils/ML_Fitting_1D_GISAXS/PosteriorV8/launch_k1_phase_a_dag_v5.py"),
        *K1_TRAINING_REQUIRED_SOURCE_FILES,
    }
    for index, relative in enumerate(sorted(required)):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"fixture-source-{index}\n", encoding="utf-8")
    return root


def _population(role: str, split: str, marker: str) -> V5K1RecipePopulation:
    return V5K1RecipePopulation(
        role=role,
        split_id=split,
        plan_sha256=_digest("balanced-plan"),
        artifact_sha256s=(_digest(f"{marker}-artifact"),),
        manifest_sha256s=(_digest(f"{marker}-manifest"),),
        recipe_sha256s=(_digest(f"{marker}-recipe"),),
        clean_group_ids=(_digest(f"{marker}-group"),),
    )


@pytest.fixture
def launch_fixture(tmp_path: Path):
    dust = tmp_path / "data" / "dust" / "user" / "zhaiyufe"
    dust.mkdir(parents=True)
    staging = _minimal_source(dust / "source-staging")
    archive = dust / "source.tar"
    source_identity = build_source_snapshot(staging, archive)
    source_root = dust / "source-extracted"
    extract_source_snapshot(
        archive,
        source_root,
        expected_sha256=source_identity["archive_sha256"],
    )
    archive.chmod(0o400)
    phase_c = build_v5_k1_phase_c_plan(formal=True)
    phase_c_path = dust / "phase-c-plan.json"
    write_v5_k1_phase_c_authoring_plan(phase_c_path, phase_c)
    receipt = build_v5_k1_train_tuning_disjointness_receipt(
        (
            _population("train", V5_K1_TRAIN_SPLIT_ID, "train"),
            _population(
                "tuning_validation", V5_K1_TUNING_SPLIT_ID, "tuning"
            ),
        )
    )
    receipt_path = dust / "train-tuning.json"
    write_v5_k1_train_tuning_disjointness_receipt(receipt_path, receipt)
    config = V5K1PhaseCHoldoutLaunchConfig(
        source_root=source_root,
        source_archive=archive,
        expected_source_archive_sha256=source_identity["archive_sha256"],
        phase_c_plan=phase_c_path,
        expected_phase_c_plan_file_sha256=sha256(phase_c_path.read_bytes()).hexdigest(),
        train_tuning_receipt=receipt_path,
        expected_train_tuning_receipt_file_sha256=sha256(
            receipt_path.read_bytes()
        ).hexdigest(),
        run_root=dust / "runs" / "phase-c-holdout-r1",
        recipes_per_shard=K1_PHASE_C_HOLDOUT_FORMAL_RECIPES_PER_SHARD,
    )
    return dust, config, phase_c


def _publish_launch_plan(dust: Path, config: V5K1PhaseCHoldoutLaunchConfig):
    plan = build_v5_k1_phase_c_holdout_launch_plan(config, allowed_root=dust)
    config.run_root.mkdir(parents=True)
    for name in ("logs", "data", "completion", "audit"):
        (config.run_root / name).mkdir()
    target = Path(plan["layout"]["plan"])
    write_v5_k1_phase_c_holdout_launch_plan(target, plan)
    return plan, target


def test_holdout_launch_plan_covers_formal_all12_without_writes(launch_fixture):
    dust, config, _ = launch_fixture

    plan = build_v5_k1_phase_c_holdout_launch_plan(config, allowed_root=dust)

    assert validate_v5_k1_phase_c_holdout_launch_plan(plan) == plan
    assert plan["array"] == {
        "task_count": 48,
        "array_spec": "0-47",
        "expected_clean_parent_count": 13_824,
    }
    assert len(plan["tasks"]) == 48
    assert set(Counter(value["branch_id"] for value in plan["tasks"]).values()) == {4}
    assert not config.run_root.exists()
    assert set(
        replay_v5_k1_phase_c_holdout_launch_inputs(plan, allowed_root=dust)
    ) == {
        "source_bundle_sha256",
        "source_archive_sha256",
        "source_manifest_sha256",
        "source_tree_sha256",
        "phase_c_plan_file_sha256",
        "phase_c_plan_sha256",
        "train_tuning_receipt_file_sha256",
        "train_tuning_receipt_sha256",
        "train_tuning_claim_sha256",
    }


def test_holdout_launch_plan_and_worker_fail_closed(launch_fixture):
    dust, config, _ = launch_fixture
    plan, target = _publish_launch_plan(dust, config)
    tampered = dict(plan)
    tampered["plan_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="self-hash"):
        validate_v5_k1_phase_c_holdout_launch_plan(tampered)

    result = run_v5_k1_phase_c_holdout_task(
        target,
        0,
        expected_plan_sha256=plan["plan_sha256"],
        dry_run=True,
        allowed_root=dust,
    )
    assert result["status"] == "checked_dry_run"
    assert result["writes_performed"] is False
    assert not Path(plan["tasks"][0]["output"]).exists()
    with pytest.raises(ValueError, match="differs from the Slurm export"):
        run_v5_k1_phase_c_holdout_task(
            target,
            0,
            expected_plan_sha256="0" * 64,
            dry_run=True,
            allowed_root=dust,
        )


def test_worker_seals_artifact_before_completion(launch_fixture, monkeypatch):
    dust, config, _ = launch_fixture
    plan, target = _publish_launch_plan(dust, config)
    task = plan["tasks"][0]

    def fake_write(shard_plan, output, *, allowed_root):
        path = Path(output)
        path.write_text("{}\n", encoding="utf-8")
        path.chmod(0o400)
        return {
            "artifact_self_sha256": _digest("artifact-self"),
            "manifest": {"manifest_sha256": _digest("manifest")},
        }

    monkeypatch.setattr(
        k1_phase_c_holdout_worker_v5,
        "write_v5_k1_phase_c_holdout_shard",
        fake_write,
    )
    monkeypatch.setattr(
        k1_phase_c_holdout_worker_v5,
        "validate_v5_k1_phase_c_holdout_shard_payload",
        lambda payload: payload,
    )
    result = run_v5_k1_phase_c_holdout_task(
        target,
        0,
        expected_plan_sha256=plan["plan_sha256"],
        allowed_root=dust,
        hostname="max-wn001",
        environment={"SLURM_JOB_ID": "123", "SLURM_ARRAY_TASK_ID": "0"},
    )
    completion = Path(task["completion"])
    assert result["status"] == "PASS"
    assert stat.S_IMODE(completion.stat().st_mode) == 0o400
    assert completion.stat().st_nlink == 1
    assert completion.stat().st_mtime_ns >= Path(task["output"]).stat().st_mtime_ns


def test_collector_publishes_actual_three_way_receipt_completion_last(
    launch_fixture, monkeypatch
):
    dust, config, phase_c = launch_fixture
    plan, target = _publish_launch_plan(dust, config)
    input_identity = replay_v5_k1_phase_c_holdout_launch_inputs(
        plan, allowed_root=dust
    )
    artifact_payloads = {}
    for task in plan["tasks"]:
        task_id = task["array_task_id"]
        artifact_path = Path(task["output"])
        completion_path = Path(task["completion"])
        artifact_path.write_text(json.dumps({"task_id": task_id}), encoding="utf-8")
        artifact_path.chmod(0o400)
        block = phase_c.sobol_blocks[task["branch_ordinal"]]
        indices = range(task["split_offset"], task["split_offset"] + task["recipe_count"])
        cell_counts = Counter(
            planned_v5_k1_phase_c_stress_cell(
                phase_c, branch_id=block.branch_id, sobol_index=index
            )
            for index in indices
        )
        manifest = {
            "phase_c_sobol_block_sha256": task["phase_c_sobol_block_sha256"],
            "branch_id": block.branch_id,
            "branch_ordinal": block.branch_ordinal,
            "selection_sha256": _digest(f"selection-{task_id}"),
            "selected_sobol_indices": list(indices),
            "recipe_count": task["recipe_count"],
            "recipe_sha256s": [
                _digest(f"holdout-recipe-{block.branch_ordinal}-{index}")
                for index in indices
            ],
            "clean_group_ids": [
                _digest(f"holdout-group-{block.branch_ordinal}-{index}")
                for index in range(
                    task["split_offset"],
                    task["split_offset"] + task["recipe_count"],
                )
            ],
            "stress_cell_counts": [
                [range_name, observation_name, cell_counts[(range_name, observation_name)]]
                for observation_name in K1_PHASE_C_OBSERVATION_STRESS_STRATA
                for range_name in K1_PHASE_C_RANGE_STRESS_STRATA
            ],
            "manifest_sha256": _digest(f"manifest-{task_id}"),
        }
        artifact_self = _digest(f"artifact-self-{task_id}")
        artifact_payloads[task_id] = {
            "artifact_self_sha256": artifact_self,
            "manifest": manifest,
        }
        completion_core = {
            "schema": V5_K1_PHASE_C_HOLDOUT_TASK_COMPLETION_SCHEMA,
            "version": V5_K1_PHASE_C_HOLDOUT_TASK_COMPLETION_VERSION,
            "status": "PASS",
            "scientific_acceptance_evidence": False,
            "training_authorization_granted": False,
            "plan_sha256": plan["plan_sha256"],
            "array_task_id": task_id,
            "slurm_job_id": str(1000 + task_id),
            "hostname": "max-wn001",
            "task": task,
            "selection_sha256": _digest(f"selection-{task_id}"),
            "artifact": {
                "path": str(artifact_path),
                "file_sha256": sha256(artifact_path.read_bytes()).hexdigest(),
                "artifact_self_sha256": artifact_self,
                "manifest_sha256": manifest["manifest_sha256"],
                "byte_count": artifact_path.stat().st_size,
                "mode_octal": "0400",
                "nlink": 1,
            },
            "immutable_input_identity_pre": input_identity,
            "immutable_input_identity_post": input_identity,
            "completion_written_after_artifact_seal": True,
        }
        completion = {
            **completion_core,
            "completion_sha256": sha256(
                canonical_json(completion_core).encode()
            ).hexdigest(),
        }
        completion_path.write_text(json.dumps(completion), encoding="utf-8")
        completion_path.chmod(0o400)

    monkeypatch.setattr(
        k1_phase_c_holdout_collector_v5,
        "validate_v5_k1_phase_c_holdout_shard_payload",
        lambda payload: artifact_payloads[payload["task_id"]],
    )
    result = collect_v5_k1_phase_c_holdout(
        target,
        expected_plan_sha256=plan["plan_sha256"],
        allowed_root=dust,
        hostname="max-wn002",
        environment={"SLURM_JOB_ID": "2000"},
    )
    receipt = Path(plan["layout"]["three_way_disjointness_receipt"])
    completion = Path(plan["layout"]["holdout_completion"])
    assert result["status"] == "PASS"
    assert result["training_authorization_granted"] is False
    assert result["phase_c_exclusion_proven"] is True
    assert result["holdout_population"]["clean_parent_count"] == 13_824
    assert stat.S_IMODE(receipt.stat().st_mode) == 0o400
    assert completion.stat().st_mtime_ns >= receipt.stat().st_mtime_ns


def test_launcher_submit_holds_reads_back_and_releases_reverse(launch_fixture):
    dust, config, _ = launch_fixture
    preview = launch_v5_k1_phase_c_holdout_dag(config, allowed_root=dust)
    assert preview["status"] == "dry_run"
    assert not config.run_root.exists()
    released = set()
    calls = []

    def runner(argv):
        command = tuple(argv)
        calls.append(command)
        if command[0] == "sbatch":
            job_id = "701" if any(value.startswith("--array=") for value in command) else "702"
            return V5CommandResult(0, f"{job_id}\n", "")
        if command[:3] == ("scontrol", "show", "job"):
            job_id = command[-1]
            dependency = " Dependency=afterok:701" if job_id == "702" else ""
            reason = "JobHeldUser" if job_id not in released else (
                "Dependency" if job_id == "702" else "Resources"
            )
            return V5CommandResult(
                0,
                f"JobId={job_id} JobState=PENDING Reason={reason}{dependency}\n",
                "",
            )
        if command[:2] == ("scontrol", "release"):
            released.add(command[-1])
            return V5CommandResult(0, "", "")
        raise AssertionError(command)

    result = launch_v5_k1_phase_c_holdout_dag(
        config,
        submit=True,
        runner=runner,
        allowed_root=dust,
        hostname="max-wgs001",
    )
    assert result["status"] == "ALL_JOBS_RELEASED"
    assert result["job_ids"] == {"holdout_array": "701", "collector": "702"}
    assert result["release_order"] == ["collector", "holdout_array"]
    assert [value for value in calls if value[:2] == ("scontrol", "release")] == [
        ("scontrol", "release", "702"),
        ("scontrol", "release", "701"),
    ]
