from __future__ import annotations

import copy
from hashlib import sha256
import json
from pathlib import Path
import stat

import pytest

from PosteriorV8.k1_balanced_dataset_launch_plan_v5 import (
    V5K1BalancedDatasetLaunchConfig,
    build_v5_k1_balanced_dataset_launch_plan,
    replay_v5_k1_balanced_dataset_launch_inputs,
    validate_v5_k1_balanced_dataset_launch_plan,
    write_v5_k1_balanced_dataset_launch_plan,
)
from PosteriorV8 import k1_balanced_dataset_worker_v5
from PosteriorV8 import k1_balanced_dataset_collector_v5
from PosteriorV8.grouped_artifact_v5 import V5ArtifactReceipt, canonical_json
from PosteriorV8.k1_balanced_dataset_collector_v5 import (
    collect_v5_k1_balanced_dataset,
)
from PosteriorV8.k1_balanced_dataset_worker_v5 import (
    V5_K1_BALANCED_DATASET_COMPLETION_SCHEMA,
    V5_K1_BALANCED_DATASET_COMPLETION_VERSION,
    run_v5_k1_balanced_dataset_task,
)
from PosteriorV8.k1_dataset_disjointness_v5 import V5K1RecipePopulation
from PosteriorV8.k1_balanced_dataset_plan_v5 import (
    K1_BALANCED_FORMAL_RECIPES_PER_SHARD,
    K1_BALANCED_FORMAL_VIEW_INDICES,
    build_frozen_v5_k1_balanced_dataset_plan,
    build_v5_k1_balanced_dataset_plan,
    write_v5_k1_balanced_dataset_authoring_plan,
)
from PosteriorV8.k1_training_chain_plan_v5 import K1_TRAINING_REQUIRED_SOURCE_FILES
from PosteriorV8.launch_k1_balanced_dataset_dag_v5 import (
    V5CommandResult,
    launch_v5_k1_balanced_dataset_dag,
)
from PosteriorV8.package_source_snapshot_v5 import (
    build_source_snapshot,
    extract_source_snapshot,
)


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
    authoring = build_v5_k1_balanced_dataset_plan(
        train_master_scramble_seed=101,
        tuning_master_scramble_seed=303,
        train_parents_per_branch=5,
        tuning_parents_per_branch=3,
    )
    authoring_path = dust / "balanced-authoring.json"
    write_v5_k1_balanced_dataset_authoring_plan(authoring_path, authoring)
    config = V5K1BalancedDatasetLaunchConfig(
        source_root=source_root,
        source_archive=archive,
        expected_source_archive_sha256=source_identity["archive_sha256"],
        balanced_plan=authoring_path,
        expected_balanced_plan_file_sha256=sha256(authoring_path.read_bytes()).hexdigest(),
        run_root=dust / "runs" / "all-k1-data-r1",
        recipes_per_shard=2,
        view_indices=(0, 1, 2),
    )
    return dust, config, authoring


def test_launch_plan_maps_every_branch_local_shard_without_writing(launch_fixture):
    dust, config, authoring = launch_fixture

    plan = build_v5_k1_balanced_dataset_launch_plan(config, allowed_root=dust)
    validated = validate_v5_k1_balanced_dataset_launch_plan(plan)

    assert validated == plan
    assert plan["array"] == {
        "task_count": 60,
        "array_spec": "0-59",
        "expected_clean_parent_counts": {"train": 60, "tuning_validation": 36},
    }
    assert plan["balanced_dataset_plan"]["plan_sha256"] == authoring.sha256
    assert not config.run_root.exists()
    assert [value["array_task_id"] for value in plan["tasks"]] == list(range(60))
    assert len({value["output"] for value in plan["tasks"]}) == 60
    assert len({value["completion"] for value in plan["tasks"]}) == 60
    assert plan["execution_contract"]["training_authorization_granted"] is False


def test_launch_plan_rejects_unreviewed_or_mutated_inputs(launch_fixture):
    dust, config, _ = launch_fixture
    with pytest.raises(ValueError, match="explicit expectation"):
        build_v5_k1_balanced_dataset_launch_plan(
            V5K1BalancedDatasetLaunchConfig(
                **{
                    **config.__dict__,
                    "expected_balanced_plan_file_sha256": "0" * 64,
                }
            ),
            allowed_root=dust,
        )
    with pytest.raises(ValueError, match="duplicate-free"):
        build_v5_k1_balanced_dataset_launch_plan(
            V5K1BalancedDatasetLaunchConfig(
                **{**config.__dict__, "view_indices": (0, 0)}
            ),
            allowed_root=dust,
        )
    plan = build_v5_k1_balanced_dataset_launch_plan(config, allowed_root=dust)
    tampered = copy.deepcopy(plan)
    tampered["tasks"][0]["recipe_count"] += 1
    with pytest.raises(ValueError, match="SHA-256"):
        validate_v5_k1_balanced_dataset_launch_plan(tampered)
    tampered_core = dict(tampered)
    tampered_core.pop("plan_sha256")
    tampered["plan_sha256"] = sha256(canonical_json(tampered_core).encode()).hexdigest()
    validate_v5_k1_balanced_dataset_launch_plan(tampered)
    with pytest.raises(RuntimeError, match="task inventory"):
        replay_v5_k1_balanced_dataset_launch_inputs(tampered, allowed_root=dust)


def test_published_source_replay_survives_new_training_inventory(launch_fixture, monkeypatch):
    from PosteriorV8 import k1_training_chain_plan_v5 as training_plan

    dust, config, _ = launch_fixture
    plan = build_v5_k1_balanced_dataset_launch_plan(config, allowed_root=dust)
    before = replay_v5_k1_balanced_dataset_launch_inputs(plan, allowed_root=dust)
    monkeypatch.setattr(
        training_plan, "K1_TRAINING_REQUIRED_SOURCE_FILES",
        (*K1_TRAINING_REQUIRED_SOURCE_FILES, Path("future_training_module.py")),
    )
    assert replay_v5_k1_balanced_dataset_launch_inputs(plan, allowed_root=dust) == before
    with pytest.raises(FileNotFoundError, match="future_training_module"):
        build_v5_k1_balanced_dataset_launch_plan(config, allowed_root=dust)

    path = config.source_root / K1_TRAINING_REQUIRED_SOURCE_FILES[0]
    path.chmod(0o600)
    path.write_text("tampered historical source\n")
    path.chmod(0o444)
    with pytest.raises(RuntimeError, match="source snapshot changed"):
        replay_v5_k1_balanced_dataset_launch_inputs(plan, allowed_root=dust)


def test_historical_replay_still_verifies_files_outside_required_inventory(launch_fixture):
    dust, config, _ = launch_fixture
    plan = build_v5_k1_balanced_dataset_launch_plan(config, allowed_root=dust)
    relative = "src/gimap/example.py"
    assert relative not in plan["source"]["required_file_sha256"]
    path = config.source_root / relative
    path.chmod(0o600)
    path.write_text("mutated file outside required inventory\n")
    path.chmod(0o444)
    with pytest.raises(ValueError, match="extracted source file identity mismatch"):
        replay_v5_k1_balanced_dataset_launch_inputs(plan, allowed_root=dust)


@pytest.mark.parametrize("relative", ["../escape.py", "/outside.py", "src/../escape.py"])
def test_historical_replay_rejects_escaping_inventory_paths(launch_fixture, relative):
    dust, config, _ = launch_fixture
    plan = build_v5_k1_balanced_dataset_launch_plan(config, allowed_root=dust)
    plan["source"]["required_file_sha256"][relative] = "0" * 64
    core = {key: value for key, value in plan.items() if key != "plan_sha256"}
    plan["plan_sha256"] = sha256(canonical_json(core).encode()).hexdigest()
    with pytest.raises(ValueError, match="historical source inventory"):
        replay_v5_k1_balanced_dataset_launch_inputs(plan, allowed_root=dust)


def _publish_launch_plan(dust: Path, config: V5K1BalancedDatasetLaunchConfig):
    plan = build_v5_k1_balanced_dataset_launch_plan(config, allowed_root=dust)
    config.run_root.mkdir(parents=True)
    (config.run_root / "logs").mkdir()
    target = Path(plan["layout"]["plan"])
    write_v5_k1_balanced_dataset_launch_plan(target, plan)
    return plan, target


def test_worker_dry_run_replays_frozen_task_without_writes(launch_fixture):
    dust, config, _ = launch_fixture
    plan, target = _publish_launch_plan(dust, config)

    result = run_v5_k1_balanced_dataset_task(
        target,
        0,
        expected_plan_sha256=plan["plan_sha256"],
        dry_run=True,
        allowed_root=dust,
    )

    assert result["status"] == "checked_dry_run"
    assert result["writes_performed"] is False
    assert result["training_authorization_granted"] is False
    assert not Path(plan["tasks"][0]["output"]).exists()


def test_worker_publishes_completion_after_a_sealed_artifact(
    launch_fixture, monkeypatch
):
    dust, config, _ = launch_fixture
    plan, target = _publish_launch_plan(dust, config)
    task = plan["tasks"][0]
    Path(task["output"]).parent.mkdir(parents=True)
    Path(task["completion"]).parent.mkdir(parents=True)

    def fake_builder(shard, output, *, allowed_root):
        artifact = Path(output)
        artifact.write_bytes(b"checked-grouped-artifact")
        artifact.chmod(0o400)
        return object(), V5ArtifactReceipt(
            artifact,
            sha256(artifact.read_bytes()).hexdigest(),
            artifact.stat().st_size,
            "a" * 64,
        )

    monkeypatch.setattr(
        k1_balanced_dataset_worker_v5,
        "build_v5_k1_balanced_grouped_shard",
        fake_builder,
    )
    result = run_v5_k1_balanced_dataset_task(
        target,
        0,
        expected_plan_sha256=plan["plan_sha256"],
        allowed_root=dust,
        hostname="max-wn001",
        environment={"SLURM_JOB_ID": "123", "SLURM_ARRAY_TASK_ID": "0"},
    )

    completion = Path(task["completion"])
    assert result["status"] == "PASS"
    assert completion.exists()
    assert stat.S_IMODE(completion.stat().st_mode) == 0o400
    assert completion.stat().st_nlink == 1
    persisted = json.loads(completion.read_text(encoding="utf-8"))
    core = dict(persisted)
    supplied = core.pop("completion_sha256")
    assert supplied == sha256(canonical_json(core).encode()).hexdigest()
    assert persisted["completion_written_after_artifact_seal"] is True


def test_collector_requires_all_completions_then_publishes_train_tune_claim(
    launch_fixture, monkeypatch
):
    dust, config, _ = launch_fixture
    plan, target = _publish_launch_plan(dust, config)
    input_identity = replay_v5_k1_balanced_dataset_launch_inputs(
        plan, allowed_root=dust
    )
    (config.run_root / "audit").mkdir()
    for task in plan["tasks"]:
        artifact = Path(task["output"])
        completion = Path(task["completion"])
        artifact.parent.mkdir(parents=True, exist_ok=True)
        completion.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_bytes(f"artifact-{task['array_task_id']}".encode())
        artifact.chmod(0o400)
        artifact_sha = sha256(artifact.read_bytes()).hexdigest()
        completion_core = {
            "schema": V5_K1_BALANCED_DATASET_COMPLETION_SCHEMA,
            "version": V5_K1_BALANCED_DATASET_COMPLETION_VERSION,
            "status": "PASS",
            "scientific_acceptance_evidence": False,
            "training_authorization_granted": False,
            "plan_sha256": plan["plan_sha256"],
            "array_task_id": task["array_task_id"],
            "slurm_job_id": f"10{task['array_task_id']}",
            "hostname": "max-wn001",
            "task": task,
            "selection_sha256": sha256(
                f"selection-{task['array_task_id']}".encode()
            ).hexdigest(),
            "artifact": {
                "path": str(artifact),
                "artifact_sha256": artifact_sha,
                "manifest_sha256": sha256(
                    f"manifest-{task['array_task_id']}".encode()
                ).hexdigest(),
                "byte_count": artifact.stat().st_size,
                "mode_octal": "0400",
                "nlink": 1,
            },
            "immutable_input_identity_pre": input_identity,
            "immutable_input_identity_post": input_identity,
            "completion_written_after_artifact_seal": True,
        }
        completion_payload = {
            **completion_core,
            "completion_sha256": sha256(
                canonical_json(completion_core).encode()
            ).hexdigest(),
        }
        completion.write_text(
            json.dumps(completion_payload, sort_keys=True), encoding="utf-8"
        )
        completion.chmod(0o400)

    def fake_population(*, role, split_id, plan_sha256, artifacts, allowed_root):
        count = plan["array"]["expected_clean_parent_counts"][role]
        return V5K1RecipePopulation(
            role=role,
            split_id=split_id,
            plan_sha256=plan_sha256,
            artifact_sha256s=tuple(value.artifact_sha256 for value in artifacts),
            manifest_sha256s=tuple(value.manifest_sha256 for value in artifacts),
            recipe_sha256s=tuple(
                sha256(f"{role}-recipe-{index}".encode()).hexdigest()
                for index in range(count)
            ),
            clean_group_ids=tuple(
                sha256(f"{role}-group-{index}".encode()).hexdigest()
                for index in range(count)
            ),
        )

    monkeypatch.setattr(
        k1_balanced_dataset_collector_v5,
        "population_from_v5_k1_grouped_artifacts",
        fake_population,
    )
    result = collect_v5_k1_balanced_dataset(
        target,
        expected_plan_sha256=plan["plan_sha256"],
        allowed_root=dust,
        hostname="max-wn002",
        environment={"SLURM_JOB_ID": "456"},
    )

    assert result["status"] == "PASS"
    assert result["phase_c_exclusion_proven"] is False
    assert result["observed_clean_parent_counts"] == {
        "train": 60,
        "tuning_validation": 36,
    }
    receipt = Path(plan["layout"]["train_tuning_disjointness_receipt"])
    completion = Path(plan["layout"]["dataset_completion"])
    assert stat.S_IMODE(receipt.stat().st_mode) == 0o400
    assert stat.S_IMODE(completion.stat().st_mode) == 0o400


@pytest.mark.parametrize("hostname", ["max-wgs001", "max-fs-display006.desy.de"])
def test_launcher_dry_run_is_write_free_and_submit_releases_in_reverse_order(
    launch_fixture, hostname,
):
    dust, config, _ = launch_fixture
    dry_run = launch_v5_k1_balanced_dataset_dag(
        config,
        allowed_root=dust,
    )
    assert dry_run["status"] == "dry_run"
    assert dry_run["writes_performed"] is False
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
            if job_id not in released:
                body = f"JobId={job_id} JobState=PENDING Reason=JobHeldUser{dependency}"
            elif job_id == "702":
                body = f"JobId={job_id} JobState=PENDING Reason=Dependency{dependency}"
            else:
                body = f"JobId={job_id} JobState=PENDING Reason=Resources"
            return V5CommandResult(0, body + "\n", "")
        if command[:2] == ("scontrol", "release"):
            released.add(command[-1])
            return V5CommandResult(0, "", "")
        raise AssertionError(command)

    formal_authoring = build_frozen_v5_k1_balanced_dataset_plan()
    formal_authoring_path = dust / "balanced-authoring-formal.json"
    write_v5_k1_balanced_dataset_authoring_plan(
        formal_authoring_path, formal_authoring
    )
    formal_run_root = dust / "runs" / "all-k1-data-formal-r1"
    formal_config = V5K1BalancedDatasetLaunchConfig(
        **{
            **config.__dict__,
            "balanced_plan": formal_authoring_path,
            "expected_balanced_plan_file_sha256": sha256(
                formal_authoring_path.read_bytes()
            ).hexdigest(),
            "run_root": formal_run_root,
            "recipes_per_shard": K1_BALANCED_FORMAL_RECIPES_PER_SHARD,
            "view_indices": K1_BALANCED_FORMAL_VIEW_INDICES,
        }
    )
    result = launch_v5_k1_balanced_dataset_dag(
        formal_config,
        submit=True,
        runner=runner,
        allowed_root=dust,
        hostname=hostname,
    )

    assert result["status"] == "ALL_JOBS_RELEASED"
    assert result["job_ids"] == {"dataset_array": "701", "collector": "702"}
    assert result["release_order"] == ["collector", "dataset_array"]
    release_calls = [value for value in calls if value[:2] == ("scontrol", "release")]
    assert release_calls == [
        ("scontrol", "release", "702"),
        ("scontrol", "release", "701"),
    ]
    for name in ("held_submission_receipt", "launch_completion"):
        path = formal_run_root / "audit" / {
            "held_submission_receipt": "held-submission-receipt-v2.json",
            "launch_completion": "launch-completion-v2.json",
        }[name]
        assert stat.S_IMODE(path.stat().st_mode) == 0o400
        assert path.stat().st_nlink == 1


def test_submit_rejects_nonformal_counts_even_after_write_free_preview(launch_fixture):
    dust, config, _ = launch_fixture

    preview = launch_v5_k1_balanced_dataset_dag(config, allowed_root=dust)
    assert preview["plan"]["configuration"]["formal_e1_dataset_production"] is False
    with pytest.raises(RuntimeError, match="exact frozen E1"):
        launch_v5_k1_balanced_dataset_dag(
            config,
            submit=True,
            allowed_root=dust,
            hostname="max-wgs001",
        )
def test_noise_v2_rebuild_uses_v4_job_and_log_names():
    from PosteriorV8 import launch_k1_balanced_dataset_dag_v5 as launcher

    plan = {
        "source": {"root": "/data/dust/user/zhaiyufe/source-test"},
        "layout": {"logs": "/data/dust/user/zhaiyufe/new-run/logs", "plan": "/data/dust/user/zhaiyufe/new-run/plan.json"},
        "plan_sha256": "a" * 64,
        "array": {"array_spec": "0-59"},
    }
    for command in (launcher._dataset_command(plan), launcher._collector_command(plan, "123")):
        labels = [item for item in command if item.startswith(("--job-name=", "--output=", "--error="))]
        assert len(labels) == 3
        assert all("k1-balanced-all12-v4" in item for item in labels)
        assert all("all12-v3" not in item for item in labels)
    for relative in (launcher.DATASET_WRAPPER_RELATIVE, launcher.COLLECTOR_WRAPPER_RELATIVE):
        body = relative.read_text()
        assert "all12-v3" not in body
        assert "k1-balanced-all12-v4" in body
