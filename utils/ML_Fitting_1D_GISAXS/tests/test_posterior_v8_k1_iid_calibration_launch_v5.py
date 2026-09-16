from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import stat

import pytest

from PosteriorV8 import k1_iid_calibration_worker_v5
from PosteriorV8.grouped_artifact_v5 import canonical_json
from PosteriorV8.k1_dataset_disjointness_v5 import (
    V5K1RecipePopulation,
    build_v5_k1_dataset_disjointness_receipt,
    write_v5_k1_dataset_disjointness_receipt,
)
from PosteriorV8.k1_iid_calibration_collector_v5 import _build_four_way_receipt
from PosteriorV8.k1_iid_calibration_launch_plan_v5 import (
    V5K1IIDCalibrationLaunchConfig,
    build_v5_k1_iid_calibration_launch_plan,
    replay_v5_k1_iid_calibration_launch_inputs,
    validate_v5_k1_iid_calibration_launch_plan,
    write_v5_k1_iid_calibration_launch_plan,
)
from PosteriorV8.k1_iid_calibration_worker_v5 import (
    run_v5_k1_iid_calibration_task,
)
from PosteriorV8.k1_training_chain_contract_v5 import (
    V5_K1_TRAIN_SPLIT_ID,
    V5_K1_TUNING_SPLIT_ID,
)
from PosteriorV8.k1_training_chain_plan_v5 import K1_TRAINING_REQUIRED_SOURCE_FILES
from PosteriorV8.k1_phase_c_contract_v5 import K1_PHASE_C_SPLIT_ID
from PosteriorV8.launch_k1_iid_calibration_dag_v5 import (
    V5CommandResult,
    launch_v5_k1_iid_calibration_dag,
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


def _population(role: str, split_id: str) -> V5K1RecipePopulation:
    plan_label = "balanced-plan" if role != "phase_c_holdout" else "phase-c-plan"
    return V5K1RecipePopulation(
        role=role,
        split_id=split_id,
        plan_sha256=_digest(plan_label),
        artifact_sha256s=(_digest(f"{role}-artifact"),),
        manifest_sha256s=(_digest(f"{role}-manifest"),),
        recipe_sha256s=(_digest(f"{role}-recipe"),),
        clean_group_ids=(_digest(f"{role}-group"),),
    )


@pytest.fixture
def calibration_launch_fixture(tmp_path: Path):
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
    receipt = build_v5_k1_dataset_disjointness_receipt(
        (
            _population("train", V5_K1_TRAIN_SPLIT_ID),
            _population("tuning_validation", V5_K1_TUNING_SPLIT_ID),
            _population("phase_c_holdout", K1_PHASE_C_SPLIT_ID),
        )
    )
    receipt_path = dust / "three-way.json"
    write_v5_k1_dataset_disjointness_receipt(receipt_path, receipt)
    config = V5K1IIDCalibrationLaunchConfig(
        source_root=source_root,
        source_archive=archive,
        expected_source_archive_sha256=source_identity["archive_sha256"],
        three_way_disjointness_receipt=receipt_path,
        expected_three_way_receipt_file_sha256=sha256(
            receipt_path.read_bytes()
        ).hexdigest(),
        run_root=dust / "runs" / "iid-calibration-r1",
    )
    return dust, config, receipt


def _publish_plan(dust: Path, config: V5K1IIDCalibrationLaunchConfig):
    plan = build_v5_k1_iid_calibration_launch_plan(config, allowed_root=dust)
    config.run_root.mkdir(parents=True)
    for name in ("logs", "data", "task-completion", "audit", "artifacts"):
        (config.run_root / name).mkdir()
    target = Path(plan["layout"]["plan"])
    write_v5_k1_iid_calibration_launch_plan(target, plan)
    return plan, target


def test_iid_calibration_plan_is_formal_write_free_and_content_bound(
    calibration_launch_fixture,
):
    dust, config, _ = calibration_launch_fixture
    plan = build_v5_k1_iid_calibration_launch_plan(config, allowed_root=dust)

    assert validate_v5_k1_iid_calibration_launch_plan(plan) == plan
    assert plan["array"] == {
        "array_spec": "0-59%60",
        "task_count": 60,
        "expected_sample_count": 160_020,
    }
    assert len(plan["tasks"]) == 60
    assert not config.run_root.exists()
    assert set(
        replay_v5_k1_iid_calibration_launch_inputs(plan, allowed_root=dust)
    ) == {
        "source_bundle_sha256",
        "source_archive_sha256",
        "three_way_receipt_file_sha256",
        "three_way_receipt_sha256",
        "phase_c_exclusion_claim_sha256",
        "calibration_population_plan_sha256",
    }


def test_iid_calibration_worker_dry_run_and_sealed_completion(
    calibration_launch_fixture,
    monkeypatch,
):
    dust, config, _ = calibration_launch_fixture
    plan, target = _publish_plan(dust, config)
    dry = run_v5_k1_iid_calibration_task(
        target,
        0,
        expected_plan_sha256=plan["plan_sha256"],
        dry_run=True,
        allowed_root=dust,
    )
    assert dry["writes_performed"] is False

    task = plan["tasks"][0]
    payload = {
        "manifest": {
            "manifest_sha256": _digest("manifest"),
        },
        "artifact_self_sha256": _digest("artifact-self"),
    }

    def fake_build(**kwargs):
        return payload

    def fake_write(output, shard, *, allowed_root):
        path = Path(output)
        path.write_text(json.dumps(shard), encoding="utf-8")
        path.chmod(0o400)
        return path

    monkeypatch.setattr(
        k1_iid_calibration_worker_v5,
        "build_v5_k1_iid_calibration_shard",
        fake_build,
    )
    monkeypatch.setattr(
        k1_iid_calibration_worker_v5,
        "write_v5_k1_iid_calibration_shard",
        fake_write,
    )
    monkeypatch.setattr(
        k1_iid_calibration_worker_v5,
        "validate_v5_k1_iid_calibration_shard",
        lambda *args, **kwargs: payload,
    )
    result = run_v5_k1_iid_calibration_task(
        target,
        0,
        expected_plan_sha256=plan["plan_sha256"],
        allowed_root=dust,
        hostname="max-wn001",
        environment={"SLURM_JOB_ID": "101", "SLURM_ARRAY_TASK_ID": "0"},
    )
    completion = Path(task["completion"])
    assert result["status"] == "PASS"
    assert stat.S_IMODE(completion.stat().st_mode) == 0o400
    assert completion.stat().st_mtime_ns >= Path(task["output"]).stat().st_mtime_ns


def test_four_way_receipt_proves_actual_calibration_exclusion(
    calibration_launch_fixture,
):
    dust, config, three_way = calibration_launch_fixture
    plan = build_v5_k1_iid_calibration_launch_plan(config, allowed_root=dust)
    receipt = _build_four_way_receipt(
        plan=plan,
        three_way=three_way,
        recipe_sha256s=(_digest("cal-recipe"),),
        clean_group_ids=(_digest("cal-group"),),
        sample_ids=(_digest("cal-sample"),),
        branch_counts={"shape=sphere|pattern=None|resolution=off": 1},
        stratum_counts={0: 1},
        artifact_identities=(),
    )
    core = dict(receipt)
    supplied = core.pop("receipt_sha256")
    assert supplied == sha256(canonical_json(core).encode()).hexdigest()
    assert receipt["all_four_populations_recipe_disjoint"] is True
    assert receipt["all_four_populations_clean_group_disjoint"] is True
    assert all(
        count == 0
        for pair in receipt["intersection_counts"].values()
        for count in pair.values()
    )


@pytest.mark.parametrize("dependency_value", [
    "afterok:801(unfulfilled)", "afterok:801_*(unfulfilled)",
    "afterany:801(unfulfilled)", "afterok:8010(unfulfilled)",
    "afterok:801_0(unfulfilled)", "afterok:801,afterany:999",
])
def test_iid_calibration_launcher_holds_and_releases_reverse(
    calibration_launch_fixture,
    dependency_value,
):
    dust, config, _ = calibration_launch_fixture
    dry = launch_v5_k1_iid_calibration_dag(config, allowed_root=dust)
    assert dry["status"] == "dry_run"
    assert not config.run_root.exists()
    released: set[str] = set()
    calls: list[tuple[str, ...]] = []

    def runner(argv):
        command = tuple(argv)
        calls.append(command)
        if command[0] == "sbatch":
            job_id = "801" if any(value.startswith("--array=") for value in command) else "802"
            return V5CommandResult(0, job_id + "\n", "")
        if command[:4] == ("scontrol", "show", "job", "--oneliner"):
            job_id = command[4]
            dependency = f" Dependency={dependency_value}" if job_id == "802" else ""
            if job_id not in released:
                body = f"JobId={job_id} JobState=PENDING Reason=JobHeldUser{dependency}"
            elif job_id == "802":
                body = f"JobId={job_id} JobState=PENDING Reason=Dependency{dependency}"
            else:
                body = f"JobId={job_id} JobState=PENDING Reason=Resources"
            return V5CommandResult(0, body + "\n", "")
        if command[:2] == ("scontrol", "release"):
            released.add(command[2])
            return V5CommandResult(0, "", "")
        raise AssertionError(command)

    arguments = dict(
        config=config,
        submit=True,
        runner=runner,
        allowed_root=dust,
        hostname="max-fs-display006",
    )
    if dependency_value not in {
        "afterok:801(unfulfilled)", "afterok:801_*(unfulfilled)"
    }:
        with pytest.raises(RuntimeError):
            launch_v5_k1_iid_calibration_dag(**arguments)
        assert released == set()
        assert not any(command[0] == "scancel" for command in calls)
        return
    result = launch_v5_k1_iid_calibration_dag(
        config,
        submit=True,
        runner=runner,
        allowed_root=dust,
        hostname="max-wgs",
    )
    assert result["status"] == "ALL_JOBS_RELEASED"
    assert result["release_order"] == ["collector", "calibration_array"]
    release_calls = [value for value in calls if value[:2] == ("scontrol", "release")]
    assert [value[2] for value in release_calls] == ["802", "801"]
