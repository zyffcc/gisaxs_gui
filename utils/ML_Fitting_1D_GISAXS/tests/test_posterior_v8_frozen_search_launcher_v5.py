from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
import shutil
import stat

import numpy as np
import pytest

from utils.ML_Fitting_1D_GISAXS.PosteriorV8.calibrated_search_threshold_v5 import (
    compatibility_stratum_from_v5_observation,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.compatibility_calibration import (
    CompatibilityCalibrationSample,
    fit_compatibility_calibration,
    write_compatibility_calibration_atomic,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.contract import (
    NUM_TOPOLOGIES,
    SPHERE,
    topology_id_for,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.exact_search_schedule_v5 import (
    V5FrozenLocalSobolSchedule,
    write_v5_frozen_local_sobol_schedule,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.frozen_search_launch_plan_v5 import (
    V5FrozenSearchLaunchConfig,
    build_v5_frozen_search_launch_plan,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.frozen_search_workload_v5 import (
    point_workloads_v5,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.launch_frozen_search_v5 import (
    V5FrozenSearchLaunchError,
    V5SearchLaunchCommandResult,
    launch_v5_frozen_search_labels,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.observation_v5 import (
    build_v5_observation_data_view,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.sobol_recipe_coordinates_v5 import (
    V5_SOBOL_RECIPE_COORDINATE_INDEX,
    V5_SOBOL_RECIPE_DIM,
    v5_sobol_recipe_design,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.sobol_design_v5 import V5DesignPoint
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.split_design_v5 import (
    V5SplitCounts,
    V5SplitPlan,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.synthetic_recipe_v5 import (
    sample_v5_clean_recipe,
)


def _copy_source(source_root: Path, target: Path) -> None:
    posterior = Path("utils/ML_Fitting_1D_GISAXS/PosteriorV8")
    domain = Path("src/gimap/features/fitting/domain")
    (target / posterior.parent).mkdir(parents=True)
    (target / domain.parent).mkdir(parents=True)
    shutil.copytree(source_root / posterior, target / posterior)
    shutil.copytree(source_root / domain, target / domain)
    # A real versioned Maxwell source snapshot is intentionally read-only.
    # This fixture is a mutable throwaway copy because the drift test below
    # must alter one copied file after the first submission check.
    for copied_root in (target / posterior, target / domain):
        for copied_path in (copied_root, *copied_root.rglob("*")):
            copied_path.chmod(copied_path.stat().st_mode | stat.S_IWUSR)


def _fixture(tmp_path):
    allowed = tmp_path / "dust"
    allowed.mkdir(parents=True)
    source_root = allowed / "versioned-source"
    repository_root = Path(__file__).parents[3]
    _copy_source(repository_root, source_root)
    contracts = allowed / "contracts"
    contracts.mkdir()
    split_plan = V5SplitPlan.create(
        V5SplitCounts(
            train=64,
            tuning_validation=8,
            calibration=1,
            test=1,
            reference=1,
            ood_topology=1,
            ood_range_width=1,
            ood_weak_component=1,
            ood_acquisition_policy=1,
        ),
        guard_band=2,
    )
    design = v5_sobol_recipe_design(scramble_seed=20260903)
    split_path = contracts / "split-plan.json"
    design_path = contracts / "sobol-design.json"
    schedule_path = contracts / "local-search-sobol.gvd5"
    split_path.write_text(split_plan.to_json(), encoding="utf-8")
    design_path.write_text(design.to_json(), encoding="utf-8")
    schedule = V5FrozenLocalSobolSchedule.generate(
        schedule_id="maxwell-k1-throughput-pilot-seeds-v1",
        point_count=2,
        base_seed=17,
    )
    write_v5_frozen_local_sobol_schedule(schedule, schedule_path)
    config = V5FrozenSearchLaunchConfig(
        source_root=source_root,
        run_root=allowed / "runs" / "pilot-001",
        split_plan=split_path,
        sobol_design=design_path,
        local_sobol_schedule=schedule_path,
        # Final 168D design: offset 56 is naturally K1 Vertical Cylinder with measured sigma.
        train_start=56,
        train_recipes=1,
        validation_start=5,
        validation_recipes=1,
        recipes_per_shard=1,
        view_indices=(0,),
        topology_schedule_id="k1-three-shape-throughput-pilot-v1",
        selected_topology_ids=(0, 1, 2),
        optimizer_schedule_id="one-call-per-seed-pilot-v1",
        direct_scout_seed_count=1,
        per_seed_forward_evaluation_limit=1,
        protocol_id="label-pipeline-throughput-pilot-v1",
        standardized_threshold_name="precalibration_standardized_pilot_gate",
        standardized_threshold_value=3.0,
        raw_threshold_name="precalibration_raw_pilot_gate",
        raw_threshold_value=0.25,
        threshold_source_id="engineering_pilot_only_thresholds_not_paper_frozen",
        compatibility_calibration=None,
        pilot_throughput_source_id="local-k1-smoke-3.4-seconds-for-6-calls",
        pilot_effective_seconds_per_exact_forward_call=3.4 / 6.0,
        runtime_safety_factor=10.0,
    )
    return allowed, source_root, config


def _calibration(path: Path) -> Path:
    recipe = sample_v5_clean_recipe(("sphere",), recipe_seed=77)
    view = next(
        value
        for value in (
            build_v5_observation_data_view(recipe, index, split_id="calibration")
            for index in range(8)
        )
        if value.acceptance_sigma_log is not None
    )
    stratum = compatibility_stratum_from_v5_observation(view)
    samples = tuple(
        CompatibilityCalibrationSample(
            sample_id=f"calibration-{index:03d}",
            independent_group_id=f"recipe-{index:03d}",
            stratum=stratum,
            score=float(index + 1),
            effective_valid_point_count=view.effective_valid_point_count,
            acquisition_policy_id=view.acquisition_policy_id,
            measurement_sigma_available=True,
        )
        for index in range(9)
    )
    artifact = fit_compatibility_calibration(
        samples,
        dataset_manifest_sha256="1" * 64,
        calibration_split_sha256="2" * 64,
        target_coverage=0.8,
        minimum_samples_per_stratum=5,
    )
    write_compatibility_calibration_atomic(path, artifact)
    return path


def test_launch_plan_binds_schedule_source_and_emits_nonautomatic_afterok_handoff(
    tmp_path,
):
    allowed, _, config = _fixture(tmp_path)
    plan = build_v5_frozen_search_launch_plan(config, allowed_root=allowed)

    assert plan["arrays"]["train"]["array_spec"] == "0-0"
    assert plan["arrays"]["tuning_validation"]["array_spec"] == "0-0"
    schedule = plan["contracts"]["local_sobol_schedule"]
    assert schedule["path"] == str(config.local_sobol_schedule)
    assert len(schedule["schedule_sha256"]) == 64
    assert len(schedule["artifact_sha256"]) == 64
    assert len(schedule["manifest_sha256"]) == 64
    assert plan["afterok_handoff"]["automatic_downstream_submission"] is False
    assert plan["afterok_handoff"]["full_trainer_eligible"] is False
    assert plan["afterok_handoff"]["dependency_template"] == (
        "afterok:{train_job_id}:{validation_job_id}"
    )
    assert all(
        str(config.run_root) in path
        for path in plan["afterok_handoff"]["train_sidecar_paths"]
    )
    gate = plan["workload_gate"]
    assert gate["exact_forward_calls_per_branch"] == 2
    assert gate["all_shards_below_timeout_ceiling"] is True
    train_workload = plan["arrays"]["train"]["windows"][0]["workload"]
    assert train_workload["observation_query_count"] == 1
    assert train_workload["branch_count_total"] == 3
    assert train_workload["exact_forward_calls_total"] == 6
    assert plan["failure_recovery"]["scale_out_permitted"] is False
    assert plan["label_contract"] == {
        "protocol_tier": "engineering_pilot",
        "label_purpose": "engineering_throughput_pilot_not_training_eligible",
        "formal_calibration_contract_smoke_only": False,
        "full_training_eligible": False,
        "paper_scale_training_labels_claimed": False,
        "shard_plan_remains_pipeline_pilot_contract": True,
    }


def test_point_workload_counts_complete_query_slot_equivalence():
    coordinates = np.zeros(V5_SOBOL_RECIPE_DIM, dtype=np.float64)
    topology_id = topology_id_for((SPHERE, SPHERE))
    coordinates[V5_SOBOL_RECIPE_COORDINATE_INDEX["discrete.topology"]] = (
        topology_id + 0.5
    ) / NUM_TOPOLOGIES
    for slot in (1, 2):
        coordinates[
            V5_SOBOL_RECIPE_COORDINATE_INDEX[f"geometry.slot_{slot}.D_policy"]
        ] = 0.5
    point = V5DesignPoint(
        sobol_index=41,
        assigned_split="train",
        ood_label=None,
        clean_group_id=sha256(b"complete-slot-workload").hexdigest(),
        unit_coordinates=tuple(coordinates),
    )
    rows = point_workloads_v5(
        (point,),
        design=v5_sobol_recipe_design(scramble_seed=20260903),
        selected_topology_ids=(topology_id,),
        view_indices=(0,),
        exact_forward_calls_per_branch=2,
    )

    assert rows[0]["branch_count_total"] == 3
    assert rows[0]["exact_forward_calls_total"] == 6


def test_launcher_is_dry_run_by_default_and_submit_only_calls_two_arrays(tmp_path):
    allowed, _, config = _fixture(tmp_path)
    preview = launch_v5_frozen_search_labels(config, allowed_root=allowed)
    assert preview["status"] == "dry_run"
    assert preview["writes_performed"] is False
    assert not config.run_root.exists()
    joined = "\n".join(
        " ".join(value) for value in preview["submission_preview"].values()
    )
    expected_source = preview["plan"]["source"]["bundle_sha256"]
    expected_launch_plan = preview["plan"]["plan_sha256"]
    assert f"POSTERIOR_V8_V5_EXPECTED_SOURCE_BUNDLE_SHA256={expected_source}" in joined
    assert f"POSTERIOR_V8_V5_LAUNCH_PLAN_SHA256={expected_launch_plan}" in joined
    assert "POSTERIOR_V8_V5_EXPECTED_SPLIT_PLAN_FILE_SHA256=" in joined
    assert "POSTERIOR_V8_V5_EXPECTED_SPLIT_PLAN_SHA256=" in joined
    assert "POSTERIOR_V8_V5_EXPECTED_SOBOL_DESIGN_FILE_SHA256=" in joined
    assert "POSTERIOR_V8_V5_EXPECTED_SOBOL_DESIGN_SHA256=" in joined
    assert "POSTERIOR_V8_V5_EXPECTED_LOCAL_SCHEDULE_ARTIFACT_SHA256=" in joined
    assert "POSTERIOR_V8_V5_EXPECTED_LOCAL_SCHEDULE_MANIFEST_SHA256=" in joined
    assert "POSTERIOR_V8_V5_EXPECTED_LOCAL_SCHEDULE_SHA256=" in joined
    assert str(config.local_sobol_schedule) in joined
    assert all(
        "--hold" in command for command in preview["submission_preview"].values()
    )

    calls = []

    def runner(argv):
        calls.append(tuple(argv))
        return V5SearchLaunchCommandResult(0, f"{9100 + len(calls)}\n", "")

    manifest = launch_v5_frozen_search_labels(
        config,
        submit=True,
        runner=runner,
        allowed_root=allowed,
        hostname="max-wgs.desy.de",
    )
    assert manifest["status"] == "submitted"
    assert manifest["job_ids"] == {"train": "9101", "tuning_validation": "9102"}
    assert manifest["afterok_handoff"]["dependency"] == "afterok:9101:9102"
    assert manifest["downstream_training_submitted"] is False
    assert manifest["arrays_released"] is True
    assert len(calls) == 3
    assert all(call[0:2] == ("sbatch", "--parsable") for call in calls[:2])
    assert all("--hold" in call for call in calls[:2])
    assert calls[2] == ("scontrol", "release", "9101", "9102")
    stored = json.loads(
        (config.run_root / "frozen-search-launch-manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert stored == manifest


def test_source_drift_between_submission_checks_preserves_failed_launch_audit(tmp_path):
    allowed, source_root, config = _fixture(tmp_path)
    calls = []

    def runner(argv):
        calls.append(tuple(argv))
        changed = (
            source_root
            / "utils/ML_Fitting_1D_GISAXS/PosteriorV8/freeze_exact_search_schedule_v5.py"
        )
        changed.write_text(changed.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        return V5SearchLaunchCommandResult(0, "9201\n", "")

    with pytest.raises(V5FrozenSearchLaunchError) as caught:
        launch_v5_frozen_search_labels(
            config,
            submit=True,
            runner=runner,
            allowed_root=allowed,
            hostname="max-wgs",
        )
    assert len(calls) == 2
    assert calls[0][0:3] == ("sbatch", "--parsable", "--hold")
    assert calls[1] == ("scancel", "9201")
    stored = json.loads(caught.value.manifest_path.read_text(encoding="utf-8"))
    assert stored["status"] == "failed"
    assert stored["failure"]["stage"] == "tuning_validation"
    assert "changed during launch" in stored["failure"]["message"]
    assert stored["job_ids"] == {"train": "9201"}
    assert stored["cancellation_attempted"] is True
    assert stored["control_attempts"][0]["stage"] == (
        "cancel_submitted_held_arrays"
    )


def test_formal_calibration_is_hash_bound_and_drift_cancels_held_array(tmp_path):
    allowed, _, pilot = _fixture(tmp_path)
    calibration = _calibration(allowed / "contracts" / "calibration.json")
    config = V5FrozenSearchLaunchConfig(
        **{
            **pilot.__dict__,
            "standardized_threshold_name": None,
            "standardized_threshold_value": None,
            "raw_threshold_name": None,
            "raw_threshold_value": None,
            "threshold_source_id": None,
            "compatibility_calibration": calibration,
        }
    )
    preview = launch_v5_frozen_search_labels(config, allowed_root=allowed)
    contract = preview["plan"]["contracts"]["compatibility_calibration"]
    assert contract["path"] == str(calibration)
    assert len(contract["identity"]["artifact_sha256"]) == 64
    joined = "\n".join(
        " ".join(value) for value in preview["submission_preview"].values()
    )
    assert "POSTERIOR_V8_V5_PROTOCOL_TIER=paper_full_calibrated" in joined
    assert "POSTERIOR_V8_V5_EXPECTED_CALIBRATION_SHA256=" in joined
    assert "POSTERIOR_V8_V5_EXPECTED_CALIBRATION_FILE_SHA256=" in joined
    assert "POSTERIOR_V8_V5_STANDARDIZED_THRESHOLD_VALUE=" not in joined
    assert preview["plan"]["label_contract"]["label_purpose"] == (
        "formal_calibrated_contract_smoke_not_training_eligible"
    )
    assert preview["plan"]["label_contract"]["formal_calibration_contract_smoke_only"]
    assert preview["plan"]["label_contract"]["full_training_eligible"] is False
    assert "contract_smoke_only_not_training_eligible" in (
        preview["plan"]["scientific_scope"]
    )

    calls = []

    def runner(argv):
        calls.append(tuple(argv))
        if argv[0] == "sbatch":
            calibration.write_bytes(calibration.read_bytes() + b"\n")
            return V5SearchLaunchCommandResult(0, "9251\n", "")
        return V5SearchLaunchCommandResult(0, "", "")

    with pytest.raises(V5FrozenSearchLaunchError) as caught:
        launch_v5_frozen_search_labels(
            config,
            submit=True,
            runner=runner,
            allowed_root=allowed,
            hostname="max-wgs",
        )
    stored = json.loads(caught.value.manifest_path.read_text(encoding="utf-8"))
    assert calls[-1] == ("scancel", "9251")
    assert stored["failure"]["stage"] == "tuning_validation"
    assert stored["cancellation_attempted"] is True


def test_second_submission_or_release_failure_cancels_every_held_job(tmp_path):
    allowed, _, config = _fixture(tmp_path)
    calls = []

    def second_fails(argv):
        calls.append(tuple(argv))
        if len(calls) == 1:
            return V5SearchLaunchCommandResult(0, "9301\n", "")
        if argv[0] == "sbatch":
            return V5SearchLaunchCommandResult(1, "", "validation rejected")
        return V5SearchLaunchCommandResult(0, "", "")

    with pytest.raises(V5FrozenSearchLaunchError) as caught:
        launch_v5_frozen_search_labels(
            config,
            submit=True,
            runner=second_fails,
            allowed_root=allowed,
            hostname="max-wgs",
        )
    stored = json.loads(caught.value.manifest_path.read_text(encoding="utf-8"))
    assert calls[-1] == ("scancel", "9301")
    assert stored["failure"]["stage"] == "tuning_validation"
    assert stored["arrays_released"] is False
    assert stored["cancellation_attempted"] is True

    allowed, _, config = _fixture(tmp_path / "release")
    calls = []

    def release_fails(argv):
        calls.append(tuple(argv))
        if argv[0] == "sbatch":
            return V5SearchLaunchCommandResult(0, f"{9400 + len(calls)}\n", "")
        if argv[0] == "scontrol":
            return V5SearchLaunchCommandResult(1, "", "release rejected")
        return V5SearchLaunchCommandResult(0, "", "")

    with pytest.raises(V5FrozenSearchLaunchError) as caught:
        launch_v5_frozen_search_labels(
            config,
            submit=True,
            runner=release_fails,
            allowed_root=allowed,
            hostname="max-wgs",
        )
    stored = json.loads(caught.value.manifest_path.read_text(encoding="utf-8"))
    assert calls[-1] == ("scancel", "9401", "9402")
    assert stored["failure"]["stage"] == "release_held_arrays"
    assert stored["arrays_released"] is False
    assert stored["cancellation_attempted"] is True


def test_launch_plan_rejects_non_k1_scale_out_and_predicted_timeout(tmp_path):
    allowed, _, config = _fixture(tmp_path)
    with pytest.raises(ValueError, match="complete K1 topology set"):
        build_v5_frozen_search_launch_plan(
            V5FrozenSearchLaunchConfig(
                **{**config.__dict__, "selected_topology_ids": (0,)}
            ),
            allowed_root=allowed,
        )
    with pytest.raises(ValueError, match="one recipe per shard"):
        build_v5_frozen_search_launch_plan(
            V5FrozenSearchLaunchConfig(
                **{**config.__dict__, "recipes_per_shard": 2}
            ),
            allowed_root=allowed,
        )
    with pytest.raises(ValueError, match="20-hour safety ceiling"):
        build_v5_frozen_search_launch_plan(
            V5FrozenSearchLaunchConfig(
                **{
                    **config.__dict__,
                    "pilot_effective_seconds_per_exact_forward_call": 20_000.0,
                }
            ),
            allowed_root=allowed,
        )


def test_launcher_rejects_non_dust_paths_and_wrapper_forbids_login_compute(tmp_path):
    allowed, _, config = _fixture(tmp_path)
    outside = V5FrozenSearchLaunchConfig(
        **{**config.__dict__, "run_root": tmp_path / "outside"}
    )
    with pytest.raises(ValueError, match="below"):
        build_v5_frozen_search_launch_plan(outside, allowed_root=allowed)
    with pytest.raises(RuntimeError, match="max-wgs"):
        launch_v5_frozen_search_labels(
            config,
            submit=True,
            allowed_root=allowed,
            hostname="workstation",
        )
    assert not config.run_root.exists()

    wrapper = (
        Path(__file__).parents[1]
        / "PosteriorV8/slurm/v5_frozen_search_labels_cpu.sbatch"
    ).read_text(encoding="utf-8")
    assert "SLURM_JOB_ID" in wrapper
    assert "max-wgs*" in wrapper
    assert "/data/dust/user/zhaiyufe/*" in wrapper
    assert "EXPECTED_SOURCE_BUNDLE_SHA256" in wrapper
    assert "POSTERIOR_V8_V5_LAUNCH_PLAN_SHA256" in wrapper
    assert "POSTERIOR_V8_V5_PROTOCOL_TIER" in wrapper
    assert "EXPECTED_CALIBRATION_SHA256" in wrapper
    assert "EXPECTED_CALIBRATION_FILE_SHA256" in wrapper
    assert "run_frozen_search_pipeline_v5" in wrapper
    assert "--execute" in wrapper
    assert "train_grouped" not in wrapper
