from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from utils.ML_Fitting_1D_GISAXS.PosteriorV8.calibrated_search_threshold_v5 import (
    compatibility_stratum_from_v5_observation,
    inspect_v5_compatibility_calibration,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.compatibility_calibration import (
    CompatibilityCalibrationSample,
    fit_compatibility_calibration,
    write_compatibility_calibration_atomic,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.exact_search_schedule_v5 import (
    write_v5_frozen_local_sobol_schedule,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.formal_production_search_contract_v5 import (
    V5FormalProductionSourceIdentity,
    V5_FORMAL_PRODUCTION_STAGE_IDS,
    plan_v5_formal_production_search_shard,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.formal_label_observation_policy_v5 import (
    select_v5_formal_label_observation,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.formal_production_search_plan_v5 import (
    build_v5_formal_production_search_plan,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.formal_production_search_worker_contract_v5 import (
    prepare_v5_formal_production_worker_shard,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.frozen_search_launch_contracts_v5 import (
    fingerprint_v5_frozen_search_source,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.frozen_search_pipeline_v5 import (
    _expected_parent_dataset,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.grouped_artifact_v5 import canonical_json
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.grouped_dataset_v5 import (
    observation_array,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.launch_formal_production_search_smoke_v5 import (
    V5FormalProductionSmokeLaunchConfig,
    V5FormalProductionSmokeLaunchError,
    V5_FORMAL_PRODUCTION_PLAN_FILENAME,
    launch_v5_formal_production_contract_smoke,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.launch_frozen_search_v5 import (
    V5SearchLaunchCommandResult,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.observation_v5 import (
    build_v5_observation_data_view,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.sobol_design_v5 import (
    materialize_v5_design_points_for_indices,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.sobol_recipe_coordinates_v5 import (
    v5_sobol_recipe_design,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.sobol_recipe_v5 import (
    materialize_v5_sobol_clean_recipe,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.split_design_v5 import (
    V5SplitCounts,
    V5SplitPlan,
)
from utils.ML_Fitting_1D_GISAXS.tests.test_posterior_v8_formal_production_search_plan_v5 import (
    _stage,
    _stage_indices,
)


def _fixture(tmp_path: Path) -> SimpleNamespace:
    repository_root = Path(__file__).parents[3]
    source_root = tmp_path / "versioned-source"
    posterior = Path("utils/ML_Fitting_1D_GISAXS/PosteriorV8")
    domain = Path("src/gimap/features/fitting/domain")
    (source_root / posterior.parent).mkdir(parents=True)
    (source_root / domain.parent).mkdir(parents=True)
    shutil.copytree(repository_root / posterior, source_root / posterior)
    shutil.copytree(repository_root / domain, source_root / domain)
    contracts = tmp_path / "contracts"
    contracts.mkdir()
    split_plan = V5SplitPlan.create(
        V5SplitCounts(
            train=256,
            tuning_validation=256,
            calibration=8,
            test=8,
            reference=8,
            ood_topology=2,
            ood_range_width=2,
            ood_weak_component=2,
            ood_acquisition_policy=2,
        ),
        guard_band=4,
    )
    design = v5_sobol_recipe_design(scramble_seed=20260903)
    by_split = {
        split: _stage_indices(split_plan, design, split)
        for split in ("train", "tuning_validation")
    }
    selected_indices = tuple(
        by_split[split][stage_id]
        for stage_id in V5_FORMAL_PRODUCTION_STAGE_IDS
        for split in ("train", "tuning_validation")
    )
    calibration_samples = []
    for point in materialize_v5_design_points_for_indices(
        split_plan, design, selected_indices
    ):
        recipe = materialize_v5_sobol_clean_recipe(point, design)
        selection = select_v5_formal_label_observation(point.sobol_index, (0, 1))
        view = build_v5_observation_data_view(
            recipe, selection.selected_view_index, split_id="calibration"
        )
        stratum = compatibility_stratum_from_v5_observation(view)
        for repeat in range(5):
            calibration_samples.append(
                CompatibilityCalibrationSample(
                    sample_id=f"{point.sobol_index}-{repeat}",
                    independent_group_id=f"{point.sobol_index}-{repeat}",
                    stratum=stratum,
                    score=float(repeat + 1),
                    effective_valid_point_count=view.effective_valid_point_count,
                    acquisition_policy_id=view.acquisition_policy_id,
                    measurement_sigma_available=True,
                )
            )
    calibration_artifact = fit_compatibility_calibration(
        calibration_samples,
        dataset_manifest_sha256="1" * 64,
        calibration_split_sha256="2" * 64,
        target_coverage=0.8,
        minimum_samples_per_stratum=5,
    )
    calibration_path = contracts / "calibration.json"
    write_compatibility_calibration_atomic(calibration_path, calibration_artifact)
    calibration = inspect_v5_compatibility_calibration(calibration_path)
    split_path = contracts / "split.json"
    design_path = contracts / "design.json"
    split_path.write_text(split_plan.to_json(), encoding="utf-8")
    design_path.write_text(design.to_json(), encoding="utf-8")
    stages = tuple(
        _stage(stage_id, 2, calibration.identity)
        for stage_id in V5_FORMAL_PRODUCTION_STAGE_IDS
    )
    schedule_paths = {}
    for stage in stages:
        path = contracts / f"{stage.stage_id.lower()}-schedule.gvd5"
        write_v5_frozen_local_sobol_schedule(stage.seed_schedule, path)
        schedule_paths[stage.stage_id] = path
    shards = tuple(
        plan_v5_formal_production_search_shard(
            split_plan=split_plan,
            sobol_design=design,
            stage=stage,
            target_split=split,
            sobol_indices=(by_split[split][stage.stage_id],),
            output_relative_path=(
                f"labels/{stage.stage_id.lower()}/{split}/shard-000000"
            ),
        )
        for stage in stages
        for split in ("train", "tuning_validation")
    )
    source = V5FormalProductionSourceIdentity.from_fingerprint(
        fingerprint_v5_frozen_search_source(source_root)
    )
    plan = build_v5_formal_production_search_plan(
        study_id="formal-runtime-focused-test",
        source=source,
        split_plan=split_plan,
        sobol_design=design,
        calibration=calibration,
        candidate_view_indices=(0, 1),
        stages=stages,
        shards=shards,
    )
    run_root = tmp_path / "run"
    run_root.mkdir()
    plan_path = run_root / V5_FORMAL_PRODUCTION_PLAN_FILENAME
    plan_path.write_text(plan.to_json(), encoding="utf-8")
    return SimpleNamespace(
        source_root=source_root,
        calibration_path=calibration_path,
        split_path=split_path,
        design_path=design_path,
        schedule_paths=schedule_paths,
        plan=plan,
        plan_path=plan_path,
        run_root=run_root,
    )


def _prepare(values: SimpleNamespace, shard=None, **overrides):
    selected = values.plan.shards[0] if shard is None else shard
    arguments = {
        "global_plan_path": values.plan_path,
        "expected_global_plan_sha256": values.plan.sha256,
        "expected_shard_plan_sha256": selected.sha256,
        "stage_id": selected.stage.stage_id,
        "target_split": selected.target_split,
        "array_task_id": 0,
        "source_root": values.source_root,
        "run_root": values.run_root,
        "split_plan_path": values.split_path,
        "sobol_design_path": values.design_path,
        "local_sobol_schedule_path": values.schedule_paths[selected.stage.stage_id],
        "calibration_path": values.calibration_path,
    }
    arguments.update(overrides)
    return prepare_v5_formal_production_worker_shard(**arguments)


def test_worker_replays_unique_shard_and_materializes_only_selected_view(tmp_path):
    values = _fixture(tmp_path)
    prepared = _prepare(values)
    member = values.plan.shards[0].recipes[0]

    assert prepared.runtime_shard.view_indices_for_recipe(0) == (
        member.observation_selection.selected_view_index,
    )
    assert prepared.output_root == (
        values.run_root / values.plan.shards[0].output_relative_path
    )
    parent, _ = _expected_parent_dataset(prepared.runtime_shard)
    assert parent.observation_count == 1
    assert int(parent.arrays[observation_array("view_index")][0]) == (
        member.observation_selection.selected_view_index
    )


def test_worker_rejects_task_mismatch_and_selected_view_drift(tmp_path):
    values = _fixture(tmp_path)
    other = values.plan.shards[1]
    with pytest.raises(ValueError, match="task/shard"):
        _prepare(values, expected_shard_plan_sha256=other.sha256)

    payload = values.plan.to_payload()
    recipe = payload["shards"][0]["recipes"][0]
    recipe["selected_view_index"] = 99
    core = {key: value for key, value in payload.items() if key != "plan_sha256"}
    payload["plan_sha256"] = sha256(canonical_json(core).encode()).hexdigest()
    values.plan_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="selected singleton view"):
        _prepare(
            values,
            expected_global_plan_sha256=payload["plan_sha256"],
        )


def test_worker_rejects_output_escape_reuse_and_source_drift(tmp_path, monkeypatch):
    values = _fixture(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (values.run_root / "labels").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="escaped"):
        _prepare(values)

    (values.run_root / "labels").unlink()
    output = values.run_root / values.plan.shards[0].output_relative_path
    output.mkdir(parents=True)
    with pytest.raises(FileExistsError, match="reuse"):
        _prepare(values)

    fingerprint = fingerprint_v5_frozen_search_source(values.source_root)
    fingerprint["bundle_sha256"] = "0" * 64
    monkeypatch.setattr(
        "utils.ML_Fitting_1D_GISAXS.PosteriorV8."
        "formal_production_search_worker_contract_v5."
        "fingerprint_v5_frozen_search_source",
        lambda _: fingerprint,
    )
    with pytest.raises(RuntimeError, match="source bundle changed"):
        _prepare(values)


def _launch_config(values: SimpleNamespace, run_root: Path):
    return V5FormalProductionSmokeLaunchConfig.from_plan(
        values.plan,
        source_root=values.source_root,
        run_root=run_root,
        split_plan=values.split_path,
        sobol_design=values.design_path,
        calibration_artifact=values.calibration_path,
        local_sobol_schedules=values.schedule_paths,
    )


def test_launcher_writes_plan_before_sbatch_and_cancels_partial_submit(
    tmp_path, monkeypatch
):
    values = _fixture(tmp_path)
    launch_root = tmp_path / "launch-partial"
    config = _launch_config(values, launch_root)
    monkeypatch.setattr(
        "utils.ML_Fitting_1D_GISAXS.PosteriorV8."
        "launch_formal_production_search_smoke_v5._preflight_stage_contracts",
        lambda *_: None,
    )
    monkeypatch.setattr(
        "utils.ML_Fitting_1D_GISAXS.PosteriorV8."
        "launch_formal_production_search_smoke_v5._verify_planned_source",
        lambda *_: values.plan.source.bundle_sha256,
    )
    calls = []

    def runner(argv):
        calls.append(tuple(argv))
        if argv[0] == "sbatch":
            assert (launch_root / V5_FORMAL_PRODUCTION_PLAN_FILENAME).is_file()
            if sum(call[0] == "sbatch" for call in calls) == 1:
                return V5SearchLaunchCommandResult(0, "8101\n", "")
            return V5SearchLaunchCommandResult(1, "", "rejected")
        return V5SearchLaunchCommandResult(0, "", "")

    with pytest.raises(V5FormalProductionSmokeLaunchError) as caught:
        launch_v5_formal_production_contract_smoke(
            config,
            submit=True,
            runner=runner,
            allowed_root=Path("/"),
            hostname="max-wgs",
        )
    assert calls[-1] == ("scancel", "8101")
    receipt = json.loads(caught.value.receipt_path.read_text(encoding="utf-8"))
    assert receipt["arrays_released_together"] is False
    assert receipt["training_submitted"] is False


def test_launcher_default_dry_run_and_atomic_six_array_release(tmp_path, monkeypatch):
    values = _fixture(tmp_path)
    launch_root = tmp_path / "launch-success"
    config = _launch_config(values, launch_root)
    monkeypatch.setattr(
        "utils.ML_Fitting_1D_GISAXS.PosteriorV8."
        "launch_formal_production_search_smoke_v5._preflight_stage_contracts",
        lambda *_: None,
    )
    monkeypatch.setattr(
        "utils.ML_Fitting_1D_GISAXS.PosteriorV8."
        "launch_formal_production_search_smoke_v5._verify_planned_source",
        lambda *_: values.plan.source.bundle_sha256,
    )
    preview = launch_v5_formal_production_contract_smoke(
        config, allowed_root=Path("/")
    )
    assert preview["status"] == "dry_run"
    assert preview["writes_performed"] is False
    assert len(preview["submission_preview"]) == 6
    assert not launch_root.exists()

    calls = []

    def runner(argv):
        calls.append(tuple(argv))
        if argv[0] == "sbatch":
            return V5SearchLaunchCommandResult(0, f"{8200 + len(calls)}\n", "")
        return V5SearchLaunchCommandResult(0, "", "")

    receipt = launch_v5_formal_production_contract_smoke(
        config,
        submit=True,
        runner=runner,
        allowed_root=Path("/"),
        hostname="max-wgs.desy.de",
    )
    assert receipt["status"] == "submitted"
    assert receipt["arrays_released_together"] is True
    assert receipt["training_submitted"] is False
    assert len([value for value in calls if value[0] == "sbatch"]) == 6
    assert calls[-1][:2] == ("scontrol", "release")
    assert len(calls[-1]) == 8
