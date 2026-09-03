from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from utils.ML_Fitting_1D_GISAXS.PosteriorV8 import (
    freeze_formal_production_search_plan_v5 as freeze_module,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.calibrated_search_threshold_v5 import (
    compatibility_stratum_from_v5_observation,
    inspect_v5_compatibility_calibration,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.compatibility_calibration import (
    CompatibilityCalibrationSample,
    fit_compatibility_calibration,
    write_compatibility_calibration_atomic,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.contract import topology_from_id
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.exact_search_schedule_v5 import (
    V5FrozenLocalSobolSchedule,
    write_v5_frozen_local_sobol_schedule,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.formal_production_search_contract_v5 import (
    V5FormalProductionSourceIdentity,
    V5_FORMAL_PRODUCTION_ALLOWED_SPLITS,
    V5_FORMAL_PRODUCTION_STAGE_IDS,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.formal_production_search_worker_contract_v5 import (
    read_v5_formal_production_global_plan,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.freeze_formal_production_search_plan_v5 import (
    V5FormalContractSmokeFreezeConfig,
    build_v5_formal_contract_smoke_plan,
    freeze_v5_formal_contract_smoke_plan,
    main,
    select_v5_formal_contract_smoke_parent_indices,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.observation_v5 import (
    build_v5_observation_data_view,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.sobol_recipe_coordinates_v5 import (
    v5_sobol_recipe_design,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.split_design_v5 import (
    V5SplitCounts,
    V5SplitPlan,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.synthetic_recipe_v5 import (
    sample_v5_clean_recipe,
)


def _source_fingerprint(label: str = "a") -> dict[str, object]:
    return {
        "source_root": "/checked/source",
        "bundle_sha256": label * 64,
        "bundle_file_count": 2,
        "required_file_sha256": {
            "PosteriorV8/executor.py": "b" * 64,
            "fitting/domain/scattering_model.py": "c" * 64,
        },
    }


def _write_calibration(path: Path) -> None:
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
            sample_id=f"freeze-calibration-{index:03d}",
            independent_group_id=f"freeze-parent-{index:03d}",
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


@pytest.fixture
def frozen_inputs(tmp_path, monkeypatch):
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
    sobol_design = v5_sobol_recipe_design(scramble_seed=20260903)
    split_path = tmp_path / "split-plan.json"
    design_path = tmp_path / "sobol-design.json"
    split_path.write_text(split_plan.to_json(), encoding="utf-8")
    design_path.write_text(sobol_design.to_json(), encoding="utf-8")
    calibration_path = tmp_path / "calibration.json"
    _write_calibration(calibration_path)
    schedule_paths = {}
    schedules = {}
    for stage_id, budget in zip(V5_FORMAL_PRODUCTION_STAGE_IDS, (2, 4, 8)):
        schedule = V5FrozenLocalSobolSchedule.generate(
            schedule_id=f"{stage_id.lower()}-checked-local-sobol-v1",
            point_count=budget,
            base_seed=100 + budget,
        )
        path = tmp_path / f"{stage_id.lower()}-local-sobol.gvd5"
        write_v5_frozen_local_sobol_schedule(schedule, path)
        schedule_paths[stage_id] = path
        schedules[stage_id] = schedule
    monkeypatch.setattr(
        freeze_module,
        "fingerprint_v5_frozen_search_source",
        lambda _source_root: _source_fingerprint(),
    )
    source_root = tmp_path / "source"
    source_root.mkdir()
    return {
        "split_plan": split_plan,
        "sobol_design": sobol_design,
        "calibration": inspect_v5_compatibility_calibration(calibration_path),
        "schedules": schedules,
        "source": V5FormalProductionSourceIdentity.from_fingerprint(_source_fingerprint()),
        "source_root": source_root,
        "split_path": split_path,
        "design_path": design_path,
        "calibration_path": calibration_path,
        "schedule_paths": schedule_paths,
        "tmp_path": tmp_path,
    }


def _config(inputs, name="global-plan.json") -> V5FormalContractSmokeFreezeConfig:
    return V5FormalContractSmokeFreezeConfig(
        study_id="paper-v5-contract-smoke-20260903",
        source_root=inputs["source_root"],
        split_plan_path=inputs["split_path"],
        sobol_design_path=inputs["design_path"],
        calibration_path=inputs["calibration_path"],
        local_sobol_schedule_paths=inputs["schedule_paths"],
        output_path=inputs["tmp_path"] / name,
    )


def test_plan_replay_is_stable_and_summary_has_exact_workload(frozen_inputs) -> None:
    config = _config(frozen_inputs)
    first = freeze_v5_formal_contract_smoke_plan(config)
    second = freeze_v5_formal_contract_smoke_plan(
        replace(config, output_path=frozen_inputs["tmp_path"] / "replay-plan.json")
    )

    assert first["plan_sha256"] == second["plan_sha256"]
    assert first["plan_file_sha256"] == second["plan_file_sha256"]
    assert first["training_promotion_enabled"] is False
    assert first["totals"]["shards"] == 6
    assert first["totals"]["queries"] == 6
    assert first["totals"]["branches"] > 0
    assert first["totals"]["exact_forward_calls"] > first["totals"]["branches"]
    payload = read_v5_formal_production_global_plan(
        config.output_path, expected_plan_sha256=first["plan_sha256"]
    )
    assert payload["plan_sha256"] == first["plan_sha256"]
    assert payload["promotion_boundary"]["training_promotion_enabled_by_this_module"] is True
    assert payload["promotion_boundary"]["membership_proof_alone_authorizes_training"] is False


def test_existing_output_is_rejected_before_source_or_artifact_reads(
    frozen_inputs, monkeypatch
) -> None:
    config = _config(frozen_inputs)
    freeze_v5_formal_contract_smoke_plan(config)
    monkeypatch.setattr(
        freeze_module,
        "fingerprint_v5_frozen_search_source",
        lambda _source_root: (_ for _ in ()).throw(AssertionError("must not fingerprint")),
    )

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        freeze_v5_formal_contract_smoke_plan(config)


def test_configurable_shards_are_disjoint_and_use_new_stage_topology_bands(
    frozen_inputs,
) -> None:
    plan = build_v5_formal_contract_smoke_plan(
        study_id="paper-v5-contract-smoke-count-three",
        source=frozen_inputs["source"],
        split_plan=frozen_inputs["split_plan"],
        sobol_design=frozen_inputs["sobol_design"],
        calibration=frozen_inputs["calibration"],
        local_sobol_schedules=frozen_inputs["schedules"],
        parents_per_stage_split=3,
        parents_per_shard=2,
    )

    indices = [recipe.sobol_index for shard in plan.shards for recipe in shard.recipes]
    assert len(indices) == len(set(indices)) == 18
    assert len(plan.shards) == 12
    expected_component_counts = {"K1": {1}, "K2": {2}, "ALL34": {3, 4}}
    for shard in plan.shards:
        assert len(shard.recipes) <= 2
        for recipe in shard.recipes:
            assert (
                len(topology_from_id(recipe.generating_topology_id))
                in (expected_component_counts[shard.stage.stage_id])
            )
            assert plan.split_plan.split_for_index(recipe.sobol_index) == shard.target_split


def test_selection_is_earliest_nonoverlapping_and_stage_complete(frozen_inputs) -> None:
    selections = select_v5_formal_contract_smoke_parent_indices(
        frozen_inputs["split_plan"],
        frozen_inputs["sobol_design"],
        parents_per_stage_split=2,
        scan_chunk_size=7,
    )

    assert tuple((value.stage_id, value.target_split) for value in selections) == tuple(
        (stage_id, split)
        for stage_id in V5_FORMAL_PRODUCTION_STAGE_IDS
        for split in V5_FORMAL_PRODUCTION_ALLOWED_SPLITS
    )
    assert all(len(value.sobol_indices) == 2 for value in selections)
    flattened = [index for value in selections for index in value.sobol_indices]
    assert len(flattened) == len(set(flattened))
    assert all(tuple(sorted(value.sobol_indices)) == value.sobol_indices for value in selections)


def test_source_drift_before_publication_fails_without_plan(frozen_inputs, monkeypatch) -> None:
    calls = iter((_source_fingerprint("a"), _source_fingerprint("d")))
    monkeypatch.setattr(
        freeze_module,
        "fingerprint_v5_frozen_search_source",
        lambda _source_root: next(calls),
    )
    config = _config(frozen_inputs)

    with pytest.raises(RuntimeError, match="source bundle changed"):
        freeze_v5_formal_contract_smoke_plan(config)
    assert not config.output_path.exists()


def test_cli_freezes_default_six_query_contract_smoke(frozen_inputs, capsys) -> None:
    config = _config(frozen_inputs, "cli-plan.json")
    result = main(
        (
            "--study-id",
            config.study_id,
            "--source-root",
            str(config.source_root),
            "--split-plan",
            str(config.split_plan_path),
            "--sobol-design",
            str(config.sobol_design_path),
            "--calibration",
            str(config.calibration_path),
            "--k1-local-sobol-schedule",
            str(config.local_sobol_schedule_paths["K1"]),
            "--k2-local-sobol-schedule",
            str(config.local_sobol_schedule_paths["K2"]),
            "--all34-local-sobol-schedule",
            str(config.local_sobol_schedule_paths["ALL34"]),
            "--output-plan",
            str(config.output_path),
        )
    )
    summary = json.loads(capsys.readouterr().out)

    assert result == 0
    assert config.output_path.is_file()
    assert summary["totals"]["queries"] == 6
    assert summary["training_promotion_enabled"] is False
    assert set(summary["local_sobol_schedule_sha256"]) == set(V5_FORMAL_PRODUCTION_STAGE_IDS)
