from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from hashlib import sha256

import numpy as np
import pytest

from utils.ML_Fitting_1D_GISAXS.PosteriorV8.amplitude_query_v5 import V5AmplitudeQuery
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.bounds_query_sampling_v5 import (
    full_range_v5_query,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.bounds_query_v5 import (
    AxisRangeDesign,
    V5BoundsQuery,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.contract import (
    ClosedInterval,
    GuiComponentBounds,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8 import exact_search_executor_v5 as executor_module
from utils.ML_Fitting_1D_GISAXS.PosteriorV8 import profiled_forward as profiled_forward_module
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.exact_search_executor_v5 import (
    V5_EXACT_SEARCH_REPRESENTATIVE_DISTANCE_ID,
    V5_EXACT_SEARCH_REPRESENTATIVE_DISTANCE_SHA256,
    V5FrozenExactSearchExecutor,
    build_v5_frozen_exact_search_protocol,
    execute_v5_frozen_branch_search,
    read_v5_exact_search_executor_artifact,
    v5_exact_search_artifact_path,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.exact_search_schedule_v5 import (
    V5FrozenExactOptimizerSchedule,
    V5FrozenLocalSobolSchedule,
    read_v5_frozen_local_sobol_schedule,
    write_v5_frozen_local_sobol_schedule,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.grouped_artifact_v5 import (
    array_manifest,
    canonical_json,
    write_checked_array_artifact,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.observation_v5 import (
    build_v5_observation_data_view,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.profiled_forward import (
    POLYTOPE_FEASIBLE_FALLBACK_STATUS,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.search_supervision_contract_v5 import (
    V5ExactSearchObservation,
    V5FrozenSearchTask,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.synthetic_recipe_v5 import (
    sample_v5_clean_recipe,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.universal_query_contract_v5 import (
    V5TopologyQuery,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.universal_query_v5 import (
    build_v5_universal_candidate_context,
)


def _schedule(*, count=8, seed=9102):
    return V5FrozenLocalSobolSchedule.generate(
        schedule_id=f"paper-direct-sobol-{count}/test",
        point_count=count,
        base_seed=seed,
    )


def _optimizer(*, scouts=2, per_seed=3):
    return V5FrozenExactOptimizerSchedule(
        schedule_id=f"paper-exact-multistart-{scouts}-{per_seed}/test",
        direct_scout_seed_count=scouts,
        per_seed_forward_evaluation_limit=per_seed,
    )


def _write_semantically_forged_artifact(artifact, path, mutate_arrays):
    arrays = {name: np.array(value, copy=True) for name, value in artifact.arrays.items()}
    mutate_arrays(arrays)
    manifest = deepcopy(dict(artifact.manifest))
    manifest["arrays"] = array_manifest(arrays)
    core = dict(manifest)
    core.pop("manifest_sha256")
    manifest["manifest_sha256"] = sha256(canonical_json(core).encode("utf-8")).hexdigest()
    write_checked_array_artifact(path, manifest=manifest, arrays=arrays)


def _write_semantically_forged_manifest(artifact, path, mutate_manifest):
    arrays = {name: np.array(value, copy=True) for name, value in artifact.arrays.items()}
    manifest = deepcopy(dict(artifact.manifest))
    mutate_manifest(manifest)
    core = dict(manifest)
    core.pop("manifest_sha256")
    manifest["manifest_sha256"] = sha256(canonical_json(core).encode("utf-8")).hexdigest()
    write_checked_array_artifact(path, manifest=manifest, arrays=arrays)


def _view_with_sigma(recipe, available: bool):
    for index in range(4):
        view = build_v5_observation_data_view(recipe, index, split_id="train")
        if (view.acceptance_sigma_log is not None) is available:
            return view
    raise AssertionError("the frozen observation policy did not expose both sigma states")


def _amplitude_query(query):
    return V5AmplitudeQuery.create(
        background=ClosedInterval(0.0, 1.0e4),
        k=ClosedInterval(1.0e-2, 1.0e4),
        component_intensities=tuple(ClosedInterval(0.0, 1.0) for _ in query.topology),
        resolution_presence_policy=query.resolution_presence_policy,
        int_res=(
            None if query.resolution_presence_policy == "absent" else ClosedInterval(0.0, 10.0)
        ),
    )


def _task(
    *,
    schedule,
    optimizer,
    sigma_available,
    threshold,
    delta=0.08,
    fixed=False,
    resolution_present=False,
):
    recipe = sample_v5_clean_recipe(("sphere",), recipe_seed=78123, pattern_id=0)
    view = _view_with_sigma(recipe, sigma_available)
    if fixed:
        bounds = (
            GuiComponentBounds(
                "sphere",
                ClosedInterval(10.0, 10.0),
                ClosedInterval(1.0, 1.0),
            ),
        )
        query = V5BoundsQuery.create(
            query_seed=44,
            generation_attempt=0,
            component_bounds=bounds,
            resolution_presence_policy="absent",
            resolution_bounds=None,
            axis_designs=(
                AxisRangeDesign("component[0].R", "fixed", "interior"),
                AxisRangeDesign("component[0].sigma_R", "fixed", "interior"),
            ),
        )
        pattern_id = 0
    else:
        query = full_range_v5_query(("sphere",), query_seed=9812)
        pattern_id = 16 if resolution_present else 0
    amplitude = _amplitude_query(query)
    if resolution_present:
        amplitude = V5AmplitudeQuery.create(
            background=ClosedInterval(10.0, 10.0),
            k=ClosedInterval(100.0, 100.0),
            component_intensities=(ClosedInterval(1.0, 1.0),),
            resolution_presence_policy="optional",
            int_res=ClosedInterval(0.1, 0.1),
        )
    context = build_v5_universal_candidate_context(
        view.preprocessed,
        view.uncertainty,
        (V5TopologyQuery(query, amplitude),),
    )
    wire_key = f"topology-{query.topology_id:02d}:wire-{pattern_id:02d}"
    branch_index = context.global_branch_keys.index(wire_key)
    protocol = build_v5_frozen_exact_search_protocol(
        protocol_id="paper-frozen-exact-search/test",
        seed_schedule=schedule,
        optimizer_schedule=optimizer,
        standardized_threshold_name="standardized_exact_gate/test",
        standardized_threshold_value=threshold,
        raw_threshold_name="raw_exact_gate/test",
        raw_threshold_value=threshold,
        threshold_source_id="unit-test-frozen-threshold",
        delta_separation_threshold=delta,
    )
    exact_observation = V5ExactSearchObservation.from_observation_view(
        view,
        curve_id=f"observation-sigma-{int(sigma_available)}",
    )
    return V5FrozenSearchTask(
        query_index=0,
        clean_group_id="clean-group-test",
        recipe_id="recipe-test",
        observation_id=exact_observation.observed_curve.curve_id,
        universal_context=context,
        exact_observation=exact_observation,
        query_catalog_artifact_id="query-catalog/test",
        query_catalog_artifact_sha256=sha256(b"query-catalog/test").hexdigest(),
        branch_index=branch_index,
        protocol=protocol,
    )


def test_direct_sobol_schedule_is_byte_replayable_and_exclusive(tmp_path):
    first = _schedule(count=8, seed=101)
    second = _schedule(count=8, seed=101)
    changed = _schedule(count=8, seed=102)
    assert np.array_equal(first.points, second.points)
    assert first.sha256 == second.sha256
    assert first.sha256 != changed.sha256
    first.verify_runtime_replay()

    path = tmp_path / "schedule.gvd5"
    receipt = write_v5_frozen_local_sobol_schedule(first, path)
    loaded, loaded_receipt = read_v5_frozen_local_sobol_schedule(path)
    assert loaded.sha256 == first.sha256
    assert np.array_equal(loaded.points, first.points)
    assert loaded_receipt.artifact_sha256 == receipt.artifact_sha256
    with pytest.raises(FileExistsError, match="overwrite"):
        write_v5_frozen_local_sobol_schedule(first, path)

    corrupt = tmp_path / "corrupt.gvd5"
    content = bytearray(path.read_bytes())
    content[len(content) // 2] ^= 0x01
    corrupt.write_bytes(content)
    with pytest.raises(ValueError):
        read_v5_frozen_local_sobol_schedule(corrupt)


def test_fixed_branch_runs_full_budget_after_positive_and_preserves_sigma_metric(tmp_path):
    schedule = _schedule(count=4)
    optimizer = _optimizer(scouts=2, per_seed=2)
    task = _task(
        schedule=schedule,
        optimizer=optimizer,
        sigma_available=True,
        threshold=1.0e12,
        fixed=True,
    )
    runner = V5FrozenExactSearchExecutor(
        output_directory=tmp_path,
        seed_schedule=schedule,
        optimizer_schedule=optimizer,
    )
    result = runner(task)

    assert result.outcome == "compatible_found"
    assert result.completed
    assert result.exact_forward_calls_used == task.protocol.exact_forward_call_budget == 4
    assert len(result.representatives) == 1
    path = v5_exact_search_artifact_path(tmp_path, task, schedule, optimizer)
    artifact = read_v5_exact_search_executor_artifact(path, task=task)
    assert artifact.receipt.artifact_sha256 == result.executor_artifact_sha256
    assert artifact.manifest["positive_early_stop_used"] is False
    assert artifact.manifest["selected_metric_name"] == task.protocol.metric_name
    assert task.protocol.representative_distance_id == V5_EXACT_SEARCH_REPRESENTATIVE_DISTANCE_ID
    assert (
        task.protocol.representative_distance_sha256
        == V5_EXACT_SEARCH_REPRESENTATIVE_DISTANCE_SHA256
    )
    assert (
        artifact.manifest["representative_distance_contract"]["implementation"]
        == "query_local_parameter_distance"
    )
    sources = artifact.manifest["source_sha256"]
    assert (
        "utils/ML_Fitting_1D_GISAXS/PosteriorV8/query_parameter_distance_v5.py"
        in sources
    )
    assert "src/gimap/features/fitting/domain/scattering_model.py" in sources
    assert "src/gimap/features/fitting/domain/physical_constraints.py" in sources
    assert artifact.manifest["ledger"]["exact_forward_calls_remaining"] == 0
    assert artifact.arrays["attempt_exact_forward_calls"].tolist() == [1, 1, 1, 1]
    assert np.all(artifact.arrays["candidate_local_unit"] == 0.5)
    assert np.all(artifact.arrays["candidate_k"] > 0.0)
    assert np.all(artifact.arrays["candidate_bounds_pass"])
    assert np.all(artifact.arrays["candidate_physics_pass"])
    with pytest.raises(FileExistsError, match="overwrite"):
        runner(task)


def test_missing_acceptance_sigma_uses_raw_metric_and_negative_requires_full_budget(tmp_path):
    schedule = _schedule(count=4, seed=130)
    optimizer = _optimizer(scouts=2, per_seed=2)
    task = _task(
        schedule=schedule,
        optimizer=optimizer,
        sigma_available=False,
        threshold=0.0,
    )
    path = tmp_path / "negative.gvd5"
    result = execute_v5_frozen_branch_search(
        task,
        seed_schedule=schedule,
        optimizer_schedule=optimizer,
        artifact_path=path,
    )

    assert result.outcome == "no_compatible_found_within_frozen_search_budget"
    assert result.completed
    assert result.exact_forward_calls_used == 4
    assert result.termination_reason == "exact_forward_budget_exhausted_without_compatible"
    artifact = read_v5_exact_search_executor_artifact(path, task=task)
    assert (
        artifact.manifest["selected_metric_name"]
        == task.protocol.missing_acceptance_sigma_metric_name
    )
    assert not np.any(artifact.arrays["candidate_standardized_metric_available"])
    assert not np.any(artifact.arrays["candidate_compatible"])


def test_full_budget_retains_multiple_delta_separated_resolution_representatives(tmp_path):
    schedule = _schedule(count=8, seed=811)
    optimizer = _optimizer(scouts=2, per_seed=3)
    task = _task(
        schedule=schedule,
        optimizer=optimizer,
        sigma_available=False,
        threshold=1.0e12,
        delta=1.0e-10,
        resolution_present=True,
    )
    path = tmp_path / "multiple.gvd5"
    result = execute_v5_frozen_branch_search(
        task,
        seed_schedule=schedule,
        optimizer_schedule=optimizer,
        artifact_path=path,
    )

    assert result.completed and result.exact_forward_calls_used == 8
    assert len(result.representatives) >= 2
    assert len({value.cluster_id for value in result.representatives}) == len(
        result.representatives
    )
    assert len({value.target_local for value in result.representatives}) == len(
        result.representatives
    )
    assert all(value.target_local[24] != 0.5 for value in result.representatives)
    artifact = read_v5_exact_search_executor_artifact(path, task=task)
    assert artifact.arrays["representative_exact_intensity"].shape[0] == len(result.representatives)
    assert np.all(artifact.arrays["candidate_resolution_present"])


def test_exception_and_incomplete_budget_fail_closed_to_unverified(monkeypatch, tmp_path):
    schedule = _schedule(count=4, seed=160)
    optimizer = _optimizer(scouts=2, per_seed=2)
    task = _task(
        schedule=schedule,
        optimizer=optimizer,
        sigma_available=False,
        threshold=1.0e12,
    )
    real_runner = executor_module.run_v5_exact_refinement
    calls = 0

    def fail_after_one(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise FloatingPointError("injected scientific failure")
        return real_runner(*args, **kwargs)

    monkeypatch.setattr(executor_module, "run_v5_exact_refinement", fail_after_one)
    path = tmp_path / "failed.gvd5"
    result = execute_v5_frozen_branch_search(
        task,
        seed_schedule=schedule,
        optimizer_schedule=optimizer,
        artifact_path=path,
    )

    assert result.outcome == "unverified"
    assert not result.completed
    assert result.exact_forward_calls_used == 1 < task.protocol.exact_forward_call_budget
    assert result.representatives == ()
    artifact = read_v5_exact_search_executor_artifact(path, task=task)
    assert artifact.manifest["failure"]["exception_type"] == "FloatingPointError"
    assert artifact.manifest["ledger"]["exact_forward_calls_remaining"] == 3

    def fail_immediately(*args, **kwargs):
        raise RuntimeError("failure before the first exact curve")

    monkeypatch.setattr(executor_module, "run_v5_exact_refinement", fail_immediately)
    empty_path = tmp_path / "failed-empty.gvd5"
    empty = execute_v5_frozen_branch_search(
        task,
        seed_schedule=schedule,
        optimizer_schedule=optimizer,
        artifact_path=empty_path,
    )
    assert empty.outcome == "unverified" and empty.exact_forward_calls_used == 0
    empty_artifact = read_v5_exact_search_executor_artifact(empty_path, task=task)
    assert empty_artifact.arrays["candidate_local_unit"].shape == (0, 26)
    assert empty_artifact.arrays["representative_exact_intensity"].shape[0] == 0


def test_slsqp_line_search_failure_keeps_only_audited_feasible_exact_fallback(
    monkeypatch, tmp_path
):
    schedule = _schedule(count=4, seed=175)
    optimizer = _optimizer(scouts=2, per_seed=2)
    task = _task(
        schedule=schedule,
        optimizer=optimizer,
        sigma_available=False,
        threshold=1.0e12,
        fixed=True,
    )
    real_minimize = profiled_forward_module.minimize

    def line_search_failure(*args, **kwargs):
        solved = real_minimize(*args, **kwargs)
        solved.success = False
        solved.status = 8
        solved.message = "Positive directional derivative for linesearch"
        return solved

    monkeypatch.setattr(profiled_forward_module, "minimize", line_search_failure)
    path = tmp_path / "feasible-fallback.gvd5"
    result = execute_v5_frozen_branch_search(
        task,
        seed_schedule=schedule,
        optimizer_schedule=optimizer,
        artifact_path=path,
    )

    assert result.completed and result.exact_forward_calls_used == 4
    artifact = read_v5_exact_search_executor_artifact(path, task=task)
    assert all(
        "not an optimality certificate" in value for value in artifact.arrays["attempt_message"]
    )
    assert all(
        f"amplitude_solver_status={POLYTOPE_FEASIBLE_FALLBACK_STATUS}" in value
        for value in artifact.arrays["attempt_message"]
    )


def test_protocol_and_semantic_artifact_tampering_fail_before_label_use(tmp_path):
    schedule = _schedule(count=4, seed=190)
    optimizer = _optimizer(scouts=2, per_seed=2)
    task = _task(
        schedule=schedule,
        optimizer=optimizer,
        sigma_available=False,
        threshold=1.0e12,
    )
    stale = replace(task, protocol=replace(task.protocol, seed_schedule_sha256="0" * 64))
    with pytest.raises(ValueError, match="seed schedule digest"):
        execute_v5_frozen_branch_search(
            stale,
            seed_schedule=schedule,
            optimizer_schedule=optimizer,
            artifact_path=tmp_path / "must-not-exist.gvd5",
        )
    assert not (tmp_path / "must-not-exist.gvd5").exists()

    path = tmp_path / "valid.gvd5"
    execute_v5_frozen_branch_search(
        task,
        seed_schedule=schedule,
        optimizer_schedule=optimizer,
        artifact_path=path,
    )
    artifact = read_v5_exact_search_executor_artifact(path, task=task)
    forged = tmp_path / "forged.gvd5"
    _write_semantically_forged_artifact(
        artifact,
        forged,
        lambda arrays: arrays["candidate_compatible"].__setitem__(0, False),
    )
    with pytest.raises(ValueError, match="compatibility flags"):
        read_v5_exact_search_executor_artifact(forged, task=task)


def test_negative_replays_every_candidate_curve_metric_and_hash(tmp_path):
    schedule = _schedule(count=4, seed=191)
    optimizer = _optimizer(scouts=2, per_seed=2)
    task = _task(
        schedule=schedule,
        optimizer=optimizer,
        sigma_available=False,
        threshold=0.0,
    )
    path = tmp_path / "negative-valid.gvd5"
    result = execute_v5_frozen_branch_search(
        task,
        seed_schedule=schedule,
        optimizer_schedule=optimizer,
        artifact_path=path,
    )
    assert result.outcome == "no_compatible_found_within_frozen_search_budget"
    artifact = read_v5_exact_search_executor_artifact(path, task=task)
    assert artifact.arrays["representative_exact_intensity"].shape[0] == 0
    assert artifact.arrays["candidate_selected_metric_value"].size > 0

    metric_forged = tmp_path / "negative-metric-forged.gvd5"
    _write_semantically_forged_artifact(
        artifact,
        metric_forged,
        lambda arrays: arrays["candidate_selected_metric_value"].__setitem__(
            0, arrays["candidate_selected_metric_value"][0] + 0.125
        ),
    )
    with pytest.raises(ValueError, match="selected exact metric"):
        read_v5_exact_search_executor_artifact(metric_forged, task=task)

    hash_forged = tmp_path / "negative-hash-forged.gvd5"
    _write_semantically_forged_artifact(
        artifact,
        hash_forged,
        lambda arrays: arrays["candidate_exact_intensity_sha256"].__setitem__(0, "0" * 64),
    )
    with pytest.raises(ValueError, match="exact GUI curve digest"):
        read_v5_exact_search_executor_artifact(hash_forged, task=task)

    k_forged = tmp_path / "negative-k-forged.gvd5"
    _write_semantically_forged_artifact(
        artifact,
        k_forged,
        lambda arrays: arrays["candidate_k"].__setitem__(0, 1.0e300),
    )
    with pytest.raises(ValueError, match="concrete frozen GUI ranges"):
        read_v5_exact_search_executor_artifact(k_forged, task=task)


def test_publish_requires_semantic_readback_and_full_task_provenance(monkeypatch, tmp_path):
    schedule = _schedule(count=4, seed=192)
    optimizer = _optimizer(scouts=2, per_seed=2)
    task = _task(
        schedule=schedule,
        optimizer=optimizer,
        sigma_available=False,
        threshold=1.0e12,
    )
    valid_path = tmp_path / "provenance-valid.gvd5"
    execute_v5_frozen_branch_search(
        task,
        seed_schedule=schedule,
        optimizer_schedule=optimizer,
        artifact_path=valid_path,
    )
    artifact = read_v5_exact_search_executor_artifact(valid_path, task=task)
    provenance_forged = tmp_path / "provenance-forged.gvd5"
    _write_semantically_forged_manifest(
        artifact,
        provenance_forged,
        lambda manifest: manifest["task"].__setitem__("query_catalog_artifact_sha256", "0" * 64),
    )
    with pytest.raises(ValueError, match="different task"):
        read_v5_exact_search_executor_artifact(provenance_forged, task=task)

    rejected_path = tmp_path / "readback-rejected.gvd5"

    def reject_readback(*args, **kwargs):
        raise ValueError("injected semantic read-back rejection")

    monkeypatch.setattr(executor_module, "read_v5_exact_search_executor_artifact", reject_readback)
    with pytest.raises(ValueError, match="semantic read-back rejection"):
        execute_v5_frozen_branch_search(
            task,
            seed_schedule=schedule,
            optimizer_schedule=optimizer,
            artifact_path=rejected_path,
        )
    assert rejected_path.is_file()
