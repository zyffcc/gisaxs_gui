from __future__ import annotations

from dataclasses import replace
from hashlib import sha256

import numpy as np
import pytest

from utils.ML_Fitting_1D_GISAXS.PosteriorV8.amplitude_query_v5 import (
    V5AmplitudeQuery,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.bounds_query_sampling_v5 import (
    full_range_v5_query,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.calibrated_search_threshold_v5 import (
    V5CalibrationArtifactIdentity,
    V5_CALIBRATED_THRESHOLD_NAME,
    bind_v5_calibrated_observation_threshold,
    compatibility_stratum_from_v5_observation,
    inspect_v5_compatibility_calibration,
    read_v5_checked_compatibility_calibration,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.compatibility_calibration import (
    CompatibilityCalibrationSample,
    UnseenAcquisitionPolicyError,
    UnseenCompatibilityStratumError,
    fit_compatibility_calibration,
    write_compatibility_calibration_atomic,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.contract import ClosedInterval
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.exact_search_executor_v5 import (
    build_v5_frozen_exact_search_protocol,
    execute_v5_frozen_branch_search,
    read_v5_exact_search_executor_artifact,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.exact_search_schedule_v5 import (
    V5FrozenExactOptimizerSchedule,
    V5FrozenLocalSobolSchedule,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.observation_v5 import (
    build_v5_observation_data_view,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.search_supervision_contract_v5 import (
    V5ExactSearchObservation,
    V5FrozenExactSearchProtocol,
    V5FrozenSearchTask,
    V5_ENGINEERING_PILOT_CLAIM_LIMITS,
    V5_PAPER_FULL_CALIBRATED_CLAIM_LIMITS,
    V5_SEARCH_PROTOCOL_TIER_PAPER_FULL_CALIBRATED,
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


def _views():
    recipe = sample_v5_clean_recipe(("sphere",), recipe_seed=77)
    views = tuple(
        build_v5_observation_data_view(recipe, index, split_id="train") for index in range(8)
    )
    sigma = next(value for value in views if value.acceptance_sigma_log is not None)
    missing = next(value for value in views if value.acceptance_sigma_log is None)
    return recipe, sigma, missing


def _artifact_for(view):
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
    return fit_compatibility_calibration(
        samples,
        dataset_manifest_sha256="1" * 64,
        calibration_split_sha256="2" * 64,
        target_coverage=0.8,
        minimum_samples_per_stratum=5,
    )


def _schedules():
    seeds = V5FrozenLocalSobolSchedule.generate(
        schedule_id="calibrated-threshold-test", point_count=2, base_seed=17
    )
    optimizer = V5FrozenExactOptimizerSchedule(
        schedule_id="calibrated-threshold-test",
        direct_scout_seed_count=1,
        per_seed_forward_evaluation_limit=1,
    )
    return seeds, optimizer


def _formal_task(view, threshold, protocol):
    geometry = full_range_v5_query(("sphere",), query_seed=9812)
    amplitude = V5AmplitudeQuery.create(
        background=ClosedInterval(0.0, 1.0e4),
        k=ClosedInterval(1.0e-2, 1.0e4),
        component_intensities=(ClosedInterval(0.0, 1.0),),
        resolution_presence_policy=geometry.resolution_presence_policy,
        int_res=ClosedInterval(0.0, 10.0),
    )
    context = build_v5_universal_candidate_context(
        view.preprocessed,
        view.uncertainty,
        (V5TopologyQuery(geometry, amplitude),),
    )
    exact = V5ExactSearchObservation.from_observation_view(
        view, curve_id="formal-calibrated-observation"
    )
    return V5FrozenSearchTask(
        query_index=0,
        clean_group_id="clean-group-test",
        recipe_id="recipe-test",
        observation_id=exact.observed_curve.curve_id,
        universal_context=context,
        exact_observation=exact,
        query_catalog_artifact_id="query-catalog/test",
        query_catalog_artifact_sha256=sha256(b"query-catalog/test").hexdigest(),
        branch_index=0,
        protocol=protocol,
        calibrated_threshold=threshold,
    )


def test_checked_artifact_binds_reserved_split_and_rejects_file_drift(tmp_path):
    _, sigma_view, _ = _views()
    artifact = _artifact_for(sigma_view)
    path = tmp_path / "calibration.json"
    write_compatibility_calibration_atomic(path, artifact)

    planned = inspect_v5_compatibility_calibration(path)
    replay = read_v5_checked_compatibility_calibration(
        path,
        expected_artifact_sha256=planned.identity.artifact_sha256,
        expected_file_sha256=planned.identity.file_sha256,
    )
    assert replay.identity == planned.identity
    assert replay.identity.artifact_sha256 == artifact.sha256
    assert replay.identity.calibration_split_id == "calibration"

    changed_bytes = tmp_path / "same-logical-artifact-different-bytes.json"
    changed_bytes.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="file SHA-256 changed"):
        read_v5_checked_compatibility_calibration(
            changed_bytes,
            expected_artifact_sha256=planned.identity.artifact_sha256,
            expected_file_sha256=planned.identity.file_sha256,
        )


def test_pre_universe_checked_identity_is_not_silently_accepted(tmp_path):
    _, sigma_view, _ = _views()
    path = tmp_path / "calibration.json"
    write_compatibility_calibration_atomic(path, _artifact_for(sigma_view))
    payload = inspect_v5_compatibility_calibration(path).identity.audit_payload()
    payload["schema"] = "gisaxs.posterior_v8.checked_compatibility_calibration_identity/v1"
    payload["version"] = "posterior_v8_reserved_split_logical_and_file_digest_binding_v1"
    for field in (
        "preregistered_design_stratum_count",
        "design_stratum_universe_version",
        "design_stratum_universe_fields",
        "design_stratum_universe_semantics",
        "design_stratum_universe_sha256",
    ):
        del payload[field]

    with pytest.raises(ValueError, match="unsupported fields"):
        V5CalibrationArtifactIdentity.from_payload(payload)


def test_formal_lookup_is_per_observation_and_fails_closed(tmp_path):
    _, sigma_view, missing_sigma_view = _views()
    path = tmp_path / "calibration.json"
    write_compatibility_calibration_atomic(path, _artifact_for(sigma_view))
    checked = inspect_v5_compatibility_calibration(path)

    selected = bind_v5_calibrated_observation_threshold(checked, sigma_view)
    assert selected.stratum == compatibility_stratum_from_v5_observation(sigma_view)
    assert selected.threshold_name == V5_CALIBRATED_THRESHOLD_NAME
    assert selected.threshold_value == 8.0
    assert selected.calibration_identity == checked.identity
    assert len(selected.stratum_calibration_sha256) == 64

    with pytest.raises(ValueError, match="requires measured or simulated"):
        bind_v5_calibrated_observation_threshold(checked, missing_sigma_view)

    unseen = None
    for seed in range(78, 130):
        recipe = sample_v5_clean_recipe(("sphere",), recipe_seed=seed)
        for index in range(8):
            candidate = build_v5_observation_data_view(recipe, index, split_id="train")
            if (
                candidate.acceptance_sigma_log is not None
                and compatibility_stratum_from_v5_observation(candidate) != selected.stratum
            ):
                unseen = candidate
                break
        if unseen is not None:
            break
    assert unseen is not None
    with pytest.raises(UnseenCompatibilityStratumError, match="no conditional"):
        bind_v5_calibrated_observation_threshold(checked, unseen)


def test_formal_lookup_accepts_registered_nuisance_policies_and_rejects_unknown_full_policy(
    tmp_path,
):
    recipe, sigma_view, _ = _views()
    second_known = build_v5_observation_data_view(
        recipe,
        sigma_view.view_index + 60,
        split_id="train",
    )
    unknown = build_v5_observation_data_view(
        recipe,
        sigma_view.view_index + 120,
        split_id="train",
    )
    stratum = compatibility_stratum_from_v5_observation(sigma_view)
    assert compatibility_stratum_from_v5_observation(second_known) == stratum
    assert compatibility_stratum_from_v5_observation(unknown) == stratum
    assert (
        len(
            {
                sigma_view.acquisition_policy_id,
                second_known.acquisition_policy_id,
                unknown.acquisition_policy_id,
            }
        )
        == 3
    )
    assert second_known.acceptance_sigma_log is not None
    assert unknown.acceptance_sigma_log is not None

    known_views = (sigma_view, second_known)
    samples = tuple(
        CompatibilityCalibrationSample(
            sample_id=f"multi-policy-calibration-{index:03d}",
            independent_group_id=f"multi-policy-recipe-{index:03d}",
            stratum=stratum,
            score=float(index + 1),
            effective_valid_point_count=known_views[index % 2].effective_valid_point_count,
            acquisition_policy_id=known_views[index % 2].acquisition_policy_id,
            measurement_sigma_available=True,
        )
        for index in range(10)
    )
    artifact = fit_compatibility_calibration(
        samples,
        dataset_manifest_sha256="4" * 64,
        calibration_split_sha256="5" * 64,
        target_coverage=0.8,
        minimum_samples_per_stratum=5,
    )
    path = tmp_path / "multi-policy-calibration.json"
    write_compatibility_calibration_atomic(path, artifact)
    checked = inspect_v5_compatibility_calibration(path)

    assert bind_v5_calibrated_observation_threshold(checked, sigma_view).threshold_value == 9.0
    assert bind_v5_calibrated_observation_threshold(checked, second_known).threshold_value == 9.0
    with pytest.raises(UnseenAcquisitionPolicyError, match="full acquisition policy"):
        bind_v5_calibrated_observation_threshold(checked, unknown)


def test_protocol_tiers_make_pilot_ineligible_and_forbid_formal_scalars(tmp_path):
    _, sigma_view, _ = _views()
    path = tmp_path / "calibration.json"
    write_compatibility_calibration_atomic(path, _artifact_for(sigma_view))
    checked = inspect_v5_compatibility_calibration(path)
    selected = bind_v5_calibrated_observation_threshold(checked, sigma_view)
    seeds, optimizer = _schedules()

    pilot = build_v5_frozen_exact_search_protocol(
        protocol_id="engineering-throughput-only",
        seed_schedule=seeds,
        optimizer_schedule=optimizer,
        standardized_threshold_name="pilot-standardized",
        standardized_threshold_value=2.0,
        raw_threshold_name="pilot-raw",
        raw_threshold_value=0.1,
        threshold_source_id="pilot-only",
    )
    assert pilot.claim_limits == V5_ENGINEERING_PILOT_CLAIM_LIMITS

    formal = build_v5_frozen_exact_search_protocol(
        protocol_id="paper-full-calibrated",
        seed_schedule=seeds,
        optimizer_schedule=optimizer,
        protocol_tier=V5_SEARCH_PROTOCOL_TIER_PAPER_FULL_CALIBRATED,
        calibration_identity=checked.identity,
    )
    assert formal.claim_limits == V5_PAPER_FULL_CALIBRATED_CLAIM_LIMITS
    assert formal.threshold_value is None
    assert V5FrozenExactSearchProtocol(**formal.audit_payload()) == formal
    task = _formal_task(sigma_view, selected, formal)
    assert task.full_training_label_permitted
    assert task.selected_threshold_value == selected.threshold_value
    assert task.selected_threshold_source_id == selected.threshold_source_id
    assert len(task.audit_sha256) == 64

    with pytest.raises(ValueError, match="requires one calibrated threshold"):
        replace(task, calibrated_threshold=None)
    with pytest.raises(ValueError, match="forbids command-line scalar"):
        build_v5_frozen_exact_search_protocol(
            protocol_id="invalid-paper-full",
            seed_schedule=seeds,
            optimizer_schedule=optimizer,
            standardized_threshold_name="forbidden",
            standardized_threshold_value=1.0,
            protocol_tier=V5_SEARCH_PROTOCOL_TIER_PAPER_FULL_CALIBRATED,
            calibration_identity=checked.identity,
        )


def test_formal_executor_artifact_identity_includes_observation_threshold(tmp_path):
    _, sigma_view, _ = _views()
    path = tmp_path / "calibration.json"
    write_compatibility_calibration_atomic(path, _artifact_for(sigma_view))
    checked = inspect_v5_compatibility_calibration(path)
    threshold = bind_v5_calibrated_observation_threshold(checked, sigma_view)
    seeds, optimizer = _schedules()
    protocol = build_v5_frozen_exact_search_protocol(
        protocol_id="paper-full-calibrated-executor",
        seed_schedule=seeds,
        optimizer_schedule=optimizer,
        protocol_tier=V5_SEARCH_PROTOCOL_TIER_PAPER_FULL_CALIBRATED,
        calibration_identity=checked.identity,
    )
    task = _formal_task(sigma_view, threshold, protocol)
    artifact_path = tmp_path / "formal-executor.gvd5"

    result = execute_v5_frozen_branch_search(
        task,
        seed_schedule=seeds,
        optimizer_schedule=optimizer,
        artifact_path=artifact_path,
    )
    replay = read_v5_exact_search_executor_artifact(artifact_path, task=task)

    assert result.completed
    assert replay.manifest["task"]["task_audit_sha256"] == task.audit_sha256
    assert replay.manifest["task"]["calibrated_threshold_sha256"] == threshold.sha256
    assert replay.manifest["selected_threshold_source_id"] == (threshold.threshold_source_id)
