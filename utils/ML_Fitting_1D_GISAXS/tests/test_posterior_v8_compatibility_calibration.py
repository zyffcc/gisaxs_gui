from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from utils.ML_Fitting_1D_GISAXS.PosteriorV8.compatibility_calibration import (
    ACQUISITION_POLICY_ID_VERSION,
    CALIBRATION_SCHEMA,
    CALIBRATION_VERSION,
    COMPATIBILITY_STRATUM_FIELDS,
    COMPATIBILITY_STRATUM_VERSION,
    DESIGN_POINT_COUNT_SEMANTICS,
    DESIGN_STRATUM_UNIVERSE_FIELDS,
    DESIGN_STRATUM_UNIVERSE_SEMANTICS,
    DESIGN_STRATUM_UNIVERSE_SHA256,
    DESIGN_STRATUM_UNIVERSE_VERSION,
    DUPLICATE_OBSERVATION_POLICY,
    EFFECTIVE_VALID_POINT_COUNT_SEMANTICS,
    FALLBACK_SEMANTICS,
    OBSERVATION_ESTIMAND,
    PREREGISTERED_DESIGN_STRATUM_COUNT,
    RESERVED_CALIBRATION_SPLIT_ID,
    SCORE_SEMANTICS,
    STRATIFICATION_SEMANTICS,
    CompatibilityCalibrationError,
    CompatibilityCalibrationSample,
    CompatibilityStratum,
    UnseenAcquisitionPolicyError,
    UnseenCompatibilityStratumError,
    acquisition_policy_payload,
    fit_compatibility_calibration,
    load_compatibility_calibration,
    make_acquisition_policy_id,
    write_compatibility_calibration_atomic,
)


def _stratum(*, points=64, noise="counting-v1", q_window="full-v1"):
    return CompatibilityStratum(
        point_count=points,
        noise_id=noise,
        q_window_id=q_window,
    )


def _provenance(stratum, *, view_index=0, effective_points=None, sigma=True):
    effective_points = effective_points or stratum.point_count
    policy_id = make_acquisition_policy_id(
        grid={
            "kind": "geometric",
            "design_point_count": stratum.point_count,
            "q_min": 0.001,
            "q_max": 1.0,
            "q_window_id": stratum.q_window_id,
        },
        mask={"mask_id": "mask-v1:0", "point_keep_probability": 1.0},
        crop={"crop_id": "crop-v1:0", "q_min": 0.001, "q_max": 1.0},
        view={
            "observation_view_version": "view-v1",
            "view_index": view_index,
            "observation_seed_derivation": "recipe-seed-plus-view-v1",
        },
        sigma={
            "noise_id": stratum.noise_id,
            "poisson_count_scale": 10000.0,
            "relative_sigma": 0.01,
            "sigma_floor_fraction": 1.0e-6,
            "sigma_log_source": "measured-sigma-over-observed-intensity",
        },
    )
    return {
        "effective_valid_point_count": effective_points,
        "acquisition_policy_id": policy_id,
        "measurement_sigma_available": sigma,
    }


def _samples(stratum, scores, prefix):
    return tuple(
        CompatibilityCalibrationSample(
            sample_id=f"{prefix}-{index:03d}",
            independent_group_id=f"recipe-{prefix}-{index:03d}",
            stratum=stratum,
            score=score,
            **_provenance(stratum),
        )
        for index, score in enumerate(scores)
    )


def _artifact():
    first = _stratum()
    second = _stratum(points=128, noise="relative-v2", q_window="mid-v2")
    samples = _samples(first, np.arange(1.0, 10.0), "a") + _samples(
        second, np.arange(11.0, 20.0), "b"
    )
    return (
        fit_compatibility_calibration(
            samples,
            dataset_manifest_sha256="1" * 64,
            calibration_split_sha256="2" * 64,
            target_coverage=0.8,
            minimum_samples_per_stratum=5,
        ),
        samples,
        first,
        second,
    )


def test_finite_sample_order_statistic_per_stratum_and_global_fallback():
    artifact, _, first, _ = _artifact()

    threshold, used_fallback = artifact.threshold_for(first)
    assert not used_fallback
    assert threshold.sample_count == 9
    assert threshold.order_statistic == 8
    assert threshold.guaranteed_coverage == pytest.approx(0.8)
    assert threshold.threshold == 8.0

    unseen = _stratum(points=256, noise="ood", q_window="ood")
    with pytest.raises(UnseenCompatibilityStratumError, match="no conditional"):
        artifact.threshold_for(unseen)
    fallback, used_fallback = artifact.threshold_for(unseen, allow_descriptive_fallback=True)
    assert used_fallback
    assert fallback is artifact.global_fallback
    assert fallback.sample_count == 18
    assert fallback.order_statistic == 16
    assert fallback.coverage_bound is None
    assert fallback.threshold == 17.0
    assert artifact.fallback_semantics == FALLBACK_SEMANTICS


def test_full_acquisition_policy_lookup_accepts_registered_and_rejects_unknown_policy():
    artifact, samples, first, _ = _artifact()
    known_policy = next(value.acquisition_policy_id for value in samples if value.stratum == first)
    unknown_policy = _provenance(first, view_index=999)["acquisition_policy_id"]

    assert artifact.threshold_for_acquisition_policy(first, known_policy).threshold == 8.0
    with pytest.raises(UnseenAcquisitionPolicyError, match="full acquisition policy"):
        artifact.threshold_for_acquisition_policy(first, unknown_policy)


def test_repeated_recipe_stratum_views_fail_closed_instead_of_max_aggregation():
    stratum = _stratum()
    samples = tuple(
        CompatibilityCalibrationSample(
            sample_id=f"recipe-{group}-view-{view}",
            independent_group_id=f"recipe-{group}",
            stratum=stratum,
            score=float(group + 1) + 0.5 * view,
            **_provenance(stratum, view_index=view),
        )
        for group in range(9)
        for view in range(2)
    )
    with pytest.raises(CompatibilityCalibrationError, match="single-view estimand"):
        fit_compatibility_calibration(
            samples,
            dataset_manifest_sha256="3" * 64,
            calibration_split_sha256="4" * 64,
            target_coverage=0.8,
            minimum_samples_per_stratum=5,
        )


def test_one_recipe_may_supply_one_view_to_each_distinct_design_stratum():
    first = _stratum()
    second = _stratum(points=128)
    samples = tuple(
        CompatibilityCalibrationSample(
            sample_id=f"recipe-{group}-stratum-{index}",
            independent_group_id=f"recipe-{group}",
            stratum=stratum,
            score=float(group + 1 + index),
            **_provenance(stratum, view_index=index, effective_points=stratum.point_count - 4),
        )
        for group in range(9)
        for index, stratum in enumerate((first, second))
    )
    artifact = fit_compatibility_calibration(
        samples,
        dataset_manifest_sha256="3" * 64,
        calibration_split_sha256="4" * 64,
        target_coverage=0.8,
        minimum_samples_per_stratum=5,
    )

    assert artifact.input_summary.observation_count == 18
    assert artifact.input_summary.independent_group_count == 9
    assert artifact.input_summary.recipe_stratum_count == 18
    assert artifact.global_fallback.sample_count == 18
    assert all(
        value.observation_count == value.calibration.sample_count for value in artifact.strata
    )


def test_artifact_records_versioned_semantics_summary_and_order_invariant_hash():
    artifact, samples, _, _ = _artifact()
    reordered = fit_compatibility_calibration(
        samples[::-1],
        dataset_manifest_sha256="1" * 64,
        calibration_split_sha256="2" * 64,
        target_coverage=0.8,
        minimum_samples_per_stratum=5,
    )

    assert artifact.schema == CALIBRATION_SCHEMA
    assert artifact.calibration_version == CALIBRATION_VERSION
    assert artifact.compatibility_stratum_version == COMPATIBILITY_STRATUM_VERSION
    assert artifact.calibration_split_id == RESERVED_CALIBRATION_SPLIT_ID
    assert artifact.compatibility_stratum_fields == COMPATIBILITY_STRATUM_FIELDS
    assert artifact.stratification_semantics == STRATIFICATION_SEMANTICS
    assert artifact.observation_estimand == OBSERVATION_ESTIMAND
    assert artifact.duplicate_observation_policy == DUPLICATE_OBSERVATION_POLICY
    assert artifact.preregistered_design_stratum_count == PREREGISTERED_DESIGN_STRATUM_COUNT == 60
    assert artifact.design_stratum_universe_version == DESIGN_STRATUM_UNIVERSE_VERSION
    assert artifact.design_stratum_universe_fields == DESIGN_STRATUM_UNIVERSE_FIELDS
    assert artifact.design_stratum_universe_semantics == DESIGN_STRATUM_UNIVERSE_SEMANTICS
    assert artifact.design_stratum_universe_sha256 == DESIGN_STRATUM_UNIVERSE_SHA256
    assert artifact.design_point_count_semantics == DESIGN_POINT_COUNT_SEMANTICS
    assert artifact.effective_valid_point_count_semantics == (EFFECTIVE_VALID_POINT_COUNT_SEMANTICS)
    assert artifact.acquisition_policy_id_version == ACQUISITION_POLICY_ID_VERSION
    assert artifact.score_semantics == SCORE_SEMANTICS
    assert "pseudo_likelihood" in artifact.score_semantics
    assert artifact.input_summary.observation_count == 18
    assert artifact.input_summary.independent_group_count == 18
    assert artifact.input_summary.recipe_stratum_count == 18
    assert artifact.input_summary.stratum_count == 2
    assert artifact.input_summary.acquisition_policy_count == 2
    assert artifact.input_summary.effective_valid_point_count_min == 64
    assert artifact.input_summary.effective_valid_point_count_max == 128
    assert artifact.input_summary.score_min == 1.0
    assert artifact.input_summary.score_median == 10.0
    assert artifact.input_summary.score_max == 19.0
    assert len(artifact.input_sha256) == 64
    assert len(artifact.sha256) == 64
    assert artifact.input_sha256 == reordered.input_sha256
    assert artifact.to_payload() == reordered.to_payload()
    policy_payload = acquisition_policy_payload(artifact.strata[0].acquisition_policy_ids[0])
    assert set(policy_payload) == {"version", "grid", "mask", "crop", "view", "sigma"}
    assert policy_payload["grid"]["design_point_count"] == 64
    assert artifact.strata[0].effective_valid_point_count_histogram[0].observation_count == 9


def test_calibration_rejects_nonfinite_negative_duplicate_and_insufficient_samples():
    stratum = _stratum()
    with pytest.raises(CompatibilityCalibrationError, match="finite"):
        CompatibilityCalibrationSample(
            sample_id="bad",
            independent_group_id="recipe-bad",
            stratum=stratum,
            score=np.nan,
            **_provenance(stratum),
        )
    with pytest.raises(CompatibilityCalibrationError, match="non-negative"):
        CompatibilityCalibrationSample(
            sample_id="bad",
            independent_group_id="recipe-bad",
            stratum=stratum,
            score=-0.1,
            **_provenance(stratum),
        )

    duplicate = CompatibilityCalibrationSample(
        sample_id="same",
        independent_group_id="recipe-same",
        stratum=stratum,
        score=1.0,
        **_provenance(stratum),
    )

    with pytest.raises(CompatibilityCalibrationError, match="measurement sigma"):
        CompatibilityCalibrationSample(
            sample_id="missing-sigma",
            independent_group_id="recipe-missing-sigma",
            stratum=stratum,
            score=1.0,
            **_provenance(stratum, sigma=False),
        )
    with pytest.raises(CompatibilityCalibrationError, match="unique"):
        fit_compatibility_calibration(
            (duplicate, duplicate),
            dataset_manifest_sha256="1" * 64,
            calibration_split_sha256="2" * 64,
            target_coverage=0.5,
            minimum_samples_per_stratum=1,
        )
    with pytest.raises(CompatibilityCalibrationError, match="insufficient"):
        fit_compatibility_calibration(
            _samples(stratum, [0.8, 1.0, 1.2], "s"),
            dataset_manifest_sha256="1" * 64,
            calibration_split_sha256="2" * 64,
            target_coverage=0.8,
            minimum_samples_per_stratum=5,
        )
    with pytest.raises(CompatibilityCalibrationError, match="finite split-conformal"):
        fit_compatibility_calibration(
            _samples(stratum, np.arange(10.0), "s"),
            dataset_manifest_sha256="1" * 64,
            calibration_split_sha256="2" * 64,
            target_coverage=0.95,
            minimum_samples_per_stratum=5,
        )


def test_canonical_json_round_trip_is_fail_closed_and_refuses_overwrite(tmp_path):
    artifact, _, _, _ = _artifact()
    path = tmp_path / "compatibility-calibration.json"
    write_compatibility_calibration_atomic(path, artifact)

    assert load_compatibility_calibration(path) == artifact
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert set(payload) == {
        "schema",
        "calibration_version",
        "score_semantics",
        "calibration_unit",
        "group_aggregation",
        "observation_estimand",
        "duplicate_observation_policy",
        "preregistered_design_stratum_count",
        "design_stratum_universe_version",
        "design_stratum_universe_fields",
        "design_stratum_universe_semantics",
        "design_stratum_universe_sha256",
        "design_point_count_semantics",
        "effective_valid_point_count_semantics",
        "acquisition_policy_id_version",
        "acquisition_policy_required_components",
        "measurement_sigma_policy",
        "fallback_semantics",
        "compatibility_stratum_version",
        "compatibility_stratum_fields",
        "stratification_semantics",
        "target_coverage",
        "minimum_samples_per_stratum",
        "input_summary",
        "input_sha256",
        "dataset_manifest_sha256",
        "calibration_split_id",
        "calibration_split_sha256",
        "global_fallback",
        "strata",
    }
    with pytest.raises(FileExistsError, match="overwrite"):
        write_compatibility_calibration_atomic(path, artifact)
    assert not list(tmp_path.glob(".compatibility-calibration.json.*.tmp"))


def test_pre_universe_v5_artifact_is_not_silently_accepted(tmp_path):
    artifact, _, _, _ = _artifact()
    payload = artifact.to_payload()
    payload["schema"] = "gisaxs.posterior_v8.compatibility_calibration/v5"
    payload["calibration_version"] = (
        "posterior_v8_reserved_split_single_view_marginal_acquisition_compatibility_v5"
    )
    for field in (
        "design_stratum_universe_version",
        "design_stratum_universe_fields",
        "design_stratum_universe_semantics",
        "design_stratum_universe_sha256",
    ):
        del payload[field]
    path = tmp_path / "pre-universe-v5.json"
    path.write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")

    with pytest.raises(CompatibilityCalibrationError, match="fields are invalid"):
        load_compatibility_calibration(path)


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda value: value.__setitem__("unknown", 1), "fields are invalid"),
        (
            lambda value: value["global_fallback"].__setitem__("unknown", 1),
            "fields are invalid",
        ),
        (
            lambda value: value.__setitem__("calibration_version", "future-v99"),
            "incompatible calibration version",
        ),
        (
            lambda value: value.__setitem__("calibration_split_id", "tuning_validation"),
            "reserved calibration split",
        ),
        (
            lambda value: value.__setitem__(
                "schema", "gisaxs.posterior_v8.compatibility_calibration/v3"
            ),
            "incompatible calibration schema",
        ),
        (
            lambda value: value.__setitem__("group_aggregation", "within_stratum_max"),
            "incompatible group aggregation",
        ),
        (
            lambda value: value.__setitem__(
                "schema", "gisaxs.posterior_v8.compatibility_calibration/v2"
            ),
            "incompatible calibration schema",
        ),
        (
            lambda value: value["strata"][0]["stratum"].__setitem__("component_count", 1),
            "stratum fields are invalid",
        ),
        (
            lambda value: value["strata"][0]["calibration"].__setitem__("order_statistic", 1),
            "order_statistic is inconsistent",
        ),
        (
            lambda value: value["strata"][0].__setitem__(
                "acquisition_policy_ids", ("legacy-incomplete-policy",)
            ),
            "complete provenance version",
        ),
        (
            lambda value: value["strata"][0]["effective_valid_point_count_histogram"][
                0
            ].__setitem__("observation_count", 1),
            "does not cover all observations",
        ),
        (
            lambda value: value["input_summary"].__setitem__("observation_count", 999),
            "counts are inconsistent",
        ),
    ],
)
def test_loader_rejects_unknown_versioned_and_inconsistent_fields(tmp_path, mutation, match):
    artifact, _, _, _ = _artifact()
    payload = artifact.to_payload()
    mutation(payload)
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")

    with pytest.raises(CompatibilityCalibrationError, match=match):
        load_compatibility_calibration(path)


def test_loader_rejects_nan_duplicate_json_field_and_symlink(tmp_path):
    artifact, _, _, _ = _artifact()
    payload = artifact.to_payload()

    nan_path = tmp_path / "nan.json"
    payload["global_fallback"]["threshold"] = float("nan")
    nan_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(CompatibilityCalibrationError, match="non-finite JSON"):
        load_compatibility_calibration(nan_path)

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema":"a","schema":"b"}', encoding="utf-8")
    with pytest.raises(CompatibilityCalibrationError, match="duplicate JSON"):
        load_compatibility_calibration(duplicate)

    target = tmp_path / "target.json"
    write_compatibility_calibration_atomic(target, artifact)
    link = tmp_path / "link.json"
    link.symlink_to(target.name)
    with pytest.raises(CompatibilityCalibrationError, match="symbolic link"):
        load_compatibility_calibration(link)


def test_stratum_validation_is_strict():
    stratum = _stratum()
    with pytest.raises(CompatibilityCalibrationError, match="q_window_id"):
        _stratum(q_window=" ")
    with pytest.raises(CompatibilityCalibrationError, match="noise_id"):
        _stratum(noise=" ")
    with pytest.raises(CompatibilityCalibrationError, match="finite"):
        CompatibilityCalibrationSample(
            sample_id="bad",
            independent_group_id="recipe-bad",
            stratum=stratum,
            score=True,
            **_provenance(stratum),
        )
    with pytest.raises(CompatibilityCalibrationError, match="cannot exceed"):
        CompatibilityCalibrationSample(
            sample_id="too-many-effective-points",
            independent_group_id="recipe-too-many-effective-points",
            stratum=stratum,
            score=1.0,
            **_provenance(stratum, effective_points=65),
        )
    with pytest.raises(CompatibilityCalibrationError, match="complete provenance version"):
        CompatibilityCalibrationSample(
            sample_id="incomplete-policy",
            independent_group_id="recipe-incomplete-policy",
            stratum=stratum,
            score=1.0,
            effective_valid_point_count=64,
            acquisition_policy_id="legacy-grid-only-policy",
            measurement_sigma_available=True,
        )
    with pytest.raises(CompatibilityCalibrationError, match="does not match"):
        CompatibilityCalibrationSample(
            sample_id="mismatched-policy",
            independent_group_id="recipe-mismatched-policy",
            stratum=stratum,
            score=1.0,
            **_provenance(_stratum(points=128), effective_points=64),
        )
    with pytest.raises(CompatibilityCalibrationError, match="finite"):
        CompatibilityCalibrationSample(
            sample_id="bad-string",
            independent_group_id="recipe-bad-string",
            stratum=stratum,
            score="1.0",
            **_provenance(stratum),
        )
    with pytest.raises(CompatibilityCalibrationError, match="positive integer"):
        CompatibilityStratum(
            point_count="64",
            noise_id="noise-v1",
            q_window_id="full-v1",
        )
    with pytest.raises(TypeError, match="component_count"):
        CompatibilityStratum(
            point_count=64,
            noise_id="noise-v1",
            q_window_id="full-v1",
            component_count=1,
        )
