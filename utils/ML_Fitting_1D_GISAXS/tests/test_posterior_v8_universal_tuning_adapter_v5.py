from dataclasses import replace
from hashlib import sha256
from types import SimpleNamespace
import json

import numpy as np
import pytest

from utils.ML_Fitting_1D_GISAXS.PosteriorV8 import universal_tuning_adapter_v5 as adapter
from utils.ML_Fitting_1D_GISAXS.PosteriorV8 import tuning_checkpoint_model_v5 as loader
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.paper_checkpoint_selector_v5 import build_v5_checkpoint_evaluation_method_binding
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.tuning_checkpoint_runtime_v5 import V5RetainedFullCheckpoint
from utils.ML_Fitting_1D_GISAXS.tests.test_posterior_v8_universal_inference_v5 import _case, _model, _budget, _thresholds
from utils.ML_Fitting_1D_GISAXS.tests.test_posterior_v8_paper_budget_evaluator_v5 import _references


def _calibrated_runner(tmp_path, *, change_curve=False, missing_sigma=False):
    from utils.ML_Fitting_1D_GISAXS.tests.test_posterior_v8_calibrated_search_threshold_v5 import _views, _artifact_for
    from utils.ML_Fitting_1D_GISAXS.tests.test_posterior_v8_universal_inference_v5 import _topology_query
    from utils.ML_Fitting_1D_GISAXS.PosteriorV8.compatibility_calibration import write_compatibility_calibration_atomic
    from utils.ML_Fitting_1D_GISAXS.PosteriorV8.calibrated_search_threshold_v5 import inspect_v5_compatibility_calibration
    from utils.ML_Fitting_1D_GISAXS.PosteriorV8.universal_query_v5 import build_v5_universal_candidate_context
    _, view, missing_view = _views()
    path = tmp_path / "calibration.json"
    write_compatibility_calibration_atomic(path, _artifact_for(view))
    calibration = inspect_v5_compatibility_calibration(path)
    if missing_sigma:
        view = missing_view
    context = build_v5_universal_candidate_context(view.preprocessed, view.uncertainty,
                                                  (_topology_query("sphere", 11),))
    curve = adapter.V5ExactSearchObservation.from_observation_view(view, curve_id="query").observed_curve
    if change_curve:
        curve = replace(curve, sigma_log=curve.sigma_log * 2)
    runner = adapter.V5UniversalCheckpointTraceRunner(
        queries={"query": (context, curve)}, thresholds=_thresholds(loose=True),
        budget=_budget(forward_evaluation_limit=8, fallback_attempt_limit=8, sobol_seeds_per_branch=8),
        source_summary_artifact_sha256="c" * 64,
        calibration=calibration, observation_views={"query": view},
    )
    return runner, view


def test_calibrated_protocol_binds_actual_observation_and_lookup(tmp_path):
    runner, view = _calibrated_runner(tmp_path)
    payload = json.loads(runner.protocol_json)
    assert runner.protocol_id == adapter.V5_CALIBRATED_UNIVERSAL_TUNING_PROTOCOL_ID
    lookup = payload["calibrated_queries"]["query"]["threshold"]
    assert lookup["threshold_value"] == 8.0
    assert lookup["acquisition_policy_id_sha256"] == sha256(view.acquisition_policy_id.encode()).hexdigest()
    assert runner._calibration_binding() == runner._calibration_json


def test_calibrated_query_rejects_changed_acceptance_sigma(tmp_path):
    with pytest.raises(ValueError, match="context encoder uncertainty disagrees with its provenance"):
        _calibrated_runner(tmp_path, change_curve=True)


def test_calibrated_query_rejects_encoder_only_uncertainty(tmp_path):
    with pytest.raises(ValueError, match="requires measured or simulated acceptance sigma"):
        _calibrated_runner(tmp_path, missing_sigma=True)


def test_calibrated_query_rejects_identity_drift_before_loading(tmp_path):
    runner, _ = _calibrated_runner(tmp_path)
    object.__setattr__(runner._calibration.identity, "file_sha256", "e" * 64)
    with pytest.raises(RuntimeError, match="binding changed before"):
        runner(None, None, {}, 35, 8)


def test_calibrated_query_rejects_policy_mutation_before_loading(tmp_path):
    runner, view = _calibrated_runner(tmp_path)
    object.__setattr__(view, "acquisition_policy_id", "unknown-policy")
    with pytest.raises((ValueError, RuntimeError)):
        runner(None, None, {}, 35, 8)


def test_calibrated_threshold_reaches_real_search(tmp_path, monkeypatch):
    runner, _ = _calibrated_runner(tmp_path)
    _, checkpoint, reference, _, _ = _fixture(tmp_path, monkeypatch)
    context, _ = runner._queries["query"]
    reference = replace(reference, query_id="query", representatives=tuple(
        replace(row, payload=replace(row.payload, query_context_sha256=context.audit_sha256))
        for row in reference.representatives
    ))
    binding = build_v5_checkpoint_evaluation_method_binding(
        checkpoint_epoch=1, checkpoint_artifact_sha256=checkpoint.checkpoint_artifact_sha256,
        checkpoint_weights_sha256=checkpoint.checkpoint_weights_sha256,
        training_result_sha256=checkpoint.training_result_sha256,
        source_summary_artifact_sha256="c" * 64, method_id="unit-network",
        base_method_protocol_id=runner.protocol_id,
        base_method_protocol_sha256=runner.protocol_sha256, inference_seed=35,
        representative_selection_policy="budget_snapshot",
    )
    actual_search = adapter.run_v5_universal_one_click_inference
    seen = []

    def observe(*args, **kwargs):
        seen.append(kwargs["thresholds"].standardized_exact_log_rmse_max)
        return actual_search(*args, **kwargs)

    monkeypatch.setattr(adapter, "run_v5_universal_one_click_inference", observe)
    result = runner(checkpoint, reference, binding, 35, 8)
    assert seen == [8.0]
    assert len(result.exact_forward_calls) == 8


@pytest.mark.parametrize("views", [None, {}, {"wrong-query": None}])
def test_calibrated_view_set_must_cover_queries(tmp_path, views):
    runner, _ = _calibrated_runner(tmp_path)
    with pytest.raises(ValueError, match="supplied together|exactly cover"):
        adapter.V5UniversalCheckpointTraceRunner(
            queries=runner._queries, thresholds=runner._thresholds, budget=runner._budget,
            source_summary_artifact_sha256="c" * 64,
            calibration=runner._calibration, observation_views=views,
        )


def test_two_observation_strata_use_distinct_thresholds_in_actual_search(tmp_path, monkeypatch):
    from utils.ML_Fitting_1D_GISAXS.tests.test_posterior_v8_calibrated_search_threshold_v5 import _views
    from utils.ML_Fitting_1D_GISAXS.tests.test_posterior_v8_universal_inference_v5 import _topology_query
    from utils.ML_Fitting_1D_GISAXS.PosteriorV8.observation_v5 import build_v5_observation_data_view
    from utils.ML_Fitting_1D_GISAXS.PosteriorV8.calibrated_search_threshold_v5 import (
        compatibility_stratum_from_v5_observation, inspect_v5_compatibility_calibration,
    )
    from utils.ML_Fitting_1D_GISAXS.PosteriorV8.compatibility_calibration import (
        CompatibilityCalibrationSample, fit_compatibility_calibration,
        write_compatibility_calibration_atomic,
    )
    from utils.ML_Fitting_1D_GISAXS.PosteriorV8.universal_query_v5 import build_v5_universal_candidate_context
    recipe, first, _ = _views()
    candidates = [build_v5_observation_data_view(recipe, i, split_id="tuning_validation") for i in range(8)]
    second = next(v for v in candidates if v.acceptance_sigma_log is not None
                  and compatibility_stratum_from_v5_observation(v)
                  != compatibility_stratum_from_v5_observation(first))
    views = {"first": first, "second": second}
    samples, queries = [], {}
    for scale, (key, view) in enumerate(views.items(), 1):
        stratum = compatibility_stratum_from_v5_observation(view)
        samples.extend(CompatibilityCalibrationSample(
            sample_id=f"{key}-{i}", independent_group_id=f"{key}-recipe-{i}",
            stratum=stratum, score=float(scale * (i + 1)),
            effective_valid_point_count=view.effective_valid_point_count,
            acquisition_policy_id=view.acquisition_policy_id, measurement_sigma_available=True,
        ) for i in range(9))
        context = build_v5_universal_candidate_context(view.preprocessed, view.uncertainty,
                                                      (_topology_query("sphere", 11),))
        curve = adapter.V5ExactSearchObservation.from_observation_view(view, curve_id=key).observed_curve
        queries[key] = (context, curve)
    artifact = fit_compatibility_calibration(samples, dataset_manifest_sha256="1" * 64,
        calibration_split_sha256="2" * 64, target_coverage=0.8, minimum_samples_per_stratum=5)
    path = tmp_path / "two-strata.json"
    write_compatibility_calibration_atomic(path, artifact)
    runner = adapter.V5UniversalCheckpointTraceRunner(
        queries=queries, thresholds=_thresholds(loose=True),
        budget=_budget(forward_evaluation_limit=8, fallback_attempt_limit=8, sobol_seeds_per_branch=8),
        source_summary_artifact_sha256="c" * 64,
        calibration=inspect_v5_compatibility_calibration(path), observation_views=views,
    )
    _, checkpoint, reference, _, _ = _fixture(tmp_path, monkeypatch)
    binding = build_v5_checkpoint_evaluation_method_binding(
        checkpoint_epoch=1, checkpoint_artifact_sha256=checkpoint.checkpoint_artifact_sha256,
        checkpoint_weights_sha256=checkpoint.checkpoint_weights_sha256,
        training_result_sha256=checkpoint.training_result_sha256,
        source_summary_artifact_sha256="c" * 64, method_id="unit-network",
        base_method_protocol_id=runner.protocol_id, base_method_protocol_sha256=runner.protocol_sha256,
        inference_seed=35, representative_selection_policy="budget_snapshot",
    )
    actual, seen = adapter.run_v5_universal_one_click_inference, []
    def observe(*args, **kwargs):
        seen.append(kwargs["thresholds"].standardized_exact_log_rmse_max)
        return actual(*args, **kwargs)
    monkeypatch.setattr(adapter, "run_v5_universal_one_click_inference", observe)
    for key, (context, _) in queries.items():
        bound = replace(reference, query_id=key, representatives=tuple(
            replace(row, payload=replace(row.payload, query_context_sha256=context.audit_sha256))
            for row in reference.representatives))
        assert len(runner(checkpoint, bound, binding, 35, 8).exact_forward_calls) == 8
    assert seen == [8.0, 16.0]


def _fixture(tmp_path, monkeypatch, *, real_model=False):
    context, curve = _case()
    reference = _references()
    reference = replace(reference, representatives=tuple(
        replace(row, payload=replace(row.payload, query_context_sha256=context.audit_sha256))
        for row in reference.representatives
    ))
    path = tmp_path.resolve() / "checkpoint.keras"
    if real_model:
        from utils.ML_Fitting_1D_GISAXS.PosteriorV8.model_v5 import build_branch_conditioned_proposal_model
        from utils.ML_Fitting_1D_GISAXS.PosteriorV8.proposal_execution_policy_v5 import V5_PROPOSAL_EXECUTION_POLICY
        model = build_branch_conditioned_proposal_model(
            max_points=context.for_model()["x"].shape[1], width=8, encoder_blocks=1,
            mixture_components=V5_PROPOSAL_EXECUTION_POLICY.mixture_component_count,
        )
        model.save(path)
    else:
        model = _model()
        model.weights = [SimpleNamespace(name="unit-weight", numpy=lambda: np.array([1.0]))]
        path.write_bytes(b"unit-model-deserialization-fixture")
        monkeypatch.setattr(loader, "_load_graph", lambda path: model)
    path.chmod(0o400)
    checkpoint = V5RetainedFullCheckpoint(
        full_epoch=1, checkpoint_path=path,
        checkpoint_artifact_sha256=sha256(path.read_bytes()).hexdigest(),
        checkpoint_weights_sha256=loader.model_weights_sha256(model),
        training_result_sha256="b" * 64,
    )
    runner = adapter.V5UniversalCheckpointTraceRunner(
        queries={reference.query_id: (context, curve)}, thresholds=_thresholds(loose=True),
        budget=_budget(forward_evaluation_limit=8, fallback_attempt_limit=8,
                       sobol_seeds_per_branch=8),
        source_summary_artifact_sha256="c" * 64,
    )
    binding = build_v5_checkpoint_evaluation_method_binding(
        checkpoint_epoch=1, checkpoint_artifact_sha256=checkpoint.checkpoint_artifact_sha256,
        checkpoint_weights_sha256=checkpoint.checkpoint_weights_sha256,
        training_result_sha256=checkpoint.training_result_sha256,
        source_summary_artifact_sha256="c" * 64, method_id="unit-network",
        base_method_protocol_id=adapter.V5_UNIVERSAL_TUNING_PROTOCOL_ID,
        base_method_protocol_sha256=runner.protocol_sha256, inference_seed=35,
        representative_selection_policy="budget_snapshot",
    )
    return runner, checkpoint, reference, binding, model


def test_real_keras_checkpoint_runs_exact_budget_without_weight_changes(tmp_path, monkeypatch):
    runner, checkpoint, reference, binding, _ = _fixture(tmp_path, monkeypatch, real_model=True)
    result = runner(checkpoint, reference, binding, 35, 8)
    assert len(result.exact_forward_calls) == 8
    assert result.representative_snapshots
    assert sha256(checkpoint.checkpoint_path.read_bytes()).hexdigest() == checkpoint.checkpoint_artifact_sha256


def test_actual_refinement_trace_uses_checkpoint_and_never_reference_truth(tmp_path, monkeypatch):
    runner, checkpoint, reference, binding, model = _fixture(tmp_path, monkeypatch)
    original = adapter.run_v5_universal_one_click_inference

    def observe(*args, **kwargs):
        assert kwargs["model"] is model
        assert kwargs["reference_modes"] == kwargs["retrieval_seeds"] == ()
        assert kwargs["target_parameter_mode_count"] == 9
        return original(*args, **kwargs)

    monkeypatch.setattr(adapter, "run_v5_universal_one_click_inference", observe)
    result = runner(checkpoint, reference, binding, 35, 8)
    assert len(result.exact_forward_calls) == 8
    assert result.candidate_emissions and result.representative_snapshots
    assert result.method_protocol_sha256_used == binding["bound_method_protocol_sha256"]
    assert all(row.payload.source_artifact_sha256 == checkpoint.checkpoint_artifact_sha256
               for row in result.candidate_emissions)


@pytest.mark.parametrize("field", ["checkpoint_weights_sha256", "base_method_protocol_sha256",
                                   "inference_seed", "source_summary_artifact_sha256"])
def test_tampered_binding_rejected_before_model_load(tmp_path, monkeypatch, field):
    runner, checkpoint, reference, binding, _ = _fixture(tmp_path, monkeypatch)
    binding[field] = 36 if field == "inference_seed" else "d" * 64
    monkeypatch.setattr(loader, "_load_graph", lambda path: pytest.fail("must not load"))
    with pytest.raises(ValueError, match="binding"):
        runner(checkpoint, reference, binding, 35, 8)


def test_wrong_query_context_rejected_before_model_load(tmp_path, monkeypatch):
    runner, checkpoint, _, binding, _ = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(loader, "_load_graph", lambda path: pytest.fail("must not load"))
    with pytest.raises(ValueError, match="query context"):
        runner(checkpoint, _references(), binding, 35, 8)


def test_loaded_weight_mutation_is_fatal(tmp_path, monkeypatch):
    runner, checkpoint, reference, binding, model = _fixture(tmp_path, monkeypatch)
    original = adapter.run_v5_universal_one_click_inference

    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        model.weights[0].numpy = lambda: np.array([2.0])
        return result

    monkeypatch.setattr(adapter, "run_v5_universal_one_click_inference", mutate)
    with pytest.raises(RuntimeError, match="weights changed"):
        runner(checkpoint, reference, binding, 35, 8)


def test_short_trace_cannot_be_padded(tmp_path, monkeypatch):
    runner, checkpoint, reference, binding, _ = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(adapter, "run_v5_universal_one_click_inference", lambda *a, **k: None)
    with pytest.raises(ValueError, match="no padding"):
        runner(checkpoint, reference, binding, 35, 8)


def test_budget_mismatch_fails_before_loading(tmp_path, monkeypatch):
    runner, checkpoint, reference, binding, _ = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(loader, "_load_graph", lambda path: pytest.fail("must not load"))
    with pytest.raises(ValueError, match="requested budget"):
        runner(checkpoint, reference, binding, 35, 7)


def test_checkpoint_replacement_during_search_is_fatal(tmp_path, monkeypatch):
    runner, checkpoint, reference, binding, _ = _fixture(tmp_path, monkeypatch)
    original = adapter.run_v5_universal_one_click_inference

    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        checkpoint.checkpoint_path.chmod(0o600)
        checkpoint.checkpoint_path.write_bytes(b"replaced during inference")
        checkpoint.checkpoint_path.chmod(0o400)
        return result

    monkeypatch.setattr(adapter, "run_v5_universal_one_click_inference", mutate)
    with pytest.raises(RuntimeError, match="file or loaded weights changed"):
        runner(checkpoint, reference, binding, 35, 8)


def test_inference_summary_cannot_disagree_with_actual_events(tmp_path, monkeypatch):
    runner, checkpoint, reference, binding, _ = _fixture(tmp_path, monkeypatch)
    original = adapter.run_v5_universal_one_click_inference

    def mismatch(*args, **kwargs):
        result = original(*args, **kwargs)
        return SimpleNamespace(context_audit_sha256=result.context_audit_sha256,
                               forward_evaluations_used=8,
                               termination_reason="exact_forward_budget_exhausted",
                               attempts=())

    monkeypatch.setattr(adapter, "run_v5_universal_one_click_inference", mismatch)
    with pytest.raises(RuntimeError, match="call ledger"):
        runner(checkpoint, reference, binding, 35, 8)


def test_insufficient_seed_horizon_is_not_a_full_budget_protocol():
    context, curve = _case()
    with pytest.raises(ValueError, match="Sobol horizon"):
        adapter.V5UniversalCheckpointTraceRunner(
            queries={"query": (context, curve)}, thresholds=_thresholds(loose=True),
            budget=_budget(forward_evaluation_limit=4096),
            source_summary_artifact_sha256="c" * 64,
        )
