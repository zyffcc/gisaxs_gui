import json
import os
from dataclasses import replace

import pytest

from utils.ML_Fitting_1D_GISAXS.PosteriorV8 import universal_tuning_adapter_v5 as adapter
from utils.ML_Fitting_1D_GISAXS.tests.test_posterior_v8_universal_tuning_adapter_v5 import _calibrated_runner, _fixture
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.paper_checkpoint_selector_v5 import build_v5_checkpoint_evaluation_method_binding


def _sealed(runner, path):
    return adapter.V5UniversalCheckpointTraceRunner(
        queries=runner._queries, thresholds=runner._thresholds, budget=runner._budget,
        source_summary_artifact_sha256=runner._source_sha,
        calibration=runner._calibration, observation_views=runner._views,
        calibration_path=path,
    )


def test_sealed_calibration_binds_guard_and_detects_same_bytes_replacement(tmp_path):
    original, _ = _calibrated_runner(tmp_path)
    path = tmp_path / "calibration.json"
    path.chmod(0o400)
    runner = _sealed(original, path)
    assert json.loads(runner.protocol_json)["calibration_file_guard"] == "0400_single_link_pre_post_v1"
    assert runner.protocol_sha256 != original.protocol_sha256
    replacement = tmp_path / "replacement.json"
    replacement.write_bytes(path.read_bytes())
    replacement.chmod(0o400)
    replacement.replace(path)
    with pytest.raises(RuntimeError, match="file identity changed"):
        runner(None, None, {}, 35, 8)


@pytest.mark.parametrize("kind", ["writable", "mode444", "hardlink", "symlink", "changed_bytes"])
def test_unsealed_or_changed_calibration_rejected(tmp_path, kind):
    runner, _ = _calibrated_runner(tmp_path)
    path = tmp_path / "calibration.json"
    if kind == "changed_bytes":
        path.write_bytes(path.read_bytes() + b"\n")
    path.chmod(0o400)
    if kind == "writable":
        path.chmod(0o600)
    elif kind == "mode444":
        path.chmod(0o444)
    elif kind == "hardlink":
        os.link(path, tmp_path / "alias.json")
    elif kind == "symlink":
        alias = tmp_path / "alias.json"
        alias.symlink_to(path)
        path = alias
    with pytest.raises((ValueError, RuntimeError)):
        _sealed(runner, path)


@pytest.mark.parametrize("field", ["artifact_sha256", "file_sha256"])
def test_sealed_file_must_reproduce_both_planned_hashes(tmp_path, field):
    runner, _ = _calibrated_runner(tmp_path)
    path = tmp_path / "calibration.json"
    path.chmod(0o400)
    object.__setattr__(runner._calibration.identity, field, "e" * 64)
    with pytest.raises(ValueError, match="SHA-256 changed after planning"):
        _sealed(runner, path)


@pytest.mark.parametrize("mutation", [None, "replace", "permissions", "content"])
def test_actual_search_rechecks_sealed_calibration_before_return(tmp_path, monkeypatch, mutation):
    original, _ = _calibrated_runner(tmp_path)
    path = tmp_path / "calibration.json"
    path.chmod(0o400)
    runner = _sealed(original, path)
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
    completed = []

    def run_and_mutate(*args, **kwargs):
        result = actual_search(*args, **kwargs)
        completed.append(result.forward_evaluations_used)
        if mutation == "replace":
            other = tmp_path / "replacement.json"
            other.write_bytes(path.read_bytes())
            other.chmod(0o400)
            other.replace(path)
        elif mutation == "permissions":
            path.chmod(0o600)
        elif mutation == "content":
            path.chmod(0o600)
            path.write_bytes(path.read_bytes() + b"\n")
            path.chmod(0o400)
        return result

    monkeypatch.setattr(adapter, "run_v5_universal_one_click_inference", run_and_mutate)
    if mutation is None:
        assert len(runner(checkpoint, reference, binding, 35, 8).exact_forward_calls) == 8
    else:
        with pytest.raises((ValueError, RuntimeError), match="calibration"):
            runner(checkpoint, reference, binding, 35, 8)
    assert completed == [8]
