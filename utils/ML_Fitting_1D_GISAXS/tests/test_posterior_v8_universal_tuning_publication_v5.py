"""Small-grid engineering integration, never evidence for paper B4096 gates."""

from dataclasses import replace
import json
import stat
import pytest

from utils.ML_Fitting_1D_GISAXS.PosteriorV8 import (
    paper_checkpoint_selector_v5 as selector,
    tuning_checkpoint_runtime_v5 as runtime,
    tuning_checkpoint_summary_io_v5 as reader,
    tuning_trace_artifact_v5 as trace_store,
    universal_tuning_adapter_v5 as adapter,
)
from utils.ML_Fitting_1D_GISAXS.tests.test_posterior_v8_tuning_checkpoint_runtime_v5 import _cohort, _config
from utils.ML_Fitting_1D_GISAXS.tests.test_posterior_v8_universal_tuning_adapter_v5 import _calibrated_runner, _fixture
from utils.ML_Fitting_1D_GISAXS.tests.test_posterior_v8_tuning_calibration_file_v5 import _sealed


@pytest.mark.parametrize("mutate_calibration", [False, True])
def test_engineering_actual_search_publishes_and_replays_lossless_snapshot_trace(
    tmp_path, monkeypatch, mutate_calibration,
):
    # Only this test uses a reduced grid. Production constants remain frozen.
    grid = (4, 8)
    for module in (selector, runtime, reader, trace_store):
        monkeypatch.setattr(module, "EXACT_FORWARD_BUDGETS", grid)
    original, _ = _calibrated_runner(tmp_path)
    calibration_path = tmp_path / "calibration.json"
    calibration_path.chmod(0o400)
    runner = _sealed(original, calibration_path)
    _, checkpoint, reference, _, _ = _fixture(tmp_path, monkeypatch)
    context, _ = runner._queries["query"]
    reference = replace(reference, query_id="query", representatives=tuple(
        replace(row, payload=replace(row.payload, query_context_sha256=context.audit_sha256))
        for row in reference.representatives
    ))
    config = replace(_config(reference), exact_forward_budgets=grid)
    if mutate_calibration:
        actual_search = adapter.run_v5_universal_one_click_inference

        def changed_after_search(*args, **kwargs):
            result = actual_search(*args, **kwargs)
            calibration_path.chmod(0o600)
            return result

        monkeypatch.setattr(adapter, "run_v5_universal_one_click_inference", changed_after_search)
    kwargs = dict(
        checkpoints=(checkpoint,), query_cohort=_cohort(reference), reference_sets=(reference,),
        config=config, equivalence_distance_matcher=lambda reference, candidate: 0.0,
        trace_runner=runner, method_id="engineering-integration-only",
        base_method_protocol_id=runner.protocol_id,
        base_method_protocol_sha256=runner.protocol_sha256, inference_seed=35,
        source_summary_artifact_sha256="c" * 64,
        output_root=tmp_path / "engineering-output", pre_publish_guard=lambda: "c" * 64,
        representative_selection_policy="budget_snapshot",
    )
    if mutate_calibration:
        with pytest.raises(ValueError, match="calibration must be read-only"):
            runtime.evaluate_v5_retained_checkpoints_on_tuning(**kwargs)
        assert not (tmp_path / "engineering-output" / "completion.json").exists()
        assert not tuple((tmp_path / "engineering-output" / "exact-traces").iterdir())
        return
    output = runtime.evaluate_v5_retained_checkpoints_on_tuning(**kwargs)
    replay = reader.read_v5_tuning_checkpoint_runtime_result(output.output_root)
    assert [row.sha256 for row in replay.evaluations] == [row.sha256 for row in output.evaluations]
    assert replay.lossless_emission_paths
    for path in (*replay.trace_paths, *replay.summary_paths,
                 *replay.lossless_emission_paths, replay.completion_path):
        assert stat.S_IMODE(path.stat().st_mode) == 0o400
        assert path.stat().st_nlink == 1
    trace = json.loads(replay.trace_paths[0].read_text())
    assert trace["exact_forward_call_budget"] == 8
    assert trace["representative_selection_policy"] == "budget_snapshot"
    assert trace["representative_snapshots"]
    assert replay.evaluations[0].paired_query_records[0].exact_forward_budgets == grid
