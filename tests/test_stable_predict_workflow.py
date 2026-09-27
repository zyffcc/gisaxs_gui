"""Stable workflow boundary checks without loading TensorFlow or fitting."""

import json

import numpy as np
import pytest

from src.gimap.features.fitting.infrastructure.adapters import workflow_v5


def _native_payload(tmp_path):
    # Deliberately alternate signs and scramble magnitude; q=0 must not shift
    # per-observation count metadata when either side is prepared for inference.
    q = np.array([0, 4, -3, 2, -8, 7, -1, 1, -6, 8, -5, 3, -7, 5, -2, 6, -4]) * 0.1
    return dict(
        output_dir=str(tmp_path),
        q=q.tolist(),
        intensity=(np.arange(len(q)) + 10.0).tolist(),
        sigma=np.ones(len(q)).tolist(),
        observation_metadata=dict(
            source="native_cbf_columns",
            valid_pixel_counts=(np.arange(len(q)) + 1).tolist(),
            intensity_unit="counts_per_pixel",
            threshold_enabled=False,
            mirror_replaced_pixels=0,
            stack_count=1,
        ),
        options=dict(method="stable"),
    )


def _record_fits(monkeypatch):
    from src.gimap.features.fitting.infrastructure.adapters import stable_blue

    calls = []

    def fit(_engine, output, item, options, _report, _cancelled):
        calls.append((item, options, output))
        return [dict(side=item["side"], best_log_rmse=0.2)]

    def forbidden_legacy(*_args, **_kwargs):
        raise AssertionError("Stable routing loaded the frozen TensorFlow engine")

    monkeypatch.setattr(stable_blue.StableEngine, "fit_and_write", fit)
    monkeypatch.setattr(workflow_v5, "WorkflowEngine", forbidden_legacy)
    return calls


def test_native_count_metadata_tracks_sorted_sides_and_q_zero(tmp_path, monkeypatch):
    calls = _record_fits(monkeypatch)
    payload = _native_payload(tmp_path)
    result = workflow_v5.run_workflow_job(payload, lambda *_a: None, lambda: False)
    assert len(calls) == 2 and result["records"][0]["status"] == "complete"
    for item, options, _output in calls:
        indices = item["indices"]
        np.testing.assert_array_equal(
            item["count"],
            np.asarray(payload["observation_metadata"]["valid_pixel_counts"])[indices],
        )
        np.testing.assert_array_equal(item["observed"], np.asarray(payload["intensity"])[indices])
        assert options["method"] == "stable"
        assert np.all(np.diff(item["q"]) > 0)
    persisted = json.loads((tmp_path / "request.json").read_text())
    assert persisted["observation_metadata"] == payload["observation_metadata"]


def test_text_batch_does_not_inherit_native_pixel_counts(tmp_path, monkeypatch):
    calls = _record_fits(monkeypatch)
    payload = _native_payload(tmp_path / "result")
    curve = tmp_path / "curve.txt"
    np.savetxt(curve, np.column_stack([payload[key] for key in ("q", "intensity", "sigma")]))
    payload["files"] = [str(curve)]
    workflow_v5.run_workflow_job(payload, lambda *_a: None, lambda: False)
    assert len(calls) == 2
    assert all(item.get("count") is None for item, _options, _output in calls)


@pytest.mark.parametrize("counts", [[1, 2], [0] * 17, [-1] * 17])
def test_malformed_native_count_metadata_is_rejected(tmp_path, monkeypatch, counts):
    calls = _record_fits(monkeypatch)
    payload = _native_payload(tmp_path)
    payload["observation_metadata"]["valid_pixel_counts"] = counts
    with pytest.raises(ValueError, match="count|Count|pixel|Pixel"):
        workflow_v5.run_workflow_job(payload, lambda *_a: None, lambda: False)
    assert not calls


def test_fixed_constraints_survive_stable_job_boundary(tmp_path, monkeypatch):
    calls = _record_fits(monkeypatch)
    payload = _native_payload(tmp_path)
    payload["options"].update(components=[2, 2], sigma_res=0.02, nu_res=2.5)
    workflow_v5.run_workflow_job(payload, lambda *_a: None, lambda: False)
    assert len(calls) == 2
    for _item, options, _output in calls:
        assert options["components"] == [2, 2]
        assert options["sigma_res"] == 0.02 and options["nu_res"] == 2.5
