"""The synthetic online benchmark times a real restart and reports both timing denominators."""

import json
from pathlib import Path
from statistics import mean

import pytest

from tools.benchmarks.insitu_synthetic import benchmark


def _check_times(detector, stage_start):
    samples = detector["per_frame"]
    assert len(samples) == detector["frames"]
    assert all(row["fit_points"] > 0 for row in samples)
    assert samples[stage_start - 1]["stage_start"]
    assert samples[stage_start - 1]["predict_s"] > 0
    for row in samples:
        assert bool(row["chain_break"]) == (row["predict_s"] > 0)
        assert all(row[name] > 0 for name in ("reduction_s", "cut_s", "chained_fit_s"))
    for name, seconds in detector["mean_seconds_per_frame"].items():
        assert seconds == pytest.approx(mean(row[name] for row in samples))
    calls = [row["predict_s"] for row in samples if row["predict_s"] > 0]
    assert detector["predict_calls"] == len(calls)
    assert detector["mean_seconds_per_predict_call"] == pytest.approx(mean(calls))
    assert len(detector["network"]["sha256"]) == 64


def test_short_synthetic_run_reduces_frames_and_predicts_at_the_stage_change(tmp_path):
    report = benchmark(tmp_path, [1], 2)
    _check_times(report["detectors"][0], 2)
    assert not list(tmp_path.glob("*.tif"))


def test_recorded_online_benchmark_covers_both_detector_sizes():
    path = Path(__file__).resolve().parents[1] / "tools/benchmarks/insitu_online.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["cpu"] and report["settings"]["workers"] == 1
    assert [r["shape"] for r in report["detectors"]] == [[1000, 1000], [2000, 2000]]
    for detector in report["detectors"]:
        _check_times(detector, report["settings"]["stage_start_frame"])
