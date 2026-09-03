from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from utils.ML_Fitting_1D_GISAXS.PosteriorV8 import gold_solver_benchmark
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.gold_solver_benchmark import (
    GOLD_BENCHMARK_SCHEMA,
    GoldBenchmarkConfig,
    run_gold_benchmark,
)


def _smoke_config():
    return GoldBenchmarkConfig(
        curves=1,
        component_schedule=(1,),
        noise_mode="clean",
        points=64,
        seed=20260902,
        sequential_max_nfev=8,
        amplitude_polish_max_nfev=8,
        joint_max_exact_evaluations=24,
    )


def test_smoke_audit_compares_identical_curve_branch_seed_and_metric(tmp_path):
    output = tmp_path / "gold-audit.json"
    payload = run_gold_benchmark(_smoke_config(), output)
    stored = json.loads(output.read_text(encoding="utf-8"))

    assert stored["schema"] == GOLD_BENCHMARK_SCHEMA
    assert len(stored["source_sha256"]) >= 10
    assert len(stored["source_sha256_aggregate"]) == 64
    assert payload["summary"]["case_count"] == 1
    assert stored["summary"]["paired_return_count"] == 1
    case = stored["cases"][0]
    assert len(case["curve"]["curve_sha256"]) == 64
    assert len(case["comparison_input_id"]) == 64
    assert case["curve"]["metric_name"] == "raw_natural_log_rmse"
    assert case["branch"]["codec_version"]
    assert case["branch"]["active_unit_coordinates"]
    assert case["sequential"]["returned"]
    assert case["joint"]["returned"]
    assert case["comparison"]["initial_metric_absolute_delta"] < 1e-12

    for method in (case["sequential"], case["joint"]):
        assert method["final_raw_log_rmse"] >= 0.0
        assert method["final_standardized_log_rmse"] is None
        assert method["exact_forward_calls"] >= 1
        assert method["wall_seconds"] >= 0.0
        assert method["bounds_satisfied"]
        assert method["best_seen"]
    assert case["sequential"]["amplitude_bounds"]["lower"][0] == 0.0
    assert (
        case["joint"]["exact_forward_calls"]
        <= case["joint"]["max_exact_evaluations"]
    )

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        run_gold_benchmark(_smoke_config(), output)


def test_cli_schedule_and_config_reject_non_v8_component_counts():
    assert gold_solver_benchmark._schedule("1, 3,4") == (1, 3, 4)
    with pytest.raises(argparse.ArgumentTypeError, match="1..4"):
        gold_solver_benchmark._schedule("1,5")
    with pytest.raises(ValueError, match="1..4"):
        GoldBenchmarkConfig(component_schedule=(0,))


def test_cpu_slurm_wrapper_is_smoke_by_default_and_exposes_formal_overrides():
    source = (
        Path(__file__).parents[1]
        / "PosteriorV8"
        / "slurm"
        / "gold_benchmark_cpu.sbatch"
    ).read_text(encoding="utf-8")

    assert "PosteriorV8.gold_solver_benchmark" in source
    assert "POSTERIOR_V8_GOLD_OUTPUT:?" in source
    assert "POSTERIOR_V8_GOLD_CURVES:-1" in source
    assert "POSTERIOR_V8_GOLD_POINTS:-64" in source
    assert "POSTERIOR_V8_GOLD_K_SCHEDULE:-2" in source
    assert "POSTERIOR_V8_GOLD_JOINT_MAX_EXACT:-64" in source
