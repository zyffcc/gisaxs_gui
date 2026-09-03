from __future__ import annotations

import json

import pytest

from utils.ML_Fitting_1D_GISAXS.PosteriorV8 import solver_benchmark
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.solver_benchmark import (
    BenchmarkConfig,
    run_benchmark,
)


def _config(*, workers: int = 1, curves: int = 1) -> BenchmarkConfig:
    return BenchmarkConfig(
        mode="oracle_k1",
        noise_mode="clean",
        curves=curves,
        starts_per_branch=1,
        max_nfev=20,
        points=64,
        seed=20260902,
        workers=workers,
    )


def test_tiny_run_is_complete_and_refuses_to_overwrite(tmp_path):
    output_dir = tmp_path / "benchmark"
    summary = run_benchmark(_config(), output_dir)

    assert summary["schema"] == solver_benchmark.BENCHMARK_SCHEMA
    assert summary["evaluation_contract"] == {
        "audit_schema": solver_benchmark.EVALUATION_AUDIT_SCHEMA,
        "reference_matching_version": solver_benchmark.REFERENCE_MATCHING_VERSION,
        "reference_set_distance_version": (
            solver_benchmark.REFERENCE_SET_DISTANCE_VERSION
        ),
    }
    assert summary["complete_schedule_curve_fraction"] == 1.0
    accounting = summary["search_accounting_totals"]
    assert {key: accounting[key] for key in (
        "expected_branches",
        "generated_branches",
        "expected_attempts",
        "attempted_refinements",
        "returned_candidates",
    )} == {
        "expected_branches": 1,
        "generated_branches": 1,
        "expected_attempts": 1,
        "attempted_refinements": 1,
        "returned_candidates": 1,
    }
    assert accounting["optimizer_converged"] in {0, 1}
    assert accounting["amplitude_polish_attempted"] == 1
    assert accounting["amplitude_polish_completed"] == 1
    assert accounting["amplitude_polish_optimizer_converged"] in {0, 1}
    assert accounting["amplitude_polish_nfev"] >= 1
    assert accounting["amplitude_polish_residual_calls"] >= 1
    assert accounting["accepted_candidates"] in {0, 1}
    status = json.loads((output_dir / "run_status.json").read_text())
    curve = json.loads((output_dir / "curves" / "curve_00000.json").read_text())
    assert status["state"] == "COMPLETE"
    assert summary["headline"]["accepted_curve_fraction"] == float(
        curve["best_accepted"] is not None
    )
    attempt = curve["attempt_records"][0]
    assert attempt["amplitude_polish_applied"] is True
    assert (
        attempt["post_polish_raw_log_rmse"]
        <= attempt["final_raw_log_rmse"] + 1e-12
    )
    assert summary["amplitude_polish"]["completed_fraction"] == 1.0
    assert summary["amplitude_polish"]["raw_log_rmse_improvement"]["median"] >= -1e-12
    if curve["best_accepted"] is not None:
        assert curve["best_accepted"]["candidate_id"] in {
            item["candidate_id"]
            for item in curve["evaluation"]["candidates"]
            if item["accepted"]
        }

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        run_benchmark(_config(), output_dir)


def test_failed_curve_task_leaves_failed_status(tmp_path, monkeypatch):
    def fail_curve_task(config, curve_index):
        raise RuntimeError(f"deliberate failure for {curve_index}")

    monkeypatch.setattr(solver_benchmark, "_curve_task", fail_curve_task)
    output_dir = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="deliberate failure"):
        run_benchmark(_config(), output_dir)

    status = json.loads((output_dir / "run_status.json").read_text())
    assert status["state"] == "FAILED"
    assert status["error"] == "RuntimeError: deliberate failure for 0"
    assert not (output_dir / "summary.json").exists()


def test_scientific_payload_is_independent_of_worker_count(tmp_path):
    serial = run_benchmark(_config(workers=1, curves=2), tmp_path / "serial")
    parallel = run_benchmark(_config(workers=2, curves=2), tmp_path / "parallel")

    assert serial["scientific_payload_sha256"] == parallel[
        "scientific_payload_sha256"
    ]
