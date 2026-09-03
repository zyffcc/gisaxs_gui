from __future__ import annotations

from collections.abc import Sequence
import json
from pathlib import Path

import pytest

from utils.ML_Fitting_1D_GISAXS.PosteriorV8.launch_grouped_warmup_v5 import (
    LAUNCH_MANIFEST_FILENAME,
    V5CommandResult,
    V5WarmupLaunchConfig,
    V5WarmupLaunchError,
    build_v5_warmup_launch_plan,
    launch_v5_grouped_warmup,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.sobol_recipe_coordinates_v5 import (
    v5_sobol_recipe_design,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.split_design_v5 import (
    V5SplitCounts,
    V5SplitPlan,
)


_REQUIRED_SOURCE = (
    "build_formal_sobol_grouped_shard_v5.py",
    "train_grouped_v5.py",
    "grouped_warmup_launch_plan_v5.py",
    "launch_grouped_warmup_v5.py",
    "slurm/v5_grouped_warmup_dataset_cpu.sbatch",
    "slurm/v5_grouped_train_gpu4.sbatch",
)


def _source_root(tmp_path: Path) -> Path:
    root = tmp_path / "versioned-source-v5.1"
    bundle = root / "utils/ML_Fitting_1D_GISAXS/PosteriorV8"
    for relative in _REQUIRED_SOURCE:
        path = bundle / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"frozen source contract: {relative}\n", encoding="utf-8")
    (bundle / "support.py").write_text("VERSION = 'v5.1'\n", encoding="utf-8")
    domain = root / "src/gimap/features/fitting/domain"
    domain.mkdir(parents=True)
    for name in ("physical_constraints.py", "scattering_model.py"):
        (domain / name).write_text(f"frozen physics contract: {name}\n", encoding="utf-8")
    return root


def _config(
    tmp_path: Path,
    *,
    train_split: int = 20_000,
    validation_split: int = 5_000,
    train_recipes: int = 14_688,
    validation_recipes: int = 3_672,
    shard_size: int = 256,
) -> tuple[V5WarmupLaunchConfig, Path, Path]:
    dust = tmp_path / "data/dust/user/zhaiyufe"
    contracts = dust / "contracts/v5.1"
    contracts.mkdir(parents=True)
    plan = V5SplitPlan.create(
        V5SplitCounts(
            train=train_split,
            tuning_validation=validation_split,
            calibration=2,
            test=2,
            reference=2,
            ood_topology=1,
            ood_range_width=1,
            ood_weak_component=1,
            ood_acquisition_policy=1,
        ),
        guard_band=8,
    )
    split_path = contracts / "split-plan.json"
    design_path = contracts / "sobol-design.json"
    split_path.write_text(plan.to_json(), encoding="utf-8")
    design_path.write_text(
        v5_sobol_recipe_design(scramble_seed=20260903).to_json(),
        encoding="utf-8",
    )
    return (
        V5WarmupLaunchConfig(
            source_root=_source_root(tmp_path),
            run_root=dust / "runs/pilot-v5.1",
            split_plan=split_path,
            sobol_design=design_path,
            train_recipes=train_recipes,
            validation_recipes=validation_recipes,
            recipes_per_shard=shard_size,
            warmup_epochs=3,
        ),
        dust,
        split_path,
    )


def test_dry_run_exactly_covers_non_divisible_paper_pilot_counts(tmp_path):
    config, dust, _ = _config(tmp_path)
    result = launch_v5_grouped_warmup(config, allowed_root=dust)
    plan = result["plan"]
    train = plan["arrays"]["train"]
    validation = plan["arrays"]["tuning_validation"]

    assert result["status"] == "dry_run"
    assert result["writes_performed"] is False
    assert train["array_spec"] == "0-57"
    assert train["task_count"] == 58
    assert train["windows"][-1]["split_offset"] == 57 * 256
    assert train["windows"][-1]["recipe_count"] == 96
    assert sum(item["recipe_count"] for item in train["windows"]) == 14_688
    assert validation["array_spec"] == "0-14"
    assert validation["task_count"] == 15
    assert validation["windows"][-1]["recipe_count"] == 88
    assert sum(item["recipe_count"] for item in validation["windows"]) == 3_672
    assert plan["configuration"]["generating_candidate_only"] is False
    assert (
        plan["configuration"]["source_topology_candidate_catalog"]
        == "all_feasible_wire_branches"
    )
    assert not config.run_root.exists()


def test_plan_rejects_outside_dust_existing_output_and_bad_hash(tmp_path):
    config, dust, split_path = _config(tmp_path, train_recipes=4, validation_recipes=2)
    outside = V5WarmupLaunchConfig(
        **{**config.__dict__, "run_root": tmp_path / "outside-run"}
    )
    with pytest.raises(ValueError, match="must be under"):
        build_v5_warmup_launch_plan(outside, allowed_root=dust)

    config.run_root.mkdir(parents=True)
    with pytest.raises(FileExistsError, match="run root"):
        build_v5_warmup_launch_plan(config, allowed_root=dust)
    config.run_root.rmdir()

    payload = json.loads(split_path.read_text(encoding="utf-8"))
    payload["counts"]["train"] += 1
    split_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="does not reproduce"):
        build_v5_warmup_launch_plan(config, allowed_root=dust)


def test_submit_uses_two_arrays_and_afterok_then_writes_immutable_manifest(
    tmp_path,
    monkeypatch,
):
    config, dust, _ = _config(tmp_path, train_recipes=513, validation_recipes=257)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    calls: list[tuple[str, ...]] = []
    replies = iter(("24390001;maxwell", "24390002", "24390003"))

    def runner(argv: Sequence[str]) -> V5CommandResult:
        calls.append(tuple(argv))
        return V5CommandResult(0, next(replies) + "\n", "")

    manifest = launch_v5_grouped_warmup(
        config,
        submit=True,
        runner=runner,
        allowed_root=dust,
        hostname="max-wgs01.desy.de",
    )

    assert len(calls) == 3
    assert "--array=0-2" in calls[0]
    assert "--array=0-1" in calls[1]
    assert "--dependency=afterok:24390001:24390002" in calls[2]
    assert all(command[:2] == ("sbatch", "--parsable") for command in calls)
    assert all(any(value.startswith("--export=") for value in command) for command in calls)
    assert all("--export=ALL" not in command for command in calls)
    assert manifest["status"] == "submitted"
    assert manifest["job_ids"] == {
        "train_dataset": "24390001",
        "validation_dataset": "24390002",
        "gpu_warmup": "24390003",
    }
    assert manifest["cancellation_attempted"] is False
    required_hashes = manifest["plan"]["source"]["required_file_sha256"]
    assert "src/gimap/features/fitting/domain/scattering_model.py" in required_hashes
    assert len(manifest["plan"]["contracts"]["split_plan"]["contract_sha256"]) == 64
    stored = json.loads(
        (config.run_root / LAUNCH_MANIFEST_FILENAME).read_text(encoding="utf-8")
    )
    assert stored == manifest
    with pytest.raises(FileExistsError, match="run root"):
        launch_v5_grouped_warmup(
            config,
            submit=True,
            runner=runner,
            allowed_root=dust,
            hostname="max-wgs01",
        )


def test_failed_second_submission_preserves_first_job_and_audit_without_cancel(tmp_path, monkeypatch):
    config, dust, _ = _config(tmp_path, train_recipes=3, validation_recipes=2)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    replies = iter(
        (
            V5CommandResult(0, "24390100\n", ""),
            V5CommandResult(1, "", "Slurm controller unavailable"),
        )
    )

    with pytest.raises(V5WarmupLaunchError) as raised:
        launch_v5_grouped_warmup(
            config,
            submit=True,
            runner=lambda argv: next(replies),
            allowed_root=dust,
            hostname="max-wgs02",
        )

    stored = json.loads(raised.value.manifest_path.read_text(encoding="utf-8"))
    assert stored["status"] == "failed"
    assert stored["failure"]["stage"] == "validation_dataset"
    assert stored["job_ids"] == {"train_dataset": "24390100"}
    assert len(stored["submission_attempts"]) == 2
    assert stored["cancellation_attempted"] is False


def test_source_drift_after_first_submission_fails_closed_before_second(tmp_path, monkeypatch):
    config, dust, _ = _config(tmp_path, train_recipes=3, validation_recipes=2)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    calls = 0

    def runner(argv: Sequence[str]) -> V5CommandResult:
        nonlocal calls
        calls += 1
        support = config.source_root / "utils/ML_Fitting_1D_GISAXS/PosteriorV8/support.py"
        support.write_text("VERSION = 'mutated'\n", encoding="utf-8")
        return V5CommandResult(0, "24390200\n", "")

    with pytest.raises(V5WarmupLaunchError) as raised:
        launch_v5_grouped_warmup(
            config,
            submit=True,
            runner=runner,
            allowed_root=dust,
            hostname="max-wgs03",
        )

    stored = json.loads(raised.value.manifest_path.read_text(encoding="utf-8"))
    assert calls == 1
    assert stored["failure"]["stage"] == "validation_dataset"
    assert stored["job_ids"] == {"train_dataset": "24390200"}
    assert "changed during launch" in stored["failure"]["message"]


def test_submit_is_restricted_to_max_wgs_and_wrapper_audits_tail(tmp_path, monkeypatch):
    config, dust, _ = _config(tmp_path, train_recipes=3, validation_recipes=2)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(RuntimeError, match="max-wgs"):
        launch_v5_grouped_warmup(
            config,
            submit=True,
            runner=lambda argv: V5CommandResult(0, "1\n", ""),
            allowed_root=dust,
            hostname="local-mac",
        )
    assert not config.run_root.exists()

    wrapper = (
        Path(__file__).parents[1]
        / "PosteriorV8/slurm/v5_grouped_warmup_dataset_cpu.sbatch"
    ).read_text(encoding="utf-8")
    assert "max-wgs*" in wrapper
    assert "POSTERIOR_V8_V5_TOTAL_RECIPES" in wrapper
    assert "shard_count=$remaining" in wrapper
    assert '--start "$shard_start"' in wrapper
    assert '--count "$shard_count"' in wrapper
    assert "--include-unverified-branches" in wrapper
    assert "--shard-index" not in wrapper
