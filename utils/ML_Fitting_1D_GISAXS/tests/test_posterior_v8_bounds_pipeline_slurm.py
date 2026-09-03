from __future__ import annotations

from pathlib import Path


SLURM_DIR = Path(__file__).parents[1] / "PosteriorV8" / "slurm"


def test_v4_audit_wrapper_is_worker_only_and_fail_closed() -> None:
    source = (SLURM_DIR / "bounds_first_v4_audit_cpu.sbatch").read_text(
        encoding="utf-8"
    )

    for variable in (
        "SLURM_JOB_ID",
        "POSTERIOR_V8_SOURCE_ROOT",
        "POSTERIOR_V8_V4_DATASET_DIR",
        "POSTERIOR_V8_V4_AUDIT_OUTPUT",
    ):
        assert f"${{{variable}:?" in source
    assert "max-wgs*" in source
    assert "/data/dust/user/zhaiyufe/*" in source
    assert "build_bounds_first_shards audit" in source
    assert '--shards "${shards[@]}"' in source
    assert '--output "$POSTERIOR_V8_V4_AUDIT_OUTPUT"' in source


def test_v4_generation_and_v3_training_are_worker_only() -> None:
    for name in (
        "bounds_first_v4_dataset_cpu.sbatch",
        "bounds_proposal_v3_gpu4.sbatch",
    ):
        source = (SLURM_DIR / name).read_text(encoding="utf-8")
        assert "${SLURM_JOB_ID:?" in source
        assert "max-wgs*" in source
