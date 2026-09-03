from __future__ import annotations

from pathlib import Path


SLURM_DIR = Path(__file__).parents[1] / "PosteriorV8" / "slurm"


def test_dataset_slurm_wrapper_tracks_current_cli_contract():
    source = (SLURM_DIR / "dataset_pilot_cpu.sbatch").read_text(encoding="utf-8")

    for required in (
        "--recipe-count",
        "--views-per-recipe",
        "--max-raw-points",
        "--topology-schedule",
    ):
        assert required in source
    for removed in ("--sample-count", "--noise-mode", "--raw-points", "--split"):
        assert removed not in source
    assert "SLURM_ARRAY_TASK_ID" in source
    assert "shard_index * recipe_count" in source


def test_benchmark_slurm_wrapper_exposes_exact_amplitude_polish_budget():
    source = (SLURM_DIR / "benchmark_cpu.sbatch").read_text(encoding="utf-8")

    assert "--amplitude-polish-max-nfev" in source
    assert "POSTERIOR_V8_AMPLITUDE_POLISH_MAX_NFEV" in source


def test_heavy_wrappers_require_dust_output_paths_from_submitter():
    for name, variable in (
        ("dataset_pilot_cpu.sbatch", "POSTERIOR_V8_OUTPUT_DIR"),
        ("calibration_cpu.sbatch", "POSTERIOR_V8_CALIBRATION_OUTPUT"),
        ("reference_bank_cpu.sbatch", "POSTERIOR_V8_REFERENCE_OUTPUT"),
        ("training_gpu4.sbatch", "POSTERIOR_V8_TRAINING_OUTPUT"),
    ):
        source = (SLURM_DIR / name).read_text(encoding="utf-8")
        assert f'${{{variable}:?set {variable}}}' in source


def test_training_wrapper_uses_current_proposal_cli_without_implicit_overwrite():
    source = (SLURM_DIR / "training_gpu4.sbatch").read_text(encoding="utf-8")

    assert "PosteriorV8.train_proposal" in source
    assert "--global-batch-size" in source
    assert "--mixture-components" in source
    assert "--mixed-precision" in source
    assert "--overwrite" not in source
