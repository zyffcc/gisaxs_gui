from pathlib import Path, PurePosixPath

import pytest

from utils.ML_Fitting_1D_GISAXS.PosteriorV8 import (
    author_k1_balanced_full_search_plan_v5 as author,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.k1_training_chain_plan_v5 import (
    K1_TRAINING_REQUIRED_SOURCE_FILES,
)


def test_plan_authoring_context_is_worker_only():
    author._assert_worker_context(  # noqa: SLF001
        {"SLURM_JOB_ID": "123"}, hostname="max-wn001.desy.de"
    )
    for hostname in ("max-wgs", "max-wgs002", "max-fs-display006.desy.de"):
        with pytest.raises(RuntimeError, match="login host"):
            author._assert_worker_context(  # noqa: SLF001
                {"SLURM_JOB_ID": "123"}, hostname=hostname
            )
    with pytest.raises(RuntimeError, match="through Slurm"):
        author._assert_worker_context({}, hostname="max-wn001")  # noqa: SLF001
    with pytest.raises(RuntimeError, match="cannot be an array"):
        author._assert_worker_context(  # noqa: SLF001
            {"SLURM_JOB_ID": "123", "SLURM_ARRAY_TASK_ID": "0"},
            hostname="max-wn001",
        )


def test_plan_authoring_budget_and_source_inventory_are_frozen():
    assert author.V5_K1_BALANCED_FULL_SEARCH_DIRECT_SCOUT_COUNT == 16
    assert author.V5_K1_BALANCED_FULL_SEARCH_PER_SEED_FORWARD_LIMIT == 256
    assert author.V5_K1_BALANCED_FULL_SEARCH_EXACT_BUDGET == 4096
    required = {value.as_posix() for value in K1_TRAINING_REQUIRED_SOURCE_FILES}
    assert (
        "utils/ML_Fitting_1D_GISAXS/PosteriorV8/"
        "author_k1_balanced_full_search_plan_v5.py"
    ) in required
    assert (
        "utils/ML_Fitting_1D_GISAXS/PosteriorV8/slurm/"
        "v5_k1_balanced_full_search_plan_author_cpu.sbatch"
    ) in required


def test_plan_authoring_main_constructs_config_from_cli(monkeypatch, tmp_path, capsys):
    captured = []
    monkeypatch.setattr(author, "_assert_worker_context", lambda: None)

    def capture(config):
        captured.append(config)
        return {"status": "TEST_ONLY"}

    monkeypatch.setattr(author, "author_v5_k1_balanced_full_search_plan", capture)
    arguments = []
    for action in author._parser()._actions:  # noqa: SLF001
        if action.required:
            value = str(tmp_path / action.dest) if action.type is Path else "a" * 64
            arguments.extend([action.option_strings[0], value])
    assert author.main(arguments) == 0
    assert len(captured) == 1
    assert isinstance(captured[0], author.V5K1BalancedFullSearchAuthoringConfig)
    assert captured[0].balanced_dataset_launch_plan_path == (
        tmp_path / "balanced_dataset_launch_plan_path"
    )
    assert "TEST_ONLY" in capsys.readouterr().out


def test_plan_authoring_cli_paths_bind_to_config_field_names():
    actions = {
        action.option_strings[0]: action.dest
        for action in author._parser()._actions  # noqa: SLF001
        if action.option_strings
    }
    assert actions["--source-root"] == "source_root"
    assert actions["--source-archive"] == "source_archive"
    assert (
        actions["--balanced-dataset-launch-plan"]
        == "balanced_dataset_launch_plan_path"
    )
    assert actions["--balanced-dataset-completion"] == "balanced_dataset_completion_path"
    assert actions["--train-tuning-receipt"] == "train_tuning_receipt_path"
    assert actions["--phase-c-completion"] == "phase_c_completion_path"
    assert actions["--three-way-receipt"] == "three_way_receipt_path"
    assert actions["--local-sobol-schedule"] == "local_sobol_schedule_path"
    assert actions["--calibration"] == "calibration_path"
    assert actions["--search-run-root"] == "search_run_root"
    assert actions["--candidate-plan"] == "candidate_plan_path"
    assert actions["--completion"] == "completion_path"


@pytest.mark.parametrize("root_type", [Path, PurePosixPath])
def test_plan_authoring_output_is_exclusive_and_read_only(tmp_path, root_type):
    target = tmp_path / "audit" / "candidate.json"
    target.parent.mkdir()
    identity = author._write_read_only_json(  # noqa: SLF001
        target, {"value": 1}, allowed_root=root_type(tmp_path)
    )
    assert identity["mode_octal"] == "0400"
    assert identity["nlink"] == 1
    assert Path(identity["path"]) == target
    with pytest.raises(FileExistsError):
        author._write_read_only_json(  # noqa: SLF001
            target, {"value": 2}, allowed_root=root_type(tmp_path)
        )
