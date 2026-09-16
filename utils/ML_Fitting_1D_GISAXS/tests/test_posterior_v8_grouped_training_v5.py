from __future__ import annotations

# ruff: noqa: E402 -- skip cleanly when the optional TensorFlow runtime is absent.

from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pytest


tf = pytest.importorskip("tensorflow")

from utils.ML_Fitting_1D_GISAXS.PosteriorV8.build_grouped_dataset_v5 import (
    V5GroupedRecipeSpec,
    build_tiny_v5_grouped_dataset,
    build_v5_grouped_solution_dataset,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.candidate_supervision_v5 import (
    SEARCH_OUTCOME_CODE,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.grouped_dataset_v5 import (
    write_v5_grouped_dataset,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.grouped_training_data_v5 import (
    RecipeMacroAdapter,
    V5_GROUPED_TRAINING_CONFIG_SCHEMA,
    V5_GROUPED_TRAINING_CONFIG_VERSION,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.grouped_training_v5 import (
    BEST_MODEL_FILE,
    BEST_MODEL_ROLE,
    FULL_CHECKPOINT_DIRECTORY,
    FULL_CHECKPOINT_SELECTION_STATUS,
    LAST_MODEL_FILE,
    RESULT_MANIFEST_FILE,
    V5_GROUPED_TRAINER_SCHEMA,
    V5_GROUPED_TRAINER_VERSION,
    V5JobLocalInputCapability,
    V5GroupedTrainingConfig,
    _validate_job_local_input_capability,
    inspect_v5_grouped_training,
    train_v5_grouped_model,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.model_v5 import (
    validate_model_v5_graph_contract,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.train_grouped_v5 import main
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.synthetic_recipe_v5 import (
    sample_v5_clean_recipe,
)


def _artifact(tmp_path: Path) -> Path:
    dataset = build_tiny_v5_grouped_dataset(
        recipe_count=1,
        split_id="train",
        generating_only=False,
    )
    path = tmp_path / "tiny.gvd5"
    write_v5_grouped_dataset(dataset, path)
    return path


def _smoke_config(**updates) -> V5GroupedTrainingConfig:
    values = {
        "warmup_epochs": 1,
        "full_epochs": 0,
        "recipes_per_replica": 1,
        "validation_recipes_per_batch": 1,
        "steps_per_epoch": 1,
        "width": 8,
        "encoder_blocks": 1,
        "mixture_components": 12,
        "train_split": "train",
        "validation_split": "train",
        "allow_same_split_for_smoke": True,
    }
    values.update(updates)
    return V5GroupedTrainingConfig(**values)


def _formal_artifact(
    tmp_path: Path,
    split: str,
    index: int,
    *,
    sobol_index: int | None = None,
    design_sha: str = "e" * 64,
) -> Path:
    split_sha = "d" * 64
    recipe = sample_v5_clean_recipe(("sphere",), recipe_seed=20260930 + index, pattern_id=0)
    spec = V5GroupedRecipeSpec(
        recipe=recipe,
        split_id=split,
        sobol_index=100 + index if sobol_index is None else sobol_index,
        split_plan_sha256=split_sha,
        sobol_design_sha256=design_sha,
    )
    dataset = build_v5_grouped_solution_dataset(
        (spec,), dataset_id=f"formal-{split}-{index}", generating_only=False
    )
    path = tmp_path / f"{split}-{index}-{design_sha[:2]}-{sobol_index}.gvd5"
    write_v5_grouped_dataset(dataset, path)
    return path


def _formal_artifacts(tmp_path: Path) -> tuple[Path, Path]:
    return (
        _formal_artifact(tmp_path, "train", 0),
        _formal_artifact(tmp_path, "tuning_validation", 1),
    )


def test_recipe_macro_adapter_keeps_complete_groups_and_unverified_state(tmp_path):
    path = _artifact(tmp_path)
    dataset, audit = inspect_v5_grouped_training(path, path, _smoke_config(), replicas=1)
    adapter = RecipeMacroAdapter(dataset.shards[0].dataset)

    _, warmup = adapter.numpy_batch((0,), phase="warmup")
    _, full = adapter.numpy_batch((0,), phase="full")

    assert warmup["search_outcome_code"].tolist() == [SEARCH_OUTCOME_CODE["compatible_found"]]
    assert SEARCH_OUTCOME_CODE["unverified"] in full["search_outcome_code"]
    assert (
        SEARCH_OUTCOME_CODE["no_compatible_found_within_frozen_search_budget"]
        not in full["search_outcome_code"]
    )
    assert audit.train_outcome_counts["unverified"] > 0
    assert not audit.full_stage_permitted
    assert audit.global_recipes_per_step == 1


def test_full_stage_refuses_solution_only_artifact_instead_of_inventing_negatives(tmp_path):
    path = _artifact(tmp_path)
    config = _smoke_config(warmup_epochs=0, full_epochs=1)

    with pytest.raises(ValueError, match="unverified branches cannot be negatives"):
        inspect_v5_grouped_training(path, path, config, replicas=1)


def test_cli_dry_run_validates_without_creating_output(tmp_path, capsys):
    train_path, validation_path = _formal_artifacts(tmp_path)
    output = tmp_path / "dry-run-output"

    assert (
        main(
            (
                "--train-dataset",
                str(train_path),
                "--validation-dataset",
                str(validation_path),
                "--output-dir",
                str(output),
                "--smoke",
                "--dry-run",
            )
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["dry_run"] is True
    assert payload["training_audit"]["unverified_rows_are_negative"] is False
    assert not output.exists()


@pytest.mark.parametrize("hostname", ["max-wgs01.desy.de", "max-fs-display006.desy.de"])
def test_cli_dry_run_rejects_maxwell_login_node_before_reading_artifacts(
    tmp_path, monkeypatch, hostname
):
    from utils.ML_Fitting_1D_GISAXS.PosteriorV8 import train_grouped_v5

    monkeypatch.setattr(
        train_grouped_v5.socket, "gethostname", lambda: hostname
    )
    with pytest.raises(RuntimeError, match="including --dry-run"):
        main(
            (
                "--train-dataset",
                str(tmp_path / "missing-train.gvd5"),
                "--validation-dataset",
                str(tmp_path / "missing-validation.gvd5"),
                "--output-dir",
                str(tmp_path / "output"),
                "--dry-run",
            )
        )
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("hostname", ["max-wgs01.desy.de", "max-fs-display006.desy.de"])
def test_direct_trainer_rejects_login_even_with_slurm_environment(tmp_path, monkeypatch, hostname):
    from utils.ML_Fitting_1D_GISAXS.PosteriorV8 import grouped_training_v5

    monkeypatch.setattr(grouped_training_v5.socket, "gethostname", lambda: hostname)
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    with pytest.raises(RuntimeError, match="forbidden on the Maxwell login node"):
        train_v5_grouped_model(tmp_path / "missing-train", tmp_path / "missing-validation",
                               tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_multiple_shards_remain_separate_and_recipe_indices_are_globally_unique(tmp_path):
    first = _formal_artifact(tmp_path, "train", 0)
    second = _formal_artifact(tmp_path, "train", 2)
    validation = _formal_artifact(tmp_path, "tuning_validation", 1)
    config = V5GroupedTrainingConfig(
        warmup_epochs=1,
        full_epochs=0,
        recipes_per_replica=1,
        steps_per_epoch=1,
    )

    collection, audit = inspect_v5_grouped_training(
        (first, second), (validation,), config, replicas=1
    )
    _, labels = collection.numpy_batch(collection.train_recipes, phase="warmup")

    assert audit.train_recipe_count == 2
    assert len(collection.shards) == 3
    assert len(np.unique(labels["clean_recipe_index"])) == 2
    assert audit.split_plan_sha256 == "d" * 64
    assert audit.sobol_design_sha256 == "e" * 64


def test_formal_shard_roles_and_path_overlap_fail_closed(tmp_path):
    train, validation = _formal_artifacts(tmp_path)
    config = V5GroupedTrainingConfig(
        warmup_epochs=1,
        full_epochs=0,
        recipes_per_replica=1,
        steps_per_epoch=1,
    )

    with pytest.raises(ValueError, match="clean split_id"):
        inspect_v5_grouped_training(validation, train, config, replicas=1)
    with pytest.raises(ValueError, match="paths overlap"):
        inspect_v5_grouped_training(train, train, config, replicas=1)


def test_formal_sobol_design_group_and_index_identity_fail_closed(tmp_path):
    train = _formal_artifact(tmp_path, "train", 0)
    same_group = _formal_artifact(tmp_path, "tuning_validation", 0)
    same_index = _formal_artifact(tmp_path, "tuning_validation", 1, sobol_index=100)
    other_design = _formal_artifact(tmp_path, "tuning_validation", 2, design_sha="f" * 64)
    config = V5GroupedTrainingConfig(
        warmup_epochs=1,
        full_epochs=0,
        recipes_per_replica=1,
        steps_per_epoch=1,
    )

    with pytest.raises(ValueError, match="clean_group_id overlaps"):
        inspect_v5_grouped_training(train, same_group, config, replicas=1)
    with pytest.raises(ValueError, match="Sobol index overlaps"):
        inspect_v5_grouped_training(train, same_index, config, replicas=1)
    with pytest.raises(ValueError, match="different Sobol plan or design"):
        inspect_v5_grouped_training(train, other_design, config, replicas=1)


def test_warmup_only_publishes_no_full_checkpoint_and_is_not_paper_model(tmp_path):
    path = _artifact(tmp_path)
    output = tmp_path / "run-v5.2"
    result = train_v5_grouped_model(
        path,
        path,
        output,
        _smoke_config(),
        strategy=tf.distribute.OneDeviceStrategy("/cpu:0"),
    )

    assert result.best_model_path.name == BEST_MODEL_FILE
    assert result.last_model_path.name == LAST_MODEL_FILE
    manifest = json.loads((output / RESULT_MANIFEST_FILE).read_text(encoding="utf-8"))
    history = json.loads(result.history_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["schema"] == V5_GROUPED_TRAINER_SCHEMA
    assert manifest["version"] == V5_GROUPED_TRAINER_VERSION
    assert manifest["best_model_role"] == BEST_MODEL_ROLE
    assert manifest["full_checkpoints"] == []
    assert history["full_checkpoints"] == []
    assert result.full_checkpoint_paths == ()
    assert not (output / FULL_CHECKPOINT_DIRECTORY).exists()
    assert result.checkpoint_selection_status == FULL_CHECKPOINT_SELECTION_STATUS
    assert result.paper_model_eligible is False
    assert manifest["checkpoint_selection"]["status"] == (
        FULL_CHECKPOINT_SELECTION_STATUS
    )
    assert manifest["paper_model_status"] == history["paper_model_status"]
    assert manifest["paper_model_status"]["run_scope"] == "engineering_warmup_only"
    assert manifest["paper_model_status"]["paper_model_eligible"] is False
    assert manifest["paper_model_status"]["operational_ranking_eligible"] is False
    assert manifest["paper_model_status"]["warmup_only_ranking_eligible"] is False
    assert "full_epochs_is_zero_warmup_only_engineering_run" in manifest[
        "paper_model_status"
    ]["paper_model_ineligibility_reasons"]
    assert len(manifest["input_artifacts"]) == 2
    assert all(len(value["artifact_sha256"]) == 64 for value in manifest["input_artifacts"])
    assert len(manifest["source_sha256"]) >= 8
    assert result.best_epoch == 1
    assert np.isfinite(result.best_validation_loss)
    for model_path in (result.best_model_path, result.last_model_path):
        validate_model_v5_graph_contract(tf.keras.models.load_model(model_path, compile=False))

    with pytest.raises(FileExistsError, match="overwrite"):
        train_v5_grouped_model(
            path,
            path,
            output,
            _smoke_config(),
            strategy=tf.distribute.OneDeviceStrategy("/cpu:0"),
        )


def test_grouped_config_binds_frozen_proposal_policy_and_warmup_ranking_status():
    config = V5GroupedTrainingConfig(warmup_epochs=1)
    payload = config.audit_payload()
    assert payload["schema"] == V5_GROUPED_TRAINING_CONFIG_SCHEMA
    assert payload["version"] == V5_GROUPED_TRAINING_CONFIG_VERSION
    assert payload["mixture_components"] == 12
    assert payload["proposal_execution_policy"]["policy"]["per_branch_top_l"] == 4
    assert payload["ranking_qualification"] == {
        "warmup_only_eligible": False,
        "configured_full_ranking_phase": False,
        "eligible_from_configuration_alone": False,
        "status": "ineligible_without_full_ranking_phase",
    }
    with pytest.raises(ValueError, match="mixture_components"):
        V5GroupedTrainingConfig(warmup_epochs=1, mixture_components=11)
    with pytest.raises(ValueError, match="proposal_execution_policy_sha256"):
        V5GroupedTrainingConfig(
            warmup_epochs=1,
            proposal_execution_policy_sha256="0" * 64,
        )


def test_slurm_runtime_refuses_non_dust_data_and_output(tmp_path, monkeypatch):
    path = _artifact(tmp_path)
    monkeypatch.setenv("SLURM_JOB_ID", "test-job")

    with pytest.raises(ValueError, match="must be under /data/dust/user/zhaiyufe"):
        train_v5_grouped_model(
            path,
            path,
            tmp_path / "run",
            _smoke_config(),
            strategy=tf.distribute.OneDeviceStrategy("/cpu:0"),
        )


def test_job_local_capability_remains_fail_closed(tmp_path, monkeypatch):
    local = _artifact(tmp_path)
    monkeypatch.setenv("SLURM_JOB_ID", "991")
    with pytest.raises(TypeError, match="opaque"):
        V5JobLocalInputCapability()
    forged = object.__new__(V5JobLocalInputCapability)
    forged._nonce = b"serialized-audit-substitution"

    with pytest.raises(RuntimeError, match="not minted"):
        _validate_job_local_input_capability(
            (local.resolve(),), forged, phase="pre_training"
        )
    with pytest.raises(ValueError, match="Slurm output"):
        train_v5_grouped_model(
            local,
            local,
            tmp_path / "non-dust-output",
            _smoke_config(),
            strategy=tf.distribute.OneDeviceStrategy("/cpu:0"),
            job_local_input_capability=forged,
        )

def test_slurm_wrapper_is_worker_only_dust_scoped_and_has_no_overwrite_switch():
    path = Path(__file__).parents[1] / "PosteriorV8" / "slurm" / "v5_grouped_train_gpu4.sbatch"
    source = path.read_text(encoding="utf-8")

    for variable in (
        "SLURM_JOB_ID",
        "POSTERIOR_V8_SOURCE_ROOT",
        "POSTERIOR_V8_V5_TRAIN_DATASETS",
        "POSTERIOR_V8_V5_VALIDATION_DATASETS",
        "POSTERIOR_V8_V5_GROUPED_OUTPUT",
    ):
        assert f"${{{variable}:?" in source
    assert "max-wgs*" in source
    assert "max-fs-display*" in source
    assert "/data/dust/user/zhaiyufe/" in source
    assert "--dry-run" in source and "--smoke" in source
    assert "--overwrite" not in source
