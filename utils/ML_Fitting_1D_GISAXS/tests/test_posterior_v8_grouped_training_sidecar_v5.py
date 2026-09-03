from __future__ import annotations

# ruff: noqa: E402 -- skip cleanly when the optional TensorFlow runtime is absent.

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pytest


tf = pytest.importorskip("tensorflow")

from utils.ML_Fitting_1D_GISAXS.PosteriorV8 import (
    grouped_training_inventory_v5,
    grouped_training_shards_v5,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.amplitude_query_v5 import V5AmplitudeQuery
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.bounds_query_sampling_v5 import (
    full_range_v5_query,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.build_grouped_dataset_v5 import (
    V5GroupedRecipeSpec,
    build_v5_grouped_solution_dataset,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.contract import ClosedInterval
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.evaluation import (
    RAW_LOG_RMSE_METRIC,
    STANDARDIZED_LOG_RMSE_METRIC,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.grouped_dataset_v5 import (
    observation_array,
    write_v5_grouped_dataset,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.grouped_training_v5 import (
    FULL_CHECKPOINT_SELECTION_STATUS,
    V5GroupedTrainingConfig,
    inspect_v5_grouped_training,
    train_v5_grouped_model,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.observation_v5 import (
    build_v5_observation_data_views,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.search_supervision_contract_v5 import (
    V5CompatibleRepresentativeReference,
    V5ExactSearchObservation,
    V5FrozenBranchSearchResult,
    V5FrozenExactSearchProtocol,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.search_evidence_receipt_v5 import (
    V5SearchEvidenceReceipt,
    V5_SEARCH_LABEL_PURPOSE_PILOT,
    V5_SEARCH_LABEL_PURPOSE_TRAINING,
    evidence_receipt_path_for_sidecar,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.search_supervision_sidecar_v5 import (
    V5UniversalSearchSpec,
    collect_v5_search_supervision_sidecar,
    write_v5_search_supervision_sidecar,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.synthetic_recipe_v5 import (
    sample_v5_clean_recipe,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.train_grouped_v5 import main
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.universal_query_v5 import (
    V5TopologyQuery,
    build_v5_universal_candidate_context,
)


def _hash(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def _protocol(identity: str = "shared") -> V5FrozenExactSearchProtocol:
    return V5FrozenExactSearchProtocol(
        protocol_id=f"v5-trainer-sidecar-test/{identity}",
        evaluator_version="authoritative-forward-evaluator/test",
        authoritative_forward_id="gui-empirical-forward/test",
        metric_name=STANDARDIZED_LOG_RMSE_METRIC,
        threshold_name="standardized-compatible",
        threshold_value=0.02,
        threshold_source_id="paper-protocol/test",
        missing_acceptance_sigma_metric_name=RAW_LOG_RMSE_METRIC,
        missing_acceptance_sigma_threshold_name="raw-compatible",
        missing_acceptance_sigma_threshold_value=0.02,
        exact_forward_call_budget=8,
        seed_schedule_id="fixed-sobol/test",
        seed_schedule_sha256=_hash("fixed-sobol/test"),
        optimizer_schedule_id="bounded-refinement/test",
        optimizer_schedule_sha256=_hash("bounded-refinement/test"),
        termination_policy_id="positive-or-exhaust/test",
        representative_distance_id="normalized-parameter/test",
        representative_distance_sha256=_hash("normalized-parameter/test"),
        delta_separation_threshold=0.08,
    )


def _amplitude_query(query) -> V5AmplitudeQuery:
    return V5AmplitudeQuery.create(
        background=ClosedInterval(0.0, 1.0e8),
        k=ClosedInterval(1.0e-2, 1.0e8),
        component_intensities=tuple(
            ClosedInterval(0.0, 1.0) for _ in query.topology
        ),
        resolution_presence_policy=query.resolution_presence_policy,
        int_res=ClosedInterval(0.0, 1.0e8),
    )


def _parent_sidecar(
    tmp_path: Path,
    *,
    split: str,
    index: int,
    with_sidecar: bool,
    protocol: V5FrozenExactSearchProtocol | None = None,
    unverified: bool = False,
    multiple_targets: bool = False,
) -> tuple[Path, Path | None]:
    seed = 41000 + index
    recipe = sample_v5_clean_recipe(("sphere",), recipe_seed=seed, pattern_id=0)
    dataset = build_v5_grouped_solution_dataset(
        (
            V5GroupedRecipeSpec(
                recipe=recipe,
                split_id=split,
                view_indices=(0,),
                sobol_index=100 + index,
                split_plan_sha256="d" * 64,
                sobol_design_sha256="e" * 64,
            ),
        ),
        dataset_id=f"trainer-sidecar-parent-{split}-{index}",
        generating_only=False,
    )
    parent = tmp_path / f"parent-{split}-{index}.gvd5"
    write_v5_grouped_dataset(dataset, parent)
    if not with_sidecar:
        return parent, None

    competitor = full_range_v5_query(("cylinder",), query_seed=seed + 1000)
    observation = build_v5_observation_data_views(
        recipe, (0,), split_id=split
    )[0]
    context = build_v5_universal_candidate_context(
        observation.preprocessed,
        observation.uncertainty,
        (
            V5TopologyQuery(recipe.query, recipe.amplitude_query),
            V5TopologyQuery(competitor, _amplitude_query(competitor)),
        ),
    )
    spec = V5UniversalSearchSpec(
        parent_observation_index=0,
        context=context,
        exact_observation=V5ExactSearchObservation.from_observation_view(
            observation,
            curve_id=str(dataset.arrays[observation_array("observation_id")][0]),
        ),
        query_catalog_artifact_id=f"query-catalog/{seed}",
        query_catalog_artifact_sha256=_hash(f"query-catalog/{seed}"),
    )
    frozen_protocol = _protocol() if protocol is None else protocol

    def runner(task):
        branch = task.branch
        identity = f"{task.observation_id}/{branch.global_key.wire_key}"
        common = {
            "universal_query_sha256": task.universal_context.audit_sha256,
            "exact_curve_sha256": task.exact_curve_sha256,
            "global_branch_key": branch.global_key,
            "context_sha256": branch.context_sha256,
            "executor_artifact_id": f"executor/{_hash(identity)[:16]}",
            "executor_artifact_sha256": _hash(f"executor/{identity}"),
        }
        if unverified and branch.global_index == 0:
            return V5FrozenBranchSearchResult(
                **common,
                outcome="unverified",
                completed=False,
                exact_forward_calls_used=1,
                termination_reason="executor_failed_before_protocol_completion",
            )
        if branch.global_index % 2:
            return V5FrozenBranchSearchResult(
                **common,
                outcome="no_compatible_found_within_frozen_search_budget",
                completed=True,
                exact_forward_calls_used=8,
                termination_reason="exact_forward_budget_exhausted_without_compatible",
            )
        first = V5CompatibleRepresentativeReference(
            artifact_id=f"exact/{_hash(identity)[:16]}",
            artifact_sha256=_hash(f"exact/{identity}"),
            representative_set_id=f"set/{_hash(identity)[:16]}",
            representative_set_sha256=_hash(f"set/{identity}"),
            cluster_id="cluster-0000",
            metric_value=0.01,
            bounds_passed=True,
            physics_passed=True,
            target_local=(0.5,) * 26,
        )
        representatives = (first,)
        varying = np.asarray(branch.condition.varying_dimension_mask, dtype=np.bool_)
        if multiple_targets and np.any(varying):
            target = np.full(26, 0.5, dtype=np.float64)
            target[varying] = 0.6
            representatives += (
                replace(
                    first,
                    artifact_id=f"{first.artifact_id}/second",
                    artifact_sha256=_hash(f"{first.artifact_sha256}/second"),
                    cluster_id="cluster-0001",
                    target_local=tuple(target),
                ),
            )
        return V5FrozenBranchSearchResult(
            **common,
            outcome="compatible_found",
            completed=True,
            exact_forward_calls_used=8,
            termination_reason="frozen_full_budget_completed_with_compatible_representatives",
            representatives=representatives,
        )

    sidecar = collect_v5_search_supervision_sidecar(
        parent,
        (spec,),
        sidecar_id=f"trainer-sidecar-{split}-{index}",
        protocol=frozen_protocol,
        runner=runner,
    )
    sidecar_path = tmp_path / f"sidecar-{split}-{index}.gvd5"
    write_v5_search_supervision_sidecar(sidecar, sidecar_path)
    return parent, sidecar_path


def _config(**updates) -> V5GroupedTrainingConfig:
    values = {
        "warmup_epochs": 1,
        "full_epochs": 1,
        "recipes_per_replica": 1,
        "validation_recipes_per_batch": 1,
        "steps_per_epoch": 1,
        "width": 8,
        "encoder_blocks": 1,
        "mixture_components": 12,
    }
    values.update(updates)
    return V5GroupedTrainingConfig(**values)


def _install_test_only_evidence_receipts(
    monkeypatch,
    sidecar_paths,
    *,
    training_eligible: bool = True,
) -> None:
    """Isolate trainer mechanics; real receipt replay has dedicated integration tests."""

    selected = tuple(Path(value) for value in sidecar_paths if value is not None)
    for sidecar_path in selected:
        evidence_receipt_path_for_sidecar(sidecar_path).write_text(
            "test-only-placeholder\n", encoding="utf-8"
        )

    def fake_reader(
        path,
        *,
        parent_dataset_path,
        sidecar_path,
        require_training_eligible=True,
        expected_consumer_role=None,
    ):
        assert not require_training_eligible or training_eligible
        assert Path(parent_dataset_path).is_file()
        assert Path(sidecar_path).is_file()
        if require_training_eligible:
            assert expected_consumer_role in {
                "gradient_training",
                "tuning_validation_only",
            }
        selected_path = Path(path)
        manifest = {
            "receipt_sha256": _hash(f"receipt/{selected_path.name}"),
            "label_purpose": (
                V5_SEARCH_LABEL_PURPOSE_TRAINING
                if training_eligible
                else V5_SEARCH_LABEL_PURPOSE_PILOT
            ),
            "full_training_eligible": training_eligible,
            "launch_source_bundle_sha256": _hash("test-launch-source-bundle"),
            "executor_source_bundle_sha256": _hash("test-executor-source-bundle"),
            "counts": {
                "branches": 1,
                "queries": 1,
                "exact_forward_calls_used": 8,
            },
        }
        return V5SearchEvidenceReceipt(
            path=selected_path,
            manifest=manifest,
            file_sha256=sha256(selected_path.read_bytes()).hexdigest(),
        )

    monkeypatch.setattr(
        grouped_training_shards_v5,
        "read_v5_search_evidence_receipt",
        fake_reader,
    )


def test_full_stage_uses_only_sidecar_parent_subset_and_preserves_multi_target_weights(
    tmp_path, monkeypatch
):
    labeled_train, train_sidecar = _parent_sidecar(
        tmp_path,
        split="train",
        index=0,
        with_sidecar=True,
        multiple_targets=True,
    )
    warmup_only_train, _ = _parent_sidecar(
        tmp_path, split="train", index=2, with_sidecar=False
    )
    validation, validation_sidecar = _parent_sidecar(
        tmp_path, split="tuning_validation", index=1, with_sidecar=True
    )
    _install_test_only_evidence_receipts(
        monkeypatch, (train_sidecar, validation_sidecar)
    )

    collection, audit = inspect_v5_grouped_training(
        (labeled_train, warmup_only_train),
        validation,
        _config(steps_per_epoch=2),
        train_sidecar_paths=(train_sidecar,),
        validation_sidecar_paths=(validation_sidecar,),
        replicas=1,
    )

    assert audit.train_recipe_count == 2
    assert audit.train_full_recipe_count == 1
    assert audit.selected_steps_per_epoch == 2
    assert audit.full_selected_steps_per_epoch == 1
    assert collection.train_recipes.tolist() == [0, 1]
    assert collection.full_train_recipes.tolist() == [0]
    assert audit.full_stage_permitted
    assert audit.train_full_outcome_counts["unverified"] == 0
    assert audit.phase_data_sources["warmup"]["train"] == (
        "grouped_parent_known_positive"
    )
    assert "labeled_parent_subset" in audit.phase_data_sources["full"]["train"]

    _, labels = collection.numpy_batch(collection.full_train_recipes, phase="full")
    positive_artifacts = labels["search_artifact_id"][labels["has_local_target"]]
    assert positive_artifacts.size > 0
    observed_multiplicities = []
    for artifact_id in np.unique(positive_artifacts):
        selected = labels["search_artifact_id"] == artifact_id
        observed_multiplicities.append(np.count_nonzero(selected))
        assert np.sum(labels["candidate_weight"][selected]) == pytest.approx(1.0)
    assert 2 in observed_multiplicities

    with pytest.raises(ValueError, match="checked search sidecar"):
        collection.numpy_batch((1,), phase="full")

    _, full_only = inspect_v5_grouped_training(
        (labeled_train, warmup_only_train),
        validation,
        _config(
            warmup_epochs=0,
            steps_per_epoch=999,
            full_steps_per_epoch=1,
        ),
        train_sidecar_paths=(train_sidecar,),
        validation_sidecar_paths=(validation_sidecar,),
        replicas=1,
    )
    assert full_only.selected_steps_per_epoch == 0
    assert full_only.full_selected_steps_per_epoch == 1


def test_full_stage_rejects_unverified_wrong_parent_and_protocol_drift(
    tmp_path, monkeypatch
):
    train, train_sidecar = _parent_sidecar(
        tmp_path, split="train", index=10, with_sidecar=True, unverified=True
    )
    validation, validation_sidecar = _parent_sidecar(
        tmp_path, split="tuning_validation", index=11, with_sidecar=True
    )
    _install_test_only_evidence_receipts(
        monkeypatch, (train_sidecar, validation_sidecar)
    )
    with pytest.raises(ValueError, match="unverified branch"):
        inspect_v5_grouped_training(
            train,
            validation,
            _config(),
            train_sidecar_paths=train_sidecar,
            validation_sidecar_paths=validation_sidecar,
        )
    with pytest.raises(ValueError, match="does not uniquely bind"):
        inspect_v5_grouped_training(
            train,
            validation,
            _config(),
            train_sidecar_paths=validation_sidecar,
            validation_sidecar_paths=train_sidecar,
        )

    drift_train, drift_train_sidecar = _parent_sidecar(
        tmp_path,
        split="train",
        index=12,
        with_sidecar=True,
        protocol=_protocol("train-drift"),
    )
    drift_validation, drift_validation_sidecar = _parent_sidecar(
        tmp_path,
        split="tuning_validation",
        index=13,
        with_sidecar=True,
        protocol=_protocol("validation-drift"),
    )
    _install_test_only_evidence_receipts(
        monkeypatch, (drift_train_sidecar, drift_validation_sidecar)
    )
    with pytest.raises(ValueError, match="different frozen search protocols"):
        inspect_v5_grouped_training(
            drift_train,
            drift_validation,
            _config(),
            train_sidecar_paths=drift_train_sidecar,
            validation_sidecar_paths=drift_validation_sidecar,
        )


def test_cli_and_two_phase_run_record_parent_sidecar_hashes_and_stage_sources(
    tmp_path, capsys, monkeypatch
):
    train, train_sidecar = _parent_sidecar(
        tmp_path, split="train", index=20, with_sidecar=True
    )
    validation, validation_sidecar = _parent_sidecar(
        tmp_path, split="tuning_validation", index=21, with_sidecar=True
    )
    _install_test_only_evidence_receipts(
        monkeypatch, (train_sidecar, validation_sidecar)
    )
    dry_output = tmp_path / "dry"
    arguments = (
        "--train-dataset",
        str(train),
        "--validation-dataset",
        str(validation),
        "--train-sidecar",
        str(train_sidecar),
        "--validation-sidecar",
        str(validation_sidecar),
        "--output-dir",
        str(dry_output),
        "--warmup-epochs",
        "1",
        "--full-epochs",
        "1",
        "--recipes-per-replica",
        "1",
        "--steps-per-epoch",
        "1",
        "--full-steps-per-epoch",
        "1",
        "--pairwise-ranking-weight",
        "0.7",
        "--local-coverage-weight",
        "0.4",
        "--operational-top-l-alignment-weight",
        "0.6",
        "--local-coverage-temperature",
        "0.08",
        "--operational-hit-rms-threshold",
        "0.06",
        "--operational-duplicate-rms-threshold",
        "0.015",
        "--dry-run",
    )
    assert main(arguments) == 0
    dry = json.loads(capsys.readouterr().out)
    assert dry["training_audit"]["full_stage_permitted"] is True
    assert len(dry["training_audit"]["train_sidecar_artifacts"]) == 1
    assert dry["config"]["pairwise_ranking_weight"] == pytest.approx(0.7)
    assert dry["config"]["local_coverage_weight"] == pytest.approx(0.4)
    assert dry["config"]["operational_top_l_alignment_weight"] == pytest.approx(0.6)
    assert dry["config"]["local_coverage_temperature"] == pytest.approx(0.08)
    assert dry["config"]["operational_hit_rms_threshold"] == pytest.approx(0.06)
    assert dry["config"]["operational_duplicate_rms_threshold"] == pytest.approx(
        0.015
    )
    assert not dry_output.exists()

    output = tmp_path / "two-stage"
    result = train_v5_grouped_model(
        train,
        validation,
        output,
        _config(
            full_epochs=2,
            pairwise_ranking_weight=0.7,
            local_coverage_weight=0.4,
        ),
        train_sidecar_paths=train_sidecar,
        validation_sidecar_paths=validation_sidecar,
        strategy=tf.distribute.OneDeviceStrategy("/cpu:0"),
    )
    plan = json.loads((output / "run_plan.json").read_text(encoding="utf-8"))
    history = json.loads(result.history_path.read_text(encoding="utf-8"))
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert [value["kind"] for value in plan["input_manifests"]] == [
        "grouped_parent",
        "frozen_search_sidecar",
        "task_bound_search_evidence_receipt",
        "grouped_parent",
        "frozen_search_sidecar",
        "task_bound_search_evidence_receipt",
    ]
    assert [value["phase"] for value in history["epochs"]] == [
        "warmup",
        "full",
        "full",
    ]
    assert plan["objectives"]["warmup"]["config"]["pairwise_ranking_weight"] == 0.0
    assert plan["objectives"]["warmup"]["config"]["local_coverage_weight"] == 0.4
    assert plan["objectives"]["warmup"]["config"][
        "operational_top_l_alignment_weight"
    ] == 1.0
    assert plan["objectives"]["full"]["config"]["pairwise_ranking_weight"] == 0.7
    assert plan["objectives"]["full"]["config"]["local_coverage_weight"] == 0.4
    assert plan["objectives"]["full"]["config"][
        "operational_top_l_alignment_weight"
    ] == 1.0
    assert history["epochs"][0]["data_source"]["train"] == (
        "grouped_parent_known_positive"
    )
    assert "sidecar" in history["epochs"][1]["data_source"]["train"]
    assert result.checkpoint_selection_status == FULL_CHECKPOINT_SELECTION_STATUS
    assert result.paper_model_eligible is False
    assert len(result.full_checkpoint_paths) == 2
    checkpoints = manifest["full_checkpoints"]
    checkpoint_paths = tuple(output / value["relative_path"] for value in checkpoints)
    assert result.full_checkpoint_paths == checkpoint_paths
    assert [value["epoch"] for value in checkpoints] == [2, 3]
    assert [value["phase_epoch"] for value in checkpoints] == [1, 2]
    assert [value.name for value in checkpoint_paths] == [
        "full_epoch_0001_global_epoch_0002.keras",
        "full_epoch_0002_global_epoch_0003.keras",
    ]
    for checkpoint, checkpoint_path in zip(checkpoints, checkpoint_paths):
        assert checkpoint_path.is_file()
        assert checkpoint["file_sha256"] == sha256(
            checkpoint_path.read_bytes()
        ).hexdigest()
        assert len(checkpoint["weights_sha256"]) == 64
        loaded_checkpoint = tf.keras.models.load_model(checkpoint_path, compile=False)
        assert len(loaded_checkpoint.weights) > 0
    assert history["full_checkpoints"] == manifest["full_checkpoints"]
    assert [value["full_checkpoint"] for value in history["epochs"][1:]] == checkpoints
    assert manifest["checkpoint_selection"]["status"] == (
        FULL_CHECKPOINT_SELECTION_STATUS
    )
    assert manifest["checkpoint_selection"]["paper_selected_checkpoint"] is None
    assert manifest["paper_model_status"][
        "paper_checkpoint_candidates_eligible_for_external_selection"
    ] is True
    assert manifest["paper_model_status"]["paper_model_eligible"] is False
    assert manifest["paper_model_status"]["paper_model_ineligibility_reasons"] == [
        "paper_checkpoint_selection_pending"
    ]
    assert len(manifest["input_artifacts"]) == 6
    assert {value["kind"] for value in manifest["input_artifacts"]} == {
        "grouped_parent",
        "frozen_search_sidecar",
        "task_bound_search_evidence_receipt",
    }
    assert all(len(value["artifact_sha256"]) == 64 for value in manifest["input_artifacts"])


def test_full_training_with_unsafe_expansion_is_explicitly_not_a_paper_candidate(
    tmp_path, monkeypatch
):
    train, train_sidecar = _parent_sidecar(
        tmp_path, split="train", index=24, with_sidecar=True
    )
    validation, validation_sidecar = _parent_sidecar(
        tmp_path, split="tuning_validation", index=25, with_sidecar=True
    )
    _install_test_only_evidence_receipts(
        monkeypatch, (train_sidecar, validation_sidecar)
    )
    monkeypatch.setattr(
        grouped_training_inventory_v5,
        "V5_TRAIN_MAX_EXPANDED_ROWS_PER_REPLICA",
        0,
    )

    output = tmp_path / "unsafe-full"
    with pytest.warns(RuntimeWarning, match="engineering override"):
        result = train_v5_grouped_model(
            train,
            validation,
            output,
            _config(allow_unsafe_sidecar_expansion_for_engineering=True),
            train_sidecar_paths=train_sidecar,
            validation_sidecar_paths=validation_sidecar,
            strategy=tf.distribute.OneDeviceStrategy("/cpu:0"),
        )

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    status = manifest["paper_model_status"]
    assert len(manifest["full_checkpoints"]) == 1
    assert status["full_training_completed"] is True
    assert status["sidecar_expansion_paper_claim_allowed"] is False
    assert status[
        "paper_checkpoint_candidates_eligible_for_external_selection"
    ] is False
    assert status["paper_model_eligible"] is False
    assert "sidecar_expansion_gate_does_not_allow_paper_claim" in status[
        "paper_model_ineligibility_reasons"
    ]


def test_full_stage_rejects_a_sidecar_without_task_bound_evidence_receipt(
    tmp_path, monkeypatch
):
    train, train_sidecar = _parent_sidecar(
        tmp_path, split="train", index=30, with_sidecar=True
    )
    validation, validation_sidecar = _parent_sidecar(
        tmp_path, split="tuning_validation", index=31, with_sidecar=True
    )

    with pytest.raises(ValueError, match="task-bound evidence receipt"):
        inspect_v5_grouped_training(
            train,
            validation,
            _config(),
            train_sidecar_paths=train_sidecar,
            validation_sidecar_paths=validation_sidecar,
        )

    collection, audit = inspect_v5_grouped_training(
        train,
        validation,
        _config(full_epochs=0),
        train_sidecar_paths=train_sidecar,
        validation_sidecar_paths=validation_sidecar,
    )
    assert not audit.full_stage_permitted
    assert collection.full_train_recipes.size == 0
    assert collection.full_validation_recipes.size == 0
    with pytest.raises(ValueError, match="eligible task-bound evidence receipt"):
        collection.numpy_batch((0,), phase="full")

    _install_test_only_evidence_receipts(
        monkeypatch,
        (train_sidecar, validation_sidecar),
        training_eligible=False,
    )
    audited_collection, audited = inspect_v5_grouped_training(
        train,
        validation,
        _config(full_epochs=0),
        train_sidecar_paths=train_sidecar,
        validation_sidecar_paths=validation_sidecar,
    )
    assert not audited.full_stage_permitted
    assert audited_collection.shards[0].full_evidence_receipt is not None
    assert not audited_collection.shards[0].full_evidence_receipt.full_training_eligible
    assert audited_collection.full_train_recipes.size == 0


def test_slurm_wrapper_supports_optional_sidecar_subset_and_independent_full_steps():
    wrapper = (
        Path(__file__).parents[1]
        / "PosteriorV8/slurm/v5_grouped_train_gpu4.sbatch"
    ).read_text(encoding="utf-8")

    for variable in (
        "POSTERIOR_V8_V5_TRAIN_SIDECARS",
        "POSTERIOR_V8_V5_VALIDATION_SIDECARS",
        "POSTERIOR_V8_FULL_STEPS_PER_EPOCH",
    ):
        assert variable in wrapper
    assert "--train-sidecar" in wrapper
    assert "--validation-sidecar" in wrapper
    assert "--full-steps-per-epoch" in wrapper
