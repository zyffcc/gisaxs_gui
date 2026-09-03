from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from utils.ML_Fitting_1D_GISAXS.PosteriorV8.bounds_first_dataset import RANGE_CODE
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.bounds_first_shards import SPLIT_CODE
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.bounds_holdout_audit import (
    BOUNDS_HOLDOUT_AUDIT_SCHEMA,
    _source_hashes,
    _validate_dataset_binding,
    _write_json_exclusive,
    build_parser,
    run_bounds_holdout_audit,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.bounds_holdout_exact import (
    OBSERVABILITY_UNKNOWN,
    exact_record,
    exact_summary,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.bounds_holdout_metrics import (
    BoundsHoldoutAuditConfig,
    local_rms,
    select_holdout_rows,
    selection_priority,
    supervised_summary,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.contract import GuiComponentParameters
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.evaluation import EvaluationThresholds
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.one_click_inference import InferenceBudget
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.rescue_inference import RescuePolicy
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.study_protocol import STUDY_PROTOCOL_VERSION


def test_config_allows_only_explicit_calibration_or_test_and_valid_best_of_n():
    assert BoundsHoldoutAuditConfig(split="calibration").split == "calibration"
    assert BoundsHoldoutAuditConfig(split="test").split == "test"
    for forbidden in ("train", "tuning_validation"):
        with pytest.raises(ValueError, match="only calibration or test"):
            BoundsHoldoutAuditConfig(split=forbidden)
    with pytest.raises(ValueError, match="oracle_best_of_n"):
        BoundsHoldoutAuditConfig(oracle_best_of_n=(1, 25))
    with pytest.raises(ValueError, match="increasing"):
        BoundsHoldoutAuditConfig(topology_ks=(8, 1))


def test_selection_reads_only_the_named_holdout_split(monkeypatch):
    arrays = {
        "assigned_split": np.asarray(
            [
                SPLIT_CODE["train"],
                SPLIT_CODE["tuning_validation"],
                SPLIT_CODE["calibration"],
                SPLIT_CODE["calibration"],
                SPLIT_CODE["test"],
            ]
        ),
        "range_regime": np.asarray([RANGE_CODE["full"]] * 5),
        "global_recipe_index": np.asarray([10, 11, 12, 12, 13]),
        "view_index": np.asarray([0, 0, 0, 1, 0]),
    }
    fake_shard = SimpleNamespace(arrays=arrays)
    monkeypatch.setattr(
        "utils.ML_Fitting_1D_GISAXS.PosteriorV8.bounds_holdout_metrics.load_shard",
        lambda _path: fake_shard,
    )
    dataset = SimpleNamespace(shard_paths=(Path("immutable-v4.npz"),))

    calibration, row_count, recipe_count = select_holdout_rows(
        dataset,
        BoundsHoldoutAuditConfig(split="calibration", maximum_examples=5, exact_maximum_examples=0),
    )
    test, _, _ = select_holdout_rows(
        dataset,
        BoundsHoldoutAuditConfig(split="test", maximum_examples=5, exact_maximum_examples=0),
    )

    assert row_count == 2
    assert recipe_count == 1
    assert len(calibration) == 1
    assert calibration[0].recipe_index == 12
    assert [item.row for item in test] == [4]
    assert selection_priority(7, 12, 0) == selection_priority(7, 12, 0)


def test_dataset_binding_requires_same_fingerprint_and_zero_holdout_leakage():
    dataset = SimpleNamespace(payload={"fingerprint_sha256": "f" * 64})
    trained = {
        "fingerprint_sha256": "f" * 64,
        "consumed_splits": ["train", "tuning_validation"],
        "excluded_splits": ["calibration", "test"],
        "calibration_test_rows_consumed": 0,
    }
    _validate_dataset_binding({"dataset_audit": trained}, dataset)
    with pytest.raises(ValueError, match="holdout isolation"):
        _validate_dataset_binding(
            {"dataset_audit": {**trained, "calibration_test_rows_consumed": 1}},
            dataset,
        )
    with pytest.raises(ValueError, match="different fingerprints"):
        _validate_dataset_binding(
            {"dataset_audit": {**trained, "fingerprint_sha256": "0" * 64}},
            dataset,
        )


def test_varying_dimension_rms_and_supervised_summary_are_dimension_normalized():
    target = np.zeros(26)
    sample = np.zeros(26)
    sample[[1, 4]] = (0.3, 0.4)
    varying = np.zeros(26, dtype=bool)
    varying[[1, 4]] = True
    assert local_rms(sample, target, varying) == pytest.approx(np.sqrt(0.125))
    assert local_rms(sample, target, np.zeros(26, dtype=bool)) == 0.0

    oracle = {
        "sample_count": 24,
        "sample_best_of_n_local_rms": {
            "1": 0.20,
            "4": 0.10,
            "12": 0.05,
            "24": 0.02,
        },
        "best_mixture_center_local_rms": 0.03,
        "compliance_counts": {
            "local_unit_bounds": 24,
            "user_bounds": 24,
            "roundtrip": 24,
            "physics": 24,
        },
        "compliance_errors": [],
    }
    records = [
        {
            "topology_rank": 1,
            "canonical_branch_rank_given_truth_topology": 1,
            "canonical_joint_branch_rank": 4,
            "oracle_branch": oracle,
        },
        {
            "topology_rank": 9,
            "canonical_branch_rank_given_truth_topology": 2,
            "canonical_joint_branch_rank": 40,
            "oracle_branch": oracle,
        },
    ]
    summary = supervised_summary(records, BoundsHoldoutAuditConfig())
    assert summary["topology_recall_at_k"] == {"1": 0.5, "3": 0.5, "8": 0.5}
    assert summary["canonical_joint_topology_branch_recall_at_k"]["32"] == 0.5
    assert summary["proposal_compliance_rate"]["roundtrip"] == 1.0
    assert summary["oracle_branch_sample_best_of_n_local_unit_rms"]["24"]["p90"] == pytest.approx(
        0.02
    )


def test_exact_summary_keeps_unknown_and_never_converts_failure_to_no_solution():
    config = BoundsHoldoutAuditConfig(exact_maximum_examples=1)
    records = [
        {
            "compatible_candidate_found": False,
            "truth_parameter_mode_recalled": None,
            "candidate_count": 0,
            "parameter_mode_count_before_observability": 0,
            "verified_status": "no_candidate_found_within_budget",
            "raw_status": "no_candidate_found_within_budget",
            "observability_status": OBSERVABILITY_UNKNOWN,
            "observability_assessment_status_distribution": {OBSERVABILITY_UNKNOWN: 1},
            "confirmed_effective_parameter_mode_count": 0,
            "forward_budget": {
                "limit": 1200,
                "used": 1024,
                "remaining": 176,
                "raw_refinement_used": 1000,
                "observability_used": 24,
                "observability_limit": 64,
            },
        }
    ]
    summary = exact_summary(records, config)
    assert summary["compatible_candidate_success_rate"] == 0.0
    assert summary["compatibility_threshold_source"] == ("unfrozen_engineering_thresholds")
    assert summary["formal_paper_compatibility_claim_allowed"] is False
    assert summary["observability_metrics_role"] == ("secondary_diagnostic_not_success_gate")
    assert summary["finite_search_failure_is_no_solution"] is False
    assert summary["observability_status_distribution"] == {OBSERVABILITY_UNKNOWN: 1}
    assert summary["budget_ledger"]["used"] == 1024
    assert summary["budget_ledger"]["used"] == (
        summary["budget_ledger"]["raw_refinement_calls"]
        + summary["budget_ledger"]["observability_calls"]
    )
    assert summary["budget_ledger"]["remaining"] == 176
    assert summary["budget_ledger"]["configured_total_limit"] == 1200
    assert summary["budget_ledger"]["observability_unknown_assessment_count"] == 1


def test_exact_record_uses_unified_refinement_and_observability_ledger(monkeypatch):
    bounds = SimpleNamespace(component_bounds=(), resolution_bounds=None, sha256="b" * 64)
    clean = SimpleNamespace(label=SimpleNamespace(bounds=bounds))
    shard = SimpleNamespace(
        arrays={
            "x": np.zeros((1, 2)),
            "point_mask": np.ones((1, 2)),
            "global_features": np.zeros((1, 2)),
        }
    )
    observed = object()
    reference = SimpleNamespace(recipe_index=7, view_index=1, row=0)
    result = SimpleNamespace(
        evaluation_report=None,
        status="no_candidate_found_within_budget",
        raw_generation=SimpleNamespace(status="candidate_budget_exhausted", candidates=()),
        raw_candidates=(),
        compatible_parameter_mode_count=0,
        effective_parameter_mode_count=0,
        observability_assessments=(SimpleNamespace(status=OBSERVABILITY_UNKNOWN),),
        total_forward_evaluation_limit=1200,
        total_forward_evaluations=1024,
        raw_refinement_forward_evaluations=1000,
        observability_exact_forward_evaluations=24,
        observability_forward_evaluation_limit=64,
        audit=SimpleNamespace(sources=()),
    )
    monkeypatch.setattr(
        "utils.ML_Fitting_1D_GISAXS.PosteriorV8.bounds_holdout_exact.reconstruct_observed",
        lambda *_args, **_kwargs: (shard, clean, observed),
    )
    monkeypatch.setattr(
        "utils.ML_Fitting_1D_GISAXS.PosteriorV8.bounds_holdout_exact."
        "run_verified_bounds_local_inference",
        lambda *_args, **_kwargs: result,
    )
    monkeypatch.setattr(
        "utils.ML_Fitting_1D_GISAXS.PosteriorV8.bounds_holdout_exact.ProductionBranchFactory",
        lambda _space: object(),
    )
    monkeypatch.setattr(
        "utils.ML_Fitting_1D_GISAXS.PosteriorV8.bounds_holdout_exact.UserSearchSpace",
        SimpleNamespace(for_components=lambda *_args, **_kwargs: object()),
    )

    record = exact_record(reference, object(), BoundsHoldoutAuditConfig())

    assert record["forward_budget"] == {
        "limit": 1200,
        "used": 1024,
        "remaining": 176,
        "raw_refinement_used": 1000,
        "observability_used": 24,
        "observability_limit": 64,
        "per_candidate_limit": 128,
        "source_usage": {},
    }
    assert record["observability_status"] == OBSERVABILITY_UNKNOWN
    assert record["observability_unknown_count"] == 1


def test_exact_record_reference_uses_the_full_synthetic_linear_solution(monkeypatch):
    bounds = SimpleNamespace(component_bounds=(), resolution_bounds=None, sha256="b" * 64)
    recipe = SimpleNamespace(
        topology_id=0,
        background=0.25,
        effective_amplitudes=(2.5,),
        resolution_effective_amplitude=0.0,
    )
    clean = SimpleNamespace(
        label=SimpleNamespace(
            bounds=bounds,
            truth_components=(GuiComponentParameters(shape="sphere", R=12.0, sigma_R=1.2),),
            truth_resolution=None,
        ),
        simulation_recipe=recipe,
    )
    shard = SimpleNamespace(
        arrays={
            "x": np.zeros((1, 2)),
            "point_mask": np.ones((1, 2)),
            "global_features": np.zeros((1, 2)),
        }
    )
    reference = SimpleNamespace(recipe_index=7, view_index=1, row=0)
    result = SimpleNamespace(
        evaluation_report=SimpleNamespace(audit_schema="evaluation-test-schema"),
        status="compatible_target_reached",
        raw_generation=SimpleNamespace(status="raw_target_reached"),
        raw_candidates=(object(),),
        effective_parameter_mode_count=0,
        observability_assessments=(),
        total_forward_evaluation_limit=4,
        total_forward_evaluations=1,
        raw_refinement_forward_evaluations=1,
        observability_exact_forward_evaluations=0,
        observability_forward_evaluation_limit=3,
        audit=SimpleNamespace(sources=()),
    )
    captured = {}

    monkeypatch.setattr(
        "utils.ML_Fitting_1D_GISAXS.PosteriorV8.bounds_holdout_exact.reconstruct_observed",
        lambda *_args, **_kwargs: (shard, clean, object()),
    )
    monkeypatch.setattr(
        "utils.ML_Fitting_1D_GISAXS.PosteriorV8.bounds_holdout_exact."
        "run_verified_bounds_local_inference",
        lambda *_args, **_kwargs: result,
    )
    monkeypatch.setattr(
        "utils.ML_Fitting_1D_GISAXS.PosteriorV8.bounds_holdout_exact.ProductionBranchFactory",
        lambda _space: object(),
    )
    monkeypatch.setattr(
        "utils.ML_Fitting_1D_GISAXS.PosteriorV8.bounds_holdout_exact.UserSearchSpace",
        SimpleNamespace(for_components=lambda *_args, **_kwargs: object()),
    )

    def fake_evaluate(*_args, reference_modes, **_kwargs):
        captured["truth"] = reference_modes[0]
        return SimpleNamespace(
            mode_recall=1.0,
            predicted_parameter_mode_count=1,
            accepted_count=1,
        )

    monkeypatch.setattr(
        "utils.ML_Fitting_1D_GISAXS.PosteriorV8.bounds_holdout_exact.evaluate_candidates",
        fake_evaluate,
    )

    record = exact_record(reference, object(), BoundsHoldoutAuditConfig())

    truth = captured["truth"]
    assert truth.linear_solution.background == recipe.background
    assert truth.linear_solution.particle_amplitudes == recipe.effective_amplitudes
    assert truth.linear_solution.resolution_amplitude == recipe.resolution_effective_amplitude
    assert record["truth_parameter_mode_recalled"] == 1.0


def test_output_is_exclusive_and_slurm_wrapper_is_worker_only(tmp_path):
    output = tmp_path / "audit-v1.json"
    _write_json_exclusive(output, {"schema": BOUNDS_HOLDOUT_AUDIT_SCHEMA})
    with pytest.raises(FileExistsError, match="overwrite"):
        _write_json_exclusive(output, {"schema": "changed"})

    script = (
        Path(__file__).parents[1] / "PosteriorV8" / "slurm" / "bounds_holdout_v3_gpu.sbatch"
    ).read_text(encoding="utf-8")
    for required in (
        "SLURM_JOB_ID:?",
        "max-wgs*",
        "POSTERIOR_V8_V4_DATASET_DIR",
        "POSTERIOR_V8_V3_TRAINING_RUN",
        "POSTERIOR_V8_V3_HOLDOUT_OUTPUT",
        "PosteriorV8.bounds_holdout_audit",
        "--exact-maximum-examples",
        "/data/dust/user/zhaiyufe/",
    ):
        assert required in script


def test_source_provenance_covers_full_posterior_v8_execution_package():
    hashes = _source_hashes()
    for required in (
        "bounds_holdout_audit.py",
        "bounds_holdout_exact.py",
        "bounds_holdout_metrics.py",
        "bounds_local_verified.py",
        "component_observability.py",
        "component_observability_diagnostics.py",
        "observability_assessment.py",
        "reduced_model_search.py",
        "profiled_forward.py",
        "profiled_refinement.py",
        "production_bridge.py",
        "rescue_inference.py",
        "reference_bank.py",
        "model_v3.py",
        "slurm/bounds_holdout_v3_gpu.sbatch",
    ):
        assert len(hashes[required]) == 64


def test_cli_supports_proposal_only_smoke_and_separate_split_label():
    args = build_parser().parse_args(
        [
            "--shards",
            "one.npz",
            "--training-run",
            "run-v3",
            "--output",
            "test-audit-v1.json",
            "--split",
            "calibration",
            "--exact-maximum-examples",
            "0",
        ]
    )
    assert args.split == "calibration"
    assert args.exact_maximum_examples == 0


def test_real_v3_artifact_proposal_only_audit_is_end_to_end(tmp_path):
    tf = pytest.importorskip("tensorflow")
    from utils.ML_Fitting_1D_GISAXS.PosteriorV8.bounds_first_shards import (
        BoundsFirstShardConfig,
        BoundsFirstShardSpec,
        recipe_group_id_for,
        split_for_recipe_group,
    )
    from utils.ML_Fitting_1D_GISAXS.PosteriorV8.build_bounds_first_shards import (
        build_shard,
    )
    from utils.ML_Fitting_1D_GISAXS.PosteriorV8.proposal_training_v3 import (
        BoundsProposalTrainingConfig,
        train_bounds_proposal,
    )

    shard_config = BoundsFirstShardConfig(501, "k1", 2, 128)
    seen = set()
    recipe_count = None
    for index in range(256):
        seen.add(split_for_recipe_group(recipe_group_id_for(shard_config, index)))
        if {"train", "tuning_validation", "test"}.issubset(seen):
            recipe_count = index + 1
            break
    assert recipe_count is not None
    shard = build_shard(
        tmp_path / "data",
        shard_config,
        BoundsFirstShardSpec(0, 0, recipe_count),
    )
    training = train_bounds_proposal(
        [shard.npz_path],
        tmp_path / "run-v3",
        BoundsProposalTrainingConfig(
            epochs=1,
            global_batch_size=2,
            seed=71,
            max_points=1000,
            width=4,
            encoder_blocks=1,
            mixture_components=2,
            shuffle_buffer=8,
            checkpoint_keep=2,
            steps_per_epoch=1,
        ),
        strategy=tf.distribute.MirroredStrategy(devices=["/cpu:0"]),
    )
    output = tmp_path / "audit" / "test-proposal-only-v1.json"
    payload = run_bounds_holdout_audit(
        [shard.npz_path],
        training.output_dir,
        output,
        BoundsHoldoutAuditConfig(
            split="test",
            maximum_examples=1,
            oracle_mixture_limit=2,
            oracle_samples_per_mixture=2,
            oracle_best_of_n=(1, 4),
            exact_maximum_examples=0,
        ),
    )

    assert output.is_file()
    assert payload["selection"]["test_rows_evaluated"] == 1
    assert payload["selection"]["train_rows_evaluated"] == 0
    assert payload["selection"]["tuning_validation_rows_evaluated"] == 0
    assert payload["exact_acceptance"]["summary"]["enabled"] is False
    assert payload["supervised_proposal"]["summary"]["example_count"] == 1
    assert len(payload["artifact"]["model_sha256"]) == 64
    assert payload["study_protocol_version"] == STUDY_PROTOCOL_VERSION
    assert len(payload["study_protocol_sha256"]) == 64
    assert len(payload["immutable_inputs"]["selected_tensor_sha256_aggregate"]) == 64

    exact_output = tmp_path / "audit" / "test-exact-v1.json"
    exact_payload = run_bounds_holdout_audit(
        [shard.npz_path],
        training.output_dir,
        exact_output,
        BoundsHoldoutAuditConfig(
            split="test",
            maximum_examples=1,
            oracle_mixture_limit=2,
            oracle_samples_per_mixture=2,
            oracle_best_of_n=(1, 4),
            exact_maximum_examples=1,
            inference_budget=InferenceBudget(
                topology_beam_size=1,
                branch_beam_size=1,
                mixture_components_per_branch=1,
                samples_per_mixture=1,
                per_candidate_forward_evaluation_limit=1,
                forward_evaluation_limit=1,
            ),
            rescue_policy=RescuePolicy(
                target_candidate_count=1,
                fallback_attempt_limit=1,
                sobol_seeds_per_branch=1,
            ),
            thresholds=EvaluationThresholds(
                raw_exact_log_rmse_max=1.0e6,
                standardized_exact_log_rmse_max=1.0e6,
                parameter_mode_distance_max=0.08,
                raw_curve_equivalence_log_rmse_max=0.02,
                reference_mode_distance_max=0.10,
            ),
        ),
    )
    exact = exact_payload["exact_acceptance"]
    assert exact["summary"]["enabled"] is True
    assert exact["summary"]["budget_ledger"]["used"] <= 1
    exact_example = exact["examples"][0]
    assert exact_example["finite_search_failure_is_no_solution"] is False
    assert exact_example["forward_budget"]["used"] == (
        exact_example["forward_budget"]["raw_refinement_used"]
        + exact_example["forward_budget"]["observability_used"]
    )
    assert exact_example["observability_unknown_count"] == (
        exact_example["observability_assessment_status_distribution"].get(OBSERVABILITY_UNKNOWN, 0)
    )

    shard_alias = tmp_path / "shard-alias.npz"
    metadata_alias = shard_alias.with_suffix(".json")
    shard_alias.symlink_to(shard.npz_path)
    metadata_alias.symlink_to(shard.metadata_path)
    with pytest.raises(ValueError, match="regular immutable"):
        run_bounds_holdout_audit(
            [shard_alias],
            training.output_dir,
            tmp_path / "symlink-shard-audit.json",
            BoundsHoldoutAuditConfig(
                maximum_examples=1,
                oracle_mixture_limit=2,
                oracle_best_of_n=(1, 4),
                exact_maximum_examples=0,
            ),
        )
    run_alias = tmp_path / "run-alias"
    run_alias.symlink_to(training.output_dir, target_is_directory=True)
    with pytest.raises(ValueError, match="must not be a symlink"):
        run_bounds_holdout_audit(
            [shard.npz_path],
            run_alias,
            tmp_path / "symlink-run-audit.json",
            BoundsHoldoutAuditConfig(
                maximum_examples=1,
                oracle_mixture_limit=2,
                oracle_best_of_n=(1, 4),
                exact_maximum_examples=0,
            ),
        )
