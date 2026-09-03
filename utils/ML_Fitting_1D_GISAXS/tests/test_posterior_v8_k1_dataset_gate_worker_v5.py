from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

import pytest


pytest.importorskip("tensorflow")

from utils.ML_Fitting_1D_GISAXS.PosteriorV8.build_grouped_dataset_v5 import (
    build_tiny_v5_grouped_dataset,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.build_k1_memorization_dataset_v5 import (
    V5_K1_DATASET_BUILDER_ROLE,
    build_v5_k1_memorization_dataset,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.grouped_dataset_v5 import (
    read_v5_grouped_dataset,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.model_v5_contract import (
    MODEL_V5_SCHEMA,
    MODEL_V5_VERSION,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.k1_phase_a_cross_platform_v5 import (
    gate_claim_sha256,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.k1_phase_a_dataset_binding_v5 import (
    publish_v5_k1_phase_a_dataset_binding,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.run_k1_memorization_gate_v5 import (
    MODEL_FILENAME,
    MODEL_PROVENANCE_FILENAME,
    RESULT_FILENAME,
    V5_K1_DATASET_GATE_PUBLISHED_RESULT_FIELDS,
    V5_K1_DATASET_GATE_ROLE,
    V5K1DatasetGateConfig,
    V5K1PhaseATrainingEvidence,
    run_v5_k1_dataset_memorization_gate,
    validate_v5_k1_memorization_dataset,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.study_protocol import protocol_payload


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
SLURM_ROOT = Path(__file__).resolve().parents[1] / "PosteriorV8" / "slurm"
_SOURCE = {
    "source_archive_sha256": "1" * 64,
    "source_manifest_sha256": "2" * 64,
    "source_tree_sha256": "3" * 64,
}
_REFERENCE_SHA256 = "4" * 64
_REFERENCE_BYTE_COUNT = 123456
_MANIFEST_SHA256 = "5" * 64
_SCIENTIFIC_SHA256 = "6" * 64
_COMPARISON_SHA256 = "7" * 64


def _marker_expected() -> dict[str, object]:
    values = {
        "expected_source": dict(_SOURCE),
        "reference_file_sha256": _REFERENCE_SHA256,
        "reference_file_byte_count": _REFERENCE_BYTE_COUNT,
        "reference_manifest_sha256": _MANIFEST_SHA256,
        "scientific_content_sha256": _SCIENTIFIC_SHA256,
        "comparison_result_sha256": _COMPARISON_SHA256,
    }
    return {
        **values,
        "expected_gate_claim_sha256": gate_claim_sha256(
            source=values["expected_source"],
            reference_file_sha256=values["reference_file_sha256"],
            reference_file_byte_count=values["reference_file_byte_count"],
            reference_manifest_sha256=values["reference_manifest_sha256"],
            scientific_content_sha256=values["scientific_content_sha256"],
            comparison_result_sha256=values["comparison_result_sha256"],
        ),
    }


def _marker_text() -> str:
    core = {
        "schema": "gisaxs.posterior_v8.sobol_cross_platform_pass_marker/v1",
        "version": "posterior_v8_atomic_exclusive_pass_after_exact_compare_v1",
        "status": "PASS",
        "comparison_result_sha256": _COMPARISON_SHA256,
        "reference_manifest_sha256": _MANIFEST_SHA256,
        "candidate_manifest_sha256": _MANIFEST_SHA256,
        "scientific_content_sha256": _SCIENTIFIC_SHA256,
        "source": dict(_SOURCE),
    }
    payload = {
        **core,
        "marker_sha256": sha256(
            json.dumps(core, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


@pytest.fixture(scope="module")
def checked_k1_dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp("v5-k1-dataset-gate")
    path = root / "k1-known-truth.gvd5"
    result = build_v5_k1_memorization_dataset(
        path,
        recipe_count=1,
        topology="sphere",
        base_seed=20260903,
        view_indices=(0,),
        pattern_id=0,
        allowed_root=root,
        hostname="max-cpu-001",
        environment={"SLURM_JOB_ID": "24390001"},
    )
    path.chmod(0o400)
    return root, path, result


@pytest.fixture(scope="module")
def phase_a_evidence(checked_k1_dataset):
    root, dataset, _ = checked_k1_dataset
    marker = root / "PASS.json"
    marker.write_text(_marker_text(), encoding="utf-8")
    marker.chmod(0o400)
    binding = root / "dataset.binding.json"
    expected = _marker_expected()
    publish_v5_k1_phase_a_dataset_binding(
        dataset,
        marker,
        binding,
        original_dataset_path=str(dataset),
        marker_expected=expected,
    )
    return V5K1PhaseATrainingEvidence(
        dataset_binding_path=binding,
        cross_platform_pass_marker_path=marker,
        original_dataset_path=str(dataset),
        **_SOURCE,
        reference_file_sha256=_REFERENCE_SHA256,
        reference_file_byte_count=_REFERENCE_BYTE_COUNT,
        reference_manifest_sha256=_MANIFEST_SHA256,
        scientific_content_sha256=_SCIENTIFIC_SHA256,
        comparison_result_sha256=_COMPARISON_SHA256,
        gate_claim_sha256=expected["expected_gate_claim_sha256"],
    )


def test_versioned_cpu_builder_publishes_checked_k1_and_refuses_overwrite(
    checked_k1_dataset,
):
    root, path, result = checked_k1_dataset
    dataset, receipt = read_v5_grouped_dataset(path)

    assert result["status"] == "published"
    assert result["plan"]["scientific_role"] == V5_K1_DATASET_BUILDER_ROLE
    assert result["plan"]["model_acceptance_evidence"] is False
    assert result["dataset"]["manifest_sha256"] == dataset.manifest["manifest_sha256"]
    assert result["dataset"]["artifact_sha256"] == receipt.artifact_sha256
    assert validate_v5_k1_memorization_dataset(dataset)["clean_parent_count"] == 1

    with pytest.raises(FileExistsError, match="overwrite"):
        build_v5_k1_memorization_dataset(
            path,
            recipe_count=1,
            topology="sphere",
            base_seed=20260903,
            view_indices=(0,),
            pattern_id=0,
            dry_run=True,
            allowed_root=root,
        )


def test_cpu_builder_dry_run_checks_without_writing_on_login_node(checked_k1_dataset):
    root, _, _ = checked_k1_dataset
    output = root / "dry-run.gvd5"
    result = build_v5_k1_memorization_dataset(
        output,
        recipe_count=2,
        topology="vertical_cylinder",
        base_seed=7,
        view_indices="0,2",
        pattern_id=0,
        dry_run=True,
        allowed_root=root,
        hostname="max-wgs001",
        environment={},
    )

    assert result["status"] == "checked_dry_run"
    assert result["writes_performed"] is False
    assert result["plan"]["topology"] == "vertical_cylinder"
    assert result["plan"]["view_indices"] == [0, 2]
    assert not output.exists()


def test_dataset_gate_rejects_multicomponent_or_nongenerating_inputs():
    k2 = build_tiny_v5_grouped_dataset(topology=("sphere", "cylinder"), pattern_id=0)
    with pytest.raises(ValueError, match="K=1"):
        validate_v5_k1_memorization_dataset(k2)

    nongenerating = build_tiny_v5_grouped_dataset(generating_only=False)
    with pytest.raises(ValueError, match="generating-candidate-only"):
        validate_v5_k1_memorization_dataset(nongenerating)


def test_gate_dry_run_binds_v52_dataset_source_and_pending_complete_criteria(
    checked_k1_dataset,
    phase_a_evidence,
):
    root, dataset, builder_result = checked_k1_dataset
    output = root / "gate-dry-run"
    result = run_v5_k1_dataset_memorization_gate(
        dataset,
        output,
        source_root=REPOSITORY_ROOT,
        evidence=phase_a_evidence,
        config=V5K1DatasetGateConfig(steps=100, width=64, encoder_blocks=3, smoke=True),
        dry_run=True,
        allowed_root=root,
        hostname="max-wgs001",
        environment={},
    )

    assert result["status"] == "checked_dry_run"
    assert result["writes_performed"] is False
    assert result["gate_executed"] is False
    assert not output.exists()
    assert result["live_model_identity"]["schema_version"] == MODEL_V5_SCHEMA
    assert result["live_model_identity"]["model_version"] == MODEL_V5_VERSION
    assert result["resolved_model_config"] == {
        "width": 32,
        "encoder_blocks": 1,
        "mixture_components": 1,
    }
    assert result["resolved_gate_config"]["steps"] == 2
    assert result["dataset"]["dataset_file_sha256"] == builder_result["dataset"]["artifact_sha256"]
    assert len(result["source"]["bundle_sha256"]) == 64
    assert result["training_evidence"]["source_snapshot"] == _SOURCE
    assert result["training_evidence"]["cross_platform_gate"][
        "gate_claim_sha256"
    ] == phase_a_evidence.gate_claim_sha256
    assert {
        "utils/ML_Fitting_1D_GISAXS/PosteriorV8/candidate_supervision_v5.py",
        "utils/ML_Fitting_1D_GISAXS/PosteriorV8/grouped_training_v5.py",
        "utils/ML_Fitting_1D_GISAXS/PosteriorV8/training_objective_v5.py",
        "src/gimap/features/fitting/domain/scattering_model.py",
        "src/gimap/features/fitting/domain/physical_constraints.py",
    } <= set(result["source"]["files"])

    complete = result["single_branch_k1_gate_contract"]
    expected = protocol_payload()["stages"]["k1_memorization"]["gates"]
    assert complete["phase_b_frozen_criteria"] == expected
    assert complete["topology_schedule"] == "single_branch_sphere_pattern0"
    assert "not_a_stochastic_single_draw" in complete["stage_a_metric_semantics"]
    assert complete["stage_b_status"] == "pending_not_executed_fail_closed"
    assert complete["single_branch_phase_b_gate_passed"] is False
    assert complete["complete_k1_proposal_exact_gate_passed"] is False
    assert set(complete["stage_b_pending_metrics"]) == {
        "branch_conditioned_local_mdn_single_draw_local_rms_median",
        "branch_conditioned_local_mdn_best_of_32_local_rms_median",
        "branch_conditioned_local_mdn_best_of_32_local_rms_p90",
        "exact_post_refine_raw_log_rmse_p90",
        "exact_post_refine_compatible_rate",
    }


def test_gate_allows_node_local_dataset_but_keeps_output_below_dust_root(
    checked_k1_dataset,
    phase_a_evidence,
    tmp_path,
):
    dataset_root, dataset, _ = checked_k1_dataset
    output_root = tmp_path / "dust"
    output_root.mkdir()
    result = run_v5_k1_dataset_memorization_gate(
        dataset,
        output_root / "gate-dry-run",
        source_root=REPOSITORY_ROOT,
        evidence=phase_a_evidence,
        config=V5K1DatasetGateConfig(smoke=True),
        dry_run=True,
        allowed_root=output_root,
        dataset_allowed_root=dataset_root,
    )

    assert result["status"] == "checked_dry_run"
    with pytest.raises(ValueError, match="output_dir must be below"):
        run_v5_k1_dataset_memorization_gate(
            dataset,
            tmp_path / "outside-dust",
            source_root=REPOSITORY_ROOT,
            evidence=phase_a_evidence,
            config=V5K1DatasetGateConfig(smoke=True),
            dry_run=True,
            allowed_root=output_root,
            dataset_allowed_root=dataset_root,
        )


def test_gate_execution_rejects_login_node_and_missing_slurm(
    checked_k1_dataset, phase_a_evidence
):
    root, dataset, _ = checked_k1_dataset
    config = V5K1DatasetGateConfig(smoke=True)
    with pytest.raises(RuntimeError, match="not max-wgs"):
        run_v5_k1_dataset_memorization_gate(
            dataset,
            root / "login-rejected",
            source_root=REPOSITORY_ROOT,
            evidence=phase_a_evidence,
            config=config,
            allowed_root=root,
            hostname="max-wgs001",
            environment={"SLURM_JOB_ID": "1"},
        )
    with pytest.raises(RuntimeError, match="SLURM_JOB_ID"):
        run_v5_k1_dataset_memorization_gate(
            dataset,
            root / "no-slurm-rejected",
            source_root=REPOSITORY_ROOT,
            evidence=phase_a_evidence,
            config=config,
            allowed_root=root,
            hostname="max-gpu-001",
            environment={},
        )


def test_gate_fails_closed_on_wrong_claim_before_any_model_output(
    checked_k1_dataset, phase_a_evidence
):
    root, dataset, _ = checked_k1_dataset
    output = root / "wrong-evidence-rejected"
    with pytest.raises(ValueError, match="gate claim"):
        run_v5_k1_dataset_memorization_gate(
            dataset,
            output,
            source_root=REPOSITORY_ROOT,
            evidence=replace(phase_a_evidence, gate_claim_sha256="0" * 64),
            config=V5K1DatasetGateConfig(smoke=True),
            dry_run=True,
            allowed_root=root,
        )
    assert not output.exists()


def test_gpu_smoke_atomically_publishes_stage_a_model_and_receipt(
    checked_k1_dataset, phase_a_evidence
):
    root, dataset, _ = checked_k1_dataset
    output = root / "stage-a-smoke"
    result = run_v5_k1_dataset_memorization_gate(
        dataset,
        output,
        source_root=REPOSITORY_ROOT,
        evidence=phase_a_evidence,
        config=V5K1DatasetGateConfig(smoke=True),
        allowed_root=root,
        hostname="max-gpu-001",
        environment={"SLURM_JOB_ID": "24390002"},
    )

    assert result["status"] == "stage_a_smoke_completed"
    assert result["scientific_role"] == V5_K1_DATASET_GATE_ROLE
    assert result["model_acceptance_evidence"] is False
    assert result["stage_a_pass_enforced"] is False
    assert result["stage_a_passed"] == result["stage_a_memorization_result"]["passed"]
    assert result["stage_a_configured_wiring_gate_passed"] == result["stage_a_passed"]
    assert result["single_branch_phase_b_gate_passed"] is False
    assert result["complete_k1_proposal_exact_gate_passed"] is False
    assert result["single_branch_k1_gate_contract"]["stage_b_status"].endswith(
        "fail_closed"
    )
    assert (output / MODEL_FILENAME).is_file()
    assert (output / MODEL_PROVENANCE_FILENAME).is_file()
    assert (output / MODEL_FILENAME).stat().st_mode & 0o222 == 0
    assert (output / MODEL_PROVENANCE_FILENAME).stat().st_mode & 0o222 == 0
    saved = json.loads((output / RESULT_FILENAME).read_text(encoding="utf-8"))
    assert saved == result
    assert (output / RESULT_FILENAME).stat().st_mode & 0o222 == 0
    assert set(saved) == V5_K1_DATASET_GATE_PUBLISHED_RESULT_FIELDS
    model_bytes = (output / MODEL_FILENAME).read_bytes()
    assert sha256(model_bytes).hexdigest() == result["model_artifact"]["sha256"]
    assert result["model_artifact"]["reload_graph_contract_passed"] is True
    provenance = json.loads(
        (output / MODEL_PROVENANCE_FILENAME).read_text(encoding="utf-8")
    )
    assert provenance["training_evidence"] == result["training_evidence"]
    assert provenance["model"]["sha256"] == result["model_artifact"]["sha256"]
    assert provenance["binding_sha256"] == result["model_artifact"]["provenance"][
        "binding_sha256"
    ]
    assert not tuple(output.glob(".*.tmp"))

    with pytest.raises(FileExistsError, match="overwrite"):
        run_v5_k1_dataset_memorization_gate(
            dataset,
            output,
            source_root=REPOSITORY_ROOT,
            evidence=phase_a_evidence,
            config=V5K1DatasetGateConfig(smoke=True),
            dry_run=True,
            allowed_root=root,
        )


def test_maxwell_wrappers_form_a_versioned_cpu_to_gpu_dag_contract():
    cpu = (SLURM_ROOT / "v5_k1_memorization_dataset_cpu.sbatch").read_text()
    gpu = (SLURM_ROOT / "v5_k1_memorization_gpu.sbatch").read_text()

    for value in (
        "POSTERIOR_V8_V5_K1_RECIPE_COUNT",
        "POSTERIOR_V8_V5_K1_TOPOLOGY",
        "POSTERIOR_V8_V5_K1_BASE_SEED",
        "POSTERIOR_V8_V5_K1_VIEW_INDICES",
        "POSTERIOR_V8_V5_K1_PATTERN_ID",
        "POSTERIOR_V8_V5_K1_DATASET_OUTPUT",
        "POSTERIOR_V8_V5_K1_DATASET_BINDING",
        "POSTERIOR_V8_V5_CROSS_PLATFORM_PASS_MARKER",
        "POSTERIOR_V8_V5_EXPECTED_CROSS_PLATFORM_GATE_CLAIM_SHA256",
        "build_k1_memorization_dataset_v5",
        "k1_phase_a_dataset_binding_v5",
    ):
        assert value in cpu
    for value in (
        "SLURM_JOB_ID",
        "POSTERIOR_V8_V5_K1_DATASET",
        "POSTERIOR_V8_V5_K1_DATASET_BINDING",
        "POSTERIOR_V8_V5_CROSS_PLATFORM_PASS_MARKER",
        "POSTERIOR_V8_V5_EXPECTED_CROSS_PLATFORM_GATE_CLAIM_SHA256",
        "POSTERIOR_V8_V5_K1_GATE_OUTPUT",
        "POSTERIOR_V8_V5_K1_WIDTH",
        "POSTERIOR_V8_V5_K1_ENCODER_BLOCKS",
        "POSTERIOR_V8_V5_K1_SMOKE",
        "POSTERIOR_V8_V5_K1_DRY_RUN",
        "run_k1_memorization_gate_v5",
        "k1_phase_a_dataset_binding_v5",
        '--dataset-root "$GISAXS_JOB_CACHE_ROOT"',
        "--constraint=GPUx1",
    ):
        assert value in gpu
    assert "/data/dust/user/zhaiyufe/" in cpu
    assert "/data/dust/user/zhaiyufe/" in gpu
    r2_logs = "GISAXS_ONE_CLICK_PAPER_V5_20260903_V5_2_R2/logs"
    assert r2_logs in cpu
    assert r2_logs in gpu
    assert "GISAXS_ONE_CLICK_PAPER_V5_20260903_V5_2/logs" not in cpu
    assert "GISAXS_ONE_CLICK_PAPER_V5_20260903_V5_2/logs" not in gpu
    assert "max-wgs*" in cpu and "max-wgs*" in gpu
    for exact_argument in (
        '--dataset-binding "$POSTERIOR_V8_JOB_DATASET_BINDING"',
        '--cross-platform-pass-marker "$POSTERIOR_V8_JOB_PASS_MARKER"',
        '--original-dataset-path "$POSTERIOR_V8_V5_K1_DATASET"',
        '--source-archive-sha256 "$POSTERIOR_V8_V5_EXPECTED_SOURCE_ARCHIVE_SHA256"',
        '--source-manifest-sha256 "$POSTERIOR_V8_V5_EXPECTED_SOURCE_MANIFEST_SHA256"',
        '--source-tree-sha256 "$POSTERIOR_V8_V5_EXPECTED_SOURCE_TREE_SHA256"',
        '--gate-claim-sha256 '
        '"$POSTERIOR_V8_V5_EXPECTED_CROSS_PLATFORM_GATE_CLAIM_SHA256"',
    ):
        assert exact_argument in gpu
