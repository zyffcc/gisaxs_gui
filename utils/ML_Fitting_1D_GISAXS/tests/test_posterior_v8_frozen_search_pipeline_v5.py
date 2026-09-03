from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from utils.ML_Fitting_1D_GISAXS.PosteriorV8 import (
    frozen_search_pipeline_v5 as pipeline,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.calibrated_search_threshold_v5 import (
    compatibility_stratum_from_v5_observation,
    inspect_v5_compatibility_calibration,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.compatibility_calibration import (
    CompatibilityCalibrationSample,
    fit_compatibility_calibration,
    write_compatibility_calibration_atomic,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.exact_search_executor_v5 import (
    build_v5_frozen_exact_search_protocol,
    read_v5_exact_search_executor_artifact,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.exact_search_schedule_v5 import (
    V5FrozenExactOptimizerSchedule,
    V5FrozenLocalSobolSchedule,
    write_v5_frozen_local_sobol_schedule,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.freeze_exact_search_schedule_v5 import (
    freeze_v5_exact_search_schedule,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.frozen_search_launch_plan_v5 import (
    fingerprint_v5_frozen_search_source,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.frozen_search_pipeline_v5 import (
    V5FrozenSearchExecution,
    V5SelectedTopologySearchSchedule,
    V5_SEARCH_PIPELINE_FORMAL_SIDECAR_PREFIX,
    V5_SEARCH_PIPELINE_PILOT_SIDECAR_PREFIX,
    execute_v5_frozen_search_shard,
    materialize_or_verify_v5_frozen_search_parent,
    plan_v5_frozen_search_shard,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.grouped_dataset_v5 import (
    observation_array,
    read_v5_grouped_dataset,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.observation_v5 import (
    build_v5_observation_data_view,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.run_frozen_search_pipeline_v5 import (
    main as pipeline_main,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.search_evidence_receipt_v5 import (
    V5_SEARCH_LABEL_PURPOSE_TRAINING,
    build_v5_search_label_binding,
    read_v5_search_evidence_receipt,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.search_supervision_sidecar_v5 import (
    branch_array,
    query_array,
    read_v5_search_supervision_sidecar,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.search_supervision_contract_v5 import (
    V5_SEARCH_PROTOCOL_TIER_PAPER_FULL_CALIBRATED,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.sobol_recipe_v5 import (
    materialize_v5_sobol_clean_recipe,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.sobol_recipe_coordinates_v5 import (
    v5_sobol_recipe_design,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.sobol_universal_query_design_v5 import (
    V5SobolUniversalTopologyQueryDesign,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.split_design_v5 import (
    V5SplitCounts,
    V5SplitPlan,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.synthetic_recipe_v5 import (
    sample_v5_clean_recipe,
)


def _split_plan() -> V5SplitPlan:
    return V5SplitPlan.create(
        V5SplitCounts(
            train=64,
            tuning_validation=8,
            calibration=1,
            test=1,
            reference=1,
            ood_topology=1,
            ood_range_width=1,
            ood_weak_component=1,
            ood_acquisition_policy=1,
        ),
        guard_band=2,
    )


def _shard_plan():
    # Final 168D design: offset 56 is naturally K1 Vertical Cylinder with measured sigma.
    result = plan_v5_frozen_search_shard(
        plan=_split_plan(),
        design=v5_sobol_recipe_design(scramble_seed=20260903),
        target_split="train",
        start=56,
        shard_index=None,
        count=1,
        view_indices=(0,),
        topology_schedule=V5SelectedTopologySearchSchedule(
            schedule_id="k1-three-shape-search-smoke-v1",
            selected_topology_ids=(0, 1, 2),
        ),
    )
    assert result.query_designs[0].generating_topology_id == 2
    return result


def _execution():
    seeds = V5FrozenLocalSobolSchedule.generate(
        schedule_id="k1-smoke-two-starts-v1", point_count=2, base_seed=17
    )
    optimizer = V5FrozenExactOptimizerSchedule(
        schedule_id="k1-smoke-one-call-per-seed-v1",
        direct_scout_seed_count=1,
        per_seed_forward_evaluation_limit=1,
    )
    protocol = build_v5_frozen_exact_search_protocol(
        protocol_id="k1-pipeline-contract-smoke-v1",
        seed_schedule=seeds,
        optimizer_schedule=optimizer,
        standardized_threshold_name="smoke_only_large_standardized_gate",
        standardized_threshold_value=1.0e9,
        raw_threshold_name="smoke_only_large_raw_gate",
        raw_threshold_value=1.0e9,
        threshold_source_id="test_only_not_a_scientific_acceptance_threshold",
    )
    return V5FrozenSearchExecution(
        seed_schedule=seeds,
        optimizer_schedule=optimizer,
        protocol=protocol,
        launch_source_bundle_sha256=fingerprint_v5_frozen_search_source(
            Path(__file__).parents[3]
        )["bundle_sha256"],
        launch_plan_sha256=sha256(b"local-k1-pipeline-smoke-plan").hexdigest(),
    )


def _calibration(path: Path, view=None):
    if view is None:
        recipe = sample_v5_clean_recipe(("sphere",), recipe_seed=77)
        view = next(
            value
            for value in (
                build_v5_observation_data_view(recipe, index, split_id="calibration")
                for index in range(8)
            )
            if value.acceptance_sigma_log is not None
        )
    stratum = compatibility_stratum_from_v5_observation(view)
    artifact = fit_compatibility_calibration(
        tuple(
            CompatibilityCalibrationSample(
                sample_id=f"calibration-{index:03d}",
                independent_group_id=f"recipe-{index:03d}",
                stratum=stratum,
                score=float(index + 1),
                effective_valid_point_count=view.effective_valid_point_count,
                acquisition_policy_id=view.acquisition_policy_id,
                measurement_sigma_available=True,
            )
            for index in range(9)
        ),
        dataset_manifest_sha256="1" * 64,
        calibration_split_sha256="2" * 64,
        target_coverage=0.8,
        minimum_samples_per_stratum=5,
    )
    write_compatibility_calibration_atomic(path, artifact)
    return inspect_v5_compatibility_calibration(path)


def test_real_k1_pipeline_completes_budget_and_records_feasible_solver_fallback(tmp_path):
    root = tmp_path / "k1-smoke"
    execution = _execution()
    source_revalidations = 0

    def unchanged_source_bundle() -> str:
        nonlocal source_revalidations
        source_revalidations += 1
        return execution.launch_source_bundle_sha256

    result = execute_v5_frozen_search_shard(
        _shard_plan(),
        execution,
        root,
        allow_local_smoke=True,
        source_bundle_fingerprint=unchanged_source_bundle,
    )

    assert result["status"] == "complete"
    assert source_revalidations == 3
    assert result["sidecar"]["queries"] == 1
    assert result["sidecar"]["branches"] == 3
    assert result["sidecar"]["outcomes"]["compatible_found"] == 3
    assert result["sidecar"]["outcomes"]["unverified"] == 0
    assert result["sidecar"]["outcomes"]["exact_forward_calls_used_total"] == 6
    assert result["label_purpose"] == (
        "engineering_throughput_pilot_not_training_eligible"
    )
    assert result["task_bound_evidence_receipt"]["full_training_eligible"] is False
    assert Path(result["task_bound_evidence_receipt"]["path"]).is_file()
    evidence_receipt = json.loads(
        Path(result["task_bound_evidence_receipt"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    assert evidence_receipt["launch_plan_sha256"] == execution.launch_plan_sha256
    assert evidence_receipt["shard_plan_sha256"] == _shard_plan().sha256
    checked_receipt = read_v5_search_evidence_receipt(
        result["task_bound_evidence_receipt"]["path"],
        parent_dataset_path=root / "grouped-parent.gvd5",
        sidecar_path=root / "search-supervision.gvd5",
        require_training_eligible=False,
    )
    assert not checked_receipt.full_training_eligible
    with pytest.raises(ValueError, match="non-training search evidence"):
        read_v5_search_evidence_receipt(
            result["task_bound_evidence_receipt"]["path"],
            parent_dataset_path=root / "grouped-parent.gvd5",
            sidecar_path=root / "search-supervision.gvd5",
            require_training_eligible=True,
        )

    sidecar, _ = read_v5_search_supervision_sidecar(
        root / "search-supervision.gvd5"
    )
    assert sidecar.manifest["sidecar_id"].startswith(
        V5_SEARCH_PIPELINE_PILOT_SIDECAR_PREFIX
    )
    assert np.all(sidecar.arrays[branch_array("runner_completed")])
    assert np.all(sidecar.arrays[branch_array("exact_forward_calls_used")] == 2)
    design = _shard_plan().query_designs[0]
    assert sidecar.arrays[query_array("query_catalog_artifact_sha256")].tolist() == [
        design.sha256
    ]
    catalog_path = root / "query-catalogs" / f"{design.clean_group_id}.json"
    assert V5SobolUniversalTopologyQueryDesign.from_json(
        catalog_path.read_text(encoding="utf-8")
    ) == design

    artifacts = [
        read_v5_exact_search_executor_artifact(path)
        for path in (root / "executor-evidence").glob("*.gvd5")
    ]
    vertical = next(
        value
        for value in artifacts
        if value.manifest["task"]["global_branch_key"] == "topology-02:wire-01"
    )
    assert vertical.manifest["outcome"] == "compatible_found"
    assert vertical.manifest["ledger"]["exact_forward_calls_used"] == 2
    messages = vertical.arrays["attempt_message"].tolist()
    assert all("amplitude_solver_status=-101" in value for value in messages)
    assert all("not an optimality certificate" in value for value in messages)

    changed_evidence = next((root / "executor-evidence").glob("*.gvd5"))
    with changed_evidence.open("ab") as stream:
        stream.write(b"post-audit-change")
    with pytest.raises(ValueError, match="changed after task-bound audit"):
        read_v5_search_evidence_receipt(
            result["task_bound_evidence_receipt"]["path"],
            parent_dataset_path=root / "grouped-parent.gvd5",
            sidecar_path=root / "search-supervision.gvd5",
            require_training_eligible=False,
        )


def test_formal_search_specs_bind_checked_threshold_per_observation(tmp_path):
    shard = _shard_plan()
    recipe = materialize_v5_sobol_clean_recipe(
        shard.points[0], shard.grouped_plan.design
    )
    view = build_v5_observation_data_view(recipe, 0, split_id="train")
    assert view.acceptance_sigma_log is not None
    calibration = _calibration(tmp_path / "calibration.json", view)
    parent = SimpleNamespace(
        arrays={observation_array("observation_id"): np.asarray(["formal-view-0"])},
        observation_count=1,
    )
    design = shard.query_designs[0]
    specs = pipeline._search_specs(
        shard,
        parent,
        (recipe,),
        {
            design.clean_group_id: (
                f"v5-sobol-universal-topology-query-design/{design.sha256}",
                design.sha256,
            )
        },
        calibration,
    )
    assert len(specs) == 1
    threshold = specs[0].calibrated_threshold
    assert threshold is not None
    assert threshold.calibration_identity == calibration.identity
    base = _execution()
    protocol = build_v5_frozen_exact_search_protocol(
        protocol_id="formal-spec-binding-regression",
        seed_schedule=base.seed_schedule,
        optimizer_schedule=base.optimizer_schedule,
        protocol_tier=V5_SEARCH_PROTOCOL_TIER_PAPER_FULL_CALIBRATED,
        calibration_identity=calibration.identity,
    )
    execution = V5FrozenSearchExecution(
        seed_schedule=base.seed_schedule,
        optimizer_schedule=base.optimizer_schedule,
        protocol=protocol,
        launch_source_bundle_sha256=base.launch_source_bundle_sha256,
        launch_plan_sha256=base.launch_plan_sha256,
        calibration=calibration,
    )
    assert execution.protocol.calibration_identity == calibration.identity
    result = execute_v5_frozen_search_shard(
        shard,
        execution,
        tmp_path / "formal-calibrated-search",
        allow_local_smoke=True,
    )
    assert result["status"] == "complete"
    assert result["label_purpose"] == (
        "formal_calibrated_contract_smoke_not_training_eligible"
    )
    assert result["full_training_label_claimed"] is False
    assert result["task_bound_evidence_receipt"]["full_training_eligible"] is False
    sidecar, _ = read_v5_search_supervision_sidecar(
        tmp_path / "formal-calibrated-search/search-supervision.gvd5"
    )
    assert sidecar.manifest["sidecar_id"].startswith(
        V5_SEARCH_PIPELINE_FORMAL_SIDECAR_PREFIX
    )
    binding = json.loads(
        sidecar.arrays[query_array("selected_threshold_binding_json")][0]
    )
    assert binding["protocol_tier"] == "paper_full_calibrated"
    assert binding["full_training_label_permitted"] is True
    assert binding["calibrated_threshold_sha256"] == threshold.sha256
    with pytest.raises(ValueError, match="plan-derived formal production authorization"):
        build_v5_search_label_binding(
            protocol=protocol,
            seed_schedule_sha256=execution.seed_schedule.sha256,
            optimizer_schedule_sha256=execution.optimizer_schedule.sha256,
            launch_source_bundle_sha256=execution.launch_source_bundle_sha256,
            launch_plan_sha256=execution.launch_plan_sha256,
            shard_plan_sha256=shard.sha256,
            label_purpose=V5_SEARCH_LABEL_PURPOSE_TRAINING,
        )


def test_existing_parent_is_replayed_strictly_and_never_overwritten(tmp_path):
    path = tmp_path / "parent.gvd5"
    first, receipt, _, reused = materialize_or_verify_v5_frozen_search_parent(
        _shard_plan(), path
    )
    assert reused is False
    original = path.read_bytes()
    second, replay_receipt, _, reused = materialize_or_verify_v5_frozen_search_parent(
        _shard_plan(), path
    )
    assert reused is True
    assert path.read_bytes() == original
    assert replay_receipt.artifact_sha256 == receipt.artifact_sha256
    assert second.manifest == first.manifest

    corrupt = tmp_path / "corrupt.gvd5"
    corrupt.write_bytes(b"not-a-checked-parent")
    with pytest.raises(ValueError):
        materialize_or_verify_v5_frozen_search_parent(_shard_plan(), corrupt)
    assert corrupt.read_bytes() == b"not-a-checked-parent"


def test_pipeline_exception_preserves_failure_audit_without_sidecar(tmp_path, monkeypatch):
    def fail_collection(*_args, **_kwargs):
        raise RuntimeError("injected frozen runner failure")

    monkeypatch.setattr(
        pipeline, "collect_v5_search_supervision_sidecar", fail_collection
    )
    root = tmp_path / "failed"
    with pytest.raises(pipeline.V5FrozenSearchPipelineError) as caught:
        execute_v5_frozen_search_shard(
            _shard_plan(), _execution(), root, allow_local_smoke=True
        )
    assert caught.value.failure_audit_path == root / "failure.json"
    failure = json.loads((root / "failure.json").read_text(encoding="utf-8"))
    assert failure["failed_stage"] == "search"
    assert failure["sidecar_published"] is False
    assert failure["task_bound_evidence_receipt_published"] is False
    assert failure["partial_executor_evidence_retained"] is True
    assert not (root / "search-supervision.gvd5").exists()
    assert not (root / "completion.json").exists()


def test_runtime_source_drift_refuses_sidecar_and_receipt_publication(tmp_path):
    execution = _execution()
    root = tmp_path / "runtime-source-drift"
    calls = 0

    def changed_source_bundle() -> str:
        nonlocal calls
        calls += 1
        return "0" * 64

    with pytest.raises(pipeline.V5FrozenSearchPipelineError) as caught:
        execute_v5_frozen_search_shard(
            _shard_plan(),
            execution,
            root,
            allow_local_smoke=True,
            source_bundle_fingerprint=changed_source_bundle,
        )
    failure = json.loads(caught.value.failure_audit_path.read_text(encoding="utf-8"))
    assert calls == 1
    assert failure["failed_stage"] == "pre_sidecar_source_fingerprint"
    assert failure["sidecar_published"] is False
    assert failure["task_bound_evidence_receipt_published"] is False
    assert failure["completion_published"] is False


def test_source_drift_during_receipt_audit_refuses_final_receipt(tmp_path):
    execution = _execution()
    root = tmp_path / "receipt-publication-source-drift"
    calls = 0

    def changes_after_receipt_audit() -> str:
        nonlocal calls
        calls += 1
        if calls < 3:
            return execution.launch_source_bundle_sha256
        return "0" * 64

    with pytest.raises(pipeline.V5FrozenSearchPipelineError) as caught:
        execute_v5_frozen_search_shard(
            _shard_plan(),
            execution,
            root,
            allow_local_smoke=True,
            source_bundle_fingerprint=changes_after_receipt_audit,
        )
    failure = json.loads(caught.value.failure_audit_path.read_text(encoding="utf-8"))
    assert calls == 3
    assert failure["failed_stage"] == "task_bound_evidence_receipt"
    assert failure["sidecar_published"] is True
    assert failure["task_bound_evidence_receipt_published"] is False
    assert failure["completion_published"] is False


def test_existing_query_catalog_tamper_is_rejected_instead_of_replaced(tmp_path):
    shard = _shard_plan()
    directory = tmp_path / "catalogs"
    bindings, _ = pipeline._publish_or_verify_query_catalogs(shard, directory)
    design = shard.query_designs[0]
    assert bindings[design.clean_group_id][1] == design.sha256
    path = directory / f"{design.clean_group_id}.json"
    original = json.loads(path.read_text(encoding="utf-8"))
    original["artifact_sha256"] = "0" * 64
    tampered = json.dumps(original, indent=2, sort_keys=True) + "\n"
    path.write_text(tampered, encoding="utf-8")

    with pytest.raises(ValueError, match="does not reproduce"):
        pipeline._publish_or_verify_query_catalogs(shard, directory)
    assert path.read_text(encoding="utf-8") == tampered


def test_login_node_refusal_and_excluded_generating_topology_fail_before_writes(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(pipeline.socket, "gethostname", lambda: "max-wgs.desy.de")
    output = tmp_path / "forbidden"
    with pytest.raises(RuntimeError, match="login node"):
        execute_v5_frozen_search_shard(
            _shard_plan(), _execution(), output, allow_local_smoke=True
        )
    assert not output.exists()

    with pytest.raises(ValueError, match="include the generating topology"):
        plan_v5_frozen_search_shard(
            plan=_split_plan(),
            design=v5_sobol_recipe_design(scramble_seed=20260903),
            target_split="train",
            start=56,
            shard_index=None,
            count=1,
            view_indices=(0,),
            topology_schedule=V5SelectedTopologySearchSchedule(
                schedule_id="wrong-subset", selected_topology_ids=(0, 1)
            ),
        )


def test_slurm_execution_requires_runtime_source_revalidation(tmp_path):
    with pytest.raises(ValueError, match="pre-publication source verifier"):
        execute_v5_frozen_search_shard(
            _shard_plan(), _execution(), tmp_path / "unverified-slurm"
        )
    assert not (tmp_path / "unverified-slurm").exists()


def test_schedule_freeze_and_worker_cli_are_dry_run_by_default(tmp_path, capsys):
    schedule_path = tmp_path / "search-schedule.gvd5"
    preview = freeze_v5_exact_search_schedule(
        schedule_id="freeze-smoke",
        point_count=2,
        base_seed=17,
        output=schedule_path,
    )
    assert preview["status"] == "dry_run"
    assert not schedule_path.exists()
    written = freeze_v5_exact_search_schedule(
        schedule_id="freeze-smoke",
        point_count=2,
        base_seed=17,
        output=schedule_path,
        write=True,
    )
    assert written["status"] == "written"
    with pytest.raises(FileExistsError):
        freeze_v5_exact_search_schedule(
            schedule_id="freeze-smoke",
            point_count=2,
            base_seed=17,
            output=schedule_path,
            write=True,
        )

    plan = _split_plan()
    design = v5_sobol_recipe_design(scramble_seed=20260903)
    plan_path = tmp_path / "split-plan.json"
    design_path = tmp_path / "design.json"
    plan_path.write_text(plan.to_json(), encoding="utf-8")
    design_path.write_text(design.to_json(), encoding="utf-8")
    source_root = Path(__file__).parents[3]
    source_sha = fingerprint_v5_frozen_search_source(source_root)["bundle_sha256"]
    output = tmp_path / "dry-run-output"
    arguments = (
        "--source-root",
        str(source_root),
        "--expected-source-bundle-sha256",
        source_sha,
        "--launch-plan-sha256",
        sha256(b"local-worker-dry-run-plan").hexdigest(),
        "--split-plan",
        str(plan_path),
        "--expected-split-plan-sha256",
        plan.sha256,
        "--expected-split-plan-file-sha256",
        sha256(plan_path.read_bytes()).hexdigest(),
        "--sobol-design",
        str(design_path),
        "--expected-sobol-design-sha256",
        design.sha256,
        "--expected-sobol-design-file-sha256",
        sha256(design_path.read_bytes()).hexdigest(),
        "--target-split",
        "train",
        "--start",
        "56",
        "--count",
        "1",
        "--view-indices",
        "0",
        "--topology-schedule-id",
        "k1-smoke",
        "--selected-topology-ids",
        "0,1,2",
        "--local-sobol-schedule",
        str(schedule_path),
        "--expected-local-schedule-sha256",
        written["schedule_sha256"],
        "--expected-local-schedule-artifact-sha256",
        written["artifact_sha256"],
        "--expected-local-schedule-manifest-sha256",
        written["manifest_sha256"],
        "--optimizer-schedule-id",
        "opt",
        "--direct-scout-seed-count",
        "1",
        "--per-seed-forward-evaluation-limit",
        "1",
        "--protocol-id",
        "protocol",
        "--standardized-threshold-name",
        "std",
        "--standardized-threshold-value",
        "1",
        "--raw-threshold-name",
        "raw",
        "--raw-threshold-value",
        "1",
        "--threshold-source-id",
        "test",
        "--output-root",
        str(output),
    )
    assert pipeline_main(arguments) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "dry_run"
    assert report["writes_performed"] is False
    assert not output.exists()

    original_plan_text = plan_path.read_text(encoding="utf-8")
    plan_path.write_text(
        json.dumps(json.loads(original_plan_text), sort_keys=True), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="file changed after launch planning"):
        pipeline_main(arguments)
    plan_path.write_text(original_plan_text, encoding="utf-8")

    original_design_text = design_path.read_text(encoding="utf-8")
    design_path.write_text(
        json.dumps(json.loads(original_design_text), sort_keys=True), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="file changed after launch planning"):
        pipeline_main(arguments)
    design_path.write_text(original_design_text, encoding="utf-8")

    changed_plan = list(arguments)
    changed_plan[changed_plan.index("--expected-split-plan-sha256") + 1] = "0" * 64
    with pytest.raises(RuntimeError, match="split plan or Sobol design changed"):
        pipeline_main(tuple(changed_plan))
    changed_design = list(arguments)
    changed_design[changed_design.index("--expected-sobol-design-sha256") + 1] = (
        "0" * 64
    )
    with pytest.raises(RuntimeError, match="split plan or Sobol design changed"):
        pipeline_main(tuple(changed_design))
    assert not output.exists()

    for expected_flag in (
        "--expected-local-schedule-sha256",
        "--expected-local-schedule-artifact-sha256",
        "--expected-local-schedule-manifest-sha256",
    ):
        changed_identity = list(arguments)
        changed_identity[changed_identity.index(expected_flag) + 1] = "0" * 64
        with pytest.raises(RuntimeError, match="schedule identity changed"):
            pipeline_main(tuple(changed_identity))
        assert not output.exists()

    replacement_path = tmp_path / "replacement-valid-schedule.gvd5"
    replacement = V5FrozenLocalSobolSchedule.generate(
        schedule_id="another-valid-schedule", point_count=2, base_seed=18
    )
    write_v5_frozen_local_sobol_schedule(replacement, replacement_path)
    changed_schedule = list(arguments)
    changed_schedule[changed_schedule.index("--local-sobol-schedule") + 1] = str(
        replacement_path
    )
    with pytest.raises(RuntimeError, match="schedule identity changed"):
        pipeline_main(tuple(changed_schedule))
    assert not output.exists()

    calibration_path = tmp_path / "compatibility-calibration.json"
    checked = _calibration(calibration_path)
    formal_arguments = list(arguments)
    for flag in (
        "--standardized-threshold-name",
        "--standardized-threshold-value",
        "--raw-threshold-name",
        "--raw-threshold-value",
        "--threshold-source-id",
    ):
        index = formal_arguments.index(flag)
        del formal_arguments[index : index + 2]
    formal_arguments.extend(
        (
            "--protocol-tier",
            "paper_full_calibrated",
            "--calibration-artifact",
            str(calibration_path),
            "--expected-calibration-sha256",
            checked.identity.artifact_sha256,
            "--expected-calibration-file-sha256",
            checked.identity.file_sha256,
        )
    )
    assert pipeline_main(tuple(formal_arguments)) == 0
    capsys.readouterr()
    for flag, message in (
        ("--expected-calibration-sha256", "logical artifact SHA-256 changed"),
        ("--expected-calibration-file-sha256", "file SHA-256 changed"),
    ):
        changed_calibration = list(formal_arguments)
        changed_calibration[changed_calibration.index(flag) + 1] = "0" * 64
        with pytest.raises(ValueError, match=message):
            pipeline_main(tuple(changed_calibration))
        assert not output.exists()

    same_logical_changed_file = tmp_path / "calibration-reencoded.json"
    same_logical_changed_file.write_bytes(calibration_path.read_bytes() + b"\n")
    changed_calibration = list(formal_arguments)
    changed_calibration[
        changed_calibration.index("--calibration-artifact") + 1
    ] = str(same_logical_changed_file)
    with pytest.raises(ValueError, match="file SHA-256 changed"):
        pipeline_main(tuple(changed_calibration))
    assert not output.exists()


def test_worker_rejects_queue_time_source_drift_before_reading_contracts(
    tmp_path, monkeypatch
):
    from utils.ML_Fitting_1D_GISAXS.PosteriorV8 import (
        run_frozen_search_pipeline_v5 as command,
    )

    monkeypatch.setattr(
        command,
        "fingerprint_v5_frozen_search_source",
        lambda _path: {"bundle_sha256": "b" * 64},
    )
    output = tmp_path / "must-not-exist"
    with pytest.raises(RuntimeError, match="changed after launch planning"):
        pipeline_main(
            (
                "--source-root",
                str(tmp_path),
                "--expected-source-bundle-sha256",
                "a" * 64,
                "--launch-plan-sha256",
                "9" * 64,
                "--split-plan",
                str(tmp_path / "missing-plan"),
                "--expected-split-plan-sha256",
                "c" * 64,
                "--expected-split-plan-file-sha256",
                "1" * 64,
                "--sobol-design",
                str(tmp_path / "missing-design"),
                "--expected-sobol-design-sha256",
                "d" * 64,
                "--expected-sobol-design-file-sha256",
                "2" * 64,
                "--target-split",
                "train",
                "--start",
                "0",
                "--count",
                "1",
                "--topology-schedule-id",
                "k1",
                "--selected-topology-ids",
                "0",
                "--local-sobol-schedule",
                str(tmp_path / "missing-schedule"),
                "--expected-local-schedule-sha256",
                "e" * 64,
                "--expected-local-schedule-artifact-sha256",
                "f" * 64,
                "--expected-local-schedule-manifest-sha256",
                "0" * 64,
                "--optimizer-schedule-id",
                "opt",
                "--direct-scout-seed-count",
                "1",
                "--per-seed-forward-evaluation-limit",
                "1",
                "--protocol-id",
                "p",
                "--standardized-threshold-name",
                "std",
                "--standardized-threshold-value",
                "1",
                "--raw-threshold-name",
                "raw",
                "--raw-threshold-value",
                "1",
                "--threshold-source-id",
                "test",
                "--output-root",
                str(output),
            )
        )
    assert not output.exists()
