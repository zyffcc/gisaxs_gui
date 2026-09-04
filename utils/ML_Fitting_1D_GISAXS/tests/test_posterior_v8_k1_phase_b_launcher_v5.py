from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from hashlib import sha256
import json
import os
from pathlib import Path

import pytest

from utils.ML_Fitting_1D_GISAXS.PosteriorV8 import k1_phase_a_cross_platform_v5
from utils.ML_Fitting_1D_GISAXS.PosteriorV8 import immutable_submission_file_v5
from utils.ML_Fitting_1D_GISAXS.PosteriorV8 import k1_staging_files_v5
from utils.ML_Fitting_1D_GISAXS.PosteriorV8 import launch_k1_phase_b_dag_v5 as launcher
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.k1_phase_b_contract_v5 import (
    PHASE_A_SOURCE_PATHS,
    PHASE_B_SOURCE_PATHS,
    V5_K1_PHASE_B_SCHEMA,
    V5_K1_PHASE_B_VERSION,
    source_identity,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.k1_phase_b_capability_v5 import (
    _consumed_phase_b_capability_payload,
    _mint_phase_b_capability,
    _validate_phase_b_capability,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.k1_phase_b_launch_inputs_v5 import (
    PHASE_A_ROOT_NAME,
    V5K1PhaseBLaunchConfig,
    expected_v5_k1_phase_a_paths,
    inspect_v5_k1_phase_b_source,
    resolve_v5_k1_phase_a_input_files,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.k1_phase_b_launch_chain_v5 import (
    V5K1PhaseBLaunchRuntime,
    copy_v5_k1_phase_b_submission_wrapper,
    inspect_v5_k1_phase_b_launch_chain,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.k1_phase_b_publication_v5 import (
    V5_K1_PHASE_B_COMPLETION_FILENAME,
    publish_v5_k1_phase_b_completed_result,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.k1_phase_b_worker_inputs_v5 import (
    V5K1PhaseBWorkerInputSpec,
    assert_v5_k1_phase_b_worker_inputs_unchanged,
    bind_v5_k1_phase_b_worker_inputs,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.launch_k1_phase_b_dag_v5 import (
    CURRENT_RUN_ROOT_NAME,
    PHASE_ROOT_NAME,
    PLAN_FILENAME,
    RECEIPT_FILENAME,
    V5CommandRequest,
    V5CommandResult,
    V5K1PhaseBLaunchError,
    build_v5_k1_phase_b_launch_plan,
    launch_v5_k1_phase_b_dag,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.package_source_snapshot_v5 import (
    build_source_snapshot,
    extract_source_snapshot,
)


WRAPPER_RELATIVE = Path(
    "utils/ML_Fitting_1D_GISAXS/PosteriorV8/slurm/v5_k1_phase_b_gate_cpu.sbatch"
)
LAUNCHER_TEST_RELATIVE = Path(
    "utils/ML_Fitting_1D_GISAXS/tests/test_posterior_v8_k1_phase_b_launcher_v5.py"
)


class _HeldScheduler:
    def __init__(
        self,
        job_ids: Sequence[str],
        *,
        fail_release_stage: str | None = None,
        hold_reason_overrides: dict[str, str] | None = None,
        dependency_overrides: dict[str, str | None] | None = None,
        release_stays_held: set[str] | None = None,
        on_submit=None,
        on_release=None,
    ) -> None:
        self._job_ids = iter(job_ids)
        self._fail_release_stage = fail_release_stage
        self._hold_reason_overrides = dict(hold_reason_overrides or {})
        self._dependency_overrides = dict(dependency_overrides or {})
        self._release_stays_held = set(release_stays_held or ())
        self._on_submit = on_submit
        self._on_release = on_release
        self.requests: list[V5CommandRequest] = []
        self.dependencies: dict[str, str | None] = {}
        self.stages: dict[str, str] = {}
        self.held: set[str] = set()
        self.cancelled: list[str] = []

    @property
    def calls(self) -> list[tuple[str, ...]]:
        return [request.argv for request in self.requests]

    def __call__(self, request: V5CommandRequest) -> V5CommandResult:
        self.requests.append(request)
        command = request.argv
        if command[0] == "sbatch":
            job_id = next(self._job_ids)
            dependency = next(
                (
                    value.removeprefix("--dependency=afterok:")
                    for value in command
                    if value.startswith("--dependency=afterok:")
                ),
                None,
            )
            stage = "formal_gate" if dependency is not None else "engineering_smoke"
            self.dependencies[job_id] = dependency
            self.stages[job_id] = stage
            self.held.add(job_id)
            if self._on_submit is not None:
                self._on_submit(stage, job_id)
            return V5CommandResult(0, job_id + "\n", "")
        if command[:4] == ("scontrol", "show", "job", "--oneliner"):
            job_id = command[4]
            dependency = self.dependencies[job_id]
            stage = self.stages[job_id]
            reason = self._hold_reason_overrides.get(
                stage,
                (
                    "JobHeldUser"
                    if job_id in self.held
                    else ("Dependency" if dependency else "Priority")
                ),
            )
            reported_dependency = self._dependency_overrides.get(stage, dependency)
            dependency_text = (
                "(null)"
                if reported_dependency is None
                else f"afterok:{reported_dependency}(unfulfilled)"
            )
            return V5CommandResult(
                0,
                f"JobId={job_id} JobState=PENDING Reason={reason} "
                f"Dependency={dependency_text}\n",
                "",
            )
        if command[:2] == ("scontrol", "release"):
            job_id = command[2]
            stage = self.stages[job_id]
            if self._on_release is not None:
                self._on_release(stage, job_id)
            if stage == self._fail_release_stage:
                return V5CommandResult(1, "", "release rejected")
            if stage not in self._release_stays_held:
                self.held.discard(job_id)
            return V5CommandResult(0, "", "")
        if command[0] == "scancel":
            self.cancelled.append(command[1])
            self.held.discard(command[1])
            return V5CommandResult(0, "", "")
        raise AssertionError(f"unexpected scheduler command: {command}")


def _canonical_sha(value: dict[str, object], self_field: str) -> str:
    core = dict(value)
    supplied = core.pop(self_field)
    reproduced = sha256(
        json.dumps(core, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    assert supplied == reproduced
    return reproduced


def _read_only(path: Path, content: bytes) -> tuple[Path, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    path.chmod(0o400)
    return path, sha256(content).hexdigest()


def _stubbed_config(tmp_path: Path, monkeypatch):
    dust = tmp_path / "data/dust/user/zhaiyufe"
    run_root = dust / "MaxwellRuns" / CURRENT_RUN_ROOT_NAME
    (run_root / "logs").mkdir(parents=True)
    source_root = dust / "source-snapshots/new-audited-snapshot"
    source_root.mkdir(parents=True)
    source_wrapper, _ = _read_only(
        source_root / WRAPPER_RELATIVE, b"#!/bin/bash\nset -euo pipefail\n"
    )
    archive, archive_sha = _read_only(
        dust / "source-archives/new-audited-source.tar", b"source archive"
    )
    paths = expected_v5_k1_phase_a_paths(run_root)
    supplied = {}
    for index, (name, path) in enumerate(paths.items(), start=1):
        _, file_sha = _read_only(path, f"{name}-{index}".encode())
        supplied[name] = (path, file_sha)
    config = V5K1PhaseBLaunchConfig(
        source_root=source_root,
        source_archive_path=archive,
        source_archive_sha256=archive_sha,
        run_root=run_root,
        phase_a_dataset_path=supplied["dataset"][0],
        phase_a_dataset_sha256=supplied["dataset"][1],
        phase_a_dataset_binding_path=supplied["dataset_binding"][0],
        phase_a_dataset_binding_sha256=supplied["dataset_binding"][1],
        phase_a_cross_platform_pass_marker_path=supplied["cross_platform_pass_marker"][0],
        phase_a_cross_platform_pass_marker_sha256=supplied["cross_platform_pass_marker"][1],
        phase_a_result_path=supplied["result"][0],
        phase_a_result_sha256=supplied["result"][1],
        phase_a_model_path=supplied["model"][0],
        phase_a_model_sha256=supplied["model"][1],
        phase_a_model_provenance_path=supplied["model_provenance"][0],
        phase_a_model_provenance_sha256=supplied["model_provenance"][1],
    )
    source = {
        "source_root": str(source_root),
        "archive_path": str(archive),
        "archive_sha256": archive_sha,
        "archive_byte_count": archive.stat().st_size,
        "manifest_sha256": "1" * 64,
        "source_tree_sha256": "2" * 64,
        "required_file_identity": {
            WRAPPER_RELATIVE.as_posix(): k1_phase_a_cross_platform_v5.file_identity(
                source_wrapper,
                name="source wrapper",
                require_read_only=True,
            )
        },
    }
    artifacts = {
        name: {
            "path": str(path),
            "sha256": file_sha,
            "byte_count": path.stat().st_size,
            "mode": 0o400,
            "device": 1,
            "inode": index,
            "mtime_ns": index,
        }
        for index, (name, (path, file_sha)) in enumerate(supplied.items(), start=1)
    }
    gate = {
        "gate_claim_sha256": "4" * 64,
        "reference_file_sha256": "5" * 64,
        "reference_file_byte_count": 123,
        "reference_manifest_sha256": "6" * 64,
        "scientific_content_sha256": "7" * 64,
        "comparison_result_sha256": "8" * 64,
    }
    phase_a = {
        "artifacts": artifacts,
        "dataset": {"dataset_file_sha256": artifacts["dataset"]["sha256"]},
        "dataset_validation": {"clean_parent_count": 512},
        "dataset_source_replay": {"all_recorded_sources_reverified": True},
        "dataset_binding_sha256": "9" * 64,
        "cross_platform_gate": gate,
        "cross_platform_gate_claim_sha256": gate["gate_claim_sha256"],
        "phase_a_result_payload_sha256": "a" * 64,
        "phase_a_model_binding_sha256": "b" * 64,
        "phase_a_binding_revalidation": {
            "phase_a_full_pass_reverified": True,
            "source_snapshot_identity": {
                "source_archive_sha256": archive_sha,
                "source_manifest_sha256": "1" * 64,
                "source_tree_sha256": "2" * 64,
            },
        },
        "source_location_semantics": {
            "recorded_phase_a_source_root_is_historical_job_local_execution_path": True
        },
    }
    monkeypatch.setattr(launcher, "inspect_v5_k1_phase_b_source", lambda config, root: source)
    monkeypatch.setattr(
        launcher,
        "inspect_v5_k1_phase_a_inputs",
        lambda config, **kwargs: phase_a,
    )
    return config, dust, source, phase_a


def test_dry_run_is_write_free_and_builds_strict_smoke_formal_afterok_plan(tmp_path, monkeypatch):
    config, dust, source, phase_a = _stubbed_config(tmp_path, monkeypatch)
    result = launch_v5_k1_phase_b_dag(config, allowed_root=dust)
    plan = result["plan"]

    assert result == {
        "status": "dry_run",
        "writes_performed": False,
        "submissions_performed": False,
        "plan": plan,
    }
    assert plan["source"] == source
    assert plan["phase_a_inputs"] == phase_a
    assert plan["stage_order"] == ["engineering_smoke", "formal_gate"]
    assert plan["dependency_edges"] == [["engineering_smoke", "formal_gate"]]
    assert (
        "--dependency=afterok:ENGINEERING_SMOKE_JOB_ID" in plan["submission_preview"]["formal_gate"]
    )
    assert "--hold" in plan["submission_preview"]["formal_gate"]
    assert "--hold" in plan["submission_preview"]["engineering_smoke"]
    assert "--kill-on-invalid-dep=yes" in plan["submission_preview"]["engineering_smoke"]
    assert "--kill-on-invalid-dep=yes" in plan["submission_preview"]["formal_gate"]
    assert plan["jobs"]["engineering_smoke"]["submit_held"] is True
    assert plan["jobs"]["formal_gate"]["submit_held"] is True
    assert plan["jobs"]["formal_gate"]["wrapper"] == plan["layout"]["submission_wrapper"]
    assert (
        plan["jobs"]["formal_gate"]["environment"]["POSTERIOR_V8_V5_K1_PHASE_B_FORMAL_ACK"] == "YES"
    )
    required_evidence_environment = {
        "POSTERIOR_V8_V5_K1_PHASE_B_EXPECTED_GATE_CLAIM_SHA256",
        "POSTERIOR_V8_V5_K1_PHASE_B_EXPECTED_REFERENCE_SHA256",
        "POSTERIOR_V8_V5_K1_PHASE_B_EXPECTED_REFERENCE_BYTE_COUNT",
        "POSTERIOR_V8_V5_K1_PHASE_B_EXPECTED_REFERENCE_MANIFEST_SHA256",
        "POSTERIOR_V8_V5_K1_PHASE_B_EXPECTED_SCIENTIFIC_SHA256",
        "POSTERIOR_V8_V5_K1_PHASE_B_EXPECTED_COMPARISON_SHA256",
    }
    for job in plan["jobs"].values():
        assert required_evidence_environment <= set(job["environment"])
        assert (
            job["environment"]["POSTERIOR_V8_V5_EXPECTED_SOURCE_ARCHIVE_BYTE_COUNT"]
            == source["archive_byte_count"]
        )
        assert job["stdout"].startswith(str(config.run_root / "logs") + "/")
        assert job["stderr"].startswith(str(config.run_root / "logs") + "/")
    assert all("--export=ALL" not in command for command in plan["submission_preview"].values())
    assert "V5_2_R2" in plan["run_root"]
    assert "/GISAXS_ONE_CLICK_PAPER_V5_20260903_V5_2/" not in json.dumps(plan)
    assert plan["full_k1_all_legal_branches_pending_fail_closed"] is True
    _canonical_sha(plan, "plan_sha256")
    assert not (config.run_root / PHASE_ROOT_NAME).exists()


def test_submit_records_exact_afterok_commands_and_self_hashed_read_only_audits(
    tmp_path, monkeypatch
):
    config, dust, _, _ = _stubbed_config(tmp_path, monkeypatch)
    runner = _HeldScheduler(("25001001", "25001002"))

    result = launch_v5_k1_phase_b_dag(
        config,
        submit=True,
        runner=runner,
        allowed_root=dust,
        hostname="max-wgs02.desy.de",
        environment={"AWS_SECRET_ACCESS_KEY": "must-not-be-captured"},
    )

    receipt = result["submission_receipt"]
    calls = runner.calls
    assert result["status"] == "all_jobs_released"
    assert receipt["status"] == "all_jobs_held"
    assert receipt["job_ids"] == {
        "engineering_smoke": "25001001",
        "formal_gate": "25001002",
    }
    submissions = [call for call in calls if call[0] == "sbatch"]
    releases = [call for call in calls if call[:2] == ("scontrol", "release")]
    inspections = [call for call in calls if call[:2] == ("scontrol", "show")]
    assert len(submissions) == 2
    assert len(inspections) == 4
    assert releases == [
        ("scontrol", "release", "25001002"),
        ("scontrol", "release", "25001001"),
    ]
    assert all(
        call[:4]
        == ("sbatch", "--parsable", "--hold", "--kill-on-invalid-dep=yes")
        for call in submissions
    )
    assert "--dependency=afterok:25001001" in submissions[1]
    assert all(any(item.startswith("--output=") for item in call) for call in submissions)
    assert all(any(item.startswith("--error=") for item in call) for call in submissions)
    assert "must-not-be-captured" not in repr(calls)
    assert "AWS_SECRET_ACCESS_KEY" not in repr(calls)
    assert receipt["secret_environment_captured"] is False
    _canonical_sha(receipt, "receipt_sha256")
    _canonical_sha(result["launch_completion"], "launch_completion_sha256")

    audit = config.run_root / PHASE_ROOT_NAME / "audit"
    stored_plan = json.loads((audit / PLAN_FILENAME).read_text())
    stored_receipt = json.loads((audit / RECEIPT_FILENAME).read_text())
    assert stored_receipt == receipt
    assert stored_plan["plan_sha256"] == receipt["plan_sha256"]
    assert (audit / PLAN_FILENAME).stat().st_mode & 0o222 == 0
    assert (audit / RECEIPT_FILENAME).stat().st_mode & 0o222 == 0
    assert Path(stored_plan["layout"]["launch_completion"]).stat().st_mode & 0o222 == 0
    pinned = Path(stored_plan["layout"]["submission_wrapper"])
    assert all(str(pinned) not in call for call in submissions)
    assert all(
        request.script_bytes == pinned.read_bytes()
        for request in runner.requests
        if request.argv[0] == "sbatch"
    )
    assert pinned.stat().st_mode & 0o777 == 0o400
    assert pinned.stat().st_nlink == 1
    assert str(config.source_root / WRAPPER_RELATIVE) not in repr(calls)
    assert not Path(stored_plan["layout"]["smoke_output"]).exists()
    assert not Path(stored_plan["layout"]["formal_output"]).exists()
    with pytest.raises(FileExistsError, match="versioned K1 Phase-B root"):
        build_v5_k1_phase_b_launch_plan(config, allowed_root=dust)


def test_non_max_wgs_and_bad_sbatch_are_fail_closed(tmp_path, monkeypatch):
    config, dust, _, _ = _stubbed_config(tmp_path, monkeypatch)
    called = False

    def runner(argv: Sequence[str]) -> V5CommandResult:
        nonlocal called
        called = True
        return V5CommandResult(0, "1\n", "")

    with pytest.raises(RuntimeError, match="max-wgs"):
        launch_v5_k1_phase_b_dag(
            config,
            submit=True,
            runner=runner,
            allowed_root=dust,
            hostname="local-mac",
        )
    assert called is False
    assert not (config.run_root / PHASE_ROOT_NAME).exists()

    with pytest.raises(V5K1PhaseBLaunchError) as raised:
        launch_v5_k1_phase_b_dag(
            config,
            submit=True,
            runner=lambda argv: V5CommandResult(0, "Submitted batch job 12\n", ""),
            allowed_root=dust,
            hostname="max-wgs01",
        )
    receipt = json.loads(raised.value.receipt_path.read_text())
    assert receipt["status"] == "submission_failed"
    assert receipt["failure"]["stage"] == "engineering_smoke"
    assert receipt["job_ids"] == {}
    _canonical_sha(receipt, "receipt_sha256")


def test_phase_a_drift_between_afterok_submissions_preserves_failure_receipt(tmp_path, monkeypatch):
    config, dust, _, phase_a = _stubbed_config(tmp_path, monkeypatch)
    inspections = 0

    def inspect(config, **kwargs):
        nonlocal inspections
        inspections += 1
        if inspections < 3:
            return phase_a
        return {**phase_a, "phase_a_result_payload_sha256": "0" * 64}

    monkeypatch.setattr(launcher, "inspect_v5_k1_phase_a_inputs", inspect)

    runner = _HeldScheduler(("25002001",))

    with pytest.raises(V5K1PhaseBLaunchError) as raised:
        launch_v5_k1_phase_b_dag(
            config,
            submit=True,
            runner=runner,
            allowed_root=dust,
            hostname="max-wgs03",
    )
    receipt = json.loads(raised.value.receipt_path.read_text())
    assert runner.cancelled == ["25002001"]
    assert receipt["job_ids"] == {"engineering_smoke": "25002001"}
    assert receipt["failure"]["stage"] == "formal_gate"
    assert "Phase-A evidence changed" in receipt["failure"]["message"]


def test_release_failure_preserves_held_receipt_but_no_official_completion(
    tmp_path, monkeypatch
):
    config, dust, _, _ = _stubbed_config(tmp_path, monkeypatch)
    def check_receipt(stage: str, job_id: str) -> None:
        del job_id
        if stage == "formal_gate":
            audit = config.run_root / PHASE_ROOT_NAME / "audit"
            assert (audit / RECEIPT_FILENAME).is_file()
            assert not (audit / launcher.LAUNCH_COMPLETION_FILENAME).exists()

    runner = _HeldScheduler(
        ("25004001", "25004002"),
        fail_release_stage="formal_gate",
        on_release=check_receipt,
    )

    with pytest.raises(V5K1PhaseBLaunchError) as raised:
        launch_v5_k1_phase_b_dag(
            config,
            submit=True,
            runner=runner,
            allowed_root=dust,
            hostname="max-wgs04",
        )
    audit = config.run_root / PHASE_ROOT_NAME / "audit"
    receipt = json.loads((audit / RECEIPT_FILENAME).read_text())
    failure = json.loads(raised.value.receipt_path.read_text())
    assert receipt["status"] == "all_jobs_held"
    assert receipt["formal_release_completed"] is False
    assert failure["status"] == "RELEASE_FAILED_NO_LAUNCH_COMPLETION"
    assert failure["official_launch_chain_complete"] is False
    assert not (audit / launcher.LAUNCH_COMPLETION_FILENAME).exists()
    assert runner.cancelled == ["25004002", "25004001"]
    _canonical_sha(failure, "release_failure_sha256")


def test_scheduler_must_confirm_initial_hold_and_cancels_unproven_job(
    tmp_path, monkeypatch
):
    config, dust, _, _ = _stubbed_config(tmp_path, monkeypatch)
    runner = _HeldScheduler(
        ("25004101",), hold_reason_overrides={"engineering_smoke": "Priority"}
    )

    with pytest.raises(V5K1PhaseBLaunchError) as raised:
        launch_v5_k1_phase_b_dag(
            config,
            submit=True,
            runner=runner,
            allowed_root=dust,
            hostname="max-wgs04",
        )
    receipt = json.loads(raised.value.receipt_path.read_text())
    assert receipt["status"] == "submission_failed"
    assert "did not retain" in receipt["failure"]["message"]
    assert runner.cancelled == ["25004101"]
    assert not (
        config.run_root / PHASE_ROOT_NAME / "audit" / launcher.LAUNCH_COMPLETION_FILENAME
    ).exists()


def test_scheduler_dependency_readback_must_match_and_cancels_both_jobs(
    tmp_path, monkeypatch
):
    config, dust, _, _ = _stubbed_config(tmp_path, monkeypatch)
    runner = _HeldScheduler(
        ("25004201", "25004202"),
        dependency_overrides={"formal_gate": "99999999"},
    )

    with pytest.raises(V5K1PhaseBLaunchError) as raised:
        launch_v5_k1_phase_b_dag(
            config,
            submit=True,
            runner=runner,
            allowed_root=dust,
            hostname="max-wgs04",
        )
    receipt = json.loads(raised.value.receipt_path.read_text())
    assert receipt["status"] == "submission_failed"
    assert "dependency drifted" in receipt["failure"]["message"]
    assert runner.cancelled == ["25004202", "25004201"]


def test_release_readback_must_clear_user_hold_before_official_completion(
    tmp_path, monkeypatch
):
    config, dust, _, _ = _stubbed_config(tmp_path, monkeypatch)
    runner = _HeldScheduler(
        ("25004301", "25004302"), release_stays_held={"formal_gate"}
    )

    with pytest.raises(V5K1PhaseBLaunchError) as raised:
        launch_v5_k1_phase_b_dag(
            config,
            submit=True,
            runner=runner,
            allowed_root=dust,
            hostname="max-wgs04",
        )
    failure = json.loads(raised.value.receipt_path.read_text())
    assert failure["status"] == "RELEASE_FAILED_NO_LAUNCH_COMPLETION"
    assert "user-held after release" in failure["failure"]["message"]
    assert runner.cancelled == ["25004302", "25004301"]
    assert not (
        config.run_root / PHASE_ROOT_NAME / "audit" / launcher.LAUNCH_COMPLETION_FILENAME
    ).exists()


def test_production_maxwell_root_forbids_all_injected_submission_seams(
    tmp_path, monkeypatch
):
    config, _, _, _ = _stubbed_config(tmp_path, monkeypatch)
    with pytest.raises(RuntimeError, match="forbids injected test seams"):
        launch_v5_k1_phase_b_dag(
            config,
            submit=True,
            runner=_HeldScheduler(("25004401", "25004402")),
        )


def test_formal_chain_requires_release_completion_and_bound_completed_smoke(
    tmp_path, monkeypatch
):
    config, dust, _, _ = _stubbed_config(tmp_path, monkeypatch)
    runner = _HeldScheduler(("25005001", "25005002"))
    result = launch_v5_k1_phase_b_dag(
        config,
        submit=True,
        runner=runner,
        allowed_root=dust,
        hostname="max-wgs05",
    )
    audit = config.run_root / PHASE_ROOT_NAME / "audit"
    plan_path = audit / PLAN_FILENAME
    plan = json.loads(plan_path.read_text())
    smoke_runtime = _launch_runtime(plan_path, "engineering_smoke")
    smoke_output = Path(plan["layout"]["smoke_output"])
    smoke_chain = inspect_v5_k1_phase_b_launch_chain(
        smoke_runtime,
        output_dir=smoke_output,
        slurm_job_id="25005001",
    )
    smoke_output.parent.mkdir(parents=True, exist_ok=True)
    smoke_capability, smoke_capability_payload = _consumed_capability(smoke_chain)
    publish_v5_k1_phase_b_completed_result(
        smoke_output,
        _smoke_result(smoke_chain, smoke_capability_payload),
        launch_binding=smoke_chain,
        capability=smoke_capability,
    )

    formal_runtime = _launch_runtime(plan_path, "formal_gate")
    formal_output = Path(plan["layout"]["formal_output"])
    evidence = inspect_v5_k1_phase_b_launch_chain(
        formal_runtime,
        output_dir=formal_output,
        slurm_job_id="25005002",
    )
    assert evidence["formal_prerequisite_evidence"]["upstream_smoke_job_id"] == (
        "25005001"
    )
    assert evidence["formal_prerequisite_evidence"]["formal_job_id"] == "25005002"
    assert evidence["formal_prerequisite_evidence"]["upstream_smoke"][
        "result_payload_sha256"
    ]
    assert result["launch_completion"]["official_launch_chain_complete"] is True

    with pytest.raises(ValueError, match="not the receipt-bound stage job"):
        inspect_v5_k1_phase_b_launch_chain(
            formal_runtime,
            output_dir=formal_output,
            slurm_job_id="99999999",
        )
    smoke_completion = smoke_output / V5_K1_PHASE_B_COMPLETION_FILENAME
    smoke_completion_bytes = smoke_completion.read_bytes()
    smoke_output.chmod(0o700)
    smoke_completion.unlink()
    smoke_output.chmod(0o500)
    assert (smoke_output / "phase-b-result.json").is_file()
    with pytest.raises(FileNotFoundError):
        inspect_v5_k1_phase_b_launch_chain(
            formal_runtime,
            output_dir=formal_output,
            slurm_job_id="25005002",
        )
    smoke_output.chmod(0o700)
    smoke_completion.write_bytes(smoke_completion_bytes)
    smoke_completion.chmod(0o400)
    smoke_output.chmod(0o500)
    smoke_output.chmod(0o700)
    with pytest.raises(ValueError, match="remains writable"):
        inspect_v5_k1_phase_b_launch_chain(
            formal_runtime,
            output_dir=formal_output,
            slurm_job_id="25005002",
        )
    smoke_output.chmod(0o500)
    Path(plan["layout"]["launch_completion"]).unlink()
    with pytest.raises(FileNotFoundError):
        inspect_v5_k1_phase_b_launch_chain(
            formal_runtime,
            output_dir=formal_output,
            slurm_job_id="25005002",
        )
    receipt_path = Path(plan["layout"]["receipt"])
    receipt_payload = json.loads(receipt_path.read_text())
    receipt_payload["receipt_sha256"] = "0" * 64
    receipt_path.unlink()
    receipt_path.write_text(json.dumps(receipt_payload))
    receipt_path.chmod(0o400)
    with pytest.raises(ValueError, match="self SHA-256 does not reproduce"):
        inspect_v5_k1_phase_b_launch_chain(
            formal_runtime,
            output_dir=formal_output,
            slurm_job_id="25005002",
        )


def test_old_diagnostic_run_and_phase_a_roots_are_rejected(tmp_path, monkeypatch):
    config, dust, _, _ = _stubbed_config(tmp_path, monkeypatch)
    old_run = dust / "MaxwellRuns/GISAXS_ONE_CLICK_PAPER_V5_20260903_V5_2"
    (old_run / "logs").mkdir(parents=True)
    with pytest.raises(ValueError, match="exactly"):
        build_v5_k1_phase_b_launch_plan(replace(config, run_root=old_run), allowed_root=dust)

    old_dataset, old_sha = _read_only(
        old_run / "k1_phase_a_v5_2_dag_v5/datasets/k1-v5-2-phase-a-v5-full-r512-sphere-v0.gvd5",
        b"old diagnostic dataset",
    )
    with pytest.raises(ValueError, match="exact full-run artifact"):
        resolve_v5_k1_phase_a_input_files(
            replace(
                config,
                phase_a_dataset_path=old_dataset,
                phase_a_dataset_sha256=old_sha,
            ),
            run_root=config.run_root,
            allowed_root=dust,
        )
    assert CURRENT_RUN_ROOT_NAME.endswith("V5_2_R2")
    assert PHASE_A_ROOT_NAME.endswith("dag_v13")


def test_phase_b_refuses_phase_a_artifacts_without_full_gate_completion(
    tmp_path, monkeypatch
):
    config, dust, _, _ = _stubbed_config(tmp_path, monkeypatch)
    completion = expected_v5_k1_phase_a_paths(config.run_root)["phase_a_completion"]
    completion.unlink()

    with pytest.raises(FileNotFoundError):
        resolve_v5_k1_phase_a_input_files(
            config,
            run_root=config.run_root,
            allowed_root=dust,
        )


def _synthetic_source_tree(root: Path) -> Path:
    required = {
        Path("AGENTS.md"),
        Path("pyproject.toml"),
        Path("requirements.txt"),
        Path("requirements-dev.txt"),
        Path("utils/__init__.py"),
        Path("src/gimap/example.py"),
        Path("docs/architecture/example.md"),
        Path("docs/research/example.md"),
        LAUNCHER_TEST_RELATIVE,
        *(Path(value) for value in PHASE_A_SOURCE_PATHS),
        *(Path(value) for value in PHASE_B_SOURCE_PATHS),
    }
    for relative in sorted(required, key=lambda value: value.as_posix()):
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f"immutable test source: {relative.as_posix()}\n")
    return root


def test_archive_snapshot_replay_does_not_depend_on_deleted_phase_a_scratch(tmp_path):
    dust = tmp_path / "data/dust/user/zhaiyufe"
    working = _synthetic_source_tree(tmp_path / "working")
    archive = dust / "source-archives/phase-b-audited.tar"
    archive.parent.mkdir(parents=True)
    archive_identity = build_source_snapshot(working, archive)
    extracted = dust / "source-snapshots/phase-b-audited"
    extracted.parent.mkdir(parents=True)
    extract_source_snapshot(
        archive,
        extracted,
        expected_sha256=archive_identity["archive_sha256"],
    )
    deleted_source = tmp_path / "slurm-24398910/source"
    deleted_dataset = tmp_path / "slurm-24398910/input.gvd5"
    deleted_source.mkdir(parents=True)
    deleted_dataset.write_bytes(b"transient")
    deleted_dataset.unlink()
    deleted_source.rmdir()
    deleted_source.parent.rmdir()
    placeholder = dust / "unused"
    config = V5K1PhaseBLaunchConfig(
        source_root=extracted,
        source_archive_path=archive,
        source_archive_sha256=archive_identity["archive_sha256"],
        run_root=placeholder,
        phase_a_dataset_path=placeholder,
        phase_a_dataset_sha256="0" * 64,
        phase_a_dataset_binding_path=placeholder,
        phase_a_dataset_binding_sha256="0" * 64,
        phase_a_cross_platform_pass_marker_path=placeholder,
        phase_a_cross_platform_pass_marker_sha256="0" * 64,
        phase_a_result_path=placeholder,
        phase_a_result_sha256="0" * 64,
        phase_a_model_path=placeholder,
        phase_a_model_sha256="0" * 64,
        phase_a_model_provenance_path=placeholder,
        phase_a_model_provenance_sha256="0" * 64,
    )

    observed = inspect_v5_k1_phase_b_source(config, dust)
    replayed_phase_a = source_identity(extracted, tuple(sorted(PHASE_A_SOURCE_PATHS)))
    assert observed["archive_sha256"] == archive_identity["archive_sha256"]
    assert observed["manifest_sha256"] == archive_identity["manifest_sha256"]
    assert len(observed["source_tree_sha256"]) == 64
    assert replayed_phase_a["bundle_sha256"]
    assert not deleted_source.exists()
    assert not deleted_dataset.exists()

    selected = extracted / next(iter(sorted(PHASE_A_SOURCE_PATHS)))
    selected.chmod(0o644)
    with pytest.raises(ValueError, match="read-only|identity mismatch"):
        inspect_v5_k1_phase_b_source(config, dust)


def test_slurm_wrapper_revalidates_every_input_before_isolated_worker_execution():
    wrapper = (
        Path(__file__).resolve().parents[1]
        / WRAPPER_RELATIVE.relative_to("utils/ML_Fitting_1D_GISAXS")
    ).read_text()
    required_environment = (
        "POSTERIOR_V8_V5_SOURCE_ARCHIVE",
        "POSTERIOR_V8_V5_EXPECTED_SOURCE_ARCHIVE_SHA256",
        "POSTERIOR_V8_V5_EXPECTED_SOURCE_ARCHIVE_BYTE_COUNT",
        "POSTERIOR_V8_V5_EXPECTED_SOURCE_MANIFEST_SHA256",
        "POSTERIOR_V8_V5_EXPECTED_SOURCE_TREE_SHA256",
        "POSTERIOR_V8_V5_K1_PHASE_B_DATASET",
        "POSTERIOR_V8_V5_K1_PHASE_B_DATASET_SHA256",
        "POSTERIOR_V8_V5_K1_PHASE_B_DATASET_BYTE_COUNT",
        "POSTERIOR_V8_V5_K1_PHASE_B_DATASET_BINDING",
        "POSTERIOR_V8_V5_K1_PHASE_B_DATASET_BINDING_SHA256",
        "POSTERIOR_V8_V5_K1_PHASE_B_DATASET_BINDING_BYTE_COUNT",
        "POSTERIOR_V8_V5_K1_PHASE_B_CROSS_PLATFORM_PASS_MARKER",
        "POSTERIOR_V8_V5_K1_PHASE_B_CROSS_PLATFORM_PASS_MARKER_SHA256",
        "POSTERIOR_V8_V5_K1_PHASE_B_CROSS_PLATFORM_PASS_MARKER_BYTE_COUNT",
        "POSTERIOR_V8_V5_K1_PHASE_B_RESULT",
        "POSTERIOR_V8_V5_K1_PHASE_B_RESULT_SHA256",
        "POSTERIOR_V8_V5_K1_PHASE_B_RESULT_BYTE_COUNT",
        "POSTERIOR_V8_V5_K1_PHASE_B_MODEL",
        "POSTERIOR_V8_V5_K1_PHASE_B_MODEL_SHA256",
        "POSTERIOR_V8_V5_K1_PHASE_B_MODEL_BYTE_COUNT",
        "POSTERIOR_V8_V5_K1_PHASE_B_MODEL_PROVENANCE",
        "POSTERIOR_V8_V5_K1_PHASE_B_MODEL_PROVENANCE_SHA256",
        "POSTERIOR_V8_V5_K1_PHASE_B_MODEL_PROVENANCE_BYTE_COUNT",
        "POSTERIOR_V8_V5_K1_PHASE_B_PHASE_A_COMPLETION",
        "POSTERIOR_V8_V5_K1_PHASE_B_PHASE_A_COMPLETION_SHA256",
        "POSTERIOR_V8_V5_K1_PHASE_B_PHASE_A_COMPLETION_BYTE_COUNT",
        "POSTERIOR_V8_V5_K1_PHASE_B_EXPECTED_GATE_CLAIM_SHA256",
        "POSTERIOR_V8_V5_K1_PHASE_B_EXPECTED_REFERENCE_SHA256",
        "POSTERIOR_V8_V5_K1_PHASE_B_EXPECTED_REFERENCE_BYTE_COUNT",
        "POSTERIOR_V8_V5_K1_PHASE_B_EXPECTED_REFERENCE_MANIFEST_SHA256",
        "POSTERIOR_V8_V5_K1_PHASE_B_EXPECTED_SCIENTIFIC_SHA256",
        "POSTERIOR_V8_V5_K1_PHASE_B_EXPECTED_COMPARISON_SHA256",
        "POSTERIOR_V8_V5_K1_PHASE_B_OUTPUT",
        "POSTERIOR_V8_V5_K1_PHASE_B_MODE",
        "POSTERIOR_V8_V5_K1_PHASE_B_LAUNCH_STAGE",
        "POSTERIOR_V8_V5_K1_PHASE_B_LAUNCH_PLAN",
        "POSTERIOR_V8_V5_K1_PHASE_B_LAUNCH_PLAN_SHA256",
        "POSTERIOR_V8_V5_K1_PHASE_B_LAUNCH_PLAN_FILE_SHA256",
        "POSTERIOR_V8_V5_K1_PHASE_B_LAUNCH_PLAN_BYTE_COUNT",
        "POSTERIOR_V8_V5_K1_PHASE_B_LAUNCH_PLAN_MODE",
        "POSTERIOR_V8_V5_K1_PHASE_B_LAUNCH_PLAN_DEVICE",
        "POSTERIOR_V8_V5_K1_PHASE_B_LAUNCH_PLAN_INODE",
        "POSTERIOR_V8_V5_K1_PHASE_B_LAUNCH_PLAN_MTIME_NS",
        "POSTERIOR_V8_V5_K1_PHASE_B_LAUNCH_PLAN_CTIME_NS",
        "POSTERIOR_V8_V5_K1_PHASE_B_LAUNCH_PLAN_NLINK",
        "POSTERIOR_V8_V5_K1_PHASE_B_LAUNCH_RECEIPT",
        "POSTERIOR_V8_V5_K1_PHASE_B_LAUNCH_COMPLETION",
    )
    for name in required_environment:
        assert f"${{{name}:?" in wrapper
    assert "GISAXS_ONE_CLICK_PAPER_V5_20260903_V5_2_R2" in wrapper
    assert "k1_phase_a_v5_2_dag_v13" in wrapper
    assert "k1_phase_a_v5_2_dag_v6" not in wrapper
    assert "k1_phase_b_v5_2_dag_v3" in wrapper
    assert "PYTHONNOUSERSITE=1" in wrapper
    assert "unset PYTHONPATH" in wrapper
    assert "python -I -c" in wrapper
    assert "python -m utils.ML_Fitting_1D_GISAXS" not in wrapper
    assert 'copy_checked "$POSTERIOR_V8_V5_K1_PHASE_B_MODEL"' in wrapper
    assert "verify-extracted" in wrapper
    binding = wrapper.index("PosteriorV8.k1_phase_a_dataset_binding_v5")
    worker = wrapper.index("PosteriorV8.run_k1_phase_b_gate_v5")
    assert wrapper.index("verify-extracted") < binding < worker
    assert '--original-dataset-path "$POSTERIOR_V8_V5_K1_PHASE_B_DATASET"' in wrapper
    assert '--original-phase-a-output-dir "$expected_model_root"' in wrapper
    assert '--original-source-root "$POSTERIOR_V8_SOURCE_ROOT"' in wrapper
    assert '--original-source-archive "$POSTERIOR_V8_V5_SOURCE_ARCHIVE"' in wrapper
    assert '--original-phase-a-result "$POSTERIOR_V8_V5_K1_PHASE_B_RESULT"' in wrapper
    assert '--original-phase-a-model "$POSTERIOR_V8_V5_K1_PHASE_B_MODEL"' in wrapper
    assert '--dataset-binding "$job_binding"' in wrapper
    assert '--cross-platform-pass-marker "$job_marker"' in wrapper
    assert '--phase-a-completion "$job_phase_a_completion"' in wrapper
    assert "verify_checked" in wrapper
    assert "source archive after worker" in wrapper
    assert "dataset binding after worker" in wrapper
    assert "PASS marker after worker" in wrapper
    assert "Phase-A completion after worker" in wrapper
    assert 'exit "$worker_status"' in wrapper
    assert '--launch-plan "$POSTERIOR_V8_V5_K1_PHASE_B_LAUNCH_PLAN"' in wrapper
    assert '--launch-receipt "$POSTERIOR_V8_V5_K1_PHASE_B_LAUNCH_RECEIPT"' in wrapper
    assert "Phase-B launch completion did not appear" in wrapper
    worker_source = (
        Path(__file__).resolve().parents[1]
        / "PosteriorV8/run_k1_phase_b_gate_v5.py"
    ).read_text()
    assert "publish_v5_k1_phase_b_completed_result" in worker_source
    assert "assert_v5_k1_phase_b_launch_chain_unchanged" in worker_source
    assert "external_smoke_review" not in worker_source


def _worker_input_spec(tmp_path: Path):
    dust = tmp_path / "data/dust/user/zhaiyufe"
    working = _synthetic_source_tree(tmp_path / "authoring")
    archive = dust / "source-archives/audited.tar"
    archive.parent.mkdir(parents=True)
    archive_identity = build_source_snapshot(working, archive)
    source_root = dust / "source-snapshots/audited"
    source_root.parent.mkdir(parents=True)
    source_snapshot = extract_source_snapshot(
        archive, source_root, expected_sha256=archive_identity["archive_sha256"]
    )

    phase_a_output = dust / "MaxwellRuns/audited/models/full"
    dataset, _ = _read_only(dust / "MaxwellRuns/audited/dataset.gvd5", b"dataset")
    dataset_binding, _ = _read_only(
        Path(str(dataset) + ".binding-v1.json"), b"dataset binding"
    )
    audit_root = phase_a_output.parents[1] / "audit"
    pass_marker, _ = _read_only(
        audit_root / "sobol-cross-platform-PASS-v1.json", b"pass marker"
    )
    result, _ = _read_only(phase_a_output / "result.json", b"result")
    model, _ = _read_only(phase_a_output / "model.keras", b"model")
    provenance, _ = _read_only(phase_a_output / "model.provenance.json", b"provenance")
    phase_a_completion, _ = _read_only(
        audit_root / "full-gate-completion-v1.json", b"completion"
    )
    authoritative_paths = {
        "source_archive": archive,
        "dataset": dataset,
        "dataset_binding": dataset_binding,
        "cross_platform_pass_marker": pass_marker,
        "phase_a_result": result,
        "phase_a_model": model,
        "phase_a_model_provenance": provenance,
        "phase_a_completion": phase_a_completion,
    }
    expected = {
        name: {
            "sha256": sha256(path.read_bytes()).hexdigest(),
            "byte_count": path.stat().st_size,
        }
        for name, path in authoritative_paths.items()
    }

    job_root = tmp_path / "job"
    job_root.mkdir()
    job_archive, _ = _read_only(job_root / "source.tar", archive.read_bytes())
    job_source = job_root / "source"
    extract_source_snapshot(
        job_archive,
        job_source,
        expected_sha256=archive_identity["archive_sha256"],
    )
    local_paths = {}
    for name, original in authoritative_paths.items():
        if name == "source_archive":
            local_paths[name] = job_archive
        else:
            local_paths[name], _ = _read_only(
                job_root / "inputs" / original.name, original.read_bytes()
            )
    spec = V5K1PhaseBWorkerInputSpec(
        allowed_root=dust,
        input_root=job_root,
        original_source_root=source_root,
        job_source_root=job_source,
        phase_a_output_dir=phase_a_output,
        authoritative_paths=authoritative_paths,
        local_paths=local_paths,
        expected=expected,
        source_manifest_sha256=source_snapshot["manifest_sha256"],
        source_tree_sha256=source_snapshot["source_tree_sha256"],
    )
    return spec, authoritative_paths, local_paths


def _replace_read_only(path: Path, content: bytes) -> None:
    # Keep the unlinked inode alive until its replacement exists.  Otherwise a
    # fast filesystem may immediately recycle the same inode and timestamp,
    # making this adversarial fixture indistinguishable from the original file.
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        path.unlink()
        path.write_bytes(content)
        path.chmod(0o400)
    finally:
        os.close(descriptor)


def test_worker_rechecks_authoritative_originals_and_local_copies_after_binding(tmp_path):
    spec, originals, local = _worker_input_spec(tmp_path)
    bound = bind_v5_k1_phase_b_worker_inputs(spec)
    assert bound["authoritative"]["dataset"]["path"] == str(originals["dataset"])
    assert bound["local"]["dataset"]["path"] == str(local["dataset"])

    dataset_bytes = originals["dataset"].read_bytes()
    _replace_read_only(originals["dataset"], dataset_bytes)
    with pytest.raises(RuntimeError, match="changed during Phase-B"):
        assert_v5_k1_phase_b_worker_inputs_unchanged(bound, spec)

    rebound = bind_v5_k1_phase_b_worker_inputs(spec)
    archive_bytes = originals["source_archive"].read_bytes()
    _replace_read_only(originals["source_archive"], archive_bytes)
    with pytest.raises(RuntimeError, match="changed during Phase-B"):
        assert_v5_k1_phase_b_worker_inputs_unchanged(rebound, spec)

    rebound = bind_v5_k1_phase_b_worker_inputs(spec)
    _replace_read_only(originals["phase_a_result"], b"changed-result")
    with pytest.raises(ValueError, match="exported identity"):
        assert_v5_k1_phase_b_worker_inputs_unchanged(rebound, spec)

    _replace_read_only(originals["phase_a_result"], b"result")
    _replace_read_only(local["phase_a_model"], b"changed-local-model")
    with pytest.raises(ValueError, match="job-local phase_a_model"):
        bind_v5_k1_phase_b_worker_inputs(spec)


def test_strict_identity_rejects_final_symlink_and_open_file_replacement(tmp_path, monkeypatch):
    target, _ = _read_only(tmp_path / "target.bin", b"original")
    replacement, _ = _read_only(tmp_path / "replacement.bin", b"replacement")
    real_open = k1_phase_a_cross_platform_v5.os.open

    def replace_with_symlink(path, flags):
        target.unlink()
        target.symlink_to(replacement)
        return real_open(path, flags)

    with monkeypatch.context() as scoped:
        scoped.setattr(k1_phase_a_cross_platform_v5.os, "open", replace_with_symlink)
        with pytest.raises(ValueError, match="symlink before open"):
            k1_phase_a_cross_platform_v5.file_identity(
                target, name="racing input", require_read_only=True
            )

    target.unlink()
    target.write_bytes(b"original")
    target.chmod(0o400)
    real_fstat = k1_staging_files_v5.os.fstat
    fstat_calls = 0

    def replace_while_reading(descriptor):
        nonlocal fstat_calls
        fstat_calls += 1
        if fstat_calls == 2:
            _replace_read_only(target, b"replacement")
        return real_fstat(descriptor)

    with monkeypatch.context() as scoped:
        scoped.setattr(k1_staging_files_v5.os, "fstat", replace_while_reading)
        with pytest.raises(RuntimeError, match="changed or was replaced"):
            k1_phase_a_cross_platform_v5.file_identity(
                target, name="racing input", require_read_only=True
            )


def _launch_runtime(plan_path: Path, stage: str) -> V5K1PhaseBLaunchRuntime:
    plan = json.loads(plan_path.read_text())
    identity = k1_phase_a_cross_platform_v5.file_identity(
        plan_path, name="test launch plan", require_read_only=True
    )
    return V5K1PhaseBLaunchRuntime(
        plan_path=plan_path,
        expected_plan_sha256=plan["plan_sha256"],
        expected_plan_file_identity=identity,
        receipt_path=Path(plan["layout"]["receipt"]),
        launch_completion_path=Path(plan["layout"]["launch_completion"]),
        launch_stage=stage,
    )


def _consumed_capability(launch_chain, recheck=None):
    selected_recheck = (lambda: None) if recheck is None else recheck
    capability = _mint_phase_b_capability(launch_chain, selected_recheck)
    _validate_phase_b_capability(capability, phase="pre_execution")
    _validate_phase_b_capability(capability, phase="post_execution")
    return capability, _consumed_phase_b_capability_payload(capability)


def _smoke_result(
    launch_chain: dict[str, object], capability_payload: dict[str, object]
) -> dict[str, object]:
    core = {
        "schema_version": V5_K1_PHASE_B_SCHEMA,
        "version": V5_K1_PHASE_B_VERSION,
        "status": "engineering_throughput_smoke_completed_fail_closed",
        "launch_chain": launch_chain,
        "job_local_capability": capability_payload,
    }
    return {
        **core,
        "result_payload_sha256": sha256(
            json.dumps(
                core, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode()
        ).hexdigest(),
    }


def test_completion_last_publication_requires_live_single_use_capability(
    tmp_path,
):
    output = tmp_path / "results/smoke"
    output.parent.mkdir(parents=True)
    launch_chain = {
        "launch_stage": "engineering_smoke",
        "slurm_job_id": "25003001",
        "launch_plan": {"plan_sha256": "1" * 64},
        "formal_prerequisite_evidence": None,
        "formal_prerequisite_evidence_sha256": None,
    }
    calls = 0

    def fail_during_publication_recheck() -> None:
        nonlocal calls
        calls += 1
        if calls == 4:
            assert output.is_dir()
            assert (output / "phase-b-result.json").is_file()
            assert not (output / V5_K1_PHASE_B_COMPLETION_FILENAME).exists()
            raise RuntimeError("authoritative input changed")

    failed_capability, failed_payload = _consumed_capability(
        launch_chain, fail_during_publication_recheck
    )

    with pytest.raises(RuntimeError, match="authoritative input changed"):
        publish_v5_k1_phase_b_completed_result(
            output,
            _smoke_result(launch_chain, failed_payload),
            launch_binding=launch_chain,
            capability=failed_capability,
        )
    assert calls == 4
    assert not output.exists()

    capability, capability_payload = _consumed_capability(launch_chain)
    published = publish_v5_k1_phase_b_completed_result(
        output,
        _smoke_result(launch_chain, capability_payload),
        launch_binding=launch_chain,
        capability=capability,
    )
    result_path = output / "phase-b-result.json"
    completion_path = output / V5_K1_PHASE_B_COMPLETION_FILENAME
    assert published["completion"]["status"] == "COMPLETE"
    assert result_path.stat().st_mode & 0o777 == 0o400
    assert completion_path.stat().st_mode & 0o777 == 0o400
    assert result_path.stat().st_nlink == completion_path.stat().st_nlink == 1
    assert output.stat().st_mode & 0o222 == 0
    completion = json.loads(completion_path.read_text())
    _canonical_sha(completion, "completion_payload_sha256")
    assert completion["result"]["sha256"] == sha256(result_path.read_bytes()).hexdigest()
    assert completion["result"]["byte_count"] == result_path.stat().st_size
    assert completion["result"]["device"] == result_path.stat().st_dev
    assert completion["result"]["inode"] == result_path.stat().st_ino
    assert completion["result"]["nlink"] == 1
    assert completion["job_local_capability"] == capability_payload
    second_output = output.parent / "forged-second-completion"
    with pytest.raises(RuntimeError, match="post-execution validation"):
        publish_v5_k1_phase_b_completed_result(
            second_output,
            _smoke_result(launch_chain, capability_payload),
            launch_binding=launch_chain,
            capability=capability,
        )
    assert not second_output.exists()


def test_strict_identity_rejects_hardlinks_and_same_content_replacement(tmp_path):
    target, _ = _read_only(tmp_path / "target.bin", b"same bytes")
    alias = tmp_path / "hardlink.bin"
    alias.hardlink_to(target)
    with pytest.raises(ValueError, match="exactly one hard link"):
        k1_phase_a_cross_platform_v5.file_identity(
            target, name="hardlinked input", require_read_only=True
        )
    alias.unlink()
    before = k1_phase_a_cross_platform_v5.file_identity(
        target, name="replaceable input", require_read_only=True
    )
    _replace_read_only(target, b"same bytes")
    after = k1_phase_a_cross_platform_v5.file_identity(
        target, name="replacement input", require_read_only=True
    )
    assert before["sha256"] == after["sha256"]
    assert before["byte_count"] == after["byte_count"]
    assert before != after
    assert {"device", "inode", "mode", "mtime_ns", "ctime_ns", "nlink"} <= set(after)


def test_pinned_wrapper_copy_rejects_preoccupation_and_source_replacement(
    tmp_path, monkeypatch
):
    source, _ = _read_only(tmp_path / "source.sbatch", b"#!/bin/bash\nexit 0\n")
    expected = k1_phase_a_cross_platform_v5.file_identity(
        source, name="source wrapper", require_read_only=True
    )
    submission = tmp_path / "submission"
    submission.mkdir(mode=0o700)
    preoccupied, _ = _read_only(submission / "pinned.sbatch", b"occupied")
    with pytest.raises(FileExistsError, match="preoccupied"):
        copy_v5_k1_phase_b_submission_wrapper(
            source, preoccupied, expected_source_identity=expected
        )

    preoccupied.unlink()
    real_open = immutable_submission_file_v5.os.open

    def preoccupy_during_exclusive_open(path, flags, *args):
        if Path(path) == preoccupied and flags & immutable_submission_file_v5.os.O_EXCL:
            preoccupied.write_bytes(b"racing owner file")
            preoccupied.chmod(0o400)
        return real_open(path, flags, *args)

    with monkeypatch.context() as scoped:
        scoped.setattr(
            immutable_submission_file_v5.os, "open", preoccupy_during_exclusive_open
        )
        with pytest.raises(FileExistsError):
            copy_v5_k1_phase_b_submission_wrapper(
                source, preoccupied, expected_source_identity=expected
            )
    assert preoccupied.read_bytes() == b"racing owner file"
    preoccupied.unlink()
    real_read = immutable_submission_file_v5.os.read
    source_open_count = 0
    copy_fd = -1
    replaced = False

    def tracked_open(path, flags, *args):
        nonlocal source_open_count, copy_fd
        descriptor = real_open(path, flags, *args)
        if Path(path) == source:
            source_open_count += 1
            if source_open_count == 2:
                copy_fd = descriptor
        return descriptor

    def replace_during_copy(descriptor, byte_count):
        nonlocal replaced
        data = real_read(descriptor, byte_count)
        if descriptor == copy_fd and data and not replaced:
            replaced = True
            _replace_read_only(source, b"#!/bin/bash\nexit 0\n")
        return data

    with monkeypatch.context() as scoped:
        scoped.setattr(immutable_submission_file_v5.os, "open", tracked_open)
        scoped.setattr(immutable_submission_file_v5.os, "read", replace_during_copy)
        with pytest.raises(RuntimeError, match="changed during pinned copy"):
            copy_v5_k1_phase_b_submission_wrapper(
                source, preoccupied, expected_source_identity=expected
            )
    assert replaced is True
    assert not preoccupied.exists()
