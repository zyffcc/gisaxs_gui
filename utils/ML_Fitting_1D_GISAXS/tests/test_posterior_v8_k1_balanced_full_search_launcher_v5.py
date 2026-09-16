from __future__ import annotations

from hashlib import sha256
from importlib import import_module
from pathlib import Path, PurePosixPath
from typing import Sequence

import pytest

from utils.ML_Fitting_1D_GISAXS.PosteriorV8.grouped_artifact_v5 import (
    canonical_json,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8 import (
    launch_k1_balanced_full_search_dag_v5 as launcher,
)
from utils.ML_Fitting_1D_GISAXS.tests.test_posterior_v8_k1_balanced_full_search_plan_v5 import (
    _plan,
)


def _sha(value: str) -> str:
    return sha256(value.encode("ascii")).hexdigest()


@pytest.mark.parametrize("host", ["max-wgs", "max-wgs001.desy.de", "max-fs-display006.desy.de"])
def test_submission_host_allowlist(host):
    from utils.ML_Fitting_1D_GISAXS.PosteriorV8.launch_k1_balanced_dataset_dag_v5 import (
        _require_submission_host,
    )
    _require_submission_host(host)


@pytest.mark.parametrize("host", ["workstation", "max-wn001", "max-wgs-evil", "max-fs-displayevil"])
def test_submission_host_rejects_nonlogin_hosts(host):
    from utils.ML_Fitting_1D_GISAXS.PosteriorV8.launch_k1_balanced_dataset_dag_v5 import (
        _require_submission_host,
    )
    with pytest.raises(RuntimeError, match="Maxwell login host"):
        _require_submission_host(host)


@pytest.mark.parametrize("host", ["max-wgs001", "max-fs-display006.desy.de"])
@pytest.mark.parametrize("stage,role", [
    ("balanced_dataset", "worker"), ("balanced_dataset", "collector"),
    ("phase_c_holdout", "worker"), ("phase_c_holdout", "collector"),
    ("iid_calibration", "worker"), ("iid_calibration", "collector"),
])
def test_data_computation_rejects_login_hosts_before_reading_inputs(tmp_path, host, stage, role):
    module = import_module(
        "utils.ML_Fitting_1D_GISAXS.PosteriorV8.k1_" + stage + "_" + role + "_v5"
    )
    environment = {"SLURM_JOB_ID": "123", "SLURM_ARRAY_TASK_ID": "0"}
    with pytest.raises(RuntimeError, match="forbidden"):
        if stage == "balanced_dataset" and role == "worker":
            module._worker_guard(dry_run=False, hostname=host, environment=environment)
        elif stage == "balanced_dataset":
            module._worker_guard(host, environment)
        else:
            name = f"run_v5_k1_{stage}_task" if role == "worker" else f"collect_v5_k1_{stage}"
            args = (tmp_path / "absent.json", 0) if role == "worker" else (tmp_path / "absent.json",)
            getattr(module, name)(*args, expected_plan_sha256="0" * 64,
                                  hostname=host, environment=environment)


def test_root_check_accepts_pure_path_without_weakening_containment(tmp_path):
    root = tmp_path.resolve() / "allowed"
    root.mkdir()
    child = root / "input.json"
    child.write_text("{}", encoding="utf-8")
    allowed = PurePosixPath(root)
    assert launcher._under_root(child, allowed, "input") == child
    with pytest.raises(ValueError, match="must remain below"):
        launcher._under_root(root, allowed, "input")
    outside = tmp_path.resolve() / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="must remain below"):
        launcher._under_root(outside, allowed, "input")
    link = root / "linked.json"
    link.symlink_to(child)
    with pytest.raises(ValueError, match="symlink"):
        launcher._under_root(link, allowed, "input")


def _input_identity() -> dict[str, object]:
    return {
        "source_root": "/data/dust/user/zhaiyufe/source-snapshots/source-v1",
        "source_bundle_sha256": _sha("source-bundle"),
        "source_archive_path": "/data/dust/user/zhaiyufe/source-archives/source-v1.tar",
        "source_archive_sha256": _sha("source-archive"),
        "local_sobol_schedule_path": "/data/dust/user/zhaiyufe/schedules/local.json",
        "local_sobol_schedule_file_sha256": _sha("schedule-file"),
        "local_sobol_schedule_sha256": _sha("schedule"),
        "local_sobol_schedule_artifact_sha256": _sha("schedule-artifact"),
        "calibration_path": "/data/dust/user/zhaiyufe/calibration/calibration.json",
        "calibration_file_sha256": _sha("calibration-file"),
        "calibration_artifact_sha256": _sha("calibration-artifact"),
    }


class _HeldScheduler:
    def __init__(self, *, wrong_held_reason: bool = False, dependency_override=None) -> None:
        self.held = {"771", "772"}
        self.dependencies = {"771": None, "772": "771"}
        self.wrong_held_reason = wrong_held_reason
        self.dependency_override = dependency_override
        self.calls: list[tuple[str, ...]] = []
        self.releases: list[str] = []
        self.cancelled: list[str] = []

    def __call__(self, argv: Sequence[str]) -> launcher.V5CommandResult:
        command = tuple(argv)
        self.calls.append(command)
        if command[0] == "sbatch":
            job_id = "772" if any(
                value.startswith("--dependency=afterok:") for value in command
            ) else "771"
            return launcher.V5CommandResult(0, job_id + "\n", "")
        if command[:4] == ("scontrol", "show", "job", "--oneliner"):
            job_id = command[4]
            dependency = self.dependencies[job_id]
            if job_id in self.held:
                reason = (
                    "Priority"
                    if self.wrong_held_reason and job_id == "771"
                    else "JobHeldUser"
                )
            else:
                reason = "Dependency" if dependency else "Priority"
            dependency_text = (
                "(null)"
                if dependency is None
                else f"afterok:{dependency}(unfulfilled)"
            )
            if job_id == "772" and self.dependency_override is not None:
                dependency_text = self.dependency_override
            return launcher.V5CommandResult(
                0,
                f"JobId={job_id} JobState=PENDING Reason={reason} "
                f"Dependency={dependency_text}\n",
                "",
            )
        if command[:2] == ("scontrol", "release"):
            job_id = command[2]
            self.releases.append(job_id)
            self.held.remove(job_id)
            return launcher.V5CommandResult(0, "", "")
        if command[0] == "scancel":
            self.cancelled.append(command[1])
            return launcher.V5CommandResult(0, "", "")
        raise AssertionError(f"unexpected command: {command}")


def _stub_inputs(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    identity = _input_identity()
    monkeypatch.setattr(
        launcher,
        "replay_v5_k1_balanced_full_search_launch_inputs",
        lambda *args, **kwargs: dict(identity),
    )
    return identity


def test_full_search_launcher_dry_run_is_write_free_and_has_exact_afterok(
    monkeypatch: pytest.MonkeyPatch,
):
    identity = _stub_inputs(monkeypatch)

    result = launcher.launch_v5_k1_balanced_full_search_dag(
        _plan(),
        source_root=identity["source_root"],
        source_archive=identity["source_archive_path"],
        local_sobol_schedule_path=identity["local_sobol_schedule_path"],
        calibration_path=identity["calibration_path"],
    )

    assert result["status"] == "dry_run"
    assert result["writes_performed"] is False
    array = result["submission_preview"]["array"]
    collector = result["submission_preview"]["collector"]
    assert "--hold" in array and "--array=0-59" in array
    assert "--hold" in collector
    assert "--dependency=afterok:ARRAY_JOB_ID" in collector


def _stub_submit_files(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[Path, dict[str, object], str]]:
    writes: list[tuple[Path, dict[str, object], str]] = []
    monkeypatch.setattr(launcher, "_prepare_layout", lambda plan: None)
    monkeypatch.setattr(
        launcher,
        "write_v5_k1_balanced_full_search_plan",
        lambda path, plan: Path(path),
    )
    monkeypatch.setattr(
        launcher,
        "_file_identity",
        lambda path, name: {
            "path": str(path),
            "file_sha256": _sha("plan-file"),
            "byte_count": 1234,
            "mode_octal": "0400",
            "nlink": 1,
        },
    )
    monkeypatch.setattr(launcher, "file_sha256", lambda path, name: _sha(name))

    def fake_write(path, core, hash_field):
        payload = {
            **core,
            hash_field: sha256(canonical_json(core).encode("utf-8")).hexdigest(),
        }
        writes.append((Path(path), payload, hash_field))
        return payload

    monkeypatch.setattr(launcher, "_write_evidence", fake_write)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    return writes


@pytest.mark.parametrize("dependency", ["afterok:771", "afterok:771_*(unfulfilled)"])
@pytest.mark.parametrize("hostname", ["max-wgs", "max-fs-display006.desy.de"])
def test_full_search_launcher_holds_reads_back_and_releases_in_reverse_order(
    monkeypatch: pytest.MonkeyPatch,
    hostname: str,
    dependency: str,
):
    identity = _stub_inputs(monkeypatch)
    writes = _stub_submit_files(monkeypatch)
    scheduler = _HeldScheduler(dependency_override=dependency)

    result = launcher.launch_v5_k1_balanced_full_search_dag(
        _plan(),
        source_root=identity["source_root"],
        source_archive=identity["source_archive_path"],
        local_sobol_schedule_path=identity["local_sobol_schedule_path"],
        calibration_path=identity["calibration_path"],
        submit=True,
        runner=scheduler,
        hostname=hostname,
    )

    assert result["status"] == "ALL_JOBS_RELEASED"
    assert result["job_ids"] == {"array": "771", "collector": "772"}
    assert scheduler.releases == ["772", "771"]
    assert scheduler.cancelled == []
    assert [value[2] for value in writes] == [
        "receipt_sha256",
        "completion_sha256",
    ]
    receipt = writes[0][1]
    assert receipt["status"] == "ALL_JOBS_HELD"
    assert receipt["release_order"] == ["collector", "array"]
    assert receipt["plan_identity"]["mode_octal"] == "0400"
    assert receipt["plan_identity"]["nlink"] == 1


@pytest.mark.parametrize("dependency", [
    "afterany:771", "afterok:7710", "afterok:771_0", "afterok:771:999",
    "afterok:771?afterok:999", "afterok:771,afterok:999",
    "(null) SubmitLine=sbatch --dependency=afterok:771",
    "afterok:771 Dependency=afterok:771", "afterok:771(fulfilled)",
])
def test_wrong_dependency_keeps_both_jobs_held(monkeypatch, dependency):
    identity = _stub_inputs(monkeypatch)
    writes = _stub_submit_files(monkeypatch)
    scheduler = _HeldScheduler(dependency_override=dependency)
    with pytest.raises(RuntimeError, match="whole-array afterok"):
        launcher.launch_v5_k1_balanced_full_search_dag(
            _plan(), source_root=identity["source_root"],
            source_archive=identity["source_archive_path"],
            local_sobol_schedule_path=identity["local_sobol_schedule_path"],
            calibration_path=identity["calibration_path"], submit=True,
            runner=scheduler, hostname="max-wgs",
        )
    assert scheduler.held == {"771", "772"}
    assert scheduler.releases == []
    assert scheduler.cancelled == []
    assert writes[-1][2] == "failure_sha256"


@pytest.mark.parametrize("hostname", ["workstation", "max-fs-displayevil"])
def test_full_search_launcher_rejects_a_non_maxwell_submission_host(
    monkeypatch: pytest.MonkeyPatch,
    hostname: str,
):
    identity = _stub_inputs(monkeypatch)

    with pytest.raises(RuntimeError, match="requires a Maxwell submission host"):
        launcher.launch_v5_k1_balanced_full_search_dag(
            _plan(),
            source_root=identity["source_root"],
            source_archive=identity["source_archive_path"],
            local_sobol_schedule_path=identity["local_sobol_schedule_path"],
            calibration_path=identity["calibration_path"],
            submit=True,
            hostname=hostname,
        )


def test_full_search_launcher_preserves_submitted_jobs_after_bad_held_readback(
    monkeypatch: pytest.MonkeyPatch,
):
    identity = _stub_inputs(monkeypatch)
    writes = _stub_submit_files(monkeypatch)
    scheduler = _HeldScheduler(wrong_held_reason=True)

    with pytest.raises(RuntimeError, match="not user-held"):
        launcher.launch_v5_k1_balanced_full_search_dag(
            _plan(),
            source_root=identity["source_root"],
            source_archive=identity["source_archive_path"],
            local_sobol_schedule_path=identity["local_sobol_schedule_path"],
            calibration_path=identity["calibration_path"],
            submit=True,
            runner=scheduler,
            hostname="max-wgs",
        )

    assert scheduler.releases == []
    assert scheduler.cancelled == []
    assert writes[-1][2] == "failure_sha256"
    assert writes[-1][1]["submitted_jobs_are_not_cancelled_automatically"] is True
