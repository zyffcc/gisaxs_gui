from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import stat

import pytest

from PosteriorV8 import k1_balanced_full_search_worker_v5 as worker
from PosteriorV8.grouped_artifact_v5 import canonical_json


def _touch(path: Path, content: str = "fixture") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _dry_run_fixture(tmp_path: Path):
    paths = {
        "plan": _touch(tmp_path / "launch-plan.json"),
        "source_root": tmp_path / "source",
        "source_archive": _touch(tmp_path / "source.zip"),
        "schedule": _touch(tmp_path / "schedule.gvd5"),
        "calibration": _touch(tmp_path / "calibration.json"),
    }
    paths["source_root"].mkdir()
    output = tmp_path / "run" / "shards" / "train" / "branch-00-shard-000000"
    selected = {
        "array_task_id": 0,
        "parent": {"artifact_sha256": "b" * 64},
        "search_output_root": str(output),
    }
    plan = {"plan_sha256": "a" * 64, "parents": [selected]}
    return paths, output, plan


def test_worker_guard_rejects_login_node_and_non_slurm_execution():
    with pytest.raises(RuntimeError, match="forbidden on max-wgs"):
        worker._worker_guard(
            dry_run=False,
            hostname="max-wgs.desy.de",
            environment={"SLURM_JOB_ID": "1", "SLURM_ARRAY_TASK_ID": "0"},
        )
    with pytest.raises(RuntimeError, match="requires a Slurm array worker"):
        worker._worker_guard(
            dry_run=False,
            hostname="max-wn001",
            environment={},
        )
    worker._worker_guard(
        dry_run=True,
        hostname="max-wgs.desy.de",
        environment={},
    )


def test_dry_run_checks_plan_identity_without_creating_output(
    tmp_path, monkeypatch
):
    paths, output, plan = _dry_run_fixture(tmp_path)
    monkeypatch.setattr(worker, "MAXWELL_DUST_ROOT", tmp_path)
    monkeypatch.setattr(
        worker,
        "read_v5_k1_balanced_full_search_plan_file",
        lambda *args, **kwargs: plan,
    )
    monkeypatch.setattr(
        worker,
        "read_v5_k1_balanced_full_search_task_input_identity",
        lambda **kwargs: {"verified_inputs_sha256": "c" * 64},
    )

    result = worker.run_v5_k1_balanced_full_search_task(
        paths["plan"],
        0,
        expected_plan_sha256="a" * 64,
        expected_plan_file_sha256="d" * 64,
        source_root=paths["source_root"],
        source_archive=paths["source_archive"],
        local_sobol_schedule_path=paths["schedule"],
        calibration_path=paths["calibration"],
        dry_run=True,
        hostname="max-wgs.desy.de",
        environment={},
    )

    assert result == {
        "status": "checked_dry_run",
        "writes_performed": False,
        "plan_sha256": "a" * 64,
        "array_task_id": 0,
        "source_parent_artifact_sha256": "b" * 64,
        "input_identity": {"verified_inputs_sha256": "c" * 64},
        "search_evidence_created": False,
        "gradient_training_authorized": False,
    }
    assert not output.exists()


def test_dry_run_refuses_to_reuse_existing_task_output(tmp_path, monkeypatch):
    paths, output, plan = _dry_run_fixture(tmp_path)
    output.mkdir(parents=True)
    monkeypatch.setattr(worker, "MAXWELL_DUST_ROOT", tmp_path)
    monkeypatch.setattr(
        worker,
        "read_v5_k1_balanced_full_search_plan_file",
        lambda *args, **kwargs: plan,
    )

    with pytest.raises(FileExistsError, match="refusing to reuse"):
        worker.run_v5_k1_balanced_full_search_task(
            paths["plan"],
            0,
            expected_plan_sha256="a" * 64,
            expected_plan_file_sha256="d" * 64,
            source_root=paths["source_root"],
            source_archive=paths["source_archive"],
            local_sobol_schedule_path=paths["schedule"],
            calibration_path=paths["calibration"],
            dry_run=True,
            hostname="max-wgs.desy.de",
            environment={},
        )


def test_completion_self_hash_and_tree_sealing_are_fail_closed(tmp_path):
    core = {"schema": "fixture/v1", "status": "PASS"}
    completion_sha = sha256(canonical_json(core).encode()).hexdigest()
    completion = tmp_path / "upstream-completion.json"
    completion.write_text(
        json.dumps(
            {**core, "completion_sha256": completion_sha},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    assert worker._completion_self_sha256(completion, completion_sha) == completion_sha
    with pytest.raises(ValueError, match="identity drifted"):
        worker._completion_self_sha256(completion, "f" * 64)

    output = tmp_path / "output"
    nested = output / "nested"
    nested.mkdir(parents=True)
    artifact = _touch(nested / "artifact.json")
    worker._seal_pipeline_tree(output)
    assert stat.S_IMODE(artifact.stat().st_mode) == 0o400
    assert stat.S_IMODE(nested.stat().st_mode) == 0o500
