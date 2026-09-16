from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import pytest

from PosteriorV8 import k1_balanced_full_search_collector_v5 as collector
from PosteriorV8.grouped_artifact_v5 import canonical_json
from PosteriorV8.k1_balanced_full_search_worker_v5 import (
    V5_K1_BALANCED_FULL_SEARCH_TASK_COMPLETION_FILENAME,
    V5_K1_BALANCED_FULL_SEARCH_TASK_COMPLETION_SCHEMA,
    V5_K1_BALANCED_FULL_SEARCH_TASK_COMPLETION_VERSION,
)


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("fixture", encoding="utf-8")
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
    plan = {
        "plan_sha256": "a" * 64,
        "parents": [{"array_task_id": index} for index in range(60)],
        "layout": {"completion": str(tmp_path / "run" / "audit" / "completion.json")},
    }
    return paths, plan


def test_collector_guard_requires_a_non_array_slurm_worker():
    with pytest.raises(RuntimeError, match="forbidden on max-wgs"):
        collector._worker_guard(
            dry_run=False,
            hostname="max-wgs.desy.de",
            environment={"SLURM_JOB_ID": "1"},
        )
    with pytest.raises(RuntimeError, match="requires a Slurm worker"):
        collector._worker_guard(
            dry_run=False,
            hostname="max-wn001",
            environment={},
        )
    with pytest.raises(RuntimeError, match="cannot be an array task"):
        collector._worker_guard(
            dry_run=False,
            hostname="max-wn001",
            environment={"SLURM_JOB_ID": "1", "SLURM_ARRAY_TASK_ID": "0"},
        )


def test_collector_dry_run_is_write_free_on_login_node(tmp_path, monkeypatch):
    paths, plan = _dry_run_fixture(tmp_path)
    monkeypatch.setattr(collector, "MAXWELL_DUST_ROOT", tmp_path)
    monkeypatch.setattr(
        collector,
        "read_v5_k1_balanced_full_search_plan_file",
        lambda *args, **kwargs: plan,
    )

    result = collector.collect_v5_k1_balanced_full_search(
        paths["plan"],
        expected_plan_sha256="a" * 64,
        expected_plan_file_sha256="b" * 64,
        source_root=paths["source_root"],
        source_archive=paths["source_archive"],
        local_sobol_schedule_path=paths["schedule"],
        calibration_path=paths["calibration"],
        dry_run=True,
        hostname="max-wgs.desy.de",
        environment={},
    )

    assert result["status"] == "checked_dry_run"
    assert result["writes_performed"] is False
    assert result["expected_task_count"] == 60
    assert result["full_search_supervision_complete"] is False
    assert not Path(plan["layout"]["completion"]).parent.exists()


def test_collector_failure_is_preserved_without_a_completion(tmp_path, monkeypatch):
    paths, plan = _dry_run_fixture(tmp_path)
    failure_path = tmp_path / "run" / "audit" / "failure.json"
    plan["layout"]["failure"] = str(failure_path)
    failure_path.parent.mkdir(parents=True)
    monkeypatch.setattr(collector, "MAXWELL_DUST_ROOT", tmp_path)
    monkeypatch.setattr(
        collector,
        "read_v5_k1_balanced_full_search_plan_file",
        lambda *args, **kwargs: plan,
    )
    monkeypatch.setattr(
        collector,
        "_collect_actual",
        lambda **kwargs: (_ for _ in ()).throw(ValueError("fixture drift")),
    )

    with pytest.raises(ValueError, match="fixture drift"):
        collector.collect_v5_k1_balanced_full_search(
            paths["plan"],
            expected_plan_sha256="a" * 64,
            expected_plan_file_sha256="b" * 64,
            source_root=paths["source_root"],
            source_archive=paths["source_archive"],
            local_sobol_schedule_path=paths["schedule"],
            calibration_path=paths["calibration"],
            hostname="max-wn001",
            environment={"SLURM_JOB_ID": "123"},
        )

    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    supplied = failure.pop("failure_sha256")
    assert supplied == sha256(canonical_json(failure).encode()).hexdigest()
    assert failure["status"] == "FAIL"
    assert failure["message"] == "fixture drift"
    assert not Path(plan["layout"]["completion"]).exists()


def test_aggregate_requires_the_frozen_48_plus_12_population():
    rows = []
    for index in range(60):
        role = "train" if index < 48 else "tuning_validation"
        rows.append(
            {
                "role": role,
                "recipe_count": 288,
                "query_count": 288,
                "branch_count": 576,
                "exact_forward_calls_used": 576 * 4096,
            }
        )
    totals = collector._aggregate(rows)
    assert totals["train"]["clean_parent_count"] == 13824
    assert totals["tuning_validation"]["clean_parent_count"] == 3456
    with pytest.raises(ValueError, match="task totals drifted"):
        collector._aggregate(rows[:-1])


def test_task_completion_requires_sealed_tree_and_exact_claims(tmp_path):
    output = tmp_path / "task-output"
    output.mkdir()
    selected = {
        "array_task_id": 7,
        "search_output_root": str(output),
    }
    core = {
        "schema": V5_K1_BALANCED_FULL_SEARCH_TASK_COMPLETION_SCHEMA,
        "version": V5_K1_BALANCED_FULL_SEARCH_TASK_COMPLETION_VERSION,
        "status": "PASS",
        "scientific_acceptance_evidence": False,
        "full_search_supervision_complete": False,
        "gradient_training_authorized": False,
        "plan_sha256": "a" * 64,
        "array_task_id": 7,
        "slurm_job_id": "123",
        "hostname": "max-wn001",
        "task_authorization": {},
        "pipeline_completion": {},
        "counts": {},
        "immutable_input_identity_pre": {"source": "b" * 64},
        "immutable_input_identity_post": {"source": "b" * 64},
        "pipeline_tree_files_sealed_before_task_completion": True,
        "task_completion_written_last": True,
    }
    payload = {
        **core,
        "completion_sha256": sha256(canonical_json(core).encode()).hexdigest(),
    }
    path = output / V5_K1_BALANCED_FULL_SEARCH_TASK_COMPLETION_FILENAME
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    path.chmod(0o400)
    output.chmod(0o500)

    checked, checked_path, _ = collector._task_completion(
        {"plan_sha256": "a" * 64}, selected
    )
    assert checked == payload
    assert checked_path == path

    output.chmod(0o700)
    path.chmod(0o600)
    with pytest.raises(ValueError, match="0500"):
        collector._task_completion({"plan_sha256": "a" * 64}, selected)
