from pathlib import Path
import os

import pytest

from PosteriorV8 import k1_balanced_full_search_replay_v5 as replay
from PosteriorV8 import k1_balanced_full_search_collector_v5 as collector


def _fixture(tmp_path, monkeypatch):
    paths = {name: tmp_path / name for name in
             ("plan", "source_root", "source_archive", "schedule", "calibration")}
    for path in paths.values():
        path.write_text("fixture")
    paths["source_root"].unlink()
    paths["source_root"].mkdir()
    audit = tmp_path / "audit"
    audit.mkdir()
    completion_path = audit / "completion.json"
    inventory_path = audit / collector.V5_K1_BALANCED_FULL_SEARCH_INVENTORY_FILENAME
    rows = tuple({"array_task_id": i, "role": "train" if i < 48 else "tuning_validation",
                  "recipe_count": 288, "query_count": 288, "branch_count": 576,
                  "exact_forward_calls_used": 576 * 4096} for i in range(60))
    plan = {"plan_sha256": "a" * 64, "parents": list(rows),
            "source": {"bundle_sha256": "b" * 64},
            "identity_authorization_sha256": "c" * 64,
            "layout": {"completion": str(completion_path)}}
    monkeypatch.setattr(collector, "MAXWELL_DUST_ROOT", tmp_path)
    monkeypatch.setattr(collector, "read_v5_k1_balanced_full_search_plan_file", lambda *a, **k: plan)
    monkeypatch.setattr(collector, "_source_input_identity", lambda **k: ({"file": "sealed-input"},))
    monkeypatch.setattr(collector, "_checked_task", lambda plan, selected: dict(selected))
    monkeypatch.setattr(replay.socket, "gethostname", lambda: "max-wn001")
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.delenv("SLURM_ARRAY_TASK_ID", raising=False)
    collector._collect_actual(plan=plan, paths=paths, inventory_path=inventory_path,
                              completion_path=completion_path, host="max-wn002",
                              environment={"SLURM_JOB_ID": "122"})
    kwargs = dict(plan_path=paths["plan"], expected_plan_sha256="a" * 64,
                  expected_plan_file_sha256="d" * 64,
                  expected_inventory_file_sha256=collector.file_sha256(inventory_path),
                  expected_completion_file_sha256=collector.file_sha256(completion_path),
                  source_root=paths["source_root"], source_archive=paths["source_archive"],
                  local_sobol_schedule_path=paths["schedule"], calibration_path=paths["calibration"])
    return kwargs, inventory_path, completion_path


def test_collection_reader_reuses_live_task_replay_and_never_grants_gradients(tmp_path, monkeypatch):
    kwargs, _, _ = _fixture(tmp_path, monkeypatch)
    before = {str(p): p.stat().st_mtime_ns for p in tmp_path.rglob("*")}
    result = replay.replay_v5_k1_balanced_full_search_collection(**kwargs)
    assert result["inventory"]["task_count"] == 60
    assert result["writes_performed"] is result["gradient_training_authorized"] is False
    assert before == {str(p): p.stat().st_mtime_ns for p in tmp_path.rglob("*")}


@pytest.mark.parametrize("kind", ["inventory_hash", "completion_hash", "source", "task", "writable"])
def test_collection_reader_rejects_drift(tmp_path, monkeypatch, kind):
    kwargs, inventory_path, _ = _fixture(tmp_path, monkeypatch)
    if kind == "inventory_hash":
        kwargs["expected_inventory_file_sha256"] = "e" * 64
    elif kind == "completion_hash":
        kwargs["expected_completion_file_sha256"] = "e" * 64
    elif kind == "source":
        monkeypatch.setattr(collector, "_source_input_identity", lambda **k: ({"file": "changed"},))
    elif kind == "task":
        monkeypatch.setattr(collector, "_checked_task", lambda plan, selected:
                            {**selected, "exact_forward_calls_used": 0})
    else:
        inventory_path.chmod(0o600)
    with pytest.raises((ValueError, RuntimeError)):
        replay.replay_v5_k1_balanced_full_search_collection(**kwargs)


@pytest.mark.parametrize("host", ["max-wgs01", "max-fs-display006"])
def test_collection_reader_refuses_login_before_opening_inputs(monkeypatch, host):
    monkeypatch.setattr(replay.socket, "gethostname", lambda: host)
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    with pytest.raises(RuntimeError, match="forbidden"):
        replay.replay_v5_k1_balanced_full_search_collection(
            Path("missing"), expected_plan_sha256="a" * 64, expected_plan_file_sha256="b" * 64,
            expected_inventory_file_sha256="c" * 64, expected_completion_file_sha256="d" * 64,
            source_root="missing", source_archive="missing", local_sobol_schedule_path="missing",
            calibration_path="missing",
        )


def test_collection_reader_rejects_drift_on_second_task_replay(tmp_path, monkeypatch):
    kwargs, _, _ = _fixture(tmp_path, monkeypatch)
    calls = []

    def changed(plan, selected):
        calls.append(selected["array_task_id"])
        return {**selected, "recipe_count": 0} if len(calls) > 60 else dict(selected)

    monkeypatch.setattr(collector, "_checked_task", changed)
    with pytest.raises(RuntimeError, match="changed during read-only replay"):
        replay.replay_v5_k1_balanced_full_search_collection(**kwargs)
    assert len(calls) == 120


def test_collection_reader_rejects_completion_older_than_inventory(tmp_path, monkeypatch):
    kwargs, inventory_path, completion_path = _fixture(tmp_path, monkeypatch)
    stamp = inventory_path.stat().st_mtime_ns - 1_000_000
    os.utime(completion_path, ns=(stamp, stamp))
    with pytest.raises(ValueError, match="predates"):
        replay.replay_v5_k1_balanced_full_search_collection(**kwargs)


def test_search_array_python_entry_also_refuses_fs_login():
    from PosteriorV8 import k1_balanced_full_search_worker_v5 as worker

    with pytest.raises(RuntimeError, match="max-fs-display"):
        worker._worker_guard(dry_run=False, hostname="max-fs-display006.desy.de",
                             environment={"SLURM_JOB_ID": "123", "SLURM_ARRAY_TASK_ID": "0"})


@pytest.mark.parametrize("replace", [False, True])
def test_collection_binds_historical_plan_across_both_replays(tmp_path, monkeypatch, replace):
    kwargs, _, _ = _fixture(tmp_path, monkeypatch)
    path = tmp_path / "producer-plan.json"
    path.write_text("{}\n")
    path.chmod(0o400)
    kwargs["balanced_dataset_launch_plan_path"] = path
    calls = []
    def source(**arguments):
        calls.append(arguments["balanced_dataset_launch_plan_path"])
        if replace and len(calls) == 2:
            other = path.with_suffix(".replacement")
            other.write_bytes(path.read_bytes())
            other.chmod(0o400)
            other.replace(path)
        return ({"file": "sealed-input"},)
    monkeypatch.setattr(replay, "_source_input_identity", source)
    if replace:
        with pytest.raises(RuntimeError, match="changed across collection"):
            replay.replay_v5_k1_balanced_full_search_collection(**kwargs)
    else:
        result = replay.replay_v5_k1_balanced_full_search_collection(**kwargs)
        assert result["producer_launch_plan_file_identity"] == replay.read_only_identity(path, "producer")
    assert calls == [path, path]
