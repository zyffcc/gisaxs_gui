"""KeptProcessJobRunner: one worker process lasts from job to job (2D Prediction keeps its model loaded),
and stays as isolated as a worker per job: a crash, a cancel or a timeout ends it and the next job starts
a fresh one."""

import importlib
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src.gimap.app.jobs import JobRequest
from src.gimap.integrations.jobs import KeptProcessJobRunner

HANDLERS = """
import os
import time

LOADED = []


def load(payload, report, is_cancelled):
    report(0, 1, "loading")
    if not LOADED:
        time.sleep(float(payload.get("seconds", 0.0)))
        LOADED.append(os.getpid())
    return {"pid": os.getpid(), "loads": len(LOADED)}


def crash(_payload, _report, _is_cancelled):
    os._exit(17)


def sleep(payload, _report, _is_cancelled):
    time.sleep(float(payload.get("seconds", 1.0)))
    return {"slept": True}
"""


def _handlers(tmp_path: Path, monkeypatch) -> str:
    name = "gimap_kept_job_handlers"
    (tmp_path / f"{name}.py").write_text(HANDLERS, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    return name


def test_the_worker_and_what_it_loaded_last_from_job_to_job(tmp_path, monkeypatch):
    module = _handlers(tmp_path, monkeypatch)
    runner = KeptProcessJobRunner()
    try:
        first = runner.run(JobRequest(handler=f"{module}:load", payload={"seconds": 0.3}, timeout_seconds=60))
        second = runner.run(JobRequest(handler=f"{module}:load", payload={"seconds": 0.3}, timeout_seconds=60))
        assert first.succeeded and second.succeeded
        assert second.value["pid"] == first.value["pid"]
        assert second.value["loads"] == 1  # loaded once, in the first job
        assert runner.alive
    finally:
        runner.shutdown()
    assert not runner.alive


def test_a_crash_ends_the_worker_and_the_next_job_starts_a_fresh_one(tmp_path, monkeypatch):
    module = _handlers(tmp_path, monkeypatch)
    runner = KeptProcessJobRunner()
    try:
        before = runner.run(JobRequest(handler=f"{module}:load", timeout_seconds=60)).value["pid"]
        crashed = runner.run(JobRequest(handler=f"{module}:crash", timeout_seconds=60))
        assert crashed.status == "failed" and crashed.error.code == "worker_crash" and "17" in crashed.error.message
        after = runner.run(JobRequest(handler=f"{module}:load", timeout_seconds=60))
        assert after.succeeded and after.value["pid"] != before
    finally:
        runner.shutdown()


def test_a_timeout_or_a_cancel_ends_the_worker(tmp_path, monkeypatch):
    module = _handlers(tmp_path, monkeypatch)
    runner = KeptProcessJobRunner()
    try:
        runner.run(JobRequest(handler=f"{module}:load", timeout_seconds=60))
        late = runner.run(JobRequest(handler=f"{module}:sleep", payload={"seconds": 5.0}, timeout_seconds=0.3))
        assert late.status == "timed_out" and late.error.code == "timeout" and late.elapsed_seconds < 3.0
        assert not runner.alive

        request = JobRequest(handler=f"{module}:sleep", payload={"seconds": 30.0}, timeout_seconds=60)
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(runner.run, request)
            deadline = time.monotonic() + 30.0
            while not runner.cancel(request.job_id):
                assert time.monotonic() < deadline, "the job never became cancellable"
                time.sleep(0.01)
            result = future.result(timeout=10.0)
        assert result.status == "cancelled" and result.error.code == "cancelled"
        assert not runner.alive
    finally:
        runner.shutdown()


def test_an_idle_worker_ends_by_itself(tmp_path, monkeypatch):
    module = _handlers(tmp_path, monkeypatch)
    runner = KeptProcessJobRunner(idle_seconds=0.3)
    try:
        assert runner.run(JobRequest(handler=f"{module}:load", timeout_seconds=60)).succeeded
        deadline = time.monotonic() + 10.0
        while runner.alive:
            assert time.monotonic() < deadline, "the idle worker did not end"
            time.sleep(0.05)
        assert runner.run(JobRequest(handler=f"{module}:load", timeout_seconds=60)).succeeded  # starts again
    finally:
        runner.shutdown()
