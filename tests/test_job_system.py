import importlib
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src.gimap.app.jobs import JobRequest
from src.gimap.integrations.jobs import LocalProcessJobRunner


HANDLERS = """
import os
import time


def succeed(payload, report, is_cancelled):
    assert not is_cancelled()
    report(1, 2, "half", {"phase": "test"})
    report(2, 2, "done")
    return {"answer": int(payload["value"]) * 2}


def fail(_payload, _report, _is_cancelled):
    raise ValueError("expected worker failure")


def crash(_payload, _report, _is_cancelled):
    os._exit(17)


def wait_for_cancel(_payload, report, is_cancelled):
    report(0, 1, "started")
    while not is_cancelled():
        time.sleep(0.01)
    return {"noticed": True}


def ignore_cancel(payload, _report, _is_cancelled):
    time.sleep(float(payload.get("seconds", 1.0)))
    return {"finished": True}
"""


def _handler_module(tmp_path: Path, monkeypatch) -> str:
    module_name = "gimap_job_test_handlers"
    (tmp_path / f"{module_name}.py").write_text(HANDLERS, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    return module_name


def test_local_process_job_success_and_progress(tmp_path: Path, monkeypatch) -> None:
    module = _handler_module(tmp_path, monkeypatch)
    runner = LocalProcessJobRunner()
    progress = []
    request = JobRequest(handler=f"{module}:succeed", payload={"value": 21})

    result = runner.run(request, progress.append)

    assert result.succeeded
    assert result.value == {"answer": 42}
    assert [item.message for item in progress] == ["half", "done"]
    assert progress[-1].fraction == 1.0


def test_local_process_job_converts_worker_exception(tmp_path: Path, monkeypatch) -> None:
    module = _handler_module(tmp_path, monkeypatch)
    result = LocalProcessJobRunner().run(
        JobRequest(handler=f"{module}:fail")
    )

    assert result.status == "failed"
    assert result.error is not None
    assert result.error.code == "handler_error"
    assert result.error.exception_type == "ValueError"
    assert "expected worker failure" in result.error.message


def test_local_process_job_contains_native_worker_crash(tmp_path: Path, monkeypatch) -> None:
    module = _handler_module(tmp_path, monkeypatch)
    result = LocalProcessJobRunner().run(JobRequest(handler=f"{module}:crash"))

    assert result.status == "failed"
    assert result.error is not None
    assert result.error.code == "worker_crash"
    assert "17" in result.error.message


def test_local_process_job_can_be_cancelled(tmp_path: Path, monkeypatch) -> None:
    module = _handler_module(tmp_path, monkeypatch)
    runner = LocalProcessJobRunner()
    request = JobRequest(handler=f"{module}:wait_for_cancel")

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(runner.run, request)
        deadline = time.monotonic() + 2.0
        while not runner.cancel(request.job_id):
            if time.monotonic() >= deadline:
                raise AssertionError("Job did not become cancellable in time.")
            time.sleep(0.01)
        result = future.result(timeout=3.0)

    assert result.status == "cancelled"
    assert result.error is not None
    assert result.error.code == "cancelled"


def test_local_process_job_timeout_terminates_worker(tmp_path: Path, monkeypatch) -> None:
    module = _handler_module(tmp_path, monkeypatch)
    request = JobRequest(
        handler=f"{module}:ignore_cancel",
        payload={"seconds": 2.0},
        timeout_seconds=0.15,
    )

    result = LocalProcessJobRunner().run(request)

    assert result.status == "timed_out"
    assert result.error is not None
    assert result.error.code == "timeout"
    assert result.elapsed_seconds < 1.5
