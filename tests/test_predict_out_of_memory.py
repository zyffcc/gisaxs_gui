"""A fit that runs out of memory says so in a plain sentence; a failed Predict job is in the log with its traceback."""

import logging
import threading
from types import SimpleNamespace

import pytest

from src.gimap.app.jobs.models import JobError, JobResult
from src.gimap.app.presentation.i18n.zh import ZH
from src.gimap.features.fitting.presentation.single.method_hint import OUT_OF_MEMORY, fit_error_text
from src.gimap.features.fitting.presentation.single.predict import PredictPageMixin

NUMPY = "Unable to allocate 7.19 MiB for an array with shape (1, 1024, 920) and data type float64"


@pytest.mark.parametrize("message", [NUMPY, "MemoryError", "_ArrayMemoryError"])
def test_out_of_memory_is_said_plainly(message):
    assert fit_error_text(message) == OUT_OF_MEMORY
    assert ZH[OUT_OF_MEMORY]


def test_other_errors_keep_their_text():
    assert fit_error_text("Fitting cancelled") == "Fitting cancelled"


def _page(result):
    jobs = SimpleNamespace(run=lambda request, progress: result)
    return SimpleNamespace(_stop_event=threading.Event(), _predict_progress=None,
                           view_model=SimpleNamespace(context=SimpleNamespace(jobs=jobs)))


def test_failed_predict_job_is_logged_with_its_traceback(caplog):
    error = JobError(code="handler_error", message=NUMPY, exception_type="_ArrayMemoryError",
                     traceback="Traceback (most recent call last):\n  File \"refine.py\", line 1\n")
    page = _page(JobResult(job_id="j", status="failed", error=error))
    with caplog.at_level(logging.WARNING), pytest.raises(RuntimeError, match="Unable to allocate"):
        PredictPageMixin._run_predict_job(page, SimpleNamespace())
    assert "_ArrayMemoryError" in caplog.text and "refine.py" in caplog.text


def test_stopped_predict_job_is_not_logged(caplog):
    error = JobError(code="handler_error", message="Fitting cancelled", exception_type="RuntimeError")
    page = _page(JobResult(job_id="j", status="failed", error=error))
    with caplog.at_level(logging.WARNING), pytest.raises(RuntimeError, match="Fitting cancelled"):
        PredictPageMixin._run_predict_job(page, SimpleNamespace())
    assert "Predict parameters failed" not in caplog.text
