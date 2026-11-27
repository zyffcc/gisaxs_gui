"""GIMaP's own processes start OpenBLAS with few threads (``shared/blas_threads``): the program's process while NumPy
and SciPy load, and every job worker before NumPy loads. The environment of the program stays as it was, so the
programs that GIMaP starts (Sputter's analysis) keep their own settings."""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from src.gimap.shared import blas_threads

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _clean_environment() -> dict:
    env = dict(os.environ)
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "GOTO_NUM_THREADS", "MKL_NUM_THREADS"):
        env.pop(name, None)
    return env


def test_capped_restores_the_environment(monkeypatch) -> None:
    monkeypatch.delenv(blas_threads.VARIABLE, raising=False)
    with blas_threads.capped():
        assert os.environ[blas_threads.VARIABLE] == str(blas_threads.threads())
    assert blas_threads.VARIABLE not in os.environ


def test_a_value_set_by_the_user_is_kept(monkeypatch) -> None:
    monkeypatch.setenv(blas_threads.VARIABLE, "2")
    with blas_threads.capped():
        assert os.environ[blas_threads.VARIABLE] == "2"
    assert os.environ[blas_threads.VARIABLE] == "2"
    blas_threads.cap_environment()
    assert os.environ[blas_threads.VARIABLE] == "2"


def test_threads_never_exceed_the_processors() -> None:
    assert 1 <= blas_threads.threads() <= max(1, min(blas_threads.THREADS, os.cpu_count() or 1))


def test_the_program_loads_numpy_and_scipy_with_the_cap() -> None:
    pytest.importorskip("threadpoolctl")
    code = textwrap.dedent(f"""
        import json, os, sys
        sys.path.insert(0, {str(PROJECT_ROOT)!r})
        from src.gimap.shared.blas_threads import VARIABLE, load_numerics
        load_numerics()
        from threadpoolctl import threadpool_info
        blas = [pool["num_threads"] for pool in threadpool_info() if pool["user_api"] == "blas"]
        print(json.dumps({{"blas": blas, "left": os.environ.get(VARIABLE)}}))
    """)
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120,
                          env=_clean_environment())
    assert done.returncode == 0, done.stderr
    outcome = json.loads(done.stdout.strip().splitlines()[-1])
    assert len(outcome["blas"]) >= 2  # NumPy's and SciPy's OpenBLAS
    assert max(outcome["blas"]) <= blas_threads.threads()
    assert outcome["left"] is None


PROBE = """
import os


def blas(payload, report, is_cancelled):
    from threadpoolctl import threadpool_info

    import scipy.linalg  # noqa: F401

    return {"variable": os.environ.get("OPENBLAS_NUM_THREADS"),
            "threads": [pool["num_threads"] for pool in threadpool_info() if pool["user_api"] == "blas"]}
"""

GUI_LIKE = """
import json
import os
import sys

sys.path.insert(1, {root!r})

from src.gimap.app.jobs import JobRequest
from src.gimap.integrations.jobs import LocalProcessJobRunner

if __name__ == "__main__":
    result = LocalProcessJobRunner().run(JobRequest(handler="gimap_blas_probe:blas", timeout_seconds=120))
    error = None if result.error is None else result.error.message
    print(json.dumps({{"value": result.value, "error": error, "parent": os.environ.get("OPENBLAS_NUM_THREADS")}}))
"""


def test_a_job_worker_starts_openblas_with_the_cap(tmp_path: Path) -> None:
    pytest.importorskip("threadpoolctl")
    (tmp_path / "gimap_blas_probe.py").write_text(PROBE, encoding="utf-8")
    script = tmp_path / "gui_like_entry.py"
    script.write_text(GUI_LIKE.format(root=str(PROJECT_ROOT)), encoding="utf-8")
    done = subprocess.run([sys.executable, str(script)], cwd=tmp_path, capture_output=True, text=True, timeout=240,
                          env=_clean_environment())
    assert done.returncode == 0, done.stderr
    outcome = json.loads(done.stdout.strip().splitlines()[-1])
    assert outcome["error"] is None
    assert outcome["value"]["variable"] == str(blas_threads.threads())
    assert max(outcome["value"]["threads"]) <= blas_threads.threads()
    assert outcome["parent"] is None
