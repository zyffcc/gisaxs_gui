"""Job and batch workers start without the GUI: no PyQt5 in a worker process.

PyQt5 ships its own msvcp140.dll (older than BornAgain's). Once PyQt5 is loaded into a worker, that
DLL serves the whole process and BornAgain 24.1 crashes in ``simulate()`` (exit 0xC0000005). PyQt5
reached a worker two ways: ``src.gimap.app`` imported its Qt runtime eagerly, and a spawned worker
ran the GUI's ``main.py`` (PyQt5 at the top) again as ``__mp_main__``.
"""

from __future__ import annotations

import importlib
import importlib.machinery
import importlib.util
import json
import os
import subprocess
import sys
import textwrap
import types
from pathlib import Path

import pytest

from src.gimap.integrations.jobs.light_main import WORKER_MAIN, light_main

PROJECT_ROOT = Path(__file__).resolve().parents[1]

WORKER_MODULES = (
    "src.gimap.app",
    "src.gimap.app.jobs",
    "src.gimap.integrations.jobs",
    "src.gimap.integrations.jobs.worker_main",
    "src.gimap.integrations.bornagain",
    "src.gimap.integrations.bornagain.worker",
    "src.gimap.integrations.tensorflow.worker",
    "src.gimap.features.xrr.infrastructure.workers",
    "src.gimap.features.analyze.infrastructure.frame_workers",
    "src.gimap.features.fitting.infrastructure.adapters.workflow_v5",
    "src.gimap.features.fitting.infrastructure.adapters.ai_pipeline",
    "src.gimap.features.fitting.infrastructure.adapters.single_fit1d",
    "src.gimap.features.trainset.infrastructure.adapters.model_contract",
)
"""What a job or batch worker imports before its handler runs (every job handler's module included)."""


def _fake_main(**attributes) -> types.ModuleType:
    fake = types.ModuleType("__main__")
    for name, value in attributes.items():
        setattr(fake, name, value)
    return fake


def test_worker_modules_do_not_import_pyqt5() -> None:
    code = textwrap.dedent(
        f"""
        import importlib, json, sys
        loaded = {{}}
        for name in {list(WORKER_MODULES)!r}:
            importlib.import_module(name)
            qt = sorted(module for module in sys.modules if module.split(".")[0] == "PyQt5")
            if qt:
                loaded[name] = qt
                break
        print(json.dumps(loaded))
        """
    )
    environment = {**os.environ, "MPLBACKEND": "Qt5Agg"}  # what a worker inherits from main.py
    completed = subprocess.run(
        [sys.executable, "-c", code], cwd=PROJECT_ROOT, env=environment, capture_output=True, text=True,
        timeout=180,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout.strip().splitlines()[-1]) == {}


def test_app_package_names_load_on_first_use() -> None:
    app = importlib.import_module("src.gimap.app")

    for name in app.__all__:
        assert getattr(app, name) is not None
        assert name in dir(app)
    assert app.AppContext is importlib.import_module("src.gimap.app.context").AppContext
    with pytest.raises(AttributeError):
        app.NoSuchName  # noqa: B018


def test_light_main_points_a_script_main_at_the_empty_worker_module(monkeypatch) -> None:
    assert importlib.util.find_spec(WORKER_MAIN) is not None
    fake = _fake_main(__file__="main.py", __spec__=None)
    monkeypatch.setitem(sys.modules, "__main__", fake)

    with light_main():
        assert fake.__spec__.name == WORKER_MAIN
        with light_main():  # nested (one lock for the whole process): left as it is
            assert fake.__spec__.name == WORKER_MAIN
        assert fake.__spec__.name == WORKER_MAIN
    assert fake.__spec__ is None


@pytest.mark.parametrize(
    "attributes",
    [
        {"__file__": "tool.py", "__spec__": importlib.machinery.ModuleSpec("tool_package.tool", None)},
        {"__spec__": None},  # an interactive session: spawn runs no main module
    ],
)
def test_light_main_leaves_other_main_modules_alone(monkeypatch, attributes) -> None:
    fake = _fake_main(**attributes)
    monkeypatch.setitem(sys.modules, "__main__", fake)
    before = fake.__spec__

    with light_main():
        assert fake.__spec__ is before
    assert fake.__spec__ is before


PROBE_HANDLER = """
import sys


def loaded(payload, report, is_cancelled):
    mp_main = sys.modules.get("__mp_main__")
    return {
        "pyqt5": sorted(name for name in sys.modules if name.split(".")[0] == "PyQt5"),
        "main": getattr(getattr(mp_main, "__spec__", None), "name", None),
    }
"""

GUI_LIKE_SCRIPT = """
\"\"\"An invented GUI entry script: PyQt5 at the top (like main.py), then one job.\"\"\"
import json
import sys

sys.path.insert(1, {root!r})

from PyQt5 import QtCore  # noqa: F401

from src.gimap.app.jobs import JobRequest
from src.gimap.integrations.jobs import LocalProcessJobRunner

if __name__ == "__main__":
    result = LocalProcessJobRunner().run(JobRequest(handler="gimap_isolation_probe:loaded", timeout_seconds=120))
    error = None if result.error is None else result.error.message
    print(json.dumps({{"status": result.status, "value": result.value, "error": error}}))
"""


def test_job_worker_of_a_pyqt5_script_loads_no_pyqt5(tmp_path: Path) -> None:
    pytest.importorskip("PyQt5.QtCore")
    (tmp_path / "gimap_isolation_probe.py").write_text(PROBE_HANDLER, encoding="utf-8")
    script = tmp_path / "gui_like_entry.py"
    script.write_text(GUI_LIKE_SCRIPT.format(root=str(PROJECT_ROOT)), encoding="utf-8")

    completed = subprocess.run(
        [sys.executable, str(script)], cwd=tmp_path, capture_output=True, text=True, timeout=240
    )

    assert completed.returncode == 0, completed.stderr
    outcome = json.loads(completed.stdout.strip().splitlines()[-1])
    assert outcome["status"] == "succeeded", outcome["error"]
    assert outcome["value"] == {"pyqt5": [], "main": WORKER_MAIN}
