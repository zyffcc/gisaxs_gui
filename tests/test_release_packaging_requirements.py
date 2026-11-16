"""The requirement files name what the program imports: nothing more, nothing missing.

``requirements.txt`` is the core (``pip install -r requirements.txt``); TensorFlow, BornAgain and the AI clients are
optional, each in its own file (``OPTIONAL``), and ``requirements-full.txt`` names them all. The program starts and
opens every workspace without the optional ones.  A package nobody imports only makes that
install longer (``tqdm`` was one).  A package the program imports that the file does not name is not
there after the install: ``psutil`` was one -- Batch Export reads the physical cores and the free
memory with it, and falls back to the operating system without it, so it is listed as optional.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
# Where the program's code is: the application, the bundled models, the research code it runs in a
# subprocess, and the command-line tools.
CODE = ("main.py", "src", "modules", "utils", "tools")
# A requirement whose module has another name than the package.
IMPORTED_AS = {
    "opencv-python": "cv2",
    "conda-pack": "conda_pack",
    "pyyaml": "yaml",
    "pyqt5": "PyQt5",
    "tensorflow-intel": "tensorflow",
}
# Imported by the program, installed by a requirement that is listed.
COMES_WITH = {"PIL": "matplotlib", "keras": "tensorflow", "sip": "pyqt5"}
# Not packages: modules of the model's training runs that are kept in the local development records
# only. The frozen model code (conditional_v2/train_teacher_path_control.py) imports them inside its
# training entry point, which the program never calls.
TRAINING_ONLY = {"train_scaled_condition", "train_generalization_controls"}
# The optional kit is outside the source scanner, but its released fit1d worker needs this
# dependency to limit BLAS after NumPy has already been imported by the host's spawned job.
EXTERNAL_RUNTIME = {"threadpoolctl"}
# This worker runs under an explicitly selected, separate interpreter, never in the host GUI.
ISOLATED_RUNTIMES = {
    "src/gimap/features/sem/infrastructure/stardist_worker.py": "requirements-sem-stardist.txt",
}


OPTIONAL = {
    "requirements-tensorflow.txt": {"tensorflow", "tensorflow-intel"},
    "requirements-bornagain.txt": {"bornagain"},
    "requirements-ai.txt": {"anthropic", "openai"},
}
"""Each optional part and the packages it adds."""


def _requirements(name: str) -> set[str]:
    names = set()
    for line in (ROOT / name).read_text(encoding="utf-8").splitlines():
        match = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9_.\-]*)", line)
        if match and not line.lstrip().startswith(("#", "-")):
            names.add(match.group(1).lower())
    return names


def _imports(tree) -> set[str]:
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imported.add(node.module.split(".")[0])
    return imported


@pytest.fixture(scope="module")
def code():
    """``(top-level modules imported anywhere, names of the program's own modules and folders)``."""
    if not hasattr(sys, "stdlib_module_names"):
        pytest.skip("needs Python 3.10 or later to tell the standard library apart")
    imported, own = set(EXTERNAL_RUNTIME), {"main", "tests", *CODE}
    for entry in CODE:
        start = ROOT / entry
        for path in [start] if start.is_file() else sorted(start.rglob("*.py")):
            parts = path.relative_to(ROOT).parts
            if "development" in parts or "__pycache__" in parts:
                continue  # research records and caches: not part of the program, nor of a release
            if parts[:2] == ("utils", "gisaxs1d_gui_kit"):
                continue  # the bundled, frozen fit1d kit; required host dependencies are EXTERNAL_RUNTIME above
            if parts[:2] == ("utils", "sputter_client"):
                # the bundled analysis package of the Sputter page, a release build of research code run as a
                # subprocess in the program's Python: its commands need NumPy, SciPy and Matplotlib (fabio when
                # present); its other imports (torch, pyFAI, xraydb) sit on research paths the page never runs
                # (tests/test_sputter_client_bundle.py: the commands start without PyTorch)
                continue
            if path.relative_to(ROOT).as_posix() in ISOLATED_RUNTIMES:
                continue  # checked against its own runtime manifest below
            if parts[0] != "src":
                # The research and model code imports its neighbours by their bare names. ``src`` is
                # only ever imported as ``src.gimap...``: its folders (``integrations/tensorflow``
                # and ``integrations/bornagain`` among them) are not top-level modules, and counting
                # them as the program's own would hide those two packages from the second test.
                own.update(parts[:-1])
                own.add(path.stem)
            try:
                tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
            except (OSError, SyntaxError):
                continue
            imported.update(_imports(tree))
    return imported - set(sys.stdlib_module_names), own


def _listed() -> set[str]:
    """Every package of the core and of the optional parts."""
    return _requirements("requirements.txt").union(*(_requirements(name) for name in OPTIONAL))


def test_every_requirement_is_imported_by_the_program(code):
    imported = {module.lower() for module in code[0]}
    unused = sorted(name for name in _listed() if IMPORTED_AS.get(name, name).lower() not in imported)
    assert not unused, f"the requirement files list packages no code imports: {unused}"


def test_the_core_install_has_none_of_the_optional_parts_and_full_has_them_all():
    core = _requirements("requirements.txt")
    for name, packages in OPTIONAL.items():
        assert _requirements(name) == packages, name
        assert not core & packages, f"{packages & core} belong to {name}, not to requirements.txt"
    full = (ROOT / "requirements-full.txt").read_text(encoding="utf-8")
    assert all(f"-r {name}" in full for name in ("requirements.txt", *OPTIONAL))
    assert "-r requirements-full.txt" in (ROOT / "requirements-release.txt").read_text(encoding="utf-8")


def test_the_program_starts_and_opens_every_workspace_without_the_optional_packages(tmp_path):
    import os
    import subprocess
    import textwrap

    script = textwrap.dedent("""
        import sys
        for name in ("tensorflow", "keras", "bornagain", "anthropic", "openai"):
            sys.modules[name] = None  # importing it raises ImportError, as when it is not installed
        from PyQt5.QtWidgets import QApplication
        app = QApplication([])
        from main import MainWindow
        from src.gimap.app import AppContext
        from src.gimap.integrations.jobs import LocalProcessJobRunner
        from src.gimap.integrations.state import (
            InMemorySessionRepository, InMemorySettingsRepository, InMemoryUserPreferencesRepository)
        window = MainWindow(AppContext(
            settings=InMemorySettingsRepository(), session=InMemorySessionRepository(),
            preferences=InMemoryUserPreferencesRepository(), jobs=LocalProcessJobRunner()))
        window.show()
        for index in range(window.mainWindowWidget.count()):
            window.mainWindowWidget.setCurrentIndex(index)
            app.processEvents()
        print("opened", window.mainWindowWidget.count())
    """)
    environment = {**os.environ, "QT_QPA_PLATFORM": "offscreen", "GIMAP_HOME": str(tmp_path)}
    result = subprocess.run([sys.executable, "-c", script], cwd=ROOT, env=environment, capture_output=True,
                            text=True, timeout=600)
    assert result.returncode == 0, result.stderr[-3000:]
    assert "opened" in result.stdout


def test_every_package_the_program_imports_is_a_requirement(code):
    imported, own = code
    listed = _listed()
    development = _requirements("requirements-dev.txt")  # pytest: the research code's own tests
    assert set(COMES_WITH.values()) <= listed
    modules = {IMPORTED_AS.get(name, name).lower() for name in listed | development}
    missing = sorted(
        name
        for name in imported - own - set(COMES_WITH) - TRAINING_ONLY
        if name.lower() not in modules
    )
    assert not missing, f"imported by the program but not in requirements.txt: {missing}"
    assert "psutil" in listed and "tqdm" not in listed


def test_fit1d_blas_controller_is_installed_in_the_host_runtime():
    assert "threadpoolctl" in _requirements("requirements.txt")
    assert "threadpoolctl" in EXTERNAL_RUNTIME


def test_isolated_sem_runtime_declares_every_worker_dependency():
    for worker, manifest in ISOLATED_RUNTIMES.items():
        imported = _imports(ast.parse((ROOT / worker).read_text(encoding="utf-8")))
        required = imported - set(sys.stdlib_module_names)
        listed = _requirements(manifest)
        modules = {IMPORTED_AS.get(name, name).lower() for name in listed}
        assert {name.lower() for name in required} <= modules
        # CSBDeep is also queried in the exported runtime provenance.
        assert {"numpy", "stardist", "csbdeep", "tensorflow"} <= listed
    assert "stardist" not in _requirements("requirements.txt")
