"""``requirements.txt`` names what the program imports: nothing more, nothing missing.

A source install is ``pip install -r requirements.txt``.  A package nobody imports only makes that
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
                continue  # frozen external kit; required host dependencies are EXTERNAL_RUNTIME above
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


def test_every_requirement_is_imported_by_the_program(code):
    imported = {module.lower() for module in code[0]}
    unused = sorted(
        name for name in _requirements("requirements.txt")
        if IMPORTED_AS.get(name, name).lower() not in imported
    )
    assert not unused, f"requirements.txt lists packages no code imports: {unused}"


def test_every_package_the_program_imports_is_a_requirement(code):
    imported, own = code
    listed = _requirements("requirements.txt")
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
