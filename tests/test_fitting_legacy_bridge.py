"""Guard the legacy fitting bridge while callers are migrated incrementally."""

from __future__ import annotations

import ast
from pathlib import Path


CONTROLLER = (
    Path(__file__).resolve().parents[1] / "controllers" / "fitting_controller.py"
)


def _imports():
    tree = ast.parse(CONTROLLER.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            yield from (alias.name.casefold() for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            yield node.module.casefold()


def test_legacy_controller_does_not_load_heavy_runtime_integrations():
    imported = tuple(_imports())

    assert not any(name == "bornagain" or name.startswith("bornagain.") for name in imported)
    assert not any(name == "tensorflow" or name.startswith("tensorflow.") for name in imported)
    assert not any(name == "keras" or name.startswith("keras.") for name in imported)


def test_legacy_controller_no_longer_owns_ai_process_or_pipeline():
    source = CONTROLLER.read_text(encoding="utf-8")
    imported = tuple(_imports())

    assert "QProcess" not in source
    assert "utils.ai_fitting_pipeline" not in imported
