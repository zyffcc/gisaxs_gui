import ast
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication, QLabel

from src.gimap.app.presentation import (
    AdvancedSection,
    EmptyState,
    ErrorBanner,
    FilePicker,
    JobStatus,
    ParameterSection,
    PlotPanel,
    ResultTable,
)
from src.gimap.app.presentation.showcase import DesignSystemShowcase


_TEST_APP = None


def _app():
    global _TEST_APP
    _TEST_APP = QApplication.instance() or QApplication([])
    return _TEST_APP


def test_shared_components_construct_without_feature_or_scientific_dependencies():
    _app()
    widgets = [
        ParameterSection("Input"),
        AdvancedSection(),
        FilePicker(),
        PlotPanel(),
        ResultTable(("A", "B")),
        JobStatus(),
        EmptyState(),
        ErrorBanner("Error", "Message"),
    ]

    assert all(widget is not None for widget in widgets)
    root = Path("src/gimap/app/presentation")
    imports = []
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imports.extend(
            node.module or ""
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        )
    assert not any("gimap.features" in name for name in imports)
    assert not any(name.startswith("tensorflow") for name in imports)
    assert not any(name.startswith("bornagain") for name in imports)


def test_advanced_section_preserves_children_and_emits_expansion_state():
    _app()
    section = AdvancedSection(expanded=False)
    child = QLabel("value")
    states = []
    section.expandedChanged.connect(states.append)
    section.add_widget(child)

    section.set_expanded(True)
    section.set_expanded(False)

    assert child.parent() is section.content
    assert states == [True, False]
    assert child.text() == "value"


def test_file_picker_and_job_status_emit_intent_signals_only():
    _app()
    picker = FilePicker()
    paths = []
    browse = []
    picker.pathChanged.connect(paths.append)
    picker.browseRequested.connect(lambda: browse.append(True))
    picker.set_path("/tmp/example.tif")
    picker.browse_button.click()

    job = JobStatus()
    pauses = []
    cancels = []
    job.pauseRequested.connect(pauses.append)
    job.cancelRequested.connect(lambda: cancels.append(True))
    job.set_state("running", "Working", progress=0.25)
    job.pause_button.click()
    job.cancel_button.click()

    assert paths[-1] == "/tmp/example.tif"
    assert browse == [True]
    assert pauses == [True]
    assert cancels == [True]
    assert job.progress_bar.value() == 250


def test_result_table_empty_state_and_showcase_construct_offscreen():
    _app()
    table = ResultTable(("Name", "State"))
    assert table.empty_label.isVisible() is False or table.rowCount() == 0
    table.set_rows((("scan", "ready"),))
    assert table.rowCount() == 1
    showcase = DesignSystemShowcase()
    assert showcase.windowTitle() == "GIMaP UI Design System"
