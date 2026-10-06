"""Wave 2b, Labs / Trainset: the steps in the shared StepRail (labs-8), run-time texts in the interface
language and composed again after a switch (labs-4), Local Run path rows with Browse… and remembered
folders (labs-9), and a file dropped on the design preview loading the reference (labs-12)."""

from __future__ import annotations

import ast
import os
import string
from pathlib import Path

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtCore import QMimeData, QPoint, Qt, QUrl
from PyQt5.QtGui import QDragEnterEvent, QDropEvent
from PyQt5.QtWidgets import QApplication, QFormLayout, QLineEdit, QPushButton, QSizePolicy, QSpinBox, QWidget

from src.gimap.app.presentation.components import StepRail
from src.gimap.app.presentation.i18n import ZH, apply_language, current_language
from src.gimap.features.trainset.presentation.bindings import validation_files
from src.gimap.features.trainset.presentation.page import TrainsetBuildPage
from src.gimap.features.trainset.presentation.sections.shell_layout import NUMBER_FIELD_MAX_WIDTH
from src.gimap.features.trainset.presentation.step_rail import rail_state
from tests.test_labs_trainset import _app, _settle, _trainset

ROOT = Path(__file__).resolve().parents[1]
TRAINSET = ROOT / "src" / "gimap" / "features" / "trainset" / "presentation"
# Calls whose literal first text argument (after the widget) is an English template shown translated.
TEMPLATE_CALLS = {"set": 1, "set_step_state": 1, "set_what_if_busy": 1, "set_what_if_result": 1}


def _fields(text: str):
    return sorted((name, spec) for _literal, name, spec, _conv in string.Formatter().parse(text) if name is not None)


def _literals(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        yield node.value
    elif isinstance(node, ast.IfExp):
        yield from _literals(node.body)
        yield from _literals(node.orelse)


def test_every_composed_trainset_template_has_its_chinese_with_the_same_fields():
    """The texts given to ``page.texts.set``, ``set_step_state`` and the what-if setters are English templates
    that are translated when shown: each is in the zh table and keeps its {fields}."""
    missing, fields = [], []
    for path in sorted(TRAINSET.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
                continue
            index = TEMPLATE_CALLS.get(node.func.attr)
            if index is None or len(node.args) <= index:
                continue
            if node.func.attr == "set" and not (
                isinstance(node.func.value, ast.Attribute) and node.func.value.attr == "texts"
            ):
                continue
            for text in _literals(node.args[index]):
                if not any(character.isalpha() for character in text) or text == "{error}":
                    continue
                if text not in ZH:
                    missing.append(f"{path.relative_to(ROOT)}:{node.lineno} {text!r}")
                elif _fields(text) != _fields(ZH[text]):
                    fields.append(text)
    assert missing == [] and fields == []


def test_the_steps_are_the_shared_step_rail_addressed_by_row():
    _app()
    page = TrainsetBuildPage()
    page.resize(1280, 800)
    page.show()
    _settle(0.1)
    rail = page.step_list
    assert isinstance(rail, StepRail) and rail.count() == len(page.STEPS) == 5
    assert page.findChild(QWidget, "trainsetStepList") is page.step_panel  # the card around it
    assert rail.currentRow() == 0 and page.back_button.isHidden()
    rows = []
    rail.currentRowChanged.connect(rows.append)
    rail.findChild(QWidget, "gimapStep_run").clicked.emit()  # a click on step 4
    assert rows == [3] and page.stack.currentIndex() == 3 and not page.back_button.isHidden()
    page.back_button.click()
    assert rail.currentRow() == 2 and page.stack.currentIndex() == 2
    rail.setCurrentRow(1)  # what Open local preview does (the binding connects it)
    assert page.stack.currentIndex() == 1

    page.set_step_state(0, "Reference loaded")
    page.set_step_state(4, "Job {job}", job=4711)
    page.set_step_state(3, "Failed")
    assert page.step_states() == ["Reference loaded", "Not started", "Not started", "Failed", "Job 4711"]
    assert [rail.state(key) for key in rail.keys()] == ["ok", "pending", "pending", "error", "busy"]
    assert rail.detail("monitor") == "Job 4711"
    assert (rail_state("Running"), rail_state("Completed"), rail_state("Timed out")) == ("busy", "ok", "error")
    page.close()


def test_run_time_texts_follow_the_language_and_are_composed_again_after_a_switch(tmp_path):
    window, binding, page = _trainset(tmp_path)
    try:
        window.show()
        window.runtime.navigate("trainset")
        for key, value in (("x", 10), ("y", 10), ("width", 64), ("height", 48)):
            page.fields[f"roi.{key}"].setValue(value)  # an ROI inside the 120 x 96 reference
        _settle(0.2)
        threshold_en = page.threshold_summary.text()
        info_en = page.design_info.text()
        roi_en = page.roi_range_label.text()
        matrix_en = page.cache_grid_summary.text()
        assert threshold_en.startswith("Reference threshold locations:")
        assert "ROI tensor:" in info_en and roi_en.startswith("BornAgain detector:")
        assert matrix_en.startswith("Matrix:")

        page.set_step_state(1, "Preview ready")
        page.set_validation_state("Preview ready", "ok")
        steps = page.step_states()
        apply_language("zh", [window])
        assert current_language() == "zh"
        # A translated choice is not an edit: the design checks stay, and the configuration keeps its values.
        assert page.step_states() == steps and page.validation_state() == "ok"
        preset = page.fields["detector.preset"]
        assert preset.currentText() == ZH["Custom"] and binding._collect_config()["detector"]["preset"] == "Custom"
        page.add_model_layer({"type": "dense", "units": 4, "activation": "linear"})
        assert page.model_layers()[-1]["activation"] == "linear"
        rail = page.step_list
        assert rail.detail("dataset") == ZH[page.step_states()[0]]  # e.g. ROI ready / Mask ready, in Chinese
        assert rail.findChild(QWidget, "gimapStep_dataset").title.text() == ZH["Dataset Design"]
        assert page.validation_badge.text() == ZH[page.validation_text()]
        assert page.trainset_action_hint.text() == ZH["Validate the detector, ROI, particle and sampling design."]
        for label, english in ((page.threshold_summary, threshold_en), (page.design_info, info_en),
                               (page.roi_range_label, roi_en), (page.cache_grid_summary, matrix_en)):
            assert label.text() != english and not label.text().startswith(english.split(":")[0])
        assert page.threshold_summary.text().startswith("参考图阈值位置")
        binding._begin_mask("ellipse")
        assert window.statusbar.currentMessage() == ZH["Draw an elliptical fixed mask in ROI coordinates"]

        apply_language("en", [window])
        assert page.threshold_summary.text() == threshold_en
        assert page.design_info.text() == info_en
        assert page.roi_range_label.text() == roi_en
        assert page.cache_grid_summary.text() == matrix_en
        assert rail.detail("dataset") == page.step_states()[0]
    finally:
        apply_language("en")
        window.close()


def test_local_run_paths_have_browse_beside_their_fields_and_numbers_keep_a_natural_width(tmp_path, monkeypatch):
    window, binding, page = _trainset(tmp_path)
    try:
        for button, field in (
            (page.local_folder_button, "project.workspace"),
            (page.local_dataset_folder_button, "runtime.dataset_output_dir"),
            (page.local_results_folder_button, "runtime.results_output_dir"),
            (page.local_python_button, "training.local_python"),
            (page.local_cache_folder_button, "simulation.grid_cache.directory"),
        ):
            assert button.text() == "Browse…" and button.toolTip().startswith("Choose ")
            assert button.parentWidget() is page.fields[field].parentWidget()  # one row: field | Browse…
            row = button.parentWidget()
            form = row.parentWidget().layout()
            assert isinstance(form, QFormLayout) and form.labelForField(row) is not None
        assert not any(
            isinstance(widget, QPushButton) and widget.text().startswith("Choose ") and widget.text().endswith("…")
            for widget in page.findChildren(QPushButton)
        )
        spin = page.fields["training.batch_size"]
        assert isinstance(spin, QSpinBox) and spin.maximumWidth() == NUMBER_FIELD_MAX_WIDTH == 240
        assert spin.sizePolicy().horizontalPolicy() == QSizePolicy.Preferred
        assert page.fields["sample.particle_label"].maximumWidth() == 240
        assert isinstance(page.fields["hpc.remote_path"], QLineEdit)
        assert page.fields["hpc.remote_path"].maximumWidth() > 10000  # text fields still stretch

        chosen = tmp_path / "datasets"
        chosen.mkdir()
        starts = []

        def choose(_parent, _title, start):
            starts.append(start)
            return str(chosen)

        monkeypatch.setattr(validation_files.QFileDialog, "getExistingDirectory", staticmethod(choose))
        page.local_dataset_folder_button.click()
        assert page.fields["runtime.dataset_output_dir"].text() == str(chosen)
        page.fields["runtime.dataset_output_dir"].setText("")
        page.local_dataset_folder_button.click()
        assert Path(starts[-1]) == chosen  # the next chooser opens where the last one ended
        assert binding.trainset_view_model.last_folder("dataset") == str(chosen)
    finally:
        window.close()


def _drop(widget: QWidget, path: Path) -> bool:
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(path))])
    point = QPoint(10, 10)
    enter = QDragEnterEvent(point, Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier)
    QApplication.sendEvent(widget, enter)
    accepted = enter.isAccepted()
    drop = QDropEvent(point, Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier)
    QApplication.sendEvent(widget, drop)
    return accepted


def test_a_file_dropped_on_the_design_preview_becomes_the_reference(tmp_path):
    window, binding, page = _trainset(tmp_path)
    try:
        other = tmp_path / "dropped.npy"
        np.save(other, np.ones((40, 50), dtype=np.float32))
        panel = page.trainset_design_preview_panel
        assert panel.acceptDrops()
        assert _drop(panel, other)
        assert Path(page.reference_path.text()) == other
        assert binding.reference_image.shape == (40, 50)
        assert page.step_states()[0] == "ROI ready" or page.step_states()[0] == "Reference loaded"
        _drop(panel, tmp_path)  # a folder is not a reference file: the reference stays
        assert Path(page.reference_path.text()) == other
    finally:
        window.close()
