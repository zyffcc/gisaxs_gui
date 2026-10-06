"""Wave 2b Labs check: a folder dropped on a Labs panel stays there (Prediction uses it as the folder batch,
Trainset answers on the status line), Undo of Reset keeps a step's template, every Prediction disclosure is
composed again after a language switch, and an exported preprocessing step is recorded by its label."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtCore import QMimeData, QPoint, Qt, QUrl
from PyQt5.QtGui import QDragEnterEvent, QDropEvent
from PyQt5.QtWidgets import QApplication, QWidget

from src.gimap.app.presentation.components import visible_toasts
from src.gimap.features.prediction.presentation.workflow_components import PredictionDisclosure
from tests.test_labs_prediction import _window
from tests.test_labs_trainset import _trainset


def _drop(widget: QWidget, path: Path) -> bool:
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(path))])
    enter = QDragEnterEvent(QPoint(10, 10), Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier)
    QApplication.sendEvent(widget, enter)
    accepted = enter.isAccepted()
    QApplication.sendEvent(widget, QDropEvent(QPoint(10, 10), Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier))
    return accepted


def test_a_folder_dropped_on_the_prediction_canvas_becomes_the_folder_batch(tmp_path):
    for index in (1, 2, 3):
        (tmp_path / f"frame_{index:05d}.cbf").write_bytes(b"not read by the scan")
    window = _window()
    binding = window.runtime.prediction
    try:
        window.runtime.navigate("predict")
        assert _drop(window.predictionPlotPanel, tmp_path)
        assert window.gisaxsPredictMultiFilesRadioButton.isChecked()
        assert Path(binding.current_parameters["input_folder"]) == tmp_path
        assert binding._available_indices == [1, 2, 3]
        assert window.components.current_page_key() == "predict"  # not passed on to the window (Analyze)
    finally:
        window.close()


def test_a_folder_dropped_on_the_design_preview_is_answered_and_keeps_the_reference(tmp_path):
    window, binding, page = _trainset(tmp_path)
    said = []
    binding.status_updated.connect(said.append)
    try:
        reference = page.reference_path.text()
        assert _drop(page.trainset_design_preview_panel, tmp_path)
        assert page.reference_path.text() == reference
        assert said[-1] == "Drop one detector image file to use it as the reference"
    finally:
        window.close()


def test_undo_of_reset_restores_a_step_as_its_template(tmp_path):
    window, binding, page = _trainset(tmp_path)
    try:
        page.set_step_state(4, "Job {job}", job=4711)
        binding._reset_clicked()
        assert page.step_entries()[4] == ("Not started", {})
        toast = visible_toasts(page)[-1]
        toast.action_button.click()  # Undo
        # The template comes back with its value, so it is shown in the language of the moment.
        assert page.step_entries()[4] == ("Job {job}", {"job": 4711})
        assert page.step_list.detail("monitor") == "Job 4711"
    finally:
        window.close()


def test_every_prediction_disclosure_is_composed_again_by_refresh_language():
    window = _window()
    binding = window.runtime.prediction
    try:
        disclosures = window.gisaxsPredictPage.findChildren(PredictionDisclosure)
        names = {disclosure.objectName() for disclosure in disclosures}
        assert len(disclosures) >= 3 and "predictionActivityDisclosure" in names  # log, model, colour ranges
        for disclosure in disclosures:
            disclosure._title = "Sentinel"
        binding.refresh_language()
        assert {disclosure.toggle.text() for disclosure in disclosures} == {"Sentinel"}
        assert {disclosure.toggle.toolTip() for disclosure in disclosures} == {"Show or hide: Sentinel"}
    finally:
        window.close()


def test_an_exported_preprocessing_step_is_recorded_by_its_label():
    window = _window()
    binding = window.runtime.prediction
    try:
        binding._step_snapshots = [{"label": "crop"}, {"label": "log"}]
        binding._current_step_index = 1
        assert binding._shown_label("steps") == "steps: log"
        assert binding._shown_label("hr") == "hr"
        binding._step_snapshots = []
        assert binding._shown_label("steps") == "steps"
    finally:
        window.close()
