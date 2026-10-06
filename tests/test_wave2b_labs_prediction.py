"""Wave 2b, Labs / 2D Prediction: CBF-only wording and the import card layout (labs-13), the empty canvas's
chooser button and dropped files (labs-12), exports as PNG + data + JSON record with a toast (labs-15),
and run-time texts composed again after a switch of the interface language (labs-4)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtCore import QMimeData, QPoint, Qt, QUrl
from PyQt5.QtGui import QDragEnterEvent, QDropEvent
from PyQt5.QtWidgets import QApplication, QLabel

from src.gimap.app.presentation.components import visible_toasts
from src.gimap.app.presentation.i18n import ZH, apply_language
from src.gimap.features.prediction.application import ExportPredictionRecord, PredictionRecordRequest
from src.gimap.features.prediction.presentation import trend_windows
from src.gimap.features.prediction.presentation.bindings import input_parameters
from src.gimap.features.prediction.presentation.workbench_layout import EMPTY_ACTION
from tests.test_labs_prediction import _app, _wait, _window

QSS = Path(__file__).resolve().parents[1] / "src/gimap/features/prediction/presentation/prediction_theme.qss"


class _Repository:
    def __init__(self):
        self.written = {}

    def write_text(self, path, content):
        self.written[Path(path)] = content
        return Path(path)


def test_the_record_names_inputs_module_model_and_steps_without_the_snapshot_images(tmp_path):
    repository = _Repository()
    request = PredictionRecordRequest(
        path=tmp_path / "frame_00007_prediction.json",
        outputs=(tmp_path / "frame_00007_prediction.png", tmp_path / "frame_00007_prediction.txt"),
        shown="hr",
        input_files=("E:/data/frame_00007.cbf", "E:/data/frame_00008.cbf"),
        stack=2,
        module={"name": "HR distribution", "id": "hr_v2", "version": "2.1", "file": "modules/hr/module.yaml"},
        model_path="modules/hr/model.keras",
        framework="tensorflow 2.15.0",
        runtime_name="tensorflow",
        runtime_version="2.15.0",
        preprocess_entry="preprocess.py:run",
        preprocess_steps=("crop", "log", "normalize"),
        applied_steps=({"label": "log", "image": np.ones((64, 32)), "floor": np.float32(1e-3), "axis": np.arange(3)},),
    )
    written = ExportPredictionRecord(repository).execute(request)
    record = json.loads(repository.written[written])
    assert written == tmp_path / "frame_00007_prediction.json"
    assert record["kind"] == "prediction" and record["shown"] == "hr"
    assert record["outputs"] == ["frame_00007_prediction.png", "frame_00007_prediction.txt"]
    assert record["input"] == {"mode": "single_file", "files": list(request.input_files), "stack": 2}
    assert record["module"]["name"] == "HR distribution" and record["model"]["runtime_version"] == "2.15.0"
    assert record["model"]["path"] == "modules/hr/model.keras" and record["model"]["framework"] == "tensorflow 2.15.0"
    assert record["preprocessing"]["steps"] == ["crop", "log", "normalize"]
    step = record["preprocessing"]["applied"][0]
    assert step["image_shape"] == [64, 32] and "image" not in step  # the snapshot is not written out
    assert step["floor"] == pytest.approx(1e-3) and step["axis"] == [0, 1, 2]


def test_the_import_card_says_cbf_only_and_keeps_its_fields_beside_their_labels():
    window = _window()
    try:
        ui = window
        assert ui.gisaxsPredictChooseGisaxsFileButton.text() == "Choose CBF frame…"
        hints = [label.text() for label in ui.predictionInputModePanel.findChildren(QLabel)]
        assert "Prediction reads CBF frames (.cbf) only." in hints
        grid = ui.gisaxsPredictStackLabel.parentWidget().layout()
        assert (grid.columnStretch(1), grid.columnStretch(2)) == (0, 1)  # the free width is an empty column
        assert ui.gisaxsPredictStackLabel.alignment() & Qt.AlignLeft
        rule = QSS.read_text(encoding="utf-8").split("QFrame#predictionModeSelector QRadioButton::indicator")[1]
        rule = rule.split("}")[0]
        for declaration in ("width: 0", "height: 0", "border: none", "image: none", "margin: 0"):
            assert declaration in rule
    finally:
        window.close()


def test_the_empty_canvas_opens_the_cbf_chooser_and_dropped_files_go_to_the_file_handling(tmp_path, monkeypatch):
    window = _window()
    binding = window.runtime.prediction
    try:
        panel = window.predictionPlotPanel
        empty = panel.empty_state
        assert empty.action_button.text() == EMPTY_ACTION and not empty.action_button.isHidden()
        asked = []

        def choose(_parent, title, _folder, filters):
            asked.append((title, filters))
            return "", ""

        monkeypatch.setattr(input_parameters.QFileDialog, "getOpenFileName", staticmethod(choose))
        empty.action_button.click()
        assert asked and asked[0][0] == "Select CBF Frame" and asked[0][1].startswith("CBF frames (*.cbf)")

        handled = []
        monkeypatch.setattr(binding, "_handle_new_file_selection", handled.append)
        frame = tmp_path / "frame_00001.cbf"
        frame.write_bytes(b"not read here")
        mime = QMimeData()
        mime.setUrls([QUrl.fromLocalFile(str(frame))])
        enter = QDragEnterEvent(QPoint(5, 5), Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier)
        QApplication.sendEvent(panel, enter)
        assert enter.isAccepted()
        QApplication.sendEvent(panel, QDropEvent(QPoint(5, 5), Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier))
        assert [Path(path) for path in handled] == [frame]
        viewport = window.gisaxsImageGraphicsView.viewport()  # on the image itself (its scene takes drags)
        enter = QDragEnterEvent(QPoint(5, 5), Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier)
        QApplication.sendEvent(viewport, enter)
        assert enter.isAccepted()
        QApplication.sendEvent(viewport, QDropEvent(QPoint(5, 5), Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier))
        assert [Path(path) for path in handled] == [frame, frame]
    finally:
        window.close()


def test_a_dropped_tiff_is_refused_with_a_message_that_says_cbf_only(tmp_path, monkeypatch):
    window = _window()
    binding = window.runtime.prediction
    shown = []
    monkeypatch.setattr(input_parameters.QMessageBox, "warning", staticmethod(lambda *args: shown.append(args)))
    try:
        tiff = tmp_path / "frame.tif"
        tiff.write_bytes(b"II*\x00")
        window.predictionWorkbenchLayout._file_dropped(str(tiff))
        assert shown and shown[0][1] == "Unsupported Format" and "CBF" in shown[0][2] and "frame.tif" in shown[0][2]
        assert binding.current_parameters.get("input_file") != str(tiff)
    finally:
        window.close()


def _prediction_shown(binding, tmp_path):
    y, x = np.mgrid[0:24, 0:24]
    binding._current_image_path = str(tmp_path / "frame_00007.cbf")
    binding._current_input_files = [str(tmp_path / "frame_00007.cbf"), str(tmp_path / "frame_00008.cbf")]
    binding._current_input_stack = 2
    binding._latest_preprocess_steps = [{"label": "log", "image": np.ones((24, 24))}]
    binding._latest_runtime = SimpleNamespace(runtime_name="tensorflow", runtime_version="2.15.0")
    binding._display_prediction({"hr": np.exp(-((x - 12) ** 2 + (y - 9) ** 2) / 20.0)})


def test_export_result_writes_a_png_its_data_and_a_record_with_one_stem_and_says_so(tmp_path, monkeypatch):
    window = _window()
    binding = window.runtime.prediction
    try:
        _prediction_shown(binding, tmp_path)
        suggested = []

        def ask(title, name):
            suggested.append((title, name))
            return str(tmp_path / "out" / name[: -len(".png")])  # typed without the extension

        (tmp_path / "out").mkdir()
        monkeypatch.setattr(binding, "_ask_save_path", ask)
        binding._on_predict_export_clicked()

        assert suggested == [("Save Prediction Result", "frame_00007_prediction.png")]
        out = tmp_path / "out"
        assert sorted(path.name for path in out.iterdir()) == [
            "frame_00007_prediction.json", "frame_00007_prediction.png", "frame_00007_prediction.txt",
        ]
        assert np.loadtxt(out / "frame_00007_prediction.txt").shape == (24, 24)
        record = json.loads((out / "frame_00007_prediction.json").read_text(encoding="utf-8"))
        assert record["input"]["files"][1].endswith("frame_00008.cbf") and record["input"]["stack"] == 2
        assert record["model"]["runtime"] == "tensorflow" and record["preprocessing"]["applied"][0]["label"] == "log"
        assert binding.current_parameters["export_path"] == str(out)  # the next dialog opens here
        toast = visible_toasts(window.gisaxsPredictPage)[-1]
        assert toast.property("level") == "ok" and "frame_00007_prediction.png" in toast.text()
        assert toast.action_button is not None and toast.action_button.text() == "Open Folder"
    finally:
        window.close()


def test_a_failed_export_is_an_error_toast_and_export_input_is_a_png_with_its_record(tmp_path, monkeypatch):
    window = _window()
    binding = window.runtime.prediction
    try:
        _prediction_shown(binding, tmp_path)
        monkeypatch.setattr(binding, "_ask_save_path", lambda *_args: str(tmp_path / "missing" / "x.png"))
        binding._on_predict_export_clicked()  # the folder does not exist: nothing can be written
        assert visible_toasts(window.gisaxsPredictPage)[-1].property("level") == "error"

        binding._current_pixmap = binding._predict_pixmap  # an input preview to export
        monkeypatch.setattr(binding, "_ask_save_path", lambda _title, name: str(tmp_path / name))
        binding._export_gisaxs_image()
        assert (tmp_path / "frame_00007_input.png").is_file()
        record = json.loads((tmp_path / "frame_00007_input.json").read_text(encoding="utf-8"))
        assert record["kind"] == "input preview" and "model" not in record and "colormap" in record["display"]
    finally:
        window.close()


def test_run_time_texts_follow_a_switch_of_the_language_both_ways():
    window = _window()
    binding = window.runtime.prediction
    try:
        window.show()
        window.runtime.navigate("predict")
        binding._available_indices = [3, 9]
        binding._update_range_tooltip()
        disclosure = window.predictionActivityDisclosure
        english = (
            window.gisaxsPredictInputReadyLabel.text(),
            window.gisaxsPredictStackLabel.toolTip(),
            disclosure.toggle.toolTip(),
            window.predictionPlotPanel.empty_state.message_label.text(),
            window.gisaxsPredictModelStatusTextLabel.text(),
        )
        assert english[1] == "Available index range: 3 - 9" and english[2] == "Show or hide: Activity log"
        apply_language("zh", [window])
        assert window.gisaxsPredictInputReadyLabel.text() == ZH["Input: Missing"]
        assert window.gisaxsPredictStackLabel.toolTip() == ZH["Available index range: {first} - {last}"].format(
            first=3, last=9)
        assert disclosure.toggle.toolTip() == ZH["Show or hide: {title}"].format(title=ZH["Activity log"])
        assert window.predictionPlotPanel.empty_state.action_button.text() == ZH[EMPTY_ACTION]
        assert window.gisaxsPredictModelStatusTextLabel.text() == ZH["Not loaded"]
        apply_language("en", [window])
        assert (
            window.gisaxsPredictInputReadyLabel.text(),
            window.gisaxsPredictStackLabel.toolTip(),
            disclosure.toggle.toolTip(),
            window.predictionPlotPanel.empty_state.message_label.text(),
            window.gisaxsPredictModelStatusTextLabel.text(),
        ) == english
    finally:
        apply_language("en")
        window.close()


def test_trend_figures_use_a_chinese_capable_font_and_all_parameters_is_not_a_name():
    _app()
    results = SimpleNamespace(get_completed_results=lambda: [])
    assert trend_windows.figure_font() == {}
    window = trend_windows.ParameterTrendWindow(results)
    try:
        assert window.parameter_combo.itemData(0) == trend_windows.ALL_PARAMETERS
        apply_language("zh")
        window.refresh_parameters_and_plot()
        assert window.parameter_combo.itemText(0) == ZH["All parameters"]
        font = trend_windows.figure_font()
        if font:  # a CJK font is installed: it follows matplotlib's own
            assert font["fontfamily"][-1] in trend_windows._CJK_FONTS
        assert window.status_label.text() == ZH["No completed parameter predictions yet."]
    finally:
        apply_language("en")
        window.close()
    _wait(lambda: True, 0.01)

