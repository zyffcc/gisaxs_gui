"""Release 0.1.0, Labs / 2D Prediction: a page that predicted nothing says nothing about batch results, the
activity log is written in the interface language, and the texts that had no Chinese are keys of the table
(static texts, the dialogs of the batch results, the literal texts of the message boxes)."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from PyQt5.QtWidgets import QLabel, QMessageBox

from src.gimap.app.presentation import i18n
from src.gimap.app.presentation.i18n import DEFAULT_LANGUAGE, apply_language, trf
from src.gimap.features.prediction.presentation.bindings import module_catalog
from src.gimap.features.prediction.presentation.export_dialog import ExportDialog
from tests.test_labs_prediction import _wait, _window

ROOT = Path(__file__).resolve().parents[1]
PREDICTION = ROOT / "src" / "gimap" / "features" / "prediction" / "presentation"
BATCH_READY = "Batch results are ready in the workspace"
WATCHING = "Watching module.yaml for saved edits..."
FOLDER_CAPTION = "Select TensorFlow SavedModel Folder"

# This round's Chinese (until it is merged into ``zh_*.py``: the fixture puts in what the table lacks).
ZH_ENTRIES = {
    # the page
    "Edit Config": "编辑配置",
    "Reload Config": "重新载入配置",
    "Framework:": "框架：",
    "Browse the shared model library only when a local model is not already available.":
        "只有本地还没有可用的模型时，才需要浏览共享模型库。",
    "Browse the shared DESY model repository, download a model, then use Model Import.":
        "浏览 DESY 的共享模型库，下载一个模型，然后用“导入模型...”载入。",
    FOLDER_CAPTION: "选择 TensorFlow SavedModel 文件夹",
    "Zoom -": "缩小",
    # the batch results table
    "All Status": "全部状态",
    "File Name": "文件名",
    "Stack Files": "叠加文件数",
    "Duration": "耗时",
    "Asc": "升序",
    "Desc": "降序",
    "Export This Result": "导出这个结果",
    "Retry Prediction": "重新预测",
    # its dialogs and the log window
    "Export Prediction Results": "导出预测结果",
    "Export Range": "导出范围",
    "Export Type": "导出类型",
    "Structured JSONL/NDJSON": "结构化 JSONL/NDJSON",
    "JPG Images (in folder)": "JPG 图像（放在文件夹中）",
    "1D Curve ASCII files": "一维曲线 ASCII 文件",
    "Multi-File Distribution Heatmap": "多文件分布热图",
    "Distribution:": "分布：",
    "Multi-File Parameter Trend": "多文件参数趋势",
    "Predict Log": "预测日志",
    "Open in Separate Window": "在单独的窗口中打开",
    # message boxes
    "Missing Dependency": "缺少依赖",
    "Install the fabio package to read CBF files (pip install fabio)": "读取 CBF 文件需要安装 fabio 包（pip install fabio）",
    "Image Load Failed": "图像载入失败",
    "Folder Not Found": "找不到文件夹",
    "Enter or choose a detector data folder.": "请输入或选择探测器数据文件夹。",
    "File Missing": "缺少文件",
    "module.yaml not found for this module.": "找不到这个模块的 module.yaml。",
    "Could not reload the selected module. Please check module.yaml.": "无法重新载入所选模块。请检查 module.yaml。",
    "No Module": "未选择模块",
    "Please select a module first.": "请先选择一个模块。",
    "Open Failed": "打开失败",
    "No results to export.": "没有可导出的结果。",
    "No completed results to export.": "没有已完成的结果可导出。",
    "Export Error": "导出出错",
    "Multi-File Results": "多文件结果",
    "The batch results panel is not available yet.": "批量结果面板还不可用。",
    "Framework": "框架",
    "The selected model requires a compatible installed framework.": "所选模型需要已安装的兼容框架。",
    "Please import a model before running prediction.": "运行预测之前请先导入模型。",
    "Prediction Error": "预测出错",
    "Invalid Parameters": "参数无效",
    "Please select a valid input file": "请选择有效的输入文件",
    "Please select a valid folder": "请选择有效的文件夹",
    "Clear Results": "清除结果",
    "Are you sure you want to clear all results?": "确定要清除全部结果吗？",
    "No results selected for export.": "没有选中要导出的结果。",
}


@pytest.fixture
def chinese(monkeypatch):
    for english, text in ZH_ENTRIES.items():
        if english not in i18n.ZH:
            monkeypatch.setitem(i18n.ZH, english, text)
            monkeypatch.setitem(i18n._TO_ENGLISH, text, english)
    yield
    apply_language(DEFAULT_LANGUAGE)


@pytest.fixture(scope="module")
def window():
    """One main window for the module, shown on the 2D Prediction page."""
    shown = _window()
    shown.show()
    shown.runtime.navigate("predict")
    _wait(lambda: False, 0.3)
    yield shown
    apply_language(DEFAULT_LANGUAGE)
    shown.close()


def _zh(english: str) -> str:
    chinese_text = i18n.ZH[english]
    assert chinese_text != english
    return chinese_text


def test_the_table_has_these_texts_once_they_are_merged():
    """After the integrator's merge every text of this round is in the Chinese table (none of them before it):
    the ``chinese`` fixture above must not hide an entry the merge left out."""
    missing = sorted(english for english in ZH_ENTRIES if english not in i18n.ZH)
    if len(missing) == len(ZH_ENTRIES):
        pytest.skip("the entries of this round are not merged into i18n/zh_labs.py yet")
    assert missing == []


def test_a_page_that_predicted_nothing_does_not_say_batch_results_are_ready(window):
    prediction = window.runtime.prediction
    assert prediction._multifile_results_widget is not None  # the (empty) result table was built
    assert prediction._multifile_results_widget.table_model.rowCount() == 0
    assert BATCH_READY not in window.predictStatusTextBrowser.toPlainText()
    assert BATCH_READY not in prediction.page_status.text
    window.runtime.navigate("trainset")
    window.runtime.navigate("predict")  # showing the page says its own last line again
    assert BATCH_READY not in window.statusbar.currentMessage()
    # Building the table again (Open Batch Results before any run) says nothing either.
    prediction._setup_multifile_ui()
    assert BATCH_READY not in window.predictStatusTextBrowser.toPlainText()


def test_the_activity_log_is_written_in_the_interface_language(window):
    prediction = window.runtime.prediction
    log = window.predictStatusTextBrowser
    said = []
    prediction.status_updated.connect(said.append)
    try:
        prediction._append_status_message(WATCHING)
        assert log.toPlainText().splitlines()[-1] == f"[INFO] {WATCHING}"

        apply_language("zh")
        prediction._open_status_text_window()  # the separate log window gets the same lines
        viewer = prediction._status_text_window_browser
        prediction._append_status_message(WATCHING, level="WARN")
        assert log.toPlainText().splitlines()[-1] == f"[WARN] {_zh(WATCHING)}"
        assert viewer.toPlainText().splitlines()[-1] == f"[WARN] {_zh(WATCHING)}"
        # The page still says the English sentence: the shell translates it for the status bar.
        assert said[-1] == WATCHING
        assert window.statusbar.currentMessage() == _zh(WATCHING)
        # A message composed with trf() is already in the interface language and is written as it is.
        composed = trf("Module selected: {name}", name="Gold on Silicon")
        assert composed == _zh("Module selected: {name}").format(name="Gold on Silicon")
        prediction._append_status_message(composed)
        assert log.toPlainText().splitlines()[-1] == f"[INFO] {composed}"
        # Text that is no table key (a file name, an error of a library) is not touched.
        prediction._append_status_message("frame_00007.cbf: unreadable header", level="ERROR")
        assert log.toPlainText().splitlines()[-1] == "[ERROR] frame_00007.cbf: unreadable header"
    finally:
        prediction.status_updated.disconnect(said.append)
        if prediction._status_text_window is not None:
            prediction._status_text_window.close()
        apply_language("en")


def test_the_model_folder_dialog_is_titled_in_the_interface_language(window, chinese, monkeypatch):
    captions = []
    monkeypatch.setattr(
        module_catalog.QFileDialog, "getExistingDirectory",
        staticmethod(lambda parent, caption, *args, **kwargs: captions.append(caption) or ""),
    )
    prediction = window.runtime.prediction
    assert prediction._select_model_folder("") == ""
    apply_language("zh")
    assert prediction._select_model_folder("") == ""
    assert captions == [FOLDER_CAPTION, _zh(FOLDER_CAPTION)]


def test_the_static_texts_of_the_page_are_table_keys_and_come_back_in_english(window, chinese):
    page = window.gisaxsPredictPage
    results = window.runtime.prediction._multifile_results_widget
    texts = {
        "Edit Config": window.gisaxsPredictEditButton.text,
        "Reload Config": window.gisaxsPredictReloadConfigButton.text,
        "Framework:": window.gisaxsPredictFrameworkLabel.text,
        "Browse the shared model library only when a local model is not already available.":
            page.findChild(QLabel, "predictionAdvancedDescription").text,
        "Browse the shared DESY model repository, download a model, then use Model Import.":
            page.findChild(QLabel, "predictModelLibraryDescription").text,
        "Zoom -": window.gisaxsImageZoomOutButton.text,
        "All Status": lambda: results.status_filter.itemText(0),
        "File Name": lambda: results.sort_combo.itemText(0),
        "Stack Files": lambda: results.sort_combo.itemText(1),
        "Duration": lambda: results.sort_combo.itemText(3),
    }
    try:
        assert {english: shown() for english, shown in texts.items()} == {english: english for english in texts}
        apply_language("zh", [window])
        assert {english: shown() for english, shown in texts.items()} == {english: _zh(english) for english in texts}
        # The sort button writes its text on every click: in the interface language too.
        seen = set()
        for _click in range(2):
            results.onSortOrderToggled()
            seen.add(results.sort_order_btn.text())
        assert seen == {_zh("Asc"), _zh("Desc")}
        # The choices are found by their data, not their (translated) text.
        assert results.sort_combo.currentData() == 0 and results.status_filter.currentData() is None
        apply_language("en", [window])
        assert {english: shown() for english, shown in texts.items()} == {english: english for english in texts}
        assert results.sort_order_btn.text() in ("Asc", "Desc")
    finally:
        apply_language("en", [window])


def test_the_dialogs_and_message_boxes_are_translated_when_they_are_shown(window, chinese):
    results = window.runtime.prediction._multifile_results_widget
    apply_language("zh")
    opened = []
    try:
        export = ExportDialog(3, 1, 2, results)
        export.setModal(False)
        opened.append(export)
        export.show()
        assert export.windowTitle() == _zh("Export Prediction Results")
        assert (export.range_box.title(), export.type_box.title()) == (_zh("Export Range"), _zh("Export Type"))
        assert [box.text() for box in (export.jsonl_check, export.jpg_check, export.ascii_check)] == [
            _zh("Structured JSONL/NDJSON"), _zh("JPG Images (in folder)"), _zh("1D Curve ASCII files"),
        ]
        results.showDistributionHeatmap()
        opened.append(results._heatmap_window)
        assert results._heatmap_window.windowTitle() == _zh("Multi-File Distribution Heatmap")
        assert results._heatmap_window.component_label.text() == _zh("Distribution:")
        results.showParameterTrend()
        opened.append(results._parameter_trend_window)
        assert results._parameter_trend_window.windowTitle() == _zh("Multi-File Parameter Trend")
        window.runtime.prediction._open_status_text_window()
        opened.append(window.runtime.prediction._status_text_window)
        assert opened[-1].windowTitle() == _zh("Predict Log")
        # A box made from two literal texts (as QMessageBox.warning(parent, title, text) makes it) gets the
        # walk every window gets when it is shown. (Showing a QMessageBox crashes the offscreen platform.)
        box = QMessageBox(QMessageBox.Warning, "Invalid Parameters", "Please select a valid input file",
                          QMessageBox.Ok, window)
        opened.append(box)
        i18n.apply_to(box, "zh")
        assert (box.windowTitle(), box.text()) == (_zh("Invalid Parameters"), _zh("Please select a valid input file"))
    finally:
        for dialog in opened:
            if dialog is not None:
                dialog.close()
        apply_language("en")


def _literals(node):
    """The string literals an argument can turn into (``a or "text"``, ``"x" if c else "y"``)."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        yield node.value
    elif isinstance(node, ast.BoolOp):
        for value in node.values:
            yield from _literals(value)
    elif isinstance(node, ast.IfExp):
        yield from _literals(node.body)
        yield from _literals(node.orelse)


def test_every_literal_text_of_a_prediction_message_box_or_menu_has_its_chinese(chinese):
    """A box or a context menu is translated when it is shown, text by text: a literal title or sentence
    given to it must be a key of the table (a composed one goes through ``trf``)."""
    missing = []
    for path in sorted(PREDICTION.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.Call):
                continue
            name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
            owner = getattr(getattr(node.func, "value", None), "id", "")
            if owner == "QMessageBox" and name in ("information", "warning", "critical", "question"):
                arguments = node.args[1:3]  # the title and the text
            elif name == "QAction":
                arguments = node.args[:1]
            else:
                continue
            for argument in arguments:
                for text in _literals(argument):
                    if text.strip() and text not in i18n.ZH:
                        missing.append(f"{path.relative_to(ROOT)}:{node.lineno} {text!r}")
    assert missing == []


def test_the_two_configuration_buttons_do_not_touch(window):
    window.predictionTechnicalModelDisclosure.set_expanded(True)
    _wait(lambda: False, 0.2)
    reload_button, edit_button = window.gisaxsPredictReloadConfigButton, window.gisaxsPredictEditButton
    assert reload_button.parentWidget() is edit_button.parentWidget()
    assert reload_button.geometry().top() == edit_button.geometry().top()
    assert edit_button.geometry().left() - reload_button.geometry().right() - 1 >= 6
