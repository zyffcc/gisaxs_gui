"""The 1D Predict window in Chinese (fitting-5): the options keep their values in any language, the window
has no English left once the new table entries are in, and its run-time texts follow a switch of the language.

``ZH_ENTRIES`` are the new English → Chinese pairs of the Fitting wave (merged into ``i18n/zh_fitting.py`` by
the integrator); the tests put them into the table themselves until then.
"""

from __future__ import annotations

import re
from types import SimpleNamespace

import pytest
from PyQt5.QtWidgets import QAbstractButton, QApplication, QComboBox, QGroupBox, QLabel

from src.gimap.features.fitting.application.workflow_v5 import validate_options

ZH_ENTRIES = {
    # -- Single analysis --------------------------------------------------------------------------------
    "Undo the last change of the model, the fitting range or the left-out points, or the last fit (Ctrl+Z)":
        "撤销模型、拟合范围或被排除点的上一次修改，或上一次拟合（Ctrl+Z）",
    "Choose a method; the button below runs it (Ctrl+Return).": "选择一种方法；用下面的按钮运行（Ctrl+Return）。",
    "One curve at a time here: {name} is open. Fit ▸ Advanced ▸ Fit Many Curves… takes several.":
        "这里一次只打开一条曲线：已打开 {name}。拟合 ▸ 高级 ▸ 拟合多条曲线… 可以处理多条。",
    "A 1D curve: columns q, I and optionally σ (.dat, .txt). Analyze ▸ Send to Fitting opens its cut here.":
        "一维曲线：q、I 两列，可选 σ（.dat、.txt）。在分析里点“发送到拟合”会把切线打开到这里。",
    # -- why a file is not a curve (the curve reader) ---------------------------------------------------
    "No data rows found (at least two numeric columns are needed).": "未找到数据行（至少需要两列数字）。",
    "No (q, I[, σ]) rows could be read.": "没有读取到任何 (q, I[, σ]) 数据行。",
    "Too few valid points (fewer than 2).": "有效数据点太少（少于 2 个）。",
    # -- In-situ series ---------------------------------------------------------------------------------
    "No curves listed — choose a folder, or Send Series to Fitting in Analyze":
        "还没有列出曲线——选择一个文件夹，或在分析中点“把序列发送到拟合”",
    # -- 1D Predict ------------------------------------------------------------------------------------
    "1D Predict · fit curves & batch": "一维预测 · 拟合曲线与批处理",
    "Prediction settings — saved for single curves and future batch / in-situ runs":
        "预测设置——保存后用于单条曲线以及之后的批处理 / 原位运行",
    "Auto / unused": "自动 / 不用",
    "Complete composition": "完整组成",
    "q unit of files without one in their header": "文件头未注明时的 q 单位",
    "q sides": "q 的两侧",
    "Each half separately": "两半分别拟合",
    "Relative σ added to counting σ": "加到计数 σ 上的相对 σ",
    "Absolute σ added to counting σ": "加到计数 σ 上的绝对 σ",
    "In-situ · 1D prediction parameters": "原位 · 一维预测参数",
    "In-situ prediction parameters": "原位预测参数",
    "Save creates a new settings snapshot for future frames. Completed frames are unchanged.":
        "保存会为之后的帧建立新的设置快照；已完成的帧不变。",
    "Edit known components / resolution, or leave them automatic.": "编辑已知的组分 / 分辨率，或保持自动。",
    "S: sphere; RC: random cylinder; VC: vertical cylinder. Repeated types are distinct components.":
        "S：球；RC：随机取向圆柱；VC：竖直圆柱。重复的类型是不同的组分。",
    "Natural-log RMSE on original positive-intensity observations only.": "只在原始的正强度观测点上计算的自然对数 RMSE。",
    "RMS of (forward − observed)/sigma, including negative observations; not a calibrated probability.":
        "(正演 − 观测)/sigma 的均方根，包括负的观测值；不是校准过的概率。",
    "Add curves (or Use current cut), then Fit curve.\nSelect a candidate to see its fit here.":
        "添加曲线（或使用当前切线），然后拟合曲线。\n选一个候选，在这里查看它的拟合。",
    "Current": "当前",
    "Stage: {stage}": "阶段：{stage}",
    "Reason: {reason}": "原因：{reason}",
    "Natural-log RMSE over the measured curve, including measurement noise. Review peak positions, overall shape "
    "and residuals; no mandatory cutoff is applied.":
        "在整条测量曲线上计算的自然对数 RMSE，包含测量噪声。请检查峰位、整体形状和残差；不设强制阈值。",
    "Lengths: nm. sigma_R/h/D: relative standard deviations.\nMixture weights are not posterior probabilities.\n"
    "Resolution sigma: nm^-1; nu: dimensionless.":
        "长度：nm。sigma_R/h/D：相对标准差。\n混合权重不是后验概率。\n分辨率 sigma：nm^-1；nu：无量纲。",
    "Select one or more 1D curves": "选择一条或多条一维曲线",
    "{count} file(s): {names}": "{count} 个文件：{names}",
    "Fit {count} files": "拟合 {count} 个文件",
    "Input: the current cut of Fitting (q in nm⁻¹); each side fitted separately":
        "输入：拟合页的当前切线（q 为 nm⁻¹）；两侧分别拟合",
    "Ready.": "就绪。",
    "A fitting / in-situ job is already running. Finish or cancel it first.": "已有拟合 / 原位任务在运行。请先等它完成或取消。",
    "Load a curve or select files first": "请先载入曲线或选择文件",
    "Finished in {seconds} s · {failures} failed files · {count} candidates saved.":
        "用时 {seconds} s 完成 · {failures} 个文件失败 · 保存了 {count} 个候选。",
    "{count} candidates saved. Review curve shape and residuals; observed-data scores include noise and are not "
    "probabilities.": "已保存 {count} 个候选。请检查曲线形状和残差；基于观测数据的得分包含噪声，不是概率。",
    "Settings saved. Existing in-situ recipes keep their captured settings.": "设置已保存。已有的原位配方保留它们记录时的设置。",
    "Cancelling… completed file results remain saved.": "正在取消…已完成文件的结果仍会保留。",
    "Saved settings v{version} for future frames.": "已为之后的帧保存设置 v{version}。",
    "Check every candidate's fit; refine it in Fitting before reporting it.":
        '请检查每个候选的拟合；报告之前先在拟合页精修。',
    'Starting Predict parameters (network + posterior)…':
        '正在开始预测参数（网络 + 后验）…',
}

ALLOWED = {"#", "logRMSE", "RMS/σ", "nm⁻¹", "Å⁻¹"}
"""Texts that stay as they are in Chinese: data headers and units."""
OPTIONS = (
    {},
    dict(q_unit="A^-1", side="negative", components=[2], cut_qz=0.25),
    dict(q_unit="nm^-1", side="positive", components=[1, 3, 3], relative_noise=0.25, absolute_noise=2.5),
    dict(method="posterior", components=[2, 2]),
    dict(method="experimental", components=[1], sigma_res=0.05, nu_res=3.5),  # a retired method opens as Predict
)


@pytest.fixture
def chinese(monkeypatch):
    """The Chinese table with this wave's entries in it (as the integrator merges them)."""
    from src.gimap.app.presentation import i18n

    for english, text in ZH_ENTRIES.items():
        text = text.replace("▸", "›")
        monkeypatch.setitem(i18n.ZH, english, text)
        monkeypatch.setitem(i18n._TO_ENGLISH, text, english)
    return i18n


def _dialog(**kwargs):
    from src.gimap.features.fitting.presentation.workflow_v5_dialog import WorkflowV5Dialog

    QApplication.instance() or QApplication([])
    return WorkflowV5Dialog(**kwargs)


def _close(dialog) -> None:
    from PyQt5.QtCore import QCoreApplication, QEvent

    dialog.close()
    dialog.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)


def _candidate(**extra) -> dict:
    row = dict(file="Current curve", side="positive", rank=1, combination="random_cylinder", best_log_rmse=.05,
               signed_weighted_rms=1.2, best_source="posterior", warnings=["few effective samples"],
               native_q=[1, 2, 3], observed=[3, 2, 1], sigma=[.1, .1, .1], display_q=[1, 2, 3], display_fit=[3, 2, 1])
    row.update(extra)
    return row


@pytest.mark.parametrize("options", OPTIONS)
def test_the_options_keep_their_values(options) -> None:
    dialog = _dialog(options=options)
    assert dialog.options() == validate_options(options)  # what was saved comes back unchanged
    assert dialog.unit.currentData() == validate_options(options)["q_unit"]
    _close(dialog)


def test_the_spin_boxes_show_their_values_without_trailing_zeros() -> None:
    options = dict(cut_qz=0.00855, relative_noise=0.1, absolute_noise=7.25)
    dialog = _dialog(options=options)
    assert dialog.cut_qz.text() == "0.00855" and dialog.use_qz.isChecked()
    assert dialog.relative_noise.text() == "0.1" and dialog.noise_floor.text() == "7.25"
    assert dialog.cut_qz.decimals() == 6 and dialog.noise_floor.decimals() == 6  # the values keep every decimal
    dialog.cut_qz.lineEdit().setText(dialog.cut_qz.text())  # the text shown reads back as the same value
    dialog.cut_qz.interpretText()
    assert dialog.options()["cut_qz"] == 0.00855 and dialog.options()["absolute_noise"] == 7.25
    dialog.noise_floor.setValue(0)
    assert dialog.noise_floor.text() == "None"  # nothing added to the counting σ
    _close(dialog)


@pytest.mark.parametrize("options", OPTIONS[1:])
def test_in_chinese_the_halves_and_units_are_read_from_their_values(options, chinese) -> None:
    dialog = _dialog(options=options)
    chinese.apply_language("zh")
    chinese.apply_to(dialog, "zh")
    try:
        assert {dialog.unit.itemText(i) for i in range(dialog.unit.count())} == {"nm⁻¹", "Å⁻¹"}  # units as they are
        assert dialog.side.itemText(1) == "q > 0 一半" and dialog.side.itemText(0) == "两半分别拟合"
        assert dialog.options() == validate_options(options)
        saved = []
        dialog.settings_changed.connect(saved.append)
        dialog.save_settings()
        assert saved == [validate_options(options)] and dialog.status.text().startswith("设置已保存")
    finally:
        chinese.apply_language("en")
        chinese.apply_to(dialog, "en")
        _close(dialog)


def test_no_english_is_left_in_the_window_in_chinese(chinese) -> None:
    dialog = _dialog()
    dialog.settings_panel.show()
    chinese.apply_language("zh")
    try:
        chinese.apply_to(dialog, "zh")
        texts = [dialog.windowTitle()]
        for widget in dialog.findChildren((QLabel, QAbstractButton)):
            texts.append(widget.text())
        texts += [box.title() for box in dialog.findChildren(QGroupBox)]
        for combo in dialog.findChildren(QComboBox):
            texts += [combo.itemText(i) for i in range(combo.count())]
        texts += [dialog.table.horizontalHeaderItem(i).text() for i in range(dialog.table.columnCount())]
        texts += [dialog.noise_floor.specialValueText()]
        left = [text for text in texts if text and text not in ALLOWED and not re.search(r"[一-鿿]", text)]
        assert left == []
        tips = [dialog.table.horizontalHeaderItem(i).toolTip() for i in (2, 3, 4)]
        tips += [dialog.component_boxes[0].toolTip(), dialog.use_qz.toolTip(), dialog.cut_qz.toolTip()]
        assert all(re.search(r"[一-鿿]", tip) for tip in tips)
        assert re.search(r"[一-鿿]", dialog.parameters.placeholderText())
        assert re.search(r"[一-鿿]", dialog.log.placeholderText())
        assert dialog.figure.texts[0].get_text().startswith("添加曲线")
        from src.gimap.features.fitting.presentation.curve_rendering import CJK_FONTS, text_font_family

        if text_font_family()[-1] in CJK_FONTS:  # drawn in a font with Chinese glyphs, not as empty boxes
            assert dialog.figure.texts[0].get_fontfamily()[-1] in CJK_FONTS
    finally:
        chinese.apply_language("en")
        _close(dialog)


def test_run_time_texts_are_in_the_interface_language_and_follow_a_switch(chinese, tmp_path) -> None:
    dialog = _dialog()
    chinese.apply_language("zh")
    try:
        chinese.apply_to(dialog, "zh")
        result = SimpleNamespace(succeeded=True, value=dict(candidates=[_candidate()], output_dir=str(tmp_path),
                                                            records=[dict(status="complete")],
                                                            summary=dict(runtime_seconds=1.5)))
        dialog._completed(result)
        assert dialog.status.text() == "用时 1.50 s 完成 · 0 个文件失败 · 保存了 1 个候选。"
        assert dialog.table.item(0, 0).text() == "当前 / +" and dialog.table.item(0, 5).text() == "网络 + 后验"
        assert dialog.table.item(0, 5).toolTip() == "阶段：posterior\nfew effective samples"
        assert dialog.parameters.toPlainText().startswith("长度：nm。")
        chinese.apply_language("en")  # the window follows a switch while open
        assert dialog.status.text() == "Finished in 1.50 s · 0 failed files · 1 candidates saved."
        assert dialog.table.item(0, 5).text() == "Network + posterior" and dialog.table.currentRow() == 0
        assert dialog.table.item(0, 5).toolTip() == "Stage: posterior\nfew effective samples"
        assert dialog.table.horizontalHeaderItem(0).text() == "Curve / side"
        assert dialog.table.horizontalHeaderItem(3).toolTip().startswith("Natural-log RMSE")
        assert dialog.parameters.toPlainText().startswith("Lengths: nm.")
    finally:
        chinese.apply_language("en")
        _close(dialog)


def test_add_curves_opens_where_the_last_curves_were(tmp_path, monkeypatch) -> None:
    from PyQt5.QtWidgets import QFileDialog

    starts = []
    picked = [str(tmp_path / "a.dat"), str(tmp_path / "b.dat")]

    def choose(_parent, _title, start, _filter):
        starts.append(start)
        return picked, ""

    monkeypatch.setattr(QFileDialog, "getOpenFileNames", staticmethod(choose))
    dialog = _dialog()
    dialog._choose_files()
    dialog._choose_files()
    assert starts == ["", str(tmp_path)]
    assert dialog.run_button.text() == "Fit 2 files"
    assert dialog.input_label.text().splitlines()[0] == "2 file(s): a.dat, b.dat"  # then the unit of q of each
    dialog._use_current()
    assert dialog.run_button.text() == "Fit curve" and dialog.files == []
    assert dialog.input_label.text() == "Input: the current cut of Fitting (q in nm⁻¹); each side fitted separately"
    _close(dialog)


def test_the_window_says_once_that_it_is_experimental() -> None:
    """One warning line under the title (the details in its tooltip); the status line only says where things are."""
    dialog = _dialog()
    hint = dialog.findChild(QLabel, "workflowV5Experimental")
    assert hint.text() == "Check every candidate's fit; refine it in Fitting before reporting it."
    assert hint.property("gimapRole") == "warning" and "probability" in hint.toolTip()
    assert dialog.status.text() == "Ready."
    assert dialog.input_label.text() == "Input: the current cut of Fitting (q in nm⁻¹); each side fitted separately"
    _close(dialog)
    dialog = _dialog(settings_only=True)  # the in-situ settings window: a plain note there, not the warning
    hint = dialog.findChild(QLabel, "workflowV5Experimental")
    assert hint.text().startswith("Save creates a new settings snapshot") and hint.property("gimapRole") is None
    _close(dialog)


def test_the_in_situ_settings_window_says_what_was_saved(chinese) -> None:
    from src.gimap.app.presentation.i18n import trf

    dialog = _dialog(settings_only=True)
    chinese.apply_to(dialog, "zh")
    try:
        assert dialog.windowTitle() == "原位 · 一维预测参数" and dialog.status.text() == "编辑已知的组分 / 分辨率，或保持自动。"
        chinese.apply_language("zh")
        dialog.show_status(lambda: trf("Saved settings v{version} for future frames.", version=3))
        assert dialog.status.text() == "已为之后的帧保存设置 v3。"
        chinese.apply_language("en")
        assert dialog.status.text() == "Saved settings v3 for future frames."
    finally:
        chinese.apply_language("en")
        chinese.apply_to(dialog, "en")
        _close(dialog)
