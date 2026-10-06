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
    "Open a curve: the residuals of the model appear here.": "打开曲线后，这里显示模型的残差。",
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
    "The selected frame and its fit appear here.": "所选帧及其拟合会显示在这里。",
    # -- 1D Predict ------------------------------------------------------------------------------------
    "1D Predict · fit curves & batch": "一维预测 · 拟合曲线与批处理",
    "Prediction settings — saved for single curves and future batch / in-situ runs":
        "预测设置——保存后用于单条曲线以及之后的批处理 / 原位运行",
    "Auto / unused": "自动 / 不用",
    "Complete composition": "完整组成",
    "Fit method": "拟合方法",
    "Text-file q unit": "文本文件的 q 单位",
    "q sides": "q 的两侧",
    "Each half separately": "两半分别拟合",
    "General V5: improve fit with four numerical steps": "通用 V5：再用四个数值步骤改进拟合",
    "Calibrate intensity amplitudes": "校准强度幅值",
    "The single-RC specialist can adjust particle, background and resolution amplitudes while keeping the neural "
    "shape parameters fixed. Broader fitting may still run when curve agreement is poor.":
        "单 RC 专用模型可以调整颗粒、背景和分辨率的幅值，同时保持神经网络给出的形状参数不变。曲线吻合较差时，"
        "仍可能运行范围更广的拟合。",
    "General V5 (experimental)": "通用 V5（实验性）",
    "Single RC specialist (experimental)": "单 RC 专用模型（实验性）",
    "Physical fit (numerical)": "物理拟合（数值）",
    "General V5 proposes multiple compositions but remains experimental. The specialist requires Complete "
    "composition = one Random cylinder and eligible native CBF counts. Other inputs, fixed resolution and poor "
    "curve agreement use numerical fallback; that fallback does not make the specialist a general model. Scores "
    "are not probabilities.":
        "通用 V5 会给出多种组成，但仍是实验性的。专用模型要求“完整组成”只有一个随机取向圆柱，并且输入是符合条件的"
        "原始 CBF 计数。其他输入、固定的分辨率或曲线吻合较差时会改用数值回退；这种回退并不会让专用模型变成通用模型。"
        "得分不是概率。",
    "Fix σ res (nm⁻¹)": "固定 σ res（nm⁻¹）",
    "Fix ν res": "固定 ν res",
    "General V5: 0.007–0.013 nm⁻¹. RC specialist / physical fit: 0.001–0.1 nm⁻¹; fixed resolution uses numerical "
    "fallback.": "通用 V5：0.007–0.013 nm⁻¹。RC 专用模型 / 物理拟合：0.001–0.1 nm⁻¹；固定分辨率时改用数值回退。",
    "General V5: 5–10. RC specialist / physical fit: 1–20; fixed resolution uses numerical fallback.":
        "通用 V5：5–10。RC 专用模型 / 物理拟合：1–20；固定分辨率时改用数值回退。",
    "Auto: 0.1% peak": "自动：峰值的 0.1%",
    "Auto: measured max": "自动：测量最大值",
    "Relative σ (if missing)": "相对 σ（文件里没有时）",
    "Absolute σ floor": "σ 绝对下限",
    "Intensity normalizer": "强度归一化值",
    "Discover combinations": "搜索的组合数",
    "Condition best combinations": "细化的最佳组合数",
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
    "Model + amplitude": "模型 + 幅值",
    "Neural model": "神经网络模型",
    "Numerical fallback": "数值回退",
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
    "Current cut — native points, q converted to nm⁻¹ by the fitting workspace.":
        "当前切线——原始数据点，q 由拟合工作区换算为 nm⁻¹。",
    "A fitting / in-situ job is already running. Finish or cancel it first.": "已有拟合 / 原位任务在运行。请先等它完成或取消。",
    "Load a curve or select files first": "请先载入曲线或选择文件",
    "Starting the experimental single-RC specialist… Checking the known composition and input scope.":
        "正在启动实验性的单 RC 专用模型…正在检查已知组成和输入范围。",
    "Starting numerical physical fitting…": "正在开始数值物理拟合…",
    "Loading experimental General V5… First run includes loading and compilation.":
        "正在载入实验性的通用 V5…第一次运行包括载入和编译。",
    "Finished in {seconds} s · {failures} failed files. {quality}": "用时 {seconds} s 完成 · {failures} 个文件失败。{quality}",
    "{count} candidates saved. Review curve shape and residuals; observed-data scores include noise and are not "
    "probabilities.": "已保存 {count} 个候选。请检查曲线形状和残差；基于观测数据的得分包含噪声，不是概率。",
    "Settings saved. Existing in-situ recipes keep their captured settings.": "设置已保存。已有的原位配方保留它们记录时的设置。",
    "Cancelling… completed file results remain saved.": "正在取消…已完成文件的结果仍会保留。",
    "Saved settings v{version} for future frames.": "已为之后的帧保存设置 v{version}。",
}

ALLOWED = {"#", "logRMSE", "RMS/σ", "nm⁻¹", "Å⁻¹"}
"""Texts that stay as they are in Chinese: data headers and units."""
OPTIONS = (
    {},
    dict(q_unit="A^-1", side="negative", method="stable", components=[2], sigma_res=0.05, nu_res=3.5),
    dict(q_unit="nm^-1", side="positive", method="experimental", components=[1, 3, 3], relative_noise=0.25,
         absolute_noise=2.5, normalizer=1e4, numerical=False, amplitude_calibration=False),
    dict(method="model", components=[2, 2], sigma_res=0.00855, nu_res=7.25, search_combinations=20,
         condition_combinations=5),
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
               signed_weighted_rms=1.2, best_source="stable_numerical_fallback", fallback_reason="poor agreement",
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
    options = dict(sigma_res=0.00855, nu_res=7.25, relative_noise=0.1)
    dialog = _dialog(options=options)
    assert dialog.sigma_res.text() == "0.00855" and dialog.nu_res.text() == "7.25"
    assert dialog.relative_noise.text() == "0.1" and dialog.noise_floor.text() == "Auto: 0.1% peak"
    assert dialog.sigma_res.decimals() == 8 and dialog.nu_res.decimals() == 6  # the values keep every decimal
    dialog.fix_sigma.setChecked(True)
    dialog.fix_nu.setChecked(True)
    dialog.sigma_res.lineEdit().setText(dialog.sigma_res.text())  # the text shown reads back as the same value
    dialog.sigma_res.interpretText()
    assert dialog.options()["sigma_res"] == 0.00855 and dialog.options()["nu_res"] == 7.25
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
        texts += [dialog.noise_floor.specialValueText(), dialog.normalizer.specialValueText()]
        left = [text for text in texts if text and text not in ALLOWED and not re.search(r"[一-鿿]", text)]
        assert left == []
        tips = [dialog.table.horizontalHeaderItem(i).toolTip() for i in (2, 3, 4)]
        tips += [dialog.method.toolTip(), dialog.sigma_res.toolTip(), dialog.amplitude_calibration.toolTip()]
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
        assert dialog.status.text().startswith("用时 1.50 s 完成") and "已保存 1 个候选" in dialog.status.text()
        assert dialog.table.item(0, 0).text() == "当前 / +" and dialog.table.item(0, 5).text() == "数值回退"
        assert dialog.table.item(0, 5).toolTip().startswith("阶段：stable_numerical_fallback\n原因：poor agreement")
        assert dialog.parameters.toPlainText().startswith("长度：nm。")
        chinese.apply_language("en")  # the window follows a switch while open
        assert dialog.status.text().startswith("Finished in 1.50 s") and "1 candidates saved" in dialog.status.text()
        assert dialog.table.item(0, 5).text() == "Numerical fallback" and dialog.table.currentRow() == 0
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
    assert dialog.run_button.text() == "Fit 2 files" and dialog.input_label.text() == "2 file(s): a.dat, b.dat"
    dialog._use_current()
    assert dialog.run_button.text() == "Fit curve" and dialog.files == []
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
