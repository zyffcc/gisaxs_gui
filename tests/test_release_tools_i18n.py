"""The tool windows in the interface language (release audit R30, R31, R43, R73).

Geometry Calibration, the Format Converter and the XRR Series Extractor compose their run-time texts with
``tr``/``trf`` and exact English templates; what the calibration engine and the readers say in English (stage
names, confidence, warnings, reasons) is translated where it is shown. ``ZH`` below is the Chinese of every
text of this round (the integrator merges it into ``i18n/zh_tools.py``); the other ``test_release_tools_*``
files use it through the ``chinese`` fixture. A guard on the source keeps bare English and f-strings out of
the calls that put a text on screen.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from src.gimap.app.presentation import i18n
from src.gimap.app.presentation.i18n import apply_language

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ("calibration", "format_converter", "xrr")

# -- Geometry Calibration ----------------------------------------------------------------------------
CALIBRATION_ZH = {
    "Preparing mask": "正在准备掩膜",
    "Estimating center": "正在估计光束中心",
    "Detecting rings": "正在检测衍射环",
    "Matching standards": "正在匹配标样",
    "Optimizing geometry": "正在优化几何",
    "Ranking candidates": "正在给候选解排序",
    "Recording calibration": "正在记录标定",
    "Calibration complete": "标定完成",
    "Reading image...": "正在读取图像…",
    "Starting calibration...": "正在开始标定…",
    "Cancelling after the current numerical step...": "当前这一步数值计算结束后取消…",
    "Calibration complete. Review the selected candidate, then Apply.": "标定完成。请检查选中的候选解，然后点击“应用”。",
    "Calibration cancelled.": "标定已取消。",
    "Calibration failed. Adjust the inputs and try again.": "标定失败。请调整输入后重试。",
    "Failed to load image.": "图像读取失败。",
    "Loaded {name} — {width} × {height} pixels{notes}. Click Auto Calibration.":
        "已载入 {name} — {width} × {height} 像素{notes}。请点击“自动标定”。",
    "energy from companion NXS": "能量来自同组的 NXS 文件",
    "multiple standard names found; comparing patterns automatically": "找到多个标样名称；将自动比较衍射图样",
    "{x} × {y} µm (metadata)": "{x} × {y} µm（来自元数据）",
    "{image}  ·  {standard}  ·  {distance} mm  ·  {rings} matched rings  ·  {confidence} confidence":
        "{image}  ·  {standard}  ·  {distance} mm  ·  匹配 {rings} 个环  ·  置信度：{confidence}",
    "High": "高",
    "Medium": "中",
    "Low": "低",
    "Detected": "检测到的环",
    "Matched": "匹配的环",
    "Other theoretical": "其他理论环",
    "Only one ring is matched; center and distance remain ambiguous.": "只匹配到一个环；中心和距离仍不确定。",
    "Matched rings have very limited azimuthal coverage; verify the overlay before applying.":
        "匹配的环在方位角上覆盖很少；应用之前请核对叠加图。",
    "Matched rings are partial GISAXS arcs; verify the overlay before applying.": "匹配的环只是 GISAXS 的部分弧段；应用之前请核对叠加图。",
    "A strong inner radial peak is unexplained by this harmonic assignment.": "内侧有一个较强的径向峰，这种级次指认无法解释它。",
    "Too few experimental ring points were available for joint refinement.": "实验环上可用的点太少，无法做联合精修。",
    "Only one calibration ring was detected. The distance is ambiguous, so candidate solutions are shown.":
        "只检测到一个标定环。距离不确定，因此列出多个候选解。",
    "Geometry was manually adjusted after automatic calibration.": "自动标定之后，几何经过了手动调整。",
    "Pixel size could not be determined. Select the detector model or enter the pixel size under Advanced configuration.":
        "无法确定像素尺寸。请选择探测器型号，或在“高级配置”中填写像素尺寸。",
    "A stable beam-center proposal could not be found.": "找不到稳定的光束中心估计。",
    "No calibration-standard match was found. Check the energy, pixel size, distance range, and visible rings.":
        "没有找到与标样匹配的解。请检查能量、像素尺寸、距离范围，以及图像上是否能看到环。",
    "Calibration candidates were rejected as physically implausible.": "候选解在物理上都不合理，已全部舍去。",
    "The detector image contains no usable pixels.": "探测器图像中没有可用的像素。",
    "No valid pixels are available for a radial profile.": "没有可用于径向分布的有效像素。",
    "Custom distance minimum must be smaller than the maximum.": "自定义距离的下限必须小于上限。",
    "This file is not a GIMaP geometry calibration.": "这个文件不是 GIMaP 的几何标定文件。",
    "Unsupported geometry calibration file version.": "不支持这个版本的几何标定文件。",
    "The selected q value is not physically reachable at this energy.": "在这个能量下，所选的 q 值在物理上达不到。",
    "A positive q value is required.": "q 值必须为正。",
    "No calibration image is loaded.": "还没有载入标定图像。",
    "No calibration result is available.": "还没有标定结果。",
    "{name} was not found: it may have been moved or deleted.": "找不到 {name}：它可能已被移动或删除。",
    "{name} could not be read: the file looks damaged, incomplete or of another kind ({error}).":
        "无法读取 {name}：文件看起来已损坏、不完整，或不是这种格式（{error}）。",
    "Could not read {name}: {error}": "无法读取 {name}：{error}",
    "Calibration Image": "读取标定图像",
    "Calibration Input": "标定输入",
    "Export Calibration": "导出标定",
    "Import Calibration": "导入标定",
    "JSON Files (*.json)": "JSON 文件 (*.json)",
    "Calibration Running": "标定正在运行",
    "Cancel calibration and close?": "取消标定并关闭窗口吗？",
    "Manual Refinement": "手动精修出错",
    "Select NXS Dataset": "选择 NXS 数据集",
    "Detector image dataset:": "探测器图像数据集：",
}

# -- Format Converter ------------------------------------------------------------------------------
CONVERTER_ZH = {
    "Use": "选用",
    "to": "至",
    "First": "首帧",
    "Middle": "中间帧",
    "Last": "末帧",
    "Add Folder": "从文件夹添加",
    "Folder:": "文件夹：",
    "Include:": "包含：",
    "Include subfolders": "同时扫描子文件夹",
    "Disabled by default to avoid accidentally loading very large folder trees.": "默认关闭，以免误读很大的文件夹树。",
    "Subfolders are not scanned unless you explicitly enable the option.": "除非勾选此项，否则不会扫描子文件夹。",
    "Preparing conversion…": "正在准备转换…",
    "Elapsed: 00:00:00": "已用时：00:00:00",
    "Open output folder": "打开输出文件夹",
    "View report": "查看报告",
    "{files} source file(s), {frames} available image(s) / frame(s).": "{files} 个源文件，共 {frames} 张可用图像 / 帧。",
    "{name} is not an NXS, CBF or TIFF file: the converter cannot read it.": "{name} 不是 NXS、CBF 或 TIFF 文件：转换器无法读取它。",
    "Some inputs could not be added": "部分输入无法添加",
    "Add detector images": "添加探测器图像",
    "Select input folder": "选择输入文件夹",
    "Select at least one input format.": "请至少选择一种输入格式。",
    "Please select a valid folder.": "请选择一个有效的文件夹。",
    "No matching files were found in that folder.": "这个文件夹里没有匹配的文件。",
    "Dataset: {path}": "数据集：{path}",
    "Shape: {shape}": "形状：{shape}",
    "Frames: 1–{count}": "帧：1–{count}",
    "Unknown": "未知",
    "Converting": "正在转换",
    "Dataset selection": "数据集选择",
    "Frame selection": "帧选择",
    "Select one or more input rows first.": "请先选中一行或多行输入。",
    "Select at least one frame.": "请至少选择一帧。",
    "The frame range is empty.": "帧范围为空。",
    "Choose output folder": "选择输出文件夹",
    "Invalid naming template": "命名模板无效",
    "CBF encoders may not preserve NaN values or every floating-point representation. A metadata sidecar is recommended.":
        "CBF 编码可能无法保留 NaN 值或某些浮点表示。建议同时写出元数据文件。",
    "Add at least one input file.": "请至少添加一个输入文件。",
    "Select at least one image or frame to convert.": "请至少选择一张要转换的图像或一帧。",
    "Output settings": "输出设置",
    "Choose an output destination.": "请选择输出位置。",
    "Invalid output settings:\n{error}": "输出设置无效：\n{error}",
    "Confirm conversion": "确认转换",
    "Convert": "转换",
    "Input:": "输入：",
    "{count} {kind} file(s)": "{count} 个 {kind} 文件",
    "{images} selected image(s) / frame(s)": "已选 {images} 张图像 / 帧",
    "Output:": "输出：",
    "Destination: {destination}": "输出位置：{destination}",
    "Naming: {naming}": "命名：{naming}",
    "Estimated output:": "预计输出：",
    "{files} file(s), approximately {size}": "{files} 个文件，约 {size}",
    "⚠ Large output: conversion can take considerable time and disk space.": "⚠ 输出量很大：转换可能需要很长时间和很多磁盘空间。",
    "Converting {name}": "正在转换 {name}",
    "Frame {frame} · {done} / {total}": "第 {frame} 帧 · {done} / {total}",
    "{name} · frame {frame} · {done} / {total}": "{name} · 第 {frame} 帧 · {done} / {total}",
    "calculating…": "正在估算…",
    "Elapsed: {elapsed}    Remaining: approximately {remaining}": "已用时：{elapsed}    剩余：约 {remaining}",
    "Conversion paused": "转换已暂停",
    "Conversion resumed": "转换已继续",
    "Cancelling after the current image…": "当前图像完成后取消…",
    "Conversion completed": "转换完成",
    "Conversion cancelled": "转换已取消",
    "Conversion could not be completed": "转换未能完成",
    "{succeeded} succeeded\n{failed} failed": "成功 {succeeded} 个\n失败 {failed} 个",
}

# -- XRR Series Extractor --------------------------------------------------------------------------
XRR_ZH = {
    "Select an NXS module file or a CBF series first.": "请先选择一个 NXS 模块文件或一个 CBF 序列。",
    "Enter the NXS motor dataset path containing theta values.": "请填写存有 theta 值的 NXS 电机数据集路径。",
    "Select an NXS module file or a folder containing CBF files.": "请选择一个 NXS 模块文件，或一个包含 CBF 文件的文件夹。",
    "No detector frames were found for the selected XRR series.": "所选的 XRR 序列中没有找到探测器帧。",
    "There are no extracted XRR points to export.": "没有可导出的 XRR 数据点。",
    "Starting XRR extraction…": "正在开始 XRR 提取…",
    "Stopping extraction…": "正在停止提取…",
    "Extracted frame {done} of {total}": "已提取第 {done} 帧，共 {total} 帧",
    "Stopped after {count} point(s).": "已停止，共提取 {count} 个点。",
    "Extracted {count} XRR point(s).": "已提取 {count} 个 XRR 数据点。",
}

ZH = {**CALIBRATION_ZH, **CONVERTER_ZH, **XRR_ZH}


def switch_to_chinese(monkeypatch) -> None:
    """Chinese with this round's entries in the table, as after the merge (the caller switches back)."""
    for english, text in ZH.items():
        monkeypatch.setitem(i18n.ZH, english, text)
        monkeypatch.setitem(i18n._TO_ENGLISH, text, english)
    apply_language("zh")


@pytest.fixture
def chinese(monkeypatch):
    """Chinese for one test; English again afterwards, whatever the test switched to."""
    switch_to_chinese(monkeypatch)
    yield
    apply_language("en")


# -- the entries themselves ------------------------------------------------------------------------

_FIELD = re.compile(r"\{(\w+)[^}]*\}")


def test_no_text_is_in_two_groups() -> None:
    assert len(ZH) == len(CALIBRATION_ZH) + len(CONVERTER_ZH) + len(XRR_ZH)


@pytest.mark.parametrize("english", sorted(ZH))
def test_a_template_keeps_its_fields_in_chinese(english) -> None:
    assert set(_FIELD.findall(english)) == set(_FIELD.findall(ZH[english])), ZH[english]
    assert english.count("\n") == ZH[english].count("\n")  # the lines of a box stay the lines


def _string_constants() -> set[str]:
    found = set()
    for feature in FEATURES:
        for path in (ROOT / "src/gimap/features" / feature).rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    found.add(node.value)
    return found


def test_every_entry_is_a_text_of_the_code() -> None:
    """A key that no longer matches the English in the code is never used: the text would stay English."""
    constants = _string_constants()
    assert sorted(english for english in ZH if english not in constants) == []


def test_a_sentence_that_names_a_button_names_its_chinese_label(chinese) -> None:
    for english, label in (
        ("Calibration complete. Review the selected candidate, then Apply.", "Apply"),
        ("Loaded {name} — {width} × {height} pixels{notes}. Click Auto Calibration.", "Auto Calibration"),
        ("Pixel size could not be determined. Select the detector model or enter the pixel size under "
         "Advanced configuration.", "Advanced configuration"),
    ):
        assert label in i18n.ZH, label
        assert f"“{i18n.ZH[label]}”" in ZH[english], (english, i18n.ZH[label])


def test_the_table_has_these_texts_once_they_are_merged() -> None:
    """After the integrator's merge every text of this round is in the Chinese table (none before it)."""
    missing = sorted(english for english in ZH if english not in i18n.ZH)
    if len(missing) == len(ZH):
        pytest.skip("the entries of this round are not merged into i18n/zh_tools.py yet")
    assert missing == []


# -- a guard on the source ---------------------------------------------------------------------------

TEXT_ARGUMENTS = {
    "setText": (0,), "setToolTip": (0,), "setPlaceholderText": (0,), "setTitle": (0,), "setWindowTitle": (0,),
    "addButton": (0,), "addItem": (0,), "show_toast": (1,), "set_state": (1,),
    "warning": (1, 2), "information": (1, 2), "question": (1, 2), "critical": (1, 2),
    "getOpenFileName": (1,), "getOpenFileNames": (1,), "getSaveFileName": (1,), "getExistingDirectory": (1,),
    "getItem": (1, 2),
}
"""Calls that put a text on screen, and which arguments are texts."""
TRANSLATED_BY_CALLEE = {"set_state", "setWindowTitle"}
"""``JobStatus.set_state`` translates its message; a window's own title is translated when it is shown."""
NOT_WORDS = {"px", "mm", "µm", "q", "Å⁻¹", "θ"}
"""Units and symbols an f-string may hold: they are never translated."""


def _english(node) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, str) and any(c.isalpha() for c in node.value)


def _words(joined: ast.JoinedStr) -> list[str]:
    literal = " ".join(str(part.value) for part in joined.values if isinstance(part, ast.Constant))
    return [word for word in re.findall(r"[^\W\d_]+[⁻¹]*", literal) if word not in NOT_WORDS]


def test_no_text_of_a_tool_window_reaches_the_screen_untranslated() -> None:
    problems = []
    for feature in FEATURES:
        for path in sorted((ROOT / "src/gimap/features" / feature / "presentation").rglob("*.py")):
            if "views" in path.parts:  # static layout: translated by the walker when the window is shown
                continue
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if not isinstance(node, ast.Call):
                    continue
                name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", None)
                if name in ("tr", "trf") and node.args and isinstance(node.args[0], ast.JoinedStr):
                    problems.append(f"{path.name}:{node.lineno} tr(f\"…\") can never match the table")
                for index in TEXT_ARGUMENTS.get(name, ()):
                    if len(node.args) <= index:
                        continue
                    text = node.args[index]
                    for branch in ([text.body, text.orelse] if isinstance(text, ast.IfExp) else [text]):
                        if isinstance(branch, ast.JoinedStr) and _words(branch):
                            problems.append(f"{path.name}:{node.lineno} {name}: formatted before the translation")
                        elif _english(branch) and name not in TRANSLATED_BY_CALLEE:
                            problems.append(f"{path.name}:{node.lineno} {name}: {branch.value!r} without tr()")
    assert problems == []
