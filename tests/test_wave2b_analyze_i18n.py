"""Analyze's run-time texts in the interface language (wave 2b: analyze-12, cross-3 f).

Every sentence Analyze composes goes through ``tr``/``trf`` with an exact English template, so the Chinese
table can hold it; ``refresh_language`` composes them again after a switch. The messages and curve titles the
application and the domain make in English are shown translated (``presentation/texts.py``); numbers, units
and names are never translated. A frame whose file names no detector says nothing of it (no “Detector:
Detector”). A source guard keeps f-strings and bare English out of the text setters.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from src.gimap.app.presentation import i18n
from src.gimap.app.presentation.i18n import apply_language, trf
from tests.test_analyze_workspace import GALAXI, _context, _done, _page, _public_page, requires_public

ROOT = Path(__file__).resolve().parents[1]
PRESENTATION = ROOT / "src" / "gimap" / "features" / "analyze" / "presentation"

ZH = {  # the Chinese of this area's new texts (the integrator merges them into the zh tables)
    "Detector: {name}": "探测器：{name}",
    "Frame: {rows} × {columns} pixels": "帧：{rows} × {columns} 像素",
    "Frame: {rows} × {columns} pixels, {size} µm pixels": "帧：{rows} × {columns} 像素，像素尺寸 {size} µm",
    "Frames in the file: {n}": "文件中的帧数：{n}",
    "Frames in the file: {n} (showing {i}, sum of {total} frames)": "文件中的帧数：{n}（显示从第 {i} 帧起 {total} 帧之和）",
    "Files listed: {n}": "列表中的文件：{n}",
    "frame {i} of {n}, sum of {total} frames": "第 {i} 帧，共 {n} 帧，{total} 帧求和",
    "{name}: {detector}, {shape}.": "{name}：{detector}，{shape}。",
    "{name}: {shape}.": "{name}：{shape}。",
    "Geometry from “{name}”: {detail}.": "几何来自“{name}”：{detail}。",
    "No geometry yet": "还没有几何",
    "Needs a geometry": "需要几何",
    "this detector": "此探测器",
    "No geometry for {detector} ({shape}) yet. Calibrate from an image of a standard, or enter the values once: "
    "they are saved as an instrument profile and used for every such frame.":
        "{detector}（{shape}）还没有几何。可以用标样图像标定，或输入一次数值：它们会保存为仪器配置，用于每一个这样的帧。",
    "No curves yet: the frame needs a geometry (step 2)": "还没有曲线：这一帧需要几何（第 2 步）",
    "No instrument profile for {detector} ({shape}). Calibrate once or enter the geometry to get q curves.":
        "{detector}（{shape}）没有仪器配置。标定一次或输入几何，即可得到 q 曲线。",
    "Horizontal cut I(qy)": "水平切线 I(qy)",
    "Box I(qz)": "框 I(qz)",
    "{title}: no valid pixels in the cut band.": "{title}：切线带中没有有效像素。",
    "Instrument profile “{name}” is not here: the geometry is matched automatically.":
        "这里没有仪器配置“{name}”：几何将自动匹配。",
    "The beam centre column lies outside the frame.": "光束中心所在的列在帧之外。",
    "Rectangle: x {x}, y {y}": "矩形：x {x}，y {y}",
    "Polygon ({n} points): x {x}, y {y}": "多边形（{n} 个点）：x {x}，y {y}",
    "{frame} · no geometry": "{frame} · 没有几何",
    "profile “{name}” (matched)": "配置 “{name}”（自动匹配）",
    "Open the curve in Fitting (q in Å⁻¹). GISAXS: the horizontal cut, {half}; GIWAXS: I(q). "
    "The arrow chooses the half.": "在拟合中打开曲线（q 以 Å⁻¹ 为单位）。GISAXS：水平切线，{half}；GIWAXS：I(q)。箭头选择用哪一半。",
    "{name}: {curves} curves in {seconds} s": "{name}：{curves} 条曲线，用时 {seconds} s",
    "A difference near one {axis} in a few points is the detector: mask it in the Mask step.":
        "只在某个 {axis} 附近几个点上的差别来自探测器：请在掩膜步骤里把它遮掉。",
}


@pytest.fixture
def chinese(monkeypatch):
    """Chinese with this area's entries; English again afterwards, whatever the test switched to."""
    for english, chinese in ZH.items():
        monkeypatch.setitem(i18n.ZH, english, chinese)
    apply_language("zh")
    yield
    apply_language("en")


@pytest.fixture
def english():
    yield
    apply_language("en")


# -- what the application and the domain say, shown in the interface language ------------------------------


def test_messages_and_curve_titles_are_translated_with_their_values_kept(chinese) -> None:
    from src.gimap.features.analyze.application import MaskShape
    from src.gimap.features.analyze.presentation.texts import (
        NO_PROFILE, PROFILE_MISSING, curve_title, mask_text, message_text,
    )

    assert curve_title("Horizontal cut I(qy)") == "水平切线 I(qy)"
    assert curve_title("Box I(qz), q∥ = 0.100–0.200 Å⁻¹") == "框 I(qz), q∥ = 0.100–0.200 Å⁻¹"  # the values stay
    assert curve_title("I(χ), q = 1.000–1.100 Å⁻¹") == "I(χ), q = 1.000–1.100 Å⁻¹"  # nothing to translate
    assert message_text("The beam centre column lies outside the frame.") == "光束中心所在的列在帧之外。"
    english = NO_PROFILE.format(detector="this detector", shape="64×64")
    assert message_text(english) == "此探测器（64×64）没有仪器配置。标定一次或输入几何，即可得到 q 曲线。"
    assert message_text(NO_PROFILE.format(detector="PILATUS 2M", shape="1679×1475")).startswith("PILATUS 2M（1679×1475）")
    assert message_text("Horizontal cut I(qy): no valid pixels in the cut band.") == "水平切线 I(qy)：切线带中没有有效像素。"
    assert message_text(PROFILE_MISSING.format(name="P03 Pilatus")) == "这里没有仪器配置“P03 Pilatus”：几何将自动匹配。"
    assert message_text("Something nobody templated.") == "Something nobody templated."
    rectangle = MaskShape("rectangle", ((10.0, 20.0), (30.5, 40.0)))
    assert mask_text(rectangle) == "矩形：x 10–30，y 20–40"
    polygon = MaskShape("polygon", ((0.0, 0.0), (5.0, 0.0), (5.0, 5.0)))
    assert mask_text(polygon) == "多边形（3 个点）：x 0–5，y 0–5"
    apply_language("en")
    assert mask_text(rectangle) == rectangle.describe()  # the same words as the domain's, in English
    assert mask_text(polygon) == polygon.describe()


def test_the_geometry_summary_says_nothing_of_a_detector_the_file_does_not_name(chinese) -> None:
    from src.gimap.app.presentation.i18n import tr
    from src.gimap.features.analyze.presentation.view_model import AnalyzeViewModel

    geometry = SimpleNamespace(distance_m=1.73, wavelength_angstrom=1.34, incidence_deg=0.463)
    profile = SimpleNamespace(name="GALAXI Pilatus 1M")
    analysis = SimpleNamespace(shape=(1043, 981), detector_name=None, geometry=geometry, kind="gisaxs",
                               resolution=SimpleNamespace(profile=profile, how="matched"))
    owner = SimpleNamespace(state=SimpleNamespace(analysis=analysis))
    english = AnalyzeViewModel.geometry_summary(owner)
    assert english.startswith("1043×981 · D = 1730.0 mm") and "Detector" not in english
    assert english.endswith("GISAXS · profile “GALAXI Pilatus 1M” (matched)")
    assert AnalyzeViewModel.geometry_summary(owner, tr).endswith("GISAXS · 配置 “GALAXI Pilatus 1M”（自动匹配）")
    analysis.resolution.how = "missing"
    assert not AnalyzeViewModel.geometry_summary(owner).endswith(" · ")  # nothing to say: no empty part
    analysis.geometry, analysis.detector_name = None, "PILATUS 2M"
    assert AnalyzeViewModel.geometry_summary(owner, tr) == "PILATUS 2M 1043×981 · 没有几何"


# -- the page: composed in the interface language, again after a switch ---------------------------------------


@requires_public
def test_the_data_step_and_the_command_bar_follow_the_language(english, monkeypatch) -> None:
    page = _public_page()
    page.add_paths([str(GALAXI)])
    _done(page)
    card = page.data_info_label.text()
    meta = page.file_meta.text()
    assert "Detector:" not in card  # the TIFF names no detector: not “Detector: Detector”
    assert card.splitlines()[0] == "Frame: 1043 × 981 pixels, 172 µm pixels" and "Files listed: 1" in card
    assert meta == "GALAXI Pilatus 1M · 1043×981 px"  # the profile it matched names the detector
    assert page.step_rail.detail("data") == meta
    assert page.step_intro["data"].text() == "galaxi_data.tif: GALAXI Pilatus 1M, 1043×981 px."
    english_texts = (card, page.step_intro["geometry"].text(), page.summary_label.text(), page.fit_button.toolTip(),
                     page.status_text(), page.top_plot.title_label.text())
    assert "Detector" not in page.summary_label.text()
    assert "the horizontal cut, both halves on |qy| (two colours);" in page.fit_button.toolTip()

    for english_text, chinese in ZH.items():
        monkeypatch.setitem(i18n.ZH, english_text, chinese)
    apply_language("zh", [page])
    page.refresh_language()
    card = page.data_info_label.text()
    assert card.splitlines()[0] == "帧：1043 × 981 像素，像素尺寸 172 µm" and "列表中的文件：1" in card
    assert page.file_meta.text() == meta  # a name and a size: the same in every language
    assert page.step_intro["data"].text() == "galaxi_data.tif：GALAXI Pilatus 1M，1043×981 px。"
    assert page.step_intro["geometry"].text().startswith("几何来自“GALAXI Pilatus 1M”：1730.0 mm")
    assert page.summary_label.text().endswith("配置 “GALAXI Pilatus 1M”（自动匹配）")
    assert page.fit_button.toolTip().startswith("在拟合中打开曲线") and i18n.ZH["Both halves on |qy| (two colours)"] in \
        page.fit_button.toolTip()
    assert "条曲线" in page.status_text()

    apply_language("en", [page])
    page.refresh_language()
    assert (page.data_info_label.text(), page.step_intro["geometry"].text(), page.summary_label.text(),
            page.fit_button.toolTip(), page.status_text(), page.top_plot.title_label.text()) == english_texts
    page.dispose()
    page.close()


def _series(path: Path, frames: int = 12) -> Path:
    import h5py

    with h5py.File(path, "w") as handle:
        handle.create_dataset("entry/instrument/detector/data",
                              data=np.random.default_rng(0).poisson(5, (frames, 64, 64)).astype(np.int32))
    return path


def test_a_frame_without_a_geometry_says_so_in_the_interface_language(tmp_path: Path, chinese) -> None:
    page = _page(_context())  # no instrument profile: no geometry
    page.add_paths([str(_series(tmp_path / "insitu.nxs"))])
    _done(page)
    page.view_model.set_sum_count(3)
    page.view_model.set_frame(2)
    page.run_analysis()
    _done(page)
    assert page.step_rail.detail("geometry") == "还没有几何"
    assert page.step_rail.detail("cuts") == "需要几何"
    assert page.file_meta.text() == "64×64 px · 第 3 帧，共 12 帧，3 帧求和"
    assert page.data_info_label.text().splitlines() == [
        "帧：64 × 64 像素", "文件中的帧数：12（显示从第 3 帧起 3 帧之和）", "列表中的文件：1"]
    assert page.step_intro["geometry"].text().startswith("此探测器（64×64）还没有几何。")
    assert page.banner_label.text() == "此探测器（64×64）没有仪器配置。标定一次或输入几何，即可得到 q 曲线。"
    # said over the empty plot (no axes of an earlier frame), not in the title
    assert page.top_plot.empty_overlay.label.text() == "还没有曲线：这一帧需要几何（第 2 步）"
    assert page.top_plot.title_label.text() == ""
    assert page.summary_label.text() == "64×64 · 没有几何"
    page.dispose()
    page.close()


# -- the Series tab -------------------------------------------------------------------------------------------

SERIES_ZH = {
    "Frame {n}: {label}": "第 {n} 帧：{label}",
    "{curve} — {n} frames": "{curve} — {n} 帧",
    "I at {axis} = {centre} ± {half}": "{axis} = {centre} ± {half} 处的 I",
    "{rows} frames × {points} points. Click or drag the horizontal band to pick a frame, drag the vertical band (and "
    "its edges) to pick a {axis} window; Open shows the frame in Analyze.":
        "{rows} 帧 × {points} 个点。点击或拖动水平带选择一帧，拖动竖直带（及其边缘）选择 {axis} 窗口；“打开”在分析页显示这一帧。",
    "In-plane sector I(q)": "面内扇区 I(q)",
}


def test_the_series_tab_is_composed_again_after_a_switch(tmp_path: Path, english, monkeypatch) -> None:
    from src.gimap.features.analyze.bootstrap import create_analyze_view_model
    from src.gimap.features.analyze.presentation.page import AnalyzePage
    from src.gimap.integrations.state import InMemoryInstrumentProfileRepository
    from src.gimap.shared.geometry import DetectorGeometry, InstrumentProfile
    from tests.test_assistant_calibration import CENTER, DISTANCE_M, PIXEL, SHAPE, WAVELENGTH, save_tiff
    from tests.test_series_map import _ring_frame, _wait

    paths = [save_tiff(tmp_path / f"insitu_{index:03d}.tif", _ring_frame(100.0 * index, seed=index))
             for index in range(3)]
    geometry = DetectorGeometry(PIXEL, PIXEL, DISTANCE_M, CENTER[0], CENTER[1], WAVELENGTH, 0.2)
    profiles = InMemoryInstrumentProfileRepository([InstrumentProfile("synthetic", geometry, None, SHAPE)])
    page = AnalyzePage(create_analyze_view_model(_context(profiles)))
    try:
        page.set_mode_choice("giwaxs")
        page.add_paths([str(path) for path in paths])
        _done(page)
        page.series_build_button.click()
        _wait(page, lambda: page._series_map is not None and page._series_map.rows == 3 and not page.batch_running())
        _done(page)
        english_texts = (page.series_profile_plot.title_label.text(), page.series_map_view.title_label.text(),
                         page.series_trace_plot.title_label.text(), page.series_info_label.text())
        row, labels = page._series_row, page._series_map.labels
        assert english_texts[0] == f"Frame {row + 1}: {labels[row]}" and english_texts[1] == "I(q) — 3 frames"
        assert english_texts[3].startswith("3 frames × ") and "to pick a q window" in english_texts[3]
        titles = [page.series_curve_combo.itemText(index) for index in range(page.series_curve_combo.count())]
        assert "In-plane sector I(q)" in titles

        for key, value in {**ZH, **SERIES_ZH}.items():
            monkeypatch.setitem(i18n.ZH, key, value)
        apply_language("zh", [page])
        page.refresh_language()
        assert page.series_profile_plot.title_label.text() == f"第 {row + 1} 帧：{labels[row]}"
        assert page.series_map_view.title_label.text() == "I(q) — 3 帧"
        assert page.series_trace_plot.title_label.text().endswith("处的 I")
        assert "选择 q 窗口" in page.series_info_label.text()
        titles = [page.series_curve_combo.itemText(index) for index in range(page.series_curve_combo.count())]
        assert "面内扇区 I(q)" in titles and page.series_curve_combo.currentData() == "radial"

        apply_language("en", [page])
        page.refresh_language()
        assert (page.series_profile_plot.title_label.text(), page.series_map_view.title_label.text(),
                page.series_trace_plot.title_label.text(), page.series_info_label.text()) == english_texts
    finally:
        page.dispose()
        page.close()



def test_the_detector_note_of_the_stages_names_the_axis_of_the_map(chinese) -> None:
    from src.gimap.features.analyze.presentation.bindings.series_stages import SeriesStagesMixin

    odd = SimpleNamespace(row=3, narrow=True, q=12.5, z=9.0)
    stages = SimpleNamespace(
        stage_representatives=lambda: [0], ranges=lambda: [(0, 9)], stage_changes=lambda: [], half_row=None,
        count=1, ninety_row=None, q=np.linspace(-90.0, 90.0, 50), odd=[odd],
    )
    series = SimpleNamespace(x_label="χ (°)", labels=[f"frame {n}" for n in range(1, 11)])
    text = SeriesStagesMixin._stages_text(None, series, stages)
    assert trf("A difference near one {axis} in a few points is the detector: mask it in the Mask step.",
               axis="χ") in text
    assert "某个 χ 附近" in text and "某个 q 附近" not in text


def test_change_along_the_series_waits_for_the_stages_with_a_short_title(english) -> None:
    from src.gimap.features.analyze.presentation.views.series_view import CHANGE_EMPTY

    page = _page(_context())
    plot = page.series_trace_plot
    page._series_stages = None
    assert page._draw_change_trace()
    assert plot.title_label.text() == "Change along the series"
    assert plot.empty_overlay.text == CHANGE_EMPTY and not plot.has_curves()
    page.dispose()
    page.close()


# -- a guard on the source -------------------------------------------------------------------------------------

TEXT_CALLS = {"_status": 0, "notify_written": 0, "show_toast": 1, "set_step_state": 2, "setText": 0, "setToolTip": 0,
              "set_title": 0, "showMessage": 0, "_intro": 1, "_series_say": 0}
"""Calls that put a text on screen, and which argument is the text (a window's own title and labels are translated
when it is shown)."""
TRANSLATED_BY_CALLEE = {"set_title", "_intro", "_series_say"}
"""These translate the English they are given (``CurvePlot.set_title``, the step intro, the Series sentence)."""


def _english(node) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, str) and any(c.isalpha() for c in node.value)


def test_no_text_on_screen_is_formatted_before_its_translation() -> None:
    problems = []
    for path in sorted(PRESENTATION.rglob("*.py")):
        if "views" in path.parts or path.name == "automation.py":  # static layout; the assistant's figures
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.Call):
                continue
            name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", None)
            if name in ("tr", "trf") and node.args and isinstance(node.args[0], ast.JoinedStr):
                problems.append(f"{path.name}:{node.lineno} tr(f\"…\") can never match the table")
            index = TEXT_CALLS.get(name)
            if index is None or len(node.args) <= index:
                continue
            text = node.args[index]
            branches = [text.body, text.orelse] if isinstance(text, ast.IfExp) else [text]
            for branch in branches:
                if isinstance(branch, ast.JoinedStr) or isinstance(branch, ast.BinOp) and any(
                        isinstance(side, ast.JoinedStr) or _english(side) for side in (branch.left, branch.right)):
                    problems.append(f"{path.name}:{node.lineno} {name}: formatted before the translation")
                elif _english(branch) and name not in TRANSLATED_BY_CALLEE:
                    problems.append(f"{path.name}:{node.lineno} {name}: {branch.value!r} without tr()")
    assert problems == []
