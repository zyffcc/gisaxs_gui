"""Wave 2b, the Results tab of the automatic analysis: a peak table that fits a 1280-px window, tables that copy
their rows, selectable result lines (assistant-15) and texts composed in the interface language (assistant-3).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PyQt5.QtCore import Qt
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QAbstractItemView, QApplication, QLabel, QTableWidget, QWidget

from src.gimap.app.presentation import i18n
from src.gimap.app.presentation.i18n import DEFAULT_LANGUAGE, apply_language
from src.gimap.app.presentation.theme import theme_manager
from src.gimap.features.assistant.presentation.guided_results import (
    GuidedResultsPanel,
    change_text,
    decision_text,
    source_text,
)
from src.gimap.features.assistant.presentation.guided_tables import (
    PEAK_COLUMNS,
    PEAK_HEADERS,
    ROLE,
    compact_orientation,
    compact_trust,
    peak_cells,
    peak_table,
)
from src.gimap.features.assistant.presentation.guided_text import badge, label

PANEL = Path(__file__).resolve().parents[1] / "src" / "gimap" / "features" / "assistant"
SPIKE = ("a spike one or two bins wide far above the background: hot pixels, a module edge or a zinger rather than "
         "diffraction (check whether the calibration image shows it too)")
PEAKS = [  # the real P08 MAPbI3 frame: the strongest peaks and their verdicts
    {"q": 0.9931, "d_A": 6.327, "fwhm": 0.01385, "caveat": "", "flags": [], "size_nm": 40.8, "size_is_lower_bound": True,
     "orientation": "mainly out-of-plane", "size_note": "Scherrer, instrument broadening not removed"},
    {"q": 1.4103, "d_A": 4.455, "fwhm": 0.02112, "caveat": "", "flags": [], "size_nm": 26.8, "size_is_lower_bound": True,
     "orientation": "only the in-plane sector is usable here: the other is shadowed (peak seen)"},
    {"q": 1.7241, "d_A": 3.644, "fwhm": 0.00312, "caveat": SPIKE, "flags": ["spike"], "orientation": "both sectors (no strong preference)"},
    {"q": 1.9935, "d_A": 3.152, "fwhm": 0.0245, "caveat": "weak (3–5σ): tentative, may be noise", "flags": ["weak"],
     "orientation": "not comparable: the in-plane sector is shadowed and the out-of-plane sector is not measured"},
    {"q": 2.4521, "d_A": 2.562, "fwhm": 0.0301, "caveat": "", "flags": ["at_edge"], "orientation": ""},
]
ZH_TEST = {
    "trust": "可信度", "OOP only": "仅 OOP", "no fit": "未拟合", "neither": "均未见", "both": "两者", "spike": "尖峰", "weak": "弱", "edge": "边缘", "n/a": "无法判断",
    "Peaks": "峰", "Geometry": "几何", "Series": "序列", "Distance": "距离", "reliable": "可靠",
    "Whether this is a crystal peak: spikes (one hot pixel, a streak or a flat-topped box from one detector row), "
    "broad halos (amorphous order) and weak peaks are flagged.": "是否为晶体峰：尖峰、晕环和弱峰都会被标出。",
    "Ring q ≈ {q} Å⁻¹: f = {f} — {texture}": "环 q ≈ {q} Å⁻¹：f = {f} —— {texture}",
    "oriented along the surface normal (out-of-plane, χ ≈ 0°)": "沿表面法线取向（面外，χ ≈ 0°）",
    "A series of {total} frames: these results are frames {first}–{last} (the final state).":
        "共 {total} 帧的序列：这些结果来自第 {first}–{last} 帧（最终状态）。",
    "at the start": "开头", "present at both": "两端都有", "shifted {move} from {q}": "移动 {move}（原在 {q}）",
    "instrument profile “{name}”": "仪器配置“{name}”",
}


def _app() -> QApplication:
    return QApplication.instance() or QApplication([])


def _report(**extra) -> dict:
    report = {
        "ok": True, "procedure": "giwaxs", "frame": "C:/data/S121_00841.tif", "peaks": PEAKS, "steps": [],
        "rings": [{"q": 0.9931, "herman": 0.61, "herman_isotropic": 0.0,
                   "texture": "oriented along the surface normal (out-of-plane, χ ≈ 0°)"}],
        "needs_attention": [], "decisions": [],
        "geometry": {"source": "instrument profile 'P08 Lambda'", "distance_mm": 233.2, "beam_center_px": [1189.0, 1497.0],
                     "wavelength_A": 0.6888, "incidence_deg": 0.4},
    }
    report.update(extra)
    return report


@pytest.fixture()
def zh(monkeypatch):
    for english, chinese in ZH_TEST.items():
        monkeypatch.setitem(i18n.ZH, english, chinese)
        monkeypatch.setitem(i18n._TO_ENGLISH, chinese, english)
    yield
    apply_language(DEFAULT_LANGUAGE)


def test_the_peak_table_is_compact_and_says_the_rest_in_its_tooltips() -> None:
    _app()
    parent = QWidget()
    table = peak_table(PEAKS, parent)
    headers = [table.horizontalHeaderItem(column).text() for column in range(table.columnCount())]
    assert headers == list(PEAK_HEADERS) == ["q", "d", "FWHM", "trust", "L", "OOP/IP"]  # the units: in the legend
    for column, (full, how) in enumerate(PEAK_COLUMNS):  # the full name, its unit and how it is obtained, on hover
        assert table.horizontalHeaderItem(column).toolTip() == f"{full}\n{how}"
    cells = [[table.item(row, column).text() for column in range(6)] for row in range(table.rowCount())]
    assert cells[0] == ["0.9931", "6.327", f"{0.01385:.3g}", "✓", "≥ 40.8", "OOP"]
    assert cells[1][3:] == ["✓", "≥ 26.8", "IP*"]  # one sector only
    assert cells[2][3] == "⚠ spike" and cells[2][5] == "both" and cells[2][4] == "—"
    assert cells[3][3] == "⚠ weak" and cells[3][5] == "n/a"
    assert cells[4][3] == "⚠ edge" and cells[4][5] == "—"
    assert table.item(2, 3).toolTip().startswith("artefact (spike)\na spike one or two bins wide")
    assert "the other is shadowed" in table.item(1, 5).toolTip() and "cut off" in table.item(4, 3).toolTip()
    # Numbers right-aligned, the verdict coloured by the theme (and again after a switch of the theme).
    for column in (0, 1, 2, 4):
        assert table.item(0, column).textAlignment() & Qt.AlignRight
    assert not table.item(0, 3).textAlignment() & Qt.AlignRight
    assert table.item(0, 3).data(ROLE) == "success" and table.item(2, 3).data(ROLE) == "warning"
    assert table.item(2, 3).foreground().color() == theme_manager().color("warning")
    assert not table.verticalHeader().isVisible()  # no row numbers: the room goes to the values
    assert table.selectionMode() == QAbstractItemView.SingleSelection  # Fit details follow the current row
    parent.deleteLater()


@pytest.mark.parametrize("language", ["en", "zh"])
def test_the_peak_table_fits_the_results_tab_of_a_1280_px_window(language, zh) -> None:
    _app()
    apply_language(language)
    parent = QWidget()
    parent.resize(330, 400)  # the Results tab at 1280 px: about 356 px, the table about 330 px of it
    failed = dict(PEAKS[0], caveat="its shape could not be fitted: look at the image", flags=["fit_failed"])
    table = peak_table([*PEAKS, failed], parent)
    table.resize(330, 300)
    parent.show()
    QApplication.processEvents()
    header = table.horizontalHeader()
    needed = sum(max(header.sectionSizeHint(column), table.sizeHintForColumn(column)) for column in range(6))
    assert needed <= table.viewport().width(), (needed, table.viewport().width())
    assert table.horizontalScrollBar().maximum() == 0  # nothing to scroll sideways
    parent.close()


def test_compact_words() -> None:
    assert compact_trust("") == "✓" and compact_trust("", at_edge=True) == "⚠ edge"
    assert compact_trust("its shape could not be fitted: look at the image") == "⚠ no fit"
    assert compact_trust("a broad halo (amorphous or liquid-like order), not a crystalline reflection") == "⚠ halo"
    assert compact_trust("something new") == "⚠ check"
    assert compact_orientation("mainly in-plane") == "IP" and compact_orientation("out-of-plane only") == "OOP only"
    assert compact_orientation("only the out-of-plane sector is measured here (no peak)") == "OOP*"
    assert compact_orientation("not measured: neither sector reaches this q on the detector") == "n/a"
    assert compact_orientation("not detected in either sector") == "neither" and compact_orientation("") == "—"
    assert peak_cells([])[:1] == []


def test_tables_copy_their_rows_and_result_lines_can_be_selected() -> None:
    _app()
    panel = GuidedResultsPanel(lambda: None)
    start = _report(peaks=[dict(PEAKS[0], q=0.9900)])
    panel.show_report(_report(frames={"total": 40, "first": 31, "summed": 10}), start)
    table = panel.peak_table
    table.selectRow(2)
    QTest.keyClick(table, Qt.Key_C, Qt.ControlModifier)  # Ctrl+C: the selected row with the column names
    copied = QApplication.clipboard().text().split("\n")
    # pasted rows carry the full column names with their units, not the short headers that fit the panel
    assert copied[0] == "\t".join(full for full, _how in PEAK_COLUMNS) and copied[1].startswith("1.724\t3.644\t")
    series = panel.findChild(QTableWidget, "guidedSeriesTable")
    assert series.property("gimapTableCopy") and series.selectionMode() == QAbstractItemView.ExtendedSelection
    assert series.item(0, 0).textAlignment() & Qt.AlignRight
    assert series.item(0, 3).text().startswith("shifted +0.003 Å⁻¹")
    assert table.property("gimapTableCopy") and table.contextMenuPolicy() == Qt.ActionsContextMenu
    selectable = [line for line in panel.findChildren(QLabel) if line.textInteractionFlags() & Qt.TextSelectableByMouse]
    texts = [line.text() for line in selectable]
    assert any(text.startswith("GIWAXS — 5 peaks") for text in texts)  # the outcome line
    assert "233.2 mm" in texts and any(text.startswith("Ring q ≈ 0.9931") for text in texts)
    assert label("x", panel).textInteractionFlags() & Qt.TextSelectableByMouse
    assert badge("x", "success", panel).textInteractionFlags() & Qt.TextSelectableByMouse
    panel.deleteLater()


def test_the_results_are_composed_in_chinese_and_again_after_a_switch(zh) -> None:
    _app()
    panel = GuidedResultsPanel(lambda: None)
    start = _report(peaks=[dict(PEAKS[0], q=0.9900)])
    panel.show_report(_report(frames={"total": 40, "first": 31, "summed": 10}), start)
    panel.peak_table.selectRow(3)
    apply_language("zh")
    panel.refresh_language()
    texts = [line.text() for line in panel.findChildren(QLabel)]
    assert "峰" in texts and "几何" in texts and "距离" in texts
    assert "环 q ≈ 0.9931 Å⁻¹：f = 0.61 —— 沿表面法线取向（面外，χ ≈ 0°）" in texts  # numbers as they are
    assert "共 40 帧的序列：这些结果来自第 31–40 帧（最终状态）。" in texts
    assert "仪器配置“P08 Lambda”" in texts
    table = panel.peak_table
    assert table.horizontalHeaderItem(3).text() == "可信度" and table.horizontalHeaderItem(0).text() == "q"
    assert table.horizontalHeaderItem(3).toolTip().endswith("是否为晶体峰：尖峰、晕环和弱峰都会被标出。")
    assert [table.item(row, 3).text() for row in range(5)] == ["✓", "✓", "⚠ 尖峰", "⚠ 弱", "⚠ 边缘"]
    assert table.item(2, 5).text() == "两者" and table.item(3, 5).text() == "无法判断" and table.item(0, 0).text() == "0.9931"
    assert table.currentRow() == 3  # the peak selected before the switch
    series = panel.findChild(QTableWidget, "guidedSeriesTable")
    assert series.horizontalHeaderItem(1).text() == "开头" and series.item(0, 3).text().startswith("移动 +0.003 Å⁻¹")
    apply_language(DEFAULT_LANGUAGE)
    panel.refresh_language()
    assert panel.peak_table.horizontalHeaderItem(3).text() == "trust" and panel.peak_table.currentRow() == 3
    panel.deleteLater()


def test_words_of_the_series_and_the_geometry() -> None:
    assert change_text("present at both") == "present at both"
    assert change_text("shifted +0.012 Å⁻¹ (+0.5%) from 1.23") == "shifted +0.012 Å⁻¹ (+0.5%) from 1.23"
    moved = "moved? 3.899 → 3.884 (-0.4%): one line shifting further than its width, or one line replacing another"
    assert change_text(moved) == moved
    assert source_text("instrument profile 'P08 Lambda'") == "instrument profile “P08 Lambda”"
    assert source_text(None) == "instrument profile" and source_text("AgBH_00001.tif") == "AgBH_00001.tif"
    assert decision_text("kept the instrument profile 'P08' this detector already has") == (
        "kept the instrument profile “P08” this detector already has")
    assert decision_text("rejected AgBH_00001.tif") == "rejected AgBH_00001.tif"


def test_every_module_of_the_package_stays_within_600_lines() -> None:
    long = {str(path.relative_to(PANEL)): count for path in PANEL.rglob("*.py")
            if (count := len(path.read_text(encoding="utf-8").splitlines())) > 600}
    assert not long, long
