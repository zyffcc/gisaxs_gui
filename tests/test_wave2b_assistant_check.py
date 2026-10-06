"""Wave 2b check pass, the automatic analysis: what the review of assistant-3/14/15 found and fixed.

The sentences a run writes (with numbers in them) are shown in the interface language by their form
(``guided_words``); a hidden answer form stays hidden through a language switch; Fit details never make the
Results tab wider than a 1280-px window gives it; the report gives the αi the run used.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import numpy as np
import pytest
from PyQt5 import sip
from PyQt5.QtCore import QCoreApplication, QEvent
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget

from src.gimap.app.presentation import i18n
from src.gimap.app.presentation.i18n import DEFAULT_LANGUAGE, apply_language
from src.gimap.features.assistant.application import (
    GOALS,
    PERMISSION_AUTO,
    AnalysisGoals,
    PipelineOptions,
    RunResults,
    StandardPipeline,
    ToolCatalog,
)
from src.gimap.features.assistant.domain import choose_halves, fit_curve
from src.gimap.features.assistant.presentation import GuidedAnalysis
from src.gimap.features.assistant.presentation.guided_details import PeakDetails
from src.gimap.features.assistant.presentation.guided_gisaxs import outcome_summary
from src.gimap.features.assistant.presentation.guided_results import GuidedResultsPanel
from src.gimap.features.assistant.presentation.guided_words import NESTED, SENTENCES, parts, words
from tests.assistant_fakes import FakeWorkbench

ROOT = Path(__file__).resolve().parents[1] / "src" / "gimap" / "features"
SOURCES = [*(ROOT / "assistant" / "domain").glob("*.py"), *(ROOT / "assistant" / "application").glob("*.py"),
           ROOT / "fitting" / "infrastructure" / "adapters" / "experimental_fit.py"]
ZH_TEST = {
    "tilted: maximum at χ ≈ ±{chi}°": "倾斜取向：极大值位于 χ ≈ ±{chi}°",
    "oriented along the surface normal (out-of-plane, χ ≈ 0°)": "沿表面法线取向（面外，χ ≈ 0°）",
    "{texture} — but the strongest measured |χ| ({chi}°) borders an unmeasured range, so the true maximum may lie "
    "inside it": "{texture} —— 但测得的最强处（|χ| = {chi}°）紧邻未测量的范围，真正的极大值可能就在其中",
    "Both halves are usable (points in {a} and {b} of their |qy| ranges) and {agree}: they are averaged where both "
    "exist, which halves the noise{extend}.": "两半都可用（分别在各自 |qy| 范围的 {a} 和 {b} 内有数据点），并且{agree}：在两半都有数据的地方取平均，噪声减半{extend}。",
    "agree within {percent} %": "相差在 {percent} % 以内",
    "; beyond {q} Å⁻¹ the longer {side} half continues alone to {reach} Å⁻¹": "；超过 {q} Å⁻¹ 后，较长的 {side} 这一半单独延续到 {reach} Å⁻¹",
    "mean of both halves up to |qy| = {q} Å⁻¹": "两半的平均，至 |qy| = {q} Å⁻¹",
    "then the {side} half alone to {q} Å⁻¹": "之后只用 {side} 这一半，至 {q} Å⁻¹",
    "a spike one or two bins wide far above the background: hot pixels, a module edge or a zinger rather than "
    "diffraction (check whether the calibration image shows it too)": "一个只有一两个 bin 宽、远高于背景的尖峰",
    "halves: {halves}": "两半：{halves}", "(shoulder)": "（肩峰）",
    "best fit {model} R {radius} nm (χ² {chi2})": "最佳拟合 {model} R {radius} nm（χ² {chi2}）",
    "Only you can answer these — then run again:": "只有你能回答这些问题 —— 然后再运行：",
}


def _app() -> QApplication:
    return QApplication.instance() or QApplication([])


def _flush() -> None:
    for _ in range(3):
        QApplication.processEvents()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)


@pytest.fixture()
def zh(monkeypatch):
    for english, chinese in ZH_TEST.items():
        monkeypatch.setitem(i18n.ZH, english, chinese)
        monkeypatch.setitem(i18n._TO_ENGLISH, chinese, english)
    apply_language("zh")
    yield
    apply_language(DEFAULT_LANGUAGE)


# -- the sentences a run writes ---------------------------------------------------------------------------


def test_a_run_sentence_is_shown_by_its_form_with_its_values_as_written(zh) -> None:
    assert words("tilted: maximum at χ ≈ ±26°") == "倾斜取向：极大值位于 χ ≈ ±26°"
    gap = ("oriented along the surface normal (out-of-plane, χ ≈ 0°) — but the strongest measured |χ| (4°) borders "
           "an unmeasured range, so the true maximum may lie inside it")
    assert words(gap) == "沿表面法线取向（面外，χ ≈ 0°） —— 但测得的最强处（|χ| = 4°）紧邻未测量的范围，真正的极大值可能就在其中"
    assert words("something no run writes") == "something no run writes" and words(None) == ""
    curve = "mean of both halves up to |qy| = 0.179 Å⁻¹; then the qy < 0 half alone to 0.277 Å⁻¹"
    assert parts(curve) == "两半的平均，至 |qy| = 0.179 Å⁻¹；之后只用 qy < 0 这一半，至 0.277 Å⁻¹"


def test_the_halves_reason_of_the_domain_is_translated_by_its_form(zh) -> None:
    q = np.linspace(-0.2, 0.25, 451)  # the qy > 0 half reaches further: "beyond … the longer … half"
    intensity = 100 * np.exp(-np.abs(q) / 0.05) + 5
    choice = choose_halves(q, intensity, np.full(q.shape, 10.0))
    assert choice.side == "mean" and "beyond" in choice.reason
    shown = words(choice.reason)
    assert shown.startswith("两半都可用（分别在各自 |qy| 范围的 100% 和 100% 内有数据点），并且相差在 0 % 以内")
    assert "；超过 0.2 Å⁻¹ 后，较长的 qy > 0 这一半单独延续到 0.25 Å⁻¹。" in shown
    note = fit_curve(q, intensity, np.sqrt(intensity), np.full(q.shape, 10.0), "mean")[3]
    assert not re.search(r"[A-Za-z]{4,}", parts(note).replace("qy", "")), parts(note)


def test_every_sentence_form_is_still_written_so_by_the_analysis() -> None:
    """A form whose wording changed in the domain or application would silently stay English: each fixed
    piece of every template must still be in what the code writes (f-strings joined as Python joins them)."""
    written = []
    for path in SOURCES:
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.JoinedStr):
                written.append("\0".join(part.value if isinstance(part, ast.Constant) else "" for part in node.values))
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                written.append(node.value)
    text = "\n".join(written)

    def found(piece: str) -> bool:  # as written, or in two strings the code joins (a sentence inside a sentence)
        spaces = [index for index, char in enumerate(piece) if char == " "]
        return piece in text or any(piece[:index].strip() in text and piece[index:].strip() in text for index in spaces)

    for template in SENTENCES:
        for piece in re.split(r"\{\w+\}", template):
            piece = piece.strip(" .;:()")
            if len(piece) >= 12:
                assert found(piece), (template, piece)
    assert {"texture", "agree", "extend"} <= NESTED


def test_calibration_verdicts_and_decisions_keep_their_numbers_and_names(monkeypatch, zh) -> None:
    from src.gimap.features.assistant.domain import assessment, compare_verdict

    for english, chinese in {
        "good: {lines} lines of the standard land within {error} of their q on average": "好：标样的 {lines} 条谱线平均落在其 q 的 {error} 以内",
        "clear: {standard} ({rings} rings, rms {rms} px)": "明确：{standard}（{rings} 个环，rms {rms} px）",
        "clear: {standard} ({rings} rings, rms {rms} px) beats {second} ({second_rings} rings, rms {second_rms} px)":
            "明确：{standard}（{rings} 个环，rms {rms} px）优于 {second}（{second_rings} 个环，rms {second_rms} px）",
        "rejected {name} ({standard})": "未采用 {name}（{standard}）", "rejected {name}": "未采用 {name}",
        "from given": "由你给出", "from {name}": "来自 {name}",
    }.items():
        monkeypatch.setitem(i18n.ZH, english, chinese)
    good = assessment({"line_check": {"lines_checked": 5, "mean_relative": 0.0005}})
    assert words(good) == "好：标样的 5 条谱线平均落在其 q 的 0.05% 以内"
    fits = [{"standard": "AgBH", "matched_rings": 6, "rms_residual_px": 0.4}, {"standard": "LaB6", "matched_rings": 3,
                                                                                  "rms_residual_px": 1.2}]
    beats = compare_verdict(fits)
    assert beats.startswith("clear: AgBH (6 rings") and " beats LaB6 " in beats
    assert words(beats) == "明确：AgBH（6 个环，rms 0.4 px）优于 LaB6（3 个环，rms 1.2 px）"  # not the short form
    assert words(compare_verdict(fits[:1])) == "明确：AgBH（6 个环，rms 0.4 px）"
    assert words("rejected AgBH_00001.tif (agbh)") == "未采用 AgBH_00001.tif（agbh）"
    assert words("rejected AgBH_00001.tif") == "未采用 AgBH_00001.tif"
    assert words("from given") == "由你给出" and words("from AgBH.poni") == "来自 AgBH.poni"  # the exact sentence first


def test_english_is_never_rewritten() -> None:
    apply_language(DEFAULT_LANGUAGE)
    sentence = "tilted: maximum at χ ≈ ±26°"
    assert words(sentence) == sentence and parts("a; b") == "a; b"


def test_the_gisaxs_outcome_uses_the_chinese_comma(zh) -> None:
    report = {"gisaxs": {"halves": {"side": "mean"}, "spacing": {"distance_nm": 53.5, "kind": "shoulder"},
                         "fit": {"solutions": [{"model": "random_cylinder", "chi2": 1.22, "components": [{"R": 6.13}]}]}}}
    line = outcome_summary(report)
    assert ", " not in line and "D ≈ 53.5 nm（肩峰）" in line and line.count("，") == 2


# -- widgets ------------------------------------------------------------------------------------------------


def test_a_put_away_answer_form_stays_hidden_through_a_language_switch(zh) -> None:
    _app()
    guided = GuidedAnalysis(lambda: None)
    guided._show_questions([{"item": "incidence angle αi", "option": "incidence_deg", "why": "", "hint": ""}])
    assert not guided.questions.isHidden()
    guided.questions.hide()  # the results of another file are shown: the questions are put away with them
    guided.refresh_language()
    assert guided.questions.isHidden() and guided.questions.title_label.text() == "只有你能回答这些问题 —— 然后再运行："
    guided.results.deleteLater()


def test_fit_details_never_make_the_results_tab_wider(zh) -> None:
    _app()
    apply_language(DEFAULT_LANGUAGE)
    holder = QWidget()
    layout = QVBoxLayout(holder)
    layout.setContentsMargins(0, 0, 0, 0)
    details = PeakDetails(lambda *args: False, holder)
    layout.addWidget(details)
    caveat = next(key for key in ZH_TEST if key.startswith("a spike"))
    peak = {"q": 0.4524, "d_A": 13.89, "fwhm": 0.05119, "snr": 32.7, "ratio_out_in": 1.59, "caveat": caveat,
            "fit": {"x": [0.3, 0.4, 0.5], "y": [1.0, 2.0, 1.0], "q_err": 0.00075, "d_err_A": 0.023, "fwhm_err": 0.0019,
                    "height": 10.46, "area": 0.5699, "background": 9.953, "slope": -1.2, "reduced_chi2": 1.39,
                    "points": 185, "window": [0.2445, 0.6575]}}
    details.show_peak(peak, {"rings": []}, map_width=300)
    holder.resize(312, 1200)
    holder.show()
    QApplication.processEvents()
    assert holder.minimumSizeHint().width() <= 312 and details.values._columns == 1  # one column of pairs
    holder.resize(560, 1200)
    QApplication.processEvents()
    assert details.values._columns == 2  # room for two
    apply_language("zh")
    details.show_peak(peak, {"rings": []}, map_width=300)
    assert details.warnings.text() == "⚠ 一个只有一两个 bin 宽、远高于背景的尖峰"  # the caveat in Chinese
    details.dispose()
    holder.deleteLater()


def test_a_q_map_left_out_of_the_layout_is_deleted_with_the_next_report() -> None:
    _app()
    panel = GuidedResultsPanel(lambda: None)  # no Analyze: no q map, so the label is not put in the layout
    report = {"ok": True, "procedure": "giwaxs", "frame": "C:/data/a.tif", "rings": [{"q": 1.0}],
              "peaks": [{"q": 1.0, "d_A": 6.28, "fwhm": 0.01, "caveat": "", "orientation": ""}]}
    panel.show_report(report)
    first = panel.q_map
    assert first is not None and panel.layout_.indexOf(first) < 0
    panel.show_report(report)
    _flush()
    assert sip.isdeleted(first) and len(panel.findChildren(QLabel, "guidedQMap")) == 1
    panel.deleteLater()


# -- the record --------------------------------------------------------------------------------------------


class _Profiled(FakeWorkbench):
    """A detector with a saved profile whose αi the run changes (the answer to the αi question)."""

    def __init__(self):
        super().__init__()
        self.incidence = 0.075

    def status(self) -> dict:
        status = super().status()
        status["geometry"] = dict(status["geometry"], incidence_deg=self.incidence, instrument_profile="P08")
        return status

    def set_incidence(self, degrees):
        self.incidence = degrees
        return super().set_incidence(degrees)


def test_the_report_gives_the_incidence_angle_the_run_used() -> None:
    workbench = _Profiled()
    catalog = ToolCatalog(workbench, AnalysisGoals(goals=tuple(GOALS), permission=PERMISSION_AUTO), RunResults())
    report = StandardPipeline(catalog, PipelineOptions(incidence_deg=0.21)).run()
    assert ("set_incidence", 0.21) in workbench.calls
    assert report["geometry"]["incidence_deg"] == pytest.approx(0.21)  # not the profile's 0.075 before the run
