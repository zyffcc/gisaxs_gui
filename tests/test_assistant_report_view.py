"""The run report: the model's findings per result plus tables built only from tool results."""

from __future__ import annotations

from src.gimap.features.assistant.application import (
    RUN_COMPLETED,
    RUN_FAILED,
    AnalysisGoals,
    LlmUsage,
    RunOutcome,
    RunResults,
    ToolCatalog,
)
from src.gimap.features.assistant.presentation.report_view import report_html
from tests.assistant_fakes import FakeWorkbench, call


def _outcome() -> RunOutcome:
    results = RunResults()
    catalog = ToolCatalog(FakeWorkbench(), AnalysisGoals(goals=("peaks", "crystallite_size")), results)
    catalog.execute(call("find_peaks", curve="radial"))
    catalog.execute(call("find_peaks", curve="in_plane", q_min=2.0, q_max=2.2))
    catalog.execute(call("crystallite_size", q_center=1.2))
    catalog.execute(call("note_missing_capability", capability="pole figure", reason="one incidence angle only"))
    catalog.execute(call(
        "submit_report",
        summary="Four peaks; sizes are lower bounds.",
        items=[
            {"item": "peaks", "status": "done", "findings": "Four peaks.", "evidence": "find_peaks(radial)", "reason": ""},
            {"item": "crystallite_size", "status": "not_available", "findings": "",
             "evidence": "crystallite_size", "reason": "No instrumental width; only a lower bound."},
        ],
        caveats=["Synthetic data."],
        suggestions=["Measure LaB6."],
    ))
    return RunOutcome(RUN_COMPLETED, "Report submitted.", [], results, LlmUsage(1200, 300, 5000, 0), model="claude-opus-5")


def test_the_report_shows_findings_reasons_and_computed_tables() -> None:
    page = report_html(_outcome(), cost=0.05, elapsed=42.0)
    assert "<h3>Summary</h3>" in page and "Four peaks; sizes are lower bounds." in page
    assert "not available" in page and "No instrumental width; only a lower bound." in page
    assert "Peaks of 'radial'" in page and "d (Å)" in page
    # A search that found nothing says why, in place of an empty table.
    assert "Peaks of 'in_plane'" in page and "No peak rises" in page
    assert "Crystallite size (Scherrer, K = 0.9)" in page and "≥ " in page
    assert "pole figure" in page and "Measure LaB6." in page
    assert "≈ $0.05" in page and "42 s" in page and "6,200 input" in page


def test_a_chinese_report_translates_the_fixed_headings() -> None:
    page = report_html(_outcome(), language="中文")
    assert "<h3>摘要</h3>" in page and "峰位表" in page and "无法得到" in page and "原因：" in page
    assert "计算结果（来自工具）" in page and "注意事项" in page and "记录的缺失功能" in page


def test_gisaxs_results_have_readable_headings() -> None:
    from src.gimap.features.assistant.application import AssistantReport, ReportItem

    results = RunResults()
    results.report = AssistantReport(
        summary="Cut at the Yoneda band.",
        items=(ReportItem("gisaxs_cut", "done", "Rows 605–610.", "", ""),
               ReportItem("in_plane_spacing", "partial", "A shoulder.", "", "Not a resolved peak."),
               ReportItem("gisaxs_fit", "done", "Cylinders.", "", "")),
        caveats=(), suggestions=(),
    )
    outcome = RunOutcome(RUN_COMPLETED, "Report submitted.", [], results, LlmUsage())
    page = report_html(outcome)
    for title in ("GISAXS cut", "In-plane spacing", "GISAXS fit"):
        assert f"<h4>{title} " in page
    assert "gisaxs_cut" not in page and "in_plane_spacing" not in page and "gisaxs_fit" not in page
    chinese = report_html(outcome, language="中文")
    assert "GISAXS 水平切线" in chinese and "面内间距" in chinese and "GISAXS 拟合" in chinese


def test_halves_are_named_in_words() -> None:
    from src.gimap.features.assistant.application.operations import describe, halves_label

    assert halves_label("both_abs") == "Both halves on |qy|" and halves_label("mean", "中文") == "两半平均"
    assert halves_label("positive", "zh") == "只用 qy > 0 一半" and halves_label("odd") == "odd"
    assert describe("set_halves", {"side": "negative"}) == "Only the qy < 0 half"


def test_a_run_without_report_explains_why() -> None:
    outcome = RunOutcome(RUN_FAILED, "The Claude API rejected the credentials.", [], RunResults(), LlmUsage())
    page = report_html(outcome)
    assert "<h3>No report</h3>" in page and "rejected the credentials" in page
