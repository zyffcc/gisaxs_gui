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


def test_calibration_reports_distinguish_ring_points_from_ring_radii():
    from src.gimap.features.assistant.application.pipeline_report import _geometry_lines, calibration_quality

    result = {"matched_rings": 3, "rms_residual_px": 0.8, "initial_point_rms_px": 1.1, "point_rms_px": 0.3}
    outcome = _outcome()
    outcome.results.calibrations.extend((result, {"rms_residual_px": 0.65}))
    expected = "Ring-point RMS 1.1 px before refinement, 0.3 px after"
    page = report_html(outcome)
    assert expected in page and "Ring-radius RMS 0.65 px" in page
    assert "精修前为 1.1 px，精修后为 0.3 px" in report_html(outcome, language="中文")
    lines = _geometry_lines({"geometry": {"distance_mm": 200}, "calibration_quality": calibration_quality(result)})
    assert "- " + expected in lines


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
    # 2π/q* is a length scale of the cut, not necessarily the particle spacing: the heading says so.
    for title in ("GISAXS cut", "Length scale 2π/q*", "GISAXS fit"):
        assert f"<h4>{title} " in page
    assert "gisaxs_cut" not in page and "in_plane_spacing" not in page and "gisaxs_fit" not in page
    assert "In-plane spacing" not in page
    chinese = report_html(outcome, language="中文")
    assert "GISAXS 水平切线" in chinese and "特征长度 2π/q*" in chinese and "GISAXS 拟合" in chinese


def test_halves_are_named_in_words() -> None:
    from src.gimap.features.assistant.application.operations import describe, halves_label

    assert halves_label("both_abs") == "Both halves on |qy|" and halves_label("mean", "中文") == "两半平均"
    assert halves_label("positive", "zh") == "只用 qy > 0 一半" and halves_label("odd") == "odd"
    assert describe("set_halves", {"side": "negative"}) == "Only the qy < 0 half"


def test_a_run_without_report_explains_why() -> None:
    outcome = RunOutcome(RUN_FAILED, "The Claude API rejected the credentials.", [], RunResults(), LlmUsage())
    page = report_html(outcome)
    assert "<h3>No report</h3>" in page and "rejected the credentials" in page


def test_computed_ring_table_reports_the_measured_range_and_conditional_pole_fit():
    from dataclasses import replace
    import numpy as np
    from src.gimap.features.assistant.domain import Profile, ring_orientation

    chi = np.arange(-89.5, 90, 1.)
    profile = Profile.of(chi, 20. + 5. * np.cos(np.deg2rad(chi))**2, np.ones_like(chi))
    ring = ring_orientation(profile, q_window=(0.99, 1.01))
    outcome = _outcome()
    outcome.results.rings.append(replace(
        ring, orientation_settings={"min_exit_deg": .3, "polarization": .99,
                                    "horizon_excluded_deg": [82, 87], "horizon_fraction": .05},
        pole_figure={"ok": True, "quality_reasons": ["reduced χ² exceeds 5"], "herman": .1,
                     "oriented_fraction": .2, "systematics": {"range": [.01, .09], "model_fill_allowed": False,
                         "range_sources": {"min": {"scenario": "horizon gap continued", "threshold_multiplier": .5,
                                                   "background": 12., "horizon_fraction": .05}}}}))
    page = report_html(outcome)
    assert "Pole fit not determined (reduced χ² exceeds 5)" in page
    assert "(stat) [0.0100, 0.0900], f_iso =" in page and "1 − φ is conditional on the background b." in page
    assert "orientation factor is not determined by the measured χ range" in page
    assert "Excluded αf &lt; 0.3°. Polarisation 0.99." in page
    assert "Excluded |χ| 82–87°" in page and "Lower limit from horizon gap continued" in page
    chinese = report_html(outcome, language="中文")
    assert "极图拟合未能确定取向" in chinese and "下限取自按邻近强度补全地平线缺口" in chinese and "|χ| 82–87°" in chinese
    assert "实测 χ 范围不足以确定取向因子" in chinese


def test_final_geometry_uses_only_the_calibration_errors_that_still_apply():
    from src.gimap.features.assistant.application.pipeline_report import _geometry_lines

    geometry = dict(distance_mm=102.6789, beam_center_px=[604.375, 125.773])
    calibration = dict(distance_mm=102.6789, beam_center_px=[602.1, 125.773],
                       uncertainty=dict(distance_mm=0.0123, beam_center_px=[0.235, 0.178]))
    text = "\n".join(_geometry_lines(dict(geometry=geometry, calibration=calibration)))
    assert "distance 102.679 ± 0.012 mm, beam centre (604.38, 125.77 ± 0.18) px" in text
