"""The two-level evaluation of GIWAXS agents: the baseline must not err, reports are scored."""

from __future__ import annotations

import copy

from tools.eval_giwaxs_agent import check_baseline, load_case, score_report

CASE = load_case()


def _report(frame: str, **changes) -> dict:
    report = {
        "ok": True,
        "frame": f"E:/data/{frame}_00001_m01.nxs",
        "geometry": {"source": "giwaxs_calib_lab6_ceo2_redone_final_00001_00002_m01.nxs"},
        "calibration_quality": {"standard": "lab6_ceo2", "mean_line_q_error_percent": 0.056},
        "peaks": [
            {"q": 1.928, "caveat": "a broad halo …", "flags": ["broad"],
             "orientation": "only the out-of-plane sector is usable here: the other is shadowed (peak seen)"},
            {"q": 2.557, "caveat": "", "flags": []},
            {"q": 2.948, "caveat": "", "flags": []},
            {"q": 3.412, "caveat": "", "flags": []},
            {"q": 4.893, "caveat": "a spike …", "flags": ["resolution_limited", "spike"]},
        ],
        "rings": [{"q": 2.948, "herman": None, "reason": "Only 40% of the orientation range …",
                   "notes": ["|χ| 55–89° is shadowed: …"]}],
        "series_hints": [],
        "needs_attention": [],
    }
    report.update(changes)
    return report


def test_the_first_level_checks_what_the_baseline_must_get_right() -> None:
    reports = [_report("116"), _report("117")]
    assert all(passed for _name, passed, _detail in check_baseline(reports, CASE["baseline"]))

    broken = copy.deepcopy(reports)
    broken[1]["peaks"][4]["flags"] = ["resolution_limited"]  # the spike taken for a peak
    broken[1]["series_hints"] = ["q = 1.864, 3.723, 4.893 Å⁻¹ follow a hexagonal series"]
    broken[0]["calibration_quality"]["mean_line_q_error_percent"] = 0.3
    broken[0]["rings"] = [{"q": 2.948, "herman": 0.42, "notes": []}]  # f faked by the shadow
    broken[0]["peaks"][0]["orientation"] = "mainly out-of-plane"
    failed = {name for name, passed, _detail in check_baseline(broken, CASE["baseline"]) if not passed}
    assert failed == {"spike at 4.893 flagged", "no false series", "line error", "ring near q = 2.95", "sectors say 'shadowed'"}


def test_reports_are_scored_for_mistakes_facts_and_findings() -> None:
    weak = (
        "The calibration used LaB6 + CeO2, lines within 0.06 %. alpha_i = 0.4 deg from the notes. "
        "Peaks at 1.655, 2.557 and 4.893 follow a hexagonal series. The 4.893 peak is a sharp Bragg "
        "reflection. The ring at 1.655 has Herman f = 0.12. The film contains Cu2O."
    )
    score = score_report(weak, CASE["report"])
    assert set(score["mistakes"]) == {"hexagonal_series", "spike_as_reflection", "herman_from_13_percent", "phase_as_fact"}
    assert score["points"] == 0 and score["facts"]["calibration_named"] and not score["facts"]["spike_called_artefact"]

    careful = (
        "There is no hexagonal series. 4.893 is a spike in every sample, an artefact, not a Bragg reflection. "
        "Only 13 % of χ is measured at 1.655, so no Herman f. The 2.94 ring sits next to the missing wedge. "
        "Comparing frame 1 with the last frames, 2.94 and 3.41 appear during deposition. "
        "3.89 is in every sample, possibly from the substrate. "
        "Hypothesis: 3.408/2.942 = 1.158, the fcc (111)/(200) ratio; 2.557 may be Cu2O (111)."
    )
    score = score_report(careful, CASE["report"])
    assert score["mistakes"] == {}
    assert score["points"] == 6 and score["max_points"] == 8  # no word on the shadow

    # Sentences from the real Claude Code run (sample 117), which the first patterns missed or misread.
    real = (
        "7 条标样线的 q 平均偏差 0.056 %，拟合良好。"
        "第 1–10 帧：只有 1.649、1.884、2.562 和 3.899；没有 2.94 和 3.41 峰。"
        "3.88–3.90 Å⁻¹ 和 2.56 Å⁻¹ 从第 1 帧起就存在，也可能来自基底或样品台。"
        "χ ≳ 57° 的区域应是被遮挡或响应很低的探测器区域。"
        "用标样或基底的峰校正样品处的距离，再判断 2 % 的晶格偏差是否真实，并区分 Cu 与 Cu2O。"
    )
    score = score_report(real, CASE["report"])
    assert score["mistakes"] == {} and score["facts"]["line_error"]
    assert score["findings"]["series_change"] and score["findings"]["shared_lines"] and score["findings"]["shadowed_region"]
