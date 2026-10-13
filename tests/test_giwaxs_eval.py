"""The two-level evaluation of GIWAXS agents: the baseline must not err, reports are scored.

Every name and number here is invented. Cases that stand on real data live in
``docs/agents/eval/local/`` (ignored by git); these tests need none of them.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from tools.eval_giwaxs_agent import check_baseline, load_case, score_report, sentences

INVENTED_CASE = {
    "name": "Invented: a thin film on glass, two samples",
    "data_root": "D:/data/abc_beamtime",
    "frames": [
        "abc_film_runA_0001/det/abc_film_runA_0001_00001.nxs",
        "abc_film_runB_0002/det/abc_film_runB_0002_00001.nxs",
    ],
    "notes": "GIWAXS, alpha_i = 0.25 deg",
    "baseline": {
        "geometry_source_contains": "abc_standard_lab6_take2",
        "standard": "lab6",
        "max_line_error_percent": 0.2,
        "reliable_peaks": [1.5, 2.2, 3.1],
        "peak_tolerance": 0.01,
        "spike": 4.2,
        "ring": {"q": 2.2, "note_contains": "shadowed", "comment": "ignored by the tool"},
        "no_series_hints": True,
        "open_questions": 0,
        "sector_note_contains": "shadowed",
    },
    "report": {
        "mistakes": [
            {"id": "cubic_series", "all": ["(?i)cubic|立方"], "unless": r"(?i)\bno\b|\bnot\b|没有|不是",
             "why": "The series was built from a halo and an artefact."},
            {"id": "artefact_called_a_peak", "all": [r"4\.2", "(?i)bragg|reflection|衍射峰"],
             "unless": "(?i)artefact|artifact|spike|伪影", "why": "One bin wide, in every sample."},
            {"id": "phase_stated_as_fact", "all": ["(?i)rutile|金红石"],
             "unless": "(?i)hypothes|may|could|consistent|可能|假设", "why": "A phase is a hypothesis."},
        ],
        "baseline_facts": [
            {"id": "calibration_stated", "any": ["(?i)lab6", "六硼化镧"]},
            {"id": "artefact_reported", "all": [r"4\.2", "(?i)artefact|artifact|spike|伪影"]},
        ],
        "findings": [
            {"id": "start_and_end_compared", "points": 2, "all": [r"(?i)first frames|第\s*1\s*帧", "(?i)appear|出现"]},
            {"id": "shared_line", "any": ["(?i)every sample|每个样品"]},
            {"id": "shadow_found", "points": 3, "any": ["(?i)shadow|遮挡"]},
        ],
    },
}


@pytest.fixture
def case(tmp_path: Path) -> dict:
    path = tmp_path / "invented-case.json"
    path.write_text(json.dumps(INVENTED_CASE, ensure_ascii=False, indent=2), encoding="utf-8")
    return load_case(path)


def _report(run: str, **changes) -> dict:
    report = {
        "ok": True,
        "frame": f"D:/data/abc_beamtime/abc_film_{run}/det/abc_film_{run}_00001.nxs",
        "geometry": {"source": "D:/data/abc_beamtime/calib/abc_standard_lab6_take2_00001.nxs"},
        "calibration_quality": {"standard": "lab6", "mean_line_q_error_percent": 0.08},
        "peaks": [
            {"q": 1.2, "caveat": "a broad halo …", "flags": ["broad"],
             "orientation": "only the out-of-plane sector is usable here: the other is shadowed (peak seen)"},
            {"q": 1.503, "caveat": "", "flags": []},
            {"q": 2.19, "caveat": "", "flags": []},
            {"q": 3.11, "caveat": "", "flags": []},
            {"q": 4.2, "caveat": "a spike …", "flags": ["resolution_limited", "spike"]},
        ],
        "rings": [{"q": 2.19, "herman": None, "reason": "Only a part of the orientation range …",
                   "notes": ["|χ| above the wedge is shadowed: …"]}],
        "series_hints": [],
        "needs_attention": [],
    }
    report.update(changes)
    return report


def test_the_first_level_checks_what_the_baseline_must_get_right(case: dict) -> None:
    reports = [_report("runA_0001"), _report("runB_0002")]
    checks = check_baseline(reports, case["baseline"])
    assert all(passed for _name, passed, _detail in checks)
    assert [name for name, _passed, _detail in checks] == [
        "every frame analysed", "calibration file", "standard", "line error", "reliable peak near 1.5",
        "reliable peak near 2.2", "reliable peak near 3.1", "spike at 4.2 flagged", "no false series",
        "ring near q = 2.2", "sectors say 'shadowed'", "open questions",
    ]

    broken = copy.deepcopy(reports)
    broken[1]["peaks"][4]["flags"] = ["resolution_limited"]  # the spike taken for a peak
    broken[1]["series_hints"] = ["q = 1.2, 2.4, 4.2 Å⁻¹ follow a cubic series"]
    broken[0]["calibration_quality"]["mean_line_q_error_percent"] = 0.35
    broken[0]["rings"] = [{"q": 2.19, "herman": 0.5, "notes": []}]  # f faked by the shadow
    broken[0]["peaks"][0]["orientation"] = "mainly out-of-plane"
    failed = {name: detail for name, passed, detail in check_baseline(broken, case["baseline"]) if not passed}
    assert set(failed) == {"spike at 4.2 flagged", "no false series", "line error", "ring near q = 2.2", "sectors say 'shadowed'"}
    assert failed["spike at 4.2 flagged"] == "not flagged in abc_film_runB_0002_00001.nxs"  # the frame is named
    assert failed["sectors say 'shadowed'"] == "not in abc_film_runA_0001_00001.nxs"


def test_every_other_fact_of_the_baseline_is_checked_too(case: dict) -> None:
    def failed(*reports: dict) -> set[str]:
        return {name for name, passed, _detail in check_baseline(list(reports), case["baseline"]) if not passed}

    good = _report("runA_0001")
    assert failed(good, {"ok": False, "frame": "x/unreadable.nxs", "error": "…", "needs_attention": []}) >= {
        "every frame analysed", "calibration file", "standard", "line error", "reliable peak near 1.5"}
    assert failed(good, _report("runB_0002", geometry={"source": "another_beamtime/standard.poni"})) == {"calibration file"}
    other = _report("runB_0002", calibration_quality={"standard": "agbh", "mean_line_q_error_percent": 0.08})
    assert failed(good, other) == {"standard"}
    assert failed(_report("runB_0002", calibration_quality={"standard": "lab6"})) == {"line error"}  # no number: not a pass
    assert failed(_report("runB_0002", needs_attention=[{"question": "which energy?"}])) == {"open questions"}
    caveated = _report("runB_0002")
    caveated["peaks"][2]["caveat"] = "weak: tentative"  # a peak with a warning is not a reliable one
    assert failed(caveated) == {"reliable peak near 2.2"}
    shifted = _report("runB_0002")
    shifted["peaks"][3]["q"] = 3.1 * 1.02  # outside the 1 % the case allows
    assert failed(shifted) == {"reliable peak near 3.1"}
    assert failed(_report("runB_0002", rings=[])) == {"ring near q = 2.2"}
    unexplained = _report("runB_0002", rings=[{"q": 2.19, "herman": None, "notes": [], "reason": "too few points"}])
    assert failed(unexplained) == {"ring near q = 2.2"}  # withheld, but not for the reason the case names


def test_a_ring_with_a_range_needs_an_orientation_factor_inside_it() -> None:
    expected = {"standard": "lab6", "ring": {"q": 2.2, "f_min": 0.3, "f_max": 0.6}}

    def ring_passes(**ring) -> bool:
        checks = check_baseline([_report("runA_0001", rings=[{"q": 2.21, "notes": [], **ring}])], expected)
        return dict((name, passed) for name, passed, _detail in checks)["ring near q = 2.2"]

    assert ring_passes(herman=0.45) and ring_passes(herman=0.3) and ring_passes(herman=0.6)
    assert not ring_passes(herman=0.7) and not ring_passes(herman=None) and not ring_passes(herman=0.45, q=2.3)
    # Only the keys a case gives are checked, apart from the facts every case fixes.
    names = [name for name, _passed, _detail in check_baseline([_report("runA_0001")], {"standard": "lab6"})]
    assert names == ["every frame analysed", "calibration file", "standard", "line error"]


def test_reports_are_scored_for_mistakes_facts_and_findings(case: dict) -> None:
    weak = (
        "The calibration used LaB6. Peaks at 1.2, 2.4 and 4.2 follow a cubic series. The 4.2 peak is a sharp "
        "Bragg reflection. The film contains rutile."
    )
    score = score_report(weak, case["report"])
    assert set(score["mistakes"]) == {"cubic_series", "artefact_called_a_peak", "phase_stated_as_fact"}
    assert score["mistakes"]["artefact_called_a_peak"] == ["The 4.2 peak is a sharp Bragg reflection."]
    assert score["why"]["phase_stated_as_fact"] == "A phase is a hypothesis."
    assert score["points"] == 0 and score["max_points"] == 6
    assert score["facts"] == {"calibration_stated": True, "artefact_reported": False}

    careful = (
        "There is no cubic series. 4.2 is a spike in every sample, an artefact, not a Bragg reflection. "
        "Comparing the first frames with the last, 2.2 and 3.1 appear during the run. "
        "Hypothesis: the 2.2 line may be rutile."
    )
    score = score_report(careful, case["report"])
    assert score["mistakes"] == {}  # negated and hedged sentences are not mistakes
    assert score["facts"] == {"calibration_stated": False, "artefact_reported": True}
    assert score["findings"] == {"start_and_end_compared": True, "shared_line": True, "shadow_found": False}
    assert score["points"] == 3 and score["max_points"] == 6  # 2 + 1 (the default); no word on the shadow

    chinese = "标样是六硼化镧。第 1 帧没有 2.2，之后才出现；没有立方序列。高角一侧被遮挡。4.2 处是一个衍射峰。"
    score = score_report(chinese, case["report"])
    assert set(score["mistakes"]) == {"artefact_called_a_peak"} and score["facts"]["calibration_stated"]
    assert score["findings"]["start_and_end_compared"] and score["findings"]["shadow_found"] and score["points"] == 5


def test_a_rule_is_matched_inside_one_sentence() -> None:
    assert sentences("One line; still the same sentence. Next one.\nA third\n\n第四句。第五句！") == [
        "One line; still the same sentence.", "Next one.", "A third", "第四句。", "第五句！"]
    assert sentences("q = 2.2 Å⁻¹, f = 0.45 and so on") == ["q = 2.2 Å⁻¹, f = 0.45 and so on"]  # decimals do not split
    rubric = {"mistakes": [{"id": "both", "all": ["alpha", "beta"]}], "findings": [{"id": "either", "any": ["alpha", "beta"]}]}
    apart = score_report("Only alpha here. Only beta there.", rubric)
    assert apart["mistakes"] == {} and apart["findings"] == {"either": True} and apart["points"] == 1
    assert score_report("Here alpha meets beta.", rubric)["mistakes"] == {"both": ["Here alpha meets beta."]}
    assert score_report("anything", {}) == {"mistakes": {}, "facts": {}, "findings": {}, "points": 0, "max_points": 0, "why": {}}
