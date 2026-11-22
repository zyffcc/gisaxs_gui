"""What the GIWAXS automatic analysis says about its own numbers, on the public P08 frame (tests/data/external).

The run is the one Run Automatic Analysis and ``tools/gimap_agent.py auto`` start: an offscreen Analyze page with
the standard pipeline on it. An αi typed in Analyze's command bar is the αi Analyze uses, so the run takes it and
does not ask for αi. The report states the peak cap, the q convention and the statistical error model used
for this floating-point frame. For the ring orientation it gives the exit-angle cut and a systematic range for f.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import pytest

from src.gimap.features.assistant.application import PipelineOptions, batch_markdown, pipeline_markdown
from src.gimap.features.assistant.application.pipeline import FROM_COMMAND_BAR
from src.gimap.features.assistant.application.pipeline_geometry import INCIDENCE_ITEM

P08 = Path(__file__).parent / "data" / "external" / "giwaxs_p08_mapi"
FRAME, PONI = "S121_MAI_A2_00841.tif", "LaB6_2021_12_DESY_P08.poni"

pytestmark = pytest.mark.skipif(not (P08 / FRAME).is_file(), reason="the public P08 frame is not in this checkout")


@pytest.fixture
def p08(tmp_path: Path):
    """An offscreen Analyze page on a copy of the P08 frame (nothing is written next to the public data)."""
    from src.gimap.app.headless_assistant import open_headless_session

    data = tmp_path / "data"
    data.mkdir()
    for name in (FRAME, PONI):
        shutil.copy2(P08 / name, data / name)
    session = open_headless_session(data / FRAME, saved_profiles=False, out_dir=tmp_path / "out")
    yield session, data / PONI
    session.close()


def _decision(report: dict, what: str) -> dict:
    return next(item for item in report["decisions"] if item["what"] == what)


def _alpha_asked(report: dict) -> bool:
    return any(item["item"] == INCIDENCE_ITEM for item in report["needs_attention"])


@pytest.mark.parametrize("how", ["command bar", "set_incidence_angle"])
def test_an_alpha_i_set_in_analyze_is_used_and_not_asked_for(p08, how: str) -> None:
    session, poni = p08
    if how == "command bar":  # typed into the αi field of Analyze's command bar before Run Automatic Analysis
        session.page.incidence_spin.setValue(0.075)
        session.page.tasks.wait(60)
    else:  # an agent's earlier step (MCP, gimap_agent.py call)
        assert not session.call("set_incidence_angle", {"degrees": 0.075}).is_error
    assert session.page.view_model.state.incidence_deg == pytest.approx(0.075)

    report = session.run_pipeline(PipelineOptions(calibration=str(poni), follow_detection=True))

    assert report["ok"] and report["procedure"] == "giwaxs"
    assert report["geometry"]["incidence_deg"] == pytest.approx(0.075)  # what the analysis used …
    assert not _alpha_asked(report)  # … so αi is not asked for, nor "0° is used" said
    alpha = _decision(report, "incidence angle")
    assert alpha["decision"] == "αi = 0.075°" and alpha["why"] == FROM_COMMAND_BAR
    assert "0° is used" not in pipeline_markdown(report)


def test_an_alpha_i_given_replaces_the_one_in_the_command_bar_and_is_the_one_used(p08) -> None:
    session, poni = p08
    session.page.incidence_spin.setValue(0.075)
    session.page.tasks.wait(60)

    report = session.run_pipeline(PipelineOptions(calibration=str(poni), incidence_deg=0.1, follow_detection=True))

    assert report["geometry"]["incidence_deg"] == pytest.approx(0.1)  # before, Analyze kept using 0.075°
    assert _decision(report, "incidence angle")["why"] == "from given; it replaces the 0.075° set in Analyze's command bar"
    assert not _alpha_asked(report)


def test_the_report_says_what_its_numbers_are_and_where_the_list_was_cut(p08) -> None:
    session, poni = p08
    report = session.run_pipeline(PipelineOptions(calibration=str(poni), incidence_deg=0.075, follow_detection=True))

    q = _decision(report, "q")
    assert q["decision"] == "kinematic (no refraction correction)"
    assert "k·sin αi = 0.0119 Å⁻¹ here (αi = 0.075°, λ = 0.6888 Å)" in q["why"]  # 2π/λ · sin αi
    assert report["error_model"] == "pixel_scatter"  # a floating-point frame
    errors = _decision(report, "uncertainties")
    assert errors["decision"] == "statistical ± and systematic range for f"
    assert "scatter of the corrected pixels" in errors["why"]
    assert "pixel-scatter uncertainty, not counting noise" in errors["why"]
    assert "pooled variance for points containing fewer than ten pixels" in errors["why"]

    search = report["peak_search"]
    assert search["max_peaks"] == 12 and search["truncated"] and search["candidates"] > 12
    # The I(q) points searched (it said "bins"), and the radial bins of the reduction as <stem>_analysis.json says.
    assert "bins" not in search and search["radial_bins"] >= search["points"] > 1000
    cap = _decision(report, "peak search")
    assert re.fullmatch(r"\d+ candidate peaks found; the 12 most prominent were fitted, \d+ of them gave a peak",
                        cap["decision"])
    assert str(search["candidates"]) in cap["decision"] and "--max-peaks N" in cap["why"]
    text = pipeline_markdown(report)
    assert f"Series hints (reliable peaks among the 12 most prominent of {search['candidates']} candidates only)" in text
    summary = batch_markdown([report])  # summary.md of the command line
    assert "Peak lists cut at the cap" in summary and f"{FRAME} (12 of {search['candidates']})" in summary

    rule = _decision(report, "texture rule")  # the margin names "along the normal" or "in the plane", not "tilted"
    assert rule["decision"].endswith("|f − f_iso| ≥ max(0.1, 3σ_f)") and "without this margin" in rule["why"]
    horizon = _decision(report, "ring orientation")
    assert "αf < 0.15° (2 × αi)" in horizon["decision"] and "factor 0.99" in horizon["decision"]
    ring = min(report["rings"], key=lambda r: abs(r["q"] - 0.997))
    assert ring["orientation_settings"]["error_model"] == "pixel_scatter"
    pole = ring["pole_figure"]
    low, high = ring["herman_systematic_range"]
    assert low <= ring["herman"] <= high
    assert ring["herman"] == pytest.approx(pole["herman_measured"])
    assert {s["threshold_multiplier"] for s in pole["systematics"]["scenarios"]} == {0.5, 1.0, 2.0}
    assert pole["systematics"]["background_rule"] == "fitted sigma"
    assert "reduced χ² exceeds 5" in pole["quality_reasons"]
    assert any("0.997" in item["why"] and "not determined" in item["why"] for item in report["needs_attention"])
    assert any("pole fit not determined. Measured f =" in line for line in text.splitlines() if "q ≈ 0.997" in line)
    assert "1 − φ is conditional on the background b" in text and "(stat) [" in text


def test_more_peaks_are_fitted_when_asked_and_a_clean_one_left_out_by_the_cap_is_listed(p08) -> None:
    session, poni = p08
    report = session.run_pipeline(
        PipelineOptions(calibration=str(poni), incidence_deg=0.075, follow_detection=True, max_peaks=40, orientation_min_exit_deg=0.35),
    )
    assert report["rings"][0]["orientation_settings"]["min_exit_deg"] == 0.35
    found = [peak["q"] for peak in report["peaks"]]
    assert len(found) > 12 and any(abs(q - 3.585) < 0.01 for q in found)  # SNR 33, dropped by the default cap
    assert report["peak_search"]["max_peaks"] == 40
    assert any(step["tool"] == "find_peaks" and step["arguments"].get("max_peaks") == 40 for step in report["steps"])
