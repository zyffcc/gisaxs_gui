"""Release 0.1.0, the automatic analysis without a window: what a report says and how a batch is summed up.

A calibration nobody named is said to be found by the search, with its full path and where it lies (no
question, no other exit code); a batch summary is titled by what ran and holds the GISAXS frames' results;
the question about a missing αi is worded for the technique that ran; the fit solutions table says its units.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from src.gimap.features.assistant.application import (
    PipelineOptions,
    StandardPipeline,
    batch_markdown,
    fit_solutions_csv,
    found_by_search,
    pipeline_markdown,
)
from src.gimap.features.assistant.application.pipeline_geometry import INCIDENCE_MISSING, SEARCHED
from src.gimap.features.assistant.application.pipeline_report import technique_run
from tests.test_assistant_calibration import POOR_AGBH, touch
from tests.test_assistant_pipeline import setup

# -- a calibration the search found (R3, R78) -------------------------------------------------------------


def _searched(report: dict) -> list[dict]:
    return [item for item in report["decisions"] if item["what"] == SEARCHED]


def _with_distance(report: dict) -> dict:
    """The report with a detector distance: the tests' workbench reports a geometry without one, and the
    Geometry section of report.md is written only for a geometry that has it."""
    return {**report, "geometry": {**report["geometry"], "distance_mm": 300.0, "beam_center_px": [512.0, 900.0]}}


def test_a_calibration_found_in_a_sibling_folder_is_said_to_be_searched_and_asks_nothing(tmp_path: Path) -> None:
    from tools import gimap_agent

    catalog, _workbench, _calibrator = setup(tmp_path)
    report = StandardPipeline(catalog, PipelineOptions(energy_kev=12.4, incidence_deg=0.2)).run()

    assert report["ok"] and report["needs_attention"] == []  # information only: nothing more is asked
    assert gimap_agent.exit_code([report]) == gimap_agent.EXIT_DONE
    (said,) = _searched(report)
    image = tmp_path / "beamtime" / "raw" / "calib" / "AgBH_00001.tif"
    assert Path(said["decision"]).is_absolute() and Path(said["decision"]).samefile(image)  # the full path
    assert said["why"] == "sibling folder calib, seen from the frame's folder"
    # The sentences other code and tests read are as they were, the new line comes after them.
    geometry = [item for item in report["decisions"] if item["what"] == "geometry"]
    assert geometry[0]["decision"] == "calibrated from AgBH_00001.tif" and geometry[0]["why"].startswith("good")
    assert report["decisions"].index(said) > report["decisions"].index(geometry[0])
    assert found_by_search(report) is said
    text = pipeline_markdown(_with_distance(report))
    found = next(line for line in text.splitlines() if line.startswith("- found by the automatic search"))
    assert said["decision"] in found and "sibling folder calib" in found
    assert text.index("- source: AgBH_00001.tif") < text.index(found) < text.index("- quality: good")
    assert f"- **{SEARCHED}**: {said['decision']} — sibling folder calib" in text  # and among the Decisions


def test_a_calibration_file_three_folders_up_is_adopted_and_the_report_says_from_where(tmp_path: Path) -> None:
    """The case of the release audit: a ready .poni of another folder tree is used as saved; the person at
    least reads where it came from (it is still no question: the search is what they asked for)."""
    catalog, _workbench, calibrator = setup(tmp_path)
    poni = touch(tmp_path / "data" / "saxs" / "agbh.poni", POOR_AGBH)
    report = StandardPipeline(catalog, PipelineOptions(energy_kev=10.0, incidence_deg=0.2)).run()

    assert report["ok"] and calibrator.calls == [] and report["needs_attention"] == []
    (said,) = _searched(report)
    assert Path(said["decision"]).samefile(poni)
    assert said["why"] == "folder data three levels up, subfolder data/saxs, seen from the frame's folder"
    assert report["calibration_quality"]["assessment"].startswith("a saved calibration file")
    assert "found by the automatic search, named by nobody" in pipeline_markdown(_with_distance(report))


def test_a_calibration_the_person_named_or_a_kept_profile_is_not_called_searched(tmp_path: Path) -> None:
    from src.gimap.features.assistant.application import GOALS, PERMISSION_AUTO, AnalysisGoals, RunResults, ToolCatalog
    from tests.assistant_fakes import FakeWorkbench

    catalog, _workbench, _calibrator = setup(tmp_path)
    poni = touch(tmp_path / "data" / "saxs" / "agbh.poni", POOR_AGBH)
    named = StandardPipeline(catalog, PipelineOptions(calibration=str(poni), energy_kev=10.0, incidence_deg=0.2)).run()
    assert named["ok"] and _searched(named) == [] and found_by_search(named) is None
    assert "found by the automatic search" not in pipeline_markdown(_with_distance(named))

    noted, _workbench, _calibrator = setup(tmp_path / "noted")  # named in the notes: the person's own too
    image = tmp_path / "noted" / "beamtime" / "raw" / "calib" / "AgBH_00001.tif"
    notes = f"12.4 keV, alpha_i = 0.2 deg, calibration {image}"
    by_notes = StandardPipeline(noted, PipelineOptions(notes=notes)).run()
    assert by_notes["ok"] and _searched(by_notes) == []

    kept = ToolCatalog(FakeWorkbench(), AnalysisGoals(goals=tuple(GOALS), permission=PERMISSION_AUTO), RunResults())
    profile = StandardPipeline(kept).run()
    assert profile["ok"] and _searched(profile) == []


def test_later_frames_of_a_batch_and_its_summary_say_that_the_calibration_was_searched(tmp_path: Path) -> None:
    first, _workbench, _calibrator = setup(tmp_path / "a")
    pipeline = StandardPipeline(first, PipelineOptions(energy_kev=12.4, incidence_deg=0.2))
    one = pipeline.run()
    assert pipeline.calibration["found_by_search"]["frame"] == "film_00001_m01.nxs"
    second, _workbench, calibrator = setup(tmp_path / "b")
    two = StandardPipeline(second, PipelineOptions(energy_kev=12.4, incidence_deg=0.2, geometry=pipeline.calibration)).run()

    assert two["ok"] and calibrator.calls == [] and two["needs_attention"] == []
    (again,) = _searched(two)
    assert again["decision"] == _searched(one)[0]["decision"]
    assert again["why"] == ("sibling folder calib, seen from the folder of film_00001_m01.nxs, "
                            "the frame it was searched for")
    summary = batch_markdown([one, two])
    notes = [line for line in summary.splitlines() if line.startswith("Calibration found by the automatic search")]
    assert len(notes) == 1 and again["decision"] in notes[0] and "2 frame(s)" in notes[0]
    assert "Open questions" not in summary  # said, not asked


# -- the summary of a batch (R11, R77) --------------------------------------------------------------------

ALPHA_GIWAXS = {"item": "incidence angle αi", "why": INCIDENCE_MISSING["giwaxs"], "option": "incidence_deg", "hint": ""}
ALPHA_GISAXS = {"item": "incidence angle αi", "why": INCIDENCE_MISSING["gisaxs"], "option": "incidence_deg", "hint": ""}


def _giwaxs(name: str = "film.tif", **extra) -> dict:
    report = {
        "ok": True, "procedure": "giwaxs", "frame": name, "geometry": {"source": "agbh.poni"},
        "peaks": [{"q": 0.899}, {"q": 1.409, "caveat": "weak: barely above the noise"}],
        "rings": [{"q": 0.899, "herman": 0.61}], "needs_attention": [],
        "decisions": [{"what": "technique", "decision": "GIWAXS", "why": "detected"}],
    }
    return {**report, **extra}


def _gisaxs(name: str = "dots.tif", **extra) -> dict:
    report = {
        "ok": True, "procedure": "gisaxs", "frame": name, "geometry": {"source": "galaxi.poni"}, "needs_attention": [],
        "decisions": [{"what": "technique", "decision": "GISAXS", "why": "detected"}],
        "gisaxs": {
            "halves": {"side": "mean"}, "spacing": {"q": 0.0117, "distance_nm": 53.5, "kind": "shoulder"},
            "fit": {"solutions": [{"model": "sphere", "chi2": 1.31, "components": [{"R": 9.4, "D": 53.4}]}]},
        },
    }
    return {**report, **extra}


NO_GEOMETRY = {
    "ok": False, "procedure": "giwaxs", "frame": "lonely.tif", "geometry": {"source": ""},
    "decisions": [{"what": "technique", "decision": "none", "why": "no geometry"}],
    "needs_attention": [{"item": "calibration", "why": "No calibration candidate gave a good geometry.",
                         "option": "calibration", "hint": ""}],
}
UNREADABLE = {"ok": False, "frame": "missing.tif", "error": "No such file: missing.tif", "needs_attention": []}


def test_a_batch_summary_is_titled_by_the_procedure_that_ran() -> None:
    assert [technique_run(report) for report in (_giwaxs(), _gisaxs(), NO_GEOMETRY, UNREADABLE)] == [
        "giwaxs", "gisaxs", None, None]
    assert batch_markdown([_giwaxs(), _giwaxs("b.tif")]).startswith("# GIWAXS — batch\n")
    assert batch_markdown([_gisaxs(), _gisaxs("b.tif")]).startswith("# GISAXS — batch\n")
    assert batch_markdown([_giwaxs(), _gisaxs()]).startswith("# GIMaP — batch\n")
    # Nothing got a technique (no geometry, a file that cannot be read): neither GIWAXS nor GISAXS.
    nothing = batch_markdown([NO_GEOMETRY, UNREADABLE])
    assert nothing.startswith("# GIMaP — batch\n") and "peaks q" not in nothing and "##" not in nothing.split("## 需要处理")[0]
    assert "| lonely.tif | needs input | — | calibration |" in nothing
    assert "| missing.tif | error: No such file: missing.tif | — | — |" in nothing
    assert "- **calibration** — No calibration candidate gave a good geometry. → `--calibration PATH`" in nothing
    # One procedure ran: the frames it did not run on are rows of its table, as before.
    one = batch_markdown([NO_GEOMETRY, _giwaxs(), UNREADABLE])
    assert one.startswith("# GIWAXS — batch\n") and "##" not in one.split("## 需要处理")[0]
    assert "| lonely.tif | needs input | 1 | — | — | — | calibration |" in one
    assert "| film.tif | OK | 1 | agbh.poni | 0.899, 1.409* | 0.899: 0.61 | — |" in one
    assert batch_markdown([_gisaxs(), UNREADABLE]).startswith("# GISAXS — batch\n")


def test_a_mixed_batch_keeps_the_results_of_its_gisaxs_frames() -> None:
    reports = [_giwaxs(needs_attention=[ALPHA_GIWAXS]), _gisaxs(needs_attention=[ALPHA_GISAXS]), UNREADABLE]
    text = batch_markdown(reports)
    assert text.startswith("# GIMaP — batch\n\n3 frame(s); each has its own report.md.\n")
    giwaxs, gisaxs, rest = text.index("## GIWAXS"), text.index("## GISAXS"), text.index("## 未分析 / Not analysed")
    assert giwaxs < gisaxs < rest < text.index("Open questions")
    assert "| film.tif | needs input | 1 | agbh.poni | 0.899, 1.409* | 0.899: 0.61 | incidence angle αi |" in text[giwaxs:gisaxs]
    # Halves, spacing and best fit of the GISAXS frame: they were dropped from a table of GIWAXS columns.
    row = "| dots.tif | needs input | mean | 53.5 (shoulder) | sphere R 9.4, D 53.4 nm, χ² 1.31 | incidence angle αi |"
    assert row in text[gisaxs:rest]
    assert "| missing.tif | error: No such file: missing.tif | — | — |" in text[rest:]
    # αi is asked once for each procedure, in its words, with the flag that answers it.
    asked = [line for line in text.splitlines() if line.startswith("- **incidence angle αi**")]
    assert len(asked) == 2 and "missing wedge" in asked[0] and "exit angle αf" in asked[1]
    assert all(line.endswith("→ `--incidence-deg A`") for line in asked)


def test_a_technique_asked_for_but_never_started_is_not_what_ran(tmp_path: Path) -> None:
    """``--technique gisaxs`` on a frame no calibration is found for: nothing ran, so neither the frame's report
    nor the summary is titled by a technique (they said GIWAXS: the procedure's name until one starts)."""
    from src.gimap.features.assistant.application import GOALS, PERMISSION_AUTO, AnalysisGoals, RunResults, ToolCatalog
    from src.gimap.features.assistant.infrastructure import LocalFileExplorer
    from tests.test_assistant_calibration import FakeCalibrator
    from tests.test_assistant_pipeline import SeriesWorkbench

    frame = touch(tmp_path / "alone" / "a" / "b" / "c" / "dots_00001_m01.nxs", "x")  # nothing for the search to find
    for technique in ("gisaxs", "giwaxs", None):
        catalog = ToolCatalog(SeriesWorkbench(frame), AnalysisGoals(goals=tuple(GOALS), permission=PERMISSION_AUTO),
                              RunResults(), explorer=LocalFileExplorer(), calibrator=FakeCalibrator())
        report = StandardPipeline(catalog, PipelineOptions(energy_kev=12.4, technique=technique)).run()
        assert not report["ok"] and [item["item"] for item in report["needs_attention"]] == ["calibration"]
        assert technique_run(report) is None, technique
        text = pipeline_markdown(report)
        assert text.startswith("# GIMaP — dots_00001_m01.nxs") and "Scherrer" not in text and "The fit compares" not in text
        summary = batch_markdown([report])
        assert summary.startswith("# GIMaP — batch\n") and "peaks q" not in summary and "halves" not in summary
        assert "| dots_00001_m01.nxs | needs input | — | calibration |" in summary
    # A procedure that started and then failed (the frame changed) is still what ran.
    changed = {**NO_GEOMETRY, "failed": "The frame changed during the run", "decisions": [
        {"what": "technique", "decision": "GISAXS", "why": "forced"}], "procedure": "gisaxs"}
    assert technique_run(changed) == "gisaxs" and batch_markdown([changed]).startswith("# GISAXS — batch\n")


def test_frames_of_one_procedure_share_one_question() -> None:
    text = batch_markdown([_gisaxs(needs_attention=[ALPHA_GISAXS]), _gisaxs("b.tif", needs_attention=[ALPHA_GISAXS])])
    assert text.count("- **incidence angle αi**") == 1 and "→ `--incidence-deg A`" in text


# -- the αi question in the words of the technique (R11) ---------------------------------------------------


def test_a_giwaxs_run_asks_for_alpha_i_with_what_it_changes_for_rings(tmp_path: Path) -> None:
    catalog, _workbench, _calibrator = setup(tmp_path)
    report = StandardPipeline(catalog, PipelineOptions(energy_kev=12.4)).run()
    (question,) = [item for item in report["needs_attention"] if item["option"] == "incidence_deg"]
    assert question["why"] == INCIDENCE_MISSING["giwaxs"] and "missing wedge" in question["why"]


@pytest.fixture(scope="module")
def galaxi():
    from src.gimap.app.headless_assistant import open_headless_session
    from tests.test_guided_gisaxs import DATA, GALAXI

    session = open_headless_session(DATA / GALAXI["frame"], saved_profiles=False)
    yield session
    session.close()


def test_a_gisaxs_run_asks_for_alpha_i_with_what_it_changes_for_the_cut(galaxi) -> None:
    from tests.test_guided_gisaxs import _geometry

    report = galaxi.run_pipeline(PipelineOptions(technique="gisaxs", geometry=_geometry(), fit=False))
    assert report["ok"] and report["procedure"] == "gisaxs"
    asked = [item for item in report["needs_attention"] if item["option"] == "incidence_deg"]
    alpha = next(item for item in asked if item["item"] == "incidence angle αi")
    assert alpha["why"] == INCIDENCE_MISSING["gisaxs"]
    for word in ("Ring positions", "missing wedge"):  # GIWAXS words mean nothing here
        assert word not in alpha["why"]
    assert "0° is used" in alpha["why"] and "exit angle αf" in alpha["why"] and "too large by αi" in alpha["why"]
    text = pipeline_markdown(report)
    assert text.startswith("# GISAXS") and "too large by αi" in text.split("## 几何 / Geometry")[0]
    assert "→ `--incidence-deg A`" in text and "missing wedge" not in text


def test_find_geometry_words_the_question_for_what_auto_reads_the_frame_as(galaxi) -> None:
    from tests.test_guided_gisaxs import _geometry

    report = galaxi.run_pipeline(PipelineOptions(geometry=_geometry(), stop_after_geometry=True))
    assert report["ok"] and report["procedure"] == "geometry" and report["measurement"] == "gisaxs"
    (alpha,) = [item for item in report["needs_attention"] if item["item"] == "incidence angle αi"]
    assert alpha["why"] == INCIDENCE_MISSING["gisaxs"]


# -- the solutions table says what it is and its units (R79) -----------------------------------------------


def test_the_fit_solutions_table_starts_with_what_it_is_and_the_units() -> None:
    fit = {"curve": "mean of both halves up to |qy| = 0.179 Å⁻¹", "solutions": [
        {"rank": 1, "model": "sphere", "chi2": 1.31, "log_rmse": 0.15, "converged": True,
         "components": [{"type": "sphere", "R": 9.4, "sigma_R": 0.3, "D": 53.4, "sigma_D": 0.4, "weight": 1.0}]},
        {"rank": 2, "model": "cylinder", "chi2": 1.6, "log_rmse": 0.2, "converged": False,
         "components": [{"type": "cylinder", "R": 6.1, "sigma_R": 0.2, "h": 12.0, "sigma_h": 0.1, "weight": 1.0}]},
    ]}
    lines = fit_solutions_csv(fit).splitlines()
    assert lines[0] == "# GISAXS fit solutions [mean of both halves up to |qy| = 0.179 Å⁻¹]: one line per component of each solution"
    assert lines[1].startswith("# units: R, h, D in nm; sigma_R, sigma_h, sigma_D are relative widths")
    assert not lines[2].startswith("#") and len(lines) == 5
    # A reader that skips the comment lines gets the same table as before.
    rows = list(csv.DictReader(line for line in lines if not line.startswith("#")))
    assert [row["model"] for row in rows] == ["sphere", "cylinder"] and set(rows[0]) >= {"R", "D", "sigma_R", "h", "chi2"}
    assert float(rows[0]["R"]) == 9.4 and rows[0]["h"] == "" and float(rows[1]["h"]) == 12.0
    assert fit_solutions_csv({"curve": "a\nb", "solutions": []}).splitlines()[0].startswith("# GISAXS fit solutions [a b]")


# -- the command line, on real images ---------------------------------------------------------------------


def test_the_command_line_prints_where_the_calibration_was_found_and_still_exits_0(tmp_path: Path, capsys) -> None:
    from tests.test_assistant_calibration import ENERGY_KEV, agbh_image, giwaxs_frame, save_tiff
    from tools import gimap_agent

    root = tmp_path / "beamtime" / "raw"
    frame = save_tiff(root / "P3HT_film" / "P3HT_00012.tif", giwaxs_frame())
    image = save_tiff(root / "calib" / "AgBH_00001.tif", agbh_image())
    notes = f"Energy: {ENERGY_KEV:.4f} keV, alpha_i = 0.2 deg, pixel size 100 um"
    out = tmp_path / "out"
    code = gimap_agent.main(["auto", str(frame), "--technique", "giwaxs", "--notes", notes, "--out", str(out),
                             "--no-saved-profiles", "--quiet"])
    printed = capsys.readouterr().out

    assert code == gimap_agent.EXIT_DONE  # said, not asked: the exit code is the one of a run with nothing open
    assert printed.startswith("# GIWAXS — batch") and "P3HT_00012.tif | OK" in printed
    note = next(line for line in printed.splitlines() if line.startswith("Calibration found by the automatic search"))
    assert str(image) in note and "sibling folder calib, seen from the frame's folder" in note and "1 frame(s)" in note
    assert "Open questions" not in printed
    report = (out / "P3HT_00012" / "report.md").read_text(encoding="utf-8")
    found = next(line for line in report.splitlines() if line.startswith("- found by the automatic search"))
    assert f"`{image}`" in found and "(sibling folder calib, seen from the frame's folder)" in found
    assert report.index("- source: AgBH_00001.tif") < report.index(found) < report.index("- quality: good")
    assert "Status: **OK**" in report and f"- **{SEARCHED}**: {image} — sibling folder calib" in report
    assert note in (out / "summary.md").read_text(encoding="utf-8")
    assert not (root / "P3HT_film" / "gimap_analysis").exists() and not (root / "calib" / "gimap_analysis").exists()
