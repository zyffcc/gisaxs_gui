"""Cross-area review items of the automatic analysis (assistant area).

Each test names what it guards: Auto's detected technique is followed once the run applied a
geometry (and a switch the procedure makes is not saved as the person's mode), the report and the
decision name the pixels a cut uses with both ends included, a run acts only on the file it started
on and ends as a failure when Analyze shows another one, a file shown after a Clear during a run is
known when the run ends, every started run is followed by finished or failed, and the report carries
the frames and the file the run analysed.
"""

from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from PyQt5.QtWidgets import QApplication

from src.gimap.features.assistant.application import (
    GOALS,
    PERMISSION_AUTO,
    AnalysisGoals,
    FrameChanged,
    PipelineOptions,
    RunResults,
    StandardPipeline,
    ToolCatalog,
    pipeline_markdown,
)
from src.gimap.features.assistant.application.gisaxs_procedure import GisaxsProcedureMixin
from src.gimap.features.assistant.application.gisaxs_report import cut_spans, gisaxs_sections, pixel_span
from src.gimap.features.assistant.presentation import GuidedAnalysis, GuiWorkbench
from src.gimap.features.assistant.presentation.gui_bridge import GuiBridge
from src.gimap.features.assistant.presentation.guided_analysis import frame_key
from tests.assistant_fakes import FakeWorkbench, call
from tests.test_assistant_gui import _app, _wait, _write_frame, analyze  # noqa: F401 - the fixture
from tests.test_assistant_pipeline import setup as series_setup  # not "setup": pytest would call it per module
from tests.test_guided_gisaxs import galaxi_run  # noqa: F401 - the fixture


def _goals() -> AnalysisGoals:
    return AnalysisGoals(goals=tuple(GOALS), permission=PERMISSION_AUTO)


def _report(frame, *, frames=None) -> dict:
    report = {"ok": True, "procedure": "giwaxs", "frame": str(frame), "peaks": [], "rings": [],
              "needs_attention": [], "decisions": [], "steps": []}
    if frames is not None:
        report["frames"] = frames
    return report


class FakeAutomation:
    """What a run asks of Analyze's automation, without a page; the file it shows can change."""

    def __init__(self, path: str):
        self.path = path
        self.calls: list[tuple] = []
        self.statuses = 0
        self.switch: tuple = ()
        """(status call number, path): from that call on, Analyze shows ``path``."""

    def status(self) -> dict:
        self.statuses += 1
        if self.switch and self.statuses >= self.switch[0]:
            self.path = self.switch[1]
        return {
            "file": Path(self.path).name, "path": self.path, "mode": "giwaxs", "measurement": "giwaxs",
            "frames": 12, "frame": 1, "summed_frames": 1, "shape": [64, 64], "curves": [],
            "geometry": {"instrument_profile": "P03", "incidence_deg": 0.4, "wavelength_A": 1.0},
        }

    def set_frame(self, frame, count, done) -> None:
        self.calls.append(("set_frame", frame, count))
        done(True, "")

    def set_mode(self, mode, done, *, remember=True) -> None:
        self.calls.append(("set_mode", mode, remember))
        done(True, "")


# -- Auto: the technique detected once the run applied a geometry --------------------------------------


class AutoWorkbench(FakeWorkbench):
    """Analyze on Auto; the frame reads as GISAXS once it has a geometry (it has one here)."""

    def status(self) -> dict:
        return {**super().status(), "mode": "auto"}


def test_auto_is_followed_once_the_geometry_is_applied() -> None:
    _app()
    assert PipelineOptions().follow_detection is False  # Process with AI: GIWAXS unless the task gives a technique (the CLI and MCP set follow_detection when no technique is given)
    guided = GuidedAnalysis(lambda: None)
    assert guided.options().follow_detection is True
    guided.results.deleteLater()

    workbench = AutoWorkbench(measurement="gisaxs")
    report = StandardPipeline(ToolCatalog(workbench, _goals(), RunResults()), PipelineOptions(follow_detection=True)).run()
    assert report["procedure"] == "gisaxs"
    assert not any(step["tool"] == "set_measurement_mode" for step in report["steps"]) and not workbench.calls

    workbench = AutoWorkbench(measurement="gisaxs")
    report = StandardPipeline(ToolCatalog(workbench, _goals(), RunResults())).run()
    assert report["procedure"] == "giwaxs" and ("set_mode", "giwaxs") in workbench.calls
    why = next(item for item in report["decisions"] if item["what"] == "measurement")["why"]
    assert why.endswith("(choose GISAXS in Analyze, or --technique gisaxs, for the GISAXS one)"), why


def test_a_mode_switch_the_procedure_makes_is_not_saved_as_the_persons_mode() -> None:
    _app()
    automation = FakeAutomation("C:/data/film_a.tif")
    status = GuiWorkbench(automation, GuiBridge()).set_mode("giwaxs")
    assert automation.calls == [("set_mode", "giwaxs", False)] and status["path"] == "C:/data/film_a.tif"


# -- the pixels a cut uses, both ends included -----------------------------------------------------------


def test_the_saved_report_and_the_decision_name_the_pixels_both_ends_included() -> None:
    cuts = {"horizontal_rows": [605.0, 610.0], "horizontal_source": "yoneda", "yoneda_alpha_f_deg": 0.15,
            "vertical_columns": [480.4, 500.6]}
    report = {"gisaxs": {"cuts": cuts}, "tables": {"status": {"shape": [1043, 981]}}}
    text = "\n".join(gisaxs_sections(report))
    assert "rows 605–609, αf = 0.15°" in text and "columns 480–500" in text
    assert cut_spans({"horizontal_rows": [1040.2, 1046.0], "vertical_columns": [975.5, 990.0]}, {"shape": [1043, 981]}) == (
        "1040–1042", "975–980")
    # The pixels Analyze's status names ([first, last], both included) win over the band edges.
    named = dict(cuts, horizontal_pixel_rows=[600, 611], vertical_pixel_columns=[470, 470])
    assert cut_spans(named, None) == ("600–611", "470–470")
    assert cut_spans({}, None) == ("?", "?") and pixel_span((605.0, 610.0)) == "605–609"

    class Probe(GisaxsProcedureMixin):
        def __init__(self):
            self.catalog = SimpleNamespace(results=SimpleNamespace(status={"shape": [1043, 981]}))
            self.decisions: list[str] = []

        def _decide(self, what, decision, why):
            self.decisions.append(decision)

    probe = Probe()
    probe._horizontal_cut(cuts)
    probe._horizontal_cut({"horizontal_source": "manual"})
    assert probe.decisions == ["at the Yoneda band, αf = 0.150° (rows 605–609)", "set by hand (rows ?)"]


def test_a_real_gisaxs_report_counts_the_rows_and_columns_as_analyze_does(galaxi_run) -> None:  # noqa: F811
    from src.gimap.features.assistant.presentation.guided_gisaxs import cut_rows
    from src.gimap.features.assistant.presentation.guided_gisaxs import pixel_span as shown_span

    session, report = galaxi_run
    reduction = session.page.view_model.state.analysis.reduction
    rows = reduction.curve("horizontal").region["rows"]
    columns = reduction.curve("vertical").region["columns"]
    rows_text, columns_text = f"{rows[0]}–{rows[1] - 1}", f"{columns[0]}–{columns[1] - 1}"
    text = pipeline_markdown(report)
    assert f": rows {rows_text}, αf = " in text and f"I(qz): columns {columns_text}" in text
    decision = next(item for item in report["decisions"] if item["what"] == "horizontal cut")["decision"]
    assert decision.endswith(f"(rows {rows_text})"), decision
    assert cut_rows(report) == rows_text and shown_span is pixel_span  # one rule for the Results tab and the report


# -- a run acts only on the file it started on ----------------------------------------------------------


def test_a_pinned_workbench_refuses_to_act_on_another_file_and_the_run_fails_clearly() -> None:
    _app()
    automation = FakeAutomation("C:/data/film_a.tif")
    automation.switch = (2, "C:/data/project_frame.nxs")  # a project opened after the run read the frame
    workbench = GuiWorkbench(automation, GuiBridge(), pin_frame=True)
    report = StandardPipeline(ToolCatalog(workbench, _goals(), RunResults())).run()

    assert workbench.start_path == "C:/data/film_a.tif" and automation.calls == []  # nothing acted on the project
    assert not report["ok"] and report["failed"].startswith(
        "The frame changed during the run: it started on film_a.tif, Analyze now shows project_frame.nxs")
    assert report["frame"] == "C:/data/film_a.tif"
    assert report["frames"] == {"total": 12, "first": 1, "summed": 1}  # the start file's, not the project's
    assert report["steps"][-1]["tool"] == "set_frame" and report["steps"][-1]["error"]
    assert [item["item"] for item in report["needs_attention"]] == ["frame"]
    assert "Status: **FAILED**" in pipeline_markdown(report)

    with pytest.raises(FrameChanged):
        workbench.set_frame(1, 1)
    unpinned = GuiWorkbench(automation, GuiBridge())  # the AI's runs are not pinned
    unpinned.status()
    assert unpinned.set_frame(1, 1)["path"] == "C:/data/project_frame.nxs"


def test_a_status_that_names_another_file_ends_the_run(tmp_path: Path) -> None:
    catalog, workbench, _calibrator = series_setup(tmp_path)
    started = str(workbench.frame)
    original = workbench.set_frame

    def set_frame(frame_number, sum_count):
        workbench.frame = tmp_path / "beamtime" / "raw" / "film" / "next_00001_m01.nxs"
        return original(frame_number, sum_count)

    workbench.set_frame = set_frame
    report = StandardPipeline(catalog, PipelineOptions(energy_kev=12.4, incidence_deg=0.2)).run()
    assert not report["ok"] and "Analyze now shows next_00001_m01.nxs" in report["failed"]
    assert report["frame"] == started and report["peaks"] == [] and report["frames"]["first"] == 1
    assert not any(step["tool"] == "find_peaks" for step in report["steps"])


def test_a_real_analyze_page_refuses_a_pinned_call_while_it_shows_another_file(analyze, tmp_path: Path) -> None:  # noqa: F811
    _window, page, _context = analyze
    film = str(page.view_model.current_path)
    workbench = GuiWorkbench(page.automation(), GuiBridge(), pin_frame=True)
    catalog = ToolCatalog(workbench, _goals(), RunResults())
    assert not catalog.execute(call("get_status")).is_error and workbench.start_path == film
    other = _write_frame(tmp_path / "other_film.tif")
    page.add_paths([other])
    assert page.tasks.wait(60)
    rows = {page.file_list.item(row).text(): row for row in range(page.file_list.count())}

    def show(name: str) -> None:
        page.file_list.setCurrentRow(next(row for text, row in rows.items() if name in text))
        assert page.tasks.wait(60)
        QApplication.processEvents()

    show(other.name)
    assert Path(str(page.view_model.current_path)).name == other.name
    widths = page.view_model.state.giwaxs.in_plane_half_width_deg
    outcome = catalog.execute(call("set_sector_widths", in_plane_half_width_deg=25.0, out_of_plane_half_width_deg=25.0))
    assert outcome.is_error and outcome.data["frame_changed"] and "Analyze now shows other_film.tif" in outcome.summary
    assert page.view_model.state.giwaxs.in_plane_half_width_deg == widths  # the other file was not touched
    show(Path(film).name)
    assert not catalog.execute(call("set_sector_widths", in_plane_half_width_deg=25.0, out_of_plane_half_width_deg=25.0)).is_error


# -- Clear during a run ---------------------------------------------------------------------------------


def _busy(guided) -> threading.Event:
    release = threading.Event()
    guided._thread = threading.Thread(target=release.wait, daemon=True)
    guided._thread.start()
    guided._stop_event = threading.Event()
    return release


def _end(guided, release: threading.Event, report: dict) -> None:
    release.set()
    guided._thread.join()
    guided._finished(report)


def test_a_file_shown_after_a_clear_during_a_run_is_known_when_the_run_ends() -> None:
    _app()
    guided = GuidedAnalysis(lambda: None)
    film, project = "C:/data/film_a.tif", "C:/data/project_frame.nxs"
    guided.frame_shown(film)
    release = _busy(guided)
    guided.files_cleared()
    guided.frame_shown(project, frame=3, summed=10)  # the project's frame, shown while the stopped run ends
    _end(guided, release, _report(film))
    assert guided.report is None and guided.results.isHidden() and not guided._reports
    assert guided._shown_path == frame_key(project) and guided._frames_given == (frame_key(project), 3, 10)
    report = _report(project, frames={"total": 12, "first": 3, "summed": 10})
    guided._finished(report)  # a run on the project's frame: its results are the frame's without another frame_shown
    assert guided._for_this_frame()

    # A file shown before the Clear is gone with it.
    guided.frame_shown(film)
    release = _busy(guided)
    guided.frame_shown("C:/data/before_clear.tif")
    guided.files_cleared()
    _end(guided, release, _report(film))
    assert guided._shown_path is None and guided._frames_given is None
    guided.results.deleteLater()


# -- every started run ends with finished or failed -----------------------------------------------------


def test_every_started_run_is_followed_by_finished_or_failed(monkeypatch) -> None:
    _app()
    guided = GuidedAnalysis(lambda: None)  # no Analyze: the run ends at once with "no image is open"
    events: list[str] = []
    guided.started.connect(lambda _text: events.append("started"))
    guided.finished.connect(lambda _report: events.append("finished"))
    guided.failed.connect(lambda _message: events.append("failed"))

    guided.run()
    _wait(lambda: not guided._busy(), 60)
    assert events == ["started", "finished"]

    events.clear()
    guided.run()
    guided.files_cleared()  # Clear during the run: Analyze still hears its end
    _wait(lambda: not guided._busy(), 60)
    assert events == ["started", "finished"] and guided.report is None

    events.clear()
    guided.report = None  # the end's report was discarded while the start of the series was analysed
    guided._compared(_report("C:/data/a.nxs"))
    assert events == ["failed"]

    events.clear()
    monkeypatch.setattr(guided.results, "show_report", lambda *_args: (_ for _ in ()).throw(RuntimeError("broken")))
    with pytest.raises(RuntimeError):
        guided._finished(_report("C:/data/a.nxs"))
    assert events == ["finished"] and guided.run_button.isEnabled()
    guided.results.deleteLater()


# -- the report names the frames and the file the run analysed -------------------------------------------


def test_the_report_carries_the_frames_and_the_file_the_run_analysed(tmp_path: Path) -> None:
    catalog, workbench, _calibrator = series_setup(tmp_path)
    report = StandardPipeline(catalog, PipelineOptions(energy_kev=12.4, incidence_deg=0.2)).run()
    assert report["ok"] and report["failed"] is None
    assert report["frames"] == {"total": 403, "first": 394, "summed": 10}
    assert report["frame"] == str(workbench.frame)
