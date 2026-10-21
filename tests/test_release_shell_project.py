"""Release 0.1.0, projects: opening one replaces the Fitting curve and the In-situ series of before (R4),
and a project whose data moved says so in one short notice with the whole list behind Details… (R5)."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt5.QtCore import QCoreApplication, QEvent
from PyQt5.QtWidgets import QApplication, QDialog, QMessageBox, QPlainTextEdit

from src.gimap.app import menus as menus_module
from src.gimap.app import project
from src.gimap.app.menus import NOTE_CHARS, SHOWN_NAMES, opened_notice
from src.gimap.app.presentation.components import visible_toasts
from src.gimap.features.fitting.application.single_fit import FitModel, new_component
from tests.test_fit_series import _series, _wait
from tests.test_project import _window


def _settle(seconds: float = 0.2) -> None:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        QApplication.processEvents()
        time.sleep(0.01)


def _deleted() -> None:
    """What was closed is gone (a window deletes itself when closed, once the event loop turns)."""
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)


@pytest.fixture
def window(monkeypatch):
    # A project opened over series results that were not saved asks first (Save / Discard / Cancel): Discard.
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *_args, **_kwargs: QMessageBox.Discard))
    shown = _window()
    yield shown
    shown.close()
    _settle(0.05)


def _work_of_another_sample(window, folder) -> None:
    """A curve with a range and a left-out point in Single analysis, a fitted In-situ series and a Compare
    series: what a window holds before another sample's project is opened in it."""
    components = window.components
    single, series = components.fitting_workspace.fit_page, components.fitting_workspace.series_page
    assert single.open_curve(folder / "run_00001_fit_input.dat")
    single.set_model(FitModel((new_component("sphere", radius=5.0, distance=40.0),)))
    single.set_range((0.2, 1.5))
    single.session.excluded.add(float(f"{single.session.all_points().q[30]:.9g}"))
    single.show_step("results")
    assert series.open_series(folder)
    series.start()
    _wait(series)
    assert series.fits and series.paths
    components.compare_page.add_folder(folder)
    assert components.compare_page.series
    single.session.side = "positive"  # the half chosen for this sample's cut
    series._halves_chosen, series._halves_origin = "negative", "chosen"  # … and the one chosen for its series


def _nothing_of_before(window, chip: str) -> None:
    components = window.components
    single, series = components.fitting_workspace.fit_page, components.fitting_workspace.series_page
    session = single.session
    assert session.curve is None and session.q_range is None and session.excluded == set()
    assert session.result is None and session.solutions == [] and not session.can_undo()
    assert session.side == "mean"  # nor its halves: the next curve starts as the page does
    # The In-situ series has its own halves: those of the series that went are not the next one's.
    assert series._halves_chosen is None and series.series_side() == "mean" and series.halves_source() == "default"
    assert series.side_combo.currentData() == "mean" and series.halves_note.isHidden()
    assert single.curve_chip.text().startswith("No curve yet") and single.status_label.text() == ""
    # ... and each page is where it starts: not on the Results of what went, no range of that curve in the fields
    assert single.step_rail.current() == "curve" and series.step_rail.current() == "curves"
    assert single.range_min_spin.value() == single.range_max_spin.value() == 0.0
    assert series.folder == "" and series.paths == [] and series.fits == {} and series._start_model is None
    assert series.frame_list.count() == 0 and series.results_table.rowCount() == 0
    assert series.folder_chip.text() == chip and series.folder_chip.toolTip() == ""
    assert series.stages_label.text() == "" and not series.save_button.isVisible()
    assert components.compare_page.series == []
    state = project.collect(components)  # what Save Project would write now
    assert state["fitting"]["single"]["curve"] is None and state["fitting"]["series"]["folder"] == ""
    assert state["fitting"]["series"]["side"] == "mean"
    assert state["compare"]["series"] == []


def test_a_project_keeps_the_halves_of_its_in_situ_series(window, tmp_path) -> None:
    """The halves are the series' own (Curves ▸ Halves of the cut), not those of the curve in Single analysis:
    a project brings them back also when that curve is no longer there."""
    import numpy as np

    from tests.test_fit_series import RADII, _cut

    menus, components = window.menus, window.components
    single, series = components.fitting_workspace.fit_page, components.fitting_workspace.series_page
    folder = tmp_path / "gimap_analysis" / "fit_input"
    folder.mkdir(parents=True)
    for index, radius in enumerate(RADII):
        np.savetxt(folder / f"cut_{index + 1:05d}_fit_input.dat", np.column_stack(_cut(radius, index, negative=1.3)))
    curve = tmp_path / "representative_fit_input.dat"
    curve.write_bytes((folder / "cut_00001_fit_input.dat").read_bytes())
    assert single.open_curve(curve, "mean") and series.open_series(folder)
    assert series.set_series_side("negative") and series.halves_source() == "chosen"
    assert menus._write_project(tmp_path / "one half.gimap")
    saved = json.loads(Path(menus.project_path).read_text(encoding="utf-8"))
    assert saved["fitting"]["series"]["side"] == "negative" and saved["fitting"]["series"]["side_from"] == "chosen"
    assert saved["fitting"]["single"]["curve"]["side"] == "mean"

    empty = tmp_path / "empty.gimap"
    empty.write_text(json.dumps({"format": "gimap-project", "version": 1, "analyze": {}}), encoding="utf-8")
    assert menus.open_project(empty)
    _settle()
    assert series.paths == [] and series.series_side() == "mean"
    curve.unlink()  # the curve Single analysis showed has moved
    assert menus.open_project(Path(menus.project_path).with_name("one half.gimap"))
    _settle()
    assert single.session.curve is None and single.session.side == "mean"
    assert len(series.paths) == 5 and series.series_side() == "negative" and series.halves_source() == "project"
    assert series.side_combo.currentData() == "negative" and series.halves_note.text() == "as saved in the project"
    assert "q < 0 half (as saved in the project)" in series.model_summary.text()


def test_opening_a_project_closes_the_curve_and_the_series_of_before(window, tmp_path) -> None:
    menus, components = window.menus, window.components
    series = components.fitting_workspace.series_page
    chip = series.folder_chip.text()
    assert chip == project.NO_SERIES_FOLDER  # the sentence the page starts with (its view's)
    assert menus._write_project(tmp_path / "other sample.gimap")  # a project with no Fitting or Compare work
    other = Path(menus.project_path)
    folder = _series(tmp_path)
    _work_of_another_sample(window, folder)

    assert menus.open_project(other)
    _settle()
    _nothing_of_before(window, chip)
    stored = window.app_context.preferences.get("fitting_single_page", {})
    assert stored.get("curve") is None  # nor is the curve of before opened again at the next start
    assert menus.save_project()  # Save Project: the opened project stays what it was
    saved = json.loads(other.read_text(encoding="utf-8"))
    assert saved["fitting"]["single"]["curve"] is None and saved["fitting"]["series"]["folder"] == ""
    assert saved["compare"]["series"] == []

    # A project file of an older kind (no Fitting part, no Compare part) closes them too.
    _work_of_another_sample(window, folder)
    older = tmp_path / "older.gimap"
    older.write_text(json.dumps({"format": "gimap-project", "version": 1, "analyze": {}}), encoding="utf-8")
    assert menus.open_project(older)
    _settle()
    _nothing_of_before(window, chip)


def test_a_project_whose_curve_and_folder_are_gone_does_not_keep_those_of_before(window, tmp_path) -> None:
    menus, components = window.menus, window.components
    single, series = components.fitting_workspace.fit_page, components.fitting_workspace.series_page
    chip = series.folder_chip.text()
    folder = _series(tmp_path)
    _work_of_another_sample(window, folder)
    gone = tmp_path / "moved away"
    moved = tmp_path / "moved.gimap"
    moved.write_text(json.dumps({
        "format": "gimap-project", "version": 1, "page": "fitting", "analyze": {},
        "fitting": {"single": {"curve": {"path": str(gone / "cut_fit_input.dat"), "unit": "angstrom", "side": "mean"}},
                    "series": {"folder": str(gone / "curves"), "pattern": "*_cut.dat", "start": "same"}},
    }), encoding="utf-8")
    assert menus.open_project(moved)
    _settle()
    _nothing_of_before(window, chip)
    assert series.pattern_edit.text() == "*_cut.dat" and series.start_same.isChecked()  # the project's choices
    assert single.session.model.components  # the model stays a model

    # ... and a project that names a curve and a folder that are there opens them as before.
    assert single.open_curve(folder / "run_00002_fit_input.dat") and series.open_series(folder, "*_fit_input.dat")
    assert menus._write_project(tmp_path / "with data.gimap")
    with_data = Path(menus.project_path)
    assert menus.open_project(moved) and single.session.curve is None and series.paths == []
    assert menus.open_project(with_data)
    _settle()
    assert single.session.curve is not None and single.session.curve.name == "run_00002_fit_input.dat"
    assert series.folder == str(folder) and len(series.paths) == 5 and series.frame_list.count() == 5


# -- a project whose data moved (R5) ---------------------------------------------------------------


def _moved_project(tmp_path) -> tuple[Path, dict]:
    """A project naming 40 frames, a curve, a series folder and three Compare series of five curves, none of
    which is there (the beamtime folder went to another drive)."""
    root = tmp_path / "beamtime on another drive"
    data = {
        "format": "gimap-project", "version": 1, "page": "compare",
        "analyze": {"files": [str(root / "frames" / f"film_{index:05d}.tif") for index in range(1, 41)]},
        "fitting": {"single": {"curve": {"path": str(root / "curves" / "film_00001_fit_input.dat")}},
                    "series": {"folder": str(root / "curves")}},
        "compare": {"series": [{"name": name, "files": [str(root / name / f"{name}_{index:05d}_fit_input.dat")
                                                         for index in range(1, 6)]} for name in ("runA", "runB", "runC")]},
    }
    path = tmp_path / "sample one.gimap"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path, data


def test_missing_lists_what_a_project_names_and_is_not_there(tmp_path) -> None:
    path, data = _moved_project(tmp_path)
    gone = project.missing(data, path)
    assert len(gone) == 40 + 1 + 1 + 15 and gone[0].endswith("film_00001.tif") and gone[-1].endswith("runC_00005_fit_input.dat")
    here = tmp_path / "here"
    here.mkdir()
    frame, curve = here / "a.tif", here / "a_fit_input.dat"
    frame.write_bytes(b"0")
    curve.write_text("0.1 1 1\n", encoding="utf-8")
    present = {"analyze": {"files": [str(frame), str(here / "b.tif"), str(here / "b.tif")]},
               "fitting": {"single": {"curve": {"path": str(curve)}}, "series": {"folder": str(here)}},
               "compare": {"series": [{"name": "s", "files": [str(curve), str(here / "c.dat")]}],
                           "arrays_file": "sample.compare.npz"}}
    assert project.missing(present, here / "sample.gimap") == [
        str(here / "b.tif"), str(here / "c.dat"), str(here / "sample.compare.npz")]  # each once, in the pages' order
    assert project.missing(present) == [str(here / "b.tif"), str(here / "c.dat")]  # no project path: no sidecar
    # A file written by hand (parts of another type) is read as far as it goes.
    assert project.missing({"analyze": [], "fitting": {"single": "x", "series": None}, "compare": {"series": "x"}}) == []
    assert project.missing({}) == []


def test_the_notice_counts_the_missing_files_and_names_the_first() -> None:
    paths = [str(Path("D:/beamtime/frames") / f"film_{index:05d}.tif") for index in range(1, 10)]
    notes = ["9 frames of the project are no longer there",
             "Compare: runA could not be read (Fewer than two curves could be read: runA_00001_fit_input.dat "
             "([Errno 2] No such file or directory: 'D:\\\\beamtime\\\\runA\\\\runA_00001_fit_input.dat'))"]
    text, summary, details = opened_notice("sample one.gimap", paths, notes)
    assert text == ("Opened the project sample one.gimap. 9 files or folders it names are no longer there: "
                    "film_00001.tif, film_00002.tif, film_00003.tif and 6 more")
    assert "Errno" not in text and "beamtime" not in text  # no raw error, no path: those are in Details…
    assert summary.startswith(text) and "saving now would write the project without what is missing" in summary
    folder = str(Path("D:/beamtime/frames"))
    # Every file, under its folder said once; every note in full.
    assert details.startswith("No longer there (9):\n" + folder + "\n    film_00001.tif\n    film_00002.tif\n")
    assert details.count(folder) == 1 and all(f"    {Path(path).name}" in details for path in paths)
    assert all(note in details for note in notes)
    elsewhere = str(Path("E:/other/run_00001_fit_input.dat"))
    _text, _summary, two_folders = opened_notice("s.gimap", [paths[0], elsewhere, paths[1]], [])
    assert two_folders == "\n".join(["No longer there (3):", folder, "    film_00001.tif", "    film_00002.tif",
                                     str(Path("E:/other")), "    run_00001_fit_input.dat"])

    few, _summary, _details = opened_notice("s.gimap", paths[:SHOWN_NAMES], [])
    assert few.endswith("film_00001.tif, film_00002.tif, film_00003.tif") and "more" not in few
    one, _summary, listed = opened_notice("s.gimap", paths[:1], [])
    assert one == "Opened the project s.gimap. film_00001.tif is no longer there."
    assert listed == "No longer there (1):\n" + folder + "\n    film_00001.tif"

    # Nothing missing, but parts not restored: the first notes, one line each; all of them in Details….
    long_note = "the Analyze set-up could not be read: " + "x" * 400
    only_notes, summary, details = opened_notice("s.gimap", [], [long_note, "second\nnote", "third", "fourth"])
    assert only_notes.startswith("Opened the project s.gimap: the Analyze set-up could not be read: xxx")
    assert only_notes.endswith("…; second note … and 2 more") and len(only_notes) < 2 * NOTE_CHARS + 60
    assert summary == only_notes and details == "What the pages could not restore:\n" + "\n".join(
        [long_note, "second\nnote", "third", "fourth"])


def test_a_project_whose_data_moved_opens_with_one_short_notice_and_details(window, tmp_path) -> None:
    path, data = _moved_project(tmp_path)
    menus = window.menus
    assert menus.open_project(path)
    _settle()
    notices = [toast for toast in visible_toasts(window) if toast.text().startswith("Opened the project")]
    assert len(notices) == 1
    notice = notices[0]
    assert notice.text() == ("Opened the project sample one.gimap. 57 files or folders it names are no longer there: "
                             "film_00001.tif, film_00002.tif, film_00003.tif and 54 more")
    assert notice.property("level") == "warning" and notice.action_button.text() == "Details…"
    assert notice.height() < 140  # a few lines, not half the window
    assert not any("Errno" in toast.text() for toast in visible_toasts(window))
    assert notice.timeout_ms == 0  # it stays until closed: Details… is the only place that lists the files

    notice.action_button.click()  # Details…: every path, and what each page said, to read and copy
    _settle(0.1)
    dialog = window.findChild(QDialog, "detailsDialog")
    assert dialog is not None and dialog.isVisible() and not dialog.isModal() and dialog.windowTitle() == "Open Project"
    listed = dialog.findChild(QPlainTextEdit, "detailsText").toPlainText()
    gone = [Path(item) for item in project.missing(data, path)]
    assert all(str(item.parent) in listed and f"    {item.name}" in listed for item in gone)
    assert listed.count("\n    ") == len(gone) == 57 and "What the pages could not restore:" in listed
    dialog.findChild(type(notice.action_button), "detailsCopy").click()
    assert QApplication.clipboard().text() == listed

    # Another project opened: the notice and the Details window of this one are about other data, and go.
    assert menus.open_project(path)  # (the same file: its notice is said again, once)
    _settle()
    _deleted()
    again = [toast for toast in visible_toasts(window) if toast.action_button is not None]
    assert len(again) == 1 and again[0] is not notice and window.findChild(QDialog, "detailsDialog") is None
    again[0].action_button.click()
    _settle(0.1)
    assert window.findChild(QDialog, "detailsDialog") is not None
    nothing_missing = tmp_path / "empty.gimap"
    nothing_missing.write_text(json.dumps({"format": "gimap-project", "version": 1}), encoding="utf-8")
    assert menus.open_project(path) and menus.open_project(nothing_missing)
    _settle()
    _deleted()
    assert [toast.text() for toast in visible_toasts(window)] == ["Opened the project empty.gimap."]
    assert window.findChild(QDialog, "detailsDialog") is None


def test_a_project_file_that_cannot_be_read_is_refused_in_words(window, monkeypatch, tmp_path) -> None:
    warned = []
    monkeypatch.setattr(menus_module, "warn", lambda _parent, title, text: warned.append((title, text)))
    assert window.menus.open_project(tmp_path / "nothing.gimap") is False
    assert warned[-1] == ("Open Project", "nothing.gimap could not be opened: No such file or directory")
    broken = tmp_path / "broken.gimap"
    broken.write_bytes(b"\x00\x01 not json")
    assert window.menus.open_project(broken) is False
    assert warned[-1] == ("Open Project", "broken.gimap could not be opened: not a GIMaP project")
    with pytest.raises(ValueError, match="not a GIMaP project"):
        project.read(broken)
    assert window.menus.project_path == "" and window.windowTitle() == "GIMaP"
