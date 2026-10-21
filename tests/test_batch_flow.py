"""Batch Export from the dialog to the last file: what the dialog promises, what a batch writes, what it says.

* The tables of every frame put no number where no frame measured one.
* A folder that takes no file is found out before the first frame; an earlier export in it is replaced or
  kept apart on the person's answer, never mixed in silence — also for the curves sent to Fitting.
* A batch that was stopped, or lost frames, does not end as a success: status line, notice, panel,
  ``README.txt`` and the record say what happened and how far it got.
* The record lists every file; the README names the files that are there.
* Sums are called sums, a last sum of fewer frames is marked; the fit trend is drawn against the frame.

Every frame is synthetic.
"""

from __future__ import annotations

import csv
import json
import os
import stat
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.gimap.features.analyze.application import (
    NO_PERMISSION,
    NOT_A_FOLDER,
    NUMBER_COLUMN,
    RECORD_FORMAT,
    BatchChoices,
    BatchRun,
    CurveTables,
    CutRegion,
    FitTable,
    batch_cost,
    batch_record,
    cannot_write,
    common_x,
    earlier_export,
    may_be_written,
    new_subfolder,
    readme_text,
    remove_export,
    written_kinds,
)
from src.gimap.features.analyze.application import output_names
from src.gimap.features.analyze.application.batch_tables import EXACT
from src.gimap.features.analyze.domain import Curve
from src.gimap.features.analyze.infrastructure.adapters import CsvCurveWriter
from src.gimap.shared.detector_io import forget_layout
from src.gimap.shared.geometry import InstrumentProfile
from tests.test_assistant_calibration import SHAPE, giwaxs_frame, save_tiff
from tests.test_giwaxs_workspace import GEOMETRY, _settle


@pytest.fixture(autouse=True)
def _nothing_remembered():
    forget_layout()
    yield
    forget_layout()


# -- tables of every frame: no number that no frame measured -------------------------------------------------------


def test_frames_on_the_same_points_up_to_rounding_are_matched_not_interpolated() -> None:
    """Another cut row moves qy in the eighth digit; a masked column or a gap takes points away; a wider frame
    has points the others lack. Each value stays the frame's own, on the row of its x: none is invented across a
    hole, and no measured edge point is lost."""
    x = np.linspace(-0.2, 0.2, 81)
    first = (x, 100.0 + np.arange(81.0))
    hole = np.ones(81, dtype=bool)
    hole[30:36] = False  # six masked columns
    second = (x[hole] + 4.5e-8, 200.0 + np.arange(81.0)[hole])
    wider_x = np.r_[x[0] - 0.01, x[0] - 0.005, x, x[-1] + 0.005] - 2e-8
    third = (wider_x, 300.0 + np.arange(wider_x.size, dtype=float))
    grid, values, how = common_x([first, second, third])
    assert "matched" in how and "nothing is interpolated" in how and "6.5e-08" in how  # −2e-8 … +4.5e-8 apart
    assert grid.size == 84  # the 81 common points and the three only the widest frame has
    np.testing.assert_array_equal(values[0][2:83], first[1])  # the first frame as it is
    assert np.isnan(values[0][[0, 1, 83]]).all()
    in_second = np.isfinite(values[1])
    assert int(in_second.sum()) == 75 and np.isnan(values[1][2 + 30:2 + 36]).all()  # the hole stays a hole
    np.testing.assert_array_equal(values[1][in_second], second[1])  # … and no count became a fraction
    np.testing.assert_array_equal(values[2], third[1])  # the edge points of the widest frame are all there
    np.testing.assert_allclose(grid[2:83], x, atol=0)  # x of the first frame that has the point


def test_frames_on_really_different_grids_are_interpolated_but_never_across_a_hole() -> None:
    x = np.linspace(0.1, 1.0, 10)
    shifted = np.delete(x + 0.03, [4, 5])  # another grid, with a hole from 0.43 to 0.73
    grid, values, how = common_x([(x, x), (shifted, shifted), (x + 0.01, x + 0.01)])
    assert "interpolated" in how and "never across a gap" in how
    np.testing.assert_allclose(grid, x)
    np.testing.assert_allclose(values[1][[1, 2, 3, 7, 8, 9]], x[[1, 2, 3, 7, 8, 9]])  # y = x on its own grid
    assert np.isnan(values[1][[0, 4, 5, 6]]).all()  # before its first point, and inside the hole
    assert np.isnan(values[2][0]) and values[2][-1] == pytest.approx(1.0)  # never extrapolated
    # a frame that reaches further by whole steps gets rows for them: no edge point is left out
    far = np.linspace(0.13, 1.33, 13)
    grid, values, _how = common_x([(x, x), (far, far)])
    assert grid[-1] == pytest.approx(1.3) and values[1][-1] == pytest.approx(1.3) and np.isnan(values[0][-1])
    assert common_x([(x, x), (x, 2 * x)])[2] == EXACT


def test_interpolated_tables_keep_what_other_frames_measured_in_a_hole_of_the_first() -> None:
    """The first frame lacks a hundred points (masked columns); the second, on another grid, measured there. The
    common x has rows in that hole too, at the first frame's step: the second frame's points are not left out."""
    x = np.linspace(0.0, 1.0, 301)
    first = np.delete(x, np.arange(100, 200))
    other = np.linspace(0.0005, 0.9995, 333)
    grid, values, how = common_x([(first, 3.0 * first), (other, 2.0 * other)])
    assert how.startswith("x: a common grid") and "across its own holes" in how
    inside = (grid > x[99]) & (grid < x[200])
    assert int(inside.sum()) >= 95 and np.isnan(values[0][inside]).all()  # the first frame has nothing there …
    np.testing.assert_allclose(values[1][inside], 2.0 * grid[inside], rtol=1e-9)  # … the second its own line
    np.testing.assert_allclose(values[0][~inside], 3.0 * grid[~inside], rtol=1e-12)  # the first frame where it measured
    assert np.isfinite(np.vstack(values)).any(axis=0).all()  # no row that no frame has a number on
    # a hole that no frame measured in stays a hole without rows
    grid, values, _how = common_x([(first, first), (np.delete(other, np.arange(110, 220)), np.delete(other, np.arange(110, 220)))])
    assert np.isfinite(np.vstack(values)).any(axis=0).all() and not ((grid > 0.37) & (grid < 0.65)).any()
    # a first frame without a single value does not take the rows of the others with it
    grid, values, _how = common_x([(x, np.full(x.size, np.nan)), (x, x), (other, other)])
    assert grid.size == 301 and np.isnan(values[0]).all() and int(np.isfinite(values[1]).sum()) == 301
    assert int(np.isfinite(values[2]).sum()) >= 299


def test_a_table_keeps_the_digits_of_the_frames_and_says_what_is_missing(tmp_path: Path) -> None:
    x = np.linspace(0.1, 1.0, 10)
    tables = CurveTables()
    tables.add("frame 1", [Curve("radial", "I(q)", x, np.full(10, 383711.41), np.ones(10), np.full(10, 9), "q (Å⁻¹)")])
    tables.add("frame 3", [Curve("radial", "I(q)", x[:-2] + 1e-9, np.full(8, 2.0), np.ones(8), np.full(8, 9), "q (Å⁻¹)")])
    (path,) = tables.write(CsvCurveWriter(), tmp_path, "run", notes=["1 frames failed and are missing here: frame 2"])
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[2].startswith("# x: every frame's own points, matched row by row")
    assert lines[3] == "# 1 frames failed and are missing here: frame 2"
    rows = list(csv.reader(lines[4:]))
    assert rows[0] == ["q (1/A)", "frame 1", "frame 3"] and rows[1] == ["0.1", "383711.41", "2"]  # 8 digits, as in the frame's file
    assert rows[-1] == ["1", "383711.41", ""]  # the second frame ends earlier: blank, not a number


# -- the folder ------------------------------------------------------------------------------------------------------


def _read_only(folder: Path) -> Path:
    folder.mkdir(parents=True)
    folder.chmod(stat.S_IRUSR | stat.S_IXUSR)
    return folder


needs_permissions = pytest.mark.skipif(os.name == "nt" or (hasattr(os, "geteuid") and os.geteuid() == 0),
                                       reason="read-only folders are set with chmod, which root ignores")


@needs_permissions
def test_a_folder_that_takes_no_file_is_told_before_anything_is_written(tmp_path: Path) -> None:
    locked = _read_only(tmp_path / "locked")
    try:
        assert cannot_write(locked) and cannot_write(locked / "run_07")  # also a subfolder that cannot be made
        assert may_be_written(locked) == NO_PERMISSION and may_be_written(locked / "run_07" / "deeper") == NO_PERMISSION
    finally:
        locked.chmod(stat.S_IRWXU)
    note = tmp_path / "notes.txt"
    note.write_text("x", encoding="utf-8")
    assert cannot_write(note) and may_be_written(note) == NOT_A_FOLDER and may_be_written(note / "below") == NOT_A_FOLDER
    fresh = tmp_path / "new" / "deeper"
    assert may_be_written(fresh) is None and not fresh.exists()  # asked without touching the disk
    assert cannot_write(fresh) is None and fresh.is_dir() and list(fresh.iterdir()) == []  # tried, nothing left behind


def test_an_earlier_export_is_found_replaced_or_kept_apart(tmp_path: Path) -> None:
    target = tmp_path / "run"
    (target / "curves").mkdir(parents=True)
    (target / "fit_input").mkdir()
    files = ["run_radial_frames.csv", "curves/a_radial.csv", "curves/a_analysis.json", "fit_input/a_fit_input.dat", "README.txt"]
    for name in files:
        (target / name).write_text("x", encoding="utf-8")
    (target / "notes of mine.txt").write_text("keep", encoding="utf-8")
    (target / "curves" / "mine.csv").write_text("keep", encoding="utf-8")
    record = target / "run_batch.json"
    record.write_text(json.dumps({"format": RECORD_FORMAT, "started": "2026-01-02 03:04", "frames": ["a", "b"],
                                  "files": [*files, "run_batch.json", "../outside.txt", "gone.csv"]}), encoding="utf-8")
    assert earlier_export(tmp_path / "empty", "run") is None and earlier_export(target, "other") is None
    found = earlier_export(target, "run", fit_inputs=True)
    assert found.when == "2026-01-02 03:04" and found.frames == 2 and found.complete
    assert sorted(path.relative_to(target).as_posix() for path in found.files) == sorted(files)  # only what is there, inside
    assert [path.name for path in found.fit_inputs] == ["a_fit_input.dat"]
    assert new_subfolder(tmp_path, "run") == tmp_path / "run_2"
    (tmp_path / "run_2").mkdir()
    assert new_subfolder(tmp_path, "run") == tmp_path / "run_3"
    assert remove_export(found) == len(files) + 1  # its files and its record
    left = sorted(path.relative_to(target).as_posix() for path in target.rglob("*") if path.is_file())
    assert left == ["curves/mine.csv", "notes of mine.txt"]  # nothing of the person's own
    assert not (target / "fit_input").exists() and (target / "curves").is_dir()  # an emptied folder goes, another stays
    # curves for Fitting of other data in the folder: found when this batch writes some (Fitting would list both)
    (target / "fit_input").mkdir()
    (target / "fit_input" / "other_fit_input.dat").write_text("x", encoding="utf-8")
    assert earlier_export(target, "run") is None
    assert [path.name for path in earlier_export(target, "run", fit_inputs=True).fit_inputs] == ["other_fit_input.dat"]
    # a record of an earlier version lists only its tables: it says so, and Replace leaves the rest
    record.write_text(json.dumps({"frames": ["a"], "files": ["run_radial_frames.csv"]}), encoding="utf-8")
    assert earlier_export(target, "run").complete is False


@pytest.mark.skipif(os.name == "nt", reason="symbolic links need extra rights on Windows")
def test_replace_never_removes_a_file_outside_the_folder_whatever_the_record_says(tmp_path: Path) -> None:
    """A record is a file anyone can edit, and a folder may hold links: Replace takes away what is IN the folder."""
    target = tmp_path / "out" / "run"
    (target / "curves").mkdir(parents=True)
    precious, elsewhere = tmp_path / "precious.csv", tmp_path / "elsewhere"
    precious.write_text("keep", encoding="utf-8")
    elsewhere.mkdir()
    for name in ("b_radial.csv", "c_fit_input.dat"):
        (elsewhere / name).write_text("keep", encoding="utf-8")
    (target / "curves" / "a_radial.csv").write_text("x", encoding="utf-8")
    (target / "linked.csv").symlink_to(precious)  # a link in the folder to a file outside it
    (target / "linked_folder").symlink_to(elsewhere, target_is_directory=True)
    (target / "fit_input").symlink_to(elsewhere, target_is_directory=True)  # curves for Fitting kept elsewhere
    listed = ["curves/a_radial.csv", "../../precious.csv", str(precious), "linked.csv", "linked_folder/b_radial.csv",
              "fit_input/c_fit_input.dat", "run_batch.json"]
    (target / "run_batch.json").write_text(json.dumps({"format": RECORD_FORMAT, "frames": ["a"], "files": listed}),
                                           encoding="utf-8")
    found = earlier_export(target, "run", fit_inputs=True)
    assert [path.relative_to(target).as_posix() for path in found.files] == ["curves/a_radial.csv"]
    assert found.fit_inputs == []  # the curves the link leads to are not this folder's
    assert remove_export(found) == 2  # the curve and the record
    assert precious.read_text(encoding="utf-8") == "keep" and (target / "linked.csv").is_symlink()
    assert sorted(path.name for path in elsewhere.iterdir()) == ["b_radial.csv", "c_fit_input.dat"]


def test_replace_is_the_default_answer_only_when_it_takes_nothing_this_export_does_not_write_again(tmp_path: Path) -> None:
    """Enter answers the question with its default button. Replace is that button only for an export of the same
    kinds of file (the same batch once more); an earlier export with tables, curves or pictures this one does not
    write again — a Batch Export, and now a Send Series to Fitting into its folder — is kept unless Replace is
    pressed, and the question says what Replace would take."""
    from src.gimap.features.analyze.domain import SeriesCorrection
    from src.gimap.features.analyze.presentation.batch_dialog import BatchExportDialog
    from tests.test_analyze_workspace import _app

    _app()
    target = tmp_path / "out" / "run"
    files = ["run_radial_frames.csv", "curves/a_radial.csv", "curves/a_analysis.json", "images/a_detector.png",
             "fit_input/a_fit_input.dat", "README.txt"]
    for name in files:
        (target / name).parent.mkdir(parents=True, exist_ok=True)
        (target / name).write_text("x", encoding="utf-8")
    (target / "run_batch.json").write_text(json.dumps({"format": RECORD_FORMAT, "frames": ["a", "b"],
                                                       "files": [*files, "run_batch.json"]}), encoding="utf-8")
    found = earlier_export(target, "run", fit_inputs=True)
    send = BatchChoices(tables=False, fit_input=True)
    assert sorted(path.relative_to(target).as_posix() for path in found.not_written_again(send)) == [
        "curves/a_analysis.json", "curves/a_radial.csv", "images/a_detector.png", "run_radial_frames.csv"]
    again = BatchChoices(tables=True, per_frame=True, fit_input=True, detector_image=True, every=3)
    assert found.not_written_again(again) == []  # every kind of it is written again (README and record always)

    def dialog(**options):
        values = dict(frames=6, files=6, stem="run", settings_text="GIWAXS", choices=again, destination=tmp_path / "out",
                      subfolder=True, series=SeriesCorrection(), curve_options=[("radial", "I(q)")])
        values.update(options)
        return BatchExportDialog(**values)

    batch = dialog()
    box, replace, keep = batch._earlier_box(batch._earlier())
    assert box.defaultButton() is replace and "Replace removes the 7 files of the earlier export first." == box.informativeText(
    ).split(" Keep Both")[0]
    narrower = dialog(choices=BatchChoices(tables=True))
    box, replace, keep = narrower._earlier_box(narrower._earlier())
    assert box.defaultButton() is keep
    assert ("Replace removes the 7 files of the earlier export first, among them 4 files of kinds this export does not "
            "write again: curves of single frames, curves for Fitting, pictures.") in box.informativeText()
    from src.gimap.app.presentation.i18n import apply_language

    try:
        apply_language("zh")
        box, _replace, _keep = narrower._earlier_box(narrower._earlier())
        assert "其中 4 个属于这次导出不会再写的类型：逐帧曲线、拟合用曲线、图片。" in box.informativeText()
    finally:
        apply_language("en")
    # Send Series to Fitting has a folder of its own beside the export; pointed into the export, it asks the same way
    sending = dialog(for_fitting=True)
    assert sending.target() == tmp_path / "out" / "run_fitting" and sending._earlier() is None
    assert sending.subfolder_check.text() == "In a subfolder named “run_fitting”"
    into = dialog(for_fitting=True, destination=target, subfolder=False)
    box, replace, keep = into._earlier_box(into._earlier())
    assert box.defaultButton() is keep and "kinds this export does not write again" in box.informativeText()
    for each in (batch, narrower, sending, into):
        each.done(0)
        each.deleteLater()


# -- the record and the README: the batch as it went ---------------------------------------------------------------


def _run(tmp_path: Path, **values) -> BatchRun:
    run = BatchRun(choices=BatchChoices(per_frame=True, fit_input=True, fit="model", fit_curves=True), destination=tmp_path,
                   stem="run")
    for key, value in values.items():
        setattr(run, key, value)
    return run


def test_the_record_and_the_readme_say_how_the_batch_went(tmp_path: Path) -> None:
    run = _run(tmp_path, requested=6, frames=["a frame 1 of 12 (+2 summed)", "a frame 4 of 12 (+2 summed)", "a frame 10 of 12 (+1 summed)"],
               numbers=[1, 4, 10], sums=[3, 3, 2], stopped=True, not_started=["a frame 7 of 12 (+2 summed)"], seconds=12.4,
               cuts=[{"frame": "a", "horizontal_rows": [1170, 1175]}, {"frame": "b", "horizontal_rows": [1169, 1174]},
                     {"frame": "c", "horizontal_rows": [1170, 1175]}],
               series_cut_text="horizontal cut: found again in each frame",
               first_metadata={"measurement": "gisaxs"},
               earlier={"action": "replaced", "when": "2026-01-02 03:04", "removed": 7, "complete": False, "record": "run_batch.json"})
    run.fail("a frame 13 of 12", "could not be read: the file looks damaged", "a.nxs")
    run.fit_frames = [{"frame": "a", "state": "ok"}, {"frame": "b", "state": "failed", "reason": "No points on the q > 0 half"}]
    files = [tmp_path / "curves" / "a_horizontal.csv", tmp_path / "fit_input" / "a_fit_input.dat", tmp_path / "run_model_fits.csv",
             tmp_path / "README.txt", tmp_path / "run_batch.json"]
    record = batch_record(run, files)
    assert record["format"] == RECORD_FORMAT and record["requested"] == 6 and record["stopped"] is True
    assert record["not_started"] == ["a frame 7 of 12 (+2 summed)"] and record["frame_numbers"] == [1, 4, 10]
    assert record["failed"] == [{"frame": "a frame 13 of 12", "why": "could not be read: the file looks damaged", "file": "a.nxs"}]
    assert record["summary"] == {"requested": 6, "exported": 3, "failed": 1, "fits_failed": 1, "stopped": True, "seconds": 12.4,
                                 "files": 5}
    assert record["files"] == ["curves/a_horizontal.csv", "fit_input/a_fit_input.dat", "run_model_fits.csv", "README.txt",
                               "run_batch.json"]  # every file, with / on every system
    assert record["cuts"][1] == {"frame": "b", "horizontal_rows": [1169, 1174]}
    assert record["sums"][-1] == {"frame": "a frame 10 of 12 (+1 summed)", "frames_summed": 2}
    assert record["earlier_export"]["action"] == "replaced"
    names = output_names(run.choices, batch="run", frame="<frame>", curve="<curve>")
    kinds = written_kinds(files, tmp_path, names)
    assert kinds == {"per_frame", "fit_input", "fit", "readme", "record"}
    text = readme_text(run.choices, batch="run", frame="a", frames=3, curves=[], halves="mean", written=kinds, run=run)
    assert "Result: 3 of 6 frames exported in 12 s." in text
    assert "Stopped by the user: 1 frames were not started." in text
    assert "1 frames failed:\n  a frame 13 of 12: could not be read: the file looks damaged" in text
    assert "1 of 2 frames were not fitted: their reasons are in the note column of run_model_fits.csv." in text
    assert "An earlier export of the same data in this folder (2026-01-02 03:04) was replaced: 7 of its files were removed" in text
    assert "written by an earlier version of GIMaP" in text
    assert "Cuts: horizontal cut: found again in each frame." in text
    assert 'Horizontal cut: detector rows 1169-1174 to 1170-1175, not the same in every frame (every frame\'s is under "cuts"' in text
    assert "Sums: every frame here is the sum of 3 frames of the source" in text
    assert "Not of 3 frames: a frame 10 of 12 (+1 summed) (2)" in text
    # the fit plot and the fitted curves were chosen and no frame gave them: named as missing, not as files
    listed, missing = text.split("Chosen, but nothing of it could be written")
    assert "run_model_fits.png" not in listed and "run_model_fits.png" in missing and "fits/<frame>_model_fit.csv" in missing
    # a folder that stopped taking files is said instead of the stop it caused
    full = _run(tmp_path, requested=6, frames=["a"], stopped=True, not_started=["b"], write_error="No space left on device")
    assert "The batch was ended because the folder took no more files: No space left on device." in readme_text(
        full.choices, batch="run", frame="a", frames=1, curves=[], run=full)
    assert batch_record(full, [])["write_error"] == "No space left on device"


def test_the_cost_of_a_batch_follows_what_is_chosen() -> None:
    frame = dict(frames=100, frame_pixels=2048 * 2048, curve_points=5000, fit_points=1800, map_cells=480_000, cake_cells=160_000)
    tables = batch_cost(BatchChoices(), **frame)
    assert tables[0] == 0 and tables[1] > 0  # nothing per frame: the tables alone
    curves = batch_cost(BatchChoices(per_frame=True), **frame)
    assert 100_000 < curves[0] < 1_000_000
    everything = batch_cost(BatchChoices(per_frame=True, fit_input=True, q_map=True, cake=True, detector_image=True,
                                         q_map_image=True, frames=True), **frame)
    assert 20e6 < everything[0] < 40e6 and everything[1] == pytest.approx(100 * everything[0], rel=0.01)  # measured: 28 MB
    assert batch_cost(BatchChoices(detector_image=True, image_format="tif"), **frame)[0] > 3 * batch_cost(
        BatchChoices(detector_image=True), **frame)[0]
    assert everything[2] > 3 * tables[2] and batch_cost(BatchChoices(), **{**frame, "frames": 200})[2] == pytest.approx(2 * tables[2])
    assert batch_cost(BatchChoices(), at_once=4, **frame)[2] == pytest.approx(tables[2] / 4 + 6.0)  # the workers start
    assert batch_cost(BatchChoices(fit="model"), **frame)[2] > tables[2] + 100 * 10


# -- the trend of the fits -------------------------------------------------------------------------------------------


def test_the_fit_trend_is_drawn_against_the_frame_with_errors_and_without_empty_series(tmp_path: Path, monkeypatch) -> None:
    from src.gimap.features.analyze.infrastructure.adapters import MatplotlibFigureWriter, figures
    from src.gimap.features.analyze.presentation.batch_model import _trend_panels

    table = FitTable("peaks")
    for number, q in ((1, 0.99721), (3, 0.99722), (5, 0.99723)):  # every second frame of the list
        table.add({"frame": f"f{number}", "A q (1/A)": q, "A q err": 2e-6, "A FWHM (1/A)": 0.02, "A FWHM err": 1e-3, "A area": 5.0,
                   "A area err": 0.2, "A note": "", "B q (1/A)": float("nan"), "B q err": float("nan"),
                   "B FWHM (1/A)": float("nan"), "B FWHM err": float("nan"), "B area": float("nan"), "B area err": float("nan"),
                   "B note": "no significant peak"}, number)
    panels = _trend_panels(table, "peaks")
    assert [label for label, _curves in panels] == ["q (1/A)", "FWHM (1/A)", "area"]
    (name, x, y, error), = panels[0][1]  # the peak that was never found is no series
    assert name == "A" and x.tolist() == [1.0, 3.0, 5.0] and error.tolist() == [2e-6] * 3
    written = table.write(CsvCurveWriter(), tmp_path / "fits.csv").read_text(encoding="utf-8").splitlines()
    header = next(line for line in written if not line.startswith("#")).split(",")
    assert header[:2] == ["frame", NUMBER_COLUMN] and written[-1].split(",")[:2] == ["f5", "5"]
    drawn = []
    monkeypatch.setattr(figures, "save_figure", lambda figure, path: drawn.append(figure) or Path(path))
    MatplotlibFigureWriter().write_panels(panels, tmp_path / "fits.png", x_label="frame")
    (figure,) = drawn
    figure.canvas.draw()
    top = figure.axes[0]
    assert top.yaxis.get_major_formatter().get_offset() == ""  # the q of the peak is on the axis, not behind “+9.972e−1”
    assert any(label.get_text().startswith("0.9972") for label in top.get_yticklabels())
    assert len(top.containers) == 1 and top.containers[0].has_yerr  # error bars
    assert top.get_title() == "A" and top.get_legend() is None  # one peak left: named above, not in an empty-looking legend
    assert all(tick == int(tick) for tick in figure.axes[-1].get_xticks())  # frame numbers are whole
    # rows without their number (an older caller): 1, 2, 3 as before
    plain = FitTable("model")
    plain.add({"frame": "a", "chi2": 1.0})
    plain.add({"frame": "b", "chi2": 2.0})
    assert plain.trend("chi2")[0][1].tolist() == [1.0, 2.0]


# -- the page ----------------------------------------------------------------------------------------------------------


def _frames(folder: Path, count: int, stem: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        save_tiff(folder / f"{stem}_{index:03d}.tif", giwaxs_frame(seed=index) * (1.0 + 0.1 * index))
    return folder


def _open(folder: Path, monkeypatch, settings=None):
    from src.gimap.features.analyze.bootstrap import create_analyze_view_model
    from src.gimap.features.analyze.presentation.bindings import batch_finish, batch_run, workspace
    from src.gimap.features.analyze.presentation.page import AnalyzePage
    from src.gimap.integrations.state import InMemoryInstrumentProfileRepository
    from tests.test_analyze_workspace import _app, _context

    _app()
    profile = InstrumentProfile("synthetic", GEOMETRY, None, SHAPE)
    page = AnalyzePage(create_analyze_view_model(_context(InMemoryInstrumentProfileRepository([profile]), settings)))
    page.resize(1400, 900)
    page.show()
    page.set_mode_choice("giwaxs")
    page.add_paths([str(folder)])
    _settle(page, lambda: page.view_model.state.analysis is not None and page.view_model.state.analysis.reduction is not None)
    page.toasts = []
    for module in (batch_finish, batch_run, workspace):
        monkeypatch.setattr(module, "show_toast", lambda parent, text, **options: page.toasts.append((options.get("level"), text, options)))
    return page


def _close(page) -> None:
    page.tasks.wait(30)
    page.dispose()
    page.close()


@pytest.fixture
def page(tmp_path: Path, monkeypatch):
    page = _open(_frames(tmp_path / "run_09", 6, "run_09"), monkeypatch)
    yield page
    _close(page)


def _through_the_dialog(monkeypatch, act=lambda dialog: None, answer: str = "replace") -> list:
    """Batch Export… with Export pressed: ``act(dialog)`` first; a question about an earlier export is answered
    with ``answer`` (and returned: what was asked)."""
    from PyQt5.QtWidgets import QDialog

    from src.gimap.features.analyze.presentation.batch_dialog import BatchExportDialog

    asked = []

    def exec_(dialog):
        act(dialog)
        dialog._ask_earlier = lambda earlier: asked.append(earlier) or answer
        dialog.accept()
        return dialog.result() if dialog.result() == QDialog.Accepted else QDialog.Rejected

    monkeypatch.setattr(BatchExportDialog, "exec_", exec_)
    return asked


def _pressing_enter(monkeypatch, act=lambda dialog: None) -> list:
    """Every answer left as it comes: Batch Export / Send Series with its Export button (after ``act(dialog)``),
    a question with the button Enter presses. Returns ``(question, that button)`` of every question asked."""
    from PyQt5.QtWidgets import QMessageBox

    from src.gimap.features.analyze.presentation.batch_dialog import BatchExportDialog

    asked = []

    def press(box):
        asked.append((box.text(), box.defaultButton().text()))
        box.defaultButton().click()
        return 0

    def export(dialog):
        act(dialog)
        dialog.export_button.click()
        return dialog.result()

    monkeypatch.setattr(QMessageBox, "exec_", press)
    monkeypatch.setattr(BatchExportDialog, "exec_", export)
    return asked


def _wait(page) -> None:
    _settle(page, lambda: not page.batch_running(), timeout_s=120)


def _record(target: Path, stem: str = "run_09") -> dict:
    return json.loads((target / f"{stem}_batch.json").read_text(encoding="utf-8"))


def _on_disk(target: Path) -> list[str]:
    return sorted(path.relative_to(target).as_posix() for path in target.rglob("*") if path.is_file())


def test_the_record_lists_every_file_and_the_dialog_names_the_files_that_are_written(page, tmp_path: Path, monkeypatch) -> None:
    page.view_model.add_region(CutRegion("Ring q 1.100", (1.04, 1.16)))
    page.run_analysis()
    _settle(page, lambda: page.view_model.state.analysis.reduction.curve("region1") is not None)
    out, said = tmp_path / "out", {}

    def choose(dialog):
        dialog.destination_edit.setText(str(out))
        for key in ("per_frame", "fit_input", "q_map", "cake", "frames"):
            dialog.output_checks[key].setChecked(True)
        dialog.fit_peaks_radio.setChecked(True)
        dialog.fit_curves_check.setChecked(True)
        said.update(fit_curves=dialog.fit_curves_file.toolTip() or dialog.fit_curves_file.text(), summary=dialog.summary_label.text(),
                    target=dialog.target_label.text(), button=dialog.export_button.text())

    _through_the_dialog(monkeypatch, choose)
    page.batch_export_dialog()
    _wait(page)
    target = out / "run_09"
    record = _record(target)
    assert sorted(record["files"]) == _on_disk(target) and record["summary"]["files"] == len(record["files"]) > 40
    assert record["format"] == RECORD_FORMAT and record["requested"] == 6 and not record["stopped"] and record["failed"] == []
    assert record["frame_numbers"] == [1, 2, 3, 4, 5, 6] and len(record["cuts"]) == 6
    # the fitted curve of a peak is named after its region, as the dialog said (it used to name the first curve)
    assert said["fit_curves"] == "→ fits/run_09_000_region1_fit.csv" and (target / "fits" / "run_09_000_region1_fit.csv").is_file()
    assert said["button"] == "Export 6 Frames" and str(target)[-20:] in said["target"]
    assert said["summary"].startswith("6 frames · curves: ") and "per frame" in said["summary"] and "about " in said["summary"]
    readme = (target / "README.txt").read_text(encoding="utf-8")
    assert "Result: 6 of 6 frames exported" in readme and "fits/<frame>_region1_fit.csv" in readme
    assert "Chosen, but nothing" not in readme
    fits = (target / "run_09_peak_fits.csv").read_text(encoding="utf-8").splitlines()
    assert next(line for line in fits if not line.startswith("#")).split(",")[:2] == ["frame", NUMBER_COLUMN]
    if hasattr(page.view_model, "series_cut"):  # the ring of the frame on screen, held for every frame, is recorded
        held = record["series_cut"]
        assert held["technique"] == "giwaxs" and len(held["chi_q_window"]) == 2 and held["frame"][1] == 0
        assert record["series_cut_text"].startswith("I(χ) ring: q ") and record["series_cut_text"].endswith("for every frame")
        assert all(cut["chi_q_window"] == held["chi_q_window"] for cut in record["cuts"])
        from src.gimap.shared.export_text import export_text
        assert f"Cuts: {export_text(record['series_cut_text'])}." in readme
    assert page.status_level() == "ok" and "Exported 6/6 frames to" in page.status_text()
    assert str(tmp_path) not in page.status_text()  # the folder by its last names: the line stays one line
    panel = page.batch_panel
    assert panel.title_label.text() == "Batch Export done" and panel.failed_label.isHidden()
    assert f"{len(record['files'])} files written" in panel.detail_label.text() and "curves/" in panel.detail_label.text()
    assert [toast[0] for toast in page.toasts][-1] == "ok" and page.toasts[-1][2]["action"][0] == "Open Folder"


def test_a_stopped_batch_says_that_it_was_stopped_and_how_far_it_got(page, tmp_path: Path) -> None:
    target = tmp_path / "stopped"
    page.run_batch(target, BatchChoices(tables=True, per_frame=True, speed="gentle"), stem="run_09")
    page.cancel_batch()  # the first frame is being reduced and finishes; the other five are dropped
    _wait(page)
    run = page._batch_run
    done = len(run.frames)
    assert 1 <= done < 6 and run.stopped and len(run.not_started) == 6 - done
    status = page.status_text()
    assert page.status_level() == "warning" and status.startswith(f"Stopped: {done} of 6 frames exported to ")
    assert f"({6 - done} not started)" in status and "Exported" not in status
    assert page.batch_panel.title_label.text() == "Batch Export stopped"
    assert page.batch_panel.now_label.text().startswith(f"{done} of 6 frames in 0:")
    assert page.toasts[-1][:2] == ("warning", f"Batch Export stopped: {done} of 6 frames exported")
    assert page.toasts[-1][2]["action"][0] == "Open Folder"
    record = _record(target)
    assert record["stopped"] is True and record["requested"] == 6 and len(record["frames"]) == done
    assert record["not_started"] == [f"run_09_{index:03d}.tif" for index in range(done, 6)]
    assert record["summary"]["exported"] == done and record["summary"]["stopped"] is True
    readme = (target / "README.txt").read_text(encoding="utf-8")
    assert f"Result: {done} of 6 frames exported" in readme and f"Stopped by the user: {6 - done} frames were not started." in readme
    table = (target / "run_09_radial_frames.csv").read_text(encoding="utf-8").splitlines()
    assert f"# the batch ended early: {done} of 6 frames are here" in table


def test_frames_that_fail_are_named_with_their_reason_and_the_batch_is_not_done(tmp_path: Path, monkeypatch) -> None:
    folder = _frames(tmp_path / "run_bad", 5, "run_bad")
    page = _open(folder, monkeypatch)
    try:
        broken = folder / "run_bad_002.tif"
        broken.write_bytes(broken.read_bytes()[:3000])  # cut off while it was copied
        (folder / "run_bad_004.tif").write_bytes(b"")
        target = tmp_path / "out"
        page.run_batch(target, BatchChoices(tables=True, speed="gentle"), stem="run_bad")
        _wait(page)
        run, panel = page._batch_run, page.batch_panel
        assert len(run.frames) == 3 and [item["frame"] for item in run.failed] == ["run_bad_002.tif", "run_bad_004.tif"]
        assert all(str(tmp_path) not in item["why"] and not item["why"].startswith("run_bad") for item in run.failed)
        assert run.failed[1] == {"frame": "run_bad_004.tif", "file": "run_bad_004.tif",
                                 "why": "is empty (0 bytes) — it may still be being written, or the copy failed."}
        status = page.status_text()
        assert page.status_level() == "warning" and "Exported 3/5 frames to " in status and status.endswith("2 帧失败") is False
        assert status.endswith("; 2 failed") and str(tmp_path) not in status and "\n" not in status
        assert panel.title_label.text() == "2 of 5 frames failed"  # not “done”
        assert panel.now_label.text().startswith("3 of 5 frames in 0:")
        listed = panel.failed_label.text().splitlines()
        assert not panel.failed_label.isHidden() and len(listed) == 2
        assert listed[1] == "run_bad_004.tif: is empty (0 bytes) — it may still be being written, or the copy failed."
        assert listed[0].startswith("run_bad_002.tif: could not be read: ") and str(tmp_path) not in listed[0]
        assert page.toasts[-1][:2] == ("warning", "Batch Export ended: 2 of 5 frames failed")
        assert page.toasts[-1][2]["action"][0] == "Open Folder"
        record = _record(target, "run_bad")
        assert [item["file"] for item in record["failed"]] == ["run_bad_002.tif", "run_bad_004.tif"]
        assert record["summary"] == {**record["summary"], "requested": 5, "exported": 3, "failed": 2}
        assert record["frame_numbers"] == [1, 2, 4]  # the place in the list, not the row of the table
        readme = (target / "README.txt").read_text(encoding="utf-8")
        assert "2 frames failed:\n  run_bad_002.tif: could not be read" in readme
        table = (target / "run_bad_radial_frames.csv").read_text(encoding="utf-8").splitlines()
        assert "# 2 frames failed and are missing here: run_bad_002.tif, run_bad_004.tif" in table
        from src.gimap.app.presentation.i18n import apply_language

        try:  # the reasons follow the interface language, the file names stay
            apply_language("zh", [page])
            page.refresh_language()
            listed = panel.failed_label.text().splitlines()
            assert listed[1].startswith("run_bad_004.tif：") and "0 bytes" not in listed[1] and "0 字节" in listed[1]
            # where the language puts the file's name inside the sentence, the sentence alone names the frame
            assert listed[0].startswith("无法读取 run_bad_002.tif：") and not listed[0].startswith("run_bad_002.tif")
            assert panel.title_label.text() == "5 帧中有 2 帧失败"
        finally:
            apply_language("en", [page])
            page.refresh_language()
    finally:
        _close(page)


def test_a_stop_after_the_last_frame_was_started_ends_as_done(page, tmp_path: Path) -> None:
    """Stop drops the frames not started; when every frame had been started, nothing is missing: the batch is done,
    and panel, status line and record say the same."""
    target = tmp_path / "late"
    page.run_batch(target, BatchChoices(tables=True, every=6, speed="gentle"), stem="run_09")  # one frame: started at once
    assert page.batch_running() and not page._batch
    page.cancel_batch()
    _wait(page)
    run = page._batch_run
    assert len(run.frames) == 1 and not run.stopped and run.not_started == []
    assert page.batch_panel.title_label.text() == "Batch Export done"
    assert page.batch_panel.now_label.text().startswith("1 frame in 0:")
    assert page.status_level() == "ok" and page.status_text().startswith("Exported 1/1 frames to ")
    assert _record(target)["stopped"] is False


@needs_permissions
def test_a_folder_that_stops_taking_files_ends_the_batch_and_says_why(page, tmp_path: Path, monkeypatch) -> None:
    """The disk is full, the drive went away: a frame fails to be written, the folder is tried, and the frames not
    started are not reduced only to fail the same way."""
    import errno

    target = tmp_path / "out"
    model = page.view_model
    reduce = model.run_job

    def run_job(job):
        if job.index == 1:
            target.chmod(stat.S_IRUSR | stat.S_IXUSR)  # from now on the folder takes no file
            raise OSError(errno.EACCES, "Permission denied", str(target / "curves"))
        return reduce(job)

    monkeypatch.setattr(model, "run_job", run_job)
    try:
        page.run_batch(target, BatchChoices(tables=True, per_frame=True, speed="gentle"), stem="run_09")
        _wait(page)
    finally:
        target.chmod(stat.S_IRWXU)
    run = page._batch_run
    assert run.write_error == "Permission denied" and len(run.frames) == 1 and len(run.not_started) == 4
    assert page.status_level() == "error" and "takes no more files (Permission denied)" in page.status_text()
    assert page.batch_panel.title_label.text() == "Batch Export stopped: the folder takes no more files"
    (failed,) = run.failed  # the file that could not be written is named from the folder of the batch, not by its whole path
    assert failed["why"] == "[Errno 13] Permission denied: 'curves'" and str(tmp_path) not in page.batch_panel.failed_label.text()


@needs_permissions
def test_an_unwritable_folder_stops_the_batch_before_the_first_frame(page, tmp_path: Path, monkeypatch) -> None:
    from PyQt5.QtWidgets import QDialog

    from src.gimap.features.analyze.presentation.batch_dialog import BatchExportDialog

    locked = _read_only(tmp_path / "locked")
    try:
        before = page._batch_run
        page.run_batch(locked / "run_09", BatchChoices(tables=True, per_frame=True), stem="run_09")
        assert not page.batch_running() and page._batch_run is before  # no frame was reduced for nothing
        assert page.status_level() == "error" and page.status_text().startswith("Cannot write to …/locked/run_09: ")
        assert page.toasts[-1][0] == "error" and page.toasts[-1][2]["action"] is None  # no dialog to go back to
        # in the dialog the folder is said while it is chosen, and Export is off
        seen = {}

        def look(dialog):
            dialog.destination_edit.setText(str(locked))
            seen.update(text=dialog.target_label.text(), role=dialog.target_label.property("gimapRole"),
                        enabled=dialog.export_button.isEnabled())
            dialog.beside_button.click()  # the way out: next to the data
            seen.update(beside=dialog.destination_edit.text(), again=dialog.export_button.isEnabled())
            return QDialog.Rejected

        monkeypatch.setattr(BatchExportDialog, "exec_", look)
        page.batch_export_dialog()
        assert seen["text"] == "This folder cannot be written (no permission to write there): choose another one."
        assert seen["role"] == "warning" and seen["enabled"] is False
        assert seen["beside"] == str(page.view_model.default_export_dir()) and seen["again"] is True
    finally:
        locked.chmod(stat.S_IRWXU)


def test_exporting_again_into_a_folder_asks_and_leaves_no_stale_files(page, tmp_path: Path, monkeypatch) -> None:
    out = tmp_path / "out"

    def first(dialog):
        dialog.destination_edit.setText(str(out))
        dialog.output_checks["per_frame"].setChecked(True)
        dialog.output_checks["fit_input"].setChecked(True)

    asked = _through_the_dialog(monkeypatch, first)
    page.batch_export_dialog()
    _wait(page)
    target = out / "run_09"
    before = _on_disk(target)
    assert asked == [] and len(before) > 30 and "earlier_export" not in _record(target)

    # again, every third frame and no curves for Fitting: Replace
    said = {}

    def second(dialog):
        said.update(note=dialog.folder_note.text(), shown=not dialog.folder_note.isHidden())
        dialog.every_spin.setValue(3)
        dialog.output_checks["fit_input"].setChecked(False)

    asked = _through_the_dialog(monkeypatch, second, "replace")
    page.batch_export_dialog()
    _wait(page)
    assert said["shown"] and said["note"].startswith("This folder already holds an export of these data (") and "6 frames" in said["note"]
    assert len(asked) == 1 and asked[0].frames == 6 and asked[0].complete and len(asked[0].to_remove) == len(before)
    record = _record(target)
    assert sorted(record["files"]) == _on_disk(target)  # nothing of the first run is left beside the new record
    assert len(record["frames"]) == 2 and not (target / "fit_input").exists()
    assert record["earlier_export"]["action"] == "replaced" and record["earlier_export"]["removed"] == len(before)
    assert "An earlier export of the same data in this folder" in (target / "README.txt").read_text(encoding="utf-8")

    # again: Keep both — a new subfolder beside the first, which stays as it is
    kept = _on_disk(target)
    asked = _through_the_dialog(monkeypatch, lambda dialog: dialog.every_spin.setValue(1), "new")
    page.batch_export_dialog()
    _wait(page)
    assert len(asked) == 1 and _on_disk(target) == kept
    second_target = out / "run_09_2"
    assert sorted(_record(second_target)["files"]) == _on_disk(second_target) and len(_record(second_target)["frames"]) == 6
    assert "earlier_export" not in _record(second_target)

    # again: Cancel — the dialog stays open, nothing is written or removed
    from PyQt5.QtWidgets import QDialog

    from src.gimap.features.analyze.presentation.batch_dialog import BatchExportDialog

    results = []

    def cancelled(dialog):
        dialog._ask_earlier = lambda earlier: "cancel"
        dialog.accept()
        results.append((dialog.result(), dialog.replaces_earlier()))
        return QDialog.Rejected

    monkeypatch.setattr(BatchExportDialog, "exec_", cancelled)
    page.batch_export_dialog()
    assert results == [(QDialog.Rejected, False)] and not page.batch_running() and _on_disk(target) == kept

    # a batch started without the question (a script): nothing is removed, and the record says what the folder held
    page.run_batch(target, BatchChoices(tables=True), stem="run_09")
    _wait(page)
    record = _record(target)
    assert record["earlier_export"]["action"] == "kept" and set(kept) - {"README.txt", "run_09_batch.json"} <= set(_on_disk(target))
    assert "Files of it that this export did not write again are still here" in (target / "README.txt").read_text(encoding="utf-8")


def test_a_second_send_to_fitting_does_not_mix_with_the_first(page, tmp_path: Path, monkeypatch) -> None:
    """Fitting ▸ In-situ series lists every curve of the folder it is given: two sends into one folder used to be
    listed, and fitted, as one series."""
    out, opened = tmp_path / "send", []
    page._send_series_to_fitting = lambda folder: opened.append(sorted(path.name for path in Path(folder).glob("*_fit_input.dat")))
    asked = _through_the_dialog(monkeypatch, lambda dialog: dialog.destination_edit.setText(str(out)))
    page.send_series_dialog()
    _wait(page)
    assert asked == [] and len(opened[-1]) == 6

    # the same folder again, with every second frame: the dialog says what is there, and Replace takes it away
    said = {}

    def every_second(dialog):
        dialog.destination_edit.setText(str(out))  # (a send does not keep its folder: the one next to the data is offered)
        said["note"] = dialog.folder_note.text()
        dialog.every_spin.setValue(2)

    asked = _through_the_dialog(monkeypatch, every_second, "replace")
    page.send_series_dialog()
    _wait(page)
    assert "already holds an export of these data" in said["note"] and len(asked) == 1 and len(asked[0].fit_inputs) == 6
    assert opened[-1] == [f"run_09_{index:03d}_fit_input.dat" for index in (0, 2, 4)]  # the three of this send, no other

    # … or Keep both: the new curves in a folder of their own, the old ones where they were
    asked = _through_the_dialog(monkeypatch, lambda dialog: (dialog.destination_edit.setText(str(out)), dialog.every_spin.setValue(3)), "new")
    page.send_series_dialog()
    _wait(page)
    assert len(asked) == 1 and opened[-1] == ["run_09_000_fit_input.dat", "run_09_003_fit_input.dat"]
    assert len(list((out / "run_09_fitting" / "fit_input").glob("*.dat"))) == 3  # a send's folder: <name>_fitting
    assert len(list((out / "run_09_fitting_2" / "fit_input").glob("*.dat"))) == 2

    # curves for Fitting that another export left in the folder (no record of these data) are asked about too
    other = tmp_path / "other" / "run_09_fitting" / "fit_input"
    other.mkdir(parents=True)
    (other / "earlier_fit_input.dat").write_text("# GIMaP Analyze fit input (q in 1/A)\n0.1 1 0.1 5\n", encoding="utf-8")

    def elsewhere(dialog):
        dialog.destination_edit.setText(str(tmp_path / "other"))
        said["curves"] = dialog.folder_note.text()

    asked = _through_the_dialog(monkeypatch, elsewhere, "replace")
    page.send_series_dialog()
    _wait(page)
    assert said["curves"] == ("This folder already holds 1 curves for Fitting: Export asks whether to replace them or to keep "
                              "both.")
    assert len(asked) == 1 and asked[0].record is None and "earlier_fit_input.dat" not in opened[-1] and len(opened[-1]) == 6


def test_send_series_after_a_batch_export_keeps_that_export_when_every_answer_is_the_default(page, tmp_path: Path,
                                                                                            monkeypatch) -> None:
    """Batch Export as offered (next to the data), then Send Series to Fitting, Enter on everything: the send has a
    folder of its own and the export stays whole — a hand-over to Fitting used to remove the tables, curves and
    pictures of the export (its record named the same data). A second send replaces the curves of the first."""
    asked = _pressing_enter(monkeypatch, lambda dialog: [dialog.output_checks[key].setChecked(True)
                                                         for key in ("per_frame", "fit_input", "detector_image")])
    page.batch_export_dialog()
    _wait(page)
    beside = page.view_model.default_export_dir()
    export = beside / "run_09"
    assert page._batch_run.destination == export and asked == []
    made = _on_disk(export)
    assert len(made) > 30 and sorted(_record(export)["files"]) == made
    opened = []
    page._send_series_to_fitting = lambda folder: opened.append(Path(folder))
    asked = _pressing_enter(monkeypatch)
    page.send_series_dialog()
    _wait(page)
    assert asked == [] and _on_disk(export) == made  # not a file of the export removed or written over
    sent = beside / "run_09_fitting"
    assert opened == [sent / "fit_input"] and len(list(opened[0].glob("*_fit_input.dat"))) == 6
    assert sorted(_record(sent)["files"]) == _on_disk(sent)
    page.send_series_dialog()  # again, as it comes: the curves of the first send are replaced, not added to
    _wait(page)
    assert len(asked) == 1 and asked[0][1] == "Replace"
    assert asked[0][0].startswith("“run_09_fitting” already holds an export of these data, made ")
    assert opened[-1] == sent / "fit_input" and len(list(opened[-1].glob("*_fit_input.dat"))) == 6
    assert _record(sent)["earlier_export"]["action"] == "replaced" and _on_disk(export) == made
    # a Batch Export straight into a folder (no subfolder): the send offered there still gets a folder of its own
    flat = tmp_path / "flat"
    _pressing_enter(monkeypatch, lambda dialog: (dialog.destination_edit.setText(str(flat)), dialog.subfolder_check.setChecked(False)))
    page.batch_export_dialog()
    _wait(page)
    kept, record = _on_disk(flat), (flat / "run_09_batch.json").read_text(encoding="utf-8")
    asked = _pressing_enter(monkeypatch)
    page.send_series_dialog()
    _wait(page)
    assert asked == [] and opened[-1] == flat / "run_09_fitting" / "fit_input"
    assert [name for name in _on_disk(flat) if not name.startswith("run_09_fitting/")] == kept
    assert (flat / "run_09_batch.json").read_text(encoding="utf-8") == record


def test_the_folder_and_every_nth_of_a_batch_are_not_carried_to_other_data(tmp_path: Path, monkeypatch) -> None:
    from PyQt5.QtWidgets import QDialog

    from src.gimap.features.analyze.presentation.batch_dialog import BatchExportDialog

    first = _frames(tmp_path / "sample_A", 4, "a")
    second = _frames(tmp_path / "sample_B", 4, "b")
    page = _open(first, monkeypatch)
    try:
        chosen = first / "results"  # a folder of the first data set's own

        def export(dialog):
            dialog.destination_edit.setText(str(chosen))
            dialog.every_spin.setValue(2)

        _through_the_dialog(monkeypatch, export)
        page.batch_export_dialog()
        _wait(page)
        assert page.view_model.batch_origin() == (str(first), "sample_A")
        seen = {}

        def look(dialog):
            seen.update(destination=dialog.destination_edit.text(), every=dialog.every_spin.value(), note=dialog.folder_note.text(),
                        button=dialog.export_button.text(), target=dialog.target_label.toolTip())
            return QDialog.Rejected

        monkeypatch.setattr(BatchExportDialog, "exec_", look)
        page.batch_export_dialog()  # the same data again: as it was left
        assert seen["destination"] == str(chosen) and seen["every"] == 2 and "other data" not in seen["note"]
        page.clear_files()
        page.add_paths([str(second)])
        _settle(page, lambda: page.view_model.state.analysis is not None and page.view_model.state.analysis.path.parent == second)
        page.batch_export_dialog()  # other data: its own folder, and every frame
        assert seen["destination"] == str(second / "gimap_analysis") and seen["every"] == 1 and seen["button"] == "Export 4 Frames"
        assert seen["target"] == str(second / "gimap_analysis" / "sample_B") and seen["note"] == ""
        # a place for results that is no folder of the data: offered again, and said
        results = tmp_path / "results"
        page.view_model.remember_batch(BatchChoices(every=3), results, True, data_folder=first, data_name="sample_A")
        page.batch_export_dialog()
        assert seen["destination"] == str(results) and seen["every"] == 1
        assert seen["note"] == "This is the folder of the last batch, which was made of other data."
    finally:
        _close(page)


def test_the_time_left_does_not_grow_while_a_frame_is_being_reduced() -> None:
    from src.gimap.features.analyze.presentation.bindings.batch_panel import BatchPanelMixin

    panel = SimpleNamespace(_batch_started=100.0, _batch_rate=None, _batch_rate_done=0, _batch_rate_at=100.0)
    left = lambda done, now: BatchPanelMixin._batch_left(panel, done, 6, now)  # noqa: E731
    assert left(0, 102.0) is None  # nothing is known before the first frame is done
    assert left(1, 103.0) == pytest.approx(15.0)  # 3 s a frame, five to go
    assert left(1, 104.0) == pytest.approx(14.0) and left(1, 107.0) == pytest.approx(11.0)  # it only goes down (it grew: 20, 22 …)
    assert left(2, 109.0) == pytest.approx(18.0)  # a frame is done: counted again from what is known now
    assert left(2, 140.0) == 0.0  # never below nothing


def test_the_export_button_is_drawn_as_the_action_of_the_dialog() -> None:
    from src.gimap.app.presentation.theme import theme_color
    from src.gimap.features.analyze.domain import SeriesCorrection
    from src.gimap.features.analyze.presentation.batch_dialog import BatchExportDialog
    from tests.test_analyze_workspace import _app

    app = _app()
    dialog = BatchExportDialog(frames=3, files=3, stem="run", settings_text="GIWAXS", choices=BatchChoices(), destination=Path("."),
                               subfolder=True, series=SeriesCorrection(), curve_options=[("radial", "I(q)")])
    try:
        dialog.show()
        app.processEvents()
        button = dialog.export_button
        assert button.isDefault() and button.isEnabled() and button.property("gimapRole") == "primary"
        image = button.grab().toImage()
        colours: dict = {}
        for x in range(2, image.width() - 2, 3):
            for y in range(2, image.height() - 2, 3):
                colour = image.pixelColor(x, y).name()
                colours[colour] = colours.get(colour, 0) + 1
        assert max(colours, key=colours.get) == theme_color("accent").name()  # as first shown, not after a restyle
    finally:
        dialog.done(0)
        dialog.deleteLater()


def test_the_dialog_says_the_cut_of_every_frame_and_finds_it_again_on_demand() -> None:
    from src.gimap.features.analyze.domain import SeriesCorrection
    from src.gimap.features.analyze.presentation.batch_dialog import BatchExportDialog
    from tests.test_analyze_workspace import _app

    _app()
    state = {"pin": True, "calls": []}

    def cut_text() -> str:
        return "horizontal cut: rows 1170–1175 for every frame" if state["pin"] else "horizontal cut: found again in each frame"

    def set_pin(on: bool) -> None:
        state["pin"] = bool(on)
        state["calls"].append(bool(on))

    def make(**options):
        return BatchExportDialog(frames=8, files=1, stem="scan", settings_text="GISAXS", choices=BatchChoices(), destination=Path("."),
                                 subfolder=True, series=SeriesCorrection(), curve_options=[("horizontal", "Horizontal cut I(qy)")],
                                 gisaxs=True, **options)

    dialog = make(cut_text=cut_text, pin_cuts=True, set_pin_cuts=set_pin, frames_text="3 sums of 3 frames (8 frames; the last sum has only 2)",
                  sum_size=3, notes=["Sum 10 of the automatic analysis was switched off."],
                  cost=lambda choices, frames: (28_000_000, 28_000_000 * frames, 150.0 * frames))
    try:
        dialog.show()
        assert dialog.frames_label.text().startswith("3 sums of 3 frames (8 frames; the last sum has only 2) — ")
        assert dialog.notes_label.text() == "Sum 10 of the automatic analysis was switched off." and not dialog.notes_label.isHidden()
        assert dialog.cut_label.text() == "horizontal cut: rows 1170–1175 for every frame" and not dialog.pin_check.isChecked()
        assert dialog.export_button.text() == "Export 8 Sums"  # eight requests here: what is exported is sums
        assert dialog.every_label.text() == "sum(s): 8 of 8"
        summary = dialog.summary_label.text().splitlines()
        assert summary[0] == ("8 sums of 3 frames · curves: Horizontal cut I(qy) · horizontal cut: rows 1170–1175 for every frame")
        assert summary[1] == "about 28 MB per frame, 224 MB in all · about 20 min · README.txt in the folder says what each file is"
        dialog.pin_check.setChecked(True)  # find the cut again in every frame
        assert state["calls"] == [False] and dialog.cut_label.text() == "horizontal cut: found again in each frame"
        assert "found again in each frame" in dialog.summary_label.text() and dialog.finds_cut_again()
        dialog.every_spin.setValue(4)
        dialog.output_checks["fit_input"].setChecked(True)
        assert dialog.summary_label.text().startswith("2 of 8 sums of 3 frames (Every 4) · ")
        assert "a fit takes: Mean of both halves" in dialog.summary_label.text() and dialog.export_button.text() == "Export 2 Sums"
    finally:
        dialog.done(0)
        dialog.deleteLater()
    plain = make()  # a view model that cannot say the cut: no empty row
    try:
        plain.show()
        assert plain.cut_label.isHidden() and plain.cut_caption.isHidden() and plain.pin_check.isHidden()
        assert plain.summary_label.text().splitlines()[0] == "8 frames · curves: Horizontal cut I(qy)"
    finally:
        plain.done(0)
        plain.deleteLater()


def test_the_cut_box_of_a_cancelled_dialog_changes_nothing(page, monkeypatch) -> None:
    from PyQt5.QtWidgets import QDialog

    from src.gimap.features.analyze.presentation.batch_dialog import BatchExportDialog

    model = page.view_model
    if not hasattr(model, "set_pin_series_cuts"):
        pytest.skip("this view model has one cut per frame only")
    assert model.pin_series_cuts is True
    seen = {}

    def tried(dialog):
        seen["before"] = dialog.cut_label.text()
        dialog.pin_check.setChecked(True)
        seen.update(after=dialog.cut_label.text(), pinned=model.pin_series_cuts)
        return QDialog.Rejected

    monkeypatch.setattr(BatchExportDialog, "exec_", tried)
    page.batch_export_dialog()
    assert seen["before"].startswith("I(χ) ring: q ") and seen["before"].endswith("for every frame")
    assert seen["after"] == "I(χ) ring: the strongest ring of each frame" and seen["pinned"] is False
    assert model.pin_series_cuts is True  # cancelled: as it was
    _through_the_dialog(monkeypatch, lambda dialog: (dialog.destination_edit.setText(str(page.view_model.state.files[0].parent / "o")),
                                                      dialog.pin_check.setChecked(True)))
    page.batch_export_dialog()
    _wait(page)
    assert model.pin_series_cuts is False  # exported with it: the choice is kept
    record = _record(page._batch_run.destination)
    assert record["series_cut"] is None and record["series_cut_text"] == "I(χ) ring: the strongest ring of each frame"
    assert all(cut["chi_q_window"] for cut in record["cuts"])  # where each frame's ring was found
    model.set_pin_series_cuts(True)


# -- sums ----------------------------------------------------------------------------------------------------------


def test_a_sum_is_called_a_sum_and_a_last_sum_of_fewer_frames_is_marked(tmp_path: Path, monkeypatch) -> None:
    from PyQt5.QtWidgets import QDialog

    from src.gimap.features.analyze.presentation.batch_dialog import BatchExportDialog
    from tests.test_analyze_nexus_sources import _ring_stack
    from tests.test_analyze_workspace import _app, _context
    from tests.test_review_analyze_run_and_clear import _profiles

    from src.gimap.features.analyze.bootstrap import create_analyze_view_model
    from src.gimap.features.analyze.presentation.page import AnalyzePage

    _app()
    stack = _ring_stack(tmp_path / "raw" / "film.h5", 8)
    page = AnalyzePage(create_analyze_view_model(_context(_profiles())))
    try:
        page.resize(1400, 900)
        page.show()
        page.set_mode_choice("giwaxs")
        page.add_paths([str(stack)])
        _settle(page, lambda: page.view_model.state.analysis is not None and page.view_model.state.analysis.reduction is not None)
        page.sum_spin.setValue(3)
        _settle(page, lambda: page.view_model.state.analysis is not None and len(page.view_model.state.analysis.summed_frames) == 2)
        assert page.data_batch_button.text() == "Batch Export 8 Frames in Sums of 3…"
        seen = {}

        def look(dialog):
            seen.update(frames=dialog.frames_label.text(), button=dialog.export_button.text(),
                        file=dialog.output_files["per_frame"].toolTip() or dialog.output_files["per_frame"].text())
            return QDialog.Rejected

        monkeypatch.setattr(BatchExportDialog, "exec_", look)
        page.batch_export_dialog()
        assert seen["frames"].startswith("3 sums of 3 frames (8 frames; the last sum has only 2) — ")
        assert seen["button"] == "Export 3 Sums" and seen["file"] == "→ curves/film_frame0000_sum3_radial.csv"
        target = tmp_path / "out"
        page.run_batch(target, BatchChoices(tables=True, per_frame=True, fit_input=True), stem="film")
        _wait(page)
        names = sorted(path.name for path in (target / "curves").glob("*_radial.csv"))
        assert names == ["film_frame0000_sum3_radial.csv", "film_frame0003_sum3_radial.csv", "film_frame0006_sum2_radial.csv"]
        record = _record(target, "film")
        assert record["frames"] == ["film.h5 frame 1 of 8 (+2 summed)", "film.h5 frame 4 of 8 (+2 summed)",
                                    "film.h5 frame 7 of 8 (+1 summed)"]
        assert [item["frames_summed"] for item in record["sums"]] == [3, 3, 2] and record["frame_numbers"] == [1, 4, 7]
        table = (target / "film_radial_frames.csv").read_text(encoding="utf-8").splitlines()
        assert ("# every column is the sum of 3 frames: its intensities are sums over them; other sums: "
                "film.h5 frame 7 of 8 (+1 summed) (2)") in table
        curve = (target / "curves" / "film_frame0006_sum2_radial.csv").read_text(encoding="utf-8").splitlines()
        assert "# frame: 7 of 8 (index 6)" in curve
        assert "# sum of 2 frames: 7 to 8 of 8 (index 6 to 7); intensities are sums over these frames" in curve
        readme = (target / "README.txt").read_text(encoding="utf-8")
        assert "Sums: every frame here is the sum of 3 frames of the source" in readme and "(+1 summed) (2)" in readme
        fit_input = (target / "fit_input" / "film_frame0003_sum3_fit_input.dat").read_text(encoding="utf-8").splitlines()
        assert "# frame: 4 of 8 (index 3)" in fit_input

        # the sum Run Automatic Analysis sets for its own result is not the person's: switched off, and said
        if hasattr(page.view_model, "sum_set_by"):
            page.view_model.set_sum_count(5, by="automatic")
            seen.clear()

            def noted(dialog):
                seen.update(frames=dialog.frames_label.text(), notes=dialog.notes_label.text(), button=dialog.export_button.text())
                return QDialog.Rejected

            monkeypatch.setattr(BatchExportDialog, "exec_", noted)
            page.batch_export_dialog()
            assert page.view_model.sum_count == 1 and page.sum_spin.value() == 1
            assert seen["notes"] == "Sum 5 of the automatic analysis was switched off for the series (Data ▸ Sum N to sum again)."
            assert seen["frames"].startswith("8 frames from 1 file") and seen["button"] == "Export 8 Frames"
            page.tasks.wait(30)
    finally:
        _close(page)


def test_the_dialog_says_the_cut_of_the_frame_read_again_without_the_automatic_sum(page, tmp_path: Path, monkeypatch) -> None:
    """Run Automatic Analysis left a Sum on screen; Batch Export switches it off and the frame on screen is read
    again. Until it is, the dialog does not name a cut and Export waits: the cut it names is the cut every frame
    gets — not the summed frame's, and not one that depends on how fast Export is pressed."""
    from PyQt5.QtWidgets import QDialog

    from src.gimap.features.analyze.presentation.batch_dialog import BatchExportDialog

    model = page.view_model
    if not hasattr(model, "sum_set_by") or not hasattr(model, "series_cut_text"):
        pytest.skip("this view model has no automatic sum or no cut held for a series")
    model.set_sum_count(2, by="automatic")  # as Run Automatic Analysis leaves it — here the Sum box still shows 1
    page.run_analysis()
    _settle(page, lambda: model.state.analysis is not None and len(model.state.analysis.summed_frames) == 1)
    assert page.sum_spin.value() == 1
    seen = {}

    def exec_(dialog):
        seen.update(first=dialog.cut_label.text(), enabled=dialog.export_button.isEnabled(), summary=dialog.summary_label.text())
        dialog.accept()  # Enter while the frame is read: nothing happens yet
        seen["early"] = dialog.result()
        _settle(page, lambda: not model.state.analysis.summed_frames and dialog.export_button.isEnabled(), timeout_s=30)
        seen.update(then=dialog.cut_label.text(), summary_then=dialog.summary_label.text())
        dialog.destination_edit.setText(str(tmp_path / "out"))
        dialog.export_button.click()
        return dialog.result()

    monkeypatch.setattr(BatchExportDialog, "exec_", exec_)
    page.batch_export_dialog()
    _wait(page)
    assert seen["first"] == "the frame on screen is read again without the sum …" and seen["enabled"] is False
    assert seen["first"] in seen["summary"] and seen["early"] == QDialog.Rejected
    single = page._batch_cut_text()
    assert seen["then"] == single and single.startswith("I(χ) ring: q ") and single in seen["summary_then"]
    record = _record(tmp_path / "out" / "run_09")
    assert record["series_cut_text"] == single and len(record["frames"]) == 6
    assert all(cut["chi_q_window"] == list(model.series_cut()["chi_q_window"]) for cut in record["cuts"])


def test_a_set_of_module_files_that_lacks_a_part_is_said_before_it_is_exported(tmp_path: Path) -> None:
    from src.gimap.features.analyze.bootstrap import create_analyze_view_model
    from src.gimap.features.analyze.presentation.page import AnalyzePage
    from tests.test_analyze_nexus_sources import _module
    from tests.test_analyze_workspace import _app, _context
    from tests.test_review_analyze_run_and_clear import _profiles

    _app()
    for number in (1, 2, 4):  # m03 is missing
        _module(tmp_path / "raw" / f"scan_00007_m{number:02d}.nxs", number, frames=5)
    whole = tmp_path / "whole"
    for number in (1, 2, 3):
        _module(whole / f"scan_00008_m{number:02d}.nxs", number, frames=5)
    page = AnalyzePage(create_analyze_view_model(_context(_profiles())))
    try:
        page.add_paths([str(tmp_path / "raw")])
        page.tasks.wait(30)
        requests = page.view_model.batch_requests()
        text, size, notes = page._batch_frames(requests)
        assert len(requests) == 5 and size == 1 and text == "5 frames from 1 scan of 3 module files"
        assert notes == ["scan_00007: a module file is missing or cannot be read yet — every frame lacks that part of the detector."]
        page.clear_files()
        page.add_paths([str(whole)])
        page.tasks.wait(30)
        assert page._batch_frames(page.view_model.batch_requests()) == ("5 frames from 1 scan of 3 module files", 1, [])
        # a set whose files are still growing: the dialog says how many frames the batch takes now
        page.view_model.settling = lambda path: 4.0
        assert page._batch_frames(page.view_model.batch_requests())[2] == [
            "scan_00008 is still arriving: the batch takes the 5 frames that are there now."]
        # the record says which files every source is made of
        run = SimpleNamespace(series_cut=None, series_cut_text="", sources=[])
        page._batch_describe(run)
        (source,) = run.sources
        assert source["name"] == "scan_00008" and source["frames"] == 5 and source["complete"] is True
        assert source["files"] == [f"scan_00008_m{number:02d}.nxs" for number in (1, 2, 3)]
        assert source["summary"].startswith("3 module files · 5 frames")
        recorded = _run(tmp_path, sources=run.sources)
        assert batch_record(recorded, [])["sources"] == run.sources
        assert "Source scan_00008: 3 module files * 5 frames" in readme_text(recorded.choices, batch="run", frame="a", frames=1,
                                                                            curves=[], run=recorded)
    finally:
        _close(page)


# -- the words -------------------------------------------------------------------------------------------------------


def test_every_sentence_of_the_flow_has_its_chinese() -> None:
    from src.gimap.app.presentation.i18n import ZH, apply_language
    from src.gimap.features.analyze.application import NO_SUCH_PLACE
    from src.gimap.features.analyze.presentation import batch_summary, batch_texts
    from src.gimap.features.analyze.presentation.bindings import batch_export
    from src.gimap.features.analyze.presentation.views.batch_export_view import BatchExportView  # noqa: F401 - its texts below

    said = [
        batch_summary.CANNOT_WRITE, batch_summary.EARLIER_EXPORT, batch_summary.EARLIER_CURVES, batch_summary.CARRIED,
        NOT_A_FOLDER, NO_SUCH_PLACE, NO_PERMISSION, batch_export.SUM_SWITCHED_OFF, batch_export.SOURCE_PART_MISSING,
        batch_export.SOURCE_ARRIVING, "… and {n} more sources that are not whole",
        "Cuts", "Find the cut again in every frame", "Next to the Data", "Summary", "Files go to {folder}",
        "Save into the folder gimap_analysis next to the data files",
        "“{folder}” already holds an export of these data, made {when} ({frames} frames).",
        "“{folder}” already holds {n} curves for Fitting.",
        "Fitting ▸ In-situ series lists every curve of that folder: the old and the new ones would be fitted as one series.",
        "Replace removes the {n} files of the earlier export first.",
        "It was written by an earlier version of GIMaP, whose record lists only its tables: the files of single frames stay.",
        "Keep Both writes this export into the new subfolder “{name}”.", "Replace", "Keep Both in a New Subfolder",
        batch_summary.READING, batch_summary.REPLACE_TAKES, batch_summary.REPLACE_TAKES_OTHERS, batch_summary.LISTED,
        *batch_summary.KIND_WORDS.values(),
        "{n} s", "{n} min", "{n} h", "1 frame", "{count} frames", "{count} of {total} frames (Every {n})",
        "{count} sums of {size} frames", "{count} of {total} sums of {size} frames (Every {n})", "+ {n} more", "curves: {names}",
        "a fit takes: {halves}", "about {time}", "about {size} per frame, {total} in all",
        "about {total} in all", "sum(s): {count} of {total}",
        "README.txt in the folder says what each file is", "Export 1 Sum", "Export {count} Sums",
        "Batch Export {count} Frames in Sums of {n}…", "{groups} sums of {n} frames ({frames} frames)",
        "{groups} sums of {n} frames ({frames} frames; the last sum has only {k})",
        "{groups} sums of {n} frames ({frames} frames; {short} sums at the end of a source hold fewer)",
        "{frames} frames from 1 scan of {modules} module files", "Choose Another Folder…",
        "The earlier export in {folder} could not be removed: {error}", "Series map {done}/{total}",
        "Series map {done}/{total}: {name} …", "{done} of {total} frames in {time}", "1 frame in {time}", "{frame}: {reason}",
        "… and {n} more: README.txt in the folder has every one", "tables and records {n}",
        "{n} files written: {where}", "{n} of {total} frames failed",
        "Batch Export stopped: the folder takes no more files", "Batch Export stopped: the folder takes no more files ({error})",
        "Batch Export ended: {n} of {total} frames failed", "Batch Export stopped: {done} of {total} frames exported",
        "Batch Export: the tables of all frames could not be written ({error})",
        "Stopped: {folder} takes no more files ({error}); {done} of {total} frames were exported",
        "Exported {done}/{total} frames to {folder}, but the tables of all frames could not be written: {error}",
        "Stopped: {done} of {total} frames exported to {folder} in {seconds} s ({left} not started)",
    ]
    assert [text for text in said if text not in ZH] == []
    run = SimpleNamespace(frames=["a"], failed=[{"frame": "b", "why": "is empty (0 bytes) — it may still be being written, or the copy failed.",
                                                 "file": "b.tif"}], requested=4, stopped=True, not_started=["c", "d"], write_error="",
                          destination=Path("/data/sample/gimap_analysis/run"), fit_frames=[], files=[])
    assert batch_texts.short_folder(run.destination) == "…/gimap_analysis/run"
    assert batch_texts.batch_end_text(run, seconds=4.2) == (
        "Stopped: 1 of 4 frames exported to …/gimap_analysis/run in 4 s (2 not started); 1 failed", "warning")
    assert batch_texts.end_title(run, stopped=True) == "Batch Export stopped"
    full = SimpleNamespace(**{**vars(run), "write_error": "No space left on device"})
    assert batch_texts.batch_end_text(full, seconds=4.2)[1] == "error"
    assert batch_texts.end_title(full) == "Batch Export stopped: the folder takes no more files"
    assert batch_texts.batch_end_text(run, seconds=4.2, tables_error="Permission denied")[1] == "error"
    try:
        apply_language("zh")
        assert batch_texts.failed_lines(run) == "b：是空的（0 字节）——可能还在写入，或者复制失败了。"
        assert batch_texts.end_toast_text(run, stopped=True) == "批量导出已停止：4 帧中导出了 1 帧"
        peaks = SimpleNamespace(frames=["a.tif"], last_fit="Ring A: q 0.9973 Å⁻¹, FWHM 0.0110; Ring B: no significant peak "
                                                           "(area < 3 × its error)")
        latest = batch_texts.latest_fit_text(peaks)  # each peak's own words: none of them stays English
        assert "Ring A: q 0.9973 Å⁻¹, FWHM 0.0110; Ring B: " in latest and "no significant peak" not in latest
        assert batch_summary.duration_text(40) == "40 秒" and batch_summary.duration_text(7200) == "2.0 小时"
    finally:
        apply_language("en")
    assert batch_summary.size_text(28_400_000) == "28 MB" and batch_summary.size_text(1_340_000_000) == "1.3 GB"
    assert batch_summary.size_text(72_000) == "72 kB" and batch_summary.duration_text(200) == "3 min"
    # the reason of a frame: without the folder of the data, and without the file's name in front
    request = SimpleNamespace(path=Path("/data/run/f_003.tif"), summed_frames=())
    assert batch_texts.failure_reason(request, "f_003.tif could not be read: the file looks damaged (ValueError: Empty TIFF "
                                               "detector image: /data/run/f_003.tif).") == (
        "could not be read: the file looks damaged (ValueError: Empty TIFF detector image: f_003.tif).", "f_003.tif")
    assert batch_texts.failure_reason(request, "No geometry.") == ("No geometry.", "f_003.tif")
    full = f"[Errno 28] No space left on device: '{Path('/out/run') / 'curves' / 'f_003_radial.csv'}'"
    assert batch_texts.failure_reason(request, full, Path("/out/run")) == (  # a file of the batch: from its folder
        f"[Errno 28] No space left on device: '{Path('curves') / 'f_003_radial.csv'}'", "f_003.tif")
    summed = SimpleNamespace(path=Path("/data/run/f_003.tif"), summed_frames=((Path("/data/other/g_001.tif"), 0),))
    assert batch_texts.failure_reason(summed, "Cannot read /data/other/g_001.tif") == ("Cannot read g_001.tif", "f_003.tif")
    assert time.monotonic() > 0  # (keeps the import: used by the page tests' helpers)
