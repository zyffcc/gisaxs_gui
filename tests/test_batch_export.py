"""Batch Export: from raw frames to a folder in a few clicks — tables with every frame a column,
per-frame files, remembered choices, and settings files that bring a whole set-up back."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from src.gimap.features.analyze.application import (
    AnalyzeSettings,
    BatchChoices,
    CurveTables,
    CutRegion,
    MaskShape,
    batch_stem,
    settings_from_record,
    settings_record,
)
from src.gimap.features.analyze.domain import Corrections, Curve, GiwaxsSettings, Sector
from src.gimap.shared.geometry import InstrumentProfile
from tests.test_assistant_calibration import SHAPE, giwaxs_frame, save_tiff
from tests.test_giwaxs_workspace import GEOMETRY, _settle


def _curve(key: str, x, y) -> Curve:
    x = np.asarray(x, dtype=float)
    return Curve(key, f"{key} title", x, np.asarray(y, dtype=float), np.ones(x.size), np.full(x.size, 10), "q (Å⁻¹)")


class _Writer:
    def __init__(self):
        self.tables = {}

    def write_table(self, path, comments, header, rows):
        self.tables[Path(path).name] = (list(comments), list(header), [list(row) for row in rows])
        return Path(path)


def test_tables_put_every_frame_in_a_column_exactly_or_say_they_interpolated(tmp_path: Path) -> None:
    tables = CurveTables()
    x = np.linspace(0.1, 1.0, 10)
    tables.add("frame 1", [_curve("radial", x, x * 1.0), _curve("region1", x[:5], x[:5])])
    tables.add("frame 2", [_curve("radial", x, x * 2.0)])
    writer = _Writer()
    tables.write(writer, tmp_path, "run")
    comments, header, rows = writer.tables["run_radial_frames.csv"]
    assert header == ["q (Å⁻¹)", "frame 1", "frame 2"] and len(rows) == 10
    assert rows[3] == pytest.approx([x[3], x[3], 2 * x[3]]) and "exact" in comments[2]
    shifted = CurveTables()
    shifted.add("a", [_curve("radial", x, x)])
    shifted.add("b", [_curve("radial", x + 0.01, x + 0.01)])
    writer = _Writer()
    shifted.write(writer, tmp_path, "run")
    comments, _header, rows = writer.tables["run_radial_frames.csv"]
    assert "interpolated" in comments[2] and rows[-1][2] == pytest.approx(x[-1])  # y = x on its own grid
    assert np.isnan(rows[0][2])  # never extrapolated


def test_choices_and_names() -> None:
    choices = BatchChoices(curves=("region1",), tables=True, per_frame=True)
    assert BatchChoices.from_dict(choices.to_dict()) == choices
    assert BatchChoices.from_dict({"tables": "yes", "unknown": 1}) == BatchChoices()  # bad values: defaults
    assert BatchChoices(tables=False).writes_anything is False and BatchChoices(q_map=True).needs_map
    assert batch_stem([Path("d/run_001.tif"), Path("d/run_002.tif")]) == "d"
    assert batch_stem([Path("d/one.nxs")]) == "one"


def test_a_settings_file_brings_the_whole_set_up_back() -> None:
    settings = AnalyzeSettings(
        mode="giwaxs", profile_name="P03", profile=InstrumentProfile("P03", GEOMETRY, None, SHAPE), incidence_deg=0.4,
        beam_center=(120.5, 300.25), sum_count=3,
        corrections=Corrections(gap_guard_px=2, mirror_fill=True, minimum=0.0,
                                mask_shapes=(MaskShape("rectangle", ((1, 2), (30, 40))),)),
        giwaxs=GiwaxsSettings(in_plane_half_width_deg=8.0, bins=900, chi_q_window=(1.0, 1.2),
                              sector=Sector(10.0, 30.0, 0.5, None), regions=(CutRegion("Ring q 1.100", (1.08, 1.12)),)),
        export=BatchChoices(curves=("region1",)).to_dict(),
    )
    record = json.loads(json.dumps(settings_record(settings)))
    assert settings_from_record(record) == settings
    with pytest.raises(ValueError):
        settings_from_record({"format": "something else"})


@pytest.fixture
def batch_page(tmp_path: Path):
    from src.gimap.features.analyze.bootstrap import create_analyze_view_model
    from src.gimap.features.analyze.presentation.page import AnalyzePage
    from src.gimap.integrations.state import InMemoryInstrumentProfileRepository
    from tests.test_analyze_workspace import _app, _context

    _app()
    folder = tmp_path / "run_07"
    folder.mkdir()
    for index in range(3):
        save_tiff(folder / f"run_07_{index:03d}.tif", giwaxs_frame(seed=index) * (1.0 + 0.2 * index))
    profile = InstrumentProfile("synthetic", GEOMETRY, None, SHAPE)
    page = AnalyzePage(create_analyze_view_model(_context(InMemoryInstrumentProfileRepository([profile]))))
    page.resize(1400, 900)
    page.show()
    page.set_mode_choice("giwaxs")
    page.add_paths([str(folder)])
    _settle(page, lambda: page.view_model.state.analysis is not None and page.view_model.state.analysis.reduction is not None)
    yield page
    page.tasks.wait(30)
    page.dispose()
    page.close()


def test_batch_export_writes_tables_and_per_frame_files_to_the_chosen_folder(batch_page, tmp_path: Path, monkeypatch) -> None:
    from PyQt5.QtWidgets import QDialog

    from src.gimap.features.analyze.presentation.batch_dialog import BatchExportDialog

    page = batch_page
    page.view_model.add_region(CutRegion("Ring q 1.100", (1.07, 1.13)))
    page.run_analysis()
    _settle(page, lambda: page.view_model.state.analysis.reduction.curve("region1_chi") is not None)
    out = tmp_path / "results"
    seen = {}

    def accept(dialog):
        seen["frames"] = dialog.frames_label.text()
        seen["curves"] = [dialog.curve_list.item(index).data(0x0100) for index in range(dialog.curve_list.count())]
        dialog.destination_edit.setText(str(out))
        dialog.output_checks["per_frame"].setChecked(True)
        dialog._check_all(False)
        for index in range(dialog.curve_list.count()):  # only the region's two curves
            item = dialog.curve_list.item(index)
            if str(item.data(0x0100)).startswith("region1"):
                item.setCheckState(2)
        seen["target"] = dialog.target_label.toolTip()  # the label may shorten a long path; the tooltip has it all
        seen["files"] = {key: label.text() for key, label in dialog.output_files.items()}
        return QDialog.Accepted

    monkeypatch.setattr(BatchExportDialog, "exec_", accept)
    page.batch_export_dialog()
    _settle(page, lambda: not page._batch and (out / "run_07" / "run_07_batch.json").exists())
    assert "3 frames" in seen["frames"] and {"radial", "region1", "region1_chi"} <= set(seen["curves"])
    assert str(out / "run_07") in seen["target"]
    assert seen["files"]["tables"] == "→ run_07_region1_frames.csv"  # every option says the file it writes
    assert seen["files"]["per_frame"] == "→ curves/run_07_000_region1.csv"
    target = out / "run_07"
    with (target / "run_07_region1_chi_frames.csv").open() as stream:
        rows = [row for row in csv.reader(line for line in stream if not line.startswith("#"))]
    assert rows[0][0].startswith("|chi|") and len(rows[0]) == 4  # x and three frames
    assert not (target / "run_07_radial_frames.csv").exists()  # not chosen
    assert len(list((target / "curves").glob("run_07_00?_region1.csv"))) == 3  # per frame, in their own folder
    readme = (target / "README.txt").read_text(encoding="utf-8")
    assert "run_07_<curve>_frames.csv" in readme and "curves/<frame>_<curve>.csv" in readme
    record = json.loads((target / "run_07_batch.json").read_text(encoding="utf-8"))
    assert len(record["frames"]) == 3 and record["choices"]["curves"] == ["region1", "region1_chi"]
    assert "Exported 3/3" in page.status_text()
    choices, destination, subfolder = page.view_model.batch_preferences()  # remembered for next time
    assert choices.per_frame and destination == str(out) and subfolder


def test_settings_saved_and_loaded_on_the_page(batch_page, tmp_path: Path) -> None:
    page = batch_page
    page.view_model.add_region(CutRegion("Spot", (0.38, 0.42), (0.0, 20.0)))
    page.mirror_fill_check.setChecked(True)
    page.in_plane_spin.setValue(12.0)
    _settle(page, lambda: page.view_model.state.corrections.mirror_fill)
    path = page.save_settings(tmp_path / "settings.json")
    assert json.loads(path.read_text(encoding="utf-8"))["profile_name"] == "synthetic"
    page.view_model.set_regions([])
    page.mirror_fill_check.setChecked(False)
    page.in_plane_spin.setValue(10.0)
    _settle(page)
    assert page.load_settings(path)
    _settle(page, lambda: len(page._region_rows) == 5)
    assert page.view_model.state.giwaxs.regions[0].name == "Spot"
    assert page.mirror_fill_check.isChecked() and page.in_plane_spin.value() == 12.0
    assert page.view_model.state.giwaxs.in_plane_half_width_deg == 12.0


def test_from_the_start_page_a_folder_goes_straight_to_the_batch_dialog(tmp_path: Path, monkeypatch) -> None:
    from PyQt5.QtWidgets import QDialog, QFileDialog

    from src.gimap.features.analyze.bootstrap import create_analyze_view_model
    from src.gimap.features.analyze.presentation.batch_dialog import BatchExportDialog
    from src.gimap.features.analyze.presentation.page import AnalyzePage
    from src.gimap.integrations.state import InMemoryInstrumentProfileRepository
    from tests.test_analyze_workspace import _app, _context

    _app()
    folder = tmp_path / "raw"
    folder.mkdir()
    for index in range(2):
        save_tiff(folder / f"f_{index}.tif", giwaxs_frame(seed=index))
    profile = InstrumentProfile("synthetic", GEOMETRY, None, SHAPE)
    page = AnalyzePage(create_analyze_view_model(_context(InMemoryInstrumentProfileRepository([profile]))))
    page.set_mode_choice("giwaxs")
    opened = []
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", staticmethod(lambda *args, **kwargs: str(folder)))
    monkeypatch.setattr(BatchExportDialog, "exec_", lambda dialog: opened.append(dialog.frames_label.text()) or QDialog.Rejected)
    page.batch_from_folder()
    _settle(page, lambda: bool(opened))
    assert "2 frames" in opened[0] and not page._batch_pending
    page.tasks.wait(30)
    page.dispose()
    page.close()


def test_every_nth_frame_is_remembered_and_counted() -> None:
    from PyQt5.QtWidgets import QDialog

    from src.gimap.features.analyze.domain import SeriesCorrection
    from src.gimap.features.analyze.presentation.batch_dialog import BatchExportDialog
    from tests.test_analyze_workspace import _app

    _app()
    assert BatchChoices.from_dict(BatchChoices(every=5).to_dict()).every == 5
    assert BatchChoices.from_dict({"every": 0}).every == 1 and BatchChoices.from_dict({"every": True}).every == 1
    dialog = BatchExportDialog(
        frames=403, files=1, stem="run", settings_text="", choices=BatchChoices(every=10), destination=Path("."),
        subfolder=True, series=SeriesCorrection(), curve_options=[("radial", "I(q)")],
    )
    assert dialog.frame_count() == 41 and "41" in dialog.export_button.text() and dialog.choices().every == 10
    dialog.every_spin.setValue(1)
    assert "403" in dialog.export_button.text()
    dialog.done(QDialog.Rejected)


def test_batch_export_buttons_appear_with_a_series(batch_page) -> None:
    page = batch_page
    assert not page.batch_export_button.isHidden() and not page.data_batch_button.isHidden()
    assert "3" in page.data_batch_button.text() and page.series_batch_button.isEnabled()
    page.clear_files()
    assert page.batch_export_button.isHidden() and page.data_batch_button.isHidden()


def test_formats_folders_fits_and_converted_frames(batch_page, tmp_path: Path) -> None:
    """Text in tabs, pictures as SVG, frames converted to NumPy, the ring peak fitted in every frame."""
    page = batch_page
    page.view_model.add_region(CutRegion("Ring q 1.100", (1.04, 1.16)))
    page.run_analysis()
    _settle(page, lambda: page.view_model.state.analysis.reduction.curve("region1") is not None)
    choices = BatchChoices(
        curves=("region1",), tables=True, per_frame=True, text_format="txt", detector_image=True, image_format="svg",
        frames=True, frame_format="npy", fit="peaks", fit_profile="gaussian", fit_start="previous", fit_curves=True,
    )
    target = tmp_path / "out"
    page.run_batch(target, choices, stem="run_07")
    _settle(page, lambda: not page._batch and (target / "README.txt").exists())
    table = (target / "run_07_region1_frames.txt").read_text()
    assert "\t" in table.splitlines()[-1]  # tab separated
    assert len(list((target / "curves").glob("*.txt"))) == 3 and len(list((target / "images").glob("*.svg"))) == 3
    frames = sorted((target / "frames").glob("*.npy"))
    assert len(frames) == 3 and np.load(frames[0]).shape == SHAPE  # the detector data, as read
    fits = (target / "run_07_peak_fits.txt").read_text().splitlines()
    header = next(line for line in fits if not line.startswith("#")).split("\t")
    rows = [line.split("\t") for line in fits if not line.startswith("#")][1:]
    q = [float(row[header.index("Ring q 1.100 q (1/A)")]) for row in rows]
    assert len(rows) == 3 and all(abs(value - 1.10) < 0.01 for value in q)  # the ring of the synthetic film
    assert (target / "run_07_peak_fits.png").exists() and len(list((target / "fits").glob("*_fit.txt"))) == 3
    readme = (target / "README.txt").read_text(encoding="utf-8")
    assert "frames/<frame>.npy" in readme and "run_07_peak_fits.txt" in readme


def test_the_model_fit_starts_from_the_previous_frame() -> None:
    from src.gimap.features.analyze.application import FitStarts, FitTable, fit_model, model_row

    calls = []

    def fitter(q, intensity, sigma, *, components=(), distance_nm=None, cancelled=None):
        calls.append((components, distance_nm))
        return [{"combination": "sphere", "best_chi2_weighted": 1.1, "best_log_rmse": 0.1,
                 "components": [{"type": "sphere", "weight": 1.0, "params": {"R": 5.0 + len(calls), "D": 50.0 + len(calls)}}]}]

    class Frame:  # just what fit_model reads
        reduction = object()

    import src.gimap.features.analyze.application.batch_fit as batch_fit

    curve = Curve("fit_input", "cut", np.linspace(0.01, 0.2, 50), np.ones(50), np.ones(50), np.ones(50), "q")
    original = batch_fit.fit_input_curve
    batch_fit.fit_input_curve = lambda _analysis: curve
    try:
        starts, table = FitStarts("previous"), FitTable("model")
        for index in range(3):
            table.add(model_row(f"frame {index + 1}", fit_model(Frame(), fitter, "auto", starts)))
    finally:
        batch_fit.fit_input_curve = original
    assert calls[0] == ((), None) and calls[1] == ((1,), 51.0) and calls[2] == ((1,), 52.0)  # the previous frame's model and D
    assert [row["c1 sphere R"] for row in table.rows] == [6.0, 7.0, 8.0]
    fresh = FitStarts("fresh")
    fresh.remember("model", {"family": 1, "D": 51.0})
    assert fresh.for_key("model") is None
