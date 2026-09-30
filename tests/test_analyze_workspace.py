"""Analyze workspace: open a CBF/NXS and get curves with zero clicks (phase-2 gate)."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PyQt5.QtWidgets import QApplication, QMessageBox

from src.gimap.app import AppContext
from src.gimap.features.analyze.application import (
    AnalysisRequest,
    AnalyzeFrame,
    geometry_from_fitting_settings,
)
from src.gimap.features.analyze.bootstrap import create_analyze_view_model
from src.gimap.features.analyze.infrastructure.adapters import DetectorIoFrameSource
from src.gimap.integrations.state import (
    InMemoryInstrumentProfileRepository,
    InMemorySessionRepository,
    InMemorySettingsRepository,
    InMemoryUserPreferencesRepository,
)
from src.gimap.shared.geometry import DetectorGeometry, InstrumentProfile

ROOT = Path(__file__).resolve().parents[1]
CBF = ROOT / "TestSAXSdata" / "jg_gisaxs_4nm_old_3ml_insitu_ds03_00001_00033.cbf"
NXS_FOLDER = ROOT / "TestSAXSdata" / "lmbdp03"
NXS = NXS_FOLDER / "jg_gisaxs_12nm_insitu_00001r3_00001_m01.nxs"

requires_data = pytest.mark.skipif(
    not (CBF.is_file() and NXS.is_file()), reason="TestSAXSdata frames are not available"
)

_APP = None


def _app() -> QApplication:
    global _APP
    _APP = QApplication.instance() or QApplication([])
    return _APP


def _pilatus_profile() -> InstrumentProfile:
    geometry = DetectorGeometry(172e-6, 172e-6, 4.2, 798.0, 1308.0, 1.0332, 0.4)
    return InstrumentProfile("P03 Pilatus", geometry, "PILATUS 2M", (1679, 1475))


def _lambda_profile() -> InstrumentProfile:
    geometry = DetectorGeometry(55e-6, 55e-6, 4.2, 1571.0, 4000.0, 1.0332, 0.4)
    return InstrumentProfile("P03 Lambda", geometry, "Lambda", (4727, 3142))


def _context(profiles=None, settings=None) -> AppContext:
    return AppContext(
        settings=InMemorySettingsRepository(settings or {}),
        session=InMemorySessionRepository(),
        preferences=InMemoryUserPreferencesRepository(),
        instrument_profiles=profiles,
    )


@requires_data
def test_folders_expand_to_frames_and_nxs_module_series_collapse_to_one() -> None:
    frames = DetectorIoFrameSource().expand([NXS_FOLDER, ROOT / "TestSAXSdata", CBF])
    names = [path.name for path in frames]
    assert names.count(NXS.name) == 1
    assert not any("_m02" in name for name in names)
    assert names.count(CBF.name) == 1
    cbfs = [name for name in names if name.endswith(".cbf")]
    assert cbfs == sorted(cbfs)


@requires_data
@pytest.mark.parametrize(
    ("path", "profile", "shape"),
    [(CBF, _pilatus_profile(), (1679, 1475)), (NXS, _lambda_profile(), (4727, 3142))],
)
def test_matched_profile_gives_gisaxs_curves_without_input(path, profile, shape) -> None:
    analyze = AnalyzeFrame(DetectorIoFrameSource(), InMemoryInstrumentProfileRepository([profile]))
    analysis = analyze(AnalysisRequest(path))

    assert analysis.shape == shape
    assert analysis.resolution.how == "matched"
    assert analysis.kind == "gisaxs"
    horizontal = analysis.reduction.curve("horizontal")
    vertical = analysis.reduction.curve("vertical")
    assert len(horizontal.x) > 100 and len(vertical.x) > 100
    assert np.all(np.isfinite(horizontal.intensity)) and np.all(horizontal.pixels > 0)
    # qy spans both sides of the specular column; qz starts at the sample horizon.
    assert horizontal.x.min() < 0 < horizontal.x.max()
    k = profile.geometry.wavevector_inv_angstrom
    assert vertical.x.min() >= k * np.sin(profile.geometry.incidence_rad) - 1e-3


@requires_data
def test_reusing_the_loaded_frame_gives_identical_curves() -> None:
    analyze = AnalyzeFrame(
        DetectorIoFrameSource(), InMemoryInstrumentProfileRepository([_pilatus_profile()])
    )
    first = analyze(AnalysisRequest(CBF))
    again = analyze(AnalysisRequest(CBF), loaded=first)
    assert again.data is first.data
    np.testing.assert_array_equal(
        again.reduction.curve("horizontal").intensity, first.reduction.curve("horizontal").intensity
    )


@requires_data
def test_without_profile_the_frame_loads_and_reports_the_missing_geometry() -> None:
    analyze = AnalyzeFrame(DetectorIoFrameSource(), InMemoryInstrumentProfileRepository())
    analysis = analyze(AnalysisRequest(CBF))

    assert analysis.reduction is None and analysis.geometry is None
    assert "No instrument profile for PILATUS 2M (1679×1475)" in analysis.messages[0]


def test_fitting_settings_convert_to_the_canonical_frame() -> None:
    values = {
        ("fitting", "detector.distance"): 1456.7,
        ("fitting", "detector.beam_center_x"): 797.48,
        ("fitting", "detector.beam_center_y"): 370.75,
        ("fitting", "detector.pixel_size_x"): 172.0,
        ("fitting", "detector.pixel_size_y"): 172.0,
        ("beam", "wavelength"): 0.1033,
        ("beam", "grazing_angle"): 0.4,
    }

    def read(section, key, default=None):
        return values.get((section, key), default)

    geometry = geometry_from_fitting_settings(read, 1679)
    # Fitting counts the beam row from the bottom: canonical y = rows - y - 0.5.
    assert geometry.beam_center_x_px == pytest.approx(797.98)
    assert geometry.beam_center_y_px == pytest.approx(1679 - 370.75 - 0.5)
    assert geometry.distance_m == pytest.approx(1.4567)
    assert geometry.wavelength_angstrom == pytest.approx(1.033)
    assert geometry.incidence_deg == 0.4

    values[("fitting", "gisaxs_input.flip_ud")] = True
    flipped = geometry_from_fitting_settings(read, 1679)
    assert flipped.beam_center_y_px == pytest.approx(370.75 + 0.5)

    del values[("fitting", "detector.distance")]
    assert geometry_from_fitting_settings(read, 1679) is None


def _page(context):
    from src.gimap.features.analyze.presentation.page import AnalyzePage

    _app()
    return AnalyzePage(create_analyze_view_model(context))


@requires_data
def test_page_shows_curves_after_opening_a_cbf_with_zero_clicks(tmp_path: Path) -> None:
    frame = tmp_path / CBF.name
    shutil.copy2(CBF, frame)
    page = _page(_context(InMemoryInstrumentProfileRepository([_pilatus_profile()])))
    shown = []
    page.analysisShown.connect(shown.append)

    page.add_paths([tmp_path])
    assert page.tasks.wait(60)

    assert len(shown) == 1
    assert page.banner.isHidden()
    assert page.top_plot.curve_count() == 1 and page.bottom_plot.curve_count() == 1
    assert page.detector_view.has_image()
    assert page.detector_view.horizontal_band.isVisible()
    assert "P03 Pilatus" in page.summary_label.text()
    assert "curves" in page.status_text()

    page.export_current()
    exported = sorted(path.name for path in (tmp_path / "gimap_analysis").iterdir())
    stem = frame.stem
    assert exported == [
        f"{stem}_analysis.json",
        f"{stem}_fit_input.dat",
        f"{stem}_horizontal.csv",
        f"{stem}_vertical.csv",
    ]
    record = json.loads((tmp_path / "gimap_analysis" / f"{stem}_analysis.json").read_text("utf-8"))
    assert record["instrument_profile"]["name"] == "P03 Pilatus"
    assert record["measurement"] == "gisaxs"
    data = np.loadtxt(tmp_path / "gimap_analysis" / f"{stem}_horizontal.csv", delimiter=",", skiprows=6)
    assert data.shape[1] == 4 and data.shape[0] > 100
    page.close()


@requires_data
def test_dragging_the_horizontal_band_recomputes_the_cut(tmp_path: Path) -> None:
    page = _page(_context(InMemoryInstrumentProfileRepository([_pilatus_profile()])))
    page.add_paths([CBF])
    assert page.tasks.wait(60)

    page.detector_view.horizontalBandChanged.emit(600.0, 610.0)
    assert page.tasks.wait(60)

    horizontal = page.view_model.state.analysis.reduction.curve("horizontal")
    assert horizontal.region["rows"] == (600, 610)
    assert page.view_model.state.analysis.reduction.markers["horizontal_source"] == "manual"
    page.close()


@requires_data
def test_missing_profile_offers_the_previous_fitting_geometry(monkeypatch) -> None:
    settings = {
        "fitting": {
            "detector": {
                "distance": 4200.0,
                "beam_center_x": 797.48,
                "beam_center_y": 370.75,
                "pixel_size_x": 172.0,
                "pixel_size_y": 172.0,
            }
        },
        "beam": {"wavelength": 0.10332, "grazing_angle": 0.4},
    }
    profiles = InMemoryInstrumentProfileRepository()
    page = _page(_context(profiles, settings))
    page.add_paths([CBF])
    assert page.tasks.wait(60)
    assert not page.banner.isHidden()
    assert not page.use_fitting_button.isHidden()
    assert "797.98" in page.use_fitting_button.toolTip()
    assert page.top_plot.curve_count() == 0

    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *args, **kwargs: QMessageBox.Yes))
    page.use_fitting_button.click()
    assert page.tasks.wait(60)

    profile = profiles.find("PILATUS 2M 1679×1475")
    assert profile is not None and profile.detector_shape == (1679, 1475)
    assert profile.geometry.beam_center_y_px == pytest.approx(1679 - 370.75 - 0.5)
    assert page.banner.isHidden()
    assert page.top_plot.curve_count() == 1
    assert page.profile_combo.findText("PILATUS 2M 1679×1475") >= 0
    page.close()


@requires_data
def test_apply_to_all_files_exports_every_listed_frame(tmp_path: Path) -> None:
    sources = sorted((ROOT / "TestSAXSdata").glob("jg_gisaxs_4nm_old_3ml_insitu_ds03_00001_*.cbf"))
    for source in sources:
        shutil.copy2(source, tmp_path / source.name)
    page = _page(_context(InMemoryInstrumentProfileRepository([_pilatus_profile()])))
    page.add_paths([tmp_path])
    assert page.tasks.wait(60)

    page.apply_to_all()
    for _ in range(len(sources) + 1):
        assert page.tasks.wait(60)

    exported = list((tmp_path / "gimap_analysis" / "curves").glob("*_horizontal.csv"))  # per-frame curves: curves/
    assert len(exported) == len(sources)
    assert page.batch_button.isEnabled()
    assert f"Exported {len(sources)}/{len(sources)}" in page.status_text()
    page.close()


def test_folder_watch_reports_a_frame_only_after_its_size_settles(tmp_path: Path) -> None:
    from src.gimap.features.analyze.presentation.folder_watch import FolderWatch

    frames = DetectorIoFrameSource()
    (tmp_path / "old.cbf").write_bytes(b"x" * 10)
    watch = FolderWatch(frames.expand, frames.settled_size)
    watch.start(tmp_path)
    assert watch.poll() == []

    new = tmp_path / "new_00001.cbf"
    new.write_bytes(b"x" * 10)
    assert watch.poll() == []  # first sighting
    new.write_bytes(b"x" * 20)
    assert watch.poll() == []  # still growing
    assert watch.poll() == [new.resolve()]
    assert watch.poll() == []  # reported once

    watch.stop()
    (tmp_path / "later.cbf").write_bytes(b"x")
    assert watch.poll() == []


@requires_data
def test_watching_follows_new_frames_and_exports_them(tmp_path: Path) -> None:
    page = _page(_context(InMemoryInstrumentProfileRepository([_pilatus_profile()])))
    page.start_watch(tmp_path)
    page.auto_export_check.setChecked(True)
    assert page.poll_watch() == []

    shutil.copy2(CBF, tmp_path / CBF.name)
    assert page.poll_watch() == []  # size seen once
    added = page.poll_watch()
    assert [path.name for path in added] == [CBF.name]
    assert page.tasks.wait(60)

    assert page.file_list.count() == 1 and page.file_list.currentRow() == 0
    assert page.top_plot.curve_count() == 1
    assert (tmp_path / "gimap_analysis" / f"{CBF.stem}_horizontal.csv").is_file()
    page.stop_watch()
    assert not page.watch_button.isChecked()
    page.dispose()
    page.close()


@requires_data
def test_fit_hands_the_signed_cut_and_the_chosen_half_to_fitting(tmp_path: Path) -> None:
    from src.gimap.features.analyze.presentation.page import AnalyzePage

    _app()
    frame = tmp_path / CBF.name
    shutil.copy2(CBF, frame)
    sent = []
    context = _context(InMemoryInstrumentProfileRepository([_pilatus_profile()]))
    page = AnalyzePage(
        create_analyze_view_model(context), send_to_fitting=lambda path, side: sent.append((path, side))
    )
    assert not page.fit_button.isEnabled()
    page.add_paths([frame])
    assert page.tasks.wait(60)
    assert page.fit_button.isEnabled()
    assert page.fit_side_actions["both_abs"].isChecked()

    page.fit_button.click()

    path, side = sent[0]
    assert path == tmp_path / "gimap_analysis" / f"{frame.stem}_fit_input.dat"
    assert side == "both_abs"  # Fitting shows both halves on |qy| in two colours
    table = np.loadtxt(path, comments="#")
    horizontal = page.view_model.state.analysis.reduction.curve("horizontal")
    assert table.shape == (len(horizontal.x), 4)  # q, I, sigma, pixels
    np.testing.assert_array_equal(table[:, 3], horizontal.pixels)
    np.testing.assert_allclose(table[:, 0], horizontal.x, rtol=1e-7)
    assert table[:, 0].min() < 0 < table[:, 0].max()

    # The user's choice is remembered in the Analyze settings.
    page.fit_side_actions["positive"].trigger()
    assert context.settings.get("analyze", "fit_side") == "positive"
    page.fit_button.click()
    assert sent[-1][1] == "positive"
    page.dispose()
    page.close()


def test_header_centre_is_used_only_on_request_and_a_session_centre_wins() -> None:
    from src.gimap.features.analyze.application import ResolveGeometry

    resolve = ResolveGeometry(InMemoryInstrumentProfileRepository([_pilatus_profile()]))
    common = dict(detector_name="PILATUS 2M", shape=(1679, 1475), header_center=(700.0, 1200.0))

    ignored = resolve(**common)
    assert ignored.center_source == "profile"
    assert ignored.geometry.beam_center_x_px == 798.0
    assert ignored.header_center == (700.0, 1200.0)

    header = resolve(**common, use_header_center=True)
    assert header.center_source == "header"
    assert (header.geometry.beam_center_x_px, header.geometry.beam_center_y_px) == (700.0, 1200.0)

    session = resolve(**common, use_header_center=True, beam_center=(810.5, 1300.0))
    assert session.center_source == "session"
    assert session.geometry.beam_center_x_px == 810.5


def test_header_centres_outside_the_frame_are_placeholders() -> None:
    from src.gimap.features.analyze.application import header_beam_center

    shape = (1679, 1475)
    assert header_beam_center({"header_beam_center": (798.0, 1308.0)}, shape) == (798.0, 1308.0)
    assert header_beam_center({"header_beam_center": (1e9, 0.0)}, shape) is None
    assert header_beam_center({"header_beam_center": None}, shape) is None
    assert header_beam_center({}, shape) is None


def test_cbf_beam_xy_header_becomes_the_file_header_centre() -> None:
    from src.gimap.shared.detector_io.metadata import extract_cbf_metadata

    header = {"_array_data.header_contents": "# Detector: PILATUS 2M\n# Beam_xy (812.50, 1302.00) pixels\n"}
    metadata = extract_cbf_metadata(header, (1679, 1475))
    assert metadata["header_beam_xy_px"] == (812.5, 1302.0)


@requires_data
def test_session_beam_centre_follows_new_frames_until_reset_or_saved(tmp_path: Path) -> None:
    sources = sorted((ROOT / "TestSAXSdata").glob("jg_gisaxs_4nm_old_3ml_insitu_ds03_00001_*.cbf"))[:2]
    for source in sources:
        shutil.copy2(source, tmp_path / source.name)
    profiles = InMemoryInstrumentProfileRepository([_pilatus_profile()])
    page = _page(_context(profiles))
    page.add_paths([tmp_path])
    assert page.tasks.wait(60)
    assert page.center_button.text().endswith("profile")

    page.set_session_center(805.0, 1300.0)
    assert page.tasks.wait(60)
    geometry = page.view_model.state.analysis.geometry
    assert (geometry.beam_center_x_px, geometry.beam_center_y_px) == (805.0, 1300.0)
    assert page.center_button.property("centerSource") == "session"
    assert page.save_center_action.isEnabled()

    # The next frame keeps the session centre (the calibrated one is not reapplied).
    page.file_list.setCurrentRow(1)
    assert page.tasks.wait(60)
    assert page.view_model.state.analysis.geometry.beam_center_x_px == 805.0

    page.save_center()
    assert page.tasks.wait(60)
    saved = profiles.find("P03 Pilatus")
    assert (saved.geometry.beam_center_x_px, saved.geometry.beam_center_y_px) == (805.0, 1300.0)
    assert saved.geometry.incidence_deg == 0.4
    assert page.view_model.state.beam_center is None
    assert page.center_button.property("centerSource") == "profile"

    page.set_session_center(790.0, 1310.0)
    assert page.tasks.wait(60)
    page.reset_center()
    assert page.tasks.wait(60)
    assert page.view_model.state.analysis.geometry.beam_center_x_px == 805.0
    page.dispose()
    page.close()


_FIT_E2E_SCRIPT = r"""
import json, shutil, sys, time
from pathlib import Path

from PyQt5.QtWidgets import QApplication

from tests.test_analyze_workspace import CBF, _context, _pilatus_profile
from src.gimap.integrations.jobs import LocalProcessJobRunner
from src.gimap.integrations.state import InMemoryInstrumentProfileRepository


def main():
    app = QApplication.instance() or QApplication([])
    from main import MainWindow

    frame = Path(sys.argv[1]) / CBF.name
    shutil.copy2(CBF, frame)
    context = _context(InMemoryInstrumentProfileRepository([_pilatus_profile()]))
    context.jobs = LocalProcessJobRunner()
    window = MainWindow(context)
    deadline = time.monotonic() + 30
    while not (getattr(window, "_initialization_completed", False) and hasattr(window, "runtime")):
        app.processEvents()
        if time.monotonic() > deadline:
            raise TimeoutError("GIMaP did not start")
    start_page = window.mainWindowWidget.currentIndex()
    page = window.components.analyze_page
    page.add_paths([frame])
    page.tasks.wait(60)
    page.fit_button.click()
    fitting = window.runtime.fitting
    single = {
        "data_source": fitting.data_source,
        "file": fitting.current_1d_data["file_path"],
        "points": len(fitting.current_1d_data["q"]),
        "q_view": window.fitQViewModeComboBox.currentData(),
        "curve_name": window.components.fitting_workspace.curve_card.name_label.text(),
    }
    # Send Series to Fitting: every listed frame is exported, then the series opens.
    second = Path(sys.argv[1]) / ("second_" + CBF.name)
    shutil.copy2(CBF, second)
    page.add_paths([second])
    page.tasks.wait(60)
    destination = Path(sys.argv[1]) / "gimap_analysis"
    page.apply_to_all(destination, then=page._send_series_to_fitting)
    page.tasks.wait(120)
    workspace = window.components.fitting_workspace
    series = workspace.insitu_series_page.ui.workflowControls
    print(json.dumps({
        **single,
        "series_context": workspace.context_stack.currentWidget() is workspace.insitu_series_page,
        "series_folder": series.sequenceFolderEdit.text(),
        "series_pattern": series.sequencePatternEdit.text(),
        "fit_inputs": sorted(path.name for path in (destination / "fit_input").glob("*_fit_input.dat")),
        "start_is_home": start_page == window.mainWindowWidget.indexOf(window.components.home_page),
        "page_after_fit": window.mainWindowWidget.currentIndex(),
        "status": page.status_text(),
    }))
    window.close()
    context.jobs.shutdown()


if __name__ == "__main__":
    main()
"""


@requires_data
def test_fit_opens_the_curve_and_the_series_in_fitting(tmp_path: Path) -> None:
    """End to end through the real window, in a fresh process (one window per process)."""
    import subprocess
    import sys

    script = tmp_path / "fit_e2e.py"
    script.write_text(_FIT_E2E_SCRIPT, encoding="utf-8")
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen", PYTHONIOENCODING="utf-8")
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    completed = subprocess.run(
        [sys.executable, str(script), str(tmp_path)],
        cwd=ROOT, env=env, capture_output=True, timeout=180,
        encoding="utf-8", errors="replace",  # the locale codec (GBK) cannot decode the log
    )
    assert completed.returncode == 0, completed.stderr[-2000:]
    result_line = next(
        line for line in reversed(completed.stdout.splitlines()) if line.startswith("{")
    )
    result = json.loads(result_line)
    assert result["start_is_home"]  # a new user starts on the Start page
    assert result["page_after_fit"] == 2
    assert result["data_source"] == "1d"
    assert result["file"].endswith("_fit_input.dat")
    assert result["points"] > 100
    assert result["q_view"] == "fold"
    assert result["curve_name"].endswith("_fit_input.dat")
    assert "Exported 2/2 frames" in result["status"]
    assert result["series_context"]
    assert Path(result["series_folder"]) == tmp_path / "gimap_analysis" / "fit_input"
    assert result["series_pattern"] == "*_fit_input.dat"
    assert result["fit_inputs"] == sorted(
        [f"{CBF.stem}_fit_input.dat", f"second_{CBF.stem}_fit_input.dat"]
    )


def test_analyze_choices_are_remembered_between_sessions() -> None:
    from src.gimap.features.analyze.presentation.page import AnalyzePage

    _app()
    profiles = InMemoryInstrumentProfileRepository([_pilatus_profile()])
    context = _context(profiles)
    first = AnalyzePage(create_analyze_view_model(context))
    first.mode_combo.setCurrentIndex(first.mode_combo.findData("giwaxs"))
    first._mode_chosen(first.mode_combo.currentIndex())
    first.profile_combo.setCurrentIndex(first.profile_combo.findData("P03 Pilatus"))
    first._profile_chosen(first.profile_combo.currentIndex())
    first.incidence_spin.setValue(0.25)
    first.auto_export_check.setChecked(True)
    first._remember(last_folder=str(ROOT))
    first.dispose()
    first.close()

    second = AnalyzePage(create_analyze_view_model(context))
    state = second.view_model.state
    assert (state.mode, state.profile_name, state.incidence_deg) == ("giwaxs", "P03 Pilatus", 0.25)
    assert second.mode_combo.currentData() == "giwaxs"
    assert second.profile_combo.currentData() == "P03 Pilatus"
    assert second.incidence_spin.value() == pytest.approx(0.25)
    assert second.auto_export_check.isChecked()
    assert second._last_folder == str(ROOT)

    profiles.delete("P03 Pilatus")  # a stale profile choice falls back to automatic
    third = AnalyzePage(create_analyze_view_model(context))
    assert third.view_model.state.profile_name is None
    for page in (second, third):
        page.dispose()
        page.close()


@requires_data
def test_edit_updates_the_profile_in_use_and_can_delete_it(monkeypatch) -> None:
    from src.gimap.features.analyze.presentation.geometry_dialog import GeometryDialog

    profiles = InMemoryInstrumentProfileRepository([_pilatus_profile()])
    page = _page(_context(profiles))
    page.add_paths([CBF])
    assert page.tasks.wait(60)

    def edit_and_save(dialog):
        assert dialog.profile_name() == "P03 Pilatus"
        assert dialog.delete_button is not None
        dialog.distance_spin.setValue(3000.0)
        dialog.incidence_spin.setValue(0.25)
        return GeometryDialog.Accepted

    monkeypatch.setattr(GeometryDialog, "exec_", edit_and_save)
    page.edit_geometry_button.click()
    assert page.tasks.wait(60)
    edited = profiles.find("P03 Pilatus").geometry
    assert edited.distance_m == pytest.approx(3.0) and edited.incidence_deg == pytest.approx(0.25)
    assert profiles.find("P03 Pilatus").detector_shape == (1679, 1475)  # binding kept
    assert page.view_model.state.analysis.geometry.distance_m == pytest.approx(3.0)

    monkeypatch.setattr(GeometryDialog, "exec_", lambda dialog: GeometryDialog.DELETED)
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *args, **kwargs: QMessageBox.Yes))
    page.edit_geometry_button.click()
    assert page.tasks.wait(60)
    assert profiles.load_all() == []
    assert not page.banner.isHidden()  # no geometry any more
    assert page.use_fitting_button.isHidden()  # and no former Fitting geometry stored
    page.dispose()
    page.close()


@requires_data
def test_image_and_plots_are_saved_as_publication_figures(tmp_path: Path) -> None:
    frame = tmp_path / CBF.name
    shutil.copy2(CBF, frame)
    page = _page(_context(InMemoryInstrumentProfileRepository([_pilatus_profile()])))
    page.add_paths([frame])
    assert page.tasks.wait(60)

    state = page.detector_view.display_state()
    image = state.pop("image")
    png = page.view_model.save_image_figure(tmp_path / "detector.png", image, **state)
    assert png.stat().st_size > 10_000 and png.read_bytes()[:4] == b"\x89PNG"

    plot = page.top_plot.figure_state()
    curves = plot.pop("curves")
    svg = page.view_model.save_curves_figure(tmp_path / "cut.svg", curves, **plot)
    assert "<svg" in svg.read_text(encoding="utf-8")[:400]
    page.dispose()
    page.close()
