"""Sputter ▸ Show in Analyze: the raw frames of one time window, summed, in Analyze.

The reader of a window (its frames in order, the geometry of the result, clear errors), the menus of the Sputter page
(Windows table, Structure Sketch, Size Trend), the hand-over in the composition root with a fake Analyze page, and once
with the real Analyze page on invented TIFF frames, whose sum is known. Invented names and values only.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtCore import QPoint
from PyQt5.QtWidgets import QApplication

from src.gimap.app.presentation.i18n import DEFAULT_LANGUAGE, apply_language
from src.gimap.app.sputter_handover import SputterHandoverMixin
from src.gimap.features.sputter.infrastructure import BinnedWindowReader
from tests.sputter_fixtures import scan_folder, windows_folder


@pytest.fixture(autouse=True)
def _english():
    apply_language(DEFAULT_LANGUAGE)
    yield
    apply_language(DEFAULT_LANGUAGE)


def _scan(tmp_path, frames: int = 10):
    folder = scan_folder(tmp_path, "zz_invented_scan_00001", frames=frames)
    return folder, sorted(path.name for path in folder.glob("*.cbf"))


def _settle(condition=lambda: False, seconds: float = 10.0) -> None:
    end = time.monotonic() + seconds
    while time.monotonic() < end and not condition():
        QApplication.processEvents()
        time.sleep(0.01)
    QApplication.processEvents()


# -- the reader ------------------------------------------------------------------------------------------------------


def test_a_window_gives_its_frames_in_order_and_the_geometry_of_the_result(tmp_path):
    scan, names = _scan(tmp_path)
    folder = windows_folder(tmp_path / "result", scan, names)
    window = BinnedWindowReader().read(folder, "bin_000002")  # in the second chunk
    assert window.window == "bin_000002" and window.count == 3
    assert window.frames == tuple(scan / name for name in names[5:8])
    assert (window.first_time_s, window.last_time_s) == pytest.approx((0.25, 0.35))
    assert window.detector == "PILATUS 1M"  # the invented header
    geometry = window.geometry
    assert (geometry.sdd_mm, geometry.alpha_i_deg, geometry.wavelength_angstrom) == pytest.approx((1000.0, 0.35, 1.0))
    assert geometry.beam_center is None  # estimated by the model: never set in Analyze
    assert geometry.estimated_beam == pytest.approx((11.5, 21.5))  # 0-based pixel centres + 0.5: GIMaP's coordinates
    given = windows_folder(tmp_path / "given", scan, names, beam=(100.25, 200.75), estimated=None,
                           request=dict(wavelength_nm=None, energy_ev=12398.419843320026))
    window = BinnedWindowReader().read(given, "bin_000000")
    assert window.frames == (scan / names[0], scan / names[1])
    assert window.geometry.beam_center == (100.25, 200.75) and window.geometry.estimated_beam is None
    assert window.geometry.wavelength_angstrom == pytest.approx(1.0)


def test_a_window_that_cannot_be_shown_says_why(tmp_path):
    from src.gimap.app.presentation.i18n.zh import ZH
    from src.gimap.features.sputter.domain import (FRAMES_MISSING, NO_MEMBERS, NO_SCAN, NO_WINDOWS, SCAN_MISSING,
                                                   UNKNOWN_WINDOW, WINDOW_TEMPLATES)
    from src.gimap.features.sputter.presentation.texts import message_text

    scan, names = _scan(tmp_path)
    folder = windows_folder(tmp_path / "result", scan, names)
    reader = BinnedWindowReader()
    with pytest.raises(ValueError) as unknown:
        reader.read(folder, "bin_000099")
    assert str(unknown.value) == UNKNOWN_WINDOW.format(window="bin_000099", folder=folder)
    (scan / names[6]).unlink()
    (scan / names[7]).unlink()
    with pytest.raises(FileNotFoundError) as missing:
        reader.read(folder, "bin_000002")
    assert str(missing.value) == FRAMES_MISSING.format(missing=2, count=3, window="bin_000002", folder=scan,
                                                       name=names[6])
    assert reader.read(folder, "bin_000000").count == 2  # the other windows are still there
    chunk = folder / "binned" / "chunk_00000.json"
    rows = json.loads(chunk.read_text(encoding="utf-8"))["rows"]
    rows[0]["raw_members"] = ["../" + names[0]]  # never a file outside the scan folder
    chunk.write_text(json.dumps({"rows": rows}), encoding="utf-8")
    with pytest.raises(ValueError) as outside:
        reader.read(folder, "bin_000000")
    assert str(outside.value) == NO_MEMBERS.format(folder=folder, window="bin_000000")
    moved = windows_folder(tmp_path / "moved", tmp_path / "zz_not_here", names)
    with pytest.raises(FileNotFoundError) as gone:
        reader.read(moved, "bin_000000")
    assert str(gone.value) == SCAN_MISSING.format(folder=tmp_path / "zz_not_here")
    (folder / "gimap_request.json").unlink()
    with pytest.raises(FileNotFoundError) as unknown_scan:
        reader.read(folder, "bin_000000")
    assert str(unknown_scan.value) == NO_SCAN.format(folder=folder)
    bare = windows_folder(tmp_path / "bare", scan, names)
    for path in (bare / "binned").glob("chunk_*.json"):
        path.unlink()
    with pytest.raises(FileNotFoundError) as no_windows:
        reader.read(bare, "bin_000000")
    assert str(no_windows.value) == NO_WINDOWS.format(folder=bare)
    # Every problem is said in the interface language, the names and numbers kept.
    assert all(template in ZH for template in WINDOW_TEMPLATES)
    try:
        apply_language("zh")
        said = message_text(missing.value)
        assert said.startswith("时间窗 bin_000002 的 3 帧中有 2 帧不在") and names[6] in said
    finally:
        apply_language(DEFAULT_LANGUAGE)


# -- the hand-over (composition root) with a fake Analyze page --------------------------------------------------------


def _profile(name, distance_mm, wavelength, detector=None):
    from src.gimap.shared.geometry import DetectorGeometry, InstrumentProfile

    return InstrumentProfile(name, DetectorGeometry(172e-6, 172e-6, distance_mm * 1e-3, 10.0, 20.0, wavelength, 0.2),
                             detector)


def test_a_profile_is_chosen_only_within_half_a_millimetre_and_a_tenth_of_a_percent():
    from src.gimap.app.sputter_handover import matching_profile

    near = _profile("zz near", 1000.4, 1.0009)
    far = _profile("zz far", 1000.6, 1.0)
    other_energy = _profile("zz other energy", 1000.0, 1.0011)
    assert matching_profile([far, other_energy, near], 1000.0, 1.0) is near
    assert matching_profile([far, other_energy], 1000.0, 1.0) is None
    assert matching_profile([near], None, 1.0) is None
    detector = _profile("zz detector", 1000.0, 1.0, "PILATUS  1M")
    assert matching_profile([near, detector], 1000.0, 1.0, "Pilatus 1M") is detector  # the frames' detector first
    assert matching_profile([near, detector], 1000.0, 1.0, "Eiger") is near


class FakeAnalyze:
    """What the hand-over uses of the Analyze page: ``summed_locked``, ``show_summed``, the files and the status."""

    def __init__(self, files=(), locked="", accepts=True):
        self.view_model = SimpleNamespace(state=SimpleNamespace(files=[Path(path) for path in files]))
        self.locked, self.accepts, self.calls = locked, accepts, []

    def summed_locked(self):
        return self.locked

    def show_summed(self, paths, **options):
        self.calls.append((list(paths), options))
        if self.accepts:
            self.view_model.state.files = [Path(path) for path in paths]
            options["on_imported"]()
        return self.accepts

    def status_text(self):
        return "an invented reason of the status line"


class Composed(SputterHandoverMixin):
    """The composed window as far as the hand-over needs it (``context``: a real ``AppContext``)."""

    def __init__(self, page, profiles=(), context=None):
        self.analyze_page, self.requested = page, []
        if context is None:
            context = SimpleNamespace(instrument_profiles=SimpleNamespace(load_all=lambda: list(profiles)))
        self.ui = SimpleNamespace(app_context=context)

    def _page_requested(self, key):
        self.requested.append(key)


@pytest.fixture
def toasts(monkeypatch):
    from src.gimap.app import sputter_handover

    said = []
    monkeypatch.setattr(sputter_handover, "show_toast", lambda parent, text, **options: said.append((text, options)))
    return said


def test_the_hand_over_replaces_the_frames_sums_them_and_names_the_geometry(tmp_path, toasts):
    scan, names = _scan(tmp_path)
    window = BinnedWindowReader().read(windows_folder(tmp_path / "result", scan, names), "bin_000001")
    page = FakeAnalyze(files=[tmp_path / "zz_other_00001.cbf"])
    composed = Composed(page, [_profile("zz far", 1003.0, 1.0), _profile("zz SAXS", 1000.0, 1.0)])
    assert composed._show_sputter_window(window) is True
    assert composed.requested == ["analyze"]
    (paths, options), = page.calls
    assert paths == list(window.frames) and len(paths) == 3
    assert options["incidence_deg"] == 0.35 and options["beam_center"] is None and options["profile_name"] == "zz SAXS"
    (text, shown), = toasts
    assert shown["level"] == "ok" and shown["timeout_ms"] > 0
    assert text.split("\n") == [
        "Window bin_000001 (0.10–0.20 s): the sum of its 3 frames is shown in Analyze.",
        "Instrument profile “zz SAXS”: distance 1000 mm, λ 1.0000 Å, as in Sputter.",
        "αi 0.35° as in Sputter. Beam centre: Analyze's own (the Sputter model estimated (11.5, 21.5) px).",
        "The frames listed in Analyze before were replaced."]
    # Again for the same window: nothing else was listed, so nothing was replaced.
    toasts.clear()
    composed._show_sputter_window(window)
    assert "replaced" not in toasts[0][0]


def test_without_a_matching_profile_the_notice_stays_and_says_what_to_enter(tmp_path, toasts):
    scan, names = _scan(tmp_path)
    window = BinnedWindowReader().read(windows_folder(tmp_path / "result", scan, names, beam=(100.25, 200.75)),
                                       "bin_000000")
    page = FakeAnalyze()
    assert Composed(page, [_profile("zz far", 1003.0, 1.0)])._show_sputter_window(window)
    options = page.calls[0][1]
    assert options["profile_name"] is None and options["beam_center"] == (100.25, 200.75)
    (text, shown), = toasts
    assert shown["level"] == "warning" and shown["timeout_ms"] == 0  # stays until closed: values to type
    assert text.split("\n") == [
        "Window bin_000000 (0.00–0.05 s): the sum of its 2 frames is shown in Analyze.",
        "No instrument profile has the Sputter geometry: enter distance 1000 mm and λ 1.0000 Å "
        "(Geometry step ▸ Geometry…).",
        "αi 0.35° as in Sputter. Beam centre (100.25, 200.75) px as in Sputter."]
    try:
        from src.gimap.app.sputter_handover import notice_text

        apply_language("zh")
        chinese = notice_text(window, None, replaced=True)
        assert chinese.startswith("时间窗 bin_000000（0.00–0.05 s）：分析页显示的是它 2 帧之和。")
        assert "请输入距离 1000 mm 和 λ 1.0000 Å" in chinese and "原先列出的帧已被替换" in chinese
    finally:
        apply_language(DEFAULT_LANGUAGE)


def test_a_busy_analyze_page_changes_nothing_and_says_why(tmp_path, toasts):
    scan, names = _scan(tmp_path)
    window = BinnedWindowReader().read(windows_folder(tmp_path / "result", scan, names), "bin_000000")
    busy = FakeAnalyze(files=[tmp_path / "zz_kept.cbf"], locked="an invented lock")
    composed = Composed(busy)
    assert composed._show_sputter_window(window) is False
    assert busy.calls == [] and composed.requested == [] and busy.view_model.state.files == [tmp_path / "zz_kept.cbf"]
    assert toasts == [("Not shown in Analyze: an invented lock", {"level": "warning"})]
    toasts.clear()
    refused = FakeAnalyze(accepts=False)
    assert Composed(refused)._show_sputter_window(window) is False
    assert toasts == [("Not shown in Analyze: an invented reason of the status line", {"level": "warning"})]


# -- the Sputter page's menus ------------------------------------------------------------------------------------------


def _sputter_page(tmp_path):
    from src.gimap.app.presentation.task_runner import TaskRunner
    from src.gimap.features.sputter.bootstrap import create_sputter_service
    from src.gimap.features.sputter.presentation.page import SputterPage
    from tests.sputter_fixtures import FakeInspector, FakeInstaller, FakeRunner

    service = create_sputter_service(runner=FakeRunner(), installer=FakeInstaller(), inspector=FakeInspector(),
                                     results_root=lambda: tmp_path / "results")
    page = SputterPage(service, task_runner=TaskRunner())
    page.resize(1400, 900)
    return page


def _close(page):
    page.dispose()
    page.tasks.wait(10)
    page.close()
    page.deleteLater()


def test_the_windows_table_the_sketch_and_the_trend_offer_show_in_analyze(tmp_path, monkeypatch):
    from src.gimap.features.sputter.presentation import window_menu

    monkeypatch.setenv("GIMAP_SPUTTER_NO_INSTALL", "1")
    menus = []
    monkeypatch.setattr(window_menu.WindowMenuMixin, "_popup", staticmethod(lambda menu, where: menus.append(menu)))
    scan, names = _scan(tmp_path)
    folder = windows_folder(tmp_path / "result", scan, names)
    page = _sputter_page(tmp_path)
    try:
        page.open_results(folder, "physical")
        # Without a target (a page on its own) the entry is there, disabled.
        page.table.customContextMenuRequested.emit(page.table.visualRect(page.table.model().index(1, 0)).center())
        texts = [action.text() for action in menus[-1].actions()]
        assert texts == ["t = 1.00 s", "Show in Analyze", "", "Copy Rows", "Copy Table"]
        assert not menus[-1].actions()[1].isEnabled()
        shown = []
        page.set_analyze_target(lambda window: shown.append(window) or True)
        page.table.customContextMenuRequested.emit(page.table.visualRect(page.table.model().index(1, 0)).center())
        menus[-1].actions()[1].trigger()
        assert [(window.window, window.count) for window in shown] == [("bin_000001", 3)]
        # The sketch: the window it shows.
        page.sketch.slider.setValue(2)
        page.sketch._menu_requested(page.sketch.slider, QPoint(4, 4))
        assert menus[-1].actions()[0].text() == "t = 1.50 s"
        menus[-1].actions()[1].trigger()
        assert shown[-1].window == "bin_000002" and shown[-1].frames == tuple(scan / name for name in names[5:8])
        # The trend: the window nearest in time to the click (1.9 s → the window at 2.0 s).
        canvas, figure = page.trend.canvas, page.trend.figure
        canvas.draw()
        x, y = figure.axes[0].transData.transform((1.9, np.mean(figure.axes[0].get_ylim())))
        ratio = canvas.device_pixel_ratio
        canvas.customContextMenuRequested.emit(QPoint(round(x / ratio), round((figure.bbox.height - y) / ratio)))
        assert menus[-1].actions()[0].text() == "t = 2.00 s"
        menus[-1].actions()[1].trigger()
        assert shown[-1].window == "bin_000003" and shown[-1].count == 2
        assert page.nearest_row(0.0) == 0 and page.nearest_row(1.26) == 2
        # A window whose frames are gone says why on the Sputter page; Analyze is not asked.
        notices = []
        monkeypatch.setattr(page, "_notify", lambda message, error=False: notices.append((message, error)))
        (scan / names[0]).unlink()
        assert page.show_window_in_analyze("bin_000000") is False and len(shown) == 3
        assert notices and notices[0][1] is True and notices[0][0].startswith("1 of the 2 frames of window bin_000000")
    finally:
        _close(page)


# -- the real Analyze page: the image is the sum of the window's frames -------------------------------------------------


def test_analyze_shows_the_sum_of_every_frame_of_the_window(tmp_path, toasts):
    from tests.test_analyze_workspace import _context, _page
    from tests.test_assistant_calibration import save_tiff
    from src.gimap.integrations.state import InMemoryInstrumentProfileRepository

    scan = tmp_path / "zz_invented_tiff_scan"
    names = []
    for index in range(1, 6):  # frame k holds k counts in every pixel
        names.append(save_tiff(scan / f"zz_invented_{index:05d}.tif", np.full((24, 36), float(index))).name)
    other = save_tiff(tmp_path / "zz_listed_before.tif", np.full((24, 36), 100.0))
    folder = windows_folder(tmp_path / "result", scan, names, per_window=(2, 3), beam=(18.0, 30.0))
    window = BinnedWindowReader().read(folder, "bin_000001")
    profiles = InMemoryInstrumentProfileRepository([_profile("zz SAXS", 1000.0, 1.0)])
    context = _context(profiles)
    page = _page(context)
    composed = Composed(page, context=context)
    try:
        page.import_paths([other], interactive=False)
        _settle(lambda: page.view_model.state.analysis is not None, 30)
        page._automatic_busy = True  # the automatic analysis keeps its frame: nothing changes
        assert composed._show_sputter_window(window) is False
        assert page.view_model.state.files == [other.resolve()] and "automatic analysis" in toasts[-1][0]
        page._automatic_busy = False
        assert composed._show_sputter_window(window) is True

        def summed():
            analysis = page.view_model.state.analysis
            return analysis is not None and analysis.frame_total == 3 and not page.tasks.is_busy()

        _settle(summed, 60)
        analysis = page.view_model.state.analysis
        assert [path.name for path in page.view_model.state.files] == names[2:]
        assert analysis.frame_total == 3 and analysis.path.name == names[2]
        assert [Path(path).name for path, _index in analysis.summed_frames] == names[3:]
        assert np.all(np.asarray(analysis.data) == 3 + 4 + 5)  # the sum of every frame of the window
        assert page.sum_spin.value() == 3 and page.view_model.sum_count == 3
        assert page.view_model.state.incidence_deg == pytest.approx(0.35)
        assert page.incidence_spin.value() == pytest.approx(0.35)
        assert page.view_model.state.profile_name == "zz SAXS" and page.profile_combo.currentData() == "zz SAXS"
        assert page.view_model.state.beam_center == (18.0, 30.0)
        assert analysis.geometry.beam_center_x_px == 18.0 and analysis.geometry.incidence_deg == pytest.approx(0.35)
        assert toasts[-1][0].endswith("The frames listed in Analyze before were replaced.")
    finally:
        page.cancel_frame_import()
        page.tasks.wait(30)
        page.dispose()
        page.close()


def test_every_new_text_is_in_the_chinese_table():
    from src.gimap.app.presentation.i18n.zh import ZH
    from src.gimap.app.sputter_handover import TEMPLATES
    from src.gimap.features.analyze.presentation.bindings.frame_import import SUMMED_LOCKED
    from src.gimap.features.sputter.presentation.window_menu import SHOW_IN_ANALYZE

    missing = [text for text in (*TEMPLATES, *SUMMED_LOCKED.values(), SHOW_IN_ANALYZE) if text not in ZH]
    assert missing == []
