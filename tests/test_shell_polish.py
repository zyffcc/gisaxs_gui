"""The window shell: Ctrl+O, Start cards and drops, the title, quitting during a job, the Start page,
the Labs status bar, Recent and the sidebar in Chinese."""

from __future__ import annotations

import os
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt5.QtCore import QMimeData, QPoint, QPointF, Qt, QUrl
from PyQt5.QtGui import QCloseEvent, QDragEnterEvent, QDragLeaveEvent, QDropEvent, QKeySequence
from PyQt5.QtWidgets import QApplication, QMessageBox, QScrollArea, QShortcut, QWidget

import src.gimap.app.main_window as main_window_module

ROOT = Path(__file__).resolve().parents[1]
GALAXI = ROOT / "tests" / "data" / "external" / "gisaxs_galaxi" / "galaxi_data.tif"
needs_galaxi = pytest.mark.skipif(not GALAXI.exists(), reason="GALAXI example not present")


def _settle(seconds: float = 0.2) -> None:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        QApplication.processEvents()
        time.sleep(0.01)


def _window():
    from main import MainWindow
    from src.gimap.app import AppContext
    from src.gimap.integrations.jobs import LocalProcessJobRunner
    from src.gimap.integrations.state import (
        InMemoryInstrumentProfileRepository,
        InMemorySessionRepository,
        InMemorySettingsRepository,
        InMemoryUserPreferencesRepository,
    )
    from src.gimap.shared.geometry import DetectorGeometry, InstrumentProfile

    QApplication.instance() or QApplication([])
    geometry = DetectorGeometry(172e-6, 172e-6, 1.73, 597.1, 719.6, 1.34, 0.463)
    profiles = InMemoryInstrumentProfileRepository([InstrumentProfile("GALAXI", geometry, None, (1043, 981))])
    context = AppContext(settings=InMemorySettingsRepository(), session=InMemorySessionRepository(),
                         preferences=InMemoryUserPreferencesRepository(), jobs=LocalProcessJobRunner(),
                         instrument_profiles=profiles)
    window = MainWindow(context)
    window.resize(1400, 900)
    window.show()
    end = time.monotonic() + 40
    while not window._initialization_completed and time.monotonic() < end:
        _settle(0.02)
    return window


def _navigate(window, key: str) -> None:
    runtime = getattr(window, "runtime", None)
    if runtime is not None:
        runtime.navigate(key)
    else:
        window.components.show_page(key)


@pytest.fixture
def window():
    shown = _window()
    yield shown
    shown.close()
    _settle(0.05)


def _drop(widget: QWidget, paths, pos: QPoint | None = None) -> bool:
    """A drag that enters ``widget`` and drops there; Qt passes it up to the nearest parent taking drops."""
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(path)) for path in paths])
    pos = pos or widget.rect().center()
    enter = QDragEnterEvent(pos, Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier)
    QApplication.sendEvent(widget, enter)
    if not enter.isAccepted():
        return False
    drop = QDropEvent(QPointF(pos), Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier)
    QApplication.sendEvent(widget, drop)
    return drop.isAccepted()


# -- Ctrl+O ------------------------------------------------------------------------------------


def _record_open(window, monkeypatch) -> list[str]:
    calls: list[str] = []
    components = window.components
    monkeypatch.setattr(components.fitting_workspace.fit_page, "open_curve_dialog", lambda: calls.append("curve"))
    monkeypatch.setattr(components.analyze_page, "open_files", lambda: calls.append("data"))
    return calls


def test_open_data_opens_a_curve_on_fitting_and_frames_everywhere_else(window, monkeypatch) -> None:
    calls = _record_open(window, monkeypatch)
    components = window.components
    action = window.menus.menu_bar.actions["open_files"]
    assert action.shortcut() == QKeySequence(QKeySequence.Open)
    _navigate(window, "fitting")
    components.fitting_workspace.show_context("single")
    _settle()
    action.trigger()
    assert calls == ["curve"] and components.current_page_key() == "fitting"
    components.fitting_workspace.show_context("insitu")  # the single page is hidden: Open Data
    _settle()
    action.trigger()
    assert calls == ["curve", "data"] and components.current_page_key() == "analyze"
    for key in ("home", "analyze"):
        calls.clear()
        _navigate(window, key)
        action.trigger()
        assert calls == ["data"]


def test_ctrl_o_on_the_fitting_page_is_not_ambiguous(window, monkeypatch) -> None:
    components = window.components
    fit_page = components.fitting_workspace.fit_page
    page_shortcuts = [shortcut for shortcut in fit_page.findChildren(QShortcut)
                      if shortcut.key() == QKeySequence("Ctrl+O")]
    if page_shortcuts:
        pytest.skip("Fitting's own Ctrl+O shortcut is still there (the Fitting package removes it)")
    calls = _record_open(window, monkeypatch)
    ambiguous = []
    for shortcut in window.findChildren(QShortcut):
        shortcut.activatedAmbiguously.connect(lambda: ambiguous.append(True))
    _navigate(window, "fitting")
    components.fitting_workspace.show_context("single")
    window.activateWindow()
    QApplication.setActiveWindow(window)
    fit_page.setFocus()
    _settle()
    from PyQt5.QtTest import QTest

    QTest.keyClick(QApplication.focusWidget() or fit_page, Qt.Key_O, Qt.ControlModifier)
    _settle()
    assert calls == ["curve"] and ambiguous == []
    _navigate(window, "home")
    _settle()
    QTest.keyClick(QApplication.focusWidget() or window, Qt.Key_O, Qt.ControlModifier)
    _settle()
    assert calls == ["curve", "data"]


# -- Start cards, Ask AI and the automatic analysis ----------------------------------------------


@needs_galaxi
def test_start_cards_choose_the_mode_and_open_data_when_none_is_listed(window, monkeypatch) -> None:
    components = window.components
    page = components.analyze_page
    chosen, toasts, dialogs = [], [], []
    monkeypatch.setattr(page, "choose_mode", chosen.append, raising=False)
    monkeypatch.setattr(main_window_module, "show_toast",
                        lambda _parent, text, **kwargs: toasts.append((text, kwargs.get("level"))))
    monkeypatch.setattr(page, "open_files", lambda: dialogs.append("files"))
    monkeypatch.setattr(page, "open_folder", lambda: dialogs.append("folder"))

    components._task_requested("giwaxs")
    assert chosen == ["giwaxs"] and dialogs == ["files"]
    assert toasts == [("Analyze mode: GIWAXS — choose Auto in the bar above to let GIMaP decide", "info")]
    components._task_requested("series")
    assert dialogs == ["files", "folder"] and len(toasts) == 1

    page.add_paths([str(GALAXI)])
    page.tasks.wait(120)
    for record in (chosen, toasts, dialogs):
        record.clear()
    components._task_requested("gisaxs")  # data are listed: the mode changes, no dialog
    assert chosen == ["gisaxs"] and dialogs == [] and len(toasts) == 1 and toasts[0][1] == "info"
    assert "GISAXS" in toasts[0][0]
    components._task_requested("series")
    assert dialogs == [] and components.current_page_key() == "analyze"


@needs_galaxi
def test_the_results_step_follows_the_frame_and_its_notes_go_to_ask_ai(window, monkeypatch) -> None:
    from types import SimpleNamespace

    components = window.components
    page = components.analyze_page
    shown = []
    monkeypatch.setattr(components.guided, "frame_shown",
                        lambda path, frame=None, summed=None: shown.append((path, frame, summed)), raising=False)
    page.add_paths([str(GALAXI)])
    page.tasks.wait(120)
    _settle()
    assert shown and shown[-1] == (str(GALAXI), 1, 1)  # the file, its first frame shown and frames summed
    shown.clear()
    page.analysisShown.emit(page.view_model.state.analysis)
    assert shown == [(str(GALAXI), 1, 1)]

    started = []
    components._assistant = SimpleNamespace(start=lambda notes="": started.append(notes), running=lambda: False,
                                            shutdown=lambda: None)
    components.guided.notes_edit.setPlainText("P03, GIWAXS, alpha_i = 0.4 deg")
    page._start_assistant()
    assert started == ["P03, GIWAXS, alpha_i = 0.4 deg"]
    components._assistant = None


# -- quitting during a job -----------------------------------------------------------------------


def test_quitting_while_a_job_runs_asks_first(window, monkeypatch) -> None:
    components = window.components
    assert components.running_jobs() == []
    monkeypatch.setattr(components.analyze_page, "batch_running", lambda: True)
    assert components.running_jobs() == ["Batch Export"]
    questions = []

    def answer(reply):
        def question(*args, **_kwargs):
            questions.append(args)
            return reply
        return question

    shutdowns = []
    shutdown = components.shutdown
    monkeypatch.setattr(components, "shutdown", lambda: (shutdowns.append(True), shutdown()))
    monkeypatch.setattr(QMessageBox, "question", answer(QMessageBox.Cancel))
    event = QCloseEvent()
    window.closeEvent(event)
    assert not event.isAccepted() and shutdowns == [] and not getattr(window, "_closed", False)
    assert questions[0][1] == "Quit GIMaP" and questions[0][2] == "Still running: Batch Export. Stop them and quit?"
    assert questions[0][4] == QMessageBox.Cancel  # the default button

    monkeypatch.setattr(QMessageBox, "question", answer(QMessageBox.Yes))
    window.close()
    assert shutdowns == [True] and window._closed and len(questions) == 2


def test_stop_them_and_quit_stops_the_fit_the_series_and_the_automatic_analysis(window, monkeypatch) -> None:
    """“Stop them and quit?” → Yes: a Single fit, an In-situ series and the automatic analysis are stopped
    too (not only Batch Export and the AI run), so no thread outlives the window."""
    import threading

    components = window.components
    fit_page = components.fitting_workspace.fit_page
    series_page = components.fitting_workspace.series_page
    finished = {}

    def stop_aware(name, is_set):
        def job():
            end = time.monotonic() + 15
            while not is_set() and time.monotonic() < end:
                time.sleep(0.01)
            finished[name] = is_set()
        return job

    fit_page._running = "local"  # what run_fit sets while a fit lasts
    fit_page.tasks.submit("fit", stop_aware("fit", fit_page._stop_event.is_set))
    series_page.running = True
    series_page.tasks.submit("series", stop_aware("series", series_page._stop.is_set))
    guided = components.guided  # its pipeline thread, as _launch starts it, ending when asked to stop
    guided._stop_event = threading.Event()
    guided._thread = threading.Thread(target=lambda: guided._stop_event.wait(15), name="gimap-guided", daemon=True)
    guided._thread.start()
    assert components.running_jobs() == ["Automatic analysis", "Fitting", "In-situ series"]
    monkeypatch.setattr(QMessageBox, "question", lambda *_args, **_kwargs: QMessageBox.Yes)
    started = time.monotonic()
    window.close()
    assert window._closed and guided._stop_event.is_set() and not guided._thread.is_alive()
    assert fit_page._stop_event.is_set() and series_page._stop.is_set()
    assert finished == {"fit": True, "series": True}  # both threads ended at once, not after 15 s
    assert time.monotonic() - started < 10


def test_quitting_without_a_job_does_not_ask(window, monkeypatch) -> None:
    asked = []
    monkeypatch.setattr(QMessageBox, "question", lambda *args: asked.append(args) or QMessageBox.Cancel)
    window.close()
    assert window._closed and asked == []


# -- drops on the whole window -------------------------------------------------------------------


def test_a_project_dropped_anywhere_opens_as_a_project(window, monkeypatch, tmp_path) -> None:
    project = tmp_path / "sample.gimap"
    project.write_text('{"format": "gimap-project", "version": 1}', encoding="utf-8")
    opened = []
    monkeypatch.setattr(window.menus, "open_project", lambda path=None: opened.append(str(path)) or True)
    components = window.components
    components.show_page("compare")
    _settle()
    assert _drop(components.compare_page, [project])
    assert _drop(window, [project])
    components.show_page("analyze")
    _settle()
    assert _drop(components.analyze_page, [project])  # Analyze leaves projects to the window
    assert opened == [str(project)] * 3


@needs_galaxi
def test_a_frame_dropped_on_the_start_page_outside_the_box_opens_in_analyze(window) -> None:
    components = window.components
    components.show_page("home")
    _settle()
    card = components.home_page.cards["giwaxs"]
    assert _drop(card, [GALAXI])
    components.analyze_page.tasks.wait(120)
    assert components.current_page_key() == "analyze"
    assert [str(path) for path in components.analyze_page.view_model.state.files] == [str(GALAXI)]


def test_a_drop_with_nothing_to_open_says_so(window, monkeypatch, tmp_path) -> None:
    note = tmp_path / "beamtime notes.txt"
    note.write_text("alpha_i 0.4 deg", encoding="utf-8")
    toasts = []
    monkeypatch.setattr(main_window_module, "show_toast",
                        lambda _parent, text, **kwargs: toasts.append((text, kwargs.get("level"))))
    components = window.components
    components.show_page("home")
    assert components.open_dropped([note]) is False
    assert components.current_page_key() == "home"
    assert [level for _text, level in toasts] == ["warning"] and "(.gimap)" in toasts[0][0]


def test_a_curve_dropped_on_fitting_opens_there(window, monkeypatch, tmp_path) -> None:
    curve = tmp_path / "cut_fit_input.dat"
    curve.write_text("0.1 10 1\n0.2 8 1\n0.3 6 1\n", encoding="utf-8")
    components = window.components
    fit_page = components.fitting_workspace.fit_page
    opened = []
    monkeypatch.setattr(fit_page, "open_curve", lambda path, side=None: opened.append((path, side)))
    _navigate(window, "fitting")
    _settle()
    assert _drop(fit_page, [curve])
    # As Open Curve: no side, so the halves chosen in Fitting stay.
    assert opened == [(str(curve), None)] and components.current_page_key() == "fitting"


# -- the window title and Save Project -----------------------------------------------------------


def test_the_title_names_the_project_and_the_frame(window) -> None:
    menus = window.menus
    assert window.windowTitle() == "GIMaP"
    menus.project_path = str(Path("C:/data/A.gimap"))
    menus._sync_title()
    assert window.windowTitle() == "GIMaP — A"
    menus._files_cleared()
    assert menus.project_path == "" and window.windowTitle() == "GIMaP"


def test_the_title_names_a_frame_that_could_not_be_read(window, tmp_path) -> None:
    """A frame chosen in Analyze that fails to load is the current one: the title names it, not the last."""
    page = window.components.analyze_page
    first, broken = tmp_path / "frame_001.tif", tmp_path / "frame_002.tif"
    page.view_model.state.files = [first, broken]
    page.view_model.state.current_index = 0
    window.menus._sync_title()  # as after the first frame was shown
    assert window.windowTitle() == "GIMaP — frame_001.tif"
    page.view_model.state.current_index = 1
    page.analysisFailed.emit("not a detector frame")
    assert window.windowTitle() == "GIMaP — frame_002.tif"


def test_open_recent_names_the_file_and_its_folder(tmp_path) -> None:
    from PyQt5.QtWidgets import QMainWindow

    from src.gimap.app.presentation.menu_bar import MainMenuBar, MenuCommands

    QApplication.instance() or QApplication([])
    frame = tmp_path / "frame_001.tif"
    frame.write_bytes(b"x")
    window = QMainWindow()
    commands = MenuCommands(recent_paths=lambda: [frame, tmp_path], open_recent=lambda _path: None,
                            clear_recent=lambda: None)
    menus = MainMenuBar(window, commands)
    menus._fill_recent()
    texts = [action.text() for action in menus.recent_menu.actions()]
    assert texts[0] == f"&1  frame_001.tif — {tmp_path.name}"
    assert texts[1].endswith("(folder)") and texts[-1] == "Clear the List"


# -- the Start page --------------------------------------------------------------------------------


def _overlapping(root: QWidget) -> list[tuple[str, str]]:
    """Visible sibling widgets under ``root`` whose rectangles share area."""
    found = []
    widgets = [w for w in root.findChildren(QWidget) if w.isVisible() and not w.isWindow()]
    by_parent: dict[int, list[QWidget]] = {}
    for widget in widgets:
        by_parent.setdefault(id(widget.parentWidget()), []).append(widget)
    for siblings in by_parent.values():
        for index, first in enumerate(siblings):
            for second in siblings[index + 1:]:
                if first.geometry().intersects(second.geometry()):
                    found.append((first.objectName() or type(first).__name__,
                                  second.objectName() or type(second).__name__))
    return found


@pytest.mark.parametrize("mode", ["light", "dark"])
def test_the_start_page_scrolls_instead_of_squeezing(mode, tmp_path) -> None:
    from src.gimap.app.presentation.theme import apply_theme

    apply_theme(mode, 9)
    shown = _window()
    try:
        components = shown.components
        frames = []
        for index in range(5):
            frame = tmp_path / f"folder_{index}" / f"frame_{index:03d}.tif"
            frame.parent.mkdir()
            frame.write_bytes(b"x")
            frames.append(str(frame))
        components.analyze_page.remember_recent(frames)
        components.show_page("home")
        home = components.home_page
        home.refresh_recent()
        _settle(0.3)
        assert home.recent_rows.count() == 5
        # The Start page no longer sets the window's minimum height (it was 852 px with 5 recent items).
        # The stack's minimum is the tallest page's; at the time of writing that is Trainset Build (764 px).
        assert home.minimumSizeHint().height() <= 640 - shown.menuBar().sizeHint().height()
        heights = {key: page.minimumSizeHint().height() for key, page in components.pages.items()}
        assert shown.minimumSizeHint().height() <= 640 or max(heights, key=heights.get) != "home", heights
        scroll = home.findChild(QScrollArea, "homeScrollArea")
        assert scroll is not None and scroll.horizontalScrollBarPolicy() == Qt.ScrollBarAlwaysOff
        for width, height in ((1024, 640), (1366, 705)):
            shown.resize(width, height)
            _settle(0.3)
            assert shown.height() == height
            assert _overlapping(scroll.widget()) == [], (width, height)
            content, viewport = scroll.widget(), scroll.viewport()
            assert content.height() >= content.minimumSizeHint().height()
            assert scroll.verticalScrollBar().isVisible() == (content.height() > viewport.height())
        shown.resize(1024, 640)
        _settle(0.3)
        assert scroll.verticalScrollBar().isVisible()  # a small window: the page scrolls
    finally:
        shown.close()
        apply_theme("light", 9)


def test_the_start_page_uses_theme_tokens_and_shows_a_drag() -> None:
    from src.gimap.app.presentation.home_page import HomePage

    assert "palette(" not in (ROOT / "src/gimap/app/presentation/home_page.py").read_text(encoding="utf-8")
    page = HomePage()
    for card in page.cards.values():
        assert card.property("card") is True and card.property("homeTask") is True
        assert card.testAttribute(Qt.WA_Hover)
    zone = page.drop_zone
    assert zone.objectName() == "homeDropZone" and not zone.styleSheet()
    dropped = []
    page.filesDropped.connect(dropped.append)
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(GALAXI))])
    enter = QDragEnterEvent(QPoint(5, 5), Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier)
    zone.dragEnterEvent(enter)
    assert enter.isAccepted() and zone.property("dragActive") is True
    zone.dragLeaveEvent(QDragLeaveEvent())
    assert zone.property("dragActive") is False
    zone.dragEnterEvent(QDragEnterEvent(QPoint(5, 5), Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier))
    assert zone.property("dragActive") is True
    zone.dropEvent(QDropEvent(QPointF(5, 5), Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier))
    assert zone.property("dragActive") is False and dropped == [[str(GALAXI)]]
    page.close()


def test_start_recent_is_the_open_recent_list_with_tags(window, monkeypatch, tmp_path) -> None:
    components = window.components
    project = tmp_path / "run.gimap"
    project.write_text('{"format": "gimap-project", "version": 1}', encoding="utf-8")
    folder = tmp_path / "frames"
    folder.mkdir()
    frame = folder / "a&b.tif"
    frame.write_bytes(b"x")
    window.app_context.preferences.set("recent_projects", [str(project)])
    components.analyze_page.remember_recent([str(frame), str(folder)])
    home = components.home_page
    home.refresh_recent()
    rows = [home.recent_rows.itemAt(index).widget() for index in range(home.recent_rows.count())]
    expected = [str(path) for path in window.menus.recent_paths()[:5]]
    assert [row.toolTip() for row in rows] == expected == [str(project), str(frame), str(folder)]
    assert [row.text() for row in rows] == [
        f"run.gimap — {tmp_path.name} (project)", "a&&b.tif — frames", f"frames — {tmp_path.name} (folder)",
    ]
    opened = []
    monkeypatch.setattr(window.menus, "open_recent", opened.append)
    rows[0].click()
    rows[1].click()
    assert opened == [str(project), str(frame)]


# -- the Labs status bar ---------------------------------------------------------------------------


def test_the_status_bar_shows_on_the_labs_pages_only(window) -> None:
    components = window.components
    components.show_page("trainset")
    assert window.statusbar.isVisible()
    runtime = getattr(window, "runtime", None)
    assert runtime is not None, "the feature runtimes did not start"
    runtime.trainset.status_updated.emit("Draw the rectangular ROI on the detector image")
    assert window.statusbar.currentMessage() == "Draw the rectangular ROI on the detector image"
    components.show_page("predict")
    assert window.statusbar.isVisible()
    for key in ("home", "analyze", "fitting", "compare"):
        components.show_page(key)
        assert not window.statusbar.isVisible(), key


# -- the sidebar ------------------------------------------------------------------------------------


def test_the_sidebar_keeps_its_language_and_shows_one_logo_when_collapsed() -> None:
    from src.gimap.app.presentation.i18n import apply_language, apply_to
    from src.gimap.app.presentation.navigation import NavigationSidebar

    sidebar = NavigationSidebar()
    try:
        apply_language("zh")
        apply_to(sidebar, "zh")
        assert sidebar.toggle_button.text() == "«  收起"
        for _ in range(2):
            sidebar.set_collapsed(True)
            assert sidebar.logo_label.isHidden() and sidebar.brand_label.isHidden()
            sidebar.set_collapsed(False)
        assert sidebar.toggle_button.text() == "«  收起"
        assert not sidebar.logo_label.isHidden()
    finally:
        apply_language("en", [sidebar])
    assert sidebar.toggle_button.text() == "«  Collapse"
    sidebar.close()
