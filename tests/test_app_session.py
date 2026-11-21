"""The application around the pages: start-up, unsaved changes and autosave, theme switches, the log, the
command line and installation, the settings import, and the smaller shell rules (toasts over a status line,
Recent, splitters, small windows, F1)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt5.QtCore import QUrl
from PyQt5.QtWidgets import QApplication, QLabel, QSplitter, QWidget

from tests.test_project import _window

ROOT = Path(__file__).resolve().parents[1]


def _app() -> QApplication:
    return QApplication.instance() or QApplication([])


def _settle(seconds: float = 0.1) -> None:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        QApplication.processEvents()
        time.sleep(0.01)


@pytest.fixture(scope="module")
def window():
    shown = _window()
    yield shown
    shown._closed = True  # no session, no questions: the page threads are stopped as on close
    shown.close()
    shown.components.shutdown()
    _settle(0.05)


# -- start-up --------------------------------------------------------------------------------------------


def test_compare_and_sputter_are_built_when_first_shown(window) -> None:
    from src.gimap.app import project

    components = window.components
    stack = window.mainWindowWidget
    assert stack.count() == 6  # placeholders keep the order and the size of the stack
    if components.built_page("compare") is None:  # (not shown by an earlier test of the module)
        data = project.collect(components, "analyze")
        assert "compare" not in data and "sputter" not in data  # never built: nothing to save
        assert components.running_jobs() == []  # and nothing to stop
    window.runtime.navigate("compare")
    page = components.built_page("compare")
    assert page is not None and stack.currentWidget() is page and components.current_page_key() == "compare"
    assert stack.count() == 6 and components.compare_page is page  # built once
    assert "compare" in project.collect(components, "compare")
    window.runtime.navigate("home")


def test_the_launcher_answers_the_command_line_without_starting_qt(capsys) -> None:
    from src.gimap.app.launcher import parse_arguments
    from src.gimap.shared.version import __version__

    with pytest.raises(SystemExit) as stopped:
        parse_arguments(["--version"])
    assert stopped.value.code == 0 and capsys.readouterr().out.strip() == f"GIMaP {__version__}"
    options = parse_arguments(["frame.tif", "folder", "sample.gimap"])
    assert options.paths == ["frame.tif", "folder", "sample.gimap"] and options.qt_arguments == []
    result = subprocess.run([sys.executable, "main.py", "--version"], cwd=ROOT, capture_output=True, text=True,
                            timeout=60)
    assert result.returncode == 0 and result.stdout.strip() == f"GIMaP {__version__}"
    assert "PyQt5" not in subprocess.run(
        [sys.executable, "-c", "import sys, main; print(sorted(m for m in sys.modules if m.startswith('PyQt5')))"],
        cwd=ROOT, capture_output=True, text=True, timeout=60).stdout  # main.py stays light for the splash


def test_the_project_table_names_the_version_the_requirements_and_the_command() -> None:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")  # (Python 3.10 has no tomllib)
    assert '[project]\nname = "gimap"' in text and 'dynamic = ["version", "dependencies"]' in text
    assert 'version = {attr = "src.gimap.shared.version.__version__"}' in text  # the one place it is written
    assert 'dependencies = {file = ["requirements.txt"]}' in text
    assert 'gimap = "src.gimap.app.launcher:launch"' in text and 'py-modules = ["main"]' in text


def test_the_splash_shows_the_version_and_the_steps_in_the_saved_language() -> None:
    from PyQt5.QtWidgets import QSplashScreen

    from src.gimap.app import startup_progress
    from src.gimap.app.presentation.i18n.zh import ZH
    from src.gimap.app.presentation.splash import StartupSplash, version_text
    from src.gimap.shared.version import __version__

    _app()
    assert __version__ in version_text("en") and __version__ in version_text("zh")
    splash = StartupSplash("zh", "dark")
    try:
        startup_progress.listen(splash.step)
        startup_progress.report("Building Analyze…")
        assert QSplashScreen.message(splash).strip() == ZH["Building Analyze…"]
    finally:
        startup_progress.listen(None)
        splash.close()
        splash.deleteLater()
    startup_progress.report("nobody listens: nothing happens")


def test_one_gimap_per_user_folder_and_a_second_start_hands_its_files_over(tmp_path) -> None:
    from src.gimap.app.single_instance import SingleInstance

    _app()
    first, second = SingleInstance(tmp_path), SingleInstance(tmp_path)
    try:
        assert first.acquire() and (tmp_path / "gimap.lock").exists()
        assert not second.acquire()
        received = []
        first.activated.connect(received.append)
        assert second.hand_over([str(tmp_path / "frame.tif")])
        end = time.monotonic() + 5
        while not received and time.monotonic() < end:
            _settle(0.02)
        assert received == [[str(tmp_path / "frame.tif")]]
    finally:
        first.release()
        second.release()
    assert SingleInstance(tmp_path).acquire()  # released: the next start owns the folder


# -- unsaved changes, autosave, recovery --------------------------------------------------------------------


def test_changed_work_is_marked_asked_about_autosaved_and_offered_back(window, tmp_path, monkeypatch) -> None:
    from src.gimap.app import document_session
    from src.gimap.app.document_session import DocumentSession, pending_recovery

    series = window.components.fitting_workspace.series_page
    pattern = series.pattern_edit.text()
    document = DocumentSession(window, tmp_path)
    window.menus.attach_document(document)
    try:
        document.mark_clean(settle=0)
        assert not document.dirty and window.windowTitle().endswith("[*]") and not window.isWindowModified()
        assert (tmp_path / "autosave" / "running.json").is_file()  # a session is open
        series.pattern_edit.setText("changed_*.dat")
        document.check()
        assert document.dirty and window.isWindowModified()
        asked = []
        monkeypatch.setattr(document_session.app_dialogs, "ask_save_changes",
                            lambda _parent, name: asked.append(name) or answer)
        answer = "cancel"
        assert document.may_close() is False and asked == [""]  # no project yet: “save the current work …?”
        answer = "discard"
        assert document.may_close() is True
        written = document.autosave()
        assert written is not None and written.is_file() and document.autosave() is None  # nothing new to write
        recovery = pending_recovery(tmp_path)  # what the next start finds after a crash
        assert recovery is not None and Path(recovery["path"]) == written and recovery["saved"]
        series.pattern_edit.setText(pattern)
        document.check()
        assert not document.dirty  # back to what was saved: no “*”
        document.closed_cleanly()
        assert pending_recovery(tmp_path) is None and not (tmp_path / "autosave" / "session.gimap").exists()
    finally:
        document.closed_cleanly()
        window.menus.document = None
        window.menus._sync_title()


# -- theme ----------------------------------------------------------------------------------------------------


def test_a_theme_switch_restyles_the_windows_and_leaves_the_application_sheet_alone() -> None:
    from src.gimap.app.presentation.theme import apply_theme, theme_manager
    from src.gimap.app.presentation.theme.engine import TEMPLATE_PROPERTY

    app = _app()
    manager = theme_manager()
    apply_theme("light", 9.0)
    start, sheet = manager.start_stamp, app.styleSheet()
    top = QWidget()
    child = QLabel("text", top)
    hidden_feature = QWidget(top)
    template = ROOT / "src/gimap/features/fitting/presentation/fitting_theme.qss"
    manager.style_widget(hidden_feature, template)
    hidden_feature.hide()
    top.show()
    try:
        before = hidden_feature.styleSheet()
        apply_theme("dark" if start.startswith("light") else "light", 9.0)
        assert app.styleSheet() == sheet  # the application sheet keeps the theme of the start
        assert top.styleSheet() == manager.window_sheet() != ""  # the window carries the theme's sheet
        assert child.palette().color(child.foregroundRole()).name() == manager.color("text").name()
        assert hidden_feature.styleSheet() == before  # hidden: rendered when shown
        hidden_feature.show()
        _settle(0.02)
        assert hidden_feature.styleSheet() != before and hidden_feature.property(TEMPLATE_PROPERTY)
        apply_theme(start.split(":")[0], float(start.split(":")[1]))
        assert top.styleSheet() == ""  # back in the theme of the start: the application's sheet again
    finally:
        apply_theme("light", 9.0)
        top.close()
        top.deleteLater()


def test_the_font_size_field_applies_once_after_the_last_click(monkeypatch) -> None:
    from src.gimap.app.presentation import settings_dialog
    from src.gimap.app.presentation.settings_dialog import SettingsDialog
    from src.gimap.integrations.state import InMemoryUserPreferencesRepository

    _app()
    dialog = SettingsDialog(preferences=InMemoryUserPreferencesRepository())
    applied = []
    monkeypatch.setattr(dialog.appearance, "set_font_pt", applied.append)
    try:
        start = dialog.font_size_spin.value()
        for step in (1, 2, 3):
            dialog.font_size_spin.setValue(start + 0.5 * step)
        assert applied == []  # arrow clicks: nothing yet
        end = time.monotonic() + settings_dialog.FONT_DELAY_MS / 1000 + 1
        while not applied and time.monotonic() < end:
            _settle(0.02)
        assert applied == [start + 1.5]
        dialog.font_size_spin.setValue(start)
        dialog.font_size_spin.editingFinished.emit()  # Enter, or the field left: at once
        assert applied == [start + 1.5, start]
    finally:
        dialog.close()
        dialog.deleteLater()


# -- the log ---------------------------------------------------------------------------------------------------


def test_the_log_keeps_prints_and_the_version_and_crash_log_marks_each_start(tmp_path) -> None:
    script = (
        "import sys; from src.gimap.app import logging_setup as l; "
        f"l.configure_logging(r'{tmp_path}', version='9.9.9'); print('a message of print'); "
        "import logging; logging.getLogger('gimap.test').warning('a warning'); l.stop_logging()")
    result = subprocess.run([sys.executable, "-c", script], cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    log = (tmp_path / "logs" / "gimap.log").read_text(encoding="utf-8")
    assert "GIMaP 9.9.9 started" in log and "a message of print" in log and "a warning" in log
    assert "GIMaP 9.9.9 started" in (tmp_path / "logs" / "crash.log").read_text(encoding="utf-8")
    assert "a message of print" in result.stdout  # the console still shows it


# -- the settings import -------------------------------------------------------------------------------------


def test_old_settings_come_once_into_the_default_folder_only_and_never_a_session(tmp_path) -> None:
    from src.gimap.integrations.state.user_store import MIGRATION_MARKER, UserStore, migrate_once

    program = tmp_path / "program"
    (program / "config").mkdir(parents=True)
    (program / ".gimap_cache").mkdir()
    (program / "config" / "user_parameters.json").write_text(json.dumps({
        "analyze": {"mode": "gisaxs", "last_folder": "D:/made/up", "recent": ["D:/made/up/a.cbf"]},
        "fitting": {"last_session": {"last_opened_file": "D:/made/up/b.cbf"}, "detector": {"distance": 1234.5}},
    }), encoding="utf-8")
    (program / ".gimap_cache" / "session.json").write_text("{}", encoding="utf-8")
    chosen = tmp_path / "chosen"
    assert migrate_once(chosen, explicit=True, project_root=program) == []  # GIMAP_HOME: nothing at all
    assert not chosen.exists()
    default = tmp_path / "default"
    assert migrate_once(default, explicit=False, project_root=program)
    settings = UserStore(default / "settings.json").settings
    assert settings["analyze"]["mode"] == "gisaxs" and settings["fitting"]["detector"]["distance"] == 1234.5
    assert "last_folder" not in settings["analyze"] and "recent" not in settings["analyze"]
    assert "last_session" not in settings["fitting"] and not (default / "session.json").exists()
    assert (default / MIGRATION_MARKER).is_file()
    (default / "settings.json").unlink()
    assert migrate_once(default, explicit=False, project_root=program) == []  # once
    assert not (default / "settings.json").exists()


# -- toasts, Recent, splitters, small windows, F1 ---------------------------------------------------------


def test_a_notice_from_before_keeps_off_a_newer_status_line() -> None:
    from src.gimap.app.presentation.components import show_toast, visible_toasts

    _app()
    page = QWidget()
    page.resize(900, 300)
    status = QLabel("Ready", page)
    status.setGeometry(16, 275, 860, 22)
    page.show()
    try:
        notice = show_toast(page, "Nothing opened: drop detector frames, a folder of them or a project", level="warning")
        _settle(0.05)
        old_width = notice.width()
        status.setProperty("level", "error")  # (as a page's status line says an error)
        status.setText("Could not analyse the frame: frame_0001.tif is not a detector image GIMaP can read "
                       "(CBF, TIFF, EDF or NeXus).")
        _settle(0.6)  # the row looks again
        text_end = status.x() + status.fontMetrics().horizontalAdvance(status.text())
        assert notice not in visible_toasts(page) or (notice.x() >= text_end and notice.width() < old_width)
        fresh = show_toast(page, "A newer notice keeps its corner", level="warning")
        _settle(0.05)
        assert fresh in visible_toasts(page) and fresh.label.text() == "A newer notice keeps its corner"
    finally:
        page.close()
        page.deleteLater()


def test_what_could_not_be_opened_leaves_recent(window, tmp_path, monkeypatch) -> None:
    from src.gimap.app import menus as menus_module

    menus = window.menus
    broken = tmp_path / "broken.gimap"
    broken.write_text("not a project", encoding="utf-8")
    window.app_context.preferences.set(menus_module.RECENT_PROJECTS_KEY, [str(broken)])
    warnings = []
    monkeypatch.setattr(menus_module, "warn", lambda _parent, _title, text: warnings.append(text))
    assert menus.open_project(str(broken)) is False
    assert window.app_context.preferences.get(menus_module.RECENT_PROJECTS_KEY) == []
    assert warnings and warnings[0].endswith("It was removed from Recent.")
    page = window.components.analyze_page
    frame = tmp_path / "frame_0001.tif"
    frame.write_text("not an image", encoding="utf-8")
    page.remember_recent([frame])
    assert page.forget_recent(frame) and not page.forget_recent(frame)


def test_a_project_opened_is_not_offered_the_last_sessions_set_up(window) -> None:
    page = window.components.analyze_page
    page._setup_offered = False
    page.decline_last_setup()
    assert page._setup_offered  # the project's own set-up wins


def test_splitters_come_back_where_they_were_left(window) -> None:
    from src.gimap.app.layout_memory import SPLITTERS_KEY, LayoutMemory

    window.runtime.navigate("analyze")
    _settle(0.1)
    memory = LayoutMemory(window)
    splitter = window.findChild(QSplitter, "analyzeSplitter")
    total = sum(splitter.sizes())
    splitter.setSizes([total // 4] + [(total - total // 4) // max(1, splitter.count() - 1)] * (splitter.count() - 1))
    memory.save()
    saved = window.app_context.preferences.get(SPLITTERS_KEY)
    assert saved["analyzeSplitter"]["count"] == splitter.count()
    first = splitter.sizes()[0]
    splitter.setSizes([total // 2] + [(total - total // 2) // max(1, splitter.count() - 1)] * (splitter.count() - 1))
    assert LayoutMemory(window).restore(window) >= 1
    assert abs(splitter.sizes()[0] - first) <= 2
    window.runtime.navigate("home")


def test_a_small_window_scrolls_the_page_instead_of_cutting_it(window) -> None:
    window.runtime.navigate("analyze")
    window.resize(1024, 640)
    _settle(0.2)
    page = window.components.analyze_page
    needed = page.minimumSizeHint()
    assert page.width() >= min(needed.width(), page.maximumWidth()) and page.height() >= needed.height()
    scroll = window.components.page_scroll
    assert scroll.horizontalScrollBar().maximum() > 0 or scroll.verticalScrollBar().maximum() > 0
    window.runtime.navigate("home")  # Start needs less: Analyze's size no longer counts
    _settle(0.2)
    home = window.components.home_page
    assert scroll.widget().minimumSizeHint() == home.minimumSizeHint().expandedTo(home.minimumSize())
    window.resize(1500, 950)
    _settle(0.1)


def test_f1_opens_the_manual_at_the_chapter_of_the_workspace_shown(window, monkeypatch) -> None:
    from src.gimap.app.presentation import help_documents

    opened = []
    monkeypatch.setattr(help_documents.QDesktopServices, "openUrl", lambda url: opened.append(QUrl(url)))
    window.runtime.navigate("analyze")
    window.menus.menu_bar.actions["user_manual"].trigger()
    forward = Path(opened[-1].toLocalFile())
    assert forward.name.endswith("_section.html") and "#5-analyze" in forward.read_text(encoding="utf-8")
    page = help_documents.write_page(help_documents.USER_MANUAL, "User Manual").read_text(encoding="utf-8")
    assert help_documents.section_anchor(page, "3.2") == "32-the-start-page"
    assert help_documents.section_anchor(page, "5") == "5-analyze"
    window.runtime.navigate("home")
