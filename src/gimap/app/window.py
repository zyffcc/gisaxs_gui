"""The main window of GIMaP and the start of a session with it (``run_window``, called by ``launcher.py``).

``MainWindow`` composes the shell (``src.gimap.app.window_view``), the feature pages
(``src.gimap.app.main_window``) and the menus (``src.gimap.app.menus``). ``main.py`` re-exports it for the
scripts and tests that import it from there.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

# Matplotlib must use the Qt backend of PyQt5.
os.environ.setdefault("MPLBACKEND", "Qt5Agg")

# Before PyQt5: find Qt's platform plugin even when GIMaP is started without Start_GIMaP.bat.
from src.gimap.app.qt_plugins import ensure_qt_platform_plugin_path

ensure_qt_platform_plugin_path()

from PyQt5.QtCore import QByteArray, Qt, QTimer  # noqa: E402
from PyQt5.QtGui import QGuiApplication
from PyQt5.QtWidgets import QApplication, QMainWindow

from src.gimap.app import AppContext
from src.gimap.app.main_window import MainWindowComponents
from src.gimap.app.menus import ApplicationMenus
from src.gimap.app.presentation.app_dialogs import ask_quit_with_jobs
from src.gimap.app.presentation.assets import app_icon
from src.gimap.app.presentation.components import show_toast
from src.gimap.app.presentation.i18n import tr, trf
from src.gimap.app.presentation.layout_metrics import LAYOUT, fit_to_screen, move_window_to_cursor_screen
from src.gimap.app.presentation.theme.appearance import Appearance
from src.gimap.app.runtime import ApplicationRuntime
from src.gimap.app.window_view import ApplicationWindowView
from src.gimap.features.fitting import application as fitting_files

logger = logging.getLogger(__name__)

WINDOW_GEOMETRY_KEY = "window.geometry"
CURVE_SUFFIXES = tuple(fitting_files.CURVE_SUFFIXES)
"""Curve files, as Fitting's reader takes them (.dat, .txt, .csv by the names of its header, NXcanSAS .h5 —
``is_curve_file``: an .h5 of detector frames goes to Analyze): dropped while Fitting is shown, they open there."""
TABLE_SUFFIXES = tuple(suffix for suffix in (".csv",) if suffix not in CURVE_SUFFIXES)
"""Tables Fitting would not read as curves: dropped while Fitting is shown, they are refused with the reason instead
of being sent anywhere. Empty since Fitting reads .csv by the names of its header."""
NOT_A_CURVE = fitting_files.UNSUPPORTED_CURVE
"""Why: Fitting's own sentence (what Open Curve… says for such a file)."""


def _dropped_paths(event) -> list[str]:
    mime = event.mimeData()
    if mime is None or not mime.hasUrls():
        return []
    return [str(Path(url.toLocalFile())) for url in mime.urls() if url.isLocalFile() and url.toLocalFile()]


class MainWindow(QMainWindow, ApplicationWindowView):
    def __init__(self, app_context: AppContext):
        super().__init__()
        self.app_context = app_context
        self._startup_time = time.monotonic()
        self._initialization_completed = False
        self.setupUi(self)
        self.setWindowTitle("GIMaP")
        self.setWindowIcon(app_icon())
        self.setMinimumSize(LAYOUT.min_window)
        # Every page has its own status line; the window's status bar is shown on the Labs pages only
        # (MainWindowComponents.show_page).
        self.components = MainWindowComponents(self)
        self.menus = ApplicationMenus(self)
        # Files dropped anywhere on the window (pages that take drops themselves handle theirs first).
        self.setAcceptDrops(True)
        # Feature runtimes start after the window is on screen.
        QTimer.singleShot(100, self._delayed_initialization)

    # -- drops ----------------------------------------------------------------------------

    def dragEnterEvent(self, event) -> None:  # noqa: N802 - Qt API
        if _dropped_paths(event):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event) -> None:  # noqa: N802 - Qt API
        if _dropped_paths(event):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event) -> None:  # noqa: N802 - Qt API
        paths = _dropped_paths(event)
        if not paths:
            event.ignore()
            return
        event.acceptProposedAction()
        self.open_dropped(paths)

    def open_dropped(self, paths) -> None:
        """A project opens as a project. On Fitting, a curve file its reader takes opens there and a .csv
        table is refused with the reason. Anything else goes to Analyze."""
        paths = [str(Path(path)) for path in paths]
        projects = [path for path in paths if path.lower().endswith(".gimap")]
        if projects:
            self.menus.open_project(projects[0])
            return
        components = self.components
        if components.current_page_key() == "fitting":
            curves = [path for path in paths if path.lower().endswith(CURVE_SUFFIXES) and fitting_files.is_curve_file(path)]
            if curves:
                # As Open Curve: no side given, so the halves chosen in Fitting stay (a side would replace them).
                workspace = components.fitting_workspace
                workspace.show_context("single")
                workspace.fit_page.open_curve_file(curves[0])  # NXcanSAS of several curves: asks which
                return
            tables = [path for path in paths if path.lower().endswith(TABLE_SUFFIXES)]
            if tables:
                show_toast(self, trf("Could not open {name}: {reason}", name=Path(tables[0]).name,
                                     reason=tr(NOT_A_CURVE)), level="warning")
                paths = [path for path in paths if path not in tables]
                if not paths:  # only tables: nothing for Analyze, whose "drop detector frames" would mislead
                    return
        components.open_dropped(paths)

    def _delayed_initialization(self) -> None:
        try:
            self.runtime = ApplicationRuntime(
                self,
                self,
            )
        except Exception as exc:  # keep the window usable; the error is shown, not hidden
            logger.exception("Deferred initialization failed: %s", exc)
            show_toast(self, trf("Some workspaces could not start: {error}", error=exc), level="error", timeout_ms=0)
        finally:
            self._initialization_completed = True
            Appearance(self.app_context.preferences).apply_language([self])

    def open_paths(self, paths) -> None:
        """Files and folders from the command line or a second start: a project opens as a project, the rest
        as dropped files (``open_dropped``); a path that is not there is said, not opened."""
        present = [str(Path(path)) for path in paths if Path(path).exists()]
        for missing in [path for path in paths if not Path(path).exists()][:1]:
            show_toast(self, trf("Could not open {name}: it is not there.", name=Path(missing).name), level="warning")
        if present:
            self.open_dropped(present)

    def bring_to_front(self) -> None:
        """A second start asked for this window: shown, not minimised, in front."""
        if self.isMinimized():
            self.showNormal()
        self.show()
        self.raise_()
        self.activateWindow()

    def restore_window_geometry(self) -> None:
        saved = self.app_context.preferences.get(WINDOW_GEOMETRY_KEY)
        if isinstance(saved, str) and saved:
            if self.restoreGeometry(QByteArray.fromBase64(saved.encode("ascii"))):
                return
        fit_to_screen(self, LAYOUT.default_window)
        move_window_to_cursor_screen(self)

    def _save_window_geometry(self) -> None:
        geometry = bytes(self.saveGeometry().toBase64()).decode("ascii")
        self.app_context.preferences.set(WINDOW_GEOMETRY_KEY, geometry)

    def closeEvent(self, event):
        """Save the session, layout and preferences, then stop background work (once).

        While a job runs (Batch Export, the automatic analysis, an AI run, a fit, an In-situ series) the
        user is asked first: an accidental Alt+F4 must not end an 800-frame export without a word.
        """
        if getattr(self, "_closed", False):
            event.accept()
            return
        document = getattr(self, "document", None)  # unsaved changes (a session started by the launcher)
        if document is not None and not document.may_close():
            event.ignore()
            return
        if not self._confirm_quit():
            event.ignore()
            return
        shutdown_sem = getattr(getattr(self, "menus", None), "shutdown_sem", None)
        if shutdown_sem is not None and shutdown_sem() is False:
            show_toast(self, tr("SEM work is finishing. Close the window again after it stops."), level="warning")
            event.ignore()
            return
        self._closed = True
        try:
            if hasattr(self, "runtime"):
                self.runtime.handle_window_close()
            self._save_window_geometry()
            layout_memory = getattr(self, "layout_memory", None)  # the splitters (a session of the launcher)
            if layout_memory is not None:
                layout_memory.save()
            tools = getattr(getattr(self, "menus", None), "tools", None)
            if tools is not None:  # tool windows still open: where they are now
                tools.remember_all()
            self.components.save_state()
            self.app_context.save_session()
        except Exception as exc:
            logger.exception("Failed to save the session on close: %s", exc)
        finally:
            # Stopping may wait up to 2 s for a job: the window (and its tool windows) take no clicks meanwhile.
            # It runs even when saving failed, so no worker outlives the window.
            self.setEnabled(False)
            try:
                self.components.shutdown()
            except Exception as exc:
                logger.exception("Failed to stop the running work on close: %s", exc)
            if self.app_context.jobs is not None:
                self.app_context.jobs.shutdown()
            if document is not None:  # closed properly: nothing to recover at the next start
                document.closed_cleanly()
            event.accept()

    def _confirm_quit(self) -> bool:
        try:
            jobs = self.components.running_jobs()
        except Exception:  # never keep the window open because a check failed
            return True
        if not jobs:
            return True
        return ask_quit_with_jobs(self, self.components.jobs_text(jobs))


def configure_high_dpi() -> None:
    """Let Qt scale by the exact screen factor (125 %, 150 % …) — call before QApplication."""
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    QGuiApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )


def _warm_up_matplotlib() -> None:
    """Build Matplotlib's font cache once the window is visible (first plots stay fast)."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
        from matplotlib.figure import Figure

        FigureCanvasQTAgg(Figure(figsize=(1, 1)))
        family = plt.rcParams.get("font.family", [])
        family = [family] if isinstance(family, str) else list(family)
        if "DejaVu Sans" not in family:
            plt.rcParams["font.family"] = ["DejaVu Sans", *family]
        plt.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass


def run_window(app, options, *, splash=None, instance=None, data_dir=None, read_only: bool = False) -> int:
    """Build the main window of a session started by ``launcher.launch`` and run it; returns the exit code.

    After the window is on screen: the work of a session that was not closed properly is offered back
    (``document_session.pending_recovery``), then the files of the command line are opened. A second start
    (``instance.activated``) brings the window to the front and opens its files.
    """
    from src.gimap.app import startup_progress
    from src.gimap.app.bootstrap import create_app_context
    from src.gimap.app.document_session import DocumentSession, pending_recovery
    from src.gimap.app.error_guard import ErrorGuard
    from src.gimap.app.layout_memory import LayoutMemory
    from src.gimap.app.presentation.app_dialogs import ask_restore_session
    from src.gimap.shared.version import __version__

    # Python's collector may run in the middle of a paint and delete Qt objects of windows closed before:
    # an access violation after a long session. Collect on a timer in the GUI thread instead (pyqtgraph's way).
    from pyqtgraph.util.garbage_collector import GarbageCollector

    app._gimap_garbage_collector = GarbageCollector(interval=1.0)
    startup_progress.report("Reading the settings…")
    app_context = create_app_context(read_only=read_only)
    # A bug in one handler must not close the application: log it, show it, go on.
    guard = ErrorGuard(Path(app_context.data_dir or ".") / "logs", version=__version__, parent=app).install()
    app.aboutToQuit.connect(guard.uninstall)
    Appearance(app_context.preferences).apply_saved()
    recovery = None if read_only else pending_recovery(app_context.data_dir)

    window = MainWindow(app_context)
    startup_progress.report("Opening the window…")
    window.layout_memory = LayoutMemory(window)
    window.document = DocumentSession(window, app_context.data_dir, autosave=not read_only)
    window.document.holding = recovery is not None
    window.menus.attach_document(window.document)
    if instance is not None:
        instance.activated.connect(lambda paths: (window.bring_to_front(), window.open_paths(paths)))
    window.restore_window_geometry()
    window.show()
    if splash is not None:
        splash.finish(window)
    QTimer.singleShot(200, _warm_up_matplotlib)

    def after_start() -> None:
        if not getattr(window, "_initialization_completed", False):
            QTimer.singleShot(100, after_start)
            return
        document = window.document
        if recovery is not None:
            when = recovery["saved"].replace("T", " ") or tr("an unknown time")
            if ask_restore_session(window, when):
                document.restore(recovery)
            else:
                document.discard_autosave()
            document.holding = False
        if options.paths:
            window.open_paths([str(Path(path).resolve()) for path in options.paths])

    QTimer.singleShot(0, after_start)
    logger.info("Window shown")
    return app.exec_()
