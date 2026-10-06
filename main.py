"""GIMaP entry point and main-window composition root."""

import os

# Matplotlib must use the Qt backend of PyQt5.
os.environ.setdefault("MPLBACKEND", "Qt5Agg")

import sys
from pathlib import Path
import time

from PyQt5.QtCore import QByteArray, Qt, QTimer
from PyQt5.QtGui import QFont, QGuiApplication
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox

from src.gimap.app import AppContext
from src.gimap.app.bootstrap import create_app_context
from src.gimap.app.main_window import MainWindowComponents
from src.gimap.app.menus import ApplicationMenus
from src.gimap.app.presentation.assets import app_icon
from src.gimap.app.presentation.i18n import tr, trf
from src.gimap.app.presentation.layout_metrics import LAYOUT, fit_to_screen, move_window_to_cursor_screen
from src.gimap.app.presentation.menu_bar import APP_VERSION
from src.gimap.app.presentation.theme.appearance import Appearance
from src.gimap.app.runtime import ApplicationRuntime
from src.gimap.app.window_view import ApplicationWindowView
from src.gimap.integrations.bornagain import BornAgainSimulator

WINDOW_GEOMETRY_KEY = "window.geometry"
CURVE_SUFFIXES = (".dat", ".txt", ".csv")
"""Curve files: dropped while Fitting is shown, they open there."""


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
        """A project opens as a project; a curve file on Fitting opens there; anything else in Analyze."""
        paths = [str(Path(path)) for path in paths]
        projects = [path for path in paths if path.lower().endswith(".gimap")]
        if projects:
            self.menus.open_project(projects[0])
            return
        components = self.components
        if components.current_page_key() == "fitting":
            curves = [path for path in paths if path.lower().endswith(CURVE_SUFFIXES)]
            if curves:
                # As Open Curve: no side given, so the halves chosen in Fitting stay (a side would replace them).
                workspace = components.fitting_workspace
                workspace.show_context("single")
                workspace.fit_page.open_curve(curves[0])
                return
        components.open_dropped(paths)

    def _delayed_initialization(self) -> None:
        try:
            self.runtime = ApplicationRuntime(
                self,
                self,
                simulation_port=BornAgainSimulator(runner=self.app_context.jobs),
            )
        except Exception as exc:  # keep the window usable; the error is shown, not hidden
            print(f"Deferred initialization failed: {exc}")
            from src.gimap.app.presentation.components import show_toast

            show_toast(self, trf("Some workspaces could not start: {error}", error=exc), level="error", timeout_ms=0)
        finally:
            self._initialization_completed = True
            Appearance(self.app_context.preferences).apply_language([self])

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
        if not self._confirm_quit():
            event.ignore()
            return
        self._closed = True
        try:
            if hasattr(self, "runtime"):
                self.runtime.handle_window_close()
            self._save_window_geometry()
            tools = getattr(getattr(self, "menus", None), "tools", None)
            if tools is not None:  # tool windows still open: where they are now
                tools.remember_all()
            self.components.save_state()
            self.app_context.save_session()
        except Exception as exc:
            print(f"Failed to save the session on close: {exc}")
        finally:
            # Stopping may wait up to 2 s for a job: the window (and its tool windows) take no clicks meanwhile.
            # It runs even when saving failed, so no worker outlives the window.
            self.setEnabled(False)
            try:
                self.components.shutdown()
            except Exception as exc:
                print(f"Failed to stop the running work on close: {exc}")
            if self.app_context.jobs is not None:
                self.app_context.jobs.shutdown()
            event.accept()

    def _confirm_quit(self) -> bool:
        try:
            jobs = self.components.running_jobs()
        except Exception:  # never keep the window open because a check failed
            return True
        if not jobs:
            return True
        answer = QMessageBox.question(
            self,
            tr("Quit GIMaP"),
            trf("Still running: {jobs}. Stop them and quit?", jobs=self.components.jobs_text(jobs)),
            QMessageBox.Yes | QMessageBox.Cancel,
            QMessageBox.Cancel,
        )
        return answer == QMessageBox.Yes


def configure_high_dpi() -> None:
    """Let Qt scale by the exact screen factor (125 %, 150 % …) — call before QApplication."""
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    QGuiApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )


def main():
    configure_high_dpi()
    app = QApplication(sys.argv)
    app.setApplicationName("GIMaP")
    app.setApplicationVersion(APP_VERSION)
    app.setOrganizationName("GIMaP")
    app.setWindowIcon(app_icon())
    if sys.platform.startswith("win"):
        app.setFont(QFont("Segoe UI"))
    app_context = create_app_context()
    # A bug in one handler must not close the application: log it, show it, go on.
    from src.gimap.app.error_guard import ErrorGuard

    guard = ErrorGuard(Path(app_context.data_dir or ".") / "logs", version=APP_VERSION, parent=app).install()
    app.aboutToQuit.connect(guard.uninstall)
    Appearance(app_context.preferences).apply_saved()

    window = MainWindow(app_context)
    window.restore_window_geometry()
    window.show()
    QTimer.singleShot(200, _warm_up_matplotlib)
    sys.exit(app.exec_())


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


if __name__ == "__main__":
    main()
