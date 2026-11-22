"""The start of GIMaP: command line, splash screen, one GIMaP per user folder, the log; then the window.

``launch()`` is what ``python main.py`` and the ``gimap`` command run. It reads the command line (``--version``
answers without starting Qt), creates the application, shows the splash within a second (``splash.py``) and
takes the user folder (``single_instance.py``). Only then does it import the analysis modules and build the
window (``window.run_window``), which takes seconds: the splash says what is loading meanwhile.

Light imports only at the top of this module: PyQt, the version and the user folder's location.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional, Sequence

from src.gimap.shared.version import APP_NAME, __version__

EXIT_OK = 0


def parse_arguments(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """``gimap [FILE_OR_FOLDER ...] [--version]``; options Qt knows (``-style …``) are left to Qt."""
    parser = argparse.ArgumentParser(
        prog="gimap", description=f"{APP_NAME}: GISAXS and GIWAXS analysis. Opens the files and folders given "
                                  "(detector frames, a folder of them, a .gimap project). "
                                  "Use 'gimap reproduce RECORD.json [--out FOLDER]' to repeat a reduction or fit.")
    parser.add_argument("paths", nargs="*", metavar="FILE_OR_FOLDER",
                        help="detector frames, a folder of frames or a project (.gimap) to open")
    parser.add_argument("--version", action="version", version=f"{APP_NAME} {__version__}")
    options, qt_arguments = parser.parse_known_args(list(sys.argv[1:] if argv is None else argv))
    options.qt_arguments = [argument for argument in qt_arguments if argument.startswith("-")]
    options.paths = [*options.paths, *[argument for argument in qt_arguments if not argument.startswith("-")]]
    return options


def saved_appearance(data_dir: Path) -> tuple[str, str]:
    """``(language, theme)`` saved in the user folder, read without the store (before the heavy imports)."""
    try:
        payload = json.loads((Path(data_dir) / "settings.json").read_text(encoding="utf-8"))
        preferences = payload.get("preferences") if isinstance(payload, dict) else {}
    except (OSError, ValueError):
        preferences = {}
    preferences = preferences if isinstance(preferences, dict) else {}
    return str(preferences.get("appearance.language") or "en"), str(preferences.get("appearance.theme") or "light")


def launch(argv: Optional[Sequence[str]] = None) -> int:
    """Run GIMaP; returns the exit code."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments[:1] == ["reproduce"]:
        from src.gimap.app.reproduce import reproduce_cli
        return reproduce_cli(arguments[1:])
    options = parse_arguments(arguments)  # --version and --help end here, before Qt starts
    # Matplotlib must use the Qt backend of PyQt5; Qt must find its platform plugin.
    os.environ.setdefault("MPLBACKEND", "Qt5Agg")
    from src.gimap.app.qt_plugins import ensure_qt_platform_plugin_path

    ensure_qt_platform_plugin_path()
    from PyQt5.QtCore import Qt
    from PyQt5.QtGui import QFont, QGuiApplication
    from PyQt5.QtWidgets import QApplication

    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    QGuiApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication([sys.argv[0] if sys.argv else "gimap", *options.qt_arguments])
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(__version__)
    app.setOrganizationName(APP_NAME)
    if sys.platform.startswith("win"):
        app.setFont(QFont("Segoe UI"))

    from src.gimap.app.presentation.i18n import normalized_language, translator
    from src.gimap.app.presentation.splash import StartupSplash
    from src.gimap.integrations.state.user_store import user_data_dir

    data_dir = user_data_dir()
    language, theme = saved_appearance(data_dir)
    # Texts composed from now on (the question below, the pages) are in the saved language; the walker that
    # translates windows as they are shown starts with the window (``Appearance.apply_language``).
    translator().language = normalized_language(language)
    splash = StartupSplash(language, theme)
    splash.show()
    app.processEvents()  # the splash on screen now
    splash.step("Starting GIMaP…")

    from src.gimap.app.single_instance import SingleInstance

    instance = SingleInstance(data_dir)
    read_only = False
    if not instance.acquire():
        from src.gimap.app.presentation.app_dialogs import ask_already_running

        splash.hide()
        choice = ask_already_running(None)
        if choice == "switch":
            _allow_foreground()
            if instance.hand_over([str(Path(path).resolve()) for path in options.paths]):
                return EXIT_OK
            choice = ask_already_running(None)  # the other one did not answer (still starting, or hung)
        if choice != "read_only":
            return EXIT_OK
        read_only = True
        splash.show()

    from src.gimap.app import logging_setup, startup_progress

    logging_setup.configure_logging(data_dir, version=__version__, second=read_only)
    startup_progress.listen(splash.step)
    splash.step("Loading the analysis modules…")
    try:
        from src.gimap.app.window import run_window

        return run_window(app, options, splash=splash, instance=instance, data_dir=data_dir, read_only=read_only)
    finally:
        startup_progress.listen(None)
        instance.release()
        logging_setup.stop_logging()


def _allow_foreground() -> None:
    """Windows lets the running GIMaP come to the front only when the process asking allows it."""
    if sys.platform.startswith("win"):
        try:
            import ctypes

            ctypes.windll.user32.AllowSetForegroundWindow(-1)  # ASFW_ANY
        except (AttributeError, OSError):
            pass


__all__ = ["launch", "parse_arguments", "saved_appearance"]
