"""Find Qt's platform plugin before the QApplication starts.

Started without Start_GIMaP.bat (pythonw.exe or main.py double-clicked, an IDE, an Anaconda Python
that was not activated), or on a computer where another program set QT_PLUGIN_PATH to its own Qt,
Qt stops with "no Qt platform plugin could be initialized". This module looks for the plugin folder
of the PyQt5 that GIMaP runs with and points Qt at it. It imports no PyQt5 module, so main.py can
call it first.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Iterable, MutableMapping

PLATFORM_ENV = "QT_QPA_PLATFORM_PLUGIN_PATH"
PLUGIN_ENV = "QT_PLUGIN_PATH"


def _platform_plugin_names(platform: str) -> tuple[str, ...]:
    if platform.startswith("win"):
        return ("qwindows.dll",)
    if platform == "darwin":
        return ("libqcocoa.dylib",)
    return ("libqxcb.so", "libqwayland-generic.so", "libqoffscreen.so")


def has_platform_plugin(platforms_dir: Path | str | None, platform: str = sys.platform) -> bool:
    """True when the folder holds the platform plugin Qt needs on this system."""
    if not platforms_dir:
        return False
    folder = Path(platforms_dir)
    return any((folder / name).is_file() for name in _platform_plugin_names(platform))


def candidate_plugin_dirs(prefix: Path | str | None = None) -> list[Path]:
    """Qt plugin folders (each with a platforms/ inside) of this Python, most specific first.

    pip's PyQt5 keeps them in PyQt5/Qt5/plugins (older wheels: PyQt5/Qt/plugins); conda's Qt in
    <env>/Library/plugins on Windows and <env>/plugins elsewhere.
    """
    dirs: list[Path] = []
    try:
        spec = importlib.util.find_spec("PyQt5")
    except (ImportError, ValueError):
        spec = None
    if spec is not None and spec.submodule_search_locations:
        for location in spec.submodule_search_locations:
            package = Path(location)
            dirs += [package / "Qt5" / "plugins", package / "Qt" / "plugins"]
    prefix = Path(sys.prefix if prefix is None else prefix)
    dirs += [prefix / "Library" / "plugins", prefix / "plugins"]
    return dirs


def ensure_qt_platform_plugin_path(
    environ: MutableMapping[str, str] | None = None,
    candidates: Iterable[Path] | None = None,
    platform: str = sys.platform,
) -> Path | None:
    """Point Qt at the platforms/ folder of the PyQt5 GIMaP runs with.

    A QT_QPA_PLATFORM_PLUGIN_PATH the user set to a folder that has the plugin is kept. Anything else
    (nothing set, a folder without the plugin, a QT_PLUGIN_PATH of another program's Qt) is replaced
    by this Python's own plugins. Returns the folder set, or None when nothing was changed (kept, or no
    folder with the plugin was found: Qt then reports its own error as before).
    """
    environ = os.environ if environ is None else environ
    if has_platform_plugin(environ.get(PLATFORM_ENV), platform):
        return None
    for plugins in (candidate_plugin_dirs() if candidates is None else candidates):
        platforms = Path(plugins) / "platforms"
        if not has_platform_plugin(platforms, platform):
            continue
        environ[PLATFORM_ENV] = str(platforms)
        # Another program's QT_PLUGIN_PATH would still load its image and style plugins, built for a
        # different Qt: use this Qt's plugins only.
        if PLUGIN_ENV in environ:
            environ[PLUGIN_ENV] = str(Path(plugins))
        if platform.startswith("win") and environ is os.environ:
            _add_conda_dll_dir()
        return platforms
    return None


def _add_conda_dll_dir() -> None:
    """conda's qwindows.dll needs the Qt DLLs in <env>/Library/bin, on PATH only when activated."""
    library_bin = Path(sys.prefix) / "Library" / "bin"
    if not library_bin.is_dir():
        return
    path = os.environ.get("PATH", "")
    if str(library_bin).lower() not in (part.lower() for part in path.split(os.pathsep)):
        os.environ["PATH"] = str(library_bin) + os.pathsep + path
    add_dll_directory = getattr(os, "add_dll_directory", None)
    if add_dll_directory is not None:
        try:
            add_dll_directory(str(library_bin))
        except OSError:
            pass
