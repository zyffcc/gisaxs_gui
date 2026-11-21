"""GIMaP entry point: ``python main.py [FILE_OR_FOLDER ...] [--version]`` (installed: ``gimap``).

The start — command line, splash screen, one GIMaP per user folder, the log — is in
``src/gimap/app/launcher.py``; the main window and the session it runs in ``src/gimap/app/window.py``
(composed from ``src.gimap.app.window_view``, ``src.gimap.app.main_window`` and ``src.gimap.app.menus``).
This file stays light, so the splash appears before the analysis modules are imported; the window's names
are still importable from here (``from main import MainWindow``).
"""

from __future__ import annotations

import sys

def main(argv=None) -> int:
    from src.gimap.app.launcher import launch

    return launch(argv)


def __getattr__(name: str):
    """``MainWindow``, ``MainWindowComponents``, ``configure_high_dpi`` …: the window module's names, imported on
    first use (what this file defined before the start-up was split)."""
    if not name.startswith("__"):
        from src.gimap.app import window

        if hasattr(window, name):
            return getattr(window, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if __name__ == "__main__":
    sys.exit(main())
