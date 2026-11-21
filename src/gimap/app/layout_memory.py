"""The positions of the splitters (Analyze's panels, Fitting's, Compare's …), kept from one session to the next.

Every ``QSplitter`` of the main window with an object name is restored from the preferences when the window is
first shown (and when a page built on first use appears: ``lazy_pages.py``), and saved when the window closes,
as ``window.geometry`` already is. A state saved for another layout (a splitter that has more or fewer panels
now) is not applied. Attached by the launcher (``window.run_window``).
"""

from __future__ import annotations

from PyQt5.QtCore import QByteArray, QTimer
from PyQt5.QtWidgets import QSplitter

SPLITTERS_KEY = "layout.splitters"


class LayoutMemory:
    def __init__(self, window):
        self.window = window
        self.preferences = window.app_context.preferences
        stored = self.preferences.get(SPLITTERS_KEY, {})
        self.saved: dict = dict(stored) if isinstance(stored, dict) else {}
        components = window.components
        on_built = getattr(components, "on_page_built", None)
        if on_built is not None:
            on_built(lambda _key, page: QTimer.singleShot(0, lambda: self.restore(page)))
        QTimer.singleShot(0, lambda: self.restore(window))

    def _splitters(self, root) -> dict[str, QSplitter]:
        found: dict[str, QSplitter] = {}
        for splitter in root.findChildren(QSplitter):
            name = splitter.objectName()
            if name and name not in found:
                found[name] = splitter
        return found

    def restore(self, root) -> int:
        """Restore the splitters under ``root`` that have a saved state; returns how many."""
        done = 0
        try:
            splitters = self._splitters(root)
        except RuntimeError:  # the page or window is gone
            return 0
        for name, splitter in splitters.items():
            entry = self.saved.get(name)
            if not isinstance(entry, dict) or entry.get("count") != splitter.count():
                continue
            state = entry.get("state")
            if isinstance(state, str) and splitter.restoreState(QByteArray.fromBase64(state.encode("ascii"))):
                done += 1
        return done

    def save(self) -> None:
        """Keep the state of every named splitter of the window (in the preferences, saved with them)."""
        try:
            splitters = self._splitters(self.window)
        except RuntimeError:
            return
        for name, splitter in splitters.items():
            self.saved[name] = {"count": splitter.count(),
                                "state": bytes(splitter.saveState().toBase64()).decode("ascii")}
        self.preferences.set(SPLITTERS_KEY, self.saved)


__all__ = ["LayoutMemory", "SPLITTERS_KEY"]
