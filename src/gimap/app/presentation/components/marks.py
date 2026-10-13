"""The marks drawn over an image or a plot, each kind switchable, with a “Marks” menu.

A view draws marks of several kinds — the beam centre, the horizon, cut bands, masks and cut regions,
highlighted pixels, a q box, a window band … — and decides for each item whether it is *wanted*
(``track(key, item, wanted)``). ``MarkLayers`` shows a wanted item only while its kind is switched on,
so switching a kind off never loses what the page asked for, and switching it on shows it again.

``menu_button()`` gives a “Marks ▾” button: one check per kind present in the view, Show All and Hide
All, and — for marks the person made — “Remove …” entries the owner adds (``add_removal``), for example
“Remove the Drawn Masks”. ``changed(key, visible)`` lets the owner remember the choice.
"""

from __future__ import annotations

from typing import Callable, Optional

from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtWidgets import QMenu, QToolButton, QWidget


DETECTOR_MARKS = (
    ("beam", "Beam centre"),
    ("horizon", "Sample horizon"),
    ("bands", "Cut bands"),
    ("shapes", "Masks and cut regions"),
    ("pixels", "Pixel overlay (masked pixels, curve sources)"),
    ("points", "Marked pixels (hot and dead)"),
    ("box", "q box"),
    ("colorbar", "Colour bar"),
)
"""The kinds of mark a detector view can draw, in the order of its Marks menu."""


class MarkLayers(QObject):
    changed = pyqtSignal(str, bool)
    """A kind of mark was switched on or off by the person (``key``, visible)."""

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._titles: dict[str, str] = {}
        self._visible: dict[str, bool] = {}
        self._items: dict[str, dict] = {}
        self._used: set[str] = set()
        self._removals: list[tuple[str, Callable[[], None], Optional[Callable[[], bool]], str]] = []

    # -- kinds and items -----------------------------------------------------------------

    def add_layer(self, key: str, title: str) -> None:
        self._titles[key] = title
        self._visible.setdefault(key, True)
        self._items.setdefault(key, {})

    def set_title(self, key: str, title: str) -> None:
        if key in self._titles:
            self._titles[key] = title

    def track(self, key: str, item, wanted: bool = True) -> None:
        """Record whether the page wants ``item`` shown, and show it if its kind is on."""
        if key not in self._titles:
            self.add_layer(key, key)
        self._items[key][item] = bool(wanted)
        if wanted:
            self._used.add(key)
        item.setVisible(bool(wanted) and self._visible[key])

    def forget(self, key: str, item) -> None:
        self._items.get(key, {}).pop(item, None)

    def wanted(self, item) -> bool:
        return any(items.get(item, False) for items in self._items.values())

    def is_visible(self, key: str) -> bool:
        return self._visible.get(key, True)

    def set_visible(self, key: str, visible: bool, *, notify: bool = False) -> None:
        if key not in self._titles:
            return
        visible = bool(visible)
        changed = self._visible.get(key, True) != visible
        self._visible[key] = visible
        for item, wanted in list(self._items[key].items()):
            try:
                item.setVisible(wanted and visible)
            except RuntimeError:  # an item deleted with its scene
                self._items[key].pop(item, None)
        if changed and notify:
            self.changed.emit(key, visible)

    def hidden(self) -> list[str]:
        return [key for key, visible in self._visible.items() if not visible]

    def set_hidden(self, keys) -> None:
        """Restore remembered choices (unknown keys are kept for kinds added later)."""
        for key in keys or ():
            self._visible[str(key)] = False
        for key in self._titles:
            self.set_visible(key, self._visible.get(key, True))

    def present(self) -> list[str]:
        """The kinds this view has drawn at least once, in the order they were added."""
        return [key for key in self._titles if key in self._used]

    # -- removals and the menu -----------------------------------------------------------

    def add_removal(self, title: str, remove: Callable[[], None], enabled: Optional[Callable[[], bool]] = None,
                    tip: str = "") -> None:
        self._removals.append((title, remove, enabled, tip))

    def menu_button(self, parent: QWidget, *, text: str = "Marks", icon=None) -> QToolButton:
        from ..i18n import tr

        button = QToolButton(parent)
        button.setObjectName("marksButton")
        button.setText(text)
        button.setToolTip(tr("Show or hide each kind of mark on this view; remove the ones you made"))
        if icon is not None:
            from PyQt5.QtCore import Qt

            button.setIcon(icon)
            button.setToolButtonStyle(Qt.ToolButtonIconOnly)
            button.setAutoRaise(True)
        button.setPopupMode(QToolButton.InstantPopup)
        menu = QMenu(button)
        menu.aboutToShow.connect(lambda: self._fill(menu))
        button.setMenu(menu)
        self.button = button
        return button

    def _fill(self, menu: QMenu) -> None:
        from ..i18n import tr

        menu.clear()
        keys = self.present()
        if not keys:
            empty = menu.addAction(tr("No marks on this view yet"))
            empty.setEnabled(False)
        for key in keys:
            action = menu.addAction(tr(self._titles[key]))
            action.setCheckable(True)
            action.setChecked(self.is_visible(key))
            action.toggled.connect(lambda on, key=key: self.set_visible(key, on, notify=True))
        if len(keys) > 1:
            menu.addSeparator()
            menu.addAction(tr("Show All Marks"), lambda: self._all(True))
            menu.addAction(tr("Hide All Marks"), lambda: self._all(False))
        if self._removals:
            menu.addSeparator()
            for title, remove, enabled, tip in self._removals:
                action = menu.addAction(tr(title), remove)
                action.setEnabled(bool(enabled()) if enabled is not None else True)
                if tip:
                    action.setToolTip(tr(tip))
            menu.setToolTipsVisible(True)

    def _all(self, visible: bool) -> None:
        for key in self.present():
            self.set_visible(key, visible, notify=True)


__all__ = ["DETECTOR_MARKS", "MarkLayers"]
