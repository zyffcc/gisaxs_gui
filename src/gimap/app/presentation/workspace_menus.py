"""The Edit, Analyze and Fitting menus: the commands of the pages, findable in a menu with their keys.

The builder knows the entries by key only; the composition root (``src.gimap.app.workspace_commands``) says
what each one does, whether it can run now, and why not. An entry that is not provided is not shown, and a
menu without any entry is not added to the menu bar.

Each time a menu opens (``aboutToShow``) its entries are enabled or disabled from the state of the pages, and a
disabled entry's tooltip says why; a state check that fails is a bug, reported once through ``sys.excepthook``
(ErrorGuard logs and shows it) while its entry says that it could not check. The keys of the pages (``Ctrl+Z``,
``F5``, ``Ctrl+E``, ``Ctrl+Shift+E``, ``Ctrl+Return``) stay the pages' own shortcuts: a menu entry shows its key
as text after a tab (``"&Undo\\tCtrl+Z"``, drawn in the menu's key column) and holds no shortcut, so only the
page's ``QShortcut`` takes the key, and only while the page has the focus. A ``Qt.WidgetShortcut`` on the entry
would not do: after Alt the menu bar has the focus, and Qt then runs the entry's key from any page. One key
never has two owners — Qt would run neither — and a stale enabled state can never run a command through its key.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Callable, Optional

from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import QAction, QMenu

from .i18n import tr, trf


@dataclass(frozen=True)
class MenuEntry:
    """What one entry of the Edit, Analyze or Fitting menu does."""

    run: Callable[[], object]
    blocked: Callable[[], str] = lambda: ""
    """``""`` when the entry can run now; else why not (English, shown translated as the entry's tooltip)."""
    detail: Optional[Callable[[], str]] = None
    """Undo / Redo: what the next step changes, in the interface language (``""``: the page does not say)."""


@dataclass(frozen=True)
class Entry:
    key: str
    text: str
    tip: str
    page_keys: tuple[str, ...] = ()
    """Keys the page owns: the first is shown in the menu (as text), never taken by it."""


@dataclass(frozen=True)
class Submenu:
    key: str
    title: str
    tip: str
    entries: tuple


SEPARATOR = None

STARTING = "GIMaP is still starting; try again in a moment."
CHECK_FAILED = "This entry could not check whether it can run; the details are in the error log."

EDIT_MENU = (
    Entry("edit_undo", "&Undo", "Undo the last change on the page shown: the set-up in Analyze, the model in "
                                "Fitting's Single analysis", ("Ctrl+Z",)),
    Entry("edit_redo", "&Redo", "Redo what was undone last (Ctrl+Y or Ctrl+Shift+Z)", ("Ctrl+Y", "Ctrl+Shift+Z")),
)
DETAIL_TEXT = {"edit_undo": "&Undo: {what}", "edit_redo": "&Redo: {what}"}
"""The text of an entry whose page says what it changes (``MenuEntry.detail``)."""

ANALYZE_MENU = (
    Entry("analyze_run_automatic", "Run &Automatic Analysis",
          "The standard procedure without AI: geometry (finding a calibration if needed), "
          "mask, cuts and results, each step with what it found"),
    Entry("analyze_stop_automatic", "S&top Automatic Analysis",
          "End the automatic analysis before its next step; what it found so far is kept"),
    Entry("analyze_repeat", "&Repeat Analysis", "Analyse the frame shown again with the set-up as it is", ("F5",)),
    SEPARATOR,
    Entry("analyze_export", "&Export Curves…",
          "Write CSV curves and a JSON record next to the data (gimap_analysis/) — Ctrl+E", ("Ctrl+E",)),
    Entry("analyze_batch", "&Batch Export…",
          "Every listed frame with the current settings, to a folder you choose — Ctrl+Shift+E", ("Ctrl+Shift+E",)),
    Entry("analyze_batch_folder", "Batch Export a F&older…",
          "A folder of raw frames straight to curves: Batch Export opens once its first frame is analysed"),
    SEPARATOR,
    Entry("analyze_send_fitting", "Send to &Fitting", "The horizontal cut (GISAXS) or I(q) (GIWAXS) opened in Fitting."),
    Entry("analyze_send_series", "Send Series to Fitti&ng…",
          "Export every listed frame (or sum of frames) and open the series in Fitting ▸ In-situ series"),
    Entry("analyze_send_compare", "Send Series to &Compare",
          "Add this series (every frame's curve) to Compare, to set it beside other samples or series"),
    SEPARATOR,
    Submenu("analyze_setup_files", "&Set-up Files", "Settings, cut regions and masks as files, for the next data set", (
        Entry("analyze_save_settings", "Save &Settings…",
              "Geometry, masks, corrections and cut regions as a file, for the next data set"),
        Entry("analyze_load_settings", "Load S&ettings…", "Use a set-up saved before"),
        SEPARATOR,
        Entry("analyze_save_cuts", "Save &Cuts…", "The GIWAXS cut regions as a .json file"),
        Entry("analyze_load_cuts", "Load C&uts…", "Cut regions saved before (.json)"),
        SEPARATOR,
        Entry("analyze_save_masks", "Save &Masks…", "The drawn masks as a .json file, to reuse with other data of this set-up"),
        Entry("analyze_load_masks", "Load M&asks…",
              "Masks saved by GIMaP (.json) or a mask image of the same size (EDF, TIFF: non-zero = masked)"),
    )),
    Submenu("analyze_figures", "Fi&gures", "The view as a figure, or its data as a table", (
        Entry("analyze_save_image", "Save &Image…", "The detector or q map as shown, with a colour bar"),
        Entry("analyze_save_map", "Save &q-Map Data…", "The intensity on a regular q grid, as a CSV table with its axes"),
        Entry("analyze_save_view", "Save &View Data…", "The q map (q∥–qz) or the cake (χ–q) as a table with its axes"),
    )),
    SEPARATOR,
    Entry("analyze_clear", "C&lear Frame List",
          "Remove every file from the list. One file: select it and press Delete, or right-click it"),
)

FITTING_MENU = (
    Entry("fitting_open_curve", "&Open Curve…", "Open a 1D curve (q, I, σ) — Analyze ▸ Send to Fitting opens its cut here"),
    Entry("fitting_fit_folder", "Fit a Fol&der of Curves…",
          "In-situ series: every curve of a folder (a series, or many samples), each fitted with the model of this "
          "page — fit one curve here first, or let each curve be predicted"),
    SEPARATOR,
    Entry("fitting_fit", "&Fit", "Run the method chosen in the Fit step (Ctrl+Return)", ("Ctrl+Return",)),
    Entry("fitting_stop", "S&top Fit", "Stop the fit; the best values so far are kept"),
    SEPARATOR,
    Entry("fitting_load_model", "&Load Model…", "A model saved from Fitting (JSON), for this curve"),
    Entry("fitting_save_model", "Save &Model…", "The model (values, fit/fixed, ranges) as JSON, to load for another curve"),
    Entry("fitting_save_data", "Save Data &and Fit…",
          "CSV: q, I, σ, the model, the residuals and each term; a JSON record of the model and the fit next to it"),
    Entry("fitting_save_plot", "Save &Plot…", "The plot as shown: PNG (image) or SVG (vector)"),
    SEPARATOR,
    Submenu("fitting_series", "&In-situ Series", "Every curve of a folder fitted with the model of Single analysis", (
        Entry("fitting_series_start", "&Start", "Fit every listed curve with the model of Single analysis"),
        Entry("fitting_series_stop", "S&top", "Stop after the frame being fitted; the frames done are kept"),
    )),
    Entry("fitting_1d_predict", "1D Predict — Fit Many &Curves…",
          "1D Predict: a list of curve files fitted one after another, with their results"),
)


def key_suffix(item: Entry) -> str:
    """``"\\t<key>"`` for an entry whose page owns a key (QMenu draws it in the key column); else ``""``."""
    return "\t" + QKeySequence(item.page_keys[0]).toString(QKeySequence.NativeText) if item.page_keys else ""


def menu_keys(layout) -> list[str]:
    """The keys of every entry of ``layout`` (``EDIT_MENU`` …), those in its submenus too."""
    keys: list[str] = []
    for item in layout:
        if isinstance(item, Submenu):
            keys += menu_keys(item.entries)
        elif item is not SEPARATOR:
            keys.append(item.key)
    return keys


class WorkspaceMenus:
    """Fills the Edit, Analyze and Fitting menus of a ``MainMenuBar`` from ``entries`` (``MenuEntry`` by key)."""

    def __init__(self, window, entries: dict, actions: dict, submenus: list):
        self.window = window
        self.entries = dict(entries or {})
        self.actions = actions
        self.submenus = submenus
        self.specs: dict[str, Entry] = {}
        self._reported: set[str] = set()  # entries whose state check failed: reported once

    def fill(self, menu: QMenu, layout) -> bool:
        """The entries of ``layout`` that are provided, in ``menu``; ``True`` when there is one at least."""
        added, pending = [], False
        for item in layout:
            if item is SEPARATOR:
                pending = bool(added)
                continue
            if isinstance(item, Submenu):
                submenu = QMenu(item.title, menu)
                submenu.menuAction().setToolTip(item.tip)
                submenu.setToolTipsVisible(True)
                if not self.fill(submenu, item.entries):
                    submenu.deleteLater()
                    continue
                if pending:
                    menu.addSeparator()
                menu.addMenu(submenu)
                self.submenus.append(submenu)
                added.append(item.key)
                pending = False
                continue
            entry = self.entries.get(item.key)
            if entry is None:
                continue
            if pending:
                menu.addSeparator()
            menu.addAction(self._action(item, entry))
            added.append(item.key)
            pending = False
        if added:
            menu.aboutToShow.connect(lambda shown=menu: self.sync(shown))
        return bool(added)

    def _action(self, item: Entry, entry: MenuEntry) -> QAction:
        action = QAction(item.text + key_suffix(item), self.window)  # the page's key shown, never taken
        action.setToolTip(item.tip)
        action.setStatusTip(item.tip)
        action.setData(item.key)
        action.triggered.connect(lambda _checked=False, key=item.key: self.run(key))
        self.actions[item.key] = action
        self.specs[item.key] = item
        return action

    def run(self, key: str) -> None:
        self.entries[key].run()

    def sync(self, menu: QMenu) -> None:
        """Enable each entry of ``menu`` that can run now; a disabled one says why in its tooltip."""
        for action in menu.actions():
            key = action.data()
            if isinstance(key, str) and key in self.specs:
                self.sync_entry(key)

    def sync_all(self) -> None:
        """Every entry, as when its menu opens (tests, and a caller that shows the state elsewhere)."""
        for key in self.specs:
            self.sync_entry(key)

    def sync_entry(self, key: str) -> None:
        action, item, entry = self.actions[key], self.specs[key], self.entries[key]
        try:
            reason = entry.blocked() or ""
            detail = entry.detail() if entry.detail is not None else None
        except RuntimeError:  # a page whose Qt object is already deleted (the window is closing)
            reason, detail = STARTING, None
        except Exception:  # a bug in a state check: ErrorGuard logs and shows it once; the other entries still sync
            if key not in self._reported:
                self._reported.add(key)
                sys.excepthook(*sys.exc_info())
            reason, detail = CHECK_FAILED, None
        action.setEnabled(not reason)
        tip = tr(reason) if reason else tr(item.tip)
        action.setToolTip(tip)
        action.setStatusTip(tip)
        if detail is not None:
            what = str(detail).replace("&", "&&")
            action.setText((trf(DETAIL_TEXT[key], what=what) if what else tr(item.text)) + key_suffix(item))


__all__ = ["ANALYZE_MENU", "CHECK_FAILED", "EDIT_MENU", "FITTING_MENU", "MenuEntry", "WorkspaceMenus", "key_suffix",
           "menu_keys"]
