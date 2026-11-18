"""Details ▸ Storage: the import caches of Analyze, their limit, and the size of the results (a mixin of
``SputterPage``).

Analyze reads the frames of a scan once per request into an import cache (often gigabytes) and reuses it. The caches
are in a local folder (``application/storage.py``); those of an older GIMaP in the user data folder are listed too
(*user data*). The section, collapsed, is filled in the background when it is opened and after every Analyze or
removal: one row per import (the request's label, its size, when an analysis last used it, where it is; the one in
use is marked), the total, and the size of the results root, which is never removed automatically. **Remove
Selected** and **Remove Unused For n days** ask first; the import in use is never removed. **Limit of the imports**
(GB, kept in the preferences; default 5, 0: no limit) goes with every Analyze: the least recently used imports are
removed while the caches exceed it, when the analysis starts and when it ends.
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

from PyQt5.QtWidgets import QMessageBox, QTableWidgetItem

from src.gimap.app.presentation.i18n import tr

from ..application import DEFAULT_LIMIT_GB, GB, PLACE_USER, total_size, unused_for
from .texts import text

STORAGE_KEY = "sputter_storage"
LIMIT_PREFERENCE = "cache_limit_gb"
SUMMARY = "Imports: {size} in {count} · {folder}"
RESULTS = "Results: {size} in {folder} (never removed automatically)"
NO_STORE = "This analysis package keeps no imports GIMaP can list."
READING = "Adding up the sizes…"
IN_USE = "{label} (in use)"
PLACES = {"local": "local folder", PLACE_USER: "user data folder (older GIMaP)"}
REMOVE_TITLE = "Remove Imports"
REMOVE_QUESTION = ("Remove {count} imports ({size})? Their scans are read again from the frames the next time they are "
                   "analysed; the results stay.")
NOTHING_UNUSED = "No import has gone unused for {days} days."
REMOVED = "Removed {count} imports ({size})."
NOT_REMOVED = "{label} could not be removed: {reason}"
STORAGE_TEMPLATES = (SUMMARY, RESULTS, NO_STORE, READING, IN_USE, *PLACES.values(), REMOVE_TITLE, REMOVE_QUESTION,
                     NOTHING_UNUSED, REMOVED, NOT_REMOVED)
"""The texts of Details ▸ Storage that need Chinese."""


def size_text(size: float) -> str:
    """"1.4 GB", "230 MB"."""
    size = float(size or 0)
    return f"{size / GB:.1f} GB" if size >= 0.95 * GB else f"{size / (1 << 20):,.0f} MB"


class StoragePanelMixin:
    """Needs the widgets of ``views/sputter_steps_view.py`` (``_storage_section``), ``service``, ``tasks``,
    ``results_picker``, ``_stored``, ``_store``, ``_notify``, ``_client_busy``."""

    def _init_storage(self) -> None:
        self._storage_caches: list = []
        self._running_label = ""
        stored = self._stored().get(LIMIT_PREFERENCE)
        self.storage_limit.blockSignals(True)
        self.storage_limit.setValue(int(stored) if isinstance(stored, (int, float)) and stored >= 0 else DEFAULT_LIMIT_GB)
        self.storage_limit.blockSignals(False)
        self.storage_limit.valueChanged.connect(self._keep_limit)
        self.storage_section.expandedChanged.connect(lambda opened: self.refresh_storage() if opened else None)
        self.storage_remove.clicked.connect(self._remove_selected)
        self.storage_remove_unused.clicked.connect(self._remove_unused)

    def cache_limit(self) -> int:
        """The limit of the import caches in bytes (0: none)."""
        return int(self.storage_limit.value()) * GB

    def _keep_limit(self, value: int) -> None:
        stored = self._stored()
        stored[LIMIT_PREFERENCE] = int(value)
        self._store(stored)

    def _store_of(self):
        client = self.service.client
        return client.storage() if client is not None else None

    def _in_use(self) -> tuple:
        return (self._running_label,) if self._client_busy and self._running_label else ()

    # -- listing --------------------------------------------------------------------------------------------------
    def refresh_storage(self) -> None:
        """List the caches and add up the sizes in the background (only while the section is open)."""
        if not self.storage_section.is_expanded():
            return
        store = self._store_of()
        if store is None:
            self.storage_summary.setText(tr(NO_STORE))
            self.storage_table.setRowCount(0)
            return
        self.storage_summary.setText(tr(READING))
        results = Path(self.results_picker.path() or self.service.results_root())

        def work():
            return store.caches(), store.folder_size(results), results, store.roots()

        self.tasks.submit(STORAGE_KEY, work, on_done=self._show_storage,
                          on_error=lambda message, _details: self.storage_summary.setText(message))

    def _show_storage(self, found) -> None:
        caches, results_size, results, roots = found
        caches = sorted(caches, key=lambda cache: cache.last_used, reverse=True)
        self._storage_caches = caches
        in_use = set(self._in_use())
        local = str(roots[0][0]) if roots else ""
        lines = [text(SUMMARY, size=size_text(total_size(caches)), count=len(caches), folder=local),
                 text(RESULTS, size=size_text(results_size), folder=str(results))]
        self.storage_summary.setText("\n".join(lines))
        table = self.storage_table
        table.setRowCount(len(caches))
        for row, cache in enumerate(caches):
            label = text(IN_USE, label=cache.label) if cache.label in in_use else cache.label
            used = datetime.fromtimestamp(cache.last_used).strftime("%Y-%m-%d %H:%M") if cache.last_used else "–"
            cells = (label, size_text(cache.size), used, tr(PLACES.get(cache.place, cache.place)))
            for column, words in enumerate(cells):
                item = QTableWidgetItem(words)
                item.setToolTip(str(Path(cache.folder) / cache.label))
                table.setItem(row, column, item)
        table.resizeColumnsToContents()

    # -- removing -------------------------------------------------------------------------------------------------
    def _remove_selected(self) -> None:
        rows = sorted({index.row() for index in self.storage_table.selectionModel().selectedRows()})
        in_use = set(self._in_use())
        chosen = [self._storage_caches[row] for row in rows
                  if row < len(self._storage_caches) and self._storage_caches[row].label not in in_use]
        self._remove(chosen)

    def _remove_unused(self) -> None:
        days = int(self.storage_days.value())
        chosen = unused_for(self._storage_caches, days, time.time(), keep=self._in_use())
        if not chosen:
            self._notify(text(NOTHING_UNUSED, days=days))
            return
        self._remove(chosen)

    def _remove(self, chosen: list) -> None:
        store = self._store_of()
        if not chosen or store is None:
            return
        size = size_text(total_size(chosen))
        answer = QMessageBox.question(self, tr(REMOVE_TITLE), text(REMOVE_QUESTION, count=len(chosen), size=size),
                                      QMessageBox.Yes | QMessageBox.Cancel, QMessageBox.Cancel)
        if answer != QMessageBox.Yes:
            return

        def work():
            failed = []
            for cache in chosen:
                try:
                    store.remove(cache)
                except OSError as exc:
                    failed.append((cache.label, str(exc)))
            return failed

        def done(failed) -> None:
            if failed:
                self._notify(text(NOT_REMOVED, label=failed[0][0], reason=failed[0][1]), error=True)
            else:
                self._notify(text(REMOVED, count=len(chosen), size=size))
            self.refresh_storage()

        self.tasks.submit(STORAGE_KEY + "_remove", work, on_done=done,
                          on_error=lambda message, _details: self._notify(message, error=True))


__all__ = ["LIMIT_PREFERENCE", "STORAGE_TEMPLATES", "StoragePanelMixin", "size_text"]
