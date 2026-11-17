"""Details ▸ Analyze options ▸ Parallel processes (a mixin of ``SputterPage``): how many processes the analysis
package runs with, for Analyze and Static Frames alike.

**Automatic** (the default) shows the number the computer can take now — one process per core, two cores left free
and at most 16, fewer when the free memory is short (``ClientService.automatic_workers``) — worked out again whenever
the box is shown or opened, and once more when an analysis starts: that number is what the package is given
(``--workers``). The other choices are 1 to the number of cores. The choice is kept in the preferences (``workers``:
a number, or ``auto``); a number larger than this computer's cores (kept on another one) counts as its cores.
"""

from __future__ import annotations

import os
from typing import Optional

from src.gimap.app.presentation.i18n import tr

from .texts import text

WORKERS_KEY = "workers"
AUTOMATIC_DATA = "auto"
AUTOMATIC = "Automatic ({count})"
AUTOMATIC_PLAIN = "Automatic"
TEMPLATES = (AUTOMATIC, AUTOMATIC_PLAIN)
"""The texts of the choice that need Chinese."""


class WorkersMixin:
    """Needs ``workers_box`` (``views/sputter_main_view.py``), ``service`` and the page's ``_stored`` / ``_store``."""

    def _init_workers(self) -> None:
        self.workers_box.aboutToShow.connect(self._fill_workers)
        self.workers_box.activated.connect(self._workers_chosen)
        self._fill_workers()

    def _cores(self) -> int:
        client = self.service.client
        return client.cores() if client is not None else max(1, os.cpu_count() or 1)

    def _automatic_workers(self) -> Optional[int]:
        client = self.service.client
        return client.automatic_workers() if client is not None else None

    def chosen_workers(self) -> Optional[int]:
        """The number the user chose (at most the cores), or None for Automatic."""
        kept = self._stored().get(WORKERS_KEY)
        if isinstance(kept, bool) or not isinstance(kept, int) or kept < 1:
            return None
        return min(kept, self._cores())

    def run_workers(self) -> Optional[int]:
        """The processes an analysis starting now runs with: the user's number, else Automatic's number now (None
        when the package's runner does not say: it then takes its own)."""
        return self.chosen_workers() or self._automatic_workers()

    def _fill_workers(self) -> None:
        """Automatic with its number now, then 1 … cores; the choice kept is selected."""
        box, automatic, chosen = self.workers_box, self._automatic_workers(), self.chosen_workers()
        box.blockSignals(True)
        box.clear()
        box.addItem(text(AUTOMATIC, count=automatic) if automatic else tr(AUTOMATIC_PLAIN), AUTOMATIC_DATA)
        for count in range(1, self._cores() + 1):
            box.addItem(str(count), count)
        box.setCurrentIndex(max(0, box.findData(chosen)) if chosen else 0)
        box.blockSignals(False)

    def _workers_chosen(self, _index: int = 0) -> None:
        data = self.workers_box.currentData()
        stored = self._stored()
        stored[WORKERS_KEY] = data if isinstance(data, int) else AUTOMATIC_DATA
        self._store(stored)


__all__ = ["AUTOMATIC", "AUTOMATIC_PLAIN", "TEMPLATES", "WORKERS_KEY", "WorkersMixin"]
