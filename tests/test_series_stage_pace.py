"""The stages of a live Series map are searched one map at a time (``series_stages._stages_then``): a map drawn while
a search is under way is searched once it ends — the map shown last, not every map between. Synthetic rows."""

from __future__ import annotations

import os
import threading
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tests.test_series_growing import _row


def test_one_stage_search_at_a_time_and_the_map_shown_last_is_searched_next(monkeypatch) -> None:
    from src.gimap.features.analyze.presentation.bindings import series_stages
    from tests.test_analyze_workspace import _app, _context, _page

    search = series_stages.stages_of
    lock, running, most, searched = threading.Lock(), [0], [0], []

    def slow(series, x_range):
        with lock:
            running[0] += 1
            most[0] = max(most[0], running[0])
        try:
            time.sleep(0.3)
            searched.append(series.rows)
            return search(series, x_range)
        finally:
            with lock:
                running[0] -= 1

    monkeypatch.setattr(series_stages, "stages_of", slow)
    page = _page(_context(None))
    try:
        known = []
        page._series_key, page._series_built = "horizontal", {}
        page._series_rows = [page._series_live_row(_row(k, known=known)) for k in range(30)]
        for _drawing in range(3):  # (a live map drawn again while the search of the first is under way)
            page._series_rows.extend(page._series_live_row(_row(len(page._series_rows) + k, known=known))
                                     for k in range(5))
            page._show_series_map(keep_view=True, live=True)
            page._find_stages()
        assert page.tasks.wait(60)
        _app().processEvents()
        assert most[0] == 1 and searched == [35, 45]  # (the map of 40 rows was never the one shown when a search ended)
        assert page._series_map.rows == 45 and "Looking for stages" not in page.series_stages_label.text()
    finally:
        page.dispose()
