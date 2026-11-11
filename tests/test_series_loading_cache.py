"""Late detector reads and previews must not restore data invalidated by Watch."""

from pathlib import Path
from threading import Event, Thread
from types import SimpleNamespace

import numpy as np
import pytest

from src.gimap.features.fitting.application import LoadCurveRequest
from src.gimap.features.fitting.presentation.single.series_frames import FitSeriesFramesMixin
from src.gimap.features.fitting.presentation.single.series_watch import file_stamp
from tests.test_fit_series import _loaded, _pages, _series, _write


def test_late_detector_worker_cannot_overwrite_a_newer_published_cache(tmp_path):
    entered, release = Event(), Event()
    path = tmp_path / "synthetic.tif"
    request = LoadCurveRequest(path, "angstrom")
    cache, previous = {}, []
    source = {"value": 1.0}

    def cut(_recipe, _token, _summed):
        value = source["value"]
        if value == 1.0:
            entered.set()
            if not release.wait(10):
                raise RuntimeError("test did not release the old detector read")
        return {"q": np.array([0.1, 0.2, 0.3]), "intensity": np.full(3, value)}

    bridge = SimpleNamespace(cut=cut)

    def loader():
        return FitSeriesFramesMixin.make_curve_loader(
            load=None, bridge=bridge, recipe=object(), why="", items={}, cache=cache)

    old_load = loader()
    worker = Thread(target=lambda: previous.append(old_load(request)))
    worker.start()
    try:
        assert entered.wait(5)
        source["value"] = 9.0
        current = loader()(request)
        np.testing.assert_array_equal(current.value.intensity, np.full(3, 9.0))
        cache[str(path)] = current  # the GUI accepted the newer frame's outcome
        release.set()
        worker.join(5)
        assert not worker.is_alive() and len(previous) == 1
        np.testing.assert_array_equal(previous[0].value.intensity, np.ones(3))
        assert cache[str(path)] is current, "a late worker changed the GUI-owned frame cache"
        np.testing.assert_array_equal(loader()(request).value.intensity, np.full(3, 9.0))
    finally:
        release.set()
        worker.join(5)


@pytest.mark.parametrize("failed", [False, True], ids=["result", "error"])
def test_watch_invalidation_rejects_a_late_preview_callback(tmp_path, monkeypatch, failed):
    single, page = _pages()
    pending = []
    try:
        # The third frame is not among the initial first/middle/last/second previews.
        monkeypatch.setattr(page, "_schedule_stages", lambda: None)
        folder = _series(tmp_path, (5.0, 5.1, 5.2, 5.3, 5.4, 5.5))
        assert page.open_series(folder)
        _loaded(page)
        monkeypatch.setattr(page.preview_tasks, "submit",
                            lambda _key, work, **callbacks: pending.append((work, callbacks)))
        page.frame_list.setCurrentRow(2)
        assert len(pending) == 1
        old_work, old_callbacks = pending[0]
        old_outcome = old_work()
        path = page.paths[2]
        old_stamp = file_stamp(path)
        _write(Path(path), 8.0, 44)
        stamps = {item: file_stamp(item) for item in page.paths}
        assert stamps[path] != old_stamp
        # Two consecutive Watch scans saw the new file; the previous fit read the old one.
        page._seen = dict(stamps)
        page._read_stamps[2] = old_stamp
        page._accept_new_curves(list(page.paths), stamps)
        assert 2 in page.queue
        if failed:
            old_callbacks["on_error"]("obsolete file error", "")
        else:
            old_callbacks["on_done"](old_outcome)
        key = (path, page._preview_unit())
        assert key not in page._preview_curves, "Watch's invalidation was undone by an old preview"
        assert path not in page._read_errors, "an obsolete read error hides the new frame"
        page._select(2)
        assert len(pending) >= 2  # the new file can be requested, rather than stuck pending forever
        new_work, new_callbacks = pending[-1]
        new_outcome = new_work()
        new_callbacks["on_done"](new_outcome)
        np.testing.assert_array_equal(page._preview_curves[key].intensity, new_outcome.value.intensity)
        assert not np.array_equal(new_outcome.value.intensity, old_outcome.value.intensity)
    finally:
        page.dispose()
        single.dispose()
