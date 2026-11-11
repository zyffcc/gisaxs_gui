"""Curve discovery and reading release the UI thread; Cancel discards unfinished imports."""

import threading

import pytest
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication

from tests.test_compare import _close, _idle, _map, _run, _shown_page, _wait, _write_curves


@pytest.mark.parametrize("folder", [False, True])
def test_file_and_folder_import_run_off_thread_and_cancel_without_partial_series(tmp_path, monkeypatch, folder):
    page = _shown_page()
    release, entered = threading.Event(), threading.Event()
    main_thread = threading.get_ident()
    threads, clicks = [], []
    files = _write_curves(tmp_path / "synthetic", _run(30, seed=1))
    reader = page.service.reader

    class Slow:
        def read(self, path):
            threads.append(threading.get_ident())
            entered.set()
            if threading.get_ident() != main_thread:
                assert release.wait(10)
            return reader.read(path)

    try:
        monkeypatch.setattr(page.service, "reader", Slow())
        if folder:
            page.add_folder(files[0].parent)
        else:
            page.add_files(files)
        assert page._reading
        QTimer.singleShot(0, lambda: clicks.append("responsive"))
        _wait(page, lambda: entered.is_set() and bool(clicks))
        assert set(threads) == {threads[0]} and main_thread not in threads
        page.import_job.cancel_button.click()
        assert not page._reading and not page.series
        release.set()
        assert page.tasks.wait(10)
        assert not page.series and len(threads) == 1
        assert page.status_label.text() == "Import cancelled. Unfinished series were not added."
    finally:
        release.set()
        _close(page)


def test_folder_discovery_does_not_block_input_and_dispose_rejects_late_results(tmp_path, monkeypatch):
    page = _shown_page()
    release, entered = threading.Event(), threading.Event()
    main_thread = threading.get_ident()
    threads = []
    _write_curves(tmp_path / "synthetic", _run(4, seed=1))
    discover = page.service.folder_kinds

    def slow(folder):
        threads.append(threading.get_ident())
        entered.set()
        if threading.get_ident() != main_thread:
            assert release.wait(10)
        return discover(folder)

    try:
        monkeypatch.setattr(page.service, "folder_kinds", slow)
        page.add_folder(tmp_path / "synthetic")
        assert page._reading
        _wait(page, entered.is_set)
        assert main_thread not in threads
        page.dispose()
        release.set()
        assert page.tasks.wait(10)
        QApplication.processEvents()
        assert page.series == [] and not page._timer.isActive()
    finally:
        release.set()
        page.tasks.wait(10)
        page.close()


@pytest.mark.parametrize("arrays", [False, True])
def test_project_restore_reads_off_thread_and_new_project_rejects_late_data(tmp_path, monkeypatch, arrays):
    import numpy as np
    from src.gimap.features.compare.presentation import reading

    page = _shown_page()
    release, entered = threading.Event(), threading.Event()
    threads = []
    main_thread = threading.get_ident()
    files = _write_curves(tmp_path / "synthetic", _run(4, seed=1))
    if arrays:
        source = _map(1)
        np.savez_compressed(tmp_path / "sample.compare.npz", x_0=source.x, image_0=source.image)
        entry = {"name": "saved map", "arrays": 0, "labels": list(source.labels)}
    else:
        entry = {"name": "saved files", "files": [str(path) for path in files]}
    restore = reading._restore_entry

    def slow(*args):
        threads.append(threading.get_ident())
        entered.set()
        assert release.wait(10)
        return restore(*args)

    try:
        monkeypatch.setattr(reading, "_restore_entry", slow)
        page.apply_project_state({"series": [entry], "arrays_file": "sample.compare.npz"}, tmp_path / "sample.gimap")
        _wait(page, entered.is_set)
        assert threads and main_thread not in threads
        with pytest.raises(ValueError, match="imports to finish"):
            page.project_state(tmp_path / "sample.gimap")
        page.apply_project_state({}, tmp_path / "empty.gimap")
        release.set()
        assert page.tasks.wait(10)
        assert page.series == [] and not page._reading
        assert page.project_state()["series"] == []
    finally:
        release.set()
        _close(page)


def test_failed_import_keeps_error_visible_and_does_not_recompare_existing_result(tmp_path, monkeypatch):
    from src.gimap.app.presentation.components import visible_toasts

    page = _shown_page()
    try:
        page.add_map(_map(1), "existing")
        _wait(page, lambda: _idle(page) and page.comparison_current())
        calls = []
        compare = page.service.compare
        monkeypatch.setattr(page.service, "compare", lambda *args: (calls.append(True), compare(*args))[1])
        page.add_files([tmp_path / "missing_a.dat", tmp_path / "missing_b.dat"])
        assert page.tasks.wait(10)
        _wait(page, lambda: _idle(page))
        assert not calls and len(page.series) == 1
        assert page.comparison_current() and page.save_button.isEnabled()
        assert page.status_label.text().startswith("Fewer than two curves could be read")
        assert any(toast.text().startswith("Fewer than two curves could be read") for toast in visible_toasts(page))
    finally:
        _close(page)
