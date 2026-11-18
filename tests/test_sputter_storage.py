"""Where Analyze keeps its import caches and how much room they take: the local folder, imports of an older GIMaP in
the user data folder, the limit (least recently used first, never the one in use, never a result) and Details ▸
Storage. Invented folders and labels only."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from src.gimap.features.sputter.application import GB, ImportCache, over_limit, unused_for
from src.gimap.features.sputter.application.storage import prune
from src.gimap.features.sputter.infrastructure import CACHES_REMOVED, ClientPackageRunner, local_data_dir
from src.gimap.features.sputter.infrastructure.client import IMPORT_MARKER, PACKAGE_MASK, STORAGE_FOLDER
from src.gimap.features.sputter.infrastructure.process import Outcome
from tests.sputter_client_fixtures import FakeClientRunner, package_folder
from tests.test_sputter_client import _request
from tests.test_sputter_main_screen import _close, _page


@pytest.fixture(autouse=True)
def _no_package_of_this_computer(monkeypatch):
    monkeypatch.delenv("GIMAP_SPUTTER_CLIENT", raising=False)


def test_the_caches_go_to_a_local_folder_not_the_roaming_profile(tmp_path):
    home = tmp_path / "zz_home"
    assert local_data_dir({"LOCALAPPDATA": str(tmp_path / "local")}, "win32", home) == tmp_path / "local" / "GIMaP"
    assert local_data_dir({}, "win32", home) == home / "AppData" / "Local" / "GIMaP"
    assert local_data_dir({}, "darwin", home) == home / "Library" / "Caches" / "GIMaP"
    assert local_data_dir({}, "linux", home) == home / ".cache" / "gimap"
    assert local_data_dir({"XDG_CACHE_HOME": str(tmp_path / "xdg")}, "linux", home) == tmp_path / "xdg" / "gimap"
    assert local_data_dir({"GIMAP_HOME": str(tmp_path / "gh")}, "win32", home) == tmp_path / "gh" / "cache"
    from src.gimap.features.sputter.bootstrap import create_client_service

    runner = create_client_service().runner  # the real one: local caches, the user data folder as the older place
    assert runner.storage_root() == Path(os.environ["GIMAP_HOME"]) / "cache"
    assert runner.legacy_root() == Path(os.environ["GIMAP_HOME"])


def _cache(data: Path, label: str, size: int, used: float, *, finished: bool = True) -> None:
    (data / label).mkdir(parents=True, exist_ok=True)
    (data / label / "frames.bin").write_bytes(b"\0" * size)
    if finished:
        (data / label / "scan.json").write_text("{}", encoding="utf-8")
        (data / (label + IMPORT_MARKER)).write_text("{}", encoding="utf-8")
        os.utime(data / (label + IMPORT_MARKER), (used, used))
    (data / (label + ".trace")).mkdir(exist_ok=True)
    (data / (label + ".trace") / "reading.json").write_text("{}", encoding="utf-8")


class _Process:
    def __init__(self):
        self.calls = []

    def __call__(self, command, *, on_line, should_stop, env=None, cwd=None, **_more):
        self.calls.append(dict(env or {}))
        on_line("[1/4] intake ...")
        return Outcome(0, [], False)


def _runner(tmp_path, process=None) -> ClientPackageRunner:
    return ClientPackageRunner(lambda: tmp_path / "local", python="zz_python", run=process or _Process(),
                               chosen=package_folder(tmp_path / "package"), local=tmp_path / "nowhere",
                               workers=lambda: 2, legacy_root=lambda: tmp_path / "roaming")


def test_caches_are_listed_and_the_least_recently_used_go_first(tmp_path):
    runner = _runner(tmp_path)
    package = runner.package()
    local = tmp_path / "local" / STORAGE_FOLDER / package.identity[:12] / "data" / PACKAGE_MASK
    roaming = tmp_path / "roaming" / STORAGE_FOLDER / package.identity[:12] / "data" / PACKAGE_MASK
    now = time.time()
    _cache(local, "gui_zz_a_00000001", 3000, now - 10 * 86400)
    _cache(local, "gui_zz_b_00000002", 2000, now - 1 * 86400)
    _cache(roaming, "gui_zz_c_00000003", 4000, now - 40 * 86400)
    (local / "not_ours").mkdir()  # the package's own folders are never listed
    caches = {cache.label: cache for cache in runner.caches.caches()}
    assert set(caches) == {"gui_zz_a_00000001", "gui_zz_b_00000002", "gui_zz_c_00000003"}
    assert caches["gui_zz_c_00000003"].place == "user data" and caches["gui_zz_a_00000001"].place == "local"
    assert caches["gui_zz_a_00000001"].size > 3000 and len(caches["gui_zz_a_00000001"].paths) == 3
    assert [cache.label for cache in over_limit(caches.values(), 6000)] == ["gui_zz_c_00000003"]
    assert [cache.label for cache in over_limit(caches.values(), 3000, keep=("gui_zz_c_00000003",))] == \
        ["gui_zz_a_00000001", "gui_zz_b_00000002"]
    assert over_limit(caches.values(), 0) == [] and over_limit(caches.values(), None) == []
    assert [cache.label for cache in unused_for(caches.values(), 7, now)] == ["gui_zz_c_00000003", "gui_zz_a_00000001"]
    removed = prune(runner.caches, 6000)
    assert [cache.label for cache in removed] == ["gui_zz_c_00000003"]
    assert not (roaming / "gui_zz_c_00000003").exists() and not (roaming / "gui_zz_c_00000003.trace").exists()
    assert not (roaming / ("gui_zz_c_00000003" + IMPORT_MARKER)).exists()
    assert runner.caches.folder_size(local) > 5000


def test_an_import_finished_in_the_user_data_folder_is_still_used(tmp_path):
    runner = _runner(tmp_path)
    package = runner.package()
    request = _request(tmp_path / "scan")
    label = request.label()
    data, _work = runner.roots(package, request)
    assert data == tmp_path / "local" / STORAGE_FOLDER / package.identity[:12] / "data" / PACKAGE_MASK
    roaming = tmp_path / "roaming" / STORAGE_FOLDER / package.identity[:12] / "data" / PACKAGE_MASK
    _cache(roaming, label, 10, time.time() - 100, finished=False)  # cut off there: the new import goes local
    assert runner.roots(package, request)[0] == data
    _cache(roaming, label, 10, time.time() - 100)
    assert runner.roots(package, request)[0] == roaming
    assert runner.import_trace(request) == roaming / (label + ".trace")
    process = _Process()
    runner = _runner(tmp_path, process)
    before = (roaming / (label + IMPORT_MARKER)).stat().st_mtime
    runner.run(request, tmp_path / "out", lambda _line: None, lambda: False)
    assert process.calls[0]["IAM_DATA"] == str(roaming)
    assert (roaming / (label + IMPORT_MARKER)).stat().st_mtime > before  # used now


def test_analyze_keeps_the_caches_under_the_limit_but_never_the_one_in_use(tmp_path):
    runner = _runner(tmp_path)
    package = runner.package()
    request = _request(tmp_path / "scan", cache_limit=5000)
    label = request.label()
    local = tmp_path / "local" / STORAGE_FOLDER / package.identity[:12] / "data" / PACKAGE_MASK
    now = time.time()
    _cache(local, label, 9000, now - 30 * 86400)  # the oldest, but the one in use
    _cache(local, "gui_zz_other_00000004", 3000, now - 5 * 86400)
    results = tmp_path / "results" / "zz_scan-1234"
    results.mkdir(parents=True)
    (results / "kept.json").write_text("{}", encoding="utf-8")
    lines = []
    runner.run(request, tmp_path / "out", lines.append, lambda: False)
    assert (local / label).is_dir() and not (local / "gui_zz_other_00000004").exists()
    assert (results / "kept.json").is_file()  # results are never removed
    removed = [line for line in lines if isinstance(line, str) and line.startswith("Import caches over the limit")]
    assert removed == [CACHES_REMOVED.format(limit="0.0 GB", count=1, size="0.0 GB")]
    lines.clear()
    runner.run(_request(tmp_path / "scan", cache_limit=0), tmp_path / "out", lines.append, lambda: False)
    assert not any(isinstance(line, str) and line.startswith("Import caches") for line in lines)


def test_details_storage_lists_the_caches_and_removes_them(tmp_path, monkeypatch):
    from PyQt5.QtWidgets import QMessageBox

    from src.gimap.features.sputter.bootstrap import create_client_service, create_sputter_service
    from src.gimap.features.sputter.infrastructure import ClientFolderReader
    from src.gimap.features.sputter.presentation.page import SputterPage
    from src.gimap.app.presentation.task_runner import TaskRunner
    from src.gimap.integrations.state import InMemoryUserPreferencesRepository
    from tests.sputter_fixtures import FakeInspector

    runner = _runner(tmp_path)
    service = create_sputter_service(inspector=FakeInspector(), results_root=lambda: tmp_path / "results",
                                     client=create_client_service(runner=runner, reader=ClientFolderReader()))
    preferences = InMemoryUserPreferencesRepository()
    page = SputterPage(service, task_runner=TaskRunner(), preferences=preferences)
    page.resize(1256, 869)
    page.show()
    assert page.storage_limit.value() == 5 and page.cache_limit() == 5 * GB  # the default limit
    page.storage_limit.setValue(12)
    assert page._stored()["cache_limit_gb"] == 12
    local = tmp_path / "local" / STORAGE_FOLDER / runner.package().identity[:12] / "data" / PACKAGE_MASK
    now = time.time()
    _cache(local, "gui_zz_a_00000001", 2 << 20, now - 50 * 86400)
    _cache(local, "gui_zz_b_00000002", 1 << 20, now - 1 * 86400)
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "r.bin").write_bytes(b"\0" * (3 << 20))
    page.show_details(True)
    page.storage_section.set_expanded(True)
    page.tasks.wait(10)
    table = page.storage_table
    assert table.rowCount() == 2 and table.item(0, 0).text() == "gui_zz_b_00000002"  # the latest used first
    assert table.item(1, 1).text() == "2 MB" and table.item(0, 3).text() == "local folder"
    summary = page.storage_summary.text().split("\n")
    assert summary[0].startswith("Imports: 3 MB in 2 · ") and summary[1].startswith("Results: 3 MB in ")
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *args, **kwargs: QMessageBox.Yes))
    page.storage_days.setValue(30)
    page.storage_remove_unused.click()
    page.tasks.wait(10)
    page.tasks.wait(10)
    assert not (local / "gui_zz_a_00000001").exists() and (local / "gui_zz_b_00000002").exists()
    assert table.rowCount() == 1
    page._client_busy, page._running_label = True, "gui_zz_b_00000002"  # the import in use is never removed
    table.selectRow(0)
    page.storage_remove.click()
    page.tasks.wait(10)
    assert (local / "gui_zz_b_00000002").exists()
    page._client_busy = False
    _close(page)


def test_a_request_carries_the_limit_of_the_page(tmp_path):
    from tests.sputter_fixtures import scan_folder
    from tests.test_sputter_main_screen import _fill, _settle

    client = FakeClientRunner()
    page = _page(tmp_path, client=client)
    page.storage_limit.setValue(0)  # no limit
    _fill(page, scan_folder(tmp_path))
    page.analyze()
    _settle(lambda: not page.analyzing())
    assert client.requests[0].cache_limit == 0
    assert ImportCache("gui_zz", tmp_path).key.endswith("gui_zz")
    _close(page)
