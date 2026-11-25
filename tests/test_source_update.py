"""Help ▸ Check for Updates… in a source folder: ``app/source_update.py`` and the update dialog.

A Git clone moves forward to its upstream only when that is a plain fast-forward that touches no file the user
edited; anything else stops before a file changes. A folder from Download ZIP gets exactly the files whose Git
hash differs from GitHub's list, each checked after download; the files it replaces are kept, files GitHub does not
have are never touched, and a failure part-way puts the folder back. Repositories are made in a temporary folder
and GitHub is a dictionary: no test reaches the network.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.gimap.app import source_update as su

GIT = shutil.which("git")
needs_git = pytest.mark.skipif(GIT is None, reason="git is not installed")


# -- what the copy is ---------------------------------------------------------------------------------------


def _source(root: Path) -> Path:
    (root / "src" / "gimap").mkdir(parents=True)
    (root / "main.py").write_text("print('gimap')\n", encoding="utf-8")
    return root


def test_detect_tells_a_clone_a_zip_folder_and_an_installed_copy_apart(tmp_path: Path, monkeypatch) -> None:
    folder = _source(tmp_path / "zip")
    assert su.detect(folder).kind == "folder"
    clone = _source(tmp_path / "clone")
    (clone / ".git").mkdir()
    assert su.detect(clone).kind == "git"
    portable = _source(tmp_path / "portable")
    (portable / "runtime").mkdir()
    (portable / "runtime" / "pythonw.exe").write_bytes(b"")
    assert su.detect(portable).kind == "installed"
    assert su.detect(tmp_path / "site-packages").kind == "installed"  # no main.py: a package installed with pip
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    assert su.detect(clone).kind == "installed"


def test_the_running_source_is_this_checkout_and_restarts_with_the_flag_the_launcher_waits_on() -> None:
    from src.gimap.app.launcher import RESTART_FLAG

    root = su.running_root()
    assert (root / "main.py").is_file() and (root / "src" / "gimap" / "app" / "source_update.py").is_file()
    command = su.restart_command(root)
    assert command[1:] == [str(root / "main.py"), RESTART_FLAG]


# -- a Git clone --------------------------------------------------------------------------------------------


class Lab:
    """A bare origin, the author's clone (who pushes) and the user's copy (where the update runs)."""

    def __init__(self, root: Path, monkeypatch) -> None:
        config = root / "gitconfig"
        config.write_text("[init]\n\tdefaultBranch = main\n[core]\n\tautocrlf = false\n", "ascii")
        monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(config))
        monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
        self.origin, self.author, self.copy = root / "origin.git", root / "author", root / "copy"
        self.git(root, "init", "--bare", "-q", str(self.origin))
        self.git(root, "clone", "-q", str(self.origin), str(self.author))
        _source(self.author)
        self.write("requirements.txt", "numpy\n")
        self.write("utils/sputter_client/MANIFEST.json", "{}\n")
        self.write("utils/sputter_client/run.py", "print(1)\nprint(2)\n")
        self.commit("first")
        self.git(self.author, "push", "-q", "origin", "main")
        self.git(root, "clone", "-q", str(self.origin), str(self.copy))

    @staticmethod
    def git(where: Path, *arguments: str) -> str:
        return subprocess.run([GIT, "-C", str(where), "-c", "user.name=t", "-c", "user.email=t@example.invalid",
                               *arguments], check=True, capture_output=True, text=True).stdout.strip()

    def write(self, name: str, text: str, where: Path | None = None) -> None:
        path = (where or self.author) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text.encode("utf-8"))

    def commit(self, message: str, where: Path | None = None) -> None:
        self.git(where or self.author, "add", "-A")
        self.git(where or self.author, "commit", "-q", "-m", message)

    def publish(self, name: str, text: str, message: str) -> None:
        self.write(name, text)
        self.commit(message)
        self.git(self.author, "push", "-q", "origin", "main")

    def installation(self) -> su.Installation:
        return su.Installation("git", self.copy)


@needs_git
def test_a_clone_moves_forward_to_the_newest_commit_and_says_what_is_new(tmp_path: Path, monkeypatch) -> None:
    lab = Lab(tmp_path, monkeypatch)
    status = su.check(lab.installation())
    assert status.state == "up_to_date" and status.current == status.latest
    lab.publish("main.py", "print('new')\n", "Say new")
    lab.publish("requirements.txt", "numpy\nscipy\n", "Need scipy")
    lab.write("notes.txt", "mine\n", lab.copy)  # untracked: never touched
    lab.write("utils/sputter_client/run.py", "print(1)\nprint(2)\n# my edit\n", lab.copy)  # not changed upstream
    status = su.check(lab.installation())
    assert status.state == "behind" and status.n_new == 2
    assert [line.split(" ", 1)[1] for line in status.new_commits] == ["Need scipy", "Say new"]
    result = su.apply(status, tmp_path / "backup")
    assert result.updated and result.new == lab.git(lab.origin, "rev-parse", "main")
    assert result.requirements == ["requirements.txt"] and result.changed_files == 2
    assert (lab.copy / "main.py").read_text() == "print('new')\n"
    assert (lab.copy / "notes.txt").read_text() == "mine\n"
    assert (lab.copy / "utils/sputter_client/run.py").read_text().endswith("# my edit\n")
    assert su.check(lab.installation()).state == "up_to_date"


@needs_git
def test_a_clone_is_left_alone_when_the_update_would_overwrite_an_edit(tmp_path: Path, monkeypatch) -> None:
    lab = Lab(tmp_path, monkeypatch)
    lab.publish("main.py", "print('new')\n", "Say new")
    lab.write("main.py", "print('my own')\n", lab.copy)
    head = lab.git(lab.copy, "rev-parse", "HEAD")
    status = su.check(lab.installation())
    assert status.state == "blocked" and status.values["files"] == "main.py"
    with pytest.raises(su.UpdateError):
        su.apply(status, tmp_path / "backup")
    assert lab.git(lab.copy, "rev-parse", "HEAD") == head
    assert (lab.copy / "main.py").read_text() == "print('my own')\n"


@needs_git
def test_a_clone_with_commits_of_its_own_or_a_rewritten_history_is_not_updated(tmp_path: Path, monkeypatch) -> None:
    lab = Lab(tmp_path, monkeypatch)
    lab.write("local.txt", "a commit of mine\n", lab.copy)
    lab.commit("mine", lab.copy)
    assert su.check(lab.installation()).state == "ahead"
    lab.publish("main.py", "print('new')\n", "Say new")
    status = su.check(lab.installation())
    assert status.state == "diverged" and "README.md" in status.message
    with pytest.raises(su.UpdateError):
        su.apply(status, tmp_path / "backup")


@needs_git
def test_files_checked_byte_for_byte_are_written_again_when_only_their_line_endings_differ(tmp_path: Path,
                                                                                         monkeypatch) -> None:
    """A clone made with autocrlf before ``.gitattributes`` said ``-text`` holds the package with CRLF; the update
    that brings the rule leaves those files alone (Git sees no change) - the updater writes them again."""
    lab = Lab(tmp_path, monkeypatch)
    old = tmp_path / "old_windows_clone"
    lab.git(tmp_path, "clone", "-q", "-c", "core.autocrlf=true", str(lab.origin), str(old))
    run = old / "utils/sputter_client/run.py"
    assert run.read_bytes() == b"print(1)\r\nprint(2)\r\n"
    lab.publish(".gitattributes", "utils/sputter_client/** -text\n", "Byte for byte")
    status = su.check(su.Installation("git", old))
    result = su.apply(status, tmp_path / "backup")
    assert result.values["rewritten"] == 2  # run.py and MANIFEST.json
    assert run.read_bytes() == b"print(1)\nprint(2)\n" and lab.git(old, "status", "--porcelain") == ""
    run.write_bytes(b"print(1)\r\nprint(3)\r\n")  # a real edit stays
    assert su.restore_byte_exact(old) == 0 and run.read_bytes() == b"print(1)\r\nprint(3)\r\n"


# -- a folder from Download ZIP -----------------------------------------------------------------------------


class FakeGitHub:
    """The answers of api.github.com and raw.githubusercontent.com for one commit of ``files``."""

    def __init__(self, files: dict, sha: str = "c" * 40, base: str = ""):
        self.files, self.sha, self.base, self.asked = dict(files), sha, base, []
        self.damaged: set = set()

    def get(self, url: str) -> bytes:
        self.asked.append(url)
        if url == f"{su.API}/commits/{su.BRANCH}":
            return json.dumps({"sha": self.sha, "commit": {"committer": {"date": "2026-11-25T22:00:00Z"},
                                                           "message": "Newest\n\nbody"}}).encode()
        if url == f"{su.API}/git/trees/{self.sha}?recursive=1":
            tree = [{"path": name, "mode": "100644", "type": "blob", "sha": su.blob_sha(data), "size": len(data)}
                    for name, data in self.files.items()]
            tree += [{"path": "../outside.txt", "mode": "100644", "type": "blob", "sha": "0" * 40, "size": 1},
                     {"path": "docs", "mode": "040000", "type": "tree", "sha": "1" * 40}]
            return json.dumps({"tree": tree, "truncated": False}).encode()
        if url.startswith(f"{su.API}/compare/"):
            return json.dumps({"total_commits": 3, "commits": [
                {"sha": "a" * 40, "commit": {"message": "Older"}}, {"sha": self.sha, "commit": {"message": "Newest"}}]}).encode()
        prefix = f"{su.RAW}/{self.sha}/"
        assert url.startswith(prefix), url
        name = url[len(prefix):].replace("%20", " ")
        return b"broken" if name in self.damaged else self.files[name]


def _zip_folder(root: Path, files: dict) -> Path:
    for name, data in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return root


OLD = {"main.py": b"print('old')\n", "src/gimap/__init__.py": b"", "requirements.txt": b"numpy\n",
       "docs/a b.md": b"same\n"}
NEW = {"main.py": b"print('new')\n", "src/gimap/__init__.py": b"", "requirements.txt": b"numpy\nscipy\n",
       "docs/a b.md": b"same\n", "src/gimap/added.py": b"x = 1\n"}


def test_a_zip_folder_gets_only_the_files_that_differ_and_keeps_what_it_replaces(tmp_path: Path) -> None:
    root = _zip_folder(tmp_path / "gisaxs_gui-main", OLD)
    (root / "my_results.csv").write_text("1,2\n")
    hub = FakeGitHub(NEW)
    status = su.check(su.Installation("folder", root), hub.get)
    assert status.state == "behind"
    assert sorted(f[0] for f in status.files) == ["main.py", "requirements.txt", "src/gimap/added.py"]
    assert status.new_commits == ["ccccccc Newest"]  # no record yet: the newest commit only
    result = su.apply(status, tmp_path / "backup", hub.get)
    assert {name: (root / name).read_bytes() for name in NEW} == NEW
    assert (root / "my_results.csv").read_text() == "1,2\n" and not (tmp_path / "outside.txt").exists()
    assert result.requirements == ["requirements.txt"] and result.changed_files == 3
    kept = Path(result.backup)
    assert (kept / "main.py").read_bytes() == b"print('old')\n" and not (kept / "src/gimap/added.py").exists()
    assert json.loads((root / su.RECORD).read_text())["commit"] == hub.sha
    assert not (root / su.STAGING).exists()
    again = su.check(su.Installation("folder", root), hub.get)
    assert again.state == "up_to_date" and again.current == hub.sha
    assert not any("raw.githubusercontent" in url for url in hub.asked[-2:])


def test_a_zip_folder_holds_files_as_githubs_zip_does_crlf_launchers_and_the_stamped_commit(tmp_path: Path) -> None:
    """GitHub's ZIP applies ``.gitattributes``: ``*.bat`` with CRLF, ``revision.json`` with the commit filled in.
    Such files are not different because of that, and what the update writes looks the same way."""
    attributes = b"*.bat text eol=crlf\nsrc/gimap/shared/revision.json export-subst\n"
    stamp = b'{"commit":"$Format:%H$"}\n'
    old_sha, hub_files = "b" * 40, {".gitattributes": attributes, "Start.bat": b"@echo off\nrem start\n",
                                    "src/gimap/shared/revision.json": stamp, "main.py": b"print('new')\n"}
    root = _zip_folder(tmp_path / "copy", {".gitattributes": attributes, "Start.bat": b"@echo off\r\nrem start\r\n",
                                           "src/gimap/shared/revision.json": b'{"commit":"' + old_sha.encode() + b'"}\n',
                                           "main.py": b"print('old')\n"})
    hub = FakeGitHub(hub_files)
    status = su.check(su.Installation("folder", root), hub.get)
    assert sorted(f[0] for f in status.files) == ["main.py", "src/gimap/shared/revision.json"]
    assert status.current == old_sha and status.n_new == 3  # the stamped commit: what is new since it
    hub.files["Start.bat"] = b"@echo off\nrem started\n"
    status = su.check(su.Installation("folder", root), hub.get)
    su.apply(status, tmp_path / "backup", hub.get)
    assert (root / "Start.bat").read_bytes() == b"@echo off\r\nrem started\r\n"
    assert (root / "src/gimap/shared/revision.json").read_bytes() == b'{"commit":"' + hub.sha.encode() + b'"}\n'
    assert su.check(su.Installation("folder", root), hub.get).state == "up_to_date"


def test_a_zip_folder_updated_before_lists_the_commits_since(tmp_path: Path) -> None:
    root = _zip_folder(tmp_path / "copy", OLD)
    (root / su.RECORD).write_text(json.dumps({"commit": "b" * 40}))
    status = su.check(su.Installation("folder", root), FakeGitHub(NEW).get)
    assert status.n_new == 3 and status.new_commits == ["ccccccc Newest", "aaaaaaa Older"]


def test_a_damaged_download_or_a_stop_changes_nothing(tmp_path: Path) -> None:
    root = _zip_folder(tmp_path / "copy", OLD)
    hub = FakeGitHub(NEW)
    hub.damaged.add("src/gimap/added.py")
    status = su.check(su.Installation("folder", root), hub.get)
    with pytest.raises(su.UpdateError, match="arrived damaged"):
        su.apply(status, tmp_path / "backup", hub.get)
    assert {name: (root / name).read_bytes() for name in OLD} == OLD and not (root / "src/gimap/added.py").exists()
    hub.damaged.clear()
    with pytest.raises(su.UpdateError, match="stopped"):
        su.apply(status, tmp_path / "backup", hub.get, cancelled=lambda: True)
    assert (root / "main.py").read_bytes() == OLD["main.py"] and not (root / su.STAGING).exists()


def test_a_file_that_cannot_be_replaced_puts_the_folder_back(tmp_path: Path, monkeypatch) -> None:
    root = _zip_folder(tmp_path / "copy", OLD)
    hub = FakeGitHub(NEW)
    status = su.check(su.Installation("folder", root), hub.get)
    real, calls = os.replace, []

    def replace(source, target):
        calls.append(target)
        if len(calls) == 3:
            raise PermissionError(13, "in use")
        return real(source, target)

    monkeypatch.setattr(su.os, "replace", replace)
    with pytest.raises(su.UpdateError, match="put back as it was"):
        su.apply(status, tmp_path / "backup", hub.get)
    assert {name: (root / name).read_bytes() for name in OLD} == OLD
    assert not (root / "src/gimap/added.py").exists() and not (root / su.RECORD).exists()


def test_offline_is_said_plainly(monkeypatch) -> None:
    def fail(*_args, **_kwargs):
        raise su.urllib.error.URLError("no route to host")

    monkeypatch.setattr(su.urllib.request, "urlopen", fail)
    with pytest.raises(su.UpdateError) as caught:
        su.http_get("https://api.github.com/x")
    assert caught.value.template.startswith("GitHub could not be reached")


# -- the dialog and the menu --------------------------------------------------------------------------------


def _dialog(actions_kw: dict):
    from PyQt5.QtWidgets import QApplication

    from src.gimap.app.presentation.update_dialog import UpdateActions, UpdateDialog

    QApplication.instance() or QApplication([])
    actions = dict(kind="folder", root="/x/gisaxs_gui-main", version="9.9", check=lambda: None,
                   apply=lambda *a: None, restart=lambda: True, open_releases=lambda: None, python="py")
    actions.update(actions_kw)
    dialog = UpdateDialog(None, UpdateActions(**actions))
    return dialog


def _wait(dialog) -> None:
    from PyQt5.QtWidgets import QApplication

    QApplication.processEvents()
    dialog._runner.wait(10)
    QApplication.processEvents()


def test_the_dialog_lists_what_is_new_and_stops_for_new_packages_before_restarting() -> None:
    status = SimpleNamespace(state="behind", latest="c" * 40, latest_date="2026-11-25T22:00:00Z", files=[1, 2],
                             download_bytes=2_500_000, new_commits=["ccccccc Newest"], n_new=1, message="", values={},
                             upstream="")
    restarts = []
    result = SimpleNamespace(new="c" * 40, changed_files=2, backup="/b/kept", requirements=["requirements.txt"])
    dialog = _dialog(dict(check=lambda: status, apply=lambda s, progress, cancelled: (progress("Downloading {done} of "
                          "{total} files…", dict(done=1, total=2)), result)[1], restart=lambda: restarts.append(1)))
    try:
        _wait(dialog)
        assert "2 files differ" in dialog.state.text() and "2.5 MB" in dialog.state.text()
        assert dialog.update_button.isEnabled() and dialog.commits.toPlainText() == "ccccccc Newest"
        dialog.update_button.click()
        _wait(dialog)
        assert "Updated to ccccccc: 2 files changed." in dialog.state.text() and "/b/kept" in dialog.state.text()
        assert dialog.commits.toPlainText() == '"py" -m pip install -r requirements.txt'
        assert restarts == [] and dialog.update_button.text() == "Restart"
        dialog.update_button.click()
        assert restarts == [1]
    finally:
        dialog.busy = False
        dialog.close()


def test_the_dialog_says_what_went_wrong_in_the_interface_language() -> None:
    from src.gimap.app.presentation.i18n import translator

    def offline():
        raise su.UpdateError("GitHub could not be reached ({error}). Check the internet connection and try again.",
                             error="timed out")

    language = translator().language
    translator().language = "zh"
    try:
        dialog = _dialog(dict(check=offline))
        _wait(dialog)
        assert dialog.state.text() == "无法连接 GitHub（timed out）。请检查网络连接后重试。"
        assert not dialog.update_button.isEnabled()
        dialog.close()
    finally:
        translator().language = language


def test_restart_starts_gimap_again_only_when_the_window_closed(monkeypatch) -> None:
    from PyQt5.QtCore import QProcess
    from PyQt5.QtWidgets import QApplication

    from src.gimap.app.help_commands import restart_window

    app = QApplication.instance() or QApplication([])
    started, quits = [], []
    monkeypatch.setattr(QProcess, "startDetached", lambda *args: started.append(args) or True)
    monkeypatch.setattr(app, "quit", lambda: quits.append(1))
    command = [sys.executable, "/x/main.py", "--restarted"]
    assert restart_window(SimpleNamespace(close=lambda: False), command) is False
    app.aboutToQuit.emit()
    assert started == [] and quits == []
    assert restart_window(SimpleNamespace(close=lambda: True), command) is True
    assert quits == [1]
    app.aboutToQuit.emit()
    app.aboutToQuit.emit()  # connected once: one new GIMaP
    assert started == [(sys.executable, ["/x/main.py", "--restarted"], "/x")]


def test_check_for_updates_opens_the_dialog_for_a_source_folder_and_the_releases_page_otherwise(monkeypatch,
                                                                                                tmp_path: Path) -> None:
    from PyQt5.QtGui import QDesktopServices
    from PyQt5.QtWidgets import QApplication, QMainWindow

    from src.gimap.app import help_commands
    from src.gimap.app.presentation.menu_bar import RELEASES_URL

    QApplication.instance() or QApplication([])
    urls, shown = [], []
    monkeypatch.setattr(QDesktopServices, "openUrl", lambda url: urls.append(url.toString()) or True)
    monkeypatch.setattr(help_commands, "show_update_dialog", lambda parent, actions: shown.append(actions))
    window = QMainWindow()
    window.app_context = SimpleNamespace(data_dir=str(tmp_path))
    commands = help_commands.HelpCommands(SimpleNamespace(window=window))
    commands.check_updates(su.Installation("installed", reason="the portable version"))
    assert urls == [RELEASES_URL] and shown == []
    root = _source(tmp_path / "copy")
    commands.check_updates(su.Installation("folder", root))
    assert len(shown) == 1 and shown[0].kind == "folder" and shown[0].root == str(root)
    window.deleteLater()
