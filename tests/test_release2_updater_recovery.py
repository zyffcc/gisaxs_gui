"""``tools/update_source.ps1`` when an update cannot go through, or did not the last time.

The scenarios of ``test_release2_updater_scenarios.py`` end well.  Here git is stopped on the way -
another program holds a file open, a window was closed in the middle - and the checkout must stay
usable: the version it had, whole, with the person's modifications where they were, and a next run
that finishes the update.  As there, everything happens on repositories in a temporary folder.
"""

from __future__ import annotations

import contextlib
import ctypes
import subprocess

import pytest

from tests.test_release2_updater_scenarios import (
    POWERSHELL,
    Lab,
    _app,
    _personal_files_are_untouched,
    _report,
    pytestmark,  # noqa: F401 - the same condition: Windows, with powershell and git
)


@pytest.fixture
def lab(tmp_path):
    return Lab(tmp_path)


@contextlib.contextmanager
def _held_open(path):
    """Another program has the file open: others may read it, nobody can replace or delete it."""
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateFileW.argtypes = [
        wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID,
        wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    kernel32.CreateFileW.restype = wintypes.HANDLE
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    generic_read, share_read, open_existing = 0x80000000, 0x1, 3
    handle = kernel32.CreateFileW(str(path), generic_read, share_read, None, open_existing, 0, None)
    assert handle not in (None, wintypes.HANDLE(-1).value), f"could not open {path}"
    try:
        yield
    finally:
        kernel32.CloseHandle(handle)


def _tracked_status(lab, checkout) -> list[str]:
    """``git status`` of the tracked files: what is modified, and what of it is staged."""
    lines = lab.git(checkout, "status", "--porcelain").splitlines()
    return [line.strip() for line in lines if not line.startswith("??")]


def test_a_file_another_program_holds_open_leaves_nothing_half_way(lab):
    lab.publish(more={"data/table.bin": "one\n"})
    other = lab.computer()
    old = lab.head(other)
    lab.write(other, {"docs/manual.md": "# Manual\nstaged line\n"})
    lab.git(other, "add", "docs/manual.md")
    lab.write(other, {"src/app.py": _app("two", "# my note\n")})
    before = _tracked_status(lab, other)
    assert len(before) == 2
    new = lab.rewrite({"data/table.bin": "two\n", "zz_new.md": "new\n"})

    with _held_open(other / "data" / "table.bin"):
        done = lab.update(other)

    assert done.returncode == 1, _report(done)
    assert "NOT UPDATED: git stopped on the way." in done.stdout
    assert "data/table.bin" in done.stdout                  # git's own words name the file
    assert "Nothing is left half way" in done.stdout
    assert "Your modifications are in the files, as before." in done.stdout
    # git had replaced other files before it stopped at this one: they are the old ones again ...
    assert lab.head(other) == old
    assert (other / "requirements.txt").read_text("utf-8") == "numpy\n"
    assert (other / "docs" / "case.json").exists()
    assert (other / "data" / "table.bin").read_text("utf-8") == "one\n"
    # ... and the modifications are back in the files, staged where they were staged.
    assert (other / "src" / "app.py").read_text("utf-8") == _app("two", "# my note\n")
    assert (other / "docs" / "manual.md").read_text("utf-8") == "# Manual\nstaged line\n"
    assert _tracked_status(lab, other) == before
    assert lab.stashes(other) == []
    (backup,) = lab.backups(other)
    assert lab.git(other, "rev-parse", backup) == old
    _personal_files_are_untouched(other)

    done = lab.update(other)        # the other program was closed

    assert done.returncode == 0, _report(done)
    assert lab.head(other) == new
    assert f"already kept as branch {backup}" in done.stdout and lab.backups(other) == [backup]
    assert (other / "data" / "table.bin").read_text("utf-8") == "two\n"
    assert (other / "src" / "app.py").read_text("utf-8") == _app("three", "# my note\n")
    assert (other / "docs" / "manual.md").read_text("utf-8") == "# Manual\nstaged line\n"
    assert lab.modified(other) == ["docs/manual.md", "src/app.py"]
    assert len(lab.stashes(other)) == 1
    _personal_files_are_untouched(other)


def test_moving_forward_leaves_nothing_half_way_either(lab):
    lab.publish(more={"data/table.bin": "one\n"})
    other = lab.computer()
    old = lab.head(other)
    lab.write(other, {"docs/manual.md": "# Manual\nmy line\n"})    # the update leaves this file
    new = lab.commit(lab.author, "third", {
        "aa_new.md": "new\n", "data/table.bin": "two\n", "src/app.py": _app("three")})
    lab.git(lab.author, "push", "--quiet", "origin", "HEAD:main")

    with _held_open(other / "data" / "table.bin"):
        done = lab.update(other)

    assert done.returncode == 1, _report(done)
    assert "NOT UPDATED: git stopped on the way." in done.stdout
    assert "Nothing is left half way" in done.stdout
    assert lab.head(other) == old
    assert (other / "src" / "app.py").read_text("utf-8") == _app("two")
    assert (other / "docs" / "manual.md").read_text("utf-8") == "# Manual\nmy line\n"
    assert lab.modified(other) == ["docs/manual.md"]
    assert lab.backups(other) == [] and lab.stashes(other) == []

    done = lab.update(other)

    assert done.returncode == 0, _report(done)
    assert lab.head(other) == new
    assert (other / "src" / "app.py").read_text("utf-8") == _app("three")
    assert (other / "docs" / "manual.md").read_text("utf-8") == "# Manual\nmy line\n"
    assert lab.modified(other) == ["docs/manual.md"]
    _personal_files_are_untouched(other)


def test_an_update_that_was_cut_off_is_finished_by_the_next_run(lab):
    """A closed window leaves new files beside git's record of the old ones: not modifications."""
    lab.publish()
    other = lab.computer()
    lab.write(other, {"docs/manual.md": "# Manual\nmy line\n"})
    new = lab.rewrite()
    lab.git(other, "fetch", "--quiet", "origin")
    lab.write(other, {"src/app.py": _app("three"), "requirements.txt": "numpy\nscipy\n"})
    (other / "docs" / "case.json").unlink()
    assert len(lab.modified(other)) == 4

    done = lab.update(other)

    assert done.returncode == 0, _report(done)
    assert "3 of them already hold what the new version holds" in done.stdout
    assert "put your modifications of 1 tracked file(s) aside" in done.stdout
    assert lab.head(other) == new
    assert lab.modified(other) == ["docs/manual.md"]
    assert (other / "docs" / "manual.md").read_text("utf-8") == "# Manual\nmy line\n"
    _personal_files_are_untouched(other)


def test_what_you_staged_is_kept_when_the_file_itself_is_the_new_version(lab):
    lab.publish()
    other = lab.computer()
    lab.write(other, {"docs/manual.md": "# Manual\nstaged line\n"})
    lab.git(other, "add", "docs/manual.md")
    lab.write(other, {"docs/manual.md": "# Manual\n"})      # in the folder: the committed text
    new = lab.rewrite()

    done = lab.update(other)

    assert done.returncode == 0, _report(done)
    assert lab.head(other) == new
    assert len(lab.stashes(other)) == 1
    assert lab.git(other, "show", "stash@{0}^2:docs/manual.md") == "# Manual\nstaged line"


def test_the_error_stream_may_be_redirected_inside_powershell(lab):
    """Windows PowerShell then makes an error of every line git writes while it downloads."""
    lab.publish()
    other = lab.computer()
    new = lab.rewrite()

    done = subprocess.run(
        [POWERSHELL, "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command",
         r"& .\tools\update_source.ps1 *> ..\update.log; exit $LASTEXITCODE"],
        cwd=other, env=lab.env, capture_output=True, timeout=300,
        encoding="utf-8", errors="replace")

    assert done.returncode == 0, _report(done)
    assert lab.head(other) == new
    assert "UPDATED: main is now at" in (lab.root / "update.log").read_text("utf-16")
    _personal_files_are_untouched(other)


def test_a_misspelled_switch_updates_nothing(lab):
    """``--dry-run`` must not be taken for "no switch": that would be a real update."""
    lab.publish()
    other = lab.computer()
    old = lab.head(other)
    lab.rewrite()

    done = lab.update(other, "--dry-run")

    assert done.returncode != 0, _report(done)
    assert lab.head(other) == old and lab.backups(other) == []
    assert lab.git(other, "rev-parse", "origin/main") == old       # it did not even download
