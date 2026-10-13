"""A source checkout can still be updated after the history of origin was rewritten.

``tools/update_source.ps1`` is what another computer runs when ``git pull`` no longer works: it
keeps the state the checkout has (a ``backup/...`` branch, a stash for modified tracked files),
moves the branch to ``origin/<branch>`` and puts the modifications back.  It never changes a file
git does not track or ignores - settings, data, notes - and it leaves the files of 1D Predict byte
for byte as the repository has them.

The scenarios run for real, on repositories made in a temporary folder (git never writes to this
one): a bare "origin", the author's clone, which rewrites the history and force-pushes, and the
clone of another computer, where the script runs.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "update_source.ps1"
POWERSHELL = shutil.which("powershell")

pytestmark = pytest.mark.skipif(
    sys.platform != "win32" or not POWERSHELL or not shutil.which("git"),
    reason="the updater is a Windows PowerShell script that drives git",
)

BACKUP = re.compile(r"backup/\d{8}-\d{6}-[0-9a-f]{7,}")
IGNORE = "config/user_settings.json\nnotes/\n*.h5\n"
# On the other computer only: ignored by git, or never told to git (``_handoff``).
PERSONAL = {"config/user_settings.json": '{"language": "zh"}\n', "notes/todo.md": "my notes\n",
            "_handoff/report.md": "a report\n"}


def _long(path: Path) -> str:
    """The path as Windows opens it even when it has 260 characters or more."""
    return "\\\\?\\" + os.path.normpath(str(path))


def _app(version: str, note: str = "") -> str:
    """A source file with two places far enough apart for git to merge changes of both."""
    return (f'VERSION = "{version}"\n\n\ndef main():\n    print(VERSION)\n    return 0\n\n\n'
            f'if __name__ == "__main__":\n    raise SystemExit(main())\n{note}')


class Lab:
    """A bare origin, the author's clone and the checkouts of other computers."""

    def __init__(self, root: Path) -> None:
        self.root = root
        config = root / "gitconfig"
        # No name and no e-mail address: a computer that only ever pulled has none either.
        config.write_text("[init]\n\tdefaultBranch = main\n[core]\n\tautocrlf = false\n", "ascii")
        self.env = {k: v for k, v in os.environ.items() if not k.upper().startswith("GIT_")}
        self.env.update(
            GIT_CONFIG_GLOBAL=str(config), GIT_CONFIG_NOSYSTEM="1", GIT_TERMINAL_PROMPT="0")
        self.origin = root / "origin.git"
        self.author = root / "author"
        self.git(root, "init", "--quiet", "--bare", str(self.origin))
        self.git(root, "clone", "--quiet", str(self.origin), str(self.author))

    def git(self, folder: Path, *arguments: str, check: bool = True) -> str:
        done = subprocess.run(
            ["git", "-c", "user.name=Author", "-c", "user.email=author@example.org",
             "-c", "core.quotepath=false", *arguments],
            cwd=folder, env=self.env, capture_output=True, timeout=120,
            encoding="utf-8", errors="replace")
        if check and done.returncode:
            raise AssertionError(f"git {' '.join(arguments)}\n{done.stdout}\n{done.stderr}")
        return done.stdout.strip()

    def write(self, folder: Path, files: dict[str, str]) -> None:
        for name, text in files.items():
            path = _long(folder / name)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as handle:
                handle.write(text.encode("utf-8"))

    def commit(self, folder: Path, message: str, files: dict[str, str]) -> str:
        self.write(folder, files)
        self.git(folder, "add", "--force", "--", *files)
        self.git(folder, "commit", "--quiet", "--message", message)
        return self.git(folder, "rev-parse", "HEAD")

    def publish(self, *, script: bool = True, more: dict[str, str] | None = None) -> None:
        """The history the other computers pulled: its second commit names a private sample."""
        files = {".gitignore": IGNORE, "src/app.py": _app("one"), "docs/manual.md": "# Manual\n",
                 "requirements.txt": "numpy\n", **(more or {})}
        if script:
            files["tools/update_source.ps1"] = SCRIPT.read_text("ascii")
        self.commit(self.author, "first", files)
        self.commit(self.author, "add a case", {"docs/case.json": '{"sample": "private"}\n'})
        self.commit(self.author, "second", {"src/app.py": _app("two")})
        self.git(self.author, "push", "--quiet", "origin", "HEAD:main")

    def computer(self, name: str = "other", *, autocrlf: bool = False) -> Path:
        """A checkout that was only ever pulled, with the files a person keeps in it.

        ``autocrlf`` is the default of Git for Windows: text files get CRLF line endings on disk.
        """
        checkout = self.root / name
        self.git(self.root, "clone", "--quiet", "--config",
                 f"core.autocrlf={str(autocrlf).lower()}", str(self.origin), str(checkout))
        # Made some time ago: git compares the content of a file as young as its own record of it,
        # and trusts that record for an older one - which is how a converted file goes unnoticed.
        long_ago = time.time() - 3600
        for tracked in self.git(checkout, "ls-files").splitlines():
            os.utime(checkout / tracked, (long_ago, long_ago))
        self.git(checkout, "update-index", "-q", "--refresh")
        self.write(checkout, PERSONAL)
        return checkout

    def rewrite(self, more: dict[str, str] | None = None) -> str:
        """The author removes the commit with the private sample and force-pushes."""
        self.git(self.author, "reset", "--quiet", "--hard", "HEAD~2")
        self.commit(self.author, "second", {"src/app.py": _app("two")})
        new = self.commit(self.author, "third", {
            "src/app.py": _app("three"), "requirements.txt": "numpy\nscipy\n",
            "model.h5": "weights\n", **(more or {})})
        self.git(self.author, "push", "--quiet", "--force", "origin", "HEAD:main")
        return new

    def advance(self) -> str:
        """The author adds a commit: nothing is rewritten."""
        files = {"src/app.py": _app("three"), "docs/new.md": "new\n"}
        new = self.commit(self.author, "third", files)
        self.git(self.author, "push", "--quiet", "origin", "HEAD:main")
        return new

    def update(self, checkout: Path, *arguments: str, path: str | None = None):
        env = dict(self.env)
        if path is not None:
            env["PATH"] = path
        # Started from another folder: the script updates the checkout it lives in.
        return subprocess.run(
            [POWERSHELL, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
             str(checkout / "tools" / "update_source.ps1"), *arguments],
            cwd=self.root, env=env, capture_output=True, timeout=300,
            encoding="utf-8", errors="replace")

    def head(self, checkout: Path) -> str:
        return self.git(checkout, "rev-parse", "HEAD")

    def backups(self, checkout: Path) -> list[str]:
        return self.git(checkout, "branch", "--list", "backup/*", "--format=%(refname)").replace(
            "refs/heads/", "").split()

    def stashes(self, checkout: Path) -> list[str]:
        return self.git(checkout, "stash", "list").splitlines()

    def modified(self, checkout: Path) -> list[str]:
        """The tracked files that differ from the commit the checkout is at."""
        return self.git(checkout, "diff", "--name-only", "HEAD").splitlines()


def _personal_files_are_untouched(checkout: Path) -> None:
    for name, text in PERSONAL.items():
        assert (checkout / name).read_bytes() == text.encode("utf-8"), name


def _report(done) -> str:
    return f"exit code {done.returncode}\n{done.stdout}\n{done.stderr}"


@pytest.fixture
def lab(tmp_path):
    return Lab(tmp_path)


def test_the_script_is_plain_ascii():
    """Windows PowerShell 5.1 reads a script without a byte-order mark in the system code page."""
    assert SCRIPT.read_bytes().isascii()
    assert len(SCRIPT.read_text("ascii").splitlines()) <= 600


def test_a_rewritten_history_is_followed_and_the_old_state_is_kept(lab):
    lab.publish()
    other = lab.computer()
    old = lab.head(other)
    new = lab.rewrite()

    done = lab.update(other)

    assert done.returncode == 0, _report(done)
    assert lab.head(other) == new
    assert lab.git(other, "symbolic-ref", "--short", "HEAD") == "main"
    assert (other / "src" / "app.py").read_text("utf-8") == _app("three")
    assert not (other / "docs" / "case.json").exists()     # the new history does not have it
    assert lab.modified(other) == []
    # Everything local was kept first: the branch is named in the report, with the way back.
    (backup,) = lab.backups(other)
    assert BACKUP.fullmatch(backup) and backup.endswith(old[:7])
    assert lab.git(other, "rev-parse", backup) == old
    assert "was rewritten" in done.stdout and "UPDATED: main is now at" in done.stdout
    assert f"git reset --hard {backup}" in done.stdout
    assert f"git branch -D {backup}" in done.stdout
    assert "python -m pip install -r requirements.txt" in done.stdout   # the list changed
    assert lab.stashes(other) == []
    _personal_files_are_untouched(other)

    again = lab.update(other)

    assert again.returncode == 0, _report(again)
    assert "ALREADY UP TO DATE" in again.stdout
    assert lab.head(other) == new and lab.backups(other) == [backup]


def test_a_modification_that_fits_is_put_back_and_the_way_back_works(lab):
    lab.publish()
    other = lab.computer()
    old = lab.head(other)
    mine = _app("two", "# my note\n")
    lab.write(other, {"src/app.py": mine})
    new = lab.rewrite()

    done = lab.update(other)

    assert done.returncode == 0, _report(done)
    assert lab.head(other) == new
    # The new version, with the local modification on top of it.
    assert (other / "src" / "app.py").read_text("utf-8") == _app("three", "# my note\n")
    assert lab.modified(other) == ["src/app.py"]
    assert len(lab.stashes(other)) == 1 and "update_source" in lab.stashes(other)[0]
    assert "a copy stays in the stash" in done.stdout
    _personal_files_are_untouched(other)

    # The two commands the report gives bring back exactly what was there.
    (backup,) = lab.backups(other)
    assert f"git reset --hard {backup}" in done.stdout and "git stash pop" in done.stdout
    lab.git(other, "reset", "--quiet", "--hard", backup)
    lab.git(other, "stash", "pop", "--quiet")
    assert lab.head(other) == old
    assert (other / "src" / "app.py").read_text("utf-8") == mine
    assert lab.stashes(other) == []


def test_a_modification_that_does_not_fit_waits_in_the_stash(lab):
    lab.publish()
    other = lab.computer()
    lab.write(other, {"src/app.py": _app("mine"), "docs/manual.md": "# Manual\nmy line\n"})
    new = lab.rewrite()

    done = lab.update(other)

    assert done.returncode == 2, _report(done)
    assert "YOUR MODIFICATIONS NEED YOU" in done.stdout and "src/app.py" in done.stdout
    # The new version is whole (no conflict markers, nothing half merged) ...
    assert lab.head(other) == new
    assert (other / "src" / "app.py").read_text("utf-8") == _app("three")
    assert lab.modified(other) == []
    # ... and nothing is lost: the stash holds both modifications, the branch holds the commit.
    assert len(lab.backups(other)) == 1
    assert len(lab.stashes(other)) == 1
    kept = lab.git(other, "stash", "show", "--patch", "stash@{0}")
    assert '+VERSION = "mine"' in kept and "+my line" in kept
    _personal_files_are_untouched(other)


def test_a_file_of_yours_in_the_way_stops_the_update_before_anything_changes(lab):
    lab.publish()
    other = lab.computer()
    old = lab.head(other)
    # The new history tracks ``model.h5``; here it is a file git ignores, with other content.
    lab.write(other, {"model.h5": "my own weights\n", "src/app.py": _app("two", "# my note\n")})
    new = lab.rewrite()

    done = lab.update(other)

    assert done.returncode == 1, _report(done)
    assert "NOT UPDATED: nothing was changed." in done.stdout and "model.h5" in done.stdout
    assert lab.head(other) == old
    assert lab.backups(other) == [] and lab.stashes(other) == []
    assert (other / "model.h5").read_text("utf-8") == "my own weights\n"
    assert (other / "src" / "app.py").read_text("utf-8") == _app("two", "# my note\n")

    # The same file with the content of the new version is not in the way.
    lab.write(other, {"model.h5": "weights\n"})

    done = lab.update(other)

    assert done.returncode == 0, _report(done)
    assert lab.head(other) == new
    assert (other / "model.h5").read_text("utf-8") == "weights\n"
    assert lab.modified(other) == ["src/app.py"]
    _personal_files_are_untouched(other)


def test_a_file_in_the_way_is_seen_at_a_path_too_long_for_windows_powershell(lab):
    """``Test-Path`` does not see 260 characters or more: the file would be overwritten unseen."""
    lab.publish()
    other = lab.computer("other-" + "x" * 40)
    room = 235 - len(str(lab.author))       # the author's own copy has to stay a short path
    if room < 60:
        pytest.skip("the temporary folder is too deep for this scenario")
    weights = "models/" + "x" * (room - 24) + "/final.weights.h5"
    assert len(str(other / weights)) >= 260 > len(str(lab.author / weights))
    old = lab.head(other)
    lab.write(other, {weights: "my own weights\n"})
    try:
        lab.rewrite({weights: "weights\n"})

        done = lab.update(other)

        assert done.returncode == 1, _report(done)
        assert "NOT UPDATED: nothing was changed." in done.stdout
        assert "final.weights.h5" in done.stdout
        assert lab.head(other) == old and lab.backups(other) == []
        with open(_long(other / weights), "rb") as handle:
            assert handle.read() == b"my own weights\n"
    finally:
        shutil.rmtree(_long(other / "models"), ignore_errors=True)      # for pytest's own cleanup


def test_line_endings_an_older_checkout_converted_are_put_right(lab):
    """1D Predict checks its model files against SHA-256 sums; ``.gitattributes`` says ``-text``.

    A checkout made before that rule holds those files with CRLF line endings.  git writes a file
    again only when its content changed, and it trusts its own record of the others: after an
    update they are still converted, ``git status`` is clean, and 1D Predict refuses its files.
    """
    protocol, table = '{\n  "model": "blue"\n}\n', "q,intensity\n0.1,2.0\n"
    lab.publish(more={
        "models/protocol.json": protocol, "models/table.csv": table, "models/notes.md": "one\n"})
    other = lab.computer(autocrlf=True)
    laptop = lab.computer("laptop", autocrlf=True)
    converted = protocol.replace("\n", "\r\n").encode()
    assert (other / "models" / "protocol.json").read_bytes() == converted
    new = lab.rewrite({".gitattributes": "models/** -text\n"})

    done = lab.update(other)

    assert done.returncode == 0, _report(done)
    assert lab.head(other) == new and "byte for byte" in done.stdout
    assert (other / "models" / "protocol.json").read_bytes() == protocol.encode()
    assert (other / "models" / "table.csv").read_bytes() == table.encode()
    assert lab.modified(other) == []
    _personal_files_are_untouched(other)

    # The two plain commands of a first update leave the files converted, and git sees nothing.
    lab.git(laptop, "fetch", "--quiet", "origin")
    lab.git(laptop, "reset", "--quiet", "--hard", "origin/main")
    assert (laptop / "models" / "protocol.json").read_bytes() == converted
    assert lab.modified(laptop) == []
    # Once something touches such a file, git shows it as modified - like a file of yours.
    os.utime(laptop / "models" / "table.csv")
    mine = "one\r\nmine\r\n"
    lab.write(laptop, {"models/notes.md": mine})
    assert lab.modified(laptop) == ["models/notes.md", "models/table.csv"]

    done = lab.update(laptop)

    assert done.returncode == 0, _report(done)
    assert "ALREADY UP TO DATE: only line endings were corrected." in done.stdout
    assert (laptop / "models" / "protocol.json").read_bytes() == protocol.encode()
    assert (laptop / "models" / "table.csv").read_bytes() == table.encode()
    # A file with a modification of yours is yours, line endings included.
    assert (laptop / "models" / "notes.md").read_bytes() == mine.encode()
    assert lab.modified(laptop) == ["models/notes.md"]

    again = lab.update(laptop)

    assert again.returncode == 0, _report(again)
    assert "ALREADY UP TO DATE: nothing was changed." in again.stdout


def test_a_commit_of_your_own_stays_on_the_backup_branch(lab):
    lab.publish()
    other = lab.computer()
    own = lab.commit(other, "my own fix", {"src/mine.py": "MINE = 1\n"})
    new = lab.rewrite()

    done = lab.update(other)

    assert done.returncode == 0, _report(done)
    assert lab.head(other) == new
    assert not (other / "src" / "mine.py").exists()
    (backup,) = lab.backups(other)
    assert lab.git(other, "rev-parse", backup) == own
    # The report tells the rewritten commits from the person's own one, and how to bring it over.
    assert "2 commit(s) you have are no longer part of it" in done.stdout
    assert "1 commit(s) of your own" in done.stdout and "my own fix" in done.stdout
    assert f"git cherry-pick {own[:7]}" in done.stdout
    lab.git(other, "cherry-pick", own[:7])
    assert (other / "src" / "mine.py").read_text("utf-8") == "MINE = 1\n"
    _personal_files_are_untouched(other)


def test_without_a_record_of_the_rewrite_no_commit_is_offered_for_bringing_over(lab):
    """Bringing a replaced commit over would put what was removed back into the new history."""
    lab.publish()
    other = lab.computer()
    old = lab.head(other)
    new = lab.rewrite()
    # Fetched some time ago, and git has dropped its log of where origin/main was before.
    lab.git(other, "fetch", "--quiet", "origin")
    (other / ".git" / "logs" / "refs" / "remotes" / "origin" / "main").unlink()

    done = lab.update(other)

    assert done.returncode == 0, _report(done)
    assert lab.head(other) == new
    (backup,) = lab.backups(other)
    assert lab.git(other, "rev-parse", backup) == old
    assert "have gone different ways" in done.stdout and "was rewritten" not in done.stdout
    assert "your own, or ones origin has replaced" in done.stdout
    assert "git cherry-pick <its number>" in done.stdout
    assert f"git cherry-pick {old[:7]}" not in done.stdout


def test_a_commit_of_your_own_on_top_of_origin_is_left_alone(lab):
    lab.publish()
    other = lab.computer()
    own = lab.commit(other, "my own fix", {"src/mine.py": "MINE = 1\n"})

    done = lab.update(other)

    assert done.returncode == 0, _report(done)
    assert "ALREADY UP TO DATE" in done.stdout and "my own fix" in done.stdout
    assert lab.head(other) == own and lab.backups(other) == []


def test_new_commits_on_origin_move_the_branch_forward(lab):
    lab.publish()
    other = lab.computer()
    old = lab.head(other)
    lab.write(other, {"docs/manual.md": "# Manual\nmy line\n"})    # a file the update leaves alone
    new = lab.advance()

    done = lab.update(other)

    assert done.returncode == 0, _report(done)
    assert "only has to move forward" in done.stdout and "git merge --ff-only" in done.stdout
    assert lab.head(other) == new
    assert (other / "docs" / "new.md").exists()
    # A fast-forward needs no backup branch and no stash: the old commit is part of the history.
    assert lab.backups(other) == [] and lab.stashes(other) == []
    assert f"git reset --hard {old[:7]}" in done.stdout
    assert lab.modified(other) == ["docs/manual.md"]
    assert (other / "docs" / "manual.md").read_text("utf-8") == "# Manual\nmy line\n"
    _personal_files_are_untouched(other)


def test_moving_forward_carries_a_modification_of_a_file_the_update_changes(lab):
    lab.publish()
    other = lab.computer()
    lab.write(other, {"src/app.py": _app("two", "# my note\n")})
    new = lab.advance()     # changes src/app.py too: git alone would refuse to move

    done = lab.update(other)

    assert done.returncode == 0, _report(done)
    assert lab.head(other) == new
    assert (other / "src" / "app.py").read_text("utf-8") == _app("three", "# my note\n")
    assert lab.backups(other) == [] and len(lab.stashes(other)) == 1
    _personal_files_are_untouched(other)


def test_a_dry_run_reports_and_changes_nothing(lab):
    lab.publish()
    other = lab.computer()
    old = lab.head(other)
    lab.write(other, {"src/app.py": _app("two", "# my note\n")})
    new = lab.rewrite()

    for switch in ("-DryRun", "-WhatIf"):       # the second time, origin was fetched already
        done = lab.update(other, switch)

        assert done.returncode == 0, _report(done)
        assert "DRY RUN: nothing was changed." in done.stdout
        assert "was rewritten" in done.stdout and "git reset --hard" in done.stdout
        assert "git stash" in done.stdout and BACKUP.search(done.stdout)
        assert lab.head(other) == old
        assert lab.backups(other) == [] and lab.stashes(other) == []
        assert lab.modified(other) == ["src/app.py"]
        assert (other / "src" / "app.py").read_text("utf-8") == _app("two", "# my note\n")
        # It did look: what origin has now is known, the branch was not moved to it.
        assert lab.git(other, "rev-parse", "origin/main") == new
        _personal_files_are_untouched(other)


def test_a_folder_that_is_not_a_checkout_is_refused(lab):
    folder = lab.root / "portable"
    (folder / "tools").mkdir(parents=True)
    shutil.copy2(SCRIPT, folder / "tools" / SCRIPT.name)

    done = lab.update(folder)

    assert done.returncode == 1, _report(done)
    assert "NOT UPDATED" in done.stdout and "not a git checkout" in done.stdout


def test_a_computer_without_git_is_told_so(lab):
    lab.publish()
    other = lab.computer()
    windows = Path(os.environ.get("SystemRoot", r"C:\Windows"))

    done = lab.update(other, path=f"{windows / 'System32'};{windows}")

    assert done.returncode == 1, _report(done)
    assert "NOT UPDATED" in done.stdout and "git was not found" in done.stdout


def test_an_unfinished_merge_is_refused(lab):
    """``git pull`` on a rewritten history starts a merge, and often leaves it unfinished."""
    lab.publish()
    other = lab.computer()
    old = lab.head(other)
    lab.rewrite()
    lab.git(other, "fetch", "--quiet", "origin")
    lab.git(other, "merge", "--no-commit", "--no-ff", "origin/main", check=False)
    assert (other / ".git" / "MERGE_HEAD").exists()

    done = lab.update(other)

    assert done.returncode == 1, _report(done)
    assert "A merge" in done.stdout and "is in progress" in done.stdout
    assert "git merge --abort" in done.stdout
    assert lab.head(other) == old and lab.backups(other) == []
    assert (other / ".git" / "MERGE_HEAD").exists()


def test_a_pull_that_merged_the_old_history_back_is_undone(lab):
    lab.publish()
    other = lab.computer()
    # A rewrite git can merge without a conflict: the old history comes back with the merge.
    lab.git(lab.author, "reset", "--quiet", "--hard", "HEAD~2")
    lab.commit(lab.author, "second", {"src/app.py": _app("two")})
    new = lab.commit(lab.author, "third", {"docs/new.md": "new\n"})
    lab.git(lab.author, "push", "--quiet", "--force", "origin", "HEAD:main")
    lab.git(other, "pull", "--quiet", "--no-rebase", "--no-edit", "origin", "main")
    merged = lab.head(other)
    assert (other / "docs" / "case.json").exists()

    done = lab.update(other)

    assert done.returncode == 0, _report(done)
    assert "was rewritten" in done.stdout and "merge(s) made here" in done.stdout
    assert lab.head(other) == new
    assert not (other / "docs" / "case.json").exists()
    (backup,) = lab.backups(other)
    assert lab.git(other, "rev-parse", backup) == merged
    assert "git cherry-pick" not in done.stdout     # a merge is not a commit to bring over
    _personal_files_are_untouched(other)


def test_the_first_update_of_a_checkout_that_does_not_have_the_script_yet(lab):
    """Fetch, take the script alone out of origin, run it: the old state is kept as always."""
    lab.publish(script=False)
    other = lab.computer()
    old = lab.head(other)
    new = lab.rewrite({"tools/update_source.ps1": SCRIPT.read_text("ascii")})
    lab.git(other, "fetch", "--quiet", "origin")
    lab.git(other, "checkout", "origin/main", "--", "tools/update_source.ps1")

    done = lab.update(other)

    assert done.returncode == 0, _report(done)
    assert lab.head(other) == new
    (backup,) = lab.backups(other)
    assert lab.git(other, "rev-parse", backup) == old
    # The script that was fetched ahead is the one of the new version: nothing to put aside.
    assert lab.stashes(other) == [] and lab.modified(other) == []
    _personal_files_are_untouched(other)


def test_names_that_are_not_ascii(lab):
    """git writes names in UTF-8; the script must find the same files on disk."""
    manual = "docs/\u8bf4\u660e.md"
    weights = "\u6a21\u578b.h5"
    lab.publish(more={manual: "# \u8bf4\u660e\n\n\n\n\u7b2c\u4e00\u7248\n"})
    other = lab.computer("\u53e6\u4e00\u53f0 \u7535\u8111")
    lab.write(other, {
        manual: "# \u8bf4\u660e\n\u6211\u7684\u7b14\u8bb0\n\n\n\u7b2c\u4e00\u7248\n",
        weights: "my own weights\n"})
    new = lab.rewrite({manual: "# \u8bf4\u660e\n\n\n\n\u7b2c\u4e8c\u7248\n", weights: "weights\n"})

    refused = lab.update(other)

    assert refused.returncode == 1, _report(refused)
    assert weights in refused.stdout
    assert (other / weights).read_text("utf-8") == "my own weights\n"
    (other / weights).unlink()

    done = lab.update(other)

    assert done.returncode == 0, _report(done)
    assert lab.head(other) == new
    assert (other / manual).read_text("utf-8") == (
        "# \u8bf4\u660e\n\u6211\u7684\u7b14\u8bb0\n\n\n\u7b2c\u4e8c\u7248\n")
    assert (other / weights).read_text("utf-8") == "weights\n"
    _personal_files_are_untouched(other)
