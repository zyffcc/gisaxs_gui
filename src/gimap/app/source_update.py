"""Update a source copy of GIMaP to the newest version of its branch on GitHub (Help ▸ Check for Updates…).

Three kinds of copy (``detect``):

- ``git``: a clone. ``check`` fetches the branch's upstream; ``apply`` moves the branch forward to it
  (``git merge --ff-only``). Nothing else is done to the clone: a branch with commits of its own or a rewritten
  history, edits of a file the new version changes, or no upstream - the update stops before it changes anything
  and says why (``tools/update_source.ps1`` handles a rewritten history on Windows).
- ``folder``: a folder from GitHub's Code ▸ Download ZIP (no ``.git``). ``check`` asks GitHub for the newest
  commit of ``main`` and its file list (two requests to api.github.com) and compares every file by its Git hash;
  ``apply`` downloads only the files that differ (raw.githubusercontent.com), checks each against its hash, keeps
  the files it replaces in a backup folder, then puts the new ones in place. Each file is compared and written as
  GitHub's ZIP holds it (``.gitattributes``: launchers with CRLF, ``revision.json`` with the commit filled in).
  Files GitHub does not have - data, results, the environment Start_GIMaP.bat made - are never touched; files
  removed on GitHub stay.
- ``installed``: the portable version, a frozen build or a package installed with pip: not updated here (the
  caller opens the releases page).

No network request is made until ``check`` runs. Messages are English templates with ``{fields}``; the
dialog translates them (``tr`` / ``trf``).
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from src.gimap.app.launcher import RESTART_FLAG

REPOSITORY = "zyffcc/gisaxs_gui"
BRANCH = "main"
API = "https://api.github.com/repos/" + REPOSITORY
RAW = "https://raw.githubusercontent.com/" + REPOSITORY
RECORD = ".gimap_source.json"  # a ZIP folder updated here: the commit it now holds (git-ignored)
STAGING = ".gimap_update"  # downloads before they replace anything (git-ignored, removed afterwards)
TIMEOUT_S = 60

Progress = Callable[[str, dict], None]


@dataclass
class Installation:
    kind: str  # "git", "folder" or "installed"
    root: Optional[Path] = None
    reason: str = ""


@dataclass
class Status:
    """What ``check`` found. ``state``: up_to_date, behind (an update is there), ahead, diverged or blocked."""

    installation: Installation
    state: str
    current: str = ""
    latest: str = ""
    latest_date: str = ""
    new_commits: list = field(default_factory=list)  # "abc1234 subject", newest first
    n_new: int = 0
    files: list = field(default_factory=list)  # folder: (path, blob sha, mode, size) to download
    download_bytes: int = 0
    message: str = ""
    values: dict = field(default_factory=dict)
    upstream: str = ""


@dataclass
class Result:
    updated: bool
    old: str = ""
    new: str = ""
    changed_files: int = 0
    requirements: list = field(default_factory=list)
    backup: str = ""
    message: str = ""
    values: dict = field(default_factory=dict)


class UpdateError(RuntimeError):
    """A failure with a message template and its values (nothing was changed unless the message says so)."""

    def __init__(self, template: str, **values):
        super().__init__(template.format(**values) if values else template)
        self.template = template
        self.values = values


# -- what this copy is ----------------------------------------------------------------------------------------


def running_root() -> Path:
    """The folder that holds ``main.py`` and ``src/gimap`` of the GIMaP running (this file is src/gimap/app/…)."""
    return Path(__file__).resolve().parents[3]


def detect(root: Optional[Path] = None) -> Installation:
    if getattr(sys, "frozen", False):
        return Installation("installed", reason="a frozen build")
    root = Path(root) if root is not None else running_root()
    if not ((root / "main.py").is_file() and (root / "src" / "gimap").is_dir()):
        return Installation("installed", root, "a package installed with pip")
    if (root / "runtime" / "python.exe").is_file() or (root / "runtime" / "pythonw.exe").is_file():
        return Installation("installed", root, "the portable version")
    if (root / ".git").exists():
        return Installation("git", root)
    return Installation("folder", root)


# -- Git clone ------------------------------------------------------------------------------------------------


def _git(root: Path, *arguments: str, check: bool = True, timeout: float = 120) -> str:
    git = shutil.which("git")
    if git is None:
        raise UpdateError("Git was not found. This folder is a Git clone: install Git (git-scm.com) to update it.")
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0", LC_ALL="C", LANG="C")
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        done = subprocess.run([git, "-C", str(root), *arguments], capture_output=True, text=True, timeout=timeout,
                              env=env, creationflags=flags, encoding="utf-8", errors="replace")
    except subprocess.TimeoutExpired:
        raise UpdateError("Git did not answer within {seconds} s (git {command}).", seconds=int(timeout),
                          command=arguments[0]) from None
    if check and done.returncode != 0:
        raise UpdateError("git {command} failed: {error}", command=arguments[0],
                          error=(done.stderr or done.stdout).strip()[-600:])
    return done.stdout.strip() if check else f"{done.returncode}\n{done.stdout}\n{done.stderr}"


def _is_ancestor(root: Path, a: str, b: str) -> bool:
    return _git(root, "merge-base", "--is-ancestor", a, b, check=False).split("\n", 1)[0] == "0"


def check_git(inst: Installation) -> Status:
    root = inst.root
    branch = _git(root, "rev-parse", "--abbrev-ref", "HEAD")
    if branch == "HEAD":
        return Status(inst, "blocked", message="This clone is not on a branch (a detached HEAD): nothing to update.")
    upstream = _git(root, "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}", check=False).split("\n")
    if upstream[0] != "0" or not upstream[1].strip():
        return Status(inst, "blocked", message="The branch {branch} has no branch on GitHub to follow (no upstream).",
                      values=dict(branch=branch))
    upstream = upstream[1].strip()
    remote = upstream.split("/", 1)[0]
    _git(root, "fetch", "--quiet", remote, timeout=300)
    head, up = _git(root, "rev-parse", "HEAD"), _git(root, "rev-parse", upstream)
    date = _git(root, "log", "-1", "--format=%cI", up)
    status = Status(inst, "up_to_date", current=head, latest=up, latest_date=date, upstream=upstream,
                    values=dict(branch=branch, upstream=upstream))
    if head == up:
        return status
    if _is_ancestor(root, up, head):
        status.state = "ahead"
        status.message = "The branch {branch} has commits that are not on GitHub ({upstream}): nothing to update."
        return status
    if not _is_ancestor(root, head, up):
        status.state = "diverged"
        status.message = ("The branch {branch} and {upstream} have gone apart (commits of your own, or the history on "
                          "GitHub was rewritten). Nothing was changed: see “Updating a source checkout” in "
                          "README.md (on Windows, Update_GIMaP.bat).")
        return status
    status.state = "behind"
    status.n_new = int(_git(root, "rev-list", "--count", f"HEAD..{upstream}"))
    log = _git(root, "log", "--format=%h %s", "-n", "30", f"HEAD..{upstream}")
    status.new_commits = [line for line in log.splitlines() if line.strip()]
    incoming = set(_git(root, "diff", "--name-only", "HEAD", upstream).splitlines())
    edited = set(_git(root, "diff", "--name-only", "HEAD").splitlines())  # edited tracked files, staged or not
    clash = sorted(incoming & edited)
    if clash:
        status.state = "blocked"
        status.message = ("You have edited files that the new version changes: {files}. Nothing was changed. Keep a "
                          "copy of your edits, undo them (or commit them), then update again.")
        status.values["files"] = ", ".join(clash[:8]) + (" …" if len(clash) > 8 else "")
    return status


def apply_git(status: Status, progress: Optional[Progress] = None) -> Result:
    root = status.installation.root
    old = _git(root, "rev-parse", "HEAD")
    if old != status.current:
        raise UpdateError("The clone changed since the check: check again.")
    _say(progress, "Moving {branch} forward to {commit}…", branch=status.values.get("branch", ""),
         commit=status.latest[:7])
    out = _git(root, "merge", "--ff-only", "--quiet", status.latest, check=False)
    if not out.startswith("0\n"):
        raise UpdateError("Git did not update the clone (nothing was changed): {error}",
                          error=out.split("\n", 1)[1].strip()[-600:])
    new = _git(root, "rev-parse", "HEAD")
    names = _git(root, "diff", "--name-only", old, new).splitlines()
    rewritten = restore_byte_exact(root)
    return Result(True, old, new, len(names), sorted(n for n in names if _is_requirements(n)),
                  values=dict(rewritten=rewritten))


def restore_byte_exact(root: Path) -> int:
    """Files the repository hands out byte for byte (``.gitattributes``: ``-text`` - the analysis package of Sputter,
    the Predict network) that a clone made before that rule holds with converted line endings: written again as the
    repository has them (``tools/update_source.ps1`` does the same). Git trusts its own record of these files and
    mostly does not notice. A file you changed stays as it is. The number written."""
    suspects = []
    for line in _git(root, "ls-files", "--eol").splitlines():
        # i/<line endings in the repository> w/<on disk> attr/<attributes>TAB<path>
        match = re.match(r"^i/(\S*)\s+w/(\S+)\s+attr/-text[^\t]*\t(.+)$", line)
        if match and match.group(1) != match.group(2):
            suspects.append(match.group(3))
    if not suspects:
        return 0
    edited = set(_git(root, "diff", "--name-only", "HEAD").splitlines())
    converted = [name for name in suspects if name not in edited or _git(
        root, "diff", "--ignore-cr-at-eol", "--quiet", "HEAD", "--", name, check=False).startswith("0\n")]
    for name in converted:  # git writes a file again only when it looks changed
        try:
            os.utime(root / name)
        except OSError:
            pass
    if converted:
        _git(root, "checkout-index", "--force", "--index", "--", *converted)
    return len(converted)


# -- folder from Download ZIP ---------------------------------------------------------------------------------


def blob_sha(data: bytes) -> str:
    """Git's hash of a file's content (what GitHub's file list gives for each file)."""
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def http_get(url: str, timeout: float = TIMEOUT_S) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "GIMaP-updater",
                                                   "Accept": "application/vnd.github+json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 (fixed https URLs)
            return response.read()
    except OSError as exc:  # URLError, HTTPError, timeouts: offline, proxy, rate limit
        raise UpdateError("GitHub could not be reached ({error}). Check the internet connection and try again.",
                          error=str(getattr(exc, "reason", exc))) from None


def read_record(root: Path) -> dict:
    try:
        record = json.loads((root / RECORD).read_text(encoding="utf-8"))
        return record if isinstance(record, dict) else {}
    except (OSError, ValueError):
        return {}


def _safe(name: str) -> bool:
    parts = Path(name).parts
    return bool(parts) and not Path(name).is_absolute() and ".." not in parts and parts[0] not in (".git", STAGING)


def _stamped(root: Path) -> str:
    """The commit a ZIP from GitHub was made of: GitHub fills ``src/gimap/shared/revision.json`` in (export-subst)."""
    try:
        commit = json.loads((root / "src" / "gimap" / "shared" / "revision.json").read_text(encoding="utf-8"))["commit"]
    except (OSError, ValueError, KeyError, TypeError):
        return ""
    return commit if re.fullmatch(r"[0-9a-f]{40}", str(commit)) else ""


def parse_attributes(text: str) -> list:
    """``.gitattributes`` as (pattern, {attribute: value}) in order (a later line wins)."""
    rules = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        pattern, *words = line.split()
        attributes = {}
        for word in words:
            if word.startswith("-"):
                attributes[word[1:]] = False
            elif "=" in word:
                key, value = word.split("=", 1)
                attributes[key] = value
            else:
                attributes[word] = True
            if word == "binary":
                attributes.update(text=False, diff=False)
        rules.append((pattern, attributes))
    return rules


def _matches(pattern: str, name: str) -> bool:
    if "/" not in pattern.rstrip("/"):
        return fnmatch.fnmatchcase(name.rsplit("/", 1)[-1], pattern)
    pattern = pattern.lstrip("/")
    if pattern.endswith("/**"):
        return name.startswith(pattern[:-2])
    return fnmatch.fnmatchcase(name, pattern)


def attributes_of(rules: list, name: str) -> dict:
    found: dict = {}
    for pattern, attributes in rules:
        if _matches(pattern, name):
            found.update(attributes)
    return found


def as_checked_out(blob: bytes, attributes: dict, commit: str) -> bytes:
    """The file as a ZIP from GitHub holds it: ``eol=crlf`` text with CRLF, ``export-subst`` placeholders filled in."""
    data = blob
    if attributes.get("export-subst"):
        data = data.replace(b"$Format:%H$", commit.encode("ascii")).replace(b"$Format:%h$", commit[:7].encode("ascii"))
    if attributes.get("eol") == "crlf" and attributes.get("text") is not False and b"\0" not in data:
        data = data.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
    return data


def _raw(get: Callable[[str], bytes], commit: str, name: str, sha: str) -> bytes:
    data = get(f"{RAW}/{commit}/{urllib.parse.quote(name)}")
    if blob_sha(data) != sha:
        raise UpdateError("{file} arrived damaged (its hash does not match GitHub's list). Nothing was changed: try "
                          "again.", file=name)
    return data


def check_folder(inst: Installation, get: Callable[[str], bytes] = http_get) -> Status:
    root = inst.root
    head = json.loads(get(f"{API}/commits/{BRANCH}"))
    latest = str(head["sha"])
    date = str(head.get("commit", {}).get("committer", {}).get("date", ""))
    subject = str(head.get("commit", {}).get("message", "")).splitlines()[0] if head.get("commit") else ""
    tree = json.loads(get(f"{API}/git/trees/{latest}?recursive=1"))
    if tree.get("truncated"):
        raise UpdateError("GitHub sent an incomplete file list. Download the new version as a ZIP instead.")
    items = [item for item in tree.get("tree", []) if item.get("type") == "blob"
             and str(item.get("mode")) != "120000" and _safe(str(item.get("path", "")))]
    rules = []
    for item in items:  # the new version's .gitattributes: how a ZIP of it holds each file
        if item["path"] == ".gitattributes":
            local = root / ".gitattributes"
            text = local.read_bytes() if local.is_file() and blob_sha(local.read_bytes()) == item["sha"] else \
                _raw(get, latest, ".gitattributes", item["sha"])
            rules = parse_attributes(text.decode("utf-8", "replace"))
    files = []
    for item in items:
        name, sha, mode = str(item["path"]), str(item["sha"]), str(item.get("mode", ""))
        attributes = attributes_of(rules, name)
        path = root / name
        if attributes.get("export-subst"):  # filled in by GitHub's ZIP: compare with the file as it would be
            expected = as_checked_out(_raw(get, latest, name, sha), attributes, latest)
            if path.is_file() and path.read_bytes() == expected:
                continue
        elif path.is_file():
            data = path.read_bytes()
            if attributes.get("eol") == "crlf" and attributes.get("text") is not False:
                data = data.replace(b"\r\n", b"\n")
            if blob_sha(data) == sha:
                continue
        files.append((name, sha, mode, int(item.get("size") or 0), attributes))
    recorded = str(read_record(root).get("commit", "")) or _stamped(root)
    status = Status(inst, "behind" if files else "up_to_date", current=recorded, latest=latest, latest_date=date,
                    files=files, download_bytes=sum(f[3] for f in files), values=dict(branch=BRANCH))
    if recorded and recorded != latest and files:
        try:
            compare = json.loads(get(f"{API}/compare/{recorded}...{latest}"))
            status.n_new = int(compare.get("total_commits") or 0)
            commits = list(reversed(compare.get("commits") or []))[:30]
            status.new_commits = [f"{c['sha'][:7]} {c['commit']['message'].splitlines()[0]}" for c in commits]
        except (UpdateError, ValueError, KeyError):
            pass
    if not status.new_commits and files:
        status.new_commits = [f"{latest[:7]} {subject}"]
    return status


def apply_folder(status: Status, backup_root: Path, get: Callable[[str], bytes] = http_get,
                 progress: Optional[Progress] = None, cancelled: Callable[[], bool] = lambda: False) -> Result:
    root = status.installation.root
    staging = root / STAGING
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    replaced: list[tuple[Path, Optional[Path]]] = []
    try:
        for k, (name, sha, _mode, _size, attributes) in enumerate(status.files, 1):
            if cancelled():
                raise UpdateError("The update was stopped. Nothing was changed.")
            _say(progress, "Downloading {done} of {total} files…", done=k, total=len(status.files))
            target = staging / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(as_checked_out(_raw(get, status.latest, name, sha), attributes, status.latest))
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        backup = Path(backup_root) / f"{stamp}-{(status.current or 'zip')[:7]}"
        _say(progress, "Putting {total} new files in place…", total=len(status.files))
        try:
            for name, _sha, mode, _size, _attributes in status.files:
                path, kept = root / name, None
                if path.exists():
                    kept = backup / name
                    kept.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, kept)
                path.parent.mkdir(parents=True, exist_ok=True)
                os.replace(staging / name, path)
                if mode == "100755" and os.name != "nt":
                    path.chmod(path.stat().st_mode | 0o111)
                replaced.append((path, kept))
        except OSError as exc:
            for path, kept in reversed(replaced):  # back to what the folder held
                if kept is not None:
                    shutil.copy2(kept, path)
                else:
                    path.unlink(missing_ok=True)
            raise UpdateError("A file could not be replaced ({error}); the folder was put back as it was. Close "
                              "programs that use GIMaP's files and try again.", error=exc.strerror or str(exc)) from None
        record = dict(repository=REPOSITORY, branch=BRANCH, commit=status.latest, date=status.latest_date,
                      updated=datetime.now().astimezone().isoformat(timespec="seconds"), files=len(status.files))
        (root / RECORD).write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    names = [f[0] for f in status.files]
    return Result(True, status.current, status.latest, len(names), sorted(n for n in names if _is_requirements(n)),
                  backup=str(backup) if any(k for _p, k in replaced) else "")


# -- both ------------------------------------------------------------------------------------------------------


def check(inst: Installation, get: Callable[[str], bytes] = http_get) -> Status:
    if inst.kind == "git":
        return check_git(inst)
    if inst.kind == "folder":
        return check_folder(inst, get)
    raise UpdateError("This copy of GIMaP ({reason}) is updated by installing the new version.", reason=inst.reason)


def apply(status: Status, backup_root: Path, get: Callable[[str], bytes] = http_get,
          progress: Optional[Progress] = None, cancelled: Callable[[], bool] = lambda: False) -> Result:
    if status.state != "behind":
        raise UpdateError("There is nothing to update.")
    if status.installation.kind == "git":
        return apply_git(status, progress)
    return apply_folder(status, backup_root, get, progress, cancelled)


def _is_requirements(name: str) -> bool:
    base = name.rsplit("/", 1)[-1]
    return base.startswith("requirements") and base.endswith(".txt")


def _say(progress: Optional[Progress], template: str, **values) -> None:
    if progress is not None:
        progress(template, values)


def restart_command(root: Path) -> list:
    """The command that starts this source copy again (the new process waits until this one has let go)."""
    executable = sys.executable
    if os.name == "nt" and Path(executable).name.lower() == "python.exe":
        windowed = Path(executable).with_name("pythonw.exe")
        executable = str(windowed) if windowed.is_file() else executable
    return [executable, str(Path(root) / "main.py"), RESTART_FLAG]
