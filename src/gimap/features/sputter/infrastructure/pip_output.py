"""What pip's output says: the stage of an installation, and why one failed.

| pip says                               | stage                                         |
|----------------------------------------|-----------------------------------------------|
| ``Collecting torch…``                  | ``resolve``: looking for a package            |
| ``Downloading torch-….whl (204.1 MB)`` | ``download``                                  |
| ``Progress 1048576 of 214012345``      | ``download_progress`` (pip ≥ 24.1, ``raw``)   |
| ``Using cached torch-….whl``           | ``cached``: nothing to download               |
| ``Installing collected packages: …``   | ``install``: Stop now waits for pip's end     |
| ``Successfully installed …``           | ``done``                                      |

A failure is classified from pip's last lines (``classify_failure``); a connection problem is looked for
before “no matching distribution”, which pip also says when it could not reach the index.
"""

from __future__ import annotations

import re
from typing import Iterable, Optional

from ..application import InstallProgress

DOWNLOAD_START, DOWNLOAD_SPAN = 0.05, 0.8
_COLLECT = re.compile(r"^\s*Collecting\s+([A-Za-z0-9_.\-]+)")
_DOWNLOAD = re.compile(r"^\s*Downloading\s+(\S+)\s*\(([^)]+)\)")
_CACHED = re.compile(r"^\s*Using cached\s+(\S+)")
_PROGRESS = re.compile(r"^\s*Progress\s+(\d+)\s+of\s+(\d+)")
FAILURES = (
    ("disk", ("Errno 28", "No space left on device", "WinError 112", "not enough space on the disk")),
    ("network", ("NewConnectionError", "getaddrinfo failed", "ProxyError", "SSLError", "CERTIFICATE_VERIFY_FAILED",
                 "timed out", "Max retries exceeded", "Temporary failure in name resolution",
                 "Name or service not known", "No connection could be made", "Network is unreachable",
                 "ConnectionResetError", "Could not fetch URL", "ReadTimeoutError", "Connection aborted")),
    ("permission", ("WinError 5", "Errno 13", "Permission denied", "Access is denied")),
    ("no_match", ("No matching distribution found", "Could not find a version that satisfies")),
    ("no_pip", ("No module named pip",)),
)


def _package(file_name: str) -> str:
    """``torch`` of ``…/torch-2.6.0%2Bcpu-cp310-cp310-win_amd64.whl``."""
    name = file_name.rsplit("/", 1)[-1]
    return name.split("-", 1)[0] or name


class PipReader:
    """Turns pip's lines into ``InstallProgress`` (keeps which package is downloading)."""

    def __init__(self):
        self.package = ""
        self.installing = False

    def read(self, line: str) -> Optional[InstallProgress]:
        found = _PROGRESS.match(line)
        if found:
            done, total = int(found.group(1)), int(found.group(2))
            if total <= 0:
                return None
            share = min(1., done / total)
            return InstallProgress("download_progress", DOWNLOAD_START + DOWNLOAD_SPAN * share,
                                   (("package", self.package or "PyTorch"), ("percent", f"{100 * share:.0f}")))
        found = _DOWNLOAD.match(line)
        if found and ".metadata" not in found.group(1):
            self.package = _package(found.group(1))
            return InstallProgress("download", None, (("package", self.package), ("size", found.group(2).strip())))
        found = _CACHED.match(line)
        if found and ".metadata" not in found.group(1):
            return InstallProgress("cached", None, (("package", _package(found.group(1))),))
        found = _COLLECT.match(line)
        if found:
            return InstallProgress("resolve", None, (("package", found.group(1)),))
        if line.lstrip().startswith("Installing collected packages"):
            self.installing = True
            return InstallProgress("install", None, cancellable=False)
        if line.lstrip().startswith("Successfully installed"):
            return InstallProgress("done", 1.0, cancellable=False)
        return None


def classify_failure(lines: Iterable[str]) -> str:
    """The kind of a failed pip run (a key of ``INSTALL_MESSAGES``): from its output, ``other`` when unknown."""
    text = "\n".join(lines)
    for kind, markers in FAILURES:
        if any(marker in text for marker in markers):
            return kind
    return "other"


__all__ = ["FAILURES", "PipReader", "classify_failure"]
