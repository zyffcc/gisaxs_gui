"""The log of a GIMaP session: ``<user data>/logs/gimap.log`` (rotating), ``crash.log`` and ``errors.log``.

The portable package starts ``pythonw.exe``, where ``print`` writes nowhere: a message such as "Failed to save
the session on close" was lost exactly where it was needed. ``configure_logging`` (called by the launcher
before the heavy imports) therefore

* writes Python ``logging`` (INFO and up) to ``gimap.log``, 1 MB per file and three older ones, the version and
  the platform on the first line of every session;
* routes ``print`` (``sys.stdout``) and ``sys.stderr`` to that log, and still to the console when there is one;
* lets ``faulthandler`` write the Python stack of every thread to ``crash.log`` when the process dies of a
  native fault (an access violation in Qt), each session marked with its start;
* logs an exception that escapes before ``ErrorGuard`` takes over ``sys.excepthook`` (during the start-up);
* forwards Qt's own warnings (``qWarning``) to the log.

``ErrorGuard`` keeps writing ``errors.log`` (the tracebacks a user can send) and logs each error here too.
Nothing of this runs in the tests: only the launcher calls it.
"""

from __future__ import annotations

import faulthandler
import logging
import platform
import sys
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional, TextIO

LOG_FILE = "gimap.log"
SECOND_LOG_FILE = "gimap-second.log"
"""The log of a second GIMaP started while one runs (two processes must not rotate one file)."""
CRASH_FILE = "crash.log"
MAX_BYTES = 1_000_000
BACKUPS = 3
FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"

logger = logging.getLogger("gimap")
_crash_stream: Optional[TextIO] = None
_handler: Optional[logging.Handler] = None


class _LogStream:
    """A text stream that sends each line to a logger (and on to the stream it replaces, when there is one)."""

    def __init__(self, target: logging.Logger, level: int, original: Optional[TextIO]):
        self.target = target
        self.level = level
        self.original = original
        self._pending = ""

    def write(self, text) -> int:
        text = str(text)
        if self.original is not None:
            try:
                self.original.write(text)
            except (OSError, ValueError):
                pass
        self._pending += text
        *lines, self._pending = self._pending.split("\n")
        for line in lines:
            if line.strip():
                self.target.log(self.level, line.rstrip())
        return len(text)

    def flush(self) -> None:
        if self.original is not None:
            try:
                self.original.flush()
            except (OSError, ValueError):
                pass

    def isatty(self) -> bool:
        return False

    @property
    def encoding(self) -> str:
        return getattr(self.original, "encoding", None) or "utf-8"


def log_folder(data_dir: Path | str) -> Path:
    return Path(data_dir) / "logs"


def configure_logging(data_dir: Path | str, *, version: str, second: bool = False) -> Path:
    """Start the session's log in ``<data_dir>/logs``; returns the log file. Called once per process."""
    global _crash_stream, _handler
    folder = log_folder(data_dir)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / (SECOND_LOG_FILE if second else LOG_FILE)
    root = logging.getLogger()
    if _handler is None:
        _handler = RotatingFileHandler(path, maxBytes=MAX_BYTES, backupCount=BACKUPS, encoding="utf-8")
        _handler.setFormatter(logging.Formatter(FORMAT))
        root.addHandler(_handler)
        root.setLevel(logging.INFO)
    logger.info("GIMaP %s started (pid %s) on %s, Python %s", version, _pid(), platform.platform(),
                platform.python_version())
    if not second and _crash_stream is None:
        try:
            crash = folder / CRASH_FILE
            if crash.is_file() and crash.stat().st_size > MAX_BYTES:  # a line per start: kept short
                crash.replace(crash.with_suffix(".old.log"))
            _crash_stream = open(crash, "a", encoding="utf-8")  # noqa: SIM115 - open for the process
            _crash_stream.write(f"==== {time.strftime('%Y-%m-%d %H:%M:%S')} GIMaP {version} started (pid {_pid()})\n")
            _crash_stream.flush()
            faulthandler.enable(file=_crash_stream, all_threads=True)
        except OSError:
            logger.warning("crash.log could not be opened; native crashes are not recorded")
    if not isinstance(sys.stdout, _LogStream):
        sys.stdout = _LogStream(logging.getLogger("gimap.stdout"), logging.INFO, sys.stdout)
    if not isinstance(sys.stderr, _LogStream):
        sys.stderr = _LogStream(logging.getLogger("gimap.stderr"), logging.WARNING, sys.stderr)
    previous = sys.excepthook

    def log_and_pass(exc_type, exc, tb):
        if not issubclass(exc_type, KeyboardInterrupt):
            logger.error("Unexpected error during the start-up", exc_info=(exc_type, exc, tb))
        previous(exc_type, exc, tb)

    sys.excepthook = log_and_pass
    _forward_qt_messages()
    return path


def _forward_qt_messages() -> None:
    """Qt's warnings and criticals (a style sheet it cannot parse, a failed connection) into the log."""
    try:
        from PyQt5.QtCore import QtCriticalMsg, QtFatalMsg, QtWarningMsg, qInstallMessageHandler
    except ImportError:
        return
    qt = logging.getLogger("gimap.qt")
    levels = {QtWarningMsg: logging.WARNING, QtCriticalMsg: logging.ERROR, QtFatalMsg: logging.CRITICAL}

    def handler(kind, _context, message) -> None:
        level = levels.get(kind)
        if level is not None:
            qt.log(level, message)

    qInstallMessageHandler(handler)


def _pid() -> int:
    import os

    return os.getpid()


def stop_logging() -> None:
    """At the end of a session: the last line, and the crash log closed (a clean exit is no crash)."""
    global _crash_stream
    logger.info("GIMaP closed")
    if _crash_stream is not None:
        try:
            faulthandler.disable()
            _crash_stream.close()
        except (OSError, ValueError):
            pass
        _crash_stream = None
    if _handler is not None:
        _handler.flush()


__all__ = ["CRASH_FILE", "LOG_FILE", "SECOND_LOG_FILE", "configure_logging", "log_folder", "logger",
           "stop_logging"]
