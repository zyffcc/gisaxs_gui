"""One GIMaP per user data folder: a lock file there, and a local socket to reach the one that runs.

Two GIMaP processes on one folder overwrite each other's settings, session and autosave (the last one closed
wins). The first start takes ``gimap.lock`` in the user folder (``QLockFile``: a lock left by a process that
died is taken over) and listens on a local socket named after the folder. A second start finds the lock
taken; the launcher then asks the user (``app_dialogs.ask_already_running``) and either hands its files to
the running GIMaP, which comes to the front (``activate``), or starts with read-only settings.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Sequence

from PyQt5.QtCore import QCoreApplication, QLockFile, QObject, pyqtSignal
from PyQt5.QtNetwork import QLocalServer, QLocalSocket

LOCK_FILE = "gimap.lock"
TIMEOUT_MS = 1500


def server_name(data_dir: Path | str) -> str:
    """The local socket of the GIMaP that uses ``data_dir`` (one per folder and user)."""
    key = os.path.normcase(str(Path(data_dir).resolve()))
    return "gimap-" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


class SingleInstance(QObject):
    """``acquire()`` the folder; ``activated(paths)`` fires when a second start hands its files over."""

    activated = pyqtSignal(list)

    def __init__(self, data_dir: Path | str, parent: QObject | None = None):
        super().__init__(parent)
        self.data_dir = Path(data_dir)
        self.name = server_name(self.data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.lock = QLockFile(str(self.data_dir / LOCK_FILE))
        self.lock.setStaleLockTime(0)  # a lock is stale when its process is gone, not when it is old
        self.server: QLocalServer | None = None
        self.owner = False

    def acquire(self) -> bool:
        """``True`` when this process now owns the folder (it then listens for later starts)."""
        if not self.lock.tryLock(100):
            return False
        self.owner = True
        QLocalServer.removeServer(self.name)  # a socket left by a process that died
        self.server = QLocalServer(self)
        self.server.setSocketOptions(QLocalServer.UserAccessOption)
        self.server.newConnection.connect(self._connected)
        self.server.listen(self.name)
        return True

    def hand_over(self, paths: Sequence[str] = ()) -> bool:
        """Ask the running GIMaP to come to the front and open ``paths``; ``False`` when it does not answer."""
        socket = QLocalSocket()
        socket.connectToServer(self.name)
        if not socket.waitForConnected(TIMEOUT_MS):
            return False
        written = socket.write((json.dumps({"open": [str(path) for path in paths]}) + "\n").encode("utf-8"))
        socket.flush()
        # A Windows pipe finishes the write in the event loop: wait for it there (not past the timeout).
        deadline = time.monotonic() + TIMEOUT_MS / 1000
        while socket.bytesToWrite() and time.monotonic() < deadline:
            QCoreApplication.processEvents()
            socket.waitForBytesWritten(50)
        done = written > 0 and socket.bytesToWrite() == 0
        socket.disconnectFromServer()
        return done

    def _connected(self) -> None:
        while self.server is not None and self.server.hasPendingConnections():
            socket = self.server.nextPendingConnection()
            socket.readyRead.connect(lambda socket=socket: self._read(socket))
            socket.disconnected.connect(socket.deleteLater)
            self._read(socket)  # what arrived with the connection

    def _read(self, socket: QLocalSocket) -> None:
        if socket.property("gimapRead") or not socket.canReadLine():
            return
        socket.setProperty("gimapRead", True)
        try:
            message = json.loads(bytes(socket.readLine()).decode("utf-8"))
        except ValueError:
            return
        paths = message.get("open") if isinstance(message, dict) else None
        self.activated.emit([str(path) for path in paths] if isinstance(paths, list) else [])

    def release(self) -> None:
        if self.server is not None:
            self.server.close()
            self.server = None
        if self.owner:
            self.lock.unlock()
            self.owner = False


__all__ = ["LOCK_FILE", "SingleInstance", "server_name"]
