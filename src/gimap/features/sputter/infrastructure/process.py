"""Run a child process, stream its lines, and stop it together with its children.

Used by Analyze and Static Frames (the client analysis package). A reader thread hands the output over line by line;
the caller's ``should_stop`` is asked every 0.2 s (not only when a line arrives: reading frames can be quiet for a
while). Stopping ends the whole process tree — on Windows ``taskkill /T /F``, elsewhere the process group — because
the package reads frames in worker processes that ``terminate()`` alone would leave running. ``default_workers`` says
how many of those workers the computer can take.
"""

from __future__ import annotations

import os
import queue
import signal
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

POLL_SECONDS = 0.2
TAIL_LINES = 40
WORKER_BYTES = 0.3e9
"""Memory of one worker of the analysis package (about 0.2 GB measured in an Analyze run, with a margin)."""
KEPT_FREE_BYTES = 1.5e9
"""Memory kept for the package's main process (about 1.1 GB measured) and the rest of the computer."""


def default_workers() -> int:
    """Worker processes of the analysis package: one per core (at most 16, two cores left free; at least one) and no
    more than the free memory holds after the main process, (free − 1.5 GB) / 0.3 GB, so a busy computer analyses
    more slowly instead of running out of memory."""
    by_cores = max(1, min(16, (os.cpu_count() or 3) - 2))
    try:
        import psutil

        available = float(psutil.virtual_memory().available)
    except Exception:  # psutil missing or unreadable: cores only
        return by_cores
    return max(1, min(by_cores, int((available - KEPT_FREE_BYTES) // WORKER_BYTES)))


def console_python() -> str:
    """GIMaP's own interpreter, as the console ``python.exe`` (``pythonw.exe`` has no output to read)."""
    exe = Path(sys.executable)
    if exe.name.lower() == "pythonw.exe":
        console = exe.with_name("python.exe")
        return str(console if console.is_file() else exe)
    return str(exe)


def child_environment(**extra: str) -> dict:
    """The environment of a child Python: unbuffered UTF-8 output, without the user's site-packages and
    ``PYTHONPATH`` (only the packages of GIMaP's own environment count)."""
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8", PYTHONNOUSERSITE="1", **extra)
    env.pop("PYTHONPATH", None)
    return env


@dataclass
class Outcome:
    code: Optional[int]
    tail: list = field(default_factory=list)
    stopped: bool = False


def _creation_flags() -> int:
    return getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0


def kill_tree(process: subprocess.Popen) -> None:
    """End ``process`` and every process it started."""
    if process.poll() is not None:
        return
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(process.pid)], stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, creationflags=_creation_flags(), check=False)
    else:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except (OSError, AttributeError):
            pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()


def _read(stream, lines: "queue.Queue[Optional[str]]") -> None:
    try:
        for line in stream:
            lines.put(line.rstrip("\r\n"))
    except (OSError, ValueError):  # the pipe closed while reading
        pass
    finally:
        lines.put(None)


def run_streaming(command: Sequence[str], *, on_line: Callable[[str], None], should_stop: Callable[[], bool],
                  can_stop: Callable[[], bool] = lambda: True, env: Optional[dict] = None, cwd: Optional[str] = None,
                  poll_s: float = POLL_SECONDS, kill: Callable[[subprocess.Popen], None] = kill_tree) -> Outcome:
    """Run ``command`` to its end, or until ``should_stop()`` (while ``can_stop()``): the process tree is then
    killed and ``Outcome.stopped`` is set. Every non-empty line goes to ``on_line``; the last ones are kept."""
    kwargs = {} if sys.platform == "win32" else {"start_new_session": True}
    process = subprocess.Popen(list(command), cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               stdin=subprocess.DEVNULL, text=True, encoding="utf-8", errors="replace",
                               creationflags=_creation_flags(), **kwargs)
    lines: "queue.Queue[Optional[str]]" = queue.Queue()
    reader = threading.Thread(target=_read, args=(process.stdout, lines), daemon=True)
    reader.start()
    outcome = Outcome(code=None)
    ended = False
    try:
        while not ended:
            try:
                line = lines.get(timeout=poll_s)
            except queue.Empty:
                line = ""
            if line is None:
                ended = True
            elif line:
                outcome.tail = (outcome.tail + [line])[-TAIL_LINES:]
                on_line(line)
            if not ended and should_stop() and can_stop():
                outcome.stopped = True
                kill(process)
                break
    except BaseException:  # a failing callback must not leave the process running
        kill(process)
        raise
    outcome.code = process.wait()
    reader.join(timeout=1)
    if process.stdout is not None:
        process.stdout.close()
    return outcome


__all__ = ["Outcome", "POLL_SECONDS", "child_environment", "console_python", "default_workers", "kill_tree",
           "run_streaming"]
