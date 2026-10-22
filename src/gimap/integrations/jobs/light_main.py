"""Spawned worker processes without the GUI: ``with light_main(): process.start()``.

A process started by multiprocessing's *spawn* first runs its parent's main module again (as
``__mp_main__``). GIMaP's ``main.py`` imports PyQt5 and the whole GUI: a worker would spend seconds
loading it, and PyQt5's own msvcp140.dll (older than BornAgain's) would then serve the worker, where
BornAgain 24.1, built with a newer MSVC, crashes in ``simulate()`` (exit 0xC0000005).

While a worker is started (``Process.start()``, or ``ProcessPoolExecutor.submit()``, which starts
workers), ``light_main()`` points ``__main__`` at the empty ``worker_main`` module. Spawn reads the
main module only then, so the parent is unchanged afterwards. ``__main__`` belongs to the whole
process: one lock serialises every start that changes it (job runner and batch workers alike).
A main module with a spec (``python -m …``, pytest) or without a file (an interactive session) is
left alone: spawn either imports it by name or does not run it at all.
"""

from __future__ import annotations

import importlib.machinery
import sys
import threading
from contextlib import contextmanager
from typing import Iterator

WORKER_MAIN = "src.gimap.integrations.jobs.worker_main"
_SPAWN_LOCK = threading.RLock()


@contextmanager
def light_main() -> Iterator[None]:
    """While workers are spawned: their ``__main__`` is ``worker_main``, not the GUI's ``main.py``."""
    with _SPAWN_LOCK:
        main = sys.modules.get("__main__")
        replace_it = (
            main is not None
            and getattr(main, "__spec__", None) is None
            and bool(getattr(main, "__file__", None))
        )
        if replace_it:
            main.__spec__ = importlib.machinery.ModuleSpec(WORKER_MAIN, None)
        try:
            yield
        finally:
            if replace_it:
                main.__spec__ = None
