"""Worker processes that reduce the frames of a batch in parallel, gently.

``ProcessFrameWorkers(count, profiles)`` runs ``run_frame_job`` in ``count`` spawned processes and
returns ``concurrent.futures.Future``s of ``FrameOutcome``s. So that a long batch does not make the
computer sluggish:

* the workers run at *below normal* priority (the desktop, the GUI and other programs come first);
* each worker's numerical libraries use one thread (``OMP_NUM_THREADS`` … while it starts), so four
  workers use four cores, not four times all of them;
* the application submits at most a few more frames than there are workers, so only those frames
  are in memory (the caller limits this; ``frames_at_once`` sizes the pool to the free memory).

A spawned process imports the program's main module again. ``main.py`` imports the whole GUI (and
BornAgain); while a job is submitted, ``__main__`` is therefore pointed at the empty
``worker_main`` module, and a worker starts with only what a frame needs.

``system_resources()`` → ``(physical cores, free memory in bytes or None)``.
"""

from __future__ import annotations

import importlib.machinery
import multiprocessing
import os
import sys
from concurrent.futures import Future, ProcessPoolExecutor
from contextlib import contextmanager
from typing import Optional, Sequence

from src.gimap.shared.geometry.instrument_profile import InstrumentProfile, best_profile

WORKER_MAIN = "src.gimap.features.analyze.infrastructure.worker_main"
SINGLE_THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")

_TOOLS = None
"""The analysis tools of this worker process (built once by ``_start_worker``)."""


class _Profiles:
    """The instrument profiles of the application, read-only, inside a worker."""

    def __init__(self, profiles: Sequence[InstrumentProfile]):
        self._profiles = {profile.name: profile for profile in profiles}

    def load_all(self) -> list[InstrumentProfile]:
        return sorted(self._profiles.values(), key=lambda item: item.name)

    def find(self, name: str) -> Optional[InstrumentProfile]:
        return self._profiles.get(name)

    def save(self, profile: InstrumentProfile) -> None:  # a batch never saves profiles
        return None

    def match(self, *, detector_name, shape) -> Optional[InstrumentProfile]:
        return best_profile(self.load_all(), detector_name=detector_name, shape=shape)


def _lower_priority() -> None:
    try:
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes

            below_normal = 0x00004000
            kernel32 = ctypes.windll.kernel32
            kernel32.GetCurrentProcess.restype = wintypes.HANDLE  # a 64-bit pseudo-handle, not an int
            kernel32.SetPriorityClass.argtypes = (wintypes.HANDLE, wintypes.DWORD)
            kernel32.SetPriorityClass(kernel32.GetCurrentProcess(), below_normal)
        else:
            os.nice(10)
    except (OSError, AttributeError):
        pass


def _start_worker(profiles: Sequence[InstrumentProfile], low_priority: bool) -> None:
    global _TOOLS
    os.environ["MPLBACKEND"] = "Agg"
    if low_priority:
        _lower_priority()
    from ..application import AnalyzeFrame, ExportAnalysis, FrameTools
    from .adapters import CsvCurveWriter, DetectorIoFrameSource, MatplotlibFigureWriter

    _TOOLS = FrameTools(
        AnalyzeFrame(DetectorIoFrameSource(), _Profiles(profiles)), ExportAnalysis(CsvCurveWriter()),
        MatplotlibFigureWriter(),
    )


def _run_job(job):
    from ..application import run_frame_job

    return run_frame_job(_TOOLS, job)


@contextmanager
def _light_main():
    """While workers are spawned: their ``__main__`` is ``worker_main``, not the GUI's ``main.py``."""
    main = sys.modules.get("__main__")
    replace_it = main is not None and getattr(main, "__spec__", None) is None and getattr(main, "__file__", None)
    if replace_it:
        main.__spec__ = importlib.machinery.ModuleSpec(WORKER_MAIN, None)
    try:
        yield
    finally:
        if replace_it:
            main.__spec__ = None


@contextmanager
def _single_threaded_libraries():
    """Workers inherit the environment when they start: one thread each for BLAS/OpenMP."""
    saved = {name: os.environ.get(name) for name in SINGLE_THREAD_VARIABLES}
    for name, value in saved.items():
        if value is None:
            os.environ[name] = "1"
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)


class ProcessFrameWorkers:
    """``count`` worker processes for ``FrameJob``s (``submit`` → ``Future[FrameOutcome]``)."""

    def __init__(self, count: int, profiles: Sequence[InstrumentProfile], *, low_priority: bool = True):
        self.count = max(1, int(count))
        self._executor = ProcessPoolExecutor(
            max_workers=self.count, mp_context=multiprocessing.get_context("spawn"),
            initializer=_start_worker, initargs=(list(profiles), bool(low_priority)),
        )

    def submit(self, job) -> Future:
        with _light_main(), _single_threaded_libraries():
            return self._executor.submit(_run_job, job)

    def shutdown(self, *, wait: bool = False) -> None:
        """Stop: frames not started are dropped; those being reduced finish in the background."""
        self._executor.shutdown(wait=wait, cancel_futures=True)


def system_resources() -> tuple[int, Optional[int]]:
    """``(physical cores, free memory in bytes)``; the memory is ``None`` when it cannot be read."""
    try:
        import psutil

        cores = psutil.cpu_count(logical=False) or os.cpu_count() or 1
        return int(cores), int(psutil.virtual_memory().available)
    except (ImportError, OSError, RuntimeError):
        pass
    cores = os.cpu_count() or 1
    if sys.platform == "win32":
        try:
            import ctypes

            class _Status(ctypes.Structure):
                _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total", ctypes.c_ulonglong),
                            ("available", ctypes.c_ulonglong), ("page_total", ctypes.c_ulonglong),
                            ("page_available", ctypes.c_ulonglong), ("virtual_total", ctypes.c_ulonglong),
                            ("virtual_available", ctypes.c_ulonglong), ("extended", ctypes.c_ulonglong)]

            status = _Status()
            status.length = ctypes.sizeof(_Status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return max(1, cores // 2), int(status.available)
        except (OSError, AttributeError):
            pass
        return max(1, cores // 2), None
    try:
        return max(1, cores // 2), int(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
    except (ValueError, OSError, AttributeError):
        return max(1, cores // 2), None


__all__ = ["ProcessFrameWorkers", "WORKER_MAIN", "system_resources"]
