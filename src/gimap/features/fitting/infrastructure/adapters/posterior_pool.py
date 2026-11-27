"""Predict's least-squares fits in worker processes (``domain.posterior.prediction.predict``'s ``pool``).

The fits of step 2 are independent of each other. On threads they barely ran at the same time: an evaluation of the
model is many small NumPy steps, and Python's lock lets one thread take them at a time. Worker processes have a lock
each. A fit gives the same numbers in a worker as in the job's own process: the same code on the same values, BLAS
on one thread (as in ``parallel.one_blas_thread``).

The workers start when the pool does (``start``, at the beginning of a Predict run), so they load the fitting code
while the network proposes; OpenBLAS starts in them with one thread (about 40 MB each). ``close`` ends them, also the
fits still running when Predict is stopped. When a worker is lost (the computer ran out of memory) or an item
cannot be sent to it, the items are computed in the job's own process (the same numbers, more slowly).
"""

from __future__ import annotations

import ctypes
import multiprocessing
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from typing import Optional

from ...domain.posterior import parallel

BLAS_VARIABLE = "OPENBLAS_NUM_THREADS"
WORKER_MEMORY = 200 * 2**20
"""Bytes of memory a worker needs at most (its code, a model's tables, a fit's arrays: about 140 MB measured):
fewer workers on a full computer."""
RESERVE = 1536 * 2**20
"""Bytes left to the job's own process and the other programs."""


def available() -> bool:
    """Worker processes need threadpoolctl, which keeps their BLAS on one thread."""
    try:
        import threadpoolctl  # noqa: F401
    except ImportError:
        return False
    return True


def _commit_left() -> Optional[int]:
    """Windows: the bytes the commit limit still allows (an allocation fails there, with physical memory to
    spare)."""
    class Status(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total_physical", ctypes.c_ulonglong),
                    ("free_physical", ctypes.c_ulonglong), ("commit_limit", ctypes.c_ulonglong),
                    ("commit_left", ctypes.c_ulonglong), ("total_virtual", ctypes.c_ulonglong),
                    ("free_virtual", ctypes.c_ulonglong), ("free_extended", ctypes.c_ulonglong)]

    status = Status()
    status.length = ctypes.sizeof(Status)
    try:
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.commit_left)
    except (AttributeError, OSError):
        pass
    return None


def free_memory() -> Optional[int]:
    """Bytes new processes can still have: the available memory, on Windows at most what the commit limit
    leaves; None when unknown."""
    try:
        import psutil

        free = int(psutil.virtual_memory().available)
    except (ImportError, OSError):
        free = None
    if sys.platform == "win32":
        left = _commit_left()
        if left is not None:
            free = left if free is None else min(free, left)
    return free


def workers_for(threads: int, free: Optional[int] = None) -> int:
    """Workers for ``threads`` (``parallel.default_threads``): no more than the free memory holds after
    ``RESERVE`` (``free``: bytes; measured when None)."""
    if free is None:
        free = free_memory()
    if free is not None:
        threads = min(threads, int((free - RESERVE) // WORKER_MEMORY))
    return max(0, int(threads))


class FitPool:
    """Worker processes for the fits of one Predict run; ``ordered`` as ``parallel.ordered``."""

    def __init__(self, workers: int):
        self.workers = int(workers)
        self._executor: Optional[ProcessPoolExecutor] = None
        self.broken = False
        """A worker was lost: everything is computed in this process from then on."""

    def start(self) -> "FitPool":
        """Start every worker now (each loads the fitting code), with one OpenBLAS thread."""
        if self._executor is not None:
            return self
        from src.gimap.integrations.jobs.light_main import light_main  # the jobs package: only where a pool starts

        with light_main():  # the workers' main module: GIMaP's light one, not a script or the window
            saved = os.environ.get(BLAS_VARIABLE)
            os.environ[BLAS_VARIABLE] = "1"
            try:
                self._executor = ProcessPoolExecutor(self.workers, mp_context=multiprocessing.get_context("spawn"),
                                                     initializer=parallel.worker_start)
                for _ in range(self.workers):  # a worker is started for each task while none is idle
                    self._executor.submit(parallel.worker_pid)
            finally:
                if saved is None:
                    os.environ.pop(BLAS_VARIABLE, None)
                else:
                    os.environ[BLAS_VARIABLE] = saved
        return self

    def ordered(self, function, items):
        """``function(*item)`` of every item in the workers, yielded in the order of ``items``; the items not yet
        started are dropped when the caller stops early."""
        items = list(items)
        futures = []
        if not self.broken:
            try:
                executor = self.start()._executor
                futures = [executor.submit(function, *item) for item in items]
            except (BrokenProcessPool, RuntimeError):
                self.broken, futures = True, []
        try:
            for index, item in enumerate(items):
                if index < len(futures) and not self.broken:
                    try:
                        yield futures[index].result()
                        continue
                    except BrokenProcessPool:
                        self.broken = True
                    except (pickle.PicklingError, TypeError, AttributeError):
                        pass  # an item that cannot be sent: computed here (a real error is raised again here)
                yield function(*item)
        finally:
            for future in futures:
                future.cancel()

    def close(self) -> None:
        """End the workers, also those still fitting (their results are not needed)."""
        executor, self._executor = self._executor, None
        if executor is None:
            return
        processes = list((getattr(executor, "_processes", None) or {}).values())
        executor.shutdown(wait=False, cancel_futures=True)
        for process in processes:
            try:
                if process.is_alive():
                    process.terminate()
            except (OSError, ValueError):
                pass

    def __enter__(self) -> "FitPool":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


__all__ = ["FitPool", "RESERVE", "WORKER_MEMORY", "available", "free_memory", "workers_for"]
