"""Predict's batch likelihoods on several threads.

Every sample's likelihood depends on that sample alone (its own size rule, and a resolution grid that the prior's
widths, at most 1 % of q_max, do not change), so the samples are split into chunks, each chunk is evaluated as one
call and the results are joined in order: the numbers are those of a single call. The heavy part, the size averages
of numpy and scipy, runs outside Python's lock, so the threads share the work. The least-squares fits of Predict's
modes are independent of each other too (``ordered``): they run together and are taken in their serial order.

BLAS (numpy's matrix products) runs on one thread while Predict runs (``one_blas_thread``): a BLAS on several
threads that is called from several threads at once splits some calls differently, and their last digits change. So
any number of Predict threads gives the numbers of one, on a busy computer too. Without threadpoolctl, which sets
that, Predict runs on one thread."""

from __future__ import annotations

import functools
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

THREADS_MAX = 12
"""More threads gained little in the timing tests (12 threads 5.8 times one, 8 threads 4.8 times)."""
CHUNK = 16
"""Samples per chunk at least: below that, a thread's share is too small to pay for itself."""
ENVIRONMENT = "GIMAP_PREDICT_THREADS"


def default_threads() -> int:
    """Threads of a Predict run in the program: ``GIMAP_PREDICT_THREADS`` if set, else half the logical processors,
    at most ``THREADS_MAX``."""
    try:
        given = int(os.environ.get(ENVIRONMENT, ""))
    except ValueError:
        given = 0
    if given > 0:
        return given
    return max(1, min(THREADS_MAX, (os.cpu_count() or 2) // 2))


def one_blas_thread(function):
    """``function`` (taking ``threads``) run with BLAS on one thread; on one thread itself without threadpoolctl."""
    @functools.wraps(function)
    def run(*args, threads: int = 1, **kwargs):
        try:
            from threadpoolctl import threadpool_limits
        except ImportError:
            return function(*args, threads=1, **kwargs)
        with threadpool_limits(limits=1, user_api="blas"):
            return function(*args, threads=threads, **kwargs)
    return run


def evaluate(likelihood, values, threads: int = 1) -> tuple[np.ndarray, ...]:
    """``likelihood(values)`` (a tuple of arrays, each with one row per sample) on ``threads`` threads."""
    values = np.atleast_2d(values)
    count = values.shape[0]
    if threads <= 1 or count < 2 * CHUNK:
        return likelihood(values)
    chunks = np.array_split(np.arange(count), min(count // CHUNK, 4 * threads))
    with ThreadPoolExecutor(max_workers=threads) as pool:
        parts = list(pool.map(lambda rows: likelihood(values[rows]), chunks))
    return tuple(np.concatenate([part[k] for part in parts]) for k in range(len(parts[0])))


def ordered(function, items, threads: int = 1):
    """``function(item)`` of every item, yielded in the order of ``items``; on ``threads`` threads the later items
    are under way while the caller handles the earlier ones. A caller that stops early (an exception, a cancelled
    run) leaves the items not yet started undone."""
    items = list(items)
    if threads <= 1 or len(items) < 2:
        for item in items:
            yield function(item)
        return
    pool = ThreadPoolExecutor(max_workers=threads)
    try:
        futures = [pool.submit(function, item) for item in items]
        for future in futures:
            yield future.result()
    finally:
        pool.shutdown(wait=True, cancel_futures=True)


__all__ = ["CHUNK", "ENVIRONMENT", "THREADS_MAX", "default_threads", "evaluate", "one_blas_thread", "ordered"]
