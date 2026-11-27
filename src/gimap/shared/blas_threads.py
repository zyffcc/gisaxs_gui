"""The threads that the OpenBLAS libraries of a GIMaP process start with.

OpenBLAS reserves its buffers for all its threads when it is loaded: about 31 MB of committed memory per thread, once
for NumPy's copy and once for SciPy's. With one thread per processor core (24 on a desktop processor with 32 logical
processors) a process holds 1.5 GB before it computes anything, and a few such processes reach the commit limit of
Windows: an array of a few MB then cannot be had. GIMaP's matrices are small, and Predict runs one BLAS thread in each
of its own threads, so four threads are enough. A value of ``OPENBLAS_NUM_THREADS`` that the user set is kept.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator

VARIABLE = "OPENBLAS_NUM_THREADS"
THREADS = 4


def threads() -> int:
    """The BLAS threads of a GIMaP process: ``THREADS``, at most the logical processors."""
    return max(1, min(THREADS, os.cpu_count() or 1))


def cap_environment() -> None:
    """In a worker that GIMaP starts, before NumPy is imported: the cap, unless a value is set already."""
    os.environ.setdefault(VARIABLE, str(threads()))


@contextmanager
def capped() -> Iterator[None]:
    """The cap while NumPy and SciPy load in the program's own process. Afterwards the environment is as before, so
    the programs that GIMaP starts (Sputter's analysis) keep their own settings."""
    if VARIABLE in os.environ:
        yield
        return
    os.environ[VARIABLE] = str(threads())
    try:
        yield
    finally:
        os.environ.pop(VARIABLE, None)


def load_numerics() -> None:
    """Load NumPy and SciPy, each with its own OpenBLAS, under the cap (nothing changes when they are loaded)."""
    with capped():
        import numpy  # noqa: F401
        import scipy.linalg  # noqa: F401


__all__ = ["THREADS", "VARIABLE", "cap_environment", "capped", "load_numerics", "threads"]
