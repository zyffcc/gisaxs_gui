"""Bound actual batch-fit curve arrays in RAM and spill older curves to an owned temporary directory.

This cache captures outcomes directly. It never reads exported fit files or writes to a data folder.
Old curves are read only when picked; the GUI requests that read through its task runner.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import RLock

import numpy as np

from ..application.batch_preview import FitCurvePreview


class TemporaryFitPreviewCache:
    def __init__(self, max_frames: int = 128, max_bytes: int = 32 * 1024**2):
        self.max_frames, self.max_bytes = max(1, int(max_frames)), max(1, int(max_bytes))
        self.closed = False
        self._memory = OrderedDict()
        self._names = {}
        self._errors = {}
        self._bytes = 0
        self._directory = None
        self._lock = RLock()

    @property
    def directory(self):
        return Path(self._directory.name) if self._directory is not None else None

    @property
    def resident_frames(self):
        return len(self._memory)

    @property
    def resident_bytes(self):
        return self._bytes

    @staticmethod
    def _size(curves):
        return sum(curve.x.nbytes + curve.observed.nbytes + curve.fitted.nbytes for curve in curves)

    def remember(self, index, curves):
        with self._lock:
            if self.closed or not curves:
                return []
            curves = tuple(curves)
            self._memory[int(index)] = curves
            self._bytes += self._size(curves)
            moved = []
            while len(self._memory) > self.max_frames or self._bytes > self.max_bytes:
                key, previous = next(iter(self._memory.items()))
                try:
                    if self._directory is None:
                        self._directory = TemporaryDirectory(prefix="gimap-fit-preview-")
                    arrays = {f"{component}_{axis}": getattr(curve, axis) for component, curve in enumerate(previous)
                              for axis in ("x", "observed", "fitted")}
                    np.savez(self.directory / f"{key}.npz", **arrays)
                    self._names[key] = tuple(curve.name for curve in previous)
                except OSError as exc:
                    # A preview cache failure cannot turn a scientifically successful fit into a failed
                    # frame. Keep its parameters and expose the cache error when that curve is requested.
                    self._errors[key] = str(exc)
                del self._memory[key]
                self._bytes -= self._size(previous)
                moved.append(key)
            return moved

    def read(self, index):
        with self._lock:
            if self.closed:
                raise ValueError("The fit preview was cleared.")
            if index in self._memory:
                return self._memory[index]
            if index in self._errors:
                raise OSError(self._errors[index])
            names = self._names[index]
            with np.load(self.directory / f"{index}.npz", allow_pickle=False) as arrays:
                return tuple(FitCurvePreview(name, arrays[f"{component}_x"].copy(),
                                             arrays[f"{component}_observed"].copy(),
                                             arrays[f"{component}_fitted"].copy())
                             for component, name in enumerate(names))

    def close(self):
        with self._lock:
            self.closed = True
            self._memory.clear()
            self._names.clear()
            self._errors.clear()
            self._bytes = 0
            if self._directory is not None:
                self._directory.cleanup()
                self._directory = None


__all__ = ["TemporaryFitPreviewCache"]
