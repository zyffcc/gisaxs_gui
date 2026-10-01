"""Stages and odd frames of a series of curves, shared by Analyze (Series), Fitting (In-situ series)
and Compare: pure NumPy, no Qt. See ``comparable.py`` and ``stages.py``."""

from .comparable import COVERAGE, Comparable, Components, OddFrame, comparable, odd_frames, principal
from .stages import GAIN, MOST, SHORTEST, SeriesStages, StageChange, choose, find_stages, oriented, progress, segment

__all__ = [
    "COVERAGE", "GAIN", "MOST", "SHORTEST", "Comparable", "Components", "OddFrame", "SeriesStages", "StageChange",
    "choose", "comparable", "find_stages", "odd_frames", "oriented", "principal", "progress", "segment",
]
