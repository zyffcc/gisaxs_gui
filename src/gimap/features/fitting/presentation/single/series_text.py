"""How the In-situ series page names a frame and its state (no Qt): the marks and colours of the states,
the short words for the halves, and file names short enough to leave room for what matters in a line.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

STATE_MARK = {"waiting": "·", "running": "▸", "done": "✓", "warn": "!", "stopped": "■", "failed": "✗"}
"""A frame's state in the list: not fitted yet, being fitted, fitted, fitted but not to be trusted as it is
(not converged, a poor fit), stopped in the middle of its fit, no fit."""
STATE_COLOR = {"warn": "warning", "stopped": "warning", "failed": "danger"}
"""The theme colour of a frame's line by its state; the other states keep the list's own colour."""
FIGURE_SPACE = " "
"""Pads the frame numbers: as wide as a digit (a space is narrower)."""
HALF_TEXT = {"mean": "mean", "positive": "q > 0", "negative": "q < 0"}
"""What a frame was fitted on, in a cell of the table (``FrameFit.halves_used``)."""
ONE_HALF = {"positive": "q > 0 half only", "negative": "q < 0 half only"}
"""Said of a frame fitted under “mean of both halves” that has one half only."""
LONG_PREFIX = 8
"""A beginning that every name shares is shown as “…” when it is longer than this."""
ANALYZE_FOLDER, FIT_INPUT_FOLDER = "gimap_analysis", "fit_input"
"""Where Analyze's Batch Export and Send Series write the curves of a series, below the folder the person
chose there: ``<data>/gimap_analysis[/<batch export>]/fit_input``."""


def series_name(folder) -> str:
    """What a series is called where its whole path has no room. Every series Analyze writes lies in
    ``gimap_analysis/fit_input``, so those names tell no series from another: it is named by the folder that
    holds ``gimap_analysis`` and, after “ · ”, the folders below it (a Batch Export) —
    ``…/sampleA/gimap_analysis/fit_input`` is “sampleA”, ``…/sampleA/gimap_analysis/run_2/fit_input``
    “sampleA · run_2”. Any other folder: its last two names (without a ``fit_input`` at the end)."""
    names = list(Path(folder).parts) if folder else []
    if names and Path(folder).anchor:
        names = names[1:]  # not the drive or the root
    kept = names[:-1] if names and names[-1].casefold() == FIT_INPUT_FOLDER else names
    at = max((place for place, name in enumerate(kept) if name.casefold() == ANALYZE_FOLDER), default=-1)
    if at > 0:
        return " · ".join([kept[at - 1]] + (["/".join(kept[at + 1:])] if kept[at + 1:] else []))
    return "/".join((kept or names)[-2:])


def _common_prefix(names: Sequence[str]) -> str:
    prefix = os.path.commonprefix(list(names))
    while prefix and prefix[-1].isdigit():  # never the first digits of a number: 00001 stays whole
        prefix = prefix[:-1]
    return prefix


def _common_suffix(names: Sequence[str]) -> str:
    suffix = os.path.commonprefix([name[::-1] for name in names])[::-1]
    while suffix and suffix[0].isalnum():  # nor the last digits or letters of what differs
        suffix = suffix[1:]
    return suffix


def short_names(names: Sequence[str]) -> list[str]:
    """The names without the ending they all share (``_fit_input.dat``) and, when it is long, without the
    beginning they all share (shown as “…”): ``…00012`` for ``sample_insitu_ds03_00001_00012_fit_input.dat``.
    Numbers stay whole and every name stays different; names that share nothing — or a single one — are kept."""
    names = list(names)
    if len(names) < 2:
        return names
    prefix, suffix = _common_prefix(names), _common_suffix(names)
    if len(prefix) <= LONG_PREFIX:
        prefix = ""
    short = [name[len(prefix):len(name) - len(suffix)] for name in names]
    if any(not text for text in short):
        return names
    return [("…" if prefix else "") + text for text in short]


__all__ = ["ANALYZE_FOLDER", "FIGURE_SPACE", "FIT_INPUT_FOLDER", "HALF_TEXT", "LONG_PREFIX", "ONE_HALF", "STATE_COLOR",
           "STATE_MARK", "series_name", "short_names"]
