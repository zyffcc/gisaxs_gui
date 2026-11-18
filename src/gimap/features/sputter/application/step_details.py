"""What a step of an Analyze run found: its detail file, read when the user opens the step and dropped when it closes.

A newer package writes a small file per step into the result's ``trace/`` folder (the import's also beside its cache,
``<data root>/<label>.trace/``, from where they are copied into every result made from that import). The step event
names it (``detail``); the kind of picture follows from its name:

* ``detector.json`` (with ``detector.npz``: ``image``, log10 of the summed counts binned by ``binning``, NaN where
  nothing; ``masked``, the share of each binned pixel the mask leaves out) — the beam centre and the pixel mask:
  ``col``, ``row`` and ``horizon_row`` in binned pixels (row 0 at the top), ``frames_summed``, ``dead``, ``hot``,
  ``shadow``, ``kept``, ``pixels``;
* ``reading.json`` — how the frames were read: ``frames_read``, ``frames_total``, ``workers``, ``seconds``,
  ``read_s_per_frame``, ``read_share``, ``mb_read``, ``slow``, ``selection`` and ``frames_read_ranges``;
* ``onset.json`` — the start of the deposition: ``times``, ``rate`` (of the onset band, per bin), ``baseline``,
  ``base_window``, ``onset_rule``, ``onset``, ``how``, ``pre_window``;
* ``windows.json`` — the time windows: ``times``, ``step``, ``max_windows``;
* ``compare.json`` — the networks compared: ``deviations`` {family: median fit deviation}, ``chosen``;
* ``joint.json`` — the joint fit: ``family``, ``total``, ``iterations`` (each with ``iteration``, ``seconds``,
  ``median``, ``sum_dev5``, ``penalty`` and ``windows``: per window ``t`` and the quantities of its solution).

Nothing here is drawn: the page draws (``presentation/step_pictures.py``); this module turns a file's content into
plain values and keeps no file open.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..domain.client import number

DETAIL_KINDS = {"detector.json": "detector", "reading.json": "reading", "onset.json": "onset",
                "windows.json": "windows", "compare.json": "compare", "joint.json": "joint"}
"""The kind of picture of each detail file the package writes."""
JOINT_KEYS = ("R_nm", "H_nm", "D_nm", "base_nm", "coverage")
"""The quantities of the joint fit's solution the joint step's picture draws."""
RANGES_SHOWN = 12
"""Runs of frames read listed in words (the picture shows them all)."""
NO_DETAIL = "The details of this step are not in {folder} (the file {name} is missing)."
BAD_DETAIL = "The details of this step cannot be read: {reason}"
DETAIL_TEMPLATES = (NO_DETAIL, BAD_DETAIL)


@dataclass(frozen=True)
class StepDetail:
    """A detail file read: its kind (``DETAIL_KINDS``; "" for a file the page does not know), its name, its JSON and
    the arrays of its ``.npz`` (empty without one)."""

    kind: str
    name: str
    doc: dict = field(default_factory=dict)
    arrays: dict = field(default_factory=dict)

    def number(self, key: str) -> Optional[float]:
        return number(self.doc.get(key))

    def numbers(self, key: str) -> list:
        """A list of the doc as floats (NaN where a value is not a number)."""
        values = self.doc.get(key)
        return [math.nan if number(value) is None else number(value) for value in values] \
            if isinstance(values, list) else []


def detail_kind(name: str) -> str:
    return DETAIL_KINDS.get(str(name), "")


def frame_ranges(detail: StepDetail) -> list:
    """The runs of frame numbers read, [(first, last), ...] in their order (empty when the import read every frame
    or does not say)."""
    found = []
    for pair in detail.doc.get("frames_read_ranges") or ():
        if isinstance(pair, (list, tuple)) and len(pair) == 2:
            first, last = number(pair[0]), number(pair[1])
            if first is not None and last is not None and last >= first:
                found.append((int(first), int(last)))
    return found


def detector_image(detail: StepDetail) -> Optional[tuple]:
    """(image, masked share) of the centre and mask detail as floats, or None when the arrays do not fit."""
    image = detail.arrays.get("image")
    if image is None:
        return None
    image = np.asarray(image, dtype=float)
    if image.ndim != 2 or not image.size:
        return None
    masked = np.asarray(detail.arrays.get("masked", np.zeros_like(image)), dtype=float)
    return image, (masked if masked.shape == image.shape else np.zeros_like(image))


def deviations(detail: StepDetail) -> list:
    """(family, median fit deviation) of the networks compared, the best first."""
    found = detail.doc.get("deviations") if isinstance(detail.doc.get("deviations"), dict) else {}
    rows = [(str(name), number(value)) for name, value in found.items() if number(value) is not None]
    return sorted(rows, key=lambda row: row[1])


def joint_iterations(detail: StepDetail) -> list:
    """The joint fit's iterations as (iteration, median fit deviation, seconds), in their order."""
    rows = []
    for item in detail.doc.get("iterations") or ():
        if isinstance(item, dict) and number(item.get("iteration")) is not None:
            rows.append((int(number(item["iteration"])), number(item.get("median")), number(item.get("seconds"))))
    return sorted(rows)


def joint_solution(detail: StepDetail) -> tuple:
    """(times, {quantity: values}) of the solution after the last iteration (``JOINT_KEYS`` that it has; empty
    without an iteration)."""
    iterations = [item for item in detail.doc.get("iterations") or () if isinstance(item, dict)]
    if not iterations:
        return [], {}
    last = max(iterations, key=lambda item: number(item.get("iteration")) or 0)
    rows = sorted((row for row in last.get("windows") or () if isinstance(row, dict) and number(row.get("t")) is not None),
                  key=lambda row: number(row["t"]))
    times = [number(row["t"]) for row in rows]
    values = {key: [math.nan if number(row.get(key)) is None else number(row.get(key)) for row in rows]
              for key in JOINT_KEYS if any(number(row.get(key)) is not None for row in rows)}
    return times, values


__all__ = ["BAD_DETAIL", "DETAIL_KINDS", "DETAIL_TEMPLATES", "JOINT_KEYS", "NO_DETAIL", "RANGES_SHOWN", "StepDetail",
           "detail_kind", "detector_image", "deviations", "frame_ranges", "joint_iterations", "joint_solution"]
