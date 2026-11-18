"""What a step of an Analyze run found, drawn (Matplotlib) with its numbers above (the live view and Details ▸ Steps).

The details are read when the user opens the step and dropped when it is closed (``clear``): the picture keeps only
the step shown.

* the beam centre and the pixel mask: the detector summed (log₁₀ of the counts, binned), the share of each binned
  pixel the mask leaves out tinted red, the direct beam marked + and the sample horizon dashed;
* reading the frames: the runs of frames read along the frame number;
* the start of the deposition: the count rate of the onset band over time, its baseline (dashed) over the window it
  is taken from (shaded), and the start (a vertical line; the rule's own, when it differs, dotted);
* the time windows: a tick at each window's time;
* the networks compared: the median fit deviation of each (lower is better), the chosen one in the accent colour;
* the joint fit: the median fit deviation after each iteration, and the solution after the last one — R, H, D and
  the continuous layer t_b (nm) with the coverage on its own axis.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QLabel

from src.gimap.app.presentation.i18n import tr, trf

from ..application import StepDetail, detector_image, deviations, frame_ranges, joint_iterations, joint_solution
from .evolution_view import SIZE_SERIES
from .live_pictures import BEAM_COLOUR, MASK_ALPHA, MASK_COLOUR, TRACE_COLOUR, _canvas, _finish
from .pattern_view import COLOUR_MAP
from .step_texts import NO_DETAILS, detail_words

FRAME_AXIS = "Frame number"
TIME_AXIS = "Time in the scan (s)"
RATE_AXIS = "Count rate (onset band)"
BASELINE = "Baseline"
ONSET = "Start of the deposition"
ONSET_RULE_LABEL = "The rule alone"
DEVIATION_AXIS = "Median fit deviation"
ITERATION_AXIS = "Iteration"
SIZE_AXIS = "Size (nm)"
COVERAGE = "Coverage"
LAST_SOLUTION = "Solution after iteration {iteration}"
PICTURE_TEMPLATES = (FRAME_AXIS, TIME_AXIS, RATE_AXIS, BASELINE, ONSET, ONSET_RULE_LABEL, DEVIATION_AXIS, ITERATION_AXIS,
                     SIZE_AXIS, COVERAGE, LAST_SOLUTION)
ONSET_COLOUR = "#e08a1e"
CHOSEN_COLOUR = "#2563eb"
OTHER_COLOUR = "#8a8a8a"
COVERAGE_COLOUR = "#17a2b8"
SIZE_COLOURS = {key: colour for key, _name, colour in SIZE_SERIES}
SIZE_NAMES = {key: name for key, name, _colour in SIZE_SERIES}


class StepDetailPicture:
    """A line of numbers and a picture in ``layout``; ``show(detail)`` draws a step's details, ``clear()`` drops them."""

    def __init__(self, layout):
        self.words = QLabel("", None)
        self.words.setObjectName("sputterStepWords")
        self.words.setWordWrap(True)
        self.words.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.words)
        self.figure, self.canvas = _canvas(layout, (8, 3.4), 200)
        self.detail: Optional[StepDetail] = None
        self.kind = ""

    def show(self, detail: Optional[StepDetail], message: str = "") -> None:
        """Draw ``detail`` (None: only ``message``)."""
        self.detail = detail
        self.kind = detail.kind if detail is not None else ""
        lines = detail_words(detail) if detail is not None else []
        self.words.setText("\n".join(lines) if lines else (message or tr(NO_DETAILS)))
        self._draw()

    def clear(self) -> None:
        """The details go (their arrays with them); the picture is empty."""
        self.detail, self.kind = None, ""
        self.words.setText("")
        if self.figure is not None:
            self.figure.clear()
            self.canvas.draw_idle()

    def redraw(self) -> None:
        if self.detail is not None:
            self.show(self.detail)

    # -- drawing --------------------------------------------------------------------------------------------------
    def _draw(self) -> None:
        if self.canvas is None:
            return
        figure = self.figure
        figure.clear()
        texts = []
        detail = self.detail
        if detail is not None:
            draw = {"detector": self._detector, "reading": self._reading, "onset": self._onset,
                    "windows": self._windows, "compare": self._compare, "joint": self._joint}.get(detail.kind)
            if draw is not None:
                texts = draw(figure, detail) or []
        _finish(figure, self.canvas, texts)

    @staticmethod
    def _detector(figure, detail: StepDetail) -> list:
        found = detector_image(detail)
        if found is None:
            return []
        image, masked = found
        axes = figure.add_axes([0.02, 0.02, 0.96, 0.96])
        axes.set_axis_off()
        finite = image[np.isfinite(image)]
        low, high = np.percentile(finite, (1.0, 99.8)) if finite.size else (0.0, 1.0)
        axes.imshow(np.ma.masked_invalid(image), cmap=COLOUR_MAP, vmin=low, vmax=max(high, low + 1e-6),
                    interpolation="nearest", origin="upper")
        tint = np.zeros(image.shape + (4,))
        tint[..., :3] = MASK_COLOUR
        tint[..., 3] = np.clip(np.nan_to_num(masked), 0.0, 1.0) * MASK_ALPHA
        axes.imshow(tint, interpolation="nearest", origin="upper")
        col, row, horizon = (detail.number(key) for key in ("col", "row", "horizon_row"))
        if horizon is not None:
            axes.axhline(horizon, color="white", linewidth=0.9, linestyle="--", alpha=0.8)
        if col is not None and row is not None:
            axes.plot([col], [row], "+", color=BEAM_COLOUR, markersize=16, markeredgewidth=2.2)
        axes.set_xlim(-0.5, image.shape[1] - 0.5)
        axes.set_ylim(image.shape[0] - 0.5, -0.5)
        return []

    @staticmethod
    def _reading(figure, detail: StepDetail) -> list:
        ranges = frame_ranges(detail)
        total = detail.number("frames_total")
        if not ranges and not total:
            return []
        axes = figure.add_axes([0.06, 0.32, 0.9, 0.4])
        spans = ranges or [(0, int(total) - 1)]
        axes.broken_barh([(first, last - first + 1) for first, last in spans], (0.0, 1.0), color=TRACE_COLOUR)
        end = max(int(total or 0), spans[-1][1] + 1)
        axes.set_xlim(0, max(1, end))
        axes.set_ylim(0, 1)
        axes.set_yticks([])
        axes.set_xlabel(tr(FRAME_AXIS))
        return [axes.xaxis.label]

    @staticmethod
    def _onset(figure, detail: StepDetail) -> list:
        times, rate = detail.numbers("times"), detail.numbers("rate")
        if not times or len(times) != len(rate):
            return []
        axes = figure.add_axes([0.1, 0.17, 0.86, 0.75])
        axes.plot(times, rate, "-", color=TRACE_COLOUR, linewidth=1.2)
        window = detail.numbers("base_window")
        if len(window) == 2 and all(math.isfinite(value) for value in window):
            axes.axvspan(window[0], window[1], color=OTHER_COLOUR, alpha=0.15, linewidth=0)
        if detail.number("baseline") is not None:
            axes.axhline(detail.number("baseline"), color=OTHER_COLOUR, linestyle="--", linewidth=1.0,
                         label=tr(BASELINE))
        onset, rule = detail.number("onset"), detail.number("onset_rule")
        if onset is not None:
            axes.axvline(onset, color=ONSET_COLOUR, linewidth=1.4, label=tr(ONSET))
        if rule is not None and (onset is None or abs(rule - onset) > 1e-6):
            axes.axvline(rule, color=ONSET_COLOUR, linewidth=1.0, linestyle=":", label=tr(ONSET_RULE_LABEL))
        axes.set_xlabel(tr(TIME_AXIS))
        axes.set_ylabel(tr(RATE_AXIS))
        axes.grid(True, alpha=0.25)
        legend = axes.legend(loc="upper left", fontsize=8, frameon=False)
        return [axes.xaxis.label, axes.yaxis.label, *legend.get_texts()]

    @staticmethod
    def _windows(figure, detail: StepDetail) -> list:
        times = [value for value in detail.numbers("times") if math.isfinite(value)]
        if not times:
            return []
        axes = figure.add_axes([0.06, 0.35, 0.9, 0.3])
        axes.eventplot([times], colors=[TRACE_COLOUR], lineoffsets=0.5, linelengths=0.8)
        low, high = min(times), max(times)
        margin = max(1.0, 0.03 * (high - low))
        axes.set_xlim(low - margin, high + margin)
        axes.set_ylim(0, 1)
        axes.set_yticks([])
        axes.set_xlabel(tr(TIME_AXIS))
        return [axes.xaxis.label]

    @staticmethod
    def _compare(figure, detail: StepDetail) -> list:
        rows = deviations(detail)
        if not rows:
            return []
        chosen = str(detail.doc.get("chosen") or "")
        axes = figure.add_axes([0.22, 0.2, 0.72, 0.72])
        names = [name for name, _value in rows][::-1]
        values = [value for _name, value in rows][::-1]
        axes.barh(range(len(rows)), values, color=[CHOSEN_COLOUR if name == chosen else OTHER_COLOUR for name in names])
        axes.set_yticks(range(len(rows)))
        axes.set_yticklabels(names)
        axes.set_xlabel(tr(DEVIATION_AXIS))
        axes.grid(True, axis="x", alpha=0.25)
        return [axes.xaxis.label]

    @staticmethod
    def _joint(figure, detail: StepDetail) -> list:
        rows = joint_iterations(detail)
        if not rows:
            return []
        left = figure.add_axes([0.08, 0.17, 0.3, 0.7])
        left.plot([row[0] for row in rows], [math.nan if row[1] is None else row[1] for row in rows], "-o",
                  color=TRACE_COLOUR, markersize=3.5)
        left.set_xlabel(tr(ITERATION_AXIS))
        left.set_ylabel(tr(DEVIATION_AXIS))
        left.grid(True, alpha=0.25)
        texts = [left.xaxis.label, left.yaxis.label]
        times, values = joint_solution(detail)
        if not times:
            return texts
        right = figure.add_axes([0.5, 0.17, 0.38, 0.7])
        for key in ("R_nm", "H_nm", "D_nm", "base_nm"):
            if key in values:
                right.plot(times, values[key], "-o", color=SIZE_COLOURS[key], markersize=3, linewidth=1.3,
                           label=tr(SIZE_NAMES[key]))
        right.set_xlabel(tr(TIME_AXIS))
        right.set_ylabel(tr(SIZE_AXIS))
        right.grid(True, alpha=0.25)
        texts += [right.xaxis.label, right.yaxis.label]
        handles, names = right.get_legend_handles_labels()
        if "coverage" in values:
            share = right.twinx()
            share.plot(times, values["coverage"], "--", color=COVERAGE_COLOUR, linewidth=1.2, label=tr(COVERAGE))
            share.set_ylim(0.0, max(1.0, max((v for v in values["coverage"] if math.isfinite(v)), default=1.0)))
            share.set_ylabel(tr(COVERAGE))
            texts.append(share.yaxis.label)
            more, more_names = share.get_legend_handles_labels()
            handles, names = handles + more, names + more_names
        right.set_xlabel(f"{tr(TIME_AXIS)} · {trf(LAST_SOLUTION, iteration=rows[-1][0])}")
        legend = right.legend(handles, names, loc="lower left", bbox_to_anchor=(0.0, 1.01), ncol=3, fontsize=7,
                              frameon=False)  # above the plot, never over a line
        return texts + list(legend.get_texts())


__all__ = ["PICTURE_TEMPLATES", "StepDetailPicture"]
