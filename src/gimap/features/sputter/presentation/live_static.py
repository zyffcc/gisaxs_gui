"""The live view of the analysis of one sample measured after its deposition (a mixin of ``LivePanelMixin``).

Three steps — read the frames of the sample, find its solution, add it to the series — on the same view as an in-situ
run (``live_panel.py``):

* reading the frames: left, the detector summed (the frames of the bare substrate first, when they are given: another
  piece of the same substrate without metal, which gives the substrate part of the image — a sample measured after its
  deposition has no frames from before it), with the pixels the mask leaves out, the direct beam and the sample
  horizon; right, the usable frames summed on the grid of q_y and exit angle; under them, how many frames are usable and
  which are left out and why (the package's words); below, the series as far as it is;
* finding the solution (and adding it): the frame solved last, measured and model image, with its values in the line
  under them; below, the series: its samples analysed before and every frame of this sample solved so far at the
  sample's nominal thickness, joined through their median — the sample's value in the result is the median over its
  frames.
"""

from __future__ import annotations

import re

from src.gimap.app.presentation.i18n import tr

from ..application import FRAME_LABEL, LiveRun
from .client_texts import shown_number
from .texts import text
from .views.sputter_main_view import TILES

CAPTION_KEYS = ("R_nm", "H_nm", "D_nm", "coverage")
"""The quantities said under the images of the frame solved last (those the solution has)."""
TILE_WORDS = {key: (caption, unit) for key, caption, unit, _tip in TILES}
DEVIATION_PART = "fit deviation {deviation}"

STATIC_STEP_NAMES = {"intake": "Read the frames of the sample", "predict": "Find its solution",
                     "report": "Add it to the series"}
STATIC_EXPLAIN = {
    "intake": ("Every frame of the sample is projected onto a grid of q_y and exit angle, without the pixels the mask "
               "leaves out; a frame with the direct or the specular beam not blocked is left out. A sample measured "
               "after its deposition has no frames from before it: the substrate part of its image comes from frames of "
               "another piece of the same substrate without metal (the bare substrate), or, without them, from the "
               "model of Si. Left: the detector summed — red: left out by the mask, +: the direct beam, dashed: the "
               "sample horizon. Right: the usable frames summed. Below: the series so far."),
    "predict": ("For each usable frame the network proposes candidate structures; the physical model draws each as an "
                "image and the one closest to the measured frame is kept. Above: the frame solved last, measured and "
                "model image. Below: the series so far, with every frame of this sample at its nominal thickness; the "
                "sample's value is their median. Quick and Refine solve four of the usable frames, Full every one."),
    "report": "The sample's values — the median over its frames — go into the series, with its report and images.",
}
SAMPLE_LINE = "{usable} of {frames} frames usable"
REFERENCE_LINE = "Bare substrate: {usable} of {frames} frames usable"
SOLVED_LINE = "{solved} of {total} frames solved"
SOLVED_OF_USABLE = "{solved} of {total} frames solved ({total} of the {usable} usable)"
SAMPLE_TITLE = "Usable frames of the sample, summed"
REFERENCE_TITLE = "Usable frames of the bare substrate, summed"
LEFT_OUT = "left out: {name} ({reason})"
BEAM_NOT_BLOCKED = "{count} saturated pixels: direct or specular beam not blocked"
"""The package's reason for a frame left out (``iampred.static``), said in the interface language; any other reason
is shown as the package words it."""
BEAM_NOT_BLOCKED_RE = re.compile(r"^(\d+) saturated pixels: direct or specular beam not blocked$")
REFERENCE_NOTE = "bare substrate: another piece without metal"
NO_REFERENCE_NOTE = "no bare substrate: the film is taken to lie on Si"
FRAME_NAME = "Frame {frame}"
STAGE_NAMES = {"network": "network", "refined": "refined"}
WAITING_FRAMES_SOLVED = "The first frames are being solved…"
STATIC_TEMPLATES = (*STATIC_STEP_NAMES.values(), *STATIC_EXPLAIN.values(), SAMPLE_LINE, REFERENCE_LINE, SOLVED_LINE,
                    SOLVED_OF_USABLE, SAMPLE_TITLE, REFERENCE_TITLE, LEFT_OUT, BEAM_NOT_BLOCKED, REFERENCE_NOTE, NO_REFERENCE_NOTE, FRAME_NAME,
                    *STAGE_NAMES.values(), WAITING_FRAMES_SOLVED)
"""The templates of the static live view that need Chinese."""


def _count(value) -> str:
    return f"{int(value or 0):,}"


class LiveStaticMixin:
    """The static half of the live view: needs what ``LivePanelMixin`` sets up (``_live_sample``, the pictures)."""

    def _static_step_line(self, key: str, run: LiveRun) -> str:
        if key == "intake":
            sample, reference = run.sample, run.reference_sample
            if sample is not None:
                return text(SAMPLE_LINE, usable=_count(sample.number("usable")), frames=_count(sample.number("frames")))
            if reference is not None:
                return text(REFERENCE_LINE, usable=_count(reference.number("usable")),
                            frames=_count(reference.number("frames")))
            return ""
        if key == "predict" and (run.frames_to_solve or run.frames_solved()):
            total = max(run.frames_to_solve, run.frames_solved())
            usable = int(run.sample.number("usable") or 0) if run.sample is not None else 0
            if usable > total:  # the mode solves a few of the usable frames
                return text(SOLVED_OF_USABLE, solved=run.frames_solved(), total=total, usable=_count(usable))
            return text(SOLVED_LINE, solved=run.frames_solved(), total=total)
        return ""

    def _static_result(self, run: LiveRun):
        info = self._live_sample
        return run.static_result(self._live_folder, info.get("series"), info.get("label", ""), info.get("nominal_nm"))

    def _live_draw_static(self, run: LiveRun, step: str, turned: bool, dirty: set) -> None:
        series_changed = turned or bool(dirty & {"frame"})
        latest = run.frame_patterns()
        self.live_plot_host.setCurrentIndex(1)
        if series_changed:  # the series as far as it is; no plot while it has nothing (a first sample, no frame yet)
            result = self._static_result(run)
            mark = None
            if step != "intake" and result is not None and latest is not None:
                shown = f"{self._live_sample.get('label', '')} {latest[1]}"
                mark = next((index for index, window in enumerate(result.windows) if window.label == shown), None)
            self.live_plot_host.setVisible(result is not None)
            self.live_evolution.draw(result, None, mark)
        if step == "intake":
            self.live_picture_host.setCurrentIndex(0)
            if turned or dirty & {"detector", "sample"}:
                only_reference = run.sample is None and run.reference_sample is not None
                self.live_frames.show(run.detector or run.reference_detector, run.sample or run.reference_sample,
                                      frames_title=tr(REFERENCE_TITLE if only_reference else SAMPLE_TITLE))
            self.live_caption.set_full_text(self._sample_caption(run))
            return
        self.live_picture_host.setCurrentIndex(1)
        if turned or "frame" in dirty:
            if latest is None:
                self.live_patterns.show_message(tr(WAITING_FRAMES_SOLVED))
            else:
                self.live_patterns.show(latest[0], latest[1], self._frame_name(latest[1]))
        self.live_caption.set_full_text(self._frame_caption(run, latest))

    @staticmethod
    def _frame_name(label: str) -> str:
        """``frame <n>`` in the interface language."""
        number = label.rsplit(" ", 1)[-1] if label.startswith(FRAME_LABEL.split(" ", 1)[0]) else label
        return text(FRAME_NAME, frame=number)

    def _sample_caption(self, run: LiveRun) -> str:
        """What the frames of the sample are: how many are used, which are left out and why, and its substrate."""
        parts = [tr(REFERENCE_NOTE if self._live_sample.get("reference") else NO_REFERENCE_NOTE)]
        sample = run.sample
        if sample is not None:
            parts.insert(0, text(SAMPLE_LINE, usable=_count(sample.number("usable")),
                                 frames=_count(sample.number("frames"))))
            for pair in (sample.get("excluded") or ())[:3]:
                if isinstance(pair, (list, tuple)) and len(pair) == 2:
                    found = BEAM_NOT_BLOCKED_RE.match(str(pair[1]))
                    reason = text(BEAM_NOT_BLOCKED, count=found.group(1)) if found else str(pair[1])
                    parts.append(text(LEFT_OUT, name=str(pair[0]), reason=reason))
        return " · ".join(parts)

    def _frame_caption(self, run: LiveRun, latest) -> str:
        if latest is None:
            return ""
        _patterns, label, _frame, event = latest
        parts = [self._frame_name(label), tr(STAGE_NAMES.get(str(event.get("stage")), "network"))]
        values = event.get("values") if isinstance(event.get("values"), dict) else {}
        for key in CAPTION_KEYS:
            value = values.get(key)
            if isinstance(value, (int, float)):
                caption, unit = TILE_WORDS[key]
                parts.append(f"{tr(caption)} {shown_number(value, 2 if key == 'coverage' else 3)} {unit}".rstrip())
        deviation = event.number("fit_deviation")
        if deviation is not None:
            parts.append(text(DEVIATION_PART, deviation=shown_number(deviation)))
        return " · ".join(parts)


__all__ = ["LiveStaticMixin", "STATIC_EXPLAIN", "STATIC_STEP_NAMES", "STATIC_TEMPLATES"]
