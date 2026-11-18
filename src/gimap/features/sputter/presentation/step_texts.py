"""The words of the step tree in the interface language: the titles of the steps, the package's one-line summaries and
what a step's details say in numbers (``step_tree.py``, ``step_pictures.py``, ``live_steps_panel.py``).

The package writes its summaries in English from a few fixed sentences (``SUMMARY_TEMPLATES``: the numbers, file and
network names are data). A summary made from one of them is said in the interface language with the same values; any
other is shown as the package wrote it.
"""

from __future__ import annotations

from src.gimap.app.presentation.i18n import tr

from ..application import (RANGES_SHOWN, STEP_TITLES, StepDetail, deviations, frame_ranges, joint_iterations)
from .client_texts import shown_number
from .reading_texts import PROCESSES, clock_time, disk_words
from .texts import matched, text

SUMMARY_TEMPLATES = (
    "found in the frames: column {col}, row {row}",
    "given: column {col}, row {row}",
    "{dead} dead, {hot} hot, {shadow} beamstop / holder pixels besides {file}",
    "{file} only (no automatic mask)",
    "{read} of {total} frames in {seconds} s, {bins} bins; disk {disk} s per frame (the disk is the slow part)",
    "{read} of {total} frames in {seconds} s, {bins} bins; disk {disk} s per frame",
    "{read} of {total} frames in {seconds} s",
    "found when the windows are made ({error})",
    "{count} windows of 1 s, {first}-{last} s, every {step} s",
    "{count} windows of 1 s, {first}-{last} s",
    "median fit deviation {deviation} (best candidate), {windows} windows, {seconds} s",
    "{iterations} iterations, median fit deviation {deviation}, metal amount {kappa} x nominal, {minutes} min",
    "{iterations} iterations, median fit deviation {deviation}, {minutes} min",
    "{iterations} iterations with {workers} processes",
    "{windows} windows, {solutions} solutions tried in {seconds} s",
    "{family} fits better ({deviations}); its result is shown first",
    "{family} fits better ({deviations})",
    "mode {mode}: the network's solutions and their time path",
    "median fit deviation {deviations}",
    "{windows} windows",
    "{onset} s ({how})",
)
"""The package's summaries (more particular ones first: the first that fits is taken)."""
SUMMARY_WORDS = ("imported before", "imported before: no frame read again", "run with the network (mode full)",
                 "did not finish (see the log of this step): the network's solutions are shown",
                 "stopped: see the log of this step", "see the log of the import", "not found in the frames")
"""The package's summaries without numbers."""
ONSET_HOW = ("given", "dark first frames: from the lit bare plateau",
             "the rate rises above 1.05 x the 0.3-1.0 s baseline")
"""How the start of the deposition was found (a value of the onset's summary and details)."""
CENTRE_WORDS = "Beam centre ≈ ({col}, {row}) px · detector binned {binning} × {binning}"
MASK_WORDS = ("{frames} frames summed · {masked} of {pixels} pixels left out by the mask (found in these frames: "
              "{dead} dead, {hot} hot, {shadow} of beamstop and holder) · dashed: the sample horizon")
READ_WORDS = "{read} of {total} frames read in {time}"
READ_ALL = "every frame read"
SELECTION_WORDS = "only the frames of {windows} time windows, every {step} s"
MB_WORDS = "{mb} MB read"
RANGES_WORDS = "Frames read: {ranges}"
RANGES_MORE = "{ranges} … ({count} runs in all)"
ONSET_WORDS = "Start of the deposition: {onset} s ({how})"
ONSET_RULE = "the rule alone: {onset} s"
BASELINE_WORDS = "baseline {baseline} (the mean rate {first}–{last} s)"
WINDOWS_WORDS = "{count} time windows, {first}–{last} s, every {step} s"
WINDOWS_MOST = "at most {count}"
CHOSEN_MARK = "{family} {deviation} (chosen)"
JOINT_WORDS = "{done} of at most {total} iterations · median fit deviation {first} → {last} · {time}"
NO_STEPS = ("This result has no record of its steps: it was made by an older analysis package (the result itself is "
            "complete).")
NO_DETAILS = "This step saved no details."
CLICK_STEP = "Click a step marked › to see what it found."
BACK_TO_RUN = "Back to the Analysis"
STEP_TEMPLATES = (*STEP_TITLES.values(), *SUMMARY_TEMPLATES, *SUMMARY_WORDS, *ONSET_HOW, CENTRE_WORDS, MASK_WORDS,
                  READ_WORDS, READ_ALL, SELECTION_WORDS, MB_WORDS, RANGES_WORDS, RANGES_MORE, ONSET_WORDS, ONSET_RULE,
                  BASELINE_WORDS, WINDOWS_WORDS, WINDOWS_MOST, CHOSEN_MARK, JOINT_WORDS, NO_STEPS, NO_DETAILS,
                  CLICK_STEP, BACK_TO_RUN)
"""The texts of the step tree that need Chinese."""


def step_title(node) -> str:
    """A step's title in the interface language (a network's step: the network's name)."""
    if node.family:
        return node.family
    return tr(STEP_TITLES.get(node.id, node.title))


def summary_text(words: str) -> str:
    """A package summary in the interface language (as it is when it was not made from a known sentence)."""
    words = str(words or "")
    if not words:
        return ""
    found = matched((*SUMMARY_WORDS, *SUMMARY_TEMPLATES), words)
    if found is None:
        return words
    template, values = found
    if "how" in values:
        values["how"] = tr(values["how"])
    return text(template, **values) if values else tr(template)


def node_summary(node) -> str:
    return summary_text(node.summary)


def _count(value) -> str:
    return f"{int(value or 0):,}"


def detail_words(detail: StepDetail) -> list:
    """What a step's details say in numbers, a line each (the picture shows the rest)."""
    kind = detail.kind
    if kind == "detector":
        return _detector_words(detail)
    if kind == "reading":
        return _reading_words(detail)
    if kind == "onset":
        lines = [text(ONSET_WORDS, onset=shown_number(detail.number("onset"), 4), how=tr(str(detail.doc.get("how") or "")))]
        rule, onset = detail.number("onset_rule"), detail.number("onset")
        if rule is not None and onset is not None and abs(rule - onset) > 1e-6:
            lines.append(text(ONSET_RULE, onset=shown_number(rule, 4)))
        window = detail.numbers("base_window")
        if detail.number("baseline") is not None and len(window) == 2:
            lines.append(text(BASELINE_WORDS, baseline=shown_number(detail.number("baseline")),
                              first=shown_number(window[0]), last=shown_number(window[1])))
        return lines
    if kind == "windows":
        times = detail.numbers("times")
        if not times:
            return []
        step = detail.number("step") or (times[1] - times[0] if len(times) > 1 else 0.0)
        words = text(WINDOWS_WORDS, count=len(times), first=shown_number(min(times), 4),
                     last=shown_number(max(times), 4), step=shown_number(step))
        most = detail.number("max_windows")
        return [words + (" · " + text(WINDOWS_MOST, count=int(most)) if most else "")]
    if kind == "compare":
        chosen = str(detail.doc.get("chosen") or "")
        return [text(CHOSEN_MARK, family=name, deviation=shown_number(value)) if name == chosen else
                f"{name} {shown_number(value)}" for name, value in deviations(detail)]
    if kind == "joint":
        rows = joint_iterations(detail)
        if not rows:
            return []
        medians = [median for _iteration, median, _seconds in rows if median is not None]
        seconds = rows[-1][2]
        return [text(JOINT_WORDS, done=rows[-1][0], total=int(detail.number("total") or rows[-1][0]),
                     first=shown_number(medians[0]) if medians else "–", last=shown_number(medians[-1]) if medians else "–",
                     time=clock_time(seconds) if seconds is not None else "–")]
    return []


def _detector_words(detail: StepDetail) -> list:
    binning = detail.number("binning") or 1.0
    col, row = detail.number("col"), detail.number("row")
    lines = []
    if col is not None and row is not None:  # binned pixel indices -> GIMaP's pixel coordinates (corners)
        lines.append(text(CENTRE_WORDS, col=f"{(col + 0.5) * binning:.1f}", row=f"{(row + 0.5) * binning:.1f}",
                          binning=f"{binning:g}"))
    pixels, kept = detail.number("pixels"), detail.number("kept")
    if pixels is not None and kept is not None:
        lines.append(text(MASK_WORDS, frames=_count(detail.number("frames_summed")), masked=_count(pixels - kept),
                          pixels=_count(pixels), dead=_count(detail.number("dead")), hot=_count(detail.number("hot")),
                          shadow=_count(detail.number("shadow"))))
    return lines


def _reading_words(detail: StepDetail) -> list:
    read, total = detail.number("frames_read"), detail.number("frames_total")
    parts = []
    if read is not None:
        seconds = detail.number("seconds")
        parts.append(text(READ_WORDS, read=_count(read), total=_count(total if total is not None else read),
                          time=clock_time(seconds) if seconds is not None else "–"))
    workers = detail.number("workers")
    if workers:
        parts.append(text(PROCESSES, count=int(workers)))
    disk = disk_words(detail.number("read_s_per_frame"), detail.number("read_share"))
    if disk:
        parts.append(disk)
    if detail.number("mb_read") is not None:
        parts.append(text(MB_WORDS, mb=f"{detail.number('mb_read'):,.0f}"))
    lines = [" · ".join(parts)] if parts else []
    selection = detail.doc.get("selection") if isinstance(detail.doc.get("selection"), dict) else None
    if selection is not None and selection.get("windows"):
        lines.append(text(SELECTION_WORDS, windows=_count(selection.get("windows")),
                          step=shown_number(selection.get("step"))))
    ranges = frame_ranges(detail)
    if ranges:
        listed = ", ".join(f"{first}–{last}" if last > first else f"{first}" for first, last in ranges[:RANGES_SHOWN])
        words = listed if len(ranges) <= RANGES_SHOWN else text(RANGES_MORE, ranges=listed, count=len(ranges))
        lines.append(text(RANGES_WORDS, ranges=words))
    elif read is not None and total is not None and read >= total:
        lines.append(tr(READ_ALL))
    return lines


__all__ = ["BACK_TO_RUN", "CLICK_STEP", "NO_DETAILS", "NO_STEPS", "ONSET_HOW", "STEP_TEMPLATES", "SUMMARY_TEMPLATES",
           "SUMMARY_WORDS", "detail_words", "node_summary", "step_title", "summary_text"]
