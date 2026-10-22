"""Stages and odd frames of the Series map (``shared/series_stages``), and their table and record.

The map's curves are compared in shape (log I, each frame's mean level removed) on the x range every
frame covers; odd frames are found first and left out of the stages. Nothing here changes the map.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from src.gimap.shared.series_stages import SHORTEST, SeriesStages, find_stages

from .ports import CurveWriter
from .series_map import SeriesMap, axis_name, series_header, view_words

STAGES_LEAST_FRAMES = SHORTEST
"""Stages are looked for in a map of at least this many frames (a stage is at least that long)."""

METHOD = (
    "log10 I compared in shape (each frame's mean removed) on the x range nearly every frame covers; odd frames: "
    "a frame matching neither the 3 frames before nor the 3 after it, by far more than the series moves per frame "
    "there — over the whole curve (> 5×), or in the 1 % of the points where it differs most from both sides in the "
    "same direction (> 8×: a spike); a difference in ~1 % of the points: the detector; odd frames do not count as "
    "neighbours; main components of the other frames; stages: optimal "
    "piecewise-straight segmentation of their scores in frame order, a stage added while it explains ≥ 5 % of "
    "the one-stage residual and more than noise would."
)


def stages_of(series: SeriesMap, x_range: Optional[Sequence[float]] = None) -> SeriesStages:
    """Raises ``ValueError`` for fewer than ``STAGES_LEAST_FRAMES`` frames or too little common x range.
    ``x_range``: compare only this part of the map's x (``None``: all of it)."""
    return find_stages(series.x, series.image, q_range=x_range)


def stage_summary(stages: SeriesStages, series: Optional[SeriesMap] = None) -> str:
    """“3 stages: frames 1–40, 41–95, 96–180 · 2 odd frames” (frames as ``series`` numbers them)."""
    number = (lambda row: row + 1) if series is None else series.number
    ranges = ", ".join(f"{number(first)}–{number(last)}" for first, last in stages.ranges())
    count = stages.count
    text = f"{count} stage{'s' if count != 1 else ''}: frames {ranges}" if count > 1 else "One stage: no change of course"
    if stages.odd:
        text += f" · {len(stages.odd)} odd frame{'s' if len(stages.odd) != 1 else ''}"
    return text


def change_words(change, axis: str = "q") -> str:
    """What grows and falls from one stage to the next, on the map's own x (``axis``: q, qy, χ …) — the words of
    the screen (``stage_text.change_text``), in English for the records."""
    grows = ", ".join(f"{x:.4g} (+{percent:.0f} %)" for x, percent in change.rises)
    drops = ", ".join(f"{x:.4g} ({percent:.0f} %)" for x, percent in change.falls)
    parts = [f"grows most at {axis} {grows}" if grows else "", f"falls most at {axis} {drops}" if drops else ""]
    return f"Stage {change.stage - 1} → {change.stage}: " + "; ".join(part for part in parts if part)


def odd_words(frame, axis: str = "q") -> str:
    """Why a frame is odd, on the map's own x (``OddFrame.reason`` says q)."""
    if frame.narrow:
        return f"differs only near {axis} {frame.q:.4g}: probably the detector, not the sample"
    return f"differs from the frames before and after it ({frame.z:.0f}× their step), most near {axis} {frame.q:.4g}"


def stages_record(series: SeriesMap, stages: SeriesStages) -> dict:
    """Frames are numbered as on the map's frame axis (``SeriesMap.numbers``: the frame in its file, or its
    place in the list — not the row, which differs for a map of every n-th frame)."""
    number = series.number
    built = series.built_with
    axis = axis_name(series)
    return {
        "software": "GIMaP", "title": "Stages of a series", "curve": series.curve, "curve_title": series.title,
        "x_label": series.x_label, "halves": series.view, "halves_in_words": view_words(series),
        "series": built.get("name"), "cut_of_the_series": built.get("cut"), "cut_in_words": built.get("cut_text"),
        "method": METHOD,
        "x_compared": [float(stages.q.min()), float(stages.q.max())], "points_compared": int(stages.q.size),
        "frames": stages.rows, "frame_numbers": list(series.numbers),
        "stages_shown": stages.count, "stages_suggested": stages.suggested,
        "unexplained_by_stages": {str(k): round(float(v), 5) for k, v in stages.unexplained.items()},
        "stages": [
            {"stage": index + 1, "first_frame": number(first), "last_frame": number(last),
             "representative_frame": number(stages.stage_representatives()[index])}
            for index, (first, last) in enumerate(stages.ranges())
        ],
        "changes": [change_words(change, axis) for change in stages.stage_changes()],
        "odd_frames": [{"frame": number(frame.row), "file": series.labels[frame.row], "why": odd_words(frame, axis),
                        "detector": frame.narrow} for frame in stages.odd],
        "components_explained": [round(float(value), 5) for value in stages.explained[: stages.scores.shape[1]]],
        "half_of_the_change_by_frame": None if stages.half_row is None else number(stages.half_row),
        "ninety_percent_by_frame": None if stages.ninety_row is None else number(stages.ninety_row),
        "settings": built.get("settings"),
    }


def export_series_stages(writer: CurveWriter, series: SeriesMap, stages: SeriesStages, path: Path) -> Path:
    """CSV: one row per frame (stage, odd, why, main components, level); the JSON record next to it."""
    odd = {frame.row: frame for frame in stages.odd}
    axis = axis_name(series)
    components = stages.scores.shape[1]
    header = ["frame", "file", "stage", "odd", "why", "level_log10"] + [f"component_{i + 1}" for i in range(components)]
    rows = []
    for row in range(series.rows):
        frame = odd.get(row)
        rows.append([series.number(row), series.labels[row], stages.stage_of(row), bool(frame),
                     odd_words(frame, axis) if frame else "", float(stages.level[row])]
                    + [float(value) for value in stages.scores[row]])
    comments = ["GIMaP Analyze series: stages and odd frames", *series_header(series),
                f"compared: {series.x_label} from {float(stages.q.min()):.6g} to {float(stages.q.max()):.6g} "
                f"({int(stages.q.size)} points)", stage_summary(stages, series)]
    comments += [change_words(change, axis) for change in stages.stage_changes()]
    written = writer.write_table(Path(path), comments, header, rows)
    writer.write_record(Path(path).with_suffix(".json"), stages_record(series, stages))
    return written


def odd_frame_refs(series: SeriesMap, stages: Optional[SeriesStages]) -> set:
    """``(path, frame_index)`` of the odd frames (to leave them out of a Batch Export)."""
    if stages is None or not series.refs:
        return set()
    return {(str(series.refs[frame.row][0]).casefold(), int(series.refs[frame.row][1])) for frame in stages.odd
            if frame.row < len(series.refs)}


def change_curve(stages: SeriesStages) -> np.ndarray:
    """How far every frame has gone along the main direction of change (the first component)."""
    return np.asarray(stages.scores[:, 0], dtype=float)


__all__ = [
    "METHOD", "STAGES_LEAST_FRAMES", "change_curve", "change_words", "export_series_stages", "odd_frame_refs",
    "odd_words", "stage_summary", "stages_of", "stages_record",
]
