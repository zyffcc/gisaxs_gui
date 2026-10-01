"""Stages and odd frames of the Series map (``shared/series_stages``), and their table and record.

The map's curves are compared in shape (log I, each frame's mean level removed) on the x range every
frame covers; odd frames are found first and left out of the stages. Nothing here changes the map.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from src.gimap.shared.series_stages import SeriesStages, find_stages

from .ports import CurveWriter
from .series_map import SeriesMap

METHOD = (
    "log10 I compared in shape (each frame's mean removed) on the x range nearly every frame covers; odd frames: "
    "a frame matching neither the frames before nor after it (by far more than the series moves per frame there; "
    "a difference in ~1 % of the points: the detector); main components of the other frames; stages: optimal "
    "piecewise-straight segmentation of their scores in frame order, a stage added while it explains ≥ 5 % of "
    "the one-stage residual and more than noise would."
)


def stages_of(series: SeriesMap, x_range: Optional[Sequence[float]] = None) -> SeriesStages:
    """Raises ``ValueError`` for fewer than three frames or too little common x range."""
    return find_stages(series.x, series.image, q_range=x_range)


def stage_summary(stages: SeriesStages) -> str:
    """“3 stages: frames 1–52, 53–170, 171–403 · 3 odd frames”."""
    ranges = ", ".join(f"{first + 1}–{last + 1}" for first, last in stages.ranges())
    count = stages.count
    text = f"{count} stage{'s' if count != 1 else ''}: frames {ranges}" if count > 1 else "One stage: no change of course"
    if stages.odd:
        text += f" · {len(stages.odd)} odd frame{'s' if len(stages.odd) != 1 else ''}"
    return text


def stages_record(series: SeriesMap, stages: SeriesStages) -> dict:
    return {
        "software": "GIMaP", "title": "Stages of a series", "curve": series.curve, "x_label": series.x_label,
        "method": METHOD,
        "x_compared": [float(stages.q.min()), float(stages.q.max())], "points_compared": int(stages.q.size),
        "frames": stages.rows, "stages_shown": stages.count, "stages_suggested": stages.suggested,
        "unexplained_by_stages": {str(k): round(float(v), 5) for k, v in stages.unexplained.items()},
        "stages": [
            {"stage": index + 1, "first_frame": first + 1, "last_frame": last + 1,
             "representative_frame": stages.stage_representatives()[index] + 1}
            for index, (first, last) in enumerate(stages.ranges())
        ],
        "changes": [change.text() for change in stages.stage_changes()],
        "odd_frames": [{"frame": frame.row + 1, "file": series.labels[frame.row], "why": frame.reason,
                        "detector": frame.narrow} for frame in stages.odd],
        "components_explained": [round(float(value), 5) for value in stages.explained[: stages.scores.shape[1]]],
        "half_of_the_change_by_frame": None if stages.half_row is None else stages.half_row + 1,
        "ninety_percent_by_frame": None if stages.ninety_row is None else stages.ninety_row + 1,
    }


def export_series_stages(writer: CurveWriter, series: SeriesMap, stages: SeriesStages, path: Path) -> Path:
    """CSV: one row per frame (stage, odd, why, main components, level); the JSON record next to it."""
    odd = {frame.row: frame for frame in stages.odd}
    components = stages.scores.shape[1]
    header = ["frame", "file", "stage", "odd", "why", "level_log10"] + [f"component_{i + 1}" for i in range(components)]
    rows = []
    for row in range(series.rows):
        frame = odd.get(row)
        rows.append([row + 1, series.labels[row], stages.stage_of(row), bool(frame), frame.reason if frame else "",
                     float(stages.level[row])] + [float(value) for value in stages.scores[row]])
    comments = ["GIMaP Analyze series: stages and odd frames", stage_summary(stages)]
    comments += [change.text() for change in stages.stage_changes()]
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
    "METHOD", "change_curve", "export_series_stages", "odd_frame_refs", "stage_summary", "stages_of", "stages_record",
]
