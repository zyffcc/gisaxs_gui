"""What the Compare page does: series from curve files or from Analyze's Series map, the comparison,
and its tables with a JSON record."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, Sequence

import numpy as np

from ..domain import Comparison, SeriesData, compare

CURVE_SUFFIXES = (".dat", ".txt", ".csv", ".xy", ".chi")
METHOD = (
    "Curves on the q range every series covers (the first series' points, the others interpolated); log10 I "
    "compared in shape (each frame's mean removed) unless the level is included; odd frames per series (a frame "
    "matching neither the frames before nor after it) left out; main components of all kept frames; end state: "
    "mean shape of the last frames; difference of two end states: RMS of their log I difference, as a percentage; "
    "groups (3+ series): Ward clustering of the end states, split where a merge distance jumps (≥ 2×)."
)


class CurveReader(Protocol):
    def read(self, path: Path) -> tuple[np.ndarray, np.ndarray, str]:
        """``(x, intensity, x_label)`` of a curve file; q in Å⁻¹."""


class TableWriter(Protocol):
    def write_table(self, path: Path, comments: Sequence[str], header: Sequence[str], rows: Sequence[Sequence]) -> Path: ...

    def write_record(self, path: Path, record: dict) -> Path: ...


@dataclass(frozen=True)
class CompareSettings:
    q_range: Optional[tuple] = None
    shape_only: bool = True
    end_frames: int = 10


def natural_key(path) -> list:
    """File names in counting order (frame_2 before frame_10)."""
    return [int(part) if part.isdigit() else part.casefold() for part in re.split(r"(\d+)", Path(path).name)]


def curve_files(folder: Path, pattern: str = "*") -> list[Path]:
    files = [path for path in Path(folder).glob(pattern or "*")
             if path.is_file() and path.suffix.lower() in CURVE_SUFFIXES and not path.name.startswith(".")]
    return sorted(files, key=natural_key)


class CompareService:
    def __init__(self, reader: CurveReader, writer: TableWriter):
        self.reader, self.writer = reader, writer

    # -- series ------------------------------------------------------------------------------

    def series_from_files(self, paths: Sequence[Path], name: Optional[str] = None) -> SeriesData:
        """One series from curve files (in counting order); frames on the first file's grid."""
        paths = sorted((Path(path) for path in paths), key=natural_key)
        if len(paths) < 2:
            raise ValueError("A series needs at least two curve files.")
        curves, failed = [], []
        for path in paths:
            try:
                curves.append((path, *self.reader.read(path)))
            except (OSError, ValueError) as exc:
                failed.append(f"{path.name} ({exc})")
        if len(curves) < 2:
            raise ValueError("Fewer than two curves could be read: " + "; ".join(failed[:3]))
        x_labels = {label for _p, _x, _y, label in curves}
        if len(x_labels) > 1:
            raise ValueError("The files have different x axes: " + ", ".join(sorted(x_labels)))
        grid = np.sort(curves[0][1])
        image = np.full((len(curves), grid.size), np.nan)
        for row, (_path, x, y, _label) in enumerate(curves):
            order = np.argsort(x)
            x, y = x[order], y[order]
            inside = (grid >= x[0]) & (grid <= x[-1])
            image[row, inside] = np.interp(grid[inside], x, y)
        folder = paths[0].parent
        return SeriesData(name or folder.name, grid, image, tuple(path.name for path, *_ in curves),
                          x_label=curves[0][3], source=str(folder), paths=tuple(str(path) for path, *_ in curves))

    def series_from_folder(self, folder: Path, pattern: str = "*") -> SeriesData:
        files = curve_files(Path(folder), pattern)
        if len(files) < 2:
            raise ValueError(f"Fewer than two curve files ({', '.join(CURVE_SUFFIXES)}) in {Path(folder).name}.")
        return self.series_from_files(files, Path(folder).name)

    @staticmethod
    def series_from_map(series_map, name: str) -> SeriesData:
        """Analyze's Series map (``x``, ``image``, ``labels``, ``x_label``)."""
        return SeriesData(name, np.asarray(series_map.x, dtype=float), np.asarray(series_map.image, dtype=float),
                          tuple(series_map.labels), x_label=str(series_map.x_label), source="Analyze")

    # -- the comparison ------------------------------------------------------------------------

    @staticmethod
    def compare(series: Sequence[SeriesData], settings: CompareSettings) -> Comparison:
        return compare(series, q_range=settings.q_range, shape_only=settings.shape_only, end_frames=settings.end_frames)

    # -- tables ----------------------------------------------------------------------------------

    def export_series_table(self, series: Sequence[SeriesData], comparison: Comparison, path: Path) -> Path:
        """CSV: one row per series (frames, odd frames, stages, half / 90 % of the change, group, distances)."""
        names = comparison.names
        header = ["series", "frames", "odd_frames", "stages", "half_of_change_frame", "ninety_percent_frame", "group"]
        header += [f"differs_from_{name}_percent" for name in names]
        rows = []
        for index, (item, result) in enumerate(zip(series, comparison.results)):
            label = lambda row: item.labels[row] if row is not None and row < len(item.labels) else ""  # noqa: E731
            rows.append([result.name, result.rows, len(result.odd), result.stages.count if result.stages else "",
                         label(result.half_row), label(result.ninety_row), comparison.groups[index]]
                        + [round(float(value), 3) for value in comparison.distance[index]])
        comments = ["GIMaP Compare: every series, its odd frames and stages, how fast it changed and where it ended",
                    comparison.group_text()]
        written = self.writer.write_table(Path(path), [line for line in comments if line], header, rows)
        self.writer.write_record(Path(path).with_suffix(".json"), self.record(series, comparison))
        return written

    def export_frames_table(self, series: Sequence[SeriesData], comparison: Comparison, path: Path) -> Path:
        """CSV: one row per frame of every series (odd and why, stage, place along the main changes)."""
        components = comparison.results[0].scores.shape[1] if comparison.results else 0
        header = ["series", "frame", "file", "odd", "why", "stage"] + [f"component_{i + 1}" for i in range(components)]
        rows = []
        for item, result in zip(series, comparison.results):
            odd = {frame.row: frame for frame in result.odd}
            for row in range(result.rows):
                frame = odd.get(row)
                stage = result.stages.stage_of(row) if result.stages else ""
                rows.append([result.name, row + 1, item.labels[row], bool(frame), frame.reason if frame else "", stage]
                            + [float(value) for value in result.scores[row]])
        written = self.writer.write_table(Path(path), ["GIMaP Compare: every frame of every series"], header, rows)
        self.writer.write_record(Path(path).with_suffix(".json"), self.record(series, comparison))
        return written

    @staticmethod
    def record(series: Sequence[SeriesData], comparison: Comparison) -> dict:
        return {
            "software": "GIMaP", "title": "Compare", "method": METHOD, "x_label": comparison.x_label,
            "x_compared": [float(comparison.compared.min()), float(comparison.compared.max())],
            "shape_only": comparison.shape_only, "end_frames": comparison.end_frames,
            "components_explained": [round(float(value), 5) for value in comparison.explained[:5]],
            "groups": comparison.group_text(),
            "series": [
                {"name": result.name, "source": item.source, "frames": result.rows, "group": comparison.groups[index],
                 "odd_frames": [{"frame": frame.row + 1, "file": item.labels[frame.row], "why": frame.reason}
                                for frame in result.odd],
                 "stages": None if result.stages is None else [
                     {"first": item.labels[first], "last": item.labels[last]} for first, last in result.stages.ranges()],
                 "half_of_change": None if result.half_row is None else item.labels[result.half_row],
                 "ninety_percent": None if result.ninety_row is None else item.labels[result.ninety_row],
                 "differs_at_the_end_percent": {other: round(float(value), 3) for other, value in
                                                zip(comparison.names, comparison.distance[index])},
                 "differs_at_the_start_percent": {other: round(float(value), 3) for other, value in
                                                  zip(comparison.names, comparison.start_distance[index])}}
                for index, (item, result) in enumerate(zip(series, comparison.results))
            ],
        }


__all__ = ["CURVE_SUFFIXES", "CompareService", "CompareSettings", "CurveReader", "METHOD", "TableWriter",
           "curve_files", "natural_key"]
