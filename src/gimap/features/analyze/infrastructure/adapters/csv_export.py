"""Write reduced curves as CSV files with a JSON provenance sidecar."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ...domain import Curve


_ASCII_UNITS = (
    ("Å⁻¹", "1/A"),
    ("⁻¹", "^-1"),
    ("Å", "A"),
    ("°", "deg"),
    ("χ", "chi"),
    ("∥", "_par"),
    ("–", "-"),
    ("·", "*"),
)


def ascii_text(text: str) -> str:
    """CSV text any locale can read (Windows tools often assume a legacy code page)."""
    for original, replacement in _ASCII_UNITS:
        text = text.replace(original, replacement)
    return text.encode("ascii", "replace").decode("ascii")


def _slug(text: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in text).strip("_")


FIT_INPUT_MARKER = "# GIMaP Analyze fit input"
SEPARATORS = {".csv": ",", ".txt": "\t", ".dat": " "}
"""The column separator of a text table, by its suffix (comma, tab, space)."""


def _separator(path: Path) -> str:
    return SEPARATORS.get(Path(path).suffix.lower(), ",")


def _cell(text: str, separator: str) -> str:
    """A cell that keeps the columns: no separator inside it (spaces of a space-separated file become _)."""
    return text.replace(" ", "_") if separator == " " else text


def _table(stream, separator: str):
    return csv.writer(stream, delimiter=separator, quoting=csv.QUOTE_MINIMAL)


class CsvCurveWriter:
    """One ``<stem>_<curve>.csv`` per curve plus ``<stem>_analysis.json``.

    The CSV starts with ``#`` comment lines (source, curve, region) so it stays
    readable by numpy.loadtxt, Origin and spreadsheet programs.
    """

    def write_xy(self, curve: Curve, path: Path, metadata: Mapping[str, Any]) -> Path:
        """``x I sigma pixels`` columns (whitespace separated) with ``#`` provenance lines.

        Programs that read two or three columns ignore ``pixels`` (the number
        of detector pixels averaged into each point).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        observation = metadata.get("observation")
        with path.open("w", encoding="ascii", errors="replace", newline="") as stream:
            stream.write(f"{FIT_INPUT_MARKER} (q in 1/A)\n")
            stream.write(f"# source: {ascii_text(str(metadata.get('source_file', '')))}\n")
            stream.write(f"# curve: {ascii_text(curve.title)}\n")
            if observation:
                stream.write(f"# observation: {json.dumps(_jsonable(observation), sort_keys=True)}\n")
            stream.write(f"# columns: {ascii_text(curve.x_label)}  I  sigma  pixels\n")
            for x, intensity, sigma, pixels in zip(curve.x, curve.intensity, curve.sigma, curve.pixels):
                stream.write(f"{x:.8g} {intensity:.8g} {sigma:.8g} {int(pixels)}\n")
        return path

    def write_table(self, path: Path, comments, header, rows) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="ascii", errors="replace", newline="") as stream:
            for line in comments:
                stream.write(f"# {ascii_text(str(line))}\n")
            separator = _separator(path)
            blank = "nan" if separator == " " else ""  # an empty cell would shift a space-separated row
            writer = _table(stream, separator)
            writer.writerow([_cell(ascii_text(str(name)), separator) for name in header])
            for row in rows:
                cells = []
                for value in row:
                    if isinstance(value, float) and value != value:
                        cells.append(blank)
                    elif isinstance(value, float):
                        cells.append(f"{value:.6g}")
                    else:
                        cells.append(_cell(ascii_text(str(value)), separator) or blank)
                writer.writerow(cells)
        return path

    def write_map(self, image, x_axis, y_axis, path: Path, labels: tuple[str, str], metadata: Mapping[str, Any]) -> Path:
        """CSV with the x axis in the first row and the y axis in the first column; empty cells are blank."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="ascii", errors="replace", newline="") as stream:
            title = metadata.get("title") or "GIMaP Analyze q map (q in 1/A): mean intensity per cell, blank = no pixel"
            stream.write(f"# {ascii_text(str(title))}\n")
            if metadata.get("row_names"):  # a series: one line per row
                for name in metadata["row_names"]:
                    stream.write(f"# frame {ascii_text(str(name))}\n")
            else:
                stream.write(f"# source: {ascii_text(str(metadata.get('source_file', '')))}\n")
                stream.write(f"# frame: {metadata.get('frame_index', 0)}\n")
            stream.write(f"# rows: {ascii_text(labels[1])} (first column); columns: {ascii_text(labels[0])} (first row)\n")
            separator = _separator(path)
            blank = "nan" if separator == " " else ""
            writer = _table(stream, separator)
            corner = _cell(f"{ascii_text(labels[1])} \\ {ascii_text(labels[0])}", separator)
            writer.writerow([corner, *(f"{value:.6g}" for value in x_axis)])
            for z, row in zip(y_axis, image):
                writer.writerow([f"{z:.6g}", *(blank if value != value else f"{value:.6g}" for value in row)])
        return path

    def write_record(self, path: Path, record: Mapping[str, Any]) -> Path:
        return write_record(path, record)

    def write(
        self,
        curves: Sequence[Curve],
        destination: Path,
        stem: str,
        metadata: Mapping[str, Any],
        text_format: str = "csv",
    ) -> list[Path]:
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []
        suffix = f".{text_format}" if f".{text_format}" in SEPARATORS else ".csv"
        separator = SEPARATORS[suffix]
        for curve in curves:
            path = destination / f"{stem}_{_slug(curve.key)}{suffix}"
            with path.open("w", encoding="ascii", errors="replace", newline="") as stream:
                stream.write("# GIMaP Analyze export (q in 1/A, angles in deg)\n")
                stream.write(f"# source: {ascii_text(str(metadata.get('source_file', '')))}\n")
                stream.write(f"# frame: {metadata.get('frame_index', 0)}\n")
                stream.write(f"# curve: {ascii_text(curve.title)}\n")
                stream.write(f"# region: {json.dumps(_jsonable(curve.region))}\n")
                writer = _table(stream, separator)
                writer.writerow(
                    [_cell(ascii_text(curve.x_label), separator), _cell(ascii_text(curve.y_label), separator), "sigma", "pixels"]
                )
                for row in zip(curve.x, curve.intensity, curve.sigma, curve.pixels):
                    writer.writerow([f"{row[0]:.8g}", f"{row[1]:.8g}", f"{row[2]:.8g}", int(row[3])])
            written.append(path)
        sidecar = destination / f"{stem}_analysis.json"
        payload = dict(metadata)
        payload["files"] = [path.name for path in written]
        sidecar.write_text(
            json.dumps(_jsonable(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        written.append(sidecar)
        return written


def write_record(path: Path, record: Mapping[str, Any]) -> Path:
    """A JSON record (settings, frames, files) next to the exported files."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(dict(record)), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


__all__ = ["CsvCurveWriter", "FIT_INPUT_MARKER", "ascii_text"]
