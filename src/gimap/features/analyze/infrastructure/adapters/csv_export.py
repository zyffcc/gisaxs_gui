"""Write reduced curves as CSV files with a JSON provenance sidecar.

What the text files hold (``export_text``):

* The symbols the application itself writes — units, axis names, mathematical signs, typographic
  punctuation — in a plain spelling: ``q (1/A)``, ``chi (deg)``, ``2theta (deg)``, ``3 x its error``,
  ``+/-``, ``chi^2_r``. A usual export is therefore plain ASCII, which every program reads alike in every
  locale (a spreadsheet program takes a CSV for the legacy code page of the computer).
* Any other character — a region, file or folder named in another script — as it is, in UTF-8. It used to
  become “?”. No byte-order mark is written: in front of the first ``#`` it would make that line data for
  numpy.loadtxt, gnuplot, Fitting's check of a fit input and every reader that skips ``#`` lines.
* The path of the source file (the ``# source:`` line) exactly as it is, symbols included: it says where the
  numbers come from, and a folder “10°C” spelled “10degC” is no folder.

Which frame the numbers are of follows the source (``_frame_lines``): ``# frame: 4 of 12 (index 3)`` — counted
from 1 as on screen, with the 0-based index the file names carry —, ``# sum of …`` for summed frames and
``# files: …`` when the frame was read from several files (the module files of a detector).

A Fitting input is written under another name and renamed when it is complete: a program that watches the
folder for ``*_fit_input.dat`` never reads half a curve.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.gimap.shared.export_text import export_text

from ...domain import Curve


def _text_file(path: Path):
    """``path`` opened for writing as UTF-8. ``errors="replace"``: a file name the system could not decode
    (it reaches Python as a lone surrogate, which UTF-8 cannot hold) is written with “?” instead of ending
    the export."""
    return path.open("w", encoding="utf-8", errors="replace", newline="")


def _slug(text: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in text).strip("_")


FIT_INPUT_MARKER = "# GIMaP Analyze fit input"
SEPARATORS = {".csv": ",", ".txt": "\t", ".dat": " "}
"""The column separator of a text table, by its suffix (comma, tab, space)."""


def _separator(path: Path) -> str:
    return SEPARATORS.get(Path(path).suffix.lower(), ",")


def _source_line(metadata: Mapping[str, Any]) -> str:
    """``# source: <path>``: the path as it is (module docstring), not :func:`export_text` of it."""
    return f"# source: {metadata.get('source_file', '')}\n"


NAMES_IN_A_LINE = 12
"""A header line names at most this many files; the JSON record beside the file has every one."""


def _names(names: Sequence[Any], limit: int = NAMES_IN_A_LINE) -> str:
    names = [Path(str(name)).name for name in names]
    if len(names) <= limit:
        return ", ".join(names)
    head = limit // 2
    return ", ".join(names[:head]) + f", ... ({len(names) - 2 * head} more) ..., " + ", ".join(names[-head:])


def _frame_lines(metadata: Mapping[str, Any]) -> list[str]:
    """The ``#`` lines that say which frame of its source a file holds (module docstring); none of them for a
    record that names no frame."""
    lines: list[str] = []
    if "frame_index" not in metadata:
        return lines
    index, count = int(metadata.get("frame_index") or 0), int(metadata.get("frame_count") or 0)
    lines.append(f"# frame: {index + 1}{f' of {count}' if count else ''} (index {index})\n")
    summed = metadata.get("summed_frames") or ()
    if isinstance(summed, (list, tuple)) and summed:
        source = str(metadata.get("source_file", ""))
        frames = [(source, index), *((str(item.get("file", "")), int(item.get("frame_index") or 0)) for item in summed)]
        if all(path == source for path, _index in frames):  # consecutive frames of one source
            first, last = frames[0][1], frames[-1][1]
            lines.append(f"# sum of {len(frames)} frames: {first + 1} to {last + 1}{f' of {count}' if count else ''} "
                         f"(index {first} to {last}); intensities are sums over these frames\n")
        else:
            lines.append(f"# sum of {len(frames)} frames: {_names([path for path, _index in frames])}; "
                         "intensities are sums over these frames\n")
    files = metadata.get("source_files") or ()
    if len(files) > 1:
        lines.append(f"# files: this frame is read from {len(files)} files: {_names(files)}\n")
    return lines


def _cell(text: str, separator: str) -> str:
    """A cell that keeps the columns: no separator inside it. The programs that read a space-separated file
    split it at any white space — also the ideographic space (U+3000) and the no-break space (U+00A0) a name
    can hold, and a tab — so each of them becomes ``_`` there."""
    if separator != " ":
        return text
    return "".join("_" if character.isspace() else character for character in text)


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
        unfinished = path.with_name(f".{path.name}.part")  # no ``*_fit_input.dat``: a watched folder skips it
        try:
            with _text_file(unfinished) as stream:
                stream.write(f"{FIT_INPUT_MARKER} (q in 1/A)\n")
                stream.write(_source_line(metadata))
                stream.writelines(_frame_lines(metadata))
                stream.write(f"# curve: {export_text(curve.title)}\n")
                if observation:
                    stream.write(f"# observation: {json.dumps(_jsonable(observation), sort_keys=True)}\n")
                stream.write(f"# columns: {export_text(curve.x_label)}  I  sigma  pixels\n")
                for x, intensity, sigma, pixels in zip(curve.x, curve.intensity, curve.sigma, curve.pixels):
                    stream.write(f"{x:.8g} {intensity:.8g} {sigma:.8g} {int(pixels)}\n")
            os.replace(unfinished, path)  # complete, or not there at all
        finally:
            if unfinished.exists():
                unfinished.unlink()
        return path

    def write_table(self, path: Path, comments, header, rows) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with _text_file(path) as stream:
            for line in comments:
                stream.write(f"# {export_text(str(line))}\n")
            separator = _separator(path)
            blank = "nan" if separator == " " else ""  # an empty cell would shift a space-separated row
            writer = _table(stream, separator)
            writer.writerow([_cell(export_text(str(name)), separator) for name in header])
            for row in rows:
                cells = []
                for value in row:
                    if isinstance(value, float) and value != value:
                        cells.append(blank)
                    elif isinstance(value, float):
                        cells.append(f"{value:.8g}")  # as the files of one frame: a table loses no digit of them
                    else:
                        cells.append(_cell(export_text(str(value)), separator) or blank)
                writer.writerow(cells)
        return path

    def write_map(self, image, x_axis, y_axis, path: Path, labels: tuple[str, str], metadata: Mapping[str, Any]) -> Path:
        """CSV with the x axis in the first row and the y axis in the first column; empty cells are blank."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with _text_file(path) as stream:
            title = metadata.get("title") or "GIMaP Analyze q map (q in 1/A): mean intensity per cell, blank = no pixel"
            stream.write(f"# {export_text(str(title))}\n")
            if metadata.get("row_names"):  # a series: one line per row
                for name in metadata["row_names"]:
                    stream.write(f"# frame {export_text(str(name))}\n")
            else:
                stream.write(_source_line(metadata))
                stream.writelines(_frame_lines({"frame_index": 0, **metadata}))
            stream.write(f"# rows: {export_text(labels[1])} (first column); columns: {export_text(labels[0])} (first row)\n")
            separator = _separator(path)
            blank = "nan" if separator == " " else ""
            writer = _table(stream, separator)
            corner = _cell(f"{export_text(labels[1])} \\ {export_text(labels[0])}", separator)
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
            with _text_file(path) as stream:
                stream.write("# GIMaP Analyze export (q in 1/A, angles in deg)\n")
                stream.write(_source_line(metadata))
                stream.writelines(_frame_lines({"frame_index": 0, **metadata}))
                stream.write(f"# curve: {export_text(curve.title)}\n")
                stream.write(f"# region: {json.dumps(_jsonable(curve.region))}\n")
                writer = _table(stream, separator)
                writer.writerow(
                    [_cell(export_text(curve.x_label), separator), _cell(export_text(curve.y_label), separator), "sigma", "pixels"]
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


__all__ = ["CsvCurveWriter", "FIT_INPUT_MARKER", "export_text"]
