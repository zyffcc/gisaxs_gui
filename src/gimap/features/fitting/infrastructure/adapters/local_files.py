"""Local file adapters of Fitting: curves, fit results and in-situ curve series."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from .curve_columns import column_q_unit, csv_columns
from .curve_units import file_q_unit
from .scattering_curve_files import load_xy_any
from .curve_tables import inspect_curve_table, load_table_curve
from . import nxcansas_curves
from ...application.curve_sources import CurveTablePreview

from ...application.models import (
    CURVE_NOT_FOUND,
    CURVE_SUFFIXES,
    DEFAULT_CURVE_PATTERN,
    ExportFitResultRequest,
    DiscoverInSituFramesRequest,
    ExportedFitResult,
    LoadCurveRequest,
    InSituSourceFrame,
    UNSUPPORTED_CURVE,
)
from ...domain import CurveData


class LocalInSituFrameRepository:
    """Curve files of an in-situ series, in natural (numeric-aware) order."""

    def discover_insitu_frames(
        self, request: DiscoverInSituFramesRequest
    ) -> tuple[InSituSourceFrame, ...]:
        root = Path(request.root).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"In-situ source folder was not found: {root}")
        pattern = request.pattern.strip() or DEFAULT_CURVE_PATTERN
        iterator = root.rglob(pattern) if request.recursive else root.glob(pattern)
        # Ordered by the place in the folder: a file may be a link to one that lies elsewhere.
        found = sorted(
            (path for path in iterator if path.is_file() and path.suffix.lower() in CURVE_SUFFIXES),
            key=lambda path: tuple(
                int(value) if value.isdigit() else value.casefold()
                for value in re.split(r"(\d+)", str(path.relative_to(root)))
            ),
        )
        return tuple(InSituSourceFrame(path=path.resolve()) for path in found)


ANALYZE_FIT_INPUT_MARKER = "# GIMaP Analyze fit input"


def read_fit_input_extras(path: Path) -> tuple[np.ndarray | None, dict]:
    """Pixel counts (4th column) and ``# observation:`` record of a GIMaP Analyze fit input.

    Other files give ``(None, {})``; a malformed row drops the pixel counts,
    never the curve.
    """
    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    if not lines or not lines[0].startswith(ANALYZE_FIT_INPUT_MARKER):
        return None, {}
    observation: dict = {}
    pixels: list[float] = []
    complete = True
    for line in lines:
        text = line.strip()
        if text.startswith("# observation:"):
            try:
                loaded = json.loads(text.split(":", 1)[1])
            except ValueError:
                loaded = {}
            observation = loaded if isinstance(loaded, dict) else {}
            continue
        if not text or text.startswith("#"):
            continue
        parts = text.split()
        try:
            pixels.append(float(parts[3]))
        except (IndexError, ValueError):
            complete = False
    return (np.asarray(pixels, dtype=float) if complete and pixels else None), observation


class LocalCurveRepository:
    def load(self, request: LoadCurveRequest) -> CurveData:
        selection = InSituSourceFrame.from_token(str(request.path))
        source = selection.path.expanduser().resolve()
        if not source.exists():
            raise FileNotFoundError(CURVE_NOT_FOUND.format(path=source))
        if nxcansas_curves.is_nxcansas_file(source) and selection.columns is None:
            # NXcanSAS (HDF5): the curve ``frame_index`` of the file, in the unit its Q states
            return nxcansas_curves.load_curve(
                InSituSourceFrame(source, selection.frame_index), request.q_source_unit)
        if source.suffix.lower() not in CURVE_SUFFIXES:
            raise ValueError(UNSUPPORTED_CURVE)
        columns, chosen_by = selection.columns, "selection"
        if columns is None and source.suffix.lower() == ".csv":
            # A CSV's columns differ by table (Fitting's own “data and fit” has two q columns): by the header's
            # names, never by place alone (``curve_columns``).
            columns, chosen_by = csv_columns(source), "header"
        loaded = load_xy_any(str(source)) if columns is None else load_table_curve(source, columns)
        pixels, observation = read_fit_input_extras(source)
        if pixels is not None and pixels.size != np.asarray(loaded.q).size:
            pixels = None  # rows were dropped as non-finite: counts no longer align
        if columns is not None:
            observation = dict(observation, curve_source={
                "path": str(source), "columns": columns.to_dict(), "chosen_by": chosen_by,
                "sigma_source": "none" if columns.sigma is None else "column",
            })
        return CurveData(
            q=loaded.q,
            intensity=loaded.I,
            error=getattr(loaded, "err", None),
            q_source_unit=request.q_source_unit,
            source_path=InSituSourceFrame(source, selection.frame_index, columns=selection.columns).token,
            pixels=pixels,
            observation=observation,
            rows_left_out=int(getattr(loaded, "meta", {}).get("rows_left_out", 0)),
        )

    def stated_q_unit(self, path) -> str | None:
        """The unit of q the file's header states (``curve_units``; an NXcanSAS file: the units of its Q,
        ``nxcansas_curves``): ``"angstrom"``, ``"nm"`` or ``None``."""
        selection = InSituSourceFrame.from_token(str(path))
        source = selection.path.expanduser()
        if not source.exists():
            raise FileNotFoundError(CURVE_NOT_FOUND.format(path=source.resolve()))  # as ``load`` says it
        if nxcansas_curves.is_nxcansas_file(source):
            return nxcansas_curves.stated_unit(source, selection.frame_index)
        columns = selection.columns
        if columns is None and source.suffix.lower() == ".csv":
            columns = csv_columns(source)
        return file_q_unit(source) if columns is None else column_q_unit(source, columns)

    def inspect_table(self, path) -> CurveTablePreview:
        source = InSituSourceFrame.from_token(str(path)).path.expanduser().resolve()
        if not source.exists():
            raise FileNotFoundError(CURVE_NOT_FOUND.format(path=source))
        if nxcansas_curves.is_nxcansas_file(source):
            return nxcansas_curves.preview(source)  # every curve of the file, one source each
        if source.suffix.lower() not in CURVE_SUFFIXES:
            raise ValueError(UNSUPPORTED_CURVE)
        return inspect_curve_table(source)


class LocalFitResultRepository:
    def export(self, request: ExportFitResultRequest) -> ExportedFitResult:
        target = Path(request.path).expanduser()
        q = np.asarray(request.q, dtype=float).reshape(-1)
        intensity = np.asarray(request.intensity, dtype=float).reshape(-1)
        count = min(q.size, intensity.size)
        if count == 0:
            raise ValueError("Fit result contains no rows")
        combined = np.column_stack([q[:count], intensity[:count]])
        delimiter = "," if target.suffix.lower() == ".csv" else "\t"
        column_header = f"{request.x_column_name}{delimiter}{request.y_column_name}"
        with target.open("w", encoding="utf-8", newline="\n") as handle:
            if request.header_lines:
                handle.write("\n".join(request.header_lines) + "\n")
            handle.write(column_header + "\n")
            np.savetxt(handle, combined, delimiter=delimiter, fmt="%.6e")
        return ExportedFitResult(path=target.resolve(), row_count=count, delimiter=delimiter)
