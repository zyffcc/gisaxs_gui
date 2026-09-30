"""Local file adapters of Fitting: curves, fit results and in-situ curve series."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from .scattering_curve_files import load_xy_any

from ...application.models import (
    CURVE_SUFFIXES,
    DEFAULT_CURVE_PATTERN,
    ExportFitResultRequest,
    DiscoverInSituFramesRequest,
    ExportedFitResult,
    LoadCurveRequest,
    InSituSourceFrame,
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
        paths = sorted(
            (
                path.resolve()
                for path in iterator
                if path.is_file() and path.suffix.lower() in CURVE_SUFFIXES
            ),
            key=lambda path: tuple(
                int(value) if value.isdigit() else value.casefold()
                for value in re.split(r"(\d+)", str(path.relative_to(root)))
            ),
        )
        return tuple(InSituSourceFrame(path=path) for path in paths)


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
        source = Path(request.path).expanduser().resolve()
        if not source.exists():
            raise FileNotFoundError(f"Curve file was not found: {source}")
        if source.suffix.lower() not in {".dat", ".txt"}:
            raise ValueError(f"Unsupported curve format: {source.suffix or '<none>'}")
        loaded = load_xy_any(str(source))
        pixels, observation = read_fit_input_extras(source)
        if pixels is not None and pixels.size != np.asarray(loaded.q).size:
            pixels = None  # rows were dropped as non-finite: counts no longer align
        return CurveData(
            q=loaded.q,
            intensity=loaded.I,
            error=getattr(loaded, "err", None),
            q_source_unit=request.q_source_unit,
            source_path=str(source),
            pixels=pixels,
            observation=observation,
        )


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
