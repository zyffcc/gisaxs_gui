"""One frame of a batch as a job that can run in another process, and how many run at once.

A batch reduces every frame, writes its files and returns what the rest of the batch needs — a
``FrameOutcome``: small things only (the chosen curves for the tables, the I(q) windows of the
peaks to fit, the curve shown live), never the frame itself, so several frames can be reduced in
parallel worker processes. The outcomes are then *folded* into the batch in frame order in the
application (tables, fits whose start is the previous frame's result, the live map).

**How many at once** (``frames_at_once``): *gentle* is one frame at a time in the application
itself (no extra processes, as before); *balanced* uses about a quarter of the physical cores (at
most 4); *fast* all but two (at most 8). Never more than half of the free memory allows — a frame
needs about ``BYTES_PER_PIXEL`` bytes per detector pixel while it is reduced (measured: 1.36 GB for
a Lambda 9M frame, 14.9 Mpx) — and never more than the frames there are. A short batch (less than
``MIN_PARALLEL_SECONDS`` of work one frame at a time, about ``SECONDS_PER_PIXEL`` per pixel and
frame) stays in the application: starting worker processes would take longer than it saves.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Optional

from ..domain import GISAXS, Curve, SeriesCorrection
from .batch_export import BALANCED, FAST, FOLDERS, GENTLE, SPEEDS, BatchChoices, chosen_curves, export_frame
from .batch_fit import FIT_MODEL, FIT_PEAKS, peak_inputs
from .models import AnalysisRequest, FrameAnalysis
from .ports import FigureWriter
from .series import CorrectSeriesFrame
from .series_map import row_label, series_row
from .use_cases import analysis_metadata, export_stem, fit_input_curve

BYTES_PER_PIXEL = 100
WORKER_BASE_BYTES = 200_000_000
"""Python, numpy, h5py and matplotlib of one worker process."""
MEMORY_SHARE = 0.5
"""The part of the free memory a batch may use."""
SECONDS_PER_PIXEL = 1.4e-7
"""Reduction time per pixel of one frame (measured: 2 s for a Lambda 9M frame)."""
MIN_PARALLEL_SECONDS = 20.0


def frames_at_once(speed: str, *, cores: int, free_bytes: Optional[int], frame_pixels: int, frames: int) -> int:
    """How many frames a batch reduces at the same time (1: in the application, one after the other)."""
    if speed == GENTLE or frames <= 2 or frames * max(1, int(frame_pixels)) * SECONDS_PER_PIXEL < MIN_PARALLEL_SECONDS:
        return 1
    cores = max(1, int(cores))
    wanted = min(4, max(2, cores // 4)) if speed == BALANCED else min(8, max(2, cores - 2))
    if free_bytes is not None:
        per_frame = BYTES_PER_PIXEL * max(1, int(frame_pixels)) + WORKER_BASE_BYTES
        wanted = min(wanted, int(MEMORY_SHARE * free_bytes // per_frame))
    return max(1, min(wanted, int(frames), cores))


@dataclass(frozen=True)
class FrameJob:
    """One frame of a batch (picklable)."""

    index: int
    request: AnalysisRequest
    choices: BatchChoices
    destination: Path
    series: Optional[SeriesCorrection] = None
    series_factor: Optional[float] = None
    """The normalisation factor of the first frame (a series normalised to it)."""
    live_key: Optional[str] = None
    """The curve shown live (a row of the growing series map)."""
    display: Optional[dict] = None
    """Pictures: ``{"log", "colormap", "detector", "qmap"}`` — the limits in intensity units, or ``None``
    for each frame's own (``write_frame_images``)."""


@dataclass
class FrameOutcome:
    """What one frame gave, small enough to cross processes."""

    index: int
    label: str
    stem: str
    written: list = field(default_factory=list)
    curves: list = field(default_factory=list)
    """The chosen curves (for the tables of every frame)."""
    live_row: Optional[tuple] = None
    """``(x, intensity, x_label, label, (path, frame_index))`` of the live curve."""
    peak_inputs: list = field(default_factory=list)
    model_curve: Optional[Curve] = None
    metadata: dict = field(default_factory=dict)
    series_info: dict = field(default_factory=dict)
    seconds: float = 0.0


@dataclass
class FrameTools:
    """What a job needs: the analysis (``AnalyzeFrame``), the curve export and the figures."""

    analyze_frame: Any
    export: Any
    figures: Optional[FigureWriter] = None


def write_frame_images(
    figures: FigureWriter, analysis: FrameAnalysis, destination: Path, *, detector: bool = True,
    q_map: bool = True, suffix: str = "png", display: Optional[dict] = None,
) -> list[Path]:
    """The detector frame and/or the q map as pictures (each only when asked for).

    ``display``: log or linear, the colour map, and fixed limits per picture (``None``: each frame's own)."""
    display = display or {}
    style = {"log_scale": bool(display.get("log", True)), "colormap": str(display.get("colormap", "viridis"))}
    stem = export_stem(analysis)
    images = []
    if detector:
        images.append(figures.write_frame(
            analysis.data, analysis.valid, destination / f"{stem}_detector.{suffix}", title=analysis.path.name,
            levels=display.get("detector"), **style,
        ))
    rsm = analysis.reduction.reciprocal_space_map if analysis.reduction is not None else None
    if q_map and rsm is not None:
        (q0, q1), (z0, z1) = rsm.q_parallel_range, rsm.qz_range
        images.append(figures.write_frame(
            rsm.image, None, destination / f"{stem}_qmap.{suffix}", title=analysis.path.name,
            extent=(q0, q1, z0, z1), origin_upper=True, x_label=rsm.x_label, y_label="qz (Å⁻¹)",
            levels=display.get("qmap"), **style,
        ))
    return images


def run_frame_job(tools: FrameTools, job: FrameJob) -> FrameOutcome:
    """Reduce the frame, write what the choices ask for, and return the outcome (thread- and process-safe)."""
    started = time.perf_counter()
    choices = job.choices
    request = replace(job.request, with_map=choices.needs_map)
    info: dict = {}
    if job.series is None or job.series.is_identity:
        analysis = tools.analyze_frame(request)
    else:
        analysis, info = CorrectSeriesFrame(tools.analyze_frame)(request, job.series, factor=job.series_factor)
    if analysis.reduction is None:
        raise ValueError(analysis.messages[0] if analysis.messages else "No geometry.")
    giwaxs = analysis.kind != GISAXS
    cake = tools.analyze_frame.cake(analysis) if choices.cake and giwaxs else None
    written = export_frame(
        tools.export, analysis, Path(job.destination), choices, cake=cake,
        extra_metadata={"series": info} if info else None,
    )
    if (choices.detector_image or choices.q_map_image) and tools.figures is not None:
        written.extend(write_frame_images(
            tools.figures, analysis, Path(job.destination) / FOLDERS["detector_image"],
            detector=choices.detector_image, q_map=choices.q_map_image, suffix=choices.image_format,
            display=job.display,
        ))
    outcome = FrameOutcome(
        index=job.index, label=row_label(analysis), stem=export_stem(analysis), written=written,
        metadata=analysis_metadata(analysis), series_info=info,
    )
    if choices.tables:
        outcome.curves = chosen_curves(analysis, choices.curves)
    if choices.fit == FIT_PEAKS and giwaxs and analysis.geometry is not None:
        outcome.peak_inputs = peak_inputs(analysis, maps=tools.analyze_frame.maps_of(analysis))
    elif choices.fit == FIT_MODEL:
        outcome.model_curve = fit_input_curve(analysis)
    if job.live_key:
        try:
            x, intensity, x_label = series_row(analysis, job.live_key)
        except ValueError:
            pass  # this frame lacks the live curve: the map skips it, the files are written
        else:
            outcome.live_row = (x, intensity, x_label, outcome.label, (Path(analysis.path), int(analysis.frame_index)))
    outcome.seconds = time.perf_counter() - started
    return outcome


__all__ = [
    "BALANCED", "BYTES_PER_PIXEL", "FAST", "GENTLE", "SPEEDS", "FrameJob", "FrameOutcome", "FrameTools",
    "frames_at_once", "run_frame_job", "write_frame_images",
]
