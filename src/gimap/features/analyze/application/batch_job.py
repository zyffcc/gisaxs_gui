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

**What it will cost** (``batch_cost``): the room on disk and the time a batch takes, roughly, from the frame
on screen and the choices — said in the dialog before the batch starts (pictures and converted frames of
a long series are gigabytes).
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
from .export_record import frame_cuts, frame_name
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
POINT_BYTES = 45
"""One point of a curve in a text file (x, I, sigma, pixels at 8 digits)."""
CELL_BYTES = 9
"""One cell of a map written as numbers."""
PICTURE_BYTES = {"png": 3_000_000, "tif": 14_000_000, "svg": 3_000_000, "pdf": 3_000_000}
"""One picture of a frame (600 dpi; measured 1 to 5 MB as PNG by the frame, 13.7 MB as TIFF)."""
PICTURE_SECONDS = 0.6
WORKERS_START_SECONDS = 6.0
"""Starting the worker processes of a batch that reduces several frames at once."""
MODEL_FIT_SECONDS = {"auto": 20.0}
"""The particle-model fit of one frame: comparing sphere and cylinders (measured: 20 to 25 s for 800 points);
one named model about half of that."""


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


def batch_cost(choices: BatchChoices, *, frames: int, frame_pixels: int, curve_points: int = 0, fit_points: int = 0,
               map_cells: int = 0, cake_cells: int = 0, at_once: int = 1) -> tuple[int, int, float]:
    """``(bytes per frame, bytes in all, seconds)`` of a batch, roughly — for saying it before the batch starts.

    ``curve_points``: the points of the chosen curves of one frame together; ``fit_points``: of its curve for
    Fitting; ``map_cells`` / ``cake_cells``: the cells of its q map / unwrapped map; ``at_once``: frames reduced
    at the same time. Measured on a 2048² and a 4727 × 3142 detector; good to a factor of two."""
    pixels, frames = max(1, int(frame_pixels)), max(0, int(frames))
    per_frame, seconds = 0, pixels * SECONDS_PER_PIXEL
    if choices.per_frame:
        per_frame += curve_points * POINT_BYTES + 8_000  # the curves and the JSON record of the frame
    if choices.fit_input:
        per_frame += fit_points * POINT_BYTES
    if choices.q_map:
        per_frame += map_cells * CELL_BYTES
    if choices.cake:
        per_frame += cake_cells * CELL_BYTES
        seconds += pixels * 5e-8
    if choices.needs_map:
        seconds += pixels * 1.5e-7
    pictures = int(choices.detector_image) + int(choices.q_map_image)
    per_frame += pictures * PICTURE_BYTES.get(choices.image_format, 3_000_000)
    seconds += pictures * (PICTURE_SECONDS + pixels * 5e-8)
    if choices.frames:
        per_frame += int(pixels * 4 * (0.3 if choices.frame_format == "hdf5" else 1.0))  # 32-bit; HDF5 is compressed
        seconds += pixels * 3e-8
    if choices.fit == FIT_MODEL:
        seconds += MODEL_FIT_SECONDS.get(choices.fit_model, 10.0)
    total = per_frame * frames + (curve_points * 13 * frames if choices.tables else 0)
    at_once = max(1, int(at_once))
    return int(per_frame), int(total), seconds * frames / at_once + (WORKERS_START_SECONDS if at_once > 1 else 0.0)


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
    number: Optional[int] = None
    """The place of the frame among the listed frames, counted from 1 (every n-th frame: 1, n + 1, …)."""


@dataclass
class FrameOutcome:
    """What one frame gave, small enough to cross processes."""

    index: int
    label: str
    """How the frame is called in tables, records and titles (``export_record.frame_name``)."""
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
    number: Optional[int] = None
    """``FrameJob.number``."""
    cuts: dict = field(default_factory=dict)
    """Where the automatic cuts of this frame were taken (``export_record.frame_cuts``)."""
    summed: int = 1
    """Frames summed into this one (1: none)."""


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
    stem, title = export_stem(analysis), frame_name(analysis)  # the title says which frame of its source it is
    images = []
    if detector:
        images.append(figures.write_frame(
            analysis.data, analysis.valid, destination / f"{stem}_detector.{suffix}", title=title,
            levels=display.get("detector"), **style,
        ))
    rsm = analysis.reduction.reciprocal_space_map if analysis.reduction is not None else None
    if q_map and rsm is not None:
        (q0, q1), (z0, z1) = rsm.q_parallel_range, rsm.qz_range
        images.append(figures.write_frame(
            rsm.image, None, destination / f"{stem}_qmap.{suffix}", title=title,
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
    metadata = analysis_metadata(analysis)
    outcome = FrameOutcome(
        index=job.index, label=frame_name(analysis), stem=export_stem(analysis), written=written,
        metadata=metadata, series_info=info, number=job.number, cuts=frame_cuts(metadata),
        summed=analysis.frame_total,
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
            # the row's name as the Series map names its rows (``series_map.row_label``)
            outcome.live_row = (x, intensity, x_label, row_label(analysis), (Path(analysis.path), int(analysis.frame_index)))
    outcome.seconds = time.perf_counter() - started
    return outcome


__all__ = [
    "BALANCED", "BYTES_PER_PIXEL", "FAST", "GENTLE", "SPEEDS", "FrameJob", "FrameOutcome", "FrameTools",
    "batch_cost", "frames_at_once", "run_frame_job", "write_frame_images",
]
