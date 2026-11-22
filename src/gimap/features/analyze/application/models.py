"""Requests and results exchanged between the Analyze view model and use cases."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from src.gimap.shared.geometry import DetectorGeometry, InstrumentProfile
from src.gimap.shared.geometry.detector_tilt import FlatDetectorError

from ..domain import BadPixels, Corrections, GisaxsCutSettings, GiwaxsSettings, NormalisationUse, Reduction
from .flat_field import FlatFieldUse

AUTO = "auto"
MODES = (AUTO, "gisaxs", "giwaxs", "transmission")

CENTER_PROFILE = "profile"
CENTER_HEADER = "header"
CENTER_SESSION = "session"

FrameRef = tuple[Path, int]
"""A frame of a detector file: ``(path, frame index)``."""


@dataclass(frozen=True)
class AnalysisRequest:
    path: Path
    frame_index: int = 0
    mode: str = AUTO
    """``auto`` classifies from the geometry; ``gisaxs``/``giwaxs``/``transmission`` force a reduction."""
    profile_name: Optional[str] = None
    """Use this profile instead of the automatic match (for frames of its size, or with ``force_profile``)."""
    forced_shapes: tuple = ()
    """Frame sizes ``(rows, columns)`` on which ``profile_name`` is used although it is for another size (the person
    said so: Use Anyway)."""
    incidence_deg: Optional[float] = None
    """Grazing angle for this measurement; ``None`` keeps the profile's value."""
    gisaxs: GisaxsCutSettings = field(default_factory=GisaxsCutSettings)
    giwaxs: GiwaxsSettings = field(default_factory=GiwaxsSettings)
    beam_center: Optional[tuple[float, float]] = None
    """Canonical beam centre replacing the profile's for this session (``None``: keep it)."""
    use_header_center: bool = False
    """Use the beam centre written in the file header, when there is one."""
    corrections: Corrections = field(default_factory=Corrections)
    """Background subtraction and valid intensity range, applied before reducing."""
    distance_m: Optional[float] = None
    """Detector distance replacing the profile's (series alignment); ``None`` keeps it."""
    summed_frames: tuple[FrameRef, ...] = ()
    """Further frames added to this one before any correction (frame summing)."""
    with_map: bool = True
    """``False``: the curves only, no q map (a series map needs one curve of every frame)."""
    beam_center_shape: Optional[tuple] = None
    """``(rows, columns)`` of the frame ``beam_center`` was set on: it is used only on frames of that
    shape (another detector gets its header or profile centre). ``None``: on every frame."""
    series_cut: Optional[dict] = field(default=None, hash=False)
    """The cut of a series this frame is cut at (the view model's ``series_cut()``: the automatic cut of one frame,
    held for every frame of a series run); the reduction's record then says so (``held_cut``). ``None``: its own."""

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"Unknown analysis mode {self.mode!r}; expected one of {MODES}.")
        object.__setattr__(
            self,
            "summed_frames",
            tuple((Path(path), int(index)) for path, index in self.summed_frames),
        )
        if self.beam_center_shape is not None:
            object.__setattr__(self, "beam_center_shape", tuple(int(value) for value in self.beam_center_shape))

    @property
    def frames(self) -> tuple[FrameRef, ...]:
        """Every frame analysed: this one first, then the summed ones."""
        return ((Path(self.path), int(self.frame_index)),) + self.summed_frames


@dataclass(frozen=True)
class GeometryResolution:
    geometry: Optional[DetectorGeometry]
    profile: Optional[InstrumentProfile]
    how: str
    """``chosen`` (the profile picked by hand), ``file`` (the ``.poni`` beside the frame), ``matched`` (a profile
    matched by detector name and frame size) or ``missing``."""
    center_source: str = CENTER_PROFILE
    """Where the beam centre of ``geometry`` came from: profile, header or session."""
    header_center: Optional[tuple[float, float]] = None
    """Canonical beam centre written in the file header, if the file has one."""
    ignored_center_shape: Optional[tuple[int, int]] = None
    """The frame shape a session beam centre was set for, when it was not used for this frame (another shape)."""
    refused_profile: Optional[str] = None
    """The profile picked by hand when it was not used because it is for frames of another size (another
    detector): a frame never takes the geometry of another detector without being told to."""
    refused_shape: Optional[tuple[int, int]] = None


@dataclass(frozen=True)
class FrameAnalysis:
    """A loaded frame and, when a geometry is known, its reduction."""

    path: Path
    frame_index: int
    frame_count: int
    data: np.ndarray
    valid: np.ndarray
    detector_name: Optional[str]
    metadata: dict
    resolution: GeometryResolution
    reduction: Optional[Reduction]
    kind: Optional[str]
    messages: tuple[str, ...] = ()
    raw_data: Optional[np.ndarray] = None
    """The frame as loaded, before corrections (``None``: same as ``data``)."""
    raw_valid: Optional[np.ndarray] = None
    corrections: Corrections = field(default_factory=Corrections)
    summed_frames: tuple[FrameRef, ...] = ()
    """Frames added to this one (``data`` is their sum)."""
    bad_pixels: Optional[BadPixels] = None
    """Hot and dead pixels left out automatically (``Corrections.bad_pixels``)."""
    drawn_mask: Optional[np.ndarray] = None
    """User mask after all image, shape and rule edits, before combining with detector validity."""
    filled_pixels: Optional[np.ndarray] = None
    """Pixels filled from the mirror side (``Corrections.mirror_fill``, GIWAXS)."""
    intensity_scale: Optional[np.ndarray] = None
    """Corrected intensity corresponding to one sample count, including flat, geometric and frame scaling."""
    flat_field: Optional[FlatFieldUse] = None
    """The flat field the frame was divided by (``Corrections.flat_field_path``): file, SHA-256, mean, pixels
    it leaves out; ``None`` without one."""
    detected_kind: Optional[str] = None
    """GISAXS or GIWAXS as Auto would classify the frame from its geometry, whatever the mode chosen
    (``kind`` is the reduction used); ``None`` without a geometry."""
    flat_detector: Optional[FlatDetectorError] = None
    """A ``.poni`` tilt the flat geometry leaves out, and how far |q| is off for it on this frame's valid pixels
    (``flat_detector.py``); ``None`` when the profile names no tilt."""
    header_values: dict = field(default_factory=dict)
    """What the header offers for the normalisation (``normalisation.header_values``; of a sum: the sums)."""
    normalisation: Optional[NormalisationUse] = None
    """The constant the frame was multiplied by and its terms (``Corrections.normalisation``); ``None``: none."""
    reference_peak_scale: Optional[float] = None
    """Factor subsequently applied to the reduced curves and q map of a series, after frame normalisation."""
    error_model_used: Optional[str] = None
    pixel_variance: Optional[np.ndarray] = None
    """Sample shot variance or supplied variance, in squared corrected pixel units."""
    extra_variance: Optional[np.ndarray] = None
    """Variance from the measured reference frames, in the same units."""
    reference_dark: Optional[dict] = None
    """Dark scales used for the reference frames, with their sources."""

    @property
    def calibrated_curve_unit(self) -> Optional[str]:
        if self.normalisation is None or self.normalisation.standard_calibration is None:
            return None
        return self.normalisation.unit if self.reference_peak_scale is None else "relative"

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(int(value) for value in self.data.shape[:2])

    @property
    def geometry(self) -> Optional[DetectorGeometry]:
        return self.resolution.geometry

    @property
    def frame_total(self) -> int:
        """Number of frames summed into ``data`` (1 without summing)."""
        return 1 + len(self.summed_frames)


__all__ = [
    "AUTO",
    "AnalysisRequest",
    "CENTER_HEADER",
    "CENTER_PROFILE",
    "CENTER_SESSION",
    "FrameAnalysis",
    "FrameRef",
    "GeometryResolution",
    "MODES",
]
