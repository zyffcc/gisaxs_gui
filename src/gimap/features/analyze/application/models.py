"""Requests and results exchanged between the Analyze view model and use cases."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from src.gimap.shared.geometry import DetectorGeometry, InstrumentProfile

from ..domain import BadPixels, Corrections, GisaxsCutSettings, GiwaxsSettings, Reduction

AUTO = "auto"
MODES = (AUTO, "gisaxs", "giwaxs")

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
    """``auto`` classifies from the geometry; ``gisaxs``/``giwaxs`` force a reduction."""
    profile_name: Optional[str] = None
    """Use this profile instead of the automatic match."""
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
    """``matched``, ``chosen`` or ``missing``."""
    center_source: str = CENTER_PROFILE
    """Where the beam centre of ``geometry`` came from: profile, header or session."""
    header_center: Optional[tuple[float, float]] = None
    """Canonical beam centre written in the file header, if the file has one."""
    ignored_center_shape: Optional[tuple[int, int]] = None
    """The frame shape a session beam centre was set for, when it was not used for this frame (another shape)."""


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
    """Pixels inside the masks drawn on the image (``Corrections.mask_shapes``)."""
    filled_pixels: Optional[np.ndarray] = None
    """Pixels filled from the mirror side (``Corrections.mirror_fill``, GIWAXS)."""
    intensity_scale: Optional[np.ndarray] = None
    """GIWAXS with intensity corrections on photon counts: the factor each pixel's counts were multiplied
    by (1 / solid angle × polarisation × absorption), so its Poisson variance is that factor × its value."""
    detected_kind: Optional[str] = None
    """GISAXS or GIWAXS as Auto would classify the frame from its geometry, whatever the mode chosen
    (``kind`` is the reduction used); ``None`` without a geometry."""

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
