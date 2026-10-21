"""Analyze ViewModel: file list, reduction settings and commands, no QWidget."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Optional

from ..application import (
    AUTO,
    FigureWriter,
    Corrections,
    CENTER_HEADER,
    CENTER_SESSION,
    GISAXS,
    AnalysisRequest,
    AnalyzeFrame,
    ExportAnalysis,
    FrameAnalysis,
    FrameSource,
    GisaxsCutSettings,
    GiwaxsSettings,
    InstrumentProfileStore,
    SaveInstrumentProfile,
)
from .folder_watch import FolderWatch
from .frames_model import FramesModelMixin
from .series_model import SeriesModelMixin
from .sources_model import SourcesModelMixin, source_stamp
from .geometry_model import FIT_SIDE_KEY, HEADER_CENTER_KEY, SECTION, GeometryModelMixin
from .batch_model import BatchModelMixin
from .export_model import EXPORT_FOLDER, ExportModelMixin
from .options_model import GAP_GUARD_KEY, OptionsModelMixin


@dataclass
class AnalyzeState:
    files: list[Path] = field(default_factory=list)
    current_index: int = -1
    frame_index: int = 0
    mode: str = AUTO
    profile_name: Optional[str] = None
    incidence_deg: Optional[float] = None
    gisaxs: GisaxsCutSettings = field(default_factory=GisaxsCutSettings)
    giwaxs: GiwaxsSettings = field(default_factory=GiwaxsSettings)
    analysis: Optional[FrameAnalysis] = None
    beam_center: Optional[tuple[float, float]] = None
    """Session override of the beam centre (canonical px); kept for every new file of the same shape."""
    center_shapes: dict = field(default_factory=dict)
    """Every centre set in this session → ``(rows, columns)`` of the frame it was set on (``None``: any
    frame). The one record of the shape: Undo and loaded settings change ``beam_center`` only."""
    corrections: Corrections = field(default_factory=Corrections)
    sum_count: int = 1
    """Consecutive frames summed into one analysis (1: none)."""

    @property
    def beam_center_shape(self) -> Optional[tuple[int, int]]:
        """``(rows, columns)`` of the frame the session centre was set on (``None``: no centre, or any frame)."""
        if self.beam_center is None:
            return None
        return self.center_shapes.get(tuple(float(value) for value in self.beam_center))


class AnalyzeViewModel(
    GeometryModelMixin, OptionsModelMixin, SourcesModelMixin, FramesModelMixin, SeriesModelMixin, ExportModelMixin,
    BatchModelMixin,
):
    def __init__(
        self,
        *,
        analyze_frame: AnalyzeFrame,
        export_analysis: ExportAnalysis,
        save_profile: SaveInstrumentProfile,
        frames: FrameSource,
        profiles: Optional[InstrumentProfileStore],
        read_setting: Callable[[str, str, Any], Any],
        settings: Any = None,
        figures: Optional[FigureWriter] = None,
    ):
        self._analyze_frame = analyze_frame
        self._export = export_analysis
        self._save_profile = save_profile
        self._frames = frames
        self._profiles = profiles
        self._read_setting = read_setting
        self.settings = settings
        self.figures = figures
        self.last_setup_path = None
        """Where the set-up of this session is kept for the next one (``None``: not kept)."""
        self.state = AnalyzeState(
            corrections=Corrections(gap_guard_px=self.stored_gap_guard(), bad_pixels=self.stored_bad_pixels())
        )
        self.watch = FolderWatch(frames.expand, frames.settled_size, source_stamp(frames))
        self._init_frames()
        self._init_sources()

    # -- files (listing and the folder watch: ``sources_model.py``) -------------------------

    def request_for(
        self, path: Path, frame_index: int = 0, *, summed: Optional[tuple] = None
    ) -> AnalysisRequest:
        """The current settings applied to a frame; ``summed`` defaults to the current sum."""
        if summed is None:
            summed = self.summed_frames_for(Path(path), frame_index)
        return AnalysisRequest(
            path=Path(path),
            frame_index=int(frame_index),
            summed_frames=summed,
            mode=self.state.mode,
            profile_name=self.state.profile_name,
            incidence_deg=self.state.incidence_deg,
            gisaxs=self.state.gisaxs,
            giwaxs=self.state.giwaxs,
            beam_center=self.state.beam_center,
            beam_center_shape=self.state.beam_center_shape,
            use_header_center=self.use_header_center,
            corrections=self.state.corrections,
        )

    def clear_files(self) -> None:
        self.state = replace(self.state, files=[], current_index=-1, frame_index=0, analysis=None)
        self.forget_frame_halves()
        self._init_frames()
        self._init_sources(keep_watch=self.watch.active)  # (a watched file that grows later comes back)

    def remove_file(self, index: int) -> Optional[Path]:
        """Take one file off the list; returns it (``None``: no such row). The file shown keeps its place in
        the list; when it is the one removed, the file that took its row (else the one before) becomes current,
        at its first frame, and nothing is analysed yet (``analysis`` is ``None``: it was the removed file's)."""
        state = self.state
        if not 0 <= int(index) < len(state.files):
            return None
        index = int(index)
        removed = state.files.pop(index)
        self._forget_source(removed)
        if index < state.current_index:
            state.current_index -= 1
        elif index == state.current_index:
            state.current_index = min(index, len(state.files) - 1)
            state.frame_index = 0
            state.analysis = None
        return removed

    def select(self, index: int) -> bool:
        if not 0 <= index < len(self.state.files):
            return False
        if index != self.state.current_index:
            self.state.current_index = index
            self.state.frame_index = 0
            self.forget_frame_halves()  # what an automatic analysis chose for the file before
        return True

    @property
    def current_path(self) -> Optional[Path]:
        index = self.state.current_index
        return self.state.files[index] if 0 <= index < len(self.state.files) else None

    # -- settings ----------------------------------------------------------------------

    def set_frame(self, frame_index: int) -> None:
        self.state.frame_index = max(0, int(frame_index))

    def set_mode(self, mode: str) -> None:
        self.state.mode = mode

    def set_profile(self, name: Optional[str]) -> None:
        self.state.profile_name = name or None

    def set_incidence(self, incidence_deg: Optional[float]) -> None:
        self.state.incidence_deg = None if incidence_deg is None else float(incidence_deg)

    def set_horizontal_band(self, low: float, high: float) -> None:
        low, high = sorted((float(low), float(high)))
        self.state.gisaxs = replace(
            self.state.gisaxs,
            horizontal_row=0.5 * (low + high),
            horizontal_half_height_px=max(0.5, 0.5 * (high - low)),
        )

    def set_vertical_band(self, low: float, high: float) -> None:
        low, high = sorted((float(low), float(high)))
        self.state.gisaxs = replace(
            self.state.gisaxs,
            vertical_column=0.5 * (low + high),
            vertical_half_width_px=max(0.5, 0.5 * (high - low)),
        )

    def set_chi_window(self, low: float, high: float) -> None:
        self.state.giwaxs = replace(self.state.giwaxs, chi_q_window=tuple(sorted((low, high))))

    def clear_chi_window(self) -> None:
        """I(χ) of the most prominent ring again (found automatically)."""
        self.state.giwaxs = replace(self.state.giwaxs, chi_q_window=None)

    def current_settings(self, export=None):
        """The set-up in use (``batch_model``), with the pin of a series (``frames_model``): a settings file, a
        project and the memory of the last session say whether every frame of a series is cut alike."""
        settings = super().current_settings(export)
        if settings.pin_series_cuts is None:
            settings = replace(settings, pin_series_cuts=bool(self.pin_series_cuts))
        return settings

    def apply_settings(self, settings) -> list[str]:
        """A loaded set-up (``batch_model``), with its pin of a series when it says one (``None``: as it is)."""
        notes = super().apply_settings(settings)
        self.set_pin_series_cuts(getattr(settings, "pin_series_cuts", None))
        return notes

    def reset_cuts(self) -> None:
        """Back to the automatic cut positions (Yoneda, beam centre, strongest ring)."""
        self.state.gisaxs = GisaxsCutSettings()
        self.state.giwaxs = GiwaxsSettings()
        self._opened_cut = None  # (found in the frame again, also for a frame opened from a Series map)

    # -- analysis ----------------------------------------------------------------------

    def request(self) -> Optional[AnalysisRequest]:
        path = self.current_path
        if path is None:
            return None
        return self._shown_request(self.request_for(path, self.state.frame_index))

    def analyze(self, request: AnalysisRequest) -> FrameAnalysis:
        """Thread-safe: reuses the frame already in memory when it is the requested one and its files are as
        they were when it was read (a module file that arrived since, a file written again: read again)."""
        stamp = self.files_stamp(request.path, request.summed_frames)
        loaded = self.state.analysis
        if loaded is not None and self._read_stamps.of(loaded, stamp) != stamp:
            loaded = None
        analysis = self._analyze_frame(request, loaded=loaded)
        self._read_stamps.keep(analysis, stamp)
        return analysis

    def source_masks(self, analysis: FrameAnalysis, keys) -> dict:
        """Thread-safe: the pixels behind each named curve (see ``AnalyzeFrame.source_masks``)."""
        return self._analyze_frame.source_masks(analysis, list(keys))

    def accept(self, analysis: FrameAnalysis) -> None:
        self.state.analysis = analysis
        if self._read_stamps.of(analysis) is None:  # not read by ``analyze``: its files as they are now
            self._read_stamps.keep(analysis, self.files_stamp(analysis.path, analysis.summed_frames))
        self.remember_frame_count(analysis.path, analysis.frame_count)
        self.frame_count(analysis.path)  # as it is now: a frame kept in memory knows the count of when it was read

    def forget_analysis(self) -> None:
        """No frame is held (the one asked for could not be analysed): nothing of the frame before is exported."""
        self.state.analysis = None

    def geometry_summary(self, translate: Callable[[str], str] = str) -> str:
        """The frame and its geometry in one line. ``translate`` turns each English template into the interface
        language before it is filled (the page passes ``tr``); names, numbers and units are kept. A frame whose
        file names no detector is described by its size alone."""
        analysis = self.state.analysis
        if analysis is None:
            return translate("No frame loaded")
        rows, columns = analysis.shape
        head = " ".join(part for part in (analysis.detector_name, f"{rows}×{columns}") if part)
        geometry = analysis.geometry
        if geometry is None:
            return translate("{frame} · no geometry").format(frame=head)
        profile = analysis.resolution.profile
        origin = {
            "matched": translate("profile “{name}” (matched)").format(name=profile.name) if profile else "",
            "chosen": translate("profile “{name}”").format(name=profile.name) if profile else "",
        }.get(analysis.resolution.how, "")
        parts = [
            head, f"D = {geometry.distance_m * 1e3:.1f} mm", f"λ = {geometry.wavelength_angstrom:.4f} Å",
            f"αi = {geometry.incidence_deg:.3f}°", (analysis.kind or "").upper(), origin,
        ]
        return " · ".join(part for part in parts if part)

    @staticmethod
    def is_gisaxs(analysis: Optional[FrameAnalysis]) -> bool:
        return analysis is not None and analysis.kind == GISAXS


__all__ = [
    "AnalyzeState",
    "AnalyzeViewModel",
    "CENTER_HEADER",
    "CENTER_SESSION",
    "EXPORT_FOLDER",
    "FIT_SIDE_KEY",
    "GAP_GUARD_KEY",
    "HEADER_CENTER_KEY",
    "SECTION",
]
