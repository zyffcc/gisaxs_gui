"""Export part of the Analyze view model: curves, fit input, batches and figures."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Optional

from ..application import (
    AnalysisRequest,
    CorrectSeriesFrame,
    FrameAnalysis,
    SeriesMap,
    export_series_map,
    export_series_track,
    write_frame_images,
    row_label,
    series_row,
)

EXPORT_FOLDER = "gimap_analysis"


class ExportModelMixin:
    """Needs ``state``, ``_analyze_frame``, ``_export``, ``figures``, ``fit_side`` and ``request_for``."""

    # -- export ------------------------------------------------------------------------

    def default_export_dir(self) -> Optional[Path]:
        path = self.current_path
        return path.parent / EXPORT_FOLDER if path is not None else None

    def export(self, destination: Optional[Path] = None) -> list[Path]:
        analysis = self.state.analysis
        if analysis is None:
            raise ValueError("Open a frame first.")
        target = Path(destination) if destination is not None else self.default_export_dir()
        return self._export(analysis, target)

    def export_q_map(self, path: Path) -> Path:
        analysis = self.state.analysis
        if analysis is None:
            raise ValueError("Open a frame first.")
        return self._export.q_map(analysis, Path(path))

    def series_row(self, request: AnalysisRequest, key: str) -> tuple:
        """Thread-safe: ``(x, intensity, x_label, row label, (path, frame_index))`` of one frame's curve."""
        analysis = self.analyze(replace(request, with_map=False))  # one curve per frame: no q map
        x, intensity, x_label = series_row(analysis, key)
        return x, intensity, x_label, row_label(analysis), (Path(analysis.path), int(analysis.frame_index))

    def cake(self, analysis: FrameAnalysis):
        """Thread-safe: the frame unwrapped onto χ–q."""
        return self._analyze_frame.cake(analysis)

    def export_cake(self, analysis: FrameAnalysis, cake, path: Path) -> Path:
        return self._export.cake(analysis, cake, Path(path))

    def export_curves(self, keys, destination: Path) -> list[Path]:
        """The named curves of the frame on screen, one CSV each, with the JSON record."""
        analysis = self.state.analysis
        if analysis is None or analysis.reduction is None:
            raise ValueError("Open a frame with a geometry first.")
        return self._export.curves(analysis, list(keys), Path(destination))

    def export_series_track(self, series: SeriesMap, track, path: Path) -> Path:
        return export_series_track(self._export.writer, series, track, Path(path))

    def export_series_map(self, series: SeriesMap, path: Path) -> Path:
        return export_series_map(self._export.writer, series, Path(path))

    def write_fit_input(self) -> Path:
        """Write the curve for Fitting next to the data and return its path."""
        analysis = self.state.analysis
        if analysis is None:
            raise ValueError("Open a frame first.")
        return self._export.fit_input(analysis, self.default_export_dir())

    def analyze_and_export(
        self,
        request: AnalysisRequest,
        destination: Path,
        series=None,
        series_state: Optional[dict] = None,
        save_images: bool = False,
    ) -> list[Path]:
        """Batch helper: reduce one file with a fresh load and export it.

        ``series`` applies the reference-peak corrections of an in-situ run;
        ``series_state`` (one dict per batch) carries the first frame's
        normalisation factor to the following frames.
        """
        info: dict = {}
        if series is None or series.is_identity:
            analysis = self._analyze_frame(request)
        else:
            factor = None if series_state is None else series_state.get("factor")
            analysis, info = CorrectSeriesFrame(self._analyze_frame)(request, series, factor=factor)
            if series_state is not None and "normalization_factor" in info and not series.per_frame:
                series_state.setdefault("factor", info["normalization_factor"])
        if analysis.reduction is None:
            raise ValueError(analysis.messages[0] if analysis.messages else "No geometry.")
        written = self._export(analysis, Path(destination), {"series": info} if info else None)
        if save_images and self.figures is not None:
            written.extend(self._save_frame_images(analysis, Path(destination)))
        return written

    def _save_frame_images(
        self, analysis: FrameAnalysis, destination: Path, *, detector: bool = True, q_map: bool = True,
        suffix: str = "png",
    ) -> list[Path]:
        """The detector frame and/or the q map as PNG next to the curves (each only when asked for)."""
        return write_frame_images(self.figures, analysis, destination, detector=detector, q_map=q_map, suffix=suffix)

    def save_image_figure(self, path: Path, image, **display) -> Path:
        """The image as displayed (log, colour map, levels) as a publication figure."""
        if self.figures is None:
            raise ValueError("Figure export is not available in this session.")
        return self.figures.write_image(image, Path(path), **display)

    def save_curves_figure(self, path: Path, curves, **labels) -> Path:
        if self.figures is None:
            raise ValueError("Figure export is not available in this session.")
        if not curves:
            raise ValueError("The plot shows no curve.")
        return self.figures.write_curves(curves, Path(path), **labels)


__all__ = ["EXPORT_FOLDER", "ExportModelMixin"]
