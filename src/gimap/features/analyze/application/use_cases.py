"""Analyze use cases: resolve geometry, reduce a frame, export, keep profiles."""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np

from src.gimap.shared.geometry import DetectorGeometry, InstrumentProfile
from src.gimap.shared.geometry.conventions import (
    canonical_from_fitting_center,
    canonical_from_index_center,
)

from ..domain import (
    GISAXS,
    GIWAXS,
    CakeMap,
    Corrections,
    GiwaxsMaps,
    Reduction,
    SymmetryCenter,
    cake_map,
    classify_measurement,
    curve_source_mask,
    has_intensity_corrections,
    symmetric_center_x,
    giwaxs_maps,
    gisaxs_maps,
    gisaxs_q_map,
    reduce_gisaxs,
    reduce_giwaxs,
    valid_pixels,
)
from .models import AUTO, AnalysisRequest, FrameAnalysis
from .export_record import (  # noqa: F401 - the record of an export; these names are imported from here
    SOFTWARE,
    _plain,
    analysis_metadata,
    export_stem,
    fit_input_curve,
    fit_input_metadata as _fit_input_metadata,
    fit_input_observation,
)
from .frame_preprocessing import FramePreprocessingMixin, FrameSum, floating
from .geometry_resolution import ResolveGeometry, center_not_used_message
from .ports import CurveWriter, FrameSource, InstrumentProfileStore


class AnalyzeFrame(FramePreprocessingMixin):
    """Load one frame and reduce it with no user input beyond the request.

    The GIWAXS q maps depend only on frame shape and geometry, so the last
    ones are kept: a folder of frames from one set-up computes them once.
    """

    def __init__(self, frames: FrameSource, profiles: Optional[InstrumentProfileStore]):
        self.frames = frames
        self.resolve = ResolveGeometry(profiles)
        self._maps_key: Optional[tuple] = None
        self._maps: Optional[GiwaxsMaps] = None
        self._gisaxs_maps_key: Optional[tuple] = None
        self._gisaxs_maps = None
        self._q_map_key: Optional[tuple] = None
        self._q_map_source: Optional[tuple] = None
        self._q_map = None
        self._init_preprocessing()

    def _load_frames(self, request: AnalysisRequest) -> tuple[np.ndarray, np.ndarray, dict]:
        """Raw data and validity of the request's frame, or of the sum of its frames."""
        image = self.frames.load(request.path, request.frame_index)
        data = np.asarray(image.data, dtype=np.float32)
        valid = valid_pixels(data, image.mask, negatives_valid=floating(image.metadata))
        common = dict(
            path=Path(request.path),
            frame_index=int(request.frame_index),
            frame_count=max(1, int(self.frames.frame_count(request.path))),
            detector_name=image.detector_name,
            metadata=dict(image.metadata or {}),
            summed_frames=request.summed_frames,
        )
        if request.summed_frames:
            total = FrameSum()  # one frame in memory at a time: Sum 40 of a 15 Mpx detector held 3 GB as a list
            total.add(data, valid)
            for path, index in request.summed_frames:
                extra = self.frames.load(path, index)
                extra_data = np.asarray(extra.data, dtype=np.float32)
                total.add(extra_data, valid_pixels(extra_data, extra.mask, negatives_valid=floating(extra.metadata)))
                del extra, extra_data
            data, valid = total.result()
        return data, valid, common

    def _giwaxs_maps(self, shape: tuple[int, int], geometry: DetectorGeometry) -> GiwaxsMaps:
        key = (shape, geometry)
        maps, cached_key = self._maps, self._maps_key
        if maps is None or cached_key != key:
            maps = giwaxs_maps(shape, geometry)
            self._maps, self._maps_key = maps, key
        return maps

    def _gisaxs_q_map(self, data: np.ndarray, valid: np.ndarray, geometry: DetectorGeometry):
        """The qy–qz map of a GISAXS frame; kept while only the cuts change (same arrays, same geometry)."""
        shape = (int(data.shape[0]), int(data.shape[1]))
        key = (shape, geometry)
        source = self._q_map_source
        if source is not None and source[0] is data and source[1] is valid and self._q_map_key == key:
            return self._q_map  # the arrays are held here, so identity cannot be reused by another frame
        if self._gisaxs_maps is None or self._gisaxs_maps_key != (shape, geometry):
            self._gisaxs_maps = gisaxs_maps(shape, geometry)
            self._gisaxs_maps_key = (shape, geometry)
        self._q_map = gisaxs_q_map(data, valid, self._gisaxs_maps)
        self._q_map_key, self._q_map_source = key, (data, valid)
        return self._q_map

    def maps_of(self, analysis: FrameAnalysis) -> GiwaxsMaps:
        """The q maps of a GIWAXS frame (cached for its shape and geometry)."""
        if analysis.geometry is None or analysis.kind == GISAXS:
            raise ValueError("This needs a GIWAXS frame with a geometry.")
        return self._giwaxs_maps((int(analysis.data.shape[0]), int(analysis.data.shape[1])), analysis.geometry)

    def cake(self, analysis: FrameAnalysis) -> CakeMap:
        """The GIWAXS frame unwrapped onto χ–q (``cake_map``), from the q maps already computed."""
        if analysis.geometry is None or analysis.kind == GISAXS:
            raise ValueError("The unwrapped view needs a GIWAXS frame with a geometry.")
        maps = self.maps_of(analysis)
        return cake_map(analysis.data, np.asarray(analysis.valid, dtype=bool) & maps.above_horizon, maps)

    def source_masks(self, analysis: FrameAnalysis, keys: Sequence[str]) -> dict[str, np.ndarray]:
        """The pixels each named curve of ``analysis`` averages (curves it does not have are skipped)."""
        reduction = analysis.reduction
        if reduction is None or analysis.geometry is None:
            return {}
        maps = None
        if reduction.kind != GISAXS:
            maps = self._giwaxs_maps((int(analysis.data.shape[0]), int(analysis.data.shape[1])), analysis.geometry)
        masks = {}
        for key in keys:
            curve = reduction.curve(key)
            if curve is not None and not curve.is_empty:
                masks[key] = curve_source_mask(curve, analysis.valid, maps)
        return masks

    def __call__(
        self, request: AnalysisRequest, *, loaded: Optional[FrameAnalysis] = None
    ) -> FrameAnalysis:
        """Reduce the requested frame(s); ``loaded`` is reused when it holds the same frames."""
        if (
            loaded is not None
            and Path(loaded.path) == Path(request.path)
            and loaded.frame_index == int(request.frame_index)
            and tuple(loaded.summed_frames) == request.summed_frames
        ):
            raw_data = loaded.raw_data if loaded.raw_data is not None else loaded.data
            raw_valid = loaded.raw_valid if loaded.raw_valid is not None else loaded.valid
            common = dict(
                path=loaded.path,
                frame_index=loaded.frame_index,
                frame_count=loaded.frame_count,
                detector_name=loaded.detector_name,
                metadata=loaded.metadata,
                summed_frames=loaded.summed_frames,
            )
        else:
            raw_data, raw_valid, common = self._load_frames(request)
        shape = (int(raw_data.shape[0]), int(raw_data.shape[1]))
        resolution = self.resolve(
            detector_name=common["detector_name"],
            shape=shape,
            profile_name=request.profile_name,
            incidence_deg=request.incidence_deg,
            beam_center=request.beam_center,
            use_header_center=request.use_header_center,
            header_center=header_beam_center(common["metadata"], shape),
            distance_m=request.distance_m,
            beam_center_shape=request.beam_center_shape,
        )
        geometry = resolution.geometry
        # What Auto would choose, in every mode: a pinned mode that disagrees is pointed out.
        detected = None if geometry is None else classify_measurement(shape, geometry)
        kind = None if geometry is None else (request.mode if request.mode != AUTO else detected)
        mirror_x = geometry.beam_center_x_px if geometry is not None and kind == GIWAXS else None
        common.update(self._preprocess(raw_data, raw_valid, request.corrections, common["metadata"], mirror_x))
        # Errors from counting statistics only for photon counts; otherwise from the scatter of the pixels.
        counting = not floating(common["metadata"]) and not request.corrections.background_path
        if kind == GIWAXS and geometry is not None and has_intensity_corrections(request.corrections):
            self._correct_intensity(common, raw_data, raw_valid, geometry, request.corrections, counting)
        data, valid = common["data"], common["valid"]
        scale = common.get("intensity_scale")
        messages: list[str] = []
        if geometry is None:
            detector = common["detector_name"] or "this detector"
            messages.append(
                f"No instrument profile for {detector} ({shape[0]}×{shape[1]}). "
                "Calibrate once or enter the geometry to get q curves."
            )
            return FrameAnalysis(
                **common, resolution=resolution, reduction=None, kind=None, messages=tuple(messages)
            )
        unused_center = center_not_used_message(resolution, shape)
        if unused_center:
            messages.append(unused_center)
        profile_shape = resolution.profile.detector_shape if resolution.profile else None
        if profile_shape is not None and tuple(profile_shape) != shape:
            messages.append(
                f"Profile “{resolution.profile.name}” is for {profile_shape[0]}×{profile_shape[1]} "
                f"frames; this frame is {shape[0]}×{shape[1]}."
            )
        if kind == GISAXS:
            reduction = reduce_gisaxs(
                data, valid, geometry, request.gisaxs, counts=counting,
                q_map=self._gisaxs_q_map(data, valid, geometry) if request.with_map else None,
            )
        else:
            reduction = reduce_giwaxs(
                data, valid, geometry, request.giwaxs, maps=self._giwaxs_maps(shape, geometry),
                with_map=request.with_map, counts=scale if scale is not None else counting,
            )
        messages.extend(reduction.warnings)
        reduction = held_cut(reduction, request.series_cut)
        return FrameAnalysis(
            **common,
            resolution=resolution,
            reduction=reduction,
            kind=kind,
            messages=tuple(messages),
            detected_kind=detected,
        )


def header_beam_center(
    metadata: Mapping[str, Any], shape: tuple[int, int]
) -> Optional[tuple[float, float]]:
    """Canonical beam centre from the file header, if the loader found a usable one."""
    center = metadata.get("header_beam_center") if metadata else None
    if center is None:
        return None
    try:
        x, y = (float(value) for value in center)
    except (TypeError, ValueError):
        return None
    rows, columns = shape
    # A header centre far outside the frame is a placeholder, not a position.
    if not (math.isfinite(x) and math.isfinite(y)):
        return None
    if not (-columns <= x <= 2 * columns and -rows <= y <= 2 * rows):
        return None
    return x, y


def _band(values) -> Optional[tuple[float, float]]:
    try:
        low, high = (float(value) for value in values)
    except (TypeError, ValueError):
        return None
    return low, high


def held_cut(reduction: Reduction, series_cut: Optional[Mapping[str, Any]]) -> Reduction:
    """``reduction`` of a frame cut at the cut of a series (``AnalysisRequest.series_cut``), with markers that say
    so: ``series_cut`` — ``{"frame": (path, index), …}`` and those of ``horizontal_rows``, ``vertical_columns`` or
    ``chi_q_window`` the frame was cut at — and ``horizontal_source`` ``series`` where the reduction alone says
    ``manual`` (it calls any row it is given “manual”). A cut the person set is not the series' (``with_cut``
    leaves it) and stays as it is; so does a reduction that took nothing from the series."""
    if not series_cut or reduction is None:
        return reduction
    markers, held = dict(reduction.markers or {}), {}
    technique = series_cut.get("technique")
    if reduction.kind == GISAXS and technique == GISAXS:
        for name, band in (("horizontal_rows", "horizontal_band"), ("vertical_columns", "vertical_band")):
            pinned, used = _band(series_cut.get(name)), _band(markers.get(band))
            # (the same centre: the width of a band stays that of the frame's request)
            if pinned is not None and used is not None and abs(sum(pinned) - sum(used)) < 1e-6:
                held[name] = pinned
        if "horizontal_rows" in held and markers.get("horizontal_source") == "manual":
            markers["horizontal_source"] = "series"
    elif reduction.kind == GIWAXS and technique == GIWAXS:
        pinned, used = _band(series_cut.get("chi_q_window")), _band(markers.get("chi_q_window"))
        if pinned is not None and used is not None and np.allclose(sorted(pinned), sorted(used)):
            held["chi_q_window"] = pinned
    if not held:
        return reduction
    frame = series_cut.get("frame")
    markers["series_cut"] = {"frame": (str(frame[0]), int(frame[1])) if frame else None, **held}
    return replace(reduction, markers=markers)


def refine_center_by_symmetry(analysis: FrameAnalysis, *, cancelled=None) -> SymmetryCenter:
    """Beam-centre x of a GISAXS frame from the left–right symmetry of its horizontal cut band."""
    reduction = analysis.reduction
    if analysis.kind != GISAXS or reduction is None or analysis.geometry is None:
        raise ValueError("Symmetry refinement needs a GISAXS frame with a geometry.")
    curve = reduction.curve("horizontal")
    if curve is None:
        raise ValueError("This frame has no horizontal cut.")
    rows = tuple(int(value) for value in curve.region.get("rows", (0, 0)))
    return symmetric_center_x(
        analysis.data, analysis.valid, rows, analysis.geometry.beam_center_x_px, cancelled=cancelled
    )


class ExportAnalysis:
    def __init__(self, writer: CurveWriter):
        self.writer = writer

    def q_map(self, analysis: FrameAnalysis, path: Path) -> Path:
        """The frame's q map (qy–qz for GISAXS, q∥–qz for GIWAXS) as a CSV table with its axes."""
        rsm = analysis.reduction.reciprocal_space_map if analysis.reduction is not None else None
        if rsm is None:
            raise ValueError("This frame has no q map yet (it needs a geometry).")
        x_axis, z_axis = rsm.axes()
        return self.writer.write_map(
            rsm.image, x_axis, z_axis, Path(path), (rsm.x_label, "qz (Å⁻¹)"), analysis_metadata(analysis)
        )

    def curves(
        self, analysis: FrameAnalysis, keys: Sequence[str], destination: Path,
        extra_metadata: Optional[Mapping[str, Any]] = None, *, text_format: str = "csv",
    ) -> list[Path]:
        """Only the named curves (for example those of one plot), one CSV each, plus the JSON record."""
        chosen = [curve for key in keys if (curve := analysis.reduction.curve(key)) is not None and not curve.is_empty]
        if not chosen:
            raise ValueError("None of these curves has data.")
        metadata = analysis_metadata(analysis)
        if extra_metadata:
            metadata.update(_plain(dict(extra_metadata)))
        return self.writer.write(chosen, Path(destination), export_stem(analysis), metadata, text_format=text_format)

    def cake(self, analysis: FrameAnalysis, cake: CakeMap, path: Path) -> Path:
        """The unwrapped view as a CSV table: q in the first row, χ in the first column."""
        q_axis, chi_axis = cake.axes()
        metadata = {**analysis_metadata(analysis), "title": "GIMaP Analyze cake: mean intensity on a chi-q grid, blank = no pixel"}
        return self.writer.write_map(cake.image, q_axis, chi_axis, Path(path), ("q (Å⁻¹)", "χ (°)"), metadata)

    def fit_input(self, analysis: FrameAnalysis, destination: Path, halves: Optional[str] = None) -> Path:
        """Write the Fitting input as ``<stem>_fit_input.dat`` (q in Å⁻¹; ``halves``: ``fit_input_observation``)."""
        curve = fit_input_curve(analysis)
        if curve is None:
            raise ValueError("Nothing to fit: this frame has no usable curve.")
        path = Path(destination) / f"{export_stem(analysis)}_fit_input.dat"
        return self.writer.write_xy(
            curve, path, _fit_input_metadata(analysis, analysis_metadata(analysis), halves)
        )

    def __call__(
        self,
        analysis: FrameAnalysis,
        destination: Path,
        extra_metadata: Optional[Mapping[str, Any]] = None,
        *,
        halves: Optional[str] = None,
    ) -> list[Path]:
        """Every curve as CSV plus the JSON record, and the Fitting input when there is one (``halves``: what a
        fit is to take of a GISAXS cut, written with it — ``fit_input_observation``)."""
        if analysis.reduction is None:
            raise ValueError("Nothing to export: this frame has no geometry yet.")
        curves = [curve for curve in analysis.reduction.curves if not curve.is_empty]
        if not curves:
            raise ValueError("Nothing to export: every curve of this frame is empty.")
        metadata = analysis_metadata(analysis)
        if extra_metadata:
            metadata.update(_plain(dict(extra_metadata)))
        written = self.writer.write(curves, Path(destination), export_stem(analysis), metadata)
        fit_curve = fit_input_curve(analysis)
        if fit_curve is not None:
            path = Path(destination) / f"{export_stem(analysis)}_fit_input.dat"
            written.append(
                self.writer.write_xy(fit_curve, path, _fit_input_metadata(analysis, metadata, halves))
            )
        return written


class SaveInstrumentProfile:
    def __init__(self, profiles: Optional[InstrumentProfileStore]):
        self.profiles = profiles

    def __call__(
        self,
        name: str,
        geometry: DetectorGeometry,
        *,
        detector_name: Optional[str],
        shape: Optional[tuple[int, int]],
        source: str,
    ) -> InstrumentProfile:
        if self.profiles is None:
            raise ValueError("Instrument profiles are not available in this session.")
        existing = self.profiles.find(name)
        if existing is not None:
            profile = existing.updated(geometry, source=source)
        else:
            profile = InstrumentProfile(
                name=name,
                geometry=geometry,
                detector_name=detector_name,
                detector_shape=shape,
                source=source,
            )
        self.profiles.save(profile)
        return profile


def geometry_from_fitting_settings(
    settings: Callable[[str, str, Any], Any], rows: int
) -> Optional[DetectorGeometry]:
    """Canonical geometry stored by the former Cut & Fitting page, or ``None``.

    The page is gone; its stored geometry is only offered once as an
    instrument profile (Analyze's “Use Previous Geometry”).

    ``settings(section, key, default)`` reads the application settings.  The
    Fitting beam-centre row counts from the bottom of its analysis image,
    which is the file flipped vertically when its “flip up/down” input option
    is on; both cases are converted here.
    """
    distance = settings("fitting", "detector.distance", None)
    center_x = settings("fitting", "detector.beam_center_x", None)
    center_y = settings("fitting", "detector.beam_center_y", None)
    pixel_x = settings("fitting", "detector.pixel_size_x", None)
    pixel_y = settings("fitting", "detector.pixel_size_y", None)
    wavelength_nm = settings("beam", "wavelength", None)
    incidence = settings("beam", "grazing_angle", 0.0)
    if None in (distance, center_x, center_y, pixel_x, pixel_y, wavelength_nm):
        return None
    if bool(settings("fitting", "gisaxs_input.flip_ud", False)):
        canonical_x, canonical_y = canonical_from_index_center(float(center_x), float(center_y))
    else:
        canonical_x, canonical_y = canonical_from_fitting_center(
            float(center_x), float(center_y), int(rows)
        )
    try:
        return DetectorGeometry(
            pixel_size_x_m=float(pixel_x) * 1e-6,
            pixel_size_y_m=float(pixel_y) * 1e-6,
            distance_m=float(distance) * 1e-3,
            beam_center_x_px=canonical_x,
            beam_center_y_px=canonical_y,
            wavelength_angstrom=float(wavelength_nm) * 10.0,
            incidence_deg=float(incidence or 0.0),
        )
    except ValueError:
        return None


def expand_sources(frames: FrameSource, paths: Sequence[str | Path]) -> list[Path]:
    return frames.expand(paths)


__all__ = [
    "AnalyzeFrame",
    "ExportAnalysis",
    "ResolveGeometry",
    "SaveInstrumentProfile",
    "analysis_metadata",
    "expand_sources",
    "export_stem",
    "fit_input_curve",
    "fit_input_observation",
    "geometry_from_fitting_settings",
    "header_beam_center",
    "held_cut",
    "refine_center_by_symmetry",
]
