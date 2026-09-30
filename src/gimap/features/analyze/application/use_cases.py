"""Analyze use cases: resolve geometry, reduce a frame, export, keep profiles."""

from __future__ import annotations

import math
from dataclasses import asdict, is_dataclass, replace
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
    Curve,
    GiwaxsMaps,
    SymmetryCenter,
    cake_map,
    classify_measurement,
    curve_source_mask,
    describe_intensity_corrections,
    has_intensity_corrections,
    sum_frames,
    symmetric_center_x,
    giwaxs_maps,
    gisaxs_maps,
    gisaxs_q_map,
    reduce_gisaxs,
    reduce_giwaxs,
    valid_pixels,
)
from .models import (
    AUTO,
    CENTER_HEADER,
    CENTER_PROFILE,
    CENTER_SESSION,
    AnalysisRequest,
    FrameAnalysis,
    GeometryResolution,
)
from .frame_preprocessing import FramePreprocessingMixin, floating
from .ports import CurveWriter, FrameSource, InstrumentProfileStore

SOFTWARE = "GIMaP Analyze 1"


class ResolveGeometry:
    """Pick the geometry for a frame: an explicit profile, else the best match.

    The beam centre is the profile's (the calibrated one) unless the session
    overrides it, or the user asked to trust file headers and the file has a
    header centre.  A session override wins over the header.
    """

    def __init__(self, profiles: Optional[InstrumentProfileStore]):
        self.profiles = profiles

    def __call__(
        self,
        *,
        detector_name: Optional[str],
        shape: tuple[int, int],
        profile_name: Optional[str] = None,
        incidence_deg: Optional[float] = None,
        beam_center: Optional[tuple[float, float]] = None,
        use_header_center: bool = False,
        header_center: Optional[tuple[float, float]] = None,
        distance_m: Optional[float] = None,
    ) -> GeometryResolution:
        profile: Optional[InstrumentProfile] = None
        how = "missing"
        if self.profiles is not None:
            if profile_name:
                profile = self.profiles.find(profile_name)
                how = "chosen" if profile is not None else "missing"
            else:
                profile = self.profiles.match(detector_name=detector_name, shape=shape)
                how = "matched" if profile is not None else "missing"
        if profile is None:
            return GeometryResolution(None, None, "missing", header_center=header_center)
        geometry = profile.geometry
        source = CENTER_PROFILE
        if beam_center is not None:
            geometry = geometry.with_beam_center(*(float(value) for value in beam_center))
            source = CENTER_SESSION
        elif use_header_center and header_center is not None:
            geometry = geometry.with_beam_center(*header_center)
            source = CENTER_HEADER
        if incidence_deg is not None:
            geometry = geometry.with_incidence(float(incidence_deg))
        if distance_m is not None:
            geometry = replace(geometry, distance_m=float(distance_m))
        return GeometryResolution(geometry, profile, how, source, header_center)


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
            frames = [(data, valid)]
            for path, index in request.summed_frames:
                extra = self.frames.load(path, index)
                extra_data = np.asarray(extra.data, dtype=np.float32)
                frames.append((extra_data, valid_pixels(extra_data, extra.mask, negatives_valid=floating(extra.metadata))))
            data, valid = sum_frames(frames)
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
        )
        geometry = resolution.geometry
        kind = None if geometry is None else (
            request.mode if request.mode != AUTO else classify_measurement(shape, geometry)
        )
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
        return FrameAnalysis(
            **common,
            resolution=resolution,
            reduction=reduction,
            kind=kind,
            messages=tuple(messages),
        )


def _plain(value: Any) -> Any:
    """JSON-friendly copy of markers (dataclasses, tuples, numpy scalars)."""
    if is_dataclass(value):
        return _plain(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def analysis_metadata(analysis: FrameAnalysis) -> dict[str, Any]:
    """Provenance written next to exported curves."""
    reduction = analysis.reduction
    profile = analysis.resolution.profile
    return {
        "software": SOFTWARE,
        "source_file": str(analysis.path),
        "frame_index": analysis.frame_index,
        "frame_count": analysis.frame_count,
        "summed_frames": [
            {"file": str(path), "frame_index": index} for path, index in analysis.summed_frames
        ],
        "detector_name": analysis.detector_name,
        "frame_shape": list(analysis.shape),
        "measurement": analysis.kind,
        "geometry": _plain(analysis.geometry) if analysis.geometry is not None else None,
        "geometry_convention": (
            "canonical pixel-corner frame, row 0 at top, beam centre = direct beam; "
            "exact grazing-incidence q, Å⁻¹"
        ),
        "instrument_profile": (
            {"name": profile.name, "source": profile.source, "updated_at": profile.updated_at}
            if profile is not None
            else None
        ),
        "corrections": _plain(analysis.corrections),
        "intensity_corrections": (
            describe_intensity_corrections(analysis.corrections) if analysis.kind == GIWAXS else None
        ),
        "bad_pixels_left_out": None if analysis.bad_pixels is None else {
            "hot": analysis.bad_pixels.hot_count, "dead": analysis.bad_pixels.dead_count,
            "defective_line_pixels": analysis.bad_pixels.line_count,
        },
        "drawn_mask_pixels": None if analysis.drawn_mask is None else int(analysis.drawn_mask.sum()),
        "pixels_filled_from_mirror": None if analysis.filled_pixels is None else int(analysis.filled_pixels.sum()),
        "markers": _plain(reduction.markers) if reduction is not None else {},
        "curves": [
            {"key": curve.key, "title": curve.title, "x_label": curve.x_label, "region": _plain(curve.region)}
            for curve in (reduction.curves if reduction is not None else ())
        ],
        "messages": list(analysis.messages),
    }


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


def fit_input_curve(analysis: FrameAnalysis) -> Optional[Curve]:
    """The curve handed to Fitting.

    GISAXS: the whole horizontal cut with signed qy — Fitting's q view shows
    the halves apart on |qy|, averages them or keeps one (fitting models are
    functions of |q|).  GIWAXS: the full I(q).  ``None`` when there is
    nothing to fit.
    """
    reduction = analysis.reduction
    if reduction is None:
        return None
    if analysis.kind == GISAXS:
        curve = reduction.curve("horizontal")
        if curve is None or curve.is_empty:
            return None
        return Curve(
            "fit_input", curve.title, curve.x, curve.intensity, curve.sigma, curve.pixels,
            curve.x_label, curve.y_label, dict(curve.region),
        )
    curve = reduction.curve("radial")
    if curve is None or curve.is_empty:
        return None
    return Curve(
        "fit_input", curve.title, curve.x, curve.intensity, curve.sigma, curve.pixels,
        curve.x_label, curve.y_label, dict(curve.region),
    )


def fit_input_observation(analysis: FrameAnalysis) -> dict[str, Any]:
    """How the fit-input points were measured, for Fitting's counting-statistics contract.

    GISAXS cuts are native detector columns (one point per column, mean counts
    per pixel, Poisson σ); background subtraction, a floating-point
    (dark-subtracted) frame and mirror filling break the counting model — σ then
    comes from the scatter of the pixels —, a valid range acts as a threshold,
    and summed frames are recorded.
    """
    corrections = analysis.corrections
    return {
        "source": "native_detector_columns" if analysis.kind == GISAXS else "radial_bins",
        "file_format": analysis.path.suffix.lower().lstrip("."),
        "intensity_unit": "counts_per_pixel",
        "counting_model_valid": (
            corrections.background_path is None and not corrections.mirror_fill and not floating(analysis.metadata)
            and analysis.intensity_scale is None
            and not (analysis.kind == GIWAXS and has_intensity_corrections(corrections))
        ),
        "threshold_enabled": corrections.minimum is not None or corrections.maximum is not None,
        "gap_guard_px": int(corrections.gap_guard_px),
        "summed_frames": analysis.frame_total,
    }


def _fit_input_metadata(analysis: FrameAnalysis, metadata: Mapping[str, Any]) -> dict[str, Any]:
    return {**metadata, "observation": fit_input_observation(analysis)}


def export_stem(analysis: FrameAnalysis) -> str:
    stem = analysis.path.stem
    if analysis.frame_count > 1:
        stem += f"_frame{analysis.frame_index:04d}"
    if analysis.summed_frames:
        stem += f"_sum{analysis.frame_total}"
    return stem


def refine_center_by_symmetry(analysis: FrameAnalysis) -> SymmetryCenter:
    """Beam-centre x of a GISAXS frame from the left–right symmetry of its horizontal cut band."""
    reduction = analysis.reduction
    if analysis.kind != GISAXS or reduction is None or analysis.geometry is None:
        raise ValueError("Symmetry refinement needs a GISAXS frame with a geometry.")
    curve = reduction.curve("horizontal")
    if curve is None:
        raise ValueError("This frame has no horizontal cut.")
    rows = tuple(int(value) for value in curve.region.get("rows", (0, 0)))
    return symmetric_center_x(analysis.data, analysis.valid, rows, analysis.geometry.beam_center_x_px)


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

    def fit_input(self, analysis: FrameAnalysis, destination: Path) -> Path:
        """Write the Fitting input as ``<stem>_fit_input.dat`` (q in Å⁻¹)."""
        curve = fit_input_curve(analysis)
        if curve is None:
            raise ValueError("Nothing to fit: this frame has no usable curve.")
        path = Path(destination) / f"{export_stem(analysis)}_fit_input.dat"
        return self.writer.write_xy(
            curve, path, _fit_input_metadata(analysis, analysis_metadata(analysis))
        )

    def __call__(
        self,
        analysis: FrameAnalysis,
        destination: Path,
        extra_metadata: Optional[Mapping[str, Any]] = None,
    ) -> list[Path]:
        """Every curve as CSV plus the JSON record, and the Fitting input when there is one."""
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
                self.writer.write_xy(fit_curve, path, _fit_input_metadata(analysis, metadata))
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
    "refine_center_by_symmetry",
]
