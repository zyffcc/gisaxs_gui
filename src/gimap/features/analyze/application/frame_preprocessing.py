"""What happens to a loaded frame before any cut: validity, masks, corrections, mirror filling.

Order: the detector's own invalid pixels → hot and dead pixels found in the frame → masks drawn
by the person → dark frame → background → flat field (its pixels that do not respond left out before the gap guard, the
division after the range, so the range applies to the values as measured) → gap guard → valid intensity range →
(GIWAXS, when asked) pixels without data filled from the mirror side (from values already divided by the flat
field) → (GIWAXS, when asked, once the geometry is known) the intensity corrections: solid angle, polarisation,
film absorption, Lorentz factor (``_correct_intensity``) → the normalisation, one constant (``_normalise``). Hot,
dead and drawn-mask pixels are part of the frame's validity; the rest are corrections, so the raw frame is kept for
re-analysis with other settings.
For a rotated detector the two mirrored rays can subtend different solid angles. When angle corrections are
enabled, interpolate the corrected donor intensities and keep targets with an undefined correction masked.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

from src.gimap.shared.detector_io.mask_files import read_mask_image
from src.gimap.shared.detector_io.output_values import floating as _floating_dtype
from src.gimap.shared.geometry import DetectorGeometry

from ..domain import (
    GIWAXS,
    Corrections,
    apply_flat_field,
    intensity_factor,
    apply_valid_range,
    find_bad_pixels,
    guard_invalid,
    has_intensity_corrections,
    mirror_fill,
    normalisation_use,
    rasterize,
    subtract_background,
)
from ..domain.normalisation import BAD_VALUE, NORMALISATION_TEMPLATES  # noqa: F401 - the page translates them
from ..domain.mask_edits import MaskShape
from ..domain.mask_bitmap import read_bitmap
from ..domain.mask_raster import evaluate_mask
from ..domain.reference_variance import reference_variance
from .flat_field import FlatFieldMixin
from .reduction_inputs import image_stamp
from .reference_frames import dark_scale, detector_counts


def floating(metadata: Optional[Mapping[str, Any]]) -> bool:
    """A floating-point frame (dark- or background-subtracted): negative values are data, not gap codes.
    The one rule of ``shared/detector_io/output_values.floating`` (a big-endian ``>f4`` is float too)."""
    return _floating_dtype((metadata or {}).get("stored_dtype", ""))


def photon_counts(metadata: Optional[Mapping[str, Any]], corrections: Corrections, *, mirror_filled: bool) -> bool:
    """Whether to propagate counting variance, including all selected corrections."""
    return effective_error_model(metadata, corrections, mirror_filled=mirror_filled) == "poisson"


def effective_error_model(metadata, corrections, *, mirror_filled: bool) -> str:
    """Mirrored pixels repeat measured photons. Otherwise the user can override the storage heuristic."""
    if mirror_filled:
        return "scatter"
    if corrections.error_model != "auto":
        return corrections.error_model
    if floating(metadata) or corrections.background_path or corrections.dark_path:
        return "scatter"
    return "poisson"


class FrameSum:
    """Frames summed one after the other, as ``domain.sum_frames`` sums a list of them: a pixel is valid only
    where every frame is, an invalid pixel adds nothing, and the sum is NaN where it is not valid. Only the
    running total is kept, so a sum of many frames needs the memory of one."""

    def __init__(self) -> None:
        self._total: Optional[np.ndarray] = None
        self._valid: Optional[np.ndarray] = None

    def add(self, data: np.ndarray, valid: np.ndarray) -> None:
        data, valid = np.asarray(data), np.asarray(valid, dtype=bool)
        if self._total is None:
            self._total, self._valid = np.where(valid, data, 0.0).astype(np.float64), valid.copy()
            return
        if data.shape != self._total.shape:
            raise ValueError(
                f"Cannot sum a {data.shape[0]}×{data.shape[1]} frame with "
                f"{self._total.shape[0]}×{self._total.shape[1]} frames."
            )
        self._valid &= valid
        self._total += np.where(valid, data, 0.0)

    def result(self) -> tuple[np.ndarray, np.ndarray]:
        if self._total is None:
            raise ValueError("No frames to sum.")
        summed = self._total.astype(np.float32)
        summed[~self._valid] = np.nan
        return summed, self._valid


class FramePreprocessingMixin(FlatFieldMixin):
    """Needs ``frames`` (a ``FrameSource``); call ``_init_preprocessing()`` in ``__init__``."""

    def _init_preprocessing(self) -> None:
        self._init_flat_field()
        self._bad_cache: Optional[tuple] = None
        self._mask_cache: Optional[tuple] = None
        self._file_mask_cache: Optional[tuple] = None
        self._intensity_key: Optional[tuple] = None
        self._intensity: Optional[np.ndarray] = None

    def _bad_pixels(self, data: np.ndarray, valid: np.ndarray, metadata: Mapping[str, Any]):
        """Hot and dead pixels of the raw frame (kept while the same frame is re-analysed)."""
        cache = self._bad_cache  # one tuple: analyses run on several threads
        if cache is None or cache[0] is not data or cache[1] is not valid:
            cache = (data, valid, find_bad_pixels(data, valid, counts=not floating(metadata)))
            self._bad_cache = cache
        return cache[2]

    def _drawn_mask(self, shapes: tuple, frame_shape: tuple[int, int], geometry=None) -> np.ndarray:
        cache = self._mask_cache
        key = (tuple(shapes), tuple(frame_shape), geometry)
        if cache is None or cache[0] != key:
            cache = (key, rasterize(shapes, frame_shape, geometry))
            self._mask_cache = cache
        return cache[1]

    def _user_mask(self, raw_data, detector_valid, corrections, geometry):
        edits = corrections.mask_shapes
        shape = raw_data.shape
        base = self._file_mask(corrections.mask_path, shape) if corrections.mask_path else None
        if not edits:
            return base
        if base is None and all(isinstance(edit, MaskShape) for edit in edits):
            used_geometry = geometry if any(edit.kind == "q_annulus" for edit in edits) else None
            return self._drawn_mask(edits, shape, used_geometry)
        return evaluate_mask(edits, raw_data, detector_valid, geometry=geometry, initial=base,
                             read_image=self._file_mask)

    def preview_mask(self, analysis, corrections: Corrections) -> np.ndarray:
        """User mask for the displayed raw frame and candidate edits, using the reduction's evaluator."""
        raw = analysis.raw_data if analysis.raw_data is not None else analysis.data
        valid = analysis.raw_valid if analysis.raw_valid is not None else analysis.valid
        if corrections.bad_pixels:
            valid = valid & ~self._bad_pixels(raw, valid, analysis.metadata).mask
        drawn = self._user_mask(raw, valid, corrections, analysis.geometry)
        return np.zeros(raw.shape, dtype=bool) if drawn is None else drawn

    def _file_mask(self, path: str, frame_shape: tuple[int, int]) -> np.ndarray:
        """Non-zero pixels of a mask image, read once per file version."""
        key = image_stamp(path, self.frames)
        cache = self._file_mask_cache
        if cache is None or cache[0] != key:
            cache = (key, read_mask_image(Path(path), self.frames))
            self._file_mask_cache = cache
        mask = cache[1]
        if mask.shape != tuple(frame_shape):
            raise ValueError(
                f"The mask {Path(path).name} is {mask.shape[0]}×{mask.shape[1]}; the frame is "
                f"{frame_shape[0]}×{frame_shape[1]}."
            )
        return mask

    def check_mask_file(self, path: str | Path, frame_shape: tuple[int, int]) -> None:
        """Raise ``ValueError`` when the mask image cannot be read or is not of ``frame_shape`` — asked before
        the mask is kept, so a mask of another detector is refused instead of failing every later analysis."""
        self._file_mask(str(path), (int(frame_shape[0]), int(frame_shape[1])))

    def _corrected(self, raw, valid, corrections, record, exposures):
        """Convert ADU, subtract references, apply validity limits and divide by the flat response."""
        data = detector_counts(raw, corrections, exposures=exposures)
        model = record["error_model_used"]
        if model == "poisson" and (corrections.dark_path or corrections.background_path or corrections.offset_adu
                                   or corrections.error_model == "poisson"):
            record["pixel_variance"] = np.maximum(data, 0)  # measured before subtracting other frames
        if model == "sigma":
            if not corrections.sigma_path:
                raise ValueError("Choose a sigma frame for the supplied error model.")
            sigma = self._reference("sigma", corrections.sigma_path)
            if sigma.data.shape != raw.shape:
                raise ValueError("The sigma frame must have the same shape as the sample.")
            valid = valid & sigma.valid & (sigma.data >= 0)
            record["pixel_variance"] = (corrections.counts_per_adu * sigma.data)**2

        dark = dark_counts = background_counts = None
        dark_coefficient = 0.0
        if corrections.dark_path:
            dark = self._reference("dark", corrections.dark_path, corrections.dark_frame)
            dark_counts = detector_counts(dark.data, corrections)
            data, valid = subtract_background(data, valid, dark_counts, dark.valid, corrections.dark_scale, what="dark")
            dark_coefficient = -corrections.dark_scale
        if corrections.background_path:
            background = self._reference("background", corrections.background_path, corrections.background_frame)
            prepared = corrections.background_already_dark_subtracted
            background_counts = detector_counts(background.data, corrections, exposures=int(not prepared))
            scale, source = dark_scale(background, dark, corrections.dark_scale, prepared, corrections.background_dark_scale)
            net, usable = background_counts, background.valid
            if scale:
                net, usable = subtract_background(net, usable, dark_counts, dark.valid, scale, what="dark")
            data, valid = subtract_background(data, valid, net, usable, corrections.background_scale)
            dark_coefficient += corrections.background_scale * scale
            record["reference_dark"] = dict(background_dark_scale=scale, source=source,
                                            background_already_dark_subtracted=prepared,
                                            background_exposure_time_s=background.exposure_s,
                                            dark_exposure_time_s=dark.exposure_s if dark is not None else None)
        flat = None
        flat_dark_scale = 0.0
        if corrections.flat_field_path:
            flat, use = self._flat_field(corrections.flat_field_path, data.shape, corrections=corrections, dark=dark)
            record["flat_field"] = use
            flat_dark_scale = use.dark_scale
            valid = valid & flat.kept
        valid = guard_invalid(valid, corrections.gap_guard_px)
        valid = apply_valid_range(data, valid, corrections.minimum, corrections.maximum)
        if model != "scatter":
            record["extra_variance"] = reference_variance(
                data, background=background_counts, background_scale=corrections.background_scale,
                dark=dark_counts, dark_coefficient=dark_coefficient, flat=flat, flat_dark_scale=flat_dark_scale)
        if flat is not None:
            data, valid = apply_flat_field(data, valid, flat)
            if model == "poisson":
                record["intensity_scale"] = 1.0 / flat.factor
            if "pixel_variance" in record:
                record["pixel_variance"] = record["pixel_variance"] / flat.factor**2
        return data, valid

    def _preprocess(
        self, raw_data: np.ndarray, raw_valid: np.ndarray, corrections: Corrections,
        metadata: Mapping[str, Any], mirror_geometry: Optional[DetectorGeometry], geometry=None, exposures=1,
        *, mask_bitmap=None,
    ) -> dict:
        """The frame to reduce and what was done to it (fields of ``FrameAnalysis``).

        ``mirror_geometry``: the detector geometry when the frame is GIWAXS
        (mirror filling needs it); ``None`` never fills.
        """
        bad = self._bad_pixels(raw_data, raw_valid, metadata) if corrections.bad_pixels else None
        usable = raw_valid
        if bad is not None and bad.mask.any():
            usable = usable & ~bad.mask
        drawn = (self._user_mask(raw_data, usable, corrections, geometry) if mask_bitmap is None
                 else read_bitmap(mask_bitmap, raw_data.shape))
        if drawn is not None and drawn.any():
            usable = usable & ~drawn
        record: dict = {"corrections": corrections, "bad_pixels": bad, "drawn_mask": drawn}
        record["error_model_used"] = effective_error_model(
            metadata, corrections, mirror_filled=corrections.mirror_fill and mirror_geometry is not None)
        if corrections.is_identity:
            record.update(data=raw_data, valid=usable)
            if corrections.error_model == "poisson" and floating(metadata):
                record["pixel_variance"] = np.maximum(raw_data, 0)
            if usable is not raw_valid:
                record["raw_valid"] = raw_valid  # re-analysing without them starts from the detector's mask
            return record
        data, valid = self._corrected(raw_data, usable, corrections, record, exposures)
        filled = None
        if corrections.mirror_fill and mirror_geometry is not None:
            if not ((mirror_geometry.tilted or mirror_geometry.sample_roll_deg) and has_intensity_corrections(corrections)):
                data, valid, filled = mirror_fill(data, valid, mirror_geometry.beam_center_x_px, geometry=mirror_geometry)
        record.update(data=data, valid=valid, raw_data=raw_data, raw_valid=raw_valid, filled_pixels=filled)
        return record

    def _intensity_factor(self, shape: tuple[int, int], geometry: DetectorGeometry, corrections) -> np.ndarray:
        """Solid angle × polarisation × film absorption × Lorentz factor of every pixel (kept for the same geometry)."""
        options = dict(
            solid_angle=bool(corrections.solid_angle), polarization=corrections.polarization,
            film_thickness_nm=corrections.film_thickness_nm, attenuation_length_um=corrections.attenuation_length_um,
            lorentz=bool(getattr(corrections, "lorentz", False)),
            lorentz_band_px=corrections.lorentz_band_px,
        )
        key = (shape, geometry, tuple(sorted(options.items())))
        cached, cached_key = self._intensity, self._intensity_key
        if cached is None or cached_key != key:
            cached = intensity_factor(shape, geometry, **options)
            self._intensity, self._intensity_key = cached, key
        return cached

    def correct_pixels(self, common: dict, raw_data, raw_valid, corrections, geometry, kind, *, mask_bitmap=None) -> bool:
        """Apply the pixel-dependent corrections shared by sample and standard. Return the counting model."""
        mirror_geometry = geometry if kind == GIWAXS else None
        common.update(self._preprocess(raw_data, raw_valid, corrections, common["metadata"], mirror_geometry, geometry,
                                       exposures=1 + len(common.get("summed_frames", ())), mask_bitmap=mask_bitmap))
        counting = common["error_model_used"] == "poisson"
        if kind == GIWAXS and geometry is not None and has_intensity_corrections(corrections):
            self._correct_intensity(common, raw_data, raw_valid, geometry, corrections, counting)
        return counting

    def _correct_intensity(self, common: dict, raw_data, raw_valid, geometry, corrections, counting: bool) -> None:
        """GIWAXS intensity corrections: ``I / factor``; the raw frame is kept for re-analysis. The scale of the
        counts is that of the flat field, when there is one, times ``1 / factor``: ``F = F_Ω F_P F_A F_ff``."""
        data, valid = common["data"], common["valid"]
        factor = self._intensity_factor((int(data.shape[0]), int(data.shape[1])), geometry, corrections)
        with np.errstate(invalid="ignore", divide="ignore"):
            corrected = (data / factor).astype(np.float32, copy=False)
        if common.get("raw_data") is None:
            common["raw_data"], common["raw_valid"] = raw_data, raw_valid
        defined = np.isfinite(factor) & (factor > 0)
        valid = valid & defined
        if corrections.mirror_fill and (geometry.tilted or geometry.sample_roll_deg):
            corrected, valid, filled = mirror_fill(corrected, valid, geometry.beam_center_x_px, geometry=geometry)
            common["filled_pixels"] = filled & defined
            valid &= defined
        common["data"], common["valid"] = corrected, valid
        for field in ("pixel_variance", "extra_variance"):
            if common.get(field) is not None:
                with np.errstate(invalid="ignore", divide="ignore"):
                    common[field] = common[field] / factor**2
        if counting:
            with np.errstate(invalid="ignore", divide="ignore"):
                scale = (1.0 / factor).astype(np.float32)
            previous = common.get("intensity_scale")  # the flat field's
            common["intensity_scale"] = scale if previous is None else (previous * scale).astype(np.float32)

    def _normalise(self, common: dict, raw_data, raw_valid, corrections, counting: bool, name: str) -> None:
        """Exposure time, monitor, transmission, thickness, absolute scale: ``c · I`` with one constant ``c`` of the
        frame (``normalisation.py``; the header values of a sum of frames are their sums). The raw frame is kept;
        photon counts get ``c`` into their scale (Poisson variance ``c × value``)."""
        use = normalisation_use(corrections.normalisation, common.get("header_values") or {}, name)
        factor = np.float32(use.factor)
        if common.get("raw_data") is None:
            common["raw_data"], common["raw_valid"] = raw_data, raw_valid
        common["data"] = (np.asarray(common["data"], dtype=np.float32) * factor).astype(np.float32, copy=False)
        for field in ("pixel_variance", "extra_variance"):
            if common.get(field) is not None:
                common[field] = common[field] * factor**2
        if counting:
            previous = common.get("intensity_scale")
            common["intensity_scale"] = (np.full(common["data"].shape, factor, dtype=np.float32) if previous is None
                                         else (previous * factor).astype(np.float32))
        common["normalisation"] = use


__all__ = ["FramePreprocessingMixin", "FrameSum", "effective_error_model", "floating", "photon_counts"]
