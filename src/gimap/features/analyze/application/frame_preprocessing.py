"""What happens to a loaded frame before any cut: validity, masks, corrections, mirror filling.

Order: the detector's own invalid pixels → hot and dead pixels found in the frame → masks drawn
by the person → dark frame → background → flat field (its pixels that do not respond left out before the gap guard, the
division after the range, so the range applies to the values as measured) → gap guard → valid intensity range →
(GIWAXS, when asked) pixels without data filled from the mirror side (from values already divided by the flat
field) → (GIWAXS, when asked, once the geometry is known) the intensity corrections: solid angle, polarisation,
film absorption, Lorentz factor (``_correct_intensity``) → the normalisation, one constant (``_normalise``). Hot,
dead and drawn-mask pixels are part of the frame's validity; the rest are corrections, so the raw frame is kept for
re-analysis with other settings.
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
    has_intensity_corrections,
    intensity_factor,
    apply_valid_range,
    find_bad_pixels,
    guard_invalid,
    mirror_fill,
    normalisation_use,
    rasterize,
    subtract_background,
    valid_pixels,
)
from ..domain.normalisation import BAD_VALUE, NORMALISATION_TEMPLATES  # noqa: F401 - the page translates them
from ..domain.mask_edits import MaskShape
from ..domain.mask_raster import evaluate_mask
from .flat_field import FlatFieldMixin


def floating(metadata: Optional[Mapping[str, Any]]) -> bool:
    """A floating-point frame (dark- or background-subtracted): negative values are data, not gap codes.
    The one rule of ``shared/detector_io/output_values.floating`` (a big-endian ``>f4`` is float too)."""
    return _floating_dtype((metadata or {}).get("stored_dtype", ""))


def photon_counts(metadata: Optional[Mapping[str, Any]], corrections: Corrections, *, mirror_filled: bool) -> bool:
    """Whether the frame's values are photon counts as measured, so a bin's σ is Poisson (√Σ / n).

    The one place this is decided (the cuts, the peak fits of a batch, the fit-input record). Not for a
    floating-point (dark-subtracted) frame, a background-subtracted one, or one with pixels filled from the
    mirror side (``mirror_filled``: mirror filling was applied — GIWAXS with a geometry): a filled pixel
    repeats photons measured on the other side, and a bin that holds both (every I(q) bin does) would count
    them twice, so Poisson σ would be too small. σ then comes from the scatter of the bin's pixels. A dark frame
    subtracted works as a background: its read noise and offset are not counts.
    """
    return (not floating(metadata) and not corrections.background_path and not corrections.dark_path
            and not mirror_filled)


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
        self._references: dict[str, tuple] = {}
        """``background`` and ``dark``: (key, (data, valid)) of the frame last read for each."""
        self._bad_cache: Optional[tuple] = None
        self._mask_cache: Optional[tuple] = None
        self._file_mask_cache: Optional[tuple] = None
        self._intensity_key: Optional[tuple] = None
        self._intensity: Optional[np.ndarray] = None

    def _reference_frame(self, kind: str, path: str, frame: int) -> tuple[np.ndarray, np.ndarray]:
        """The background or dark frame (``kind``) and its valid pixels, loaded once per file version."""
        path = Path(str(path))
        stat = path.stat()
        key = (str(path), int(frame), stat.st_mtime_ns, stat.st_size)
        cached = self._references.get(kind)  # one tuple per kind: analyses run on several threads
        if cached is None or cached[0] != key:
            image = self.frames.load(path, int(frame))
            data = np.asarray(image.data, dtype=np.float32)
            cached = (key, (data, valid_pixels(data, image.mask, negatives_valid=floating(image.metadata))))
            self._references[kind] = cached
        return cached[1]

    def _background_frame(self, corrections: Corrections) -> tuple[np.ndarray, np.ndarray]:
        return self._reference_frame("background", corrections.background_path, corrections.background_frame)

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
        stat = Path(path).stat()
        key = (str(path), stat.st_mtime_ns, stat.st_size)
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

    def _corrected(
        self, data: np.ndarray, valid: np.ndarray, corrections: Corrections
    ) -> tuple[np.ndarray, np.ndarray]:
        """Dark frame, background, the flat field's pixels that do not respond, the gap guard (around invalid
        pixels, those too), the range (on the values as measured), then the division by the flat field."""
        if corrections.dark_path:
            dark, dark_valid = self._reference_frame("dark", corrections.dark_path, corrections.dark_frame)
            data, valid = subtract_background(data, valid, dark, dark_valid, corrections.dark_scale, what="dark")
        if corrections.background_path:
            background, background_valid = self._background_frame(corrections)
            data, valid = subtract_background(
                data, valid, background, background_valid, corrections.background_scale
            )
        flat = None
        if corrections.flat_field_path:
            flat = self._flat_field(corrections.flat_field_path, (int(data.shape[0]), int(data.shape[1])))[0]
            valid = valid & flat.kept
        valid = guard_invalid(valid, corrections.gap_guard_px)
        valid = apply_valid_range(data, valid, corrections.minimum, corrections.maximum)
        if flat is not None:
            data, valid = apply_flat_field(data, valid, flat)
        return data, valid

    def _preprocess(
        self, raw_data: np.ndarray, raw_valid: np.ndarray, corrections: Corrections,
        metadata: Mapping[str, Any], mirror_center_x: Optional[float], geometry=None,
    ) -> dict:
        """The frame to reduce and what was done to it (fields of ``FrameAnalysis``).

        ``mirror_center_x``: the beam-centre column when the frame is GIWAXS with a geometry
        (mirror filling needs it); ``None`` never fills.
        """
        bad = self._bad_pixels(raw_data, raw_valid, metadata) if corrections.bad_pixels else None
        usable = raw_valid
        if bad is not None and bad.mask.any():
            usable = usable & ~bad.mask
        drawn = self._user_mask(raw_data, usable, corrections, geometry)
        if drawn is not None and drawn.any():
            usable = usable & ~drawn
        record: dict = {"corrections": corrections, "bad_pixels": bad, "drawn_mask": drawn}
        if corrections.is_identity:
            record.update(data=raw_data, valid=usable)
            if usable is not raw_valid:
                record["raw_valid"] = raw_valid  # re-analysing without them starts from the detector's mask
            return record
        data, valid = self._corrected(raw_data, usable, corrections)
        if corrections.flat_field_path:
            record["flat_field"] = self._flat_field(corrections.flat_field_path)[1]
        filled = None
        if corrections.mirror_fill and mirror_center_x is not None:
            data, valid, filled = mirror_fill(data, valid, mirror_center_x)
        record.update(data=data, valid=valid, raw_data=raw_data, raw_valid=raw_valid, filled_pixels=filled)
        return record

    def _intensity_factor(self, shape: tuple[int, int], geometry: DetectorGeometry, corrections) -> np.ndarray:
        """Solid angle × polarisation × film absorption × Lorentz factor of every pixel (kept for the same geometry)."""
        options = dict(
            solid_angle=bool(corrections.solid_angle), polarization=corrections.polarization,
            film_thickness_nm=corrections.film_thickness_nm, attenuation_length_um=corrections.attenuation_length_um,
            lorentz=bool(getattr(corrections, "lorentz", False)),
        )
        key = (shape, geometry, tuple(sorted(options.items())))
        cached, cached_key = self._intensity, self._intensity_key
        if cached is None or cached_key != key:
            cached = intensity_factor(shape, geometry, **options)
            self._intensity, self._intensity_key = cached, key
        return cached

    def correct_pixels(self, common: dict, raw_data, raw_valid, corrections, geometry, kind) -> bool:
        """Apply the pixel-dependent corrections shared by sample and standard. Return the counting model."""
        mirror_x = geometry.beam_center_x_px if geometry is not None and kind == GIWAXS else None
        common.update(self._preprocess(raw_data, raw_valid, corrections, common["metadata"], mirror_x, geometry))
        counting = photon_counts(common["metadata"], corrections, mirror_filled=common.get("filled_pixels") is not None)
        self._flat_scale(common, corrections, counting)
        if kind == GIWAXS and geometry is not None and has_intensity_corrections(corrections):
            self._correct_intensity(common, raw_data, raw_valid, geometry, corrections, counting)
        return counting

    def _flat_scale(self, common: dict, corrections, counting: bool) -> None:
        """Photon counts divided by a flat field: each pixel's counts were multiplied by ``1 / F_ff``, so its
        Poisson variance is that factor × its value (``intensity_scale``)."""
        if counting and common.get("flat_field") is not None:
            with np.errstate(invalid="ignore", divide="ignore"):
                common["intensity_scale"] = (1.0 / self._flat_field(corrections.flat_field_path)[0].factor).astype(
                    np.float32)

    def _correct_intensity(self, common: dict, raw_data, raw_valid, geometry, corrections, counting: bool) -> None:
        """GIWAXS intensity corrections: ``I / factor``; the raw frame is kept for re-analysis. The scale of the
        counts is that of the flat field, when there is one, times ``1 / factor``: ``F = F_Ω F_P F_A F_ff``."""
        data, valid = common["data"], common["valid"]
        factor = self._intensity_factor((int(data.shape[0]), int(data.shape[1])), geometry, corrections)
        with np.errstate(invalid="ignore", divide="ignore"):
            corrected = (data / factor).astype(np.float32, copy=False)
        if common.get("raw_data") is None:
            common["raw_data"], common["raw_valid"] = raw_data, raw_valid
        common["data"], common["valid"] = corrected, valid & np.isfinite(factor) & (factor > 0)
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
        if counting:
            previous = common.get("intensity_scale")
            common["intensity_scale"] = (np.full(common["data"].shape, factor, dtype=np.float32) if previous is None
                                         else (previous * factor).astype(np.float32))
        common["normalisation"] = use


__all__ = ["FramePreprocessingMixin", "FrameSum", "floating", "photon_counts"]
