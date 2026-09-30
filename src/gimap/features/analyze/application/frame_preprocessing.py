"""What happens to a loaded frame before any cut: validity, masks, corrections, mirror filling.

Order: the detector's own invalid pixels → hot and dead pixels found in the frame → masks drawn
by the person → background → gap guard → valid intensity range → (GIWAXS, when asked) pixels
without data filled from the mirror side → (GIWAXS, when asked, once the geometry is known) the
intensity corrections: solid angle, polarisation, film absorption (``_correct_intensity``). Hot, dead and drawn-mask pixels are part of the frame's
validity; the rest are corrections, so the raw frame is kept for re-analysis with other settings.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

from src.gimap.shared.geometry import DetectorGeometry

from ..domain import (
    Corrections,
    intensity_factor,
    apply_valid_range,
    find_bad_pixels,
    guard_invalid,
    mirror_fill,
    rasterize,
    subtract_background,
    valid_pixels,
)


def floating(metadata: Optional[Mapping[str, Any]]) -> bool:
    """A floating-point frame (dark- or background-subtracted): negative values are data, not gap codes."""
    return str((metadata or {}).get("stored_dtype", "")).startswith("float")


class FramePreprocessingMixin:
    """Needs ``frames`` (a ``FrameSource``); call ``_init_preprocessing()`` in ``__init__``."""

    def _init_preprocessing(self) -> None:
        self._background_key: Optional[tuple] = None
        self._background: Optional[tuple[np.ndarray, np.ndarray]] = None
        self._bad_cache: Optional[tuple] = None
        self._mask_cache: Optional[tuple] = None
        self._file_mask_cache: Optional[tuple] = None
        self._intensity_key: Optional[tuple] = None
        self._intensity: Optional[np.ndarray] = None

    def _background_frame(self, corrections: Corrections) -> tuple[np.ndarray, np.ndarray]:
        """The background frame and its valid pixels, loaded once per file version."""
        path = Path(str(corrections.background_path))
        stat = path.stat()
        key = (str(path), int(corrections.background_frame), stat.st_mtime_ns, stat.st_size)
        cached, cached_key = self._background, self._background_key
        if cached is None or cached_key != key:
            image = self.frames.load(path, int(corrections.background_frame))
            data = np.asarray(image.data, dtype=np.float32)
            cached = (data, valid_pixels(data, image.mask, negatives_valid=floating(image.metadata)))
            self._background, self._background_key = cached, key
        return cached

    def _bad_pixels(self, data: np.ndarray, valid: np.ndarray, metadata: Mapping[str, Any]):
        """Hot and dead pixels of the raw frame (kept while the same frame is re-analysed)."""
        cache = self._bad_cache  # one tuple: analyses run on several threads
        if cache is None or cache[0] is not data or cache[1] is not valid:
            cache = (data, valid, find_bad_pixels(data, valid, counts=not floating(metadata)))
            self._bad_cache = cache
        return cache[2]

    def _drawn_mask(self, shapes: tuple, frame_shape: tuple[int, int]) -> np.ndarray:
        cache = self._mask_cache
        key = (tuple(shapes), tuple(frame_shape))
        if cache is None or cache[0] != key:
            cache = (key, rasterize(shapes, frame_shape))
            self._mask_cache = cache
        return cache[1]

    def _file_mask(self, path: str, frame_shape: tuple[int, int]) -> np.ndarray:
        """Non-zero pixels of a mask image, read once per file version."""
        stat = Path(path).stat()
        key = (str(path), stat.st_mtime_ns, stat.st_size)
        cache = self._file_mask_cache
        if cache is None or cache[0] != key:
            image = self.frames.load(Path(path), 0)
            cache = (key, np.asarray(image.data) != 0)
            self._file_mask_cache = cache
        mask = cache[1]
        if mask.shape != tuple(frame_shape):
            raise ValueError(
                f"The mask {Path(path).name} is {mask.shape[0]}×{mask.shape[1]}; the frame is "
                f"{frame_shape[0]}×{frame_shape[1]}."
            )
        return mask

    def _corrected(
        self, data: np.ndarray, valid: np.ndarray, corrections: Corrections
    ) -> tuple[np.ndarray, np.ndarray]:
        """Background, then the gap guard (around invalid pixels), then the range."""
        if corrections.background_path:
            background, background_valid = self._background_frame(corrections)
            data, valid = subtract_background(
                data, valid, background, background_valid, corrections.background_scale
            )
        valid = guard_invalid(valid, corrections.gap_guard_px)
        return data, apply_valid_range(data, valid, corrections.minimum, corrections.maximum)

    def _preprocess(
        self, raw_data: np.ndarray, raw_valid: np.ndarray, corrections: Corrections,
        metadata: Mapping[str, Any], mirror_center_x: Optional[float],
    ) -> dict:
        """The frame to reduce and what was done to it (fields of ``FrameAnalysis``).

        ``mirror_center_x``: the beam-centre column when the frame is GIWAXS with a geometry
        (mirror filling needs it); ``None`` never fills.
        """
        shape = (int(raw_data.shape[0]), int(raw_data.shape[1]))
        bad = self._bad_pixels(raw_data, raw_valid, metadata) if corrections.bad_pixels else None
        drawn = self._drawn_mask(corrections.mask_shapes, shape) if corrections.mask_shapes else None
        if corrections.mask_path:
            file_mask = self._file_mask(corrections.mask_path, shape)
            drawn = file_mask if drawn is None else drawn | file_mask
        usable = raw_valid
        if bad is not None and bad.mask.any():
            usable = usable & ~bad.mask
        if drawn is not None and drawn.any():
            usable = usable & ~drawn
        record: dict = {"corrections": corrections, "bad_pixels": bad, "drawn_mask": drawn}
        if corrections.is_identity:
            record.update(data=raw_data, valid=usable)
            if usable is not raw_valid:
                record["raw_valid"] = raw_valid  # re-analysing without them starts from the detector's mask
            return record
        data, valid = self._corrected(raw_data, usable, corrections)
        filled = None
        if corrections.mirror_fill and mirror_center_x is not None:
            data, valid, filled = mirror_fill(data, valid, mirror_center_x)
        record.update(data=data, valid=valid, raw_data=raw_data, raw_valid=raw_valid, filled_pixels=filled)
        return record

    def _intensity_factor(self, shape: tuple[int, int], geometry: DetectorGeometry, corrections) -> np.ndarray:
        """Solid angle × polarisation × film absorption of every pixel (kept for the same geometry)."""
        options = dict(
            solid_angle=bool(corrections.solid_angle), polarization=corrections.polarization,
            film_thickness_nm=corrections.film_thickness_nm, attenuation_length_um=corrections.attenuation_length_um,
        )
        key = (shape, geometry, tuple(sorted(options.items())))
        cached, cached_key = self._intensity, self._intensity_key
        if cached is None or cached_key != key:
            cached = intensity_factor(shape, geometry, **options)
            self._intensity, self._intensity_key = cached, key
        return cached

    def _correct_intensity(self, common: dict, raw_data, raw_valid, geometry, corrections, counting: bool) -> None:
        """GIWAXS intensity corrections: ``I / factor``; the raw frame is kept for re-analysis."""
        data, valid = common["data"], common["valid"]
        factor = self._intensity_factor((int(data.shape[0]), int(data.shape[1])), geometry, corrections)
        with np.errstate(invalid="ignore", divide="ignore"):
            corrected = (data / factor).astype(np.float32, copy=False)
        if common.get("raw_data") is None:
            common["raw_data"], common["raw_valid"] = raw_data, raw_valid
        common["data"], common["valid"] = corrected, valid & np.isfinite(factor) & (factor > 0)
        if counting:
            with np.errstate(invalid="ignore", divide="ignore"):
                common["intensity_scale"] = (1.0 / factor).astype(np.float32)


__all__ = ["FramePreprocessingMixin", "floating"]
