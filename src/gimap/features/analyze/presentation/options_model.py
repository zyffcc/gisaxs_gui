"""Reduction options of the Analyze view model: corrections and GIWAXS cuts.

These live for the session (they are not written to the settings): a
background frame or a custom sector belongs to one set of measurements.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Optional

from ..application import (
    MAX_GAP_GUARD_PX,
    X_AXIS_Q,
    X_AXIS_TWO_THETA,
    Corrections,
    CutRegion,
    MaskShape,
    PlaceRegions,
    QBox,
    Sector,
    SeriesCorrection,
)
from .geometry_model import SECTION

GAP_GUARD_KEY = "gap_guard_px"
BAD_PIXELS_KEY = "bad_pixels"
DEFAULT_GAP_GUARD_PX = 3
"""Pixels next to gaps and bad pixels ignored by default (as the former Cut & Fitting page did)."""


class OptionsModelMixin:
    """Needs ``state`` (``corrections``, ``giwaxs``), ``settings`` and ``_read_setting``."""

    # -- corrections -------------------------------------------------------------------

    def set_background(self, path: Optional[str | Path], *, frame: int = 0, scale: float = 1.0) -> None:
        """Subtract ``scale ×`` this frame from every analysed frame (``None`` stops)."""
        self.state.corrections = replace(
            self.state.corrections,
            background_path=str(path) if path else None,
            background_frame=max(0, int(frame)),
            background_scale=float(scale),
        )

    def set_valid_range(self, minimum: Optional[float], maximum: Optional[float]) -> None:
        if minimum is not None and maximum is not None and minimum > maximum:
            minimum, maximum = maximum, minimum
        self.state.corrections = replace(
            self.state.corrections,
            minimum=None if minimum is None else float(minimum),
            maximum=None if maximum is None else float(maximum),
        )

    def reset_corrections(self) -> None:
        """No background or range; the gap guard, bad pixels, masks and mirror filling stay (the set-up's)."""
        current = self.state.corrections
        self.state.corrections = Corrections(
            gap_guard_px=current.gap_guard_px, bad_pixels=current.bad_pixels, mask_shapes=current.mask_shapes,
            mask_path=current.mask_path, mirror_fill=current.mirror_fill,
        )

    # -- masks and mirror filling ---------------------------------------------------------

    def set_mask_shapes(self, shapes) -> None:
        self.state.corrections = replace(self.state.corrections, mask_shapes=tuple(shapes))

    def add_mask_shape(self, shape: MaskShape) -> None:
        self.set_mask_shapes((*self.state.corrections.mask_shapes, shape))

    def set_mask_path(self, path: Optional[str | Path]) -> None:
        self.state.corrections = replace(self.state.corrections, mask_path=str(path) if path else None)

    def set_mirror_fill(self, enabled: bool) -> None:
        self.state.corrections = replace(self.state.corrections, mirror_fill=bool(enabled))

    # -- cut regions ------------------------------------------------------------------------

    def set_regions(self, regions) -> None:
        self.state.giwaxs = replace(self.state.giwaxs, regions=tuple(regions))

    def add_region(self, region: CutRegion) -> int:
        """Append a region; returns its index."""
        self.set_regions((*self.state.giwaxs.regions, region))
        return len(self.state.giwaxs.regions) - 1

    def update_region(self, index: int, region: CutRegion) -> None:
        regions = list(self.state.giwaxs.regions)
        regions[int(index)] = region
        self.set_regions(regions)

    def remove_region(self, index: int) -> None:
        regions = list(self.state.giwaxs.regions)
        del regions[int(index)]
        self.set_regions(regions)

    # Placing regions by hand (thread-safe: they read the frame and the cached q maps only).

    def region_at_pixel(self, analysis, x: float, y: float) -> tuple[float, float]:
        return PlaceRegions(self._analyze_frame).at_pixel(analysis, x, y)

    def pick_region(self, analysis, kind: str, q: float, chi: float, *, name: str, chi_band=None):
        return PlaceRegions(self._analyze_frame).pick(analysis, kind, q, chi, name=name, chi_band=chi_band)

    def snap_region(self, analysis, region: CutRegion):
        return PlaceRegions(self._analyze_frame).snap(analysis, region)

    def stored_bad_pixels(self) -> bool:
        """Settings ▸ ``analyze.bad_pixels`` (on unless turned off)."""
        value = self._read_setting(SECTION, BAD_PIXELS_KEY, True)
        return str(value).strip().lower() not in ("false", "0", "no", "off")

    def set_intensity_corrections(
        self, *, solid_angle: bool, polarization: Optional[float], film_thickness_nm: Optional[float],
        attenuation_length_um: Optional[float],
    ) -> None:
        """GIWAXS intensity corrections (``None``: that one off)."""
        self.state.corrections = replace(
            self.state.corrections, solid_angle=bool(solid_angle),
            polarization=None if polarization is None else float(polarization),
            film_thickness_nm=None if film_thickness_nm is None else float(film_thickness_nm),
            attenuation_length_um=None if attenuation_length_um is None else float(attenuation_length_um),
        )

    def set_bad_pixels(self, enabled: bool) -> None:
        """Leave out isolated hot and dead pixels found in each frame (remembered)."""
        self.state.corrections = replace(self.state.corrections, bad_pixels=bool(enabled))
        if self.settings is not None:
            self.settings.set(SECTION, BAD_PIXELS_KEY, bool(enabled))
            self.settings.save()

    def stored_gap_guard(self) -> int:
        """Settings ▸ ``analyze.gap_guard_px`` (remembered, unlike the other corrections)."""
        try:
            value = int(self._read_setting(SECTION, GAP_GUARD_KEY, DEFAULT_GAP_GUARD_PX))
        except (TypeError, ValueError):
            value = DEFAULT_GAP_GUARD_PX
        return max(0, min(MAX_GAP_GUARD_PX, value))

    def set_gap_guard(self, pixels: int) -> None:
        """Ignore pixels within ``pixels`` of a detector gap or bad pixel (0: off)."""
        pixels = max(0, min(MAX_GAP_GUARD_PX, int(pixels)))
        self.state.corrections = replace(self.state.corrections, gap_guard_px=pixels)
        if self.settings is not None:
            self.settings.set(SECTION, GAP_GUARD_KEY, pixels)
            self.settings.save()

    # -- GIWAXS cuts -------------------------------------------------------------------

    def set_sector_widths(self, in_plane_deg: float, out_of_plane_deg: float) -> None:
        self.state.giwaxs = replace(
            self.state.giwaxs,
            in_plane_half_width_deg=max(0.5, min(45.0, float(in_plane_deg))),
            out_of_plane_half_width_deg=max(0.5, min(45.0, float(out_of_plane_deg))),
        )

    def set_radial_bins(self, bins: Optional[int]) -> None:
        self.state.giwaxs = replace(self.state.giwaxs, bins=int(bins) if bins else None)

    def set_x_axis(self, axis: str) -> None:
        axis = axis if axis in (X_AXIS_Q, X_AXIS_TWO_THETA) else X_AXIS_Q
        self.state.giwaxs = replace(self.state.giwaxs, x_axis=axis)

    def set_sector(
        self,
        chi: Optional[tuple[float, float]],
        q_range: Optional[tuple[Optional[float], Optional[float]]] = None,
    ) -> None:
        """A custom χ sector (degrees) with an optional q range; ``None`` removes it."""
        if chi is None:
            self.state.giwaxs = replace(self.state.giwaxs, sector=None)
            return
        low, high = sorted(float(value) for value in chi)
        q_min, q_max = q_range if q_range is not None else (None, None)
        self.state.giwaxs = replace(
            self.state.giwaxs, sector=Sector(low, high, q_min, q_max)
        )

    def set_box(
        self, q_parallel: Optional[tuple[float, float]], qz: Optional[tuple[float, float]] = None
    ) -> None:
        """A q∥–qz rectangle for I(qz) and I(q∥) box profiles; ``None`` removes it."""
        if q_parallel is None or qz is None:
            self.state.giwaxs = replace(self.state.giwaxs, box=None)
            return
        self.state.giwaxs = replace(
            self.state.giwaxs,
            box=QBox(tuple(sorted(map(float, q_parallel))), tuple(sorted(map(float, qz)))),
        )

    # -- in-situ series ---------------------------------------------------------------

    @property
    def series_options(self) -> SeriesCorrection:
        """The series corrections chosen last time (settings ``analyze.series``)."""
        stored = self._read_setting(SECTION, "series", None)
        if not isinstance(stored, dict):
            return SeriesCorrection()
        fields = SeriesCorrection.__dataclass_fields__
        try:
            return SeriesCorrection(**{key: value for key, value in stored.items() if key in fields})
        except (TypeError, ValueError):
            return SeriesCorrection()

    def remember_series_options(self, series: SeriesCorrection) -> None:
        if self.settings is None:
            return
        self.settings.set(SECTION, "series", dict(series.__dict__))
        self.settings.save()


__all__ = ["DEFAULT_GAP_GUARD_PX", "GAP_GUARD_KEY", "OptionsModelMixin"]
