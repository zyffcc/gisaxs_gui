"""Geometry part of the Analyze view model: profiles, beam centre, fit side.

The beam centre of a frame is the instrument profile's (the calibrated one)
unless the user moves it for this session, or opts in to file-header
centres (Settings ▸ Analyze).  A session centre stays for every following
frame until it is reset or saved into the profile.
"""

from __future__ import annotations

from typing import Any, Optional

from ..application import (
    DetectorGeometry,
    InstrumentProfile,
    SymmetryCenter,
    geometry_from_fitting_settings,
    refine_center_by_symmetry,
)

SECTION = "analyze"
HEADER_CENTER_KEY = "use_header_beam_center"
FIT_SIDE_KEY = "fit_side"
BOTH = "both_abs"
FIT_SIDES = (BOTH, "mean", "negative", "positive")
"""How Fitting shows the horizontal cut: both halves on |qy| (two colours),
their average, or one half.  Fitting's own q view applies it."""


class GeometryModelMixin:
    """Needs ``state``, ``settings``, ``_read_setting``, ``_save_profile`` and ``_profiles``."""

    # -- beam centre -------------------------------------------------------------------

    @property
    def use_header_center(self) -> bool:
        """Settings ▸ Analyze: trust the beam centre in file headers (off by default)."""
        return bool(self._read_setting(SECTION, HEADER_CENTER_KEY, False))

    def set_beam_center(self, x: float, y: float) -> None:
        """Use this centre for this and every following frame until reset or saved."""
        self.state.beam_center = (float(x), float(y))

    def clear_beam_center(self) -> None:
        self.state.beam_center = None

    def refine_center_x(self) -> SymmetryCenter:
        """GISAXS: move the centre column to the symmetry axis of the horizontal cut (session)."""
        analysis = self.state.analysis
        if analysis is None or analysis.geometry is None:
            raise ValueError("Open a frame with a geometry first.")
        result = refine_center_by_symmetry(analysis)
        self.set_beam_center(result.x_px, analysis.geometry.beam_center_y_px)
        return result

    def center_state(self) -> dict[str, Any]:
        """What the beam-centre control shows: position, origin and the available actions."""
        analysis = self.state.analysis
        geometry = analysis.geometry if analysis is not None else None
        resolution = analysis.resolution if analysis is not None else None
        profile = resolution.profile if resolution is not None else None
        return {
            "center": (
                (geometry.beam_center_x_px, geometry.beam_center_y_px) if geometry else None
            ),
            "source": resolution.center_source if resolution is not None else None,
            "profile_name": profile.name if profile is not None else None,
            "profile_center": (
                (profile.geometry.beam_center_x_px, profile.geometry.beam_center_y_px)
                if profile is not None
                else None
            ),
            "header_center": resolution.header_center if resolution is not None else None,
            "overridden": self.state.beam_center is not None,
        }

    def save_center_to_profile(self) -> InstrumentProfile:
        """Store the centre in use in the instrument profile and end the session override."""
        analysis = self.state.analysis
        if analysis is None or analysis.geometry is None or analysis.resolution.profile is None:
            raise ValueError("Open a frame with an instrument profile first.")
        profile = analysis.resolution.profile
        geometry = analysis.geometry
        saved = self._save_profile(
            profile.name,
            profile.geometry.with_beam_center(geometry.beam_center_x_px, geometry.beam_center_y_px),
            detector_name=profile.detector_name,
            shape=profile.detector_shape,
            source="beam centre set in Analyze",
        )
        self.state.beam_center = None
        return saved

    # -- fitting hand-over ------------------------------------------------------------

    @property
    def fit_side(self) -> str:
        side = self._read_setting(SECTION, FIT_SIDE_KEY, BOTH)
        return side if side in FIT_SIDES else BOTH

    def set_fit_side(self, side: str) -> None:
        if side not in FIT_SIDES or self.settings is None:
            return
        self.settings.set(SECTION, FIT_SIDE_KEY, side)
        self.settings.save()

    # -- profiles ----------------------------------------------------------------------

    def profile_names(self) -> list[str]:
        if self._profiles is None:
            return []
        return [profile.name for profile in self._profiles.load_all()]

    def suggested_profile_name(self) -> str:
        analysis = self.state.analysis
        if analysis is None:
            return "Detector"
        detector = " ".join(str(analysis.detector_name or "").split()) or "Detector"
        return f"{detector} {analysis.shape[0]}×{analysis.shape[1]}"

    def fitting_geometry(self) -> Optional[DetectorGeometry]:
        """The former Cut & Fitting geometry in the canonical frame of this frame, if stored."""
        analysis = self.state.analysis
        if analysis is None:
            return None
        return geometry_from_fitting_settings(self._read_setting, analysis.shape[0])

    def save_profile(self, name: str, geometry: DetectorGeometry, *, source: str) -> InstrumentProfile:
        analysis = self.state.analysis
        profile = self._save_profile(
            name,
            geometry,
            detector_name=analysis.detector_name if analysis else None,
            shape=analysis.shape if analysis else None,
            source=source,
        )
        self.state.profile_name = None  # let the new profile match automatically
        return profile

    def delete_profile(self, name: str) -> bool:
        if self._profiles is None or not self._profiles.delete(name):
            return False
        if self.state.profile_name == name:
            self.state.profile_name = None
        return True


__all__ = ["BOTH", "FIT_SIDES", "FIT_SIDE_KEY", "GeometryModelMixin", "HEADER_CENTER_KEY", "SECTION"]
