"""Sputter ▸ Show in Analyze: the raw frames of one time window of a Sputter result, summed, in Analyze.

The Sputter page reads the window (``WindowFrames``: its frames in order and the geometry the result was made with)
and hands it here. Analyze replaces the frames listed by them and shows their sum (``AnalyzePage.show_summed``: Sum =
their number, the first frame shown), with αi as in Sputter and, when one was given for Predict, the beam centre; a
centre the model estimated itself is only named. Distance and wavelength come from Analyze's instrument profiles: a
profile within 0.5 mm and 0.1% of the Sputter values is chosen (one of the detector the frames name first); none is
created, and without one the notice says which values to enter. The notice names the window (its time span, the
frames summed) and the geometry, and says when frames listed before were replaced; the next window's notice replaces
it. While Analyze cannot change its list (the automatic analysis, a batch, a watched folder) nothing changes and the
notice says why.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional

from .presentation.components import show_toast, visible_toasts
from .presentation.i18n import tr, trf

DISTANCE_TOLERANCE_MM = 0.5
WAVELENGTH_TOLERANCE = 1e-3
"""Relative: a profile whose wavelength is within 0.1% of the Sputter one is the same."""

SHOWN = "Window {window} ({span}): the sum of its {count} frames is shown in Analyze."
SHOWN_ONE = "Window {window} ({span}): its one frame is shown in Analyze."
PROFILE_USED = "Instrument profile “{name}”: distance {distance} mm, λ {wavelength} Å, as in Sputter."
NO_PROFILE = ("No instrument profile has the Sputter geometry: enter distance {distance} mm and λ {wavelength} Å "
              "(Geometry step ▸ Geometry…).")
NO_GEOMETRY = "The Sputter result does not record its distance and wavelength."
INCIDENCE_SET = "αi {alpha}° as in Sputter."
CENTRE_SET = "Beam centre ({x}, {y}) px as in Sputter."
CENTRE_ESTIMATED = "Beam centre: Analyze's own (the Sputter model estimated ({x}, {y}) px)."
CENTRE_UNKNOWN = "Beam centre: Analyze's own (the Sputter model estimated its own)."
REPLACED = "The frames listed in Analyze before were replaced."
NOT_SHOWN = "Not shown in Analyze: {reason}"
TEMPLATES = (SHOWN, SHOWN_ONE, PROFILE_USED, NO_PROFILE, NO_GEOMETRY, INCIDENCE_SET, CENTRE_SET, CENTRE_ESTIMATED,
             CENTRE_UNKNOWN, REPLACED, NOT_SHOWN)
NOTICE_MS = 15000
"""How long the notice stays when the geometry is complete (one that asks for values stays until closed)."""


def matching_profile(profiles: Iterable, distance_mm: Optional[float], wavelength_angstrom: Optional[float],
                     detector: str = ""):
    """The instrument profile with the Sputter distance (±0.5 mm) and wavelength (±0.1%), or None; among several,
    one whose detector is the one the frames name comes first, then the order of ``profiles``."""
    if not distance_mm or not wavelength_angstrom:
        return None
    found = [profile for profile in profiles
             if abs(profile.geometry.distance_m * 1e3 - distance_mm) <= DISTANCE_TOLERANCE_MM
             and abs(profile.geometry.wavelength_angstrom - wavelength_angstrom) <= WAVELENGTH_TOLERANCE * wavelength_angstrom]
    detector = " ".join(str(detector or "").split()).casefold()
    same = [profile for profile in found if detector and detector in " ".join(str(profile.detector_name or "").split()).casefold()]
    return (same or found or [None])[0]


def _key(path) -> str:
    try:
        path = Path(path).resolve()
    except OSError:
        path = Path(path)
    return os.path.normcase(str(path))


def span_text(window) -> str:
    """``20.00–20.45 s`` (the times of the window's first and last frame), ``t = 20.22 s`` or ``""``."""
    first, last = window.first_time_s, window.last_time_s
    if first is not None and last is not None and last > first:
        return f"{first:.2f}–{last:.2f} s"
    time = first if first is not None else window.time_s
    return "" if time is None else f"t = {time:.2f} s"


def notice_text(window, profile, *, replaced: bool) -> str:
    """The lines of the notice, in the interface language: the window and its frames, distance and wavelength,
    αi and the beam centre, and whether frames listed before were replaced."""
    geometry = window.geometry
    if window.count == 1:
        lines = [trf(SHOWN_ONE, window=window.window, span=span_text(window))]
    else:
        lines = [trf(SHOWN, window=window.window, span=span_text(window), count=window.count)]
    wavelength = geometry.wavelength_angstrom
    if profile is not None:
        lines.append(trf(PROFILE_USED, name=profile.name, distance=f"{profile.geometry.distance_m * 1e3:g}",
                         wavelength=f"{profile.geometry.wavelength_angstrom:.4f}"))
    elif geometry.sdd_mm and wavelength:
        lines.append(trf(NO_PROFILE, distance=f"{geometry.sdd_mm:g}", wavelength=f"{wavelength:.4f}"))
    else:
        lines.append(tr(NO_GEOMETRY))
    centre, estimated = geometry.beam_center, geometry.estimated_beam
    if centre is not None:
        beam = trf(CENTRE_SET, x=f"{centre[0]:.2f}", y=f"{centre[1]:.2f}")
    elif estimated is not None:
        beam = trf(CENTRE_ESTIMATED, x=f"{estimated[0]:.1f}", y=f"{estimated[1]:.1f}")
    else:
        beam = tr(CENTRE_UNKNOWN)
    alpha = geometry.alpha_i_deg
    lines.append(" ".join(([trf(INCIDENCE_SET, alpha=f"{alpha:g}")] if alpha is not None else []) + [beam]))
    if replaced:
        lines.append(tr(REPLACED))
    return "\n".join(lines)


class SputterHandoverMixin:
    """Needs the composed window: ``ui`` (its ``app_context``), ``analyze_page`` and ``_page_requested``."""

    def _instrument_profiles(self) -> list:
        repository = getattr(getattr(self.ui, "app_context", None), "instrument_profiles", None)
        try:
            return list(repository.load_all()) if repository is not None else []
        except (OSError, ValueError):  # an unreadable profile store: no profile matches
            return []

    def _show_sputter_window(self, window) -> bool:
        """Sputter ▸ Show in Analyze for ``window`` (``WindowFrames``); ``False`` when Analyze refused (said in a
        notice; Analyze stays as it was)."""
        page = self.analyze_page
        locked = page.summed_locked()
        if locked:
            show_toast(self.ui, trf(NOT_SHOWN, reason=locked), level="warning")
            return False
        geometry = window.geometry
        profile = matching_profile(self._instrument_profiles(), geometry.sdd_mm, geometry.wavelength_angstrom,
                                   window.detector)
        listed = [_key(path) for path in page.view_model.state.files]
        replaced = bool(listed) and listed != [_key(path) for path in window.frames]
        self._page_requested("analyze")
        accepted = page.show_summed(
            list(window.frames), incidence_deg=geometry.alpha_i_deg, beam_center=geometry.beam_center,
            profile_name=profile.name if profile is not None else None,
            on_imported=lambda: self._sputter_window_listed(window, profile, replaced))
        if not accepted:
            show_toast(self.ui, trf(NOT_SHOWN, reason=page.status_text()), level="warning")
        return bool(accepted)

    _window_notice = None
    """The notice of the window shown last (one at a time: the next window's replaces it)."""

    def _sputter_window_listed(self, window, profile, replaced: bool) -> None:
        """The frames are listed (the sum is being read): say what is shown and with which geometry."""
        if self._window_notice is not None and self._window_notice in visible_toasts(self.ui):
            self._window_notice.close()
        complete = profile is not None
        self._window_notice = show_toast(self.ui, notice_text(window, profile, replaced=replaced),
                                         level="ok" if complete else "warning", timeout_ms=NOTICE_MS if complete else 0)


__all__ = ["DISTANCE_TOLERANCE_MM", "NOT_SHOWN", "SputterHandoverMixin", "TEMPLATES", "WAVELENGTH_TOLERANCE",
           "matching_profile", "notice_text", "span_text"]
