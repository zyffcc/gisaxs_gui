"""The geometry of one frame: the instrument profile, and where its beam centre comes from.

The beam centre is the profile's (the calibrated one) unless the session overrides it, or the
user asked to trust file headers and the file has a header centre. A session centre is a
position on one detector: it is used only on frames of the shape it was set on, so opening a
frame of another detector (another set-up) never takes it over; that frame gets its header or
profile centre and a message saying so.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Callable, Optional

from src.gimap.shared.geometry import InstrumentProfile

from .models import CENTER_HEADER, CENTER_PROFILE, CENTER_SESSION, GeometryResolution
from .ports import InstrumentProfileStore

CENTER_NOT_USED = (
    "The beam centre you set for {set_shape} frames is not used for this {shape} frame (profile centre used)."
)
CENTER_NOT_USED_HEADER = (
    "The beam centre you set for {set_shape} frames is not used for this {shape} frame (file header centre used)."
)


def shape_text(shape) -> str:
    """``1043×981``: rows × columns of a frame."""
    rows, columns = (int(value) for value in shape)
    return f"{rows}×{columns}"


def _same_shape(first, second) -> bool:
    return tuple(int(value) for value in first) == tuple(int(value) for value in second)


class ResolveGeometry:
    """Pick the geometry for a frame: an explicit profile, else the best match.

    The beam centre is the profile's (the calibrated one) unless the session
    overrides it, or the user asked to trust file headers and the file has a
    header centre.  A session override wins over the header — on frames of
    the shape it was set on (``beam_center_shape``; ``None``: any frame).
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
        beam_center_shape: Optional[tuple[int, int]] = None,
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
        fits = beam_center is not None and (beam_center_shape is None or _same_shape(beam_center_shape, shape))
        ignored = (
            tuple(int(value) for value in beam_center_shape)
            if beam_center is not None and not fits and beam_center_shape is not None else None
        )
        if fits:
            geometry = geometry.with_beam_center(*(float(value) for value in beam_center))
            source = CENTER_SESSION
        elif use_header_center and header_center is not None:
            geometry = geometry.with_beam_center(*header_center)
            source = CENTER_HEADER
        if incidence_deg is not None:
            geometry = geometry.with_incidence(float(incidence_deg))
        if distance_m is not None:
            geometry = replace(geometry, distance_m=float(distance_m))
        return GeometryResolution(geometry, profile, how, source, header_center, ignored)


def center_not_used_message(
    resolution: GeometryResolution, shape, translate: Callable[[str], str] = str,
) -> Optional[str]:
    """Why the session beam centre was not used for this frame (``None`` when it was, or none was set).

    ``translate`` turns the English template into the interface language before it is filled in.
    """
    if resolution is None or resolution.ignored_center_shape is None or resolution.geometry is None:
        return None
    template = CENTER_NOT_USED_HEADER if resolution.center_source == CENTER_HEADER else CENTER_NOT_USED
    return translate(template).format(set_shape=shape_text(resolution.ignored_center_shape), shape=shape_text(shape))


__all__ = [
    "CENTER_NOT_USED",
    "CENTER_NOT_USED_HEADER",
    "ResolveGeometry",
    "center_not_used_message",
    "shape_text",
]
