"""The geometry of one frame: the instrument profile, and where its beam centre comes from.

The beam centre is the profile's (the calibrated one) unless the session overrides it, or the
user asked to trust file headers and the file has a header centre. A session centre is a
position on one detector: it is used only on frames of the shape it was set on, so opening a
frame of another detector (another set-up) never takes it over; that frame gets its header or
profile centre and a message saying so.
"""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from .models import CENTER_HEADER, CENTER_PROFILE, CENTER_SESSION, GeometryResolution
from .ports import InstrumentProfileStore

CENTER_NOT_USED = (
    "The beam centre you set for {set_shape} frames is not used for this {shape} frame (profile centre used)."
)
CENTER_NOT_USED_HEADER = (
    "The beam centre you set for {set_shape} frames is not used for this {shape} frame (file header centre used)."
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


def shape_text(shape) -> str:
    """``1043×981``: rows × columns of a frame."""
    rows, columns = (int(value) for value in shape)
    return f"{rows}×{columns}"


def _made_from(profile, beside, path) -> bool:
    """Whether a saved profile was made from the ``.poni`` beside the frame or for frames of its folder (its source
    names the file or the folder: a centre or geometry saved in Analyze, the file loaded): then it is the newer word
    on these frames and wins over the file itself."""
    source = str(profile.source or "")
    named = beside.source.split(": ", 1)[-1]
    return bool(source) and (named in source or str(Path(path).parent) in source)


def _same_shape(first, second) -> bool:
    return tuple(int(value) for value in first) == tuple(int(value) for value in second)


class ResolveGeometry:
    """Pick the geometry for a frame.

    In this order: the profile picked by hand, when it is for frames of this size (or the person said to use it
    anyway: ``force_profile``); the ``.poni`` beside the frame (``geometry_files.SiblingPoni``), when it fits the
    frame; a profile matched by detector name and frame size; else none. A profile picked by hand for frames of
    another size is not used, and the resolution says so (``refused_profile``): a frame of another detector never
    takes the geometry in use silently. A ``.poni`` carries no incident angle: when it is used, the profile matched
    for the detector, if it has one, still gives α_i (a GISAXS frame is not reduced at α_i = 0 because of it).

    The beam centre is the geometry's (the calibrated one) unless the session overrides it, or the user asked to
    trust file headers and the file has a header centre. A session override wins over the header — on frames of
    the shape it was set on (``beam_center_shape``; ``None``: any frame).
    """

    def __init__(self, profiles: Optional[InstrumentProfileStore], sibling=None):
        self.profiles = profiles
        self.sibling = sibling

    def _pick(self, profile_name, detector_name, shape, path, pixel_size_m, force_profile):
        refused = None
        if self.profiles is not None and profile_name:
            profile = self.profiles.find(profile_name)
            if profile is not None:
                fits = profile.detector_shape is None or _same_shape(profile.detector_shape, shape)
                if fits or force_profile:
                    return profile, "chosen", None
                refused = profile
        matched = self.profiles.match(detector_name=detector_name, shape=shape) if self.profiles is not None else None
        beside = self.sibling(path, shape, pixel_size_m, detector_name) if self.sibling is not None and path else None
        if beside is not None and not (matched is not None and _made_from(matched, beside, path)):
            incidence = matched.geometry.incidence_deg if matched is not None else 0.0
            if incidence and not beside.geometry.incidence_deg:
                # A .poni carries no incident angle: the profile of this detector keeps giving it.
                beside = replace(beside, geometry=beside.geometry.with_incidence(float(incidence)))
            return beside, "file", refused
        if matched is not None:
            return matched, "matched", refused
        return None, "missing", refused

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
        path=None,
        pixel_size_m=None,
        force_profile: bool = False,
    ) -> GeometryResolution:
        profile, how, refused = self._pick(profile_name, detector_name, shape, path, pixel_size_m, force_profile)
        refusal = dict(refused_profile=refused.name, refused_shape=tuple(refused.detector_shape)) if refused else {}
        if profile is None:
            return GeometryResolution(None, None, "missing", header_center=header_center, **refusal)
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
        return GeometryResolution(geometry, profile, how, source, header_center, ignored, **refusal)


PROFILE_REFUSED = ("Profile “{name}” is for {profile_shape} frames; this {shape} frame is from another detector, so it "
                   "is not used.")


def refused_profile_message(resolution: GeometryResolution, shape) -> Optional[str]:
    """Why the profile picked by hand was not used for this frame (``None`` when it was, or none was picked)."""
    if resolution is None or not resolution.refused_profile or resolution.refused_shape is None:
        return None
    return PROFILE_REFUSED.format(name=resolution.refused_profile, profile_shape=shape_text(resolution.refused_shape),
                                  shape=shape_text(shape))


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
    "PROFILE_REFUSED",
    "refused_profile_message",
    "CENTER_NOT_USED",
    "CENTER_NOT_USED_HEADER",
    "ResolveGeometry",
    "center_not_used_message",
    "header_beam_center",
    "shape_text",
]
