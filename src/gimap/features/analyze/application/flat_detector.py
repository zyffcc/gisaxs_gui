"""What GIMaP's flat detector costs on the frame shown, when its geometry comes from a tilted ``.poni``.

The instrument profile keeps the tilt the flat geometry leaves out (``InstrumentProfile.dropped_tilt``). For a
frame analysed with that profile the tilted ``.poni`` geometry is compared, on the frame's valid pixels, with the
flat geometry GIMaP made from that same ``.poni`` (``DetectorTilt.flat_geometry``: its direct beam and distance;
``shared/geometry/detector_tilt.flat_detector_q_error``). So the estimate is what the tilt alone costs: a beam
centre moved afterwards (the GISAXS symmetry axis, a centre set by hand) or another distance does not change it.
The result goes into the frame's record (``<stem>_analysis.json``), the status the automatic analysis reads, and —
above 1° of tilt or 0.5 % in |q| — a message on the frame (the status line in Analyze). The last comparison is
kept, so re-analysing the same frame does not compute it again.
"""

from __future__ import annotations

import hashlib
from typing import Optional

import numpy as np

from src.gimap.shared.geometry import DetectorGeometry, InstrumentProfile
from src.gimap.shared.geometry.detector_tilt import FlatDetectorError, flat_detector_q_error, relative_q_error_map

FLAT_DETECTOR_MESSAGE = (
    "The calibration describes a detector tilted by {tilt}°; GIMaP's geometry is a flat detector perpendicular to "
    "the beam, so the tilt is left out and |q| is off by up to {error} % on this frame."
)
"""A frame's message (``FrameAnalysis.messages``) when the tilt left out matters (``FlatDetectorError.needs_attention``)."""

_map: tuple = (None, None)
"""The last per-pixel error map, for its (shape, tilt): the frames of a series share it."""
_last: tuple = (None, None)
"""The last comparison, for its map and valid pixels: re-analysing a frame does not compute it again."""


def flat_detector_error(
    profile: Optional[InstrumentProfile], geometry: Optional[DetectorGeometry], shape: tuple[int, int],
    valid: np.ndarray,
) -> Optional[FlatDetectorError]:
    """The comparison for the frame, ``None`` when the profile names no dropped tilt (or there is no geometry).
    ``geometry`` (the one in use) only says that the frame is in q: the tilt is compared with the flat geometry made
    from its own ``.poni``, so a centre moved since is not counted as the tilt's error."""
    global _map, _last
    tilt = getattr(profile, "dropped_tilt", None) if profile is not None else None
    if tilt is None or geometry is None:
        return None
    shape = tuple(int(value) for value in shape)
    usable = np.asarray(valid, dtype=bool)
    flat = tilt.flat_geometry()  # the .poni's own direct beam and distance; |q| errors depend on neither λ nor αi
    where = (shape, tilt)
    cached, last = _map, _last  # read once: a batch on a worker thread may replace them meanwhile
    if cached[0] != where:
        cached = _map = (where, relative_q_error_map(tilt, flat, shape))
    key = (where, hashlib.sha1(np.packbits(usable).tobytes()).hexdigest())
    if last[0] == key:
        return last[1]
    estimate = flat_detector_q_error(tilt, flat, shape, usable, error_map=cached[1])
    _last = (key, estimate)
    return estimate


def flat_detector_message(estimate: Optional[FlatDetectorError]) -> Optional[str]:
    """The frame's message for ``estimate``; ``None`` below the limits."""
    if estimate is None or not estimate.needs_attention:
        return None
    error = "?" if estimate.q_error_max is None else f"{100 * estimate.q_error_max:.2g}"
    return FLAT_DETECTOR_MESSAGE.format(tilt=f"{estimate.tilt_deg:.2f}", error=error)


def flat_detector_record(estimate: Optional[FlatDetectorError]) -> dict:
    """The keys added to the record's ``geometry``: the tilt left out (degrees, and the three rotations) and the
    largest and RMS relative |q| error (fractions). ``None`` values when no tilt is known."""
    if estimate is None:
        return {"dropped_tilt_deg": None, "dropped_rotations_deg": None, "flat_detector_q_error_max": None,
                "flat_detector_q_error_rms": None}
    values = estimate.to_dict()
    return {
        "dropped_tilt_deg": values["dropped_tilt_deg"], "dropped_rotations_deg": values["dropped_rotations_deg"],
        "flat_detector_q_error_max": values["q_error_max"], "flat_detector_q_error_rms": values["q_error_rms"],
        "flat_detector_pixels_compared": values["pixels_compared"], "flat_detector_limits": values["limits"],
    }


__all__ = ["FLAT_DETECTOR_MESSAGE", "flat_detector_error", "flat_detector_message", "flat_detector_record"]
