"""A calibrant measured in transmission is not a GISAXS sample.

Auto tells GISAXS from GIWAXS by the largest scattering angle, so a silver behenate frame at a SAXS distance is
"GISAXS" and its rings would be fitted with particles for minutes. When nobody gave an incidence angle and the frame
shows complete sharp rings about the direct beam (``analyze/domain/ring_frame.py``), the run says so and fits no
particle model; a given αi or technique is the person's statement and is never second-guessed."""

from __future__ import annotations

from typing import Optional

TRANSMISSION_ITEM = "technique"
TRANSMISSION_DECISION = "a transmission frame of a calibrant or powder ({rings} complete rings about the beam centre)"
TRANSMISSION_WHY = (
    "no incidence angle is known and the frame shows {rings} sharp rings concentric with the direct beam that run "
    "through every direction the detector covers, as a powder in transmission gives; a grazing-incidence frame is "
    "shadowed below the sample horizon"
)
TRANSMISSION_ATTENTION = (
    "This frame looks like a transmission measurement of a calibrant or powder ({rings} complete rings about the "
    "beam centre, no incidence angle), not like a GISAXS sample: its cuts were written, but no particle model was "
    "fitted."
)
TRANSMISSION_HINT = (
    "If it is a calibrant, use it in Tools ▸ Geometry Calibration. If it is a grazing-incidence frame, give αi or "
    "the technique and run again."
)
NO_FIT = "the frame is a transmission frame of a calibrant or powder, not a GISAXS sample"


def transmission_rings(workbench, status: Optional[dict], incidence_given: bool) -> Optional[int]:
    """The number of complete rings when the frame is to be treated as a transmission frame; ``None`` otherwise
    (αi given or set, no geometry, a workbench that cannot look, or fewer rings than the rule asks for)."""
    geometry = (status or {}).get("geometry") or {}
    if incidence_given or not geometry or abs(float(geometry.get("incidence_deg") or 0.0)) > 1e-9:
        return None
    look = getattr(workbench, "ring_frame", None)
    if look is None:
        return None
    try:
        found = look() or {}
    except Exception:  # a frame that cannot be judged is analysed as before
        return None
    return int(found.get("complete_rings") or 0) if found.get("is_ring_frame") else None


__all__ = ["NO_FIT", "TRANSMISSION_ATTENTION", "TRANSMISSION_DECISION", "TRANSMISSION_HINT", "TRANSMISSION_ITEM",
           "TRANSMISSION_WHY", "transmission_rings"]
