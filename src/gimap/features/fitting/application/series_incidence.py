"""Frames of another incidence angle in an In-situ series.

A run can switch the incidence angle between frames (a reference angle, and a few others between spray pulses).
Analyze writes the αi a cut was computed with into the ``# observation:`` line of its fit input (``alpha_i_deg``).
A series fits the frames of one angle: that of the model set up in Single analysis (Model ▸ Geometry). A frame
measured at another angle is not fitted and not chained — its curve would be fitted with the wrong angle, and a
chain would carry the result of one angle into the start of another. The frame is listed with the reason
(state ``other angle``). Frames whose file states no αi, and models without an αi (0), are fitted as before.
"""

from __future__ import annotations

import math
import re
from typing import Mapping, Optional

TOLERANCE_DEG = 0.005
"""Two angles closer than this are the same (the αi of one motor position, written to four digits)."""
OTHER_ANGLE = ("measured at αi {frame}°, the model's αi is {model}°: frames of another angle are not fitted in "
               "this series (set the model up on a frame of that angle for a series of its own)")
_PATTERN = re.compile(re.escape(OTHER_ANGLE).replace(r"\{frame\}", r"(?P<frame>.+?)").replace(
    r"\{model\}", r"(?P<model>.+?)"), re.DOTALL)


def _angle(value) -> Optional[float]:
    try:
        angle = float(value)
    except (TypeError, ValueError):
        return None
    return angle if math.isfinite(angle) and 0.0 < angle < 90.0 else None


def frame_incidence(observation: Optional[Mapping]) -> Optional[float]:
    """The αi a curve file states (``alpha_i_deg`` of its observation), or ``None``."""
    return _angle(observation.get("alpha_i_deg")) if isinstance(observation, Mapping) else None


def model_incidence(model) -> Optional[float]:
    """The model's αi (Model ▸ Geometry), or ``None`` when it has none (0)."""
    try:
        return _angle(model.globals["alpha_i"].value)
    except (AttributeError, KeyError, TypeError):
        return None


def other_angle(model, observation: Optional[Mapping]) -> str:
    """Why a curve is not fitted with ``model`` (``OTHER_ANGLE``), or ``""`` when its angle is the model's or
    either is not known."""
    frame, wanted = frame_incidence(observation), model_incidence(model)
    if frame is None or wanted is None or abs(frame - wanted) <= TOLERANCE_DEG:
        return ""
    return OTHER_ANGLE.format(frame=f"{frame:.4g}", model=f"{wanted:.4g}")


def other_angle_values(message: str) -> Optional[dict]:
    """The values of an ``OTHER_ANGLE`` reason (to say it again in another language), or ``None``."""
    found = _PATTERN.fullmatch(str(message))
    return found.groupdict() if found else None


__all__ = ["OTHER_ANGLE", "TOLERANCE_DEG", "frame_incidence", "model_incidence", "other_angle", "other_angle_values"]
