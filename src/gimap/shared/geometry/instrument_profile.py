"""Named, reusable detector geometry ("instrument profile").

A profile is what calibration produces and what loading a file should reuse:
one :class:`DetectorGeometry` plus the facts used to recognise the detector
(its name from the file header and its frame shape).  Persistence lives in
``src.gimap.integrations.state.instrument_profiles``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Optional

from .detector_geometry import DetectorGeometry
from .detector_tilt import DetectorTilt

SCHEMA_VERSION = 1


def _normalized_name(value: Optional[str]) -> str:
    return " ".join(str(value or "").split()).casefold()


@dataclass(frozen=True)
class InstrumentProfile:
    name: str
    geometry: DetectorGeometry
    detector_name: Optional[str] = None
    detector_shape: Optional[tuple[int, int]] = None
    source: str = ""
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    dropped_tilt: Optional[DetectorTilt] = None
    """The tilt of the ``.poni`` the geometry was read from, which the flat ``geometry`` leaves out (``None``: none
    known). Analyze estimates from it how far |q| is off on each frame (``detector_tilt.flat_detector_q_error``)."""

    def __post_init__(self) -> None:
        if not str(self.name).strip():
            raise ValueError("An instrument profile needs a name.")
        if self.detector_shape is not None:
            rows, columns = (int(value) for value in self.detector_shape)
            if rows <= 0 or columns <= 0:
                raise ValueError(f"Invalid detector shape {self.detector_shape!r}.")
            object.__setattr__(self, "detector_shape", (rows, columns))

    def match_score(self, *, detector_name: Optional[str], shape: Optional[tuple[int, int]]) -> int:
        """0 = unrelated; higher is a better match for a loaded file.

        A known shape that differs rules the profile out, because a q map for
        another frame size is never right.  Otherwise name and shape each add.
        """
        score = 0
        if self.detector_shape is not None and shape is not None:
            if tuple(int(value) for value in shape[:2]) != self.detector_shape:
                return 0
            score += 1
        wanted = _normalized_name(detector_name)
        own = _normalized_name(self.detector_name)
        if wanted and own and (wanted == own or wanted.startswith(own) or own.startswith(wanted)):
            score += 2
        return score

    def updated(
        self, geometry: DetectorGeometry, *, source: str, dropped_tilt: Optional[DetectorTilt] = None
    ) -> "InstrumentProfile":
        """The profile with a new geometry; its ``dropped_tilt`` is the one given (a new geometry from elsewhere has
        none known)."""
        return replace(
            self,
            geometry=geometry,
            source=source,
            updated_at=datetime.now(timezone.utc).isoformat(),
            dropped_tilt=dropped_tilt,
        )

    def to_dict(self) -> dict[str, Any]:
        values = {
            "name": self.name,
            "detector_name": self.detector_name,
            "detector_shape": list(self.detector_shape) if self.detector_shape else None,
            "geometry": asdict(self.geometry),
            "source": self.source,
            "updated_at": self.updated_at,
        }
        if self.dropped_tilt is not None:  # written only when there is one: older files stay as they were
            values["dropped_tilt"] = self.dropped_tilt.to_dict()
        return values

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "InstrumentProfile":
        shape = values.get("detector_shape")
        return cls(
            name=str(values["name"]),
            geometry=DetectorGeometry(**dict(values["geometry"])),
            detector_name=values.get("detector_name"),
            detector_shape=tuple(shape) if shape else None,
            source=str(values.get("source", "")),
            updated_at=str(values.get("updated_at", "")),
            dropped_tilt=DetectorTilt.from_dict(values.get("dropped_tilt")),
        )


def best_profile(
    profiles: Iterable[InstrumentProfile],
    *,
    detector_name: Optional[str],
    shape: Optional[tuple[int, int]],
) -> Optional[InstrumentProfile]:
    """Highest-scoring profile for a loaded frame; ties go to the newest."""
    ranked = [
        (profile.match_score(detector_name=detector_name, shape=shape), profile.updated_at, profile)
        for profile in profiles
    ]
    ranked = [item for item in ranked if item[0] > 0]
    if not ranked:
        return None
    return max(ranked, key=lambda item: (item[0], item[1]))[2]
