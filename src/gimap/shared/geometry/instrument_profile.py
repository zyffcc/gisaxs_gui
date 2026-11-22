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
from .detector_response import DetectorResponse
from .detector_tilt import DetectorTilt

SCHEMA_VERSION = 1
TILT_KEY, OLD_TILT_KEY = "tilt", "dropped_tilt"
"""The key a profile's ``.poni`` tilt is saved under. Up to 0.2.0 it was ``dropped_tilt`` (when the geometry left
the tilt out); the tilt is applied now, so it is written as ``tilt`` and the old key is still read."""
OTHER_SCALE_SETUP = "This standard calibration belongs to another instrument set-up."


def _normalized_name(value: Optional[str]) -> str:
    return " ".join(str(value or "").split()).casefold()


def scale_geometry(geometry: DetectorGeometry) -> dict[str, float]:
    """Detector dimensions, orientation and wavelength used to identify the calibrated set-up."""
    return {key: getattr(geometry, key) for key in (
        "distance_m", "wavelength_angstrom", "pixel_size_x_m", "pixel_size_y_m",
        "rot1_rad", "rot2_rad", "rot3_rad")}


@dataclass(frozen=True)
class InstrumentProfile:
    name: str
    geometry: DetectorGeometry
    detector_name: Optional[str] = None
    detector_shape: Optional[tuple[int, int]] = None
    source: str = ""
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    dropped_tilt: Optional[DetectorTilt] = None
    """The tilt of the ``.poni`` the geometry was read from (``None``: none known). The name is historical: up to
    0.2 the geometry left this tilt out. Now a geometry without rotations takes the tilt's rotations
    (``__post_init__``), so a profile saved by an older GIMaP is read with its detector tilted as the ``.poni`` says.
    """
    standard_calibration: Optional[dict] = field(default=None, hash=False)
    """Absolute intensity scale and its measurement record, saved separately from the geometry source."""
    detector_response: DetectorResponse = field(default_factory=DetectorResponse)

    def __post_init__(self) -> None:
        if not str(self.name).strip():
            raise ValueError("An instrument profile needs a name.")
        if self.detector_shape is not None:
            rows, columns = (int(value) for value in self.detector_shape)
            if rows <= 0 or columns <= 0:
                raise ValueError(f"Invalid detector shape {self.detector_shape!r}.")
            object.__setattr__(self, "detector_shape", (rows, columns))
        tilt = self.dropped_tilt
        if tilt is not None and not self.geometry.requires_projection:
            object.__setattr__(
                self, "geometry", replace(self.geometry, rot1_rad=tilt.rot1_rad, rot2_rad=tilt.rot2_rad,
                                          rot3_rad=tilt.rot3_rad, orientation=tilt.orientation)
            )

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

    def check_scale_setup(self, record, geometry, shape) -> None:
        required_shape = record.get("detector_shape")
        if (self.name != record.get("instrument_profile") or geometry is None
                or scale_geometry(geometry) != record.get("instrument_geometry")
                or (required_shape and shape and list(shape) != required_shape)):
            raise ValueError(OTHER_SCALE_SETUP)

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
            standard_calibration=(self.standard_calibration if scale_geometry(self.geometry) == scale_geometry(geometry)
                                  else None),
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
            values[TILT_KEY] = self.dropped_tilt.to_dict()
        if self.standard_calibration is not None:
            values["standard_calibration"] = self.standard_calibration
        if self.detector_response != DetectorResponse():
            values["detector_response"] = asdict(self.detector_response)
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
            dropped_tilt=DetectorTilt.from_dict(values.get(TILT_KEY) or values.get(OLD_TILT_KEY)),
            standard_calibration=values.get("standard_calibration"),
            detector_response=DetectorResponse(**(values.get("detector_response") or {})),
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
