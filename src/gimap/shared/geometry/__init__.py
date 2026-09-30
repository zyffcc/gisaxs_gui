"""Detector geometry and pixel → q mapping shared by every feature.

See ``docs/architecture/geometry.md`` for the conventions.
"""

from .detector_geometry import DetectorGeometry
from .instrument_profile import InstrumentProfile, best_profile
from .q_mapping import (
    ExitAngleModel,
    GrazingQMap,
    TransmissionMap,
    exit_directions,
    grazing_q_map,
    grazing_q_region,
    pixel_center_displacements,
    region_displacements,
    scattering_vectors,
    signed_parallel,
    transmission_map,
)

__all__ = [
    "DetectorGeometry",
    "ExitAngleModel",
    "GrazingQMap",
    "InstrumentProfile",
    "TransmissionMap",
    "best_profile",
    "exit_directions",
    "grazing_q_map",
    "grazing_q_region",
    "pixel_center_displacements",
    "region_displacements",
    "scattering_vectors",
    "signed_parallel",
    "transmission_map",
]
