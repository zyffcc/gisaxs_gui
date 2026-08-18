"""不依赖 GUI、runtime 与 I/O 的 Calibration domain API。"""

from .models import (
    CalibrationCandidate,
    CalibrationResult,
    CalibrationStandard,
    DetectorImage,
    MatchedRing,
)
from .requests import CalibrationRequest
from .manual_refinement import (
    MANUAL_REFINEMENT_WARNING,
    commit_manual_refinement,
    geometry_change_is_significant,
    preview_manual_candidate,
    select_calibration_candidate,
)
from .ring_geometry import (
    TheoreticalRingOverlay,
    manual_ring_distance,
    theoretical_ring_overlays,
)
from .scientific_kernel import (
    STANDARDS,
    available_standards,
    distance_from_ring_radius,
    energy_to_wavelength,
    q_to_ring_radius_m,
    q_to_ring_radius_px,
)
from .standard_selection import (
    detect_standard_keys,
    standard_display_name,
    standard_options,
    standard_q_values,
)

__all__ = [
    "STANDARDS",
    "CalibrationCandidate",
    "CalibrationResult",
    "CalibrationRequest",
    "CalibrationStandard",
    "DetectorImage",
    "MatchedRing",
    "MANUAL_REFINEMENT_WARNING",
    "TheoreticalRingOverlay",
    "available_standards",
    "commit_manual_refinement",
    "detect_standard_keys",
    "distance_from_ring_radius",
    "energy_to_wavelength",
    "geometry_change_is_significant",
    "manual_ring_distance",
    "preview_manual_candidate",
    "q_to_ring_radius_m",
    "q_to_ring_radius_px",
    "select_calibration_candidate",
    "standard_display_name",
    "standard_options",
    "standard_q_values",
    "theoretical_ring_overlays",
]
