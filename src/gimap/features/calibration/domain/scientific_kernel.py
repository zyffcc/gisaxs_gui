"""目标架构下对现有纯 Calibration scientific primitives 的稳定入口。"""

from calibration.candidate_ranker import rank_candidates
from calibration.center_estimator import estimate_center_candidates
from calibration.geometry_model import (
    distance_from_ring_radius,
    energy_to_wavelength,
    q_to_ring_radius_m,
    q_to_ring_radius_px,
)
from calibration.optimizer import refine_candidate
from calibration.peak_detector import DetectedPeak, detect_radial_peaks
from calibration.peak_matcher import generate_distance_candidates, rematch_candidate
from calibration.preprocessing import AnalysisImage, preprocess_detector_image
from calibration.radial_profile import (
    RadialProfile,
    calculate_azimuthal_profile,
    calculate_radial_profile,
)
from calibration.standards import STANDARDS, available_standards

__all__ = [
    "AnalysisImage",
    "DetectedPeak",
    "RadialProfile",
    "STANDARDS",
    "available_standards",
    "calculate_azimuthal_profile",
    "calculate_radial_profile",
    "detect_radial_peaks",
    "distance_from_ring_radius",
    "energy_to_wavelength",
    "estimate_center_candidates",
    "generate_distance_candidates",
    "preprocess_detector_image",
    "q_to_ring_radius_m",
    "q_to_ring_radius_px",
    "rank_candidates",
    "refine_candidate",
    "rematch_candidate",
]
