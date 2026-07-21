from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .candidate_ranker import rank_candidates
from .center_estimator import estimate_center_candidates
from .geometry_model import energy_to_wavelength
from .models import CalibrationCandidate, CalibrationResult, DetectorImage
from .optimizer import refine_candidate
from .peak_detector import detect_radial_peaks
from .peak_matcher import generate_distance_candidates, rematch_candidate
from .preprocessing import preprocess_detector_image
from .radial_profile import calculate_azimuthal_profile
from .standards import STANDARDS


class CalibrationError(RuntimeError):
    pass


class CalibrationCancelled(CalibrationError):
    pass


ProgressCallback = Callable[[int, str], None]


def _coarse_peaks(analysis, center_x: float, center_y: float):
    """Detect powder rings using robust azimuthal consensus."""
    profile = calculate_azimuthal_profile(
        analysis.signal, analysis.valid, center_x, center_y,
        angle_bins=360, percentile=60.0,
    )
    peaks = detect_radial_peaks(profile, min_radius_px=10.0)
    # Central beamstop edges and support hardware are not powder rings.
    return [peak for peak in peaks if peak.radius_px >= 40.0]


def _refinement_seeds(
    candidates: list[CalibrationCandidate],
    *,
    group_limit: int = 6,
    per_group: int = 3,
) -> list[CalibrationCandidate]:
    """Keep distinct center and distance hypotheses for nonlinear refinement.

    Partial WAXS arcs can create several short-distance coincidences at one
    center. Taking only the globally highest raw scores lets those aliases
    crowd out the physically correct distance before geometry is optimized.
    """
    groups: dict[tuple, list[CalibrationCandidate]] = {}
    for candidate in candidates:
        key = (
            candidate.standard_key,
            round(candidate.center_x_px, 1),
            round(candidate.center_y_px, 1),
            candidate.center_method,
        )
        groups.setdefault(key, []).append(candidate)
    globally_ordered = sorted(
        groups.values(),
        key=lambda items: max(item.score + 40.0 * item.center_support for item in items),
        reverse=True,
    )
    # Auto-standard mode must not let one standard consume every refinement
    # slot before competing standards have been geometrically tested.
    ordered_groups: list[list[CalibrationCandidate]] = []
    for standard_key in dict.fromkeys(
        item.standard_key for item in candidates
    ):
        best = next(
            (items for items in globally_ordered if items[0].standard_key == standard_key),
            None,
        )
        if best is not None:
            ordered_groups.append(best)
    for items in globally_ordered:
        if items not in ordered_groups:
            ordered_groups.append(items)
        if len(ordered_groups) >= group_limit:
            break
    ordered_groups = ordered_groups[:group_limit]
    selected: list[CalibrationCandidate] = []
    for items in ordered_groups:
        distances: list[float] = []
        for candidate in sorted(items, key=lambda item: item.score, reverse=True):
            if any(abs(candidate.distance_mm - old) / max(candidate.distance_mm, old, 1.0) < 0.25 for old in distances):
                continue
            selected.append(candidate)
            distances.append(candidate.distance_mm)
            if len(distances) >= per_group:
                break
    selected.sort(key=lambda item: item.score + 40.0 * item.center_support, reverse=True)
    return selected


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class CalibrationEngine:
    def __init__(self, progress: Optional[ProgressCallback] = None, cancelled: Optional[Callable[[], bool]] = None):
        self.progress = progress or (lambda _value, _stage: None)
        self.cancelled = cancelled or (lambda: False)

    def _stage(self, value: int, text: str) -> None:
        if self.cancelled():
            raise CalibrationCancelled("Calibration cancelled by user.")
        self.progress(value, text)

    def calibrate(
        self,
        image: DetectorImage,
        *,
        energy_kev: float,
        standard_key: str = "auto",
        estimated_distance_mm: float | None = None,
        distance_range_mm: tuple[float, float] = (30.0, 10_000.0),
        pixel_size_x_m: float | None = None,
        pixel_size_y_m: float | None = None,
        subtract_background: bool = True,
    ) -> CalibrationResult:
        self._stage(5, "Preparing mask")
        px = float(pixel_size_x_m or image.pixel_size_x_m or 0.0)
        py = float(pixel_size_y_m or image.pixel_size_y_m or px)
        if px <= 0 or py <= 0:
            raise CalibrationError(
                "Pixel size could not be determined. Select the detector model or enter the pixel size in Advanced Settings."
            )
        wavelength = energy_to_wavelength(energy_kev)
        analysis = preprocess_detector_image(image, subtract_background=subtract_background)

        self._stage(20, "Estimating center")
        metadata_center = None
        if image.beam_center_x_px is not None and image.beam_center_y_px is not None:
            metadata_center = (image.beam_center_x_px, image.beam_center_y_px)
        centers = estimate_center_candidates(analysis, metadata_center, source_data=image.data)
        if not centers:
            raise CalibrationError("A stable beam-center proposal could not be found.")

        standard_keys = list(STANDARDS) if standard_key == "auto" else [standard_key]
        raw_candidates: list[CalibrationCandidate] = []
        pixel_mean = 0.5 * (px + py)
        total_profiles = max(1, len(centers))
        for index, center in enumerate(centers):
            self._stage(35 + int(15 * index / total_profiles), "Detecting rings")
            peaks = _coarse_peaks(analysis, center.x_px, center.y_px)
            for key in standard_keys:
                generated = generate_distance_candidates(
                    peaks, STANDARDS[key], energy_kev, pixel_mean,
                    center.x_px, center.y_px, analysis.valid_fraction, distance_range_mm,
                )
                for candidate in generated:
                    candidate.center_method = center.method
                    candidate.center_support = center.support
                raw_candidates.extend(generated)
        if not raw_candidates:
            raise CalibrationError(
                "No calibration-standard match was found. Check the energy, pixel size, distance range, and visible rings."
            )

        self._stage(55, "Matching standards")
        preliminary = rank_candidates(raw_candidates, estimated_distance_mm, limit=120)
        refinement_seeds = _refinement_seeds(preliminary)
        refined: list[CalibrationCandidate] = []
        for index, candidate in enumerate(refinement_seeds):
            self._stage(62 + int(23 * index / max(1, len(refinement_seeds))), "Optimizing geometry")
            candidate = refine_candidate(analysis, candidate, wavelength, px, py)
            peaks = _coarse_peaks(analysis, candidate.center_x_px, candidate.center_y_px)
            candidate = rematch_candidate(candidate, peaks, STANDARDS[candidate.standard_key], energy_kev, pixel_mean)
            refined.append(candidate)

        self._stage(90, "Ranking candidates")
        ranked_candidates = rank_candidates(refined, estimated_distance_mm, limit=20)
        candidates = ranked_candidates[:3]
        for key in standard_keys:
            if any(item.standard_key == key for item in candidates):
                continue
            alternative = next(
                (item for item in ranked_candidates if item.standard_key == key),
                None,
            )
            if alternative is not None:
                candidates.append(alternative)
        for item in ranked_candidates:
            if item not in candidates:
                candidates.append(item)
            if len(candidates) >= 5:
                break
        candidates.sort(key=lambda item: item.score, reverse=True)
        candidates = candidates[:5]
        if not candidates:
            raise CalibrationError("Calibration candidates were rejected as physically implausible.")
        best = candidates[0]
        if best.matched_ring_count < 2:
            best.warnings.append(
                "Only one calibration ring was detected. The distance is ambiguous, so candidate solutions are shown."
            )
        path = Path(image.source_path)
        stat = path.stat()
        self._stage(96, "Recording calibration")
        result = CalibrationResult(
            source_image=str(path),
            source_image_size=int(stat.st_size),
            source_image_mtime_ns=int(stat.st_mtime_ns),
            image_sha256=_file_sha256(path),
            energy_kev=float(energy_kev),
            wavelength_angstrom=float(wavelength),
            detector_name=image.detector_name,
            pixel_size_x_m=px,
            pixel_size_y_m=py,
            selected_candidate=best,
            candidates=candidates,
            calibration_timestamp=datetime.now(timezone.utc).isoformat(),
            metadata={
                "loader": image.metadata,
                "estimated_distance_mm": estimated_distance_mm,
                "distance_range_mm": list(distance_range_mm),
            },
        )
        self._stage(100, "Calibration complete")
        return result
