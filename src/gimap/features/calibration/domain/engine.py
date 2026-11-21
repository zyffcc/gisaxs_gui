"""Calibration scientific workflow engine."""

from __future__ import annotations

import hashlib
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .candidate_ranker import rank_candidates
from .center_estimator import estimate_center_candidates
from .geometry_model import energy_to_wavelength
from .center_polish import polished_centers
from .gisaxs_features import LitHalf, follows_horizontal_line, specular_rod_columns, without_columns
from .models import CalibrationCandidate, CalibrationResult, DetectorImage
from .optimizer import estimate_uncertainty, refine_candidate
from .peak_detector import detect_radial_peaks
from .peak_matcher import generate_distance_candidates, rematch_candidate
from .preprocessing import preprocess_detector_image
from .radial_profile import calculate_azimuthal_profile
from .standards import STANDARDS
from .tilt_fit import refine_tilt


class CalibrationError(RuntimeError):
    pass


class CalibrationCancelled(CalibrationError):
    pass


ProgressCallback = Callable[[int, str], None]
GRAZING_NOTE = "Grazing-incidence image: the specular rod is left out of the ring fit."
BAND_NOTE = "A horizontal band above the beam (the Yoneda band) is not taken for a ring."
OUT_OF_RANGE = ("No solution lies within the distance range. Widen the range, or check the energy and the "
                "pixel size.")
TILT_MODES = ("auto", "on", "off")
"""Refine rot1/rot2: ``auto`` with three or more matched rings, ``on`` with two or more, ``off`` never."""
MIN_RING_RADIUS_PX = 40.0
WINDOW_COVERAGE = 0.02
"""Least share of the azimuth with usable pixels for a radius to count as observable (AgBH order completeness)."""


def _rings(analysis, center_x: float, center_y: float, lit: Optional[LitHalf] = None):
    """(radial peaks that may be powder rings, observable radius window, number of horizontal bands left out). With
    ``lit`` (a grazing-incidence image) only the half of the image the sample does not shadow is profiled."""
    valid = analysis.valid if lit is None else lit.valid(center_y)
    profile = calculate_azimuthal_profile(
        analysis.signal, valid, center_x, center_y,
        angle_bins=360, percentile=60.0,
    )
    peaks = detect_radial_peaks(profile, min_radius_px=10.0)
    # Central beamstop edges and support hardware are not powder rings.
    peaks = [peak for peak in peaks if peak.radius_px >= MIN_RING_RADIUS_PX]
    # The Yoneda band of a grazing-incidence image is a strong radial peak above the beam, not a ring.
    rings = [peak for peak in peaks if not follows_horizontal_line(analysis, center_x, center_y, peak.radius_px)]
    observed = profile.radius_px[(profile.counts >= WINDOW_COVERAGE * 360) & (profile.radius_px >= MIN_RING_RADIUS_PX)]
    window = (float(observed.min()), float(observed.max())) if observed.size else None
    return rings, window, len(peaks) - len(rings)


def _coarse_peaks(analysis, center_x: float, center_y: float):
    """Detect powder rings using robust azimuthal consensus."""
    return _rings(analysis, center_x, center_y)[0]


def _in_range(candidate: CalibrationCandidate, distance_range_mm: tuple[float, float]) -> bool:
    return distance_range_mm[0] <= candidate.distance_mm <= distance_range_mm[1]


def _refinement_seeds(
    candidates: list[CalibrationCandidate],
    *,
    group_limit: int = 6,
    per_group: int = 3,
    estimated_distance_mm: float | None = None,
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
    if estimated_distance_mm and not any(
        abs(item.distance_mm - estimated_distance_mm) <= 0.25 * estimated_distance_mm for item in selected
    ):
        # A stated distance is the user's knowledge: its best match is always refined, not crowded out.
        near = [item for item in candidates if abs(item.distance_mm - estimated_distance_mm) <= 0.25 * estimated_distance_mm]
        if near:
            selected.append(max(near, key=lambda item: item.score + 40.0 * item.center_support))
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
        tilt: str = "auto",
        rot3_deg: float = 0.0,
        energy_source: str = "",
    ) -> CalibrationResult:
        """``tilt``: refine pyFAI's rot1 and rot2 with the centre and the distance (``TILT_MODES``); ``rot3_deg`` is
        kept as given (a turn about the beam does not change the rings). ``energy_source`` (where the energy came
        from) is written into the result's metadata."""
        if tilt not in TILT_MODES:
            raise CalibrationError(f"Unknown tilt mode {tilt!r}.")
        self._stage(5, "Preparing mask")
        px = float(pixel_size_x_m or image.pixel_size_x_m or 0.0)
        py = float(pixel_size_y_m or image.pixel_size_y_m or px)
        if px <= 0 or py <= 0:
            raise CalibrationError(
                "Pixel size could not be determined. Select the detector model or enter the pixel size "
                "under Advanced configuration."
            )
        wavelength = energy_to_wavelength(energy_kev)
        analysis = preprocess_detector_image(image, subtract_background=subtract_background)
        # Grazing incidence: the specular rod crosses every ring and pulls a ring fit; it is left out of both.
        rod = specular_rod_columns(analysis)
        ring_image = without_columns(analysis, rod)
        lit = LitHalf(ring_image) if rod is not None else None
        bands = 0

        self._stage(20, "Estimating center")
        metadata_center = None
        if image.beam_center_x_px is not None and image.beam_center_y_px is not None:
            metadata_center = (image.beam_center_x_px, image.beam_center_y_px)
        centers = estimate_center_candidates(analysis, metadata_center, source_data=image.data)
        if not centers:
            raise CalibrationError("A stable beam-center proposal could not be found.")
        if rod is not None:
            # Half arcs from coarse voting cells: a centre a few pixels off loses thin rings from the profile.
            centers = centers + polished_centers(ring_image, centers)

        standard_keys = list(STANDARDS) if standard_key == "auto" else [standard_key]
        raw_candidates: list[CalibrationCandidate] = []
        pixel_mean = 0.5 * (px + py)
        total_profiles = max(1, len(centers))
        for index, center in enumerate(centers):
            self._stage(35 + int(15 * index / total_profiles), "Detecting rings")
            peaks, window, dropped = _rings(ring_image, center.x_px, center.y_px, lit)
            bands = max(bands, dropped)
            for key in standard_keys:
                generated = generate_distance_candidates(
                    peaks, STANDARDS[key], energy_kev, pixel_mean,
                    center.x_px, center.y_px, analysis.valid_fraction, distance_range_mm, window,
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
        preliminary = rank_candidates(raw_candidates, estimated_distance_mm, limit=120, grazing=rod is not None)
        refinement_seeds = _refinement_seeds(preliminary, estimated_distance_mm=estimated_distance_mm)
        refined: list[CalibrationCandidate] = []
        for index, candidate in enumerate(refinement_seeds):
            self._stage(62 + int(23 * index / max(1, len(refinement_seeds))), "Optimizing geometry")
            candidate = refine_candidate(analysis, candidate, wavelength, px, py, distance_range_mm)
            peaks, window, _ = _rings(ring_image, candidate.center_x_px, candidate.center_y_px, lit)
            candidate = rematch_candidate(candidate, peaks, STANDARDS[candidate.standard_key], energy_kev, pixel_mean,
                                          window)
            if _in_range(candidate, distance_range_mm):  # the range is a limit for every step, not only the first
                refined.append(candidate)
        if not refined:
            raise CalibrationError(OUT_OF_RANGE)

        self._stage(90, "Ranking candidates")
        ranked_candidates = rank_candidates(refined, estimated_distance_mm, limit=20, grazing=rod is not None)
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
        if tilt != "off":
            self._stage(92, "Refining detector tilt")
            for candidate in candidates:
                refine_tilt(ring_image, candidate, wavelength, px, py, rot3_rad=math.radians(rot3_deg),
                            distance_range_mm=distance_range_mm, force=tilt == "on")
        for candidate in candidates:  # 1σ of the centre and distance (and tilt) of the candidates shown
            estimate_uncertainty(ring_image if candidate.tilt_refined else analysis, candidate, wavelength, px, py)
        best = candidates[0]
        for candidate in candidates:
            if rod is not None and GRAZING_NOTE not in candidate.warnings:
                candidate.warnings.append(GRAZING_NOTE)
            if bands and BAND_NOTE not in candidate.warnings:
                candidate.warnings.append(BAND_NOTE)
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
                # Needed to express the beam centre in other pages' conventions.
                "image_shape": [int(value) for value in np.shape(image.data)[:2]],
                "estimated_distance_mm": estimated_distance_mm,
                "distance_range_mm": list(distance_range_mm),
                "tilt_mode": tilt,
                **({"energy_source": energy_source} if energy_source else {}),
            },
        )
        self._stage(100, "Calibration complete")
        return result
