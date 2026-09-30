"""Reduce the frames of an in-situ series with reference-peak corrections."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Optional

import numpy as np

from ..domain import (
    GIWAXS,
    SeriesCorrection,
    aligned_distance,
    locate_reference_peak,
    normalization_factor,
    q_from_two_theta,
)
from .models import AnalysisRequest, FrameAnalysis

ALIGN_ITERATIONS = 3
ALIGN_TOLERANCE = 1e-6


def _radial_q(analysis: FrameAnalysis) -> tuple[np.ndarray, np.ndarray]:
    radial = analysis.reduction.curve("radial") if analysis.reduction is not None else None
    if radial is None or radial.is_empty:
        raise ValueError("The frame has no I(q) to find the reference peak in.")
    x = radial.x
    if analysis.reduction is not None and radial.x_label.startswith("2θ"):
        x = q_from_two_theta(x, analysis.geometry.wavelength_angstrom)
    return x, radial.intensity


def scaled(analysis: FrameAnalysis, factor: float) -> FrameAnalysis:
    """Every intensity of the reduction multiplied by ``factor`` (q axes unchanged)."""
    reduction = analysis.reduction
    curves = tuple(
        replace(curve, intensity=curve.intensity * factor, sigma=curve.sigma * factor)
        for curve in reduction.curves
    )
    rsm = reduction.reciprocal_space_map
    if rsm is not None:
        rsm = replace(rsm, image=rsm.image * np.float32(factor))
    return replace(analysis, reduction=replace(reduction, curves=curves, reciprocal_space_map=rsm))


class CorrectSeriesFrame:
    """One frame of a series: align the distance, then normalise, as requested."""

    def __init__(self, analyze_frame):
        self.analyze = analyze_frame

    def __call__(
        self,
        request: AnalysisRequest,
        series: SeriesCorrection,
        *,
        factor: Optional[float] = None,
    ) -> tuple[FrameAnalysis, dict[str, Any]]:
        analysis = self.analyze(request)
        if series.is_identity:
            return analysis, {}
        if analysis.reduction is None or analysis.kind != GIWAXS:
            raise ValueError("Reference-peak corrections need a GIWAXS frame with a geometry.")
        info: dict[str, Any] = {"reference_q": series.reference_q}
        if series.align_distance:
            distance = analysis.geometry.distance_m
            for _ in range(ALIGN_ITERATIONS):
                peak_q, _ = locate_reference_peak(*_radial_q(analysis), series.reference_q, series.half_width)
                new_distance = aligned_distance(distance, peak_q, series.reference_q)
                change = abs(new_distance - distance) / distance
                distance = new_distance
                analysis = self.analyze(replace(request, distance_m=distance), loaded=analysis)
                if change < ALIGN_TOLERANCE:
                    break
            info["aligned_distance_m"] = distance
        if series.normalize:
            if factor is None or series.per_frame:
                _, peak = locate_reference_peak(*_radial_q(analysis), series.reference_q, series.half_width)
                factor = normalization_factor(peak, series.target_intensity)
            analysis = scaled(analysis, factor)
            info["normalization_factor"] = factor
        return analysis, info


__all__ = ["CorrectSeriesFrame", "scaled"]
