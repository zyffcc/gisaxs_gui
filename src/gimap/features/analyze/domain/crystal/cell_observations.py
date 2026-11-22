"""Measured radial centres, assignment selection and transmitted-channel refraction."""

from __future__ import annotations

from dataclasses import dataclass
import math
from collections.abc import Mapping

import numpy as np

from .fibre import incident_kz, map_qz
from .assignment import Assignment, Candidate
from .reflections import ReflectionFamily


@dataclass(frozen=True)
class CellPeak:
    hkl: tuple[int, int, int]
    q: float
    sigma: float
    q_parallel: float | None = None
    qz: float | None = None
    equivalents: tuple[tuple[int, int, int], ...] = ()
    source_index: int | None = None
    chi_range_deg: tuple[float, float] | None = None


@dataclass(frozen=True)
class Refraction:
    k: float  # wavevector in Å⁻¹
    incidence_deg: float
    critical_deg: float
    map_critical_deg: float = 0.0

    def __post_init__(self):
        if not np.isfinite(self.k) or self.k <= 0 or any(
            not 0 <= angle < 90 for angle in (self.incidence_deg, self.critical_deg, self.map_critical_deg)
        ):
            raise ValueError("Refraction requires a positive wavevector and physical grazing angles.")


AMBIGUOUS = "No single reflection family dominates this peak."
CROSS_PHASE = "Reflections from more than one phase lie in the assignment window."
TENTATIVE = "The indexing assignment is tentative because the phase symmetry is unknown."
INVALID_PEAK = "The peak needs a finite positive centre and standard error."
NO_POSITION = "The peak has no measured direction for a refraction correction."
BELOW_CRITICAL = "The peak has no propagating transmitted exit wave at this direction."
NO_DIRECTION = "The peak has no measured in-plane and out-of-plane coordinates."
POLE_TIE = "The measured direction does not distinguish two different fibre poles."
SECTOR_POLE = "The sector does not select one fibre pole angle."


def _components(candidate, rings):
    if candidate.components:
        return candidate.components
    ring = next((r for r in rings if r.label == candidate.label and r.q == candidate.q), None)
    if ring is not None and ring.families:
        return ring.families
    # Compatibility with a single labelled candidate without its individual indices.
    label = candidate.label
    if len(candidate.families) > 1:
        return ()
    try:
        hkl = tuple(map(int, label)) if len(label) == 3 and label.isdigit() else tuple(map(int, label.split()))
    except ValueError:
        return ()
    return (ReflectionFamily(hkl, candidate.intensity, (hkl,)),) if len(hkl) == 3 else ()


def _family(choices, with_atoms, rings):
    components = [_components(candidate, rings) for candidate in choices]
    if not all(components):
        return None
    ranked = sorted((family for group in components for family in group), key=lambda f: -f.intensity)
    if len(ranked) == 1:
        return ranked[0]
    if with_atoms and all(f.intensity > 0 for f in ranked) and ranked[0].intensity >= 3 * sum(f.intensity for f in ranked[1:]):
        return ranked[0]
    return None


def _candidate(record):
    families = tuple(ReflectionFamily(tuple(f["hkl"]), float(f["intensity"]), tuple(map(tuple, f["members"])))
                     for f in record.get("families", ()))
    return Candidate(record["phase"], record["hkl"], record.get("q_hkl") or 0.,
                     float(record.get("intensity") or 0.), components=families)


def select_peaks(phase_name, assignments, *, with_atoms=False, rings=()):
    """Select at most one hkl per measurement; intensities from different phases are incomparable."""
    peaks, excluded = [], []
    for i, item in enumerate(assignments):
        q_parallel, qz = None, None
        tentative = False
        if isinstance(item, Mapping):
            if not item.get("phase") or not item.get("hkl"):
                continue
            q_parallel, qz = item.get("q_parallel"), item.get("qz")
            tentative = bool(item.get("tentative"))
            item = Assignment(item["q"], item.get("q_err"), item.get("fwhm"), item.get("tolerance"),
                              _candidate(item), alternatives=tuple(_candidate(a) for a in item.get("alternatives", ())))
        if item.match is None:
            continue
        choices = (item.match, *item.alternatives)
        phases = {candidate.phase for candidate in choices}
        if phase_name not in phases:
            continue
        reason = ""
        if tentative:
            reason = TENTATIVE
        elif len(phases) > 1:
            reason = CROSS_PHASE
        elif not _valid_centre(item.q, item.q_err):
            reason = INVALID_PEAK
        else:
            family = _family(choices, with_atoms, rings)
            if family is None:
                reason = AMBIGUOUS
            else:
                peaks.append(CellPeak(family.representative, float(item.q), float(item.q_err), q_parallel, qz,
                                      equivalents=family.members, source_index=i))
        if reason:
            excluded.append({"source_index": i, "q": finite_number(item.q), "reason": reason})
    return peaks, excluded


def _valid_centre(q, sigma):
    return all(v is not None and math.isfinite(v) and v > 0 for v in (q, sigma))


def finite_number(value):
    return float(value) if value is not None and math.isfinite(value) else None


def corrected(peak, optics):
    """True T-channel radial q, its σ at fixed map polar angle, and unit direction in the film."""
    if not _valid_centre(peak.q, peak.sigma):
        raise ValueError(INVALID_PEAK)
    change = optics is not None and optics.critical_deg != optics.map_critical_deg
    if peak.q_parallel is None or peak.qz is None or not np.isfinite([peak.q_parallel, peak.qz]).all():
        if change:
            raise ValueError(NO_POSITION)
        return peak.q, peak.sigma, None
    norm = math.hypot(peak.q_parallel, peak.qz)
    if norm == 0 or peak.qz < 0:
        raise ValueError(NO_DIRECTION)
    lateral, vertical = abs(peak.q_parallel) / norm, peak.qz / norm
    if not change:
        return peak.q, peak.sigma, (lateral, vertical)
    k, ai = optics.k, optics.incidence_deg
    # Inverse of the overlay's map_qz in the coordinate system which supplied the peak.
    ac_map = math.sin(math.radians(optics.map_critical_deg))
    exit_map = (peak.q * vertical - incident_kz(k, ai, optics.map_critical_deg)) / k
    sine = math.hypot(exit_map, ac_map)
    ac = math.sin(math.radians(optics.critical_deg))
    if exit_map < 0 or sine > 1 or sine <= ac:
        raise ValueError(BELOW_CRITICAL)
    true_z = float(map_qz(sine, k, ai, optics.critical_deg))
    true_parallel = peak.q * lateral
    q = math.hypot(true_parallel, true_z)
    # dq_true/dq_map along the measured radial profile, including the Jacobian of q_z.
    dz = exit_map / math.sqrt(sine**2 - ac**2)
    slope = (true_parallel * lateral + true_z * dz * vertical) / q
    sigma = abs(slope) * peak.sigma
    if not _valid_centre(q, sigma):
        raise ValueError(BELOW_CRITICAL)
    return q, sigma, (true_parallel / q, true_z / q)
