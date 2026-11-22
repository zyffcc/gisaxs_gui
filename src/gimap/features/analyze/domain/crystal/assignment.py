"""Assign the rings found in I(q) to the reflections of the phases given.

A ring at q_0 ± σ(q_0) with full width FWHM is assigned to the nearest reflection (powder ring position |q|) of
any phase when

    |q_0 − q_hkl| ≤ δ,    δ = max( 3 σ(q_0), FWHM / 2 )

The window takes the larger of the statistical error and the half width: σ(q_0) of a strong ring is often
10⁻⁴ Å⁻¹, smaller than the cell constants or the geometry are known, while a reflection within the half width
still lies under the peak. The relative spacing difference is (d_0 − d_hkl)/d_hkl = −Δq/q_0.
Other reflections within δ (of the same or another phase) are listed as alternatives, nearest first: the
assignment is the nearest, not a proof.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

from .reflections import ReflectionFamily

ERROR_FACTOR = 3.0
"""Statistical errors of q_0 within which a reflection is accepted."""
WIDTH_FRACTION = 0.5
"""Fraction of the FWHM within which a reflection is accepted."""


@dataclass(frozen=True)
class Candidate:
    phase: str
    label: str
    q: float
    intensity: float = 1.0
    multiplicity: int = 1
    families: tuple[str, ...] = ()
    components: tuple[ReflectionFamily, ...] = ()

    def family_records(self):
        return [{"hkl": list(part.representative), "intensity": part.intensity,
                 "members": [list(hkl) for hkl in part.members]} for part in self.components]


@dataclass(frozen=True)
class Assignment:
    q: float
    q_err: float
    fwhm: float
    tolerance: float
    match: Optional[Candidate] = None
    delta_q: Optional[float] = None
    alternatives: tuple[Candidate, ...] = field(default_factory=tuple)

    @property
    def assigned(self) -> bool:
        return self.match is not None

    @property
    def delta_sigma(self) -> Optional[float]:
        if self.delta_q is None or not (self.q_err > 0 and math.isfinite(self.q_err)):
            return None
        return self.delta_q / self.q_err

    @property
    def relative_d(self) -> Optional[float]:
        """(d_obs − d_hkl)/d_hkl, with d = 2π/q."""
        return None if self.delta_q is None or self.q <= 0 else -self.delta_q / self.q

    def record(self) -> dict:
        nearest = self.match
        return {
            "q": self.q, "q_err": self.q_err, "fwhm": self.fwhm, "tolerance": self.tolerance,
            "assigned": self.assigned,
            "ambiguous": bool(self.alternatives or (nearest and len(nearest.families) > 1)),
            "phase": None if nearest is None else nearest.phase,
            "hkl": None if nearest is None else nearest.label,
            "families": nearest.family_records() if nearest is not None else [],
            "intensity": nearest.intensity if nearest is not None else None,
            "q_hkl": None if nearest is None else nearest.q,
            "d_hkl_A": None if nearest is None else 2.0 * math.pi / nearest.q,
            "delta_q": self.delta_q, "delta_sigma": self.delta_sigma, "relative_d": self.relative_d,
            "alternatives": [
                {"phase": item.phase, "hkl": item.label, "q_hkl": item.q, "delta_q": self.q - item.q,
                 "relative_d": (item.q - self.q) / self.q if self.q > 0 else None,
                 "families": item.family_records(), "intensity": item.intensity}
                for item in self.alternatives
            ],
        }


def tolerance(q_err: float, fwhm: float) -> float:
    error = q_err if q_err is not None and math.isfinite(q_err) and q_err > 0 else 0.0
    width = fwhm if fwhm is not None and math.isfinite(fwhm) and fwhm > 0 else 0.0
    return max(ERROR_FACTOR * error, WIDTH_FRACTION * width)


def assign(q: float, q_err: float, fwhm: float, candidates: Sequence[Candidate]) -> Assignment:
    """The assignment of one ring (see the module docstring)."""
    window = tolerance(q_err, fwhm)
    inside = sorted((item for item in candidates if abs(q - item.q) <= window), key=lambda item: abs(q - item.q))
    if not inside:
        return Assignment(q, q_err, fwhm, window)
    best = inside[0]
    return Assignment(q, q_err, fwhm, window, best, q - best.q, tuple(inside[1:]))


def assign_all(peaks: Sequence[tuple[float, float, float]], candidates: Sequence[Candidate]) -> list[Assignment]:
    """``assign`` for each ``(q_0, σ(q_0), FWHM)``."""
    return [assign(float(q), float(error), float(width), candidates) for q, error, width in peaks]


__all__ = ["Assignment", "Candidate", "ERROR_FACTOR", "WIDTH_FRACTION", "assign", "assign_all", "tolerance"]
