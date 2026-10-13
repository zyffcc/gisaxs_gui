"""Crystallite size (coherence length) from a peak width with the Scherrer equation.

In q: L = 2πK / Δq, with Δq the FWHM in Å⁻¹ after removing the instrumental
width (Gaussian convolution: Δq² = FWHM² − FWHM_instr²).  Without an
instrumental width the result is a lower bound: in GIWAXS the beam footprint,
sample–detector distance and pixel size broaden every peak, and strain or
paracrystalline disorder broaden them too.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from .peaks import BROAD_RELATIVE_WIDTH

DEFAULT_SHAPE_FACTOR = 0.9


@dataclass(frozen=True)
class CrystalliteSize:
    q: float
    fwhm: Optional[float]
    corrected_fwhm: Optional[float]
    size: Optional[float]
    """Å."""
    size_err: Optional[float]
    lower_bound: bool
    shape_factor: float
    reason: str
    notes: tuple[str, ...]


def scherrer_size(
    q: float,
    fwhm: Optional[float],
    fwhm_err: Optional[float] = None,
    *,
    shape_factor: float = DEFAULT_SHAPE_FACTOR,
    instrumental_fwhm: float = 0.0,
    bin_width: Optional[float] = None,
) -> CrystalliteSize:
    notes = [f"Scherrer with K = {shape_factor:g}; strain and disorder also broaden peaks."]
    if fwhm is None or not math.isfinite(fwhm) or fwhm <= 0:
        return CrystalliteSize(q, fwhm, None, None, None, False, shape_factor, "No peak width is available for this peak.", tuple(notes))
    if fwhm > BROAD_RELATIVE_WIDTH * q:
        return CrystalliteSize(
            q, fwhm, None, None, None, False, shape_factor,
            f"The peak is a broad halo (FWHM {fwhm:.3g} Å⁻¹ = {fwhm / q:.0%} of its q): amorphous or "
            f"liquid-like order, not crystallites. Its correlation length 2π/FWHM ≈ "
            f"{2.0 * math.pi / fwhm / 10.0:.2g} nm is not a crystal size.",
            tuple(notes),
        )
    instrumental = max(0.0, float(instrumental_fwhm or 0.0))
    if instrumental >= fwhm:
        return CrystalliteSize(
            q, fwhm, None, None, None, False, shape_factor,
            f"The peak (FWHM {fwhm:.4g} Å⁻¹) is not broader than the instrumental width "
            f"{instrumental:.4g} Å⁻¹, so its size cannot be resolved.",
            tuple(notes),
        )
    corrected = math.sqrt(fwhm ** 2 - instrumental ** 2)
    size = 2.0 * math.pi * shape_factor / corrected
    size_err = None
    if fwhm_err is not None and math.isfinite(fwhm_err) and fwhm_err > 0:
        size_err = size * (fwhm / corrected) * fwhm_err / corrected
    lower_bound = instrumental == 0.0
    if lower_bound:
        notes.append(
            "No instrumental or geometric broadening was subtracted: the size is a lower bound "
            "of the coherence length."
        )
    if bin_width is not None and bin_width > 0 and fwhm < 3.0 * bin_width:
        lower_bound = True
        notes.append(
            f"The peak spans only {fwhm / bin_width:.1f} radial bins: its width is limited by the "
            "binning, so the true size can be larger."
        )
    return CrystalliteSize(q, fwhm, corrected, size, size_err, lower_bound, shape_factor, "", tuple(notes))


__all__ = ["CrystalliteSize", "DEFAULT_SHAPE_FACTOR", "scherrer_size"]
