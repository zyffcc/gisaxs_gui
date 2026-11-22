"""Crystal orientation from GIWAXS profiles.

χ follows Analyze: 0° along the surface normal (out of plane, qz) and ±90° in
the sample plane (q∥).  Two views:

* ``compare_sectors`` — at each peak, the net intensity per pixel in the
  in-plane and out-of-plane sectors, their ratio and which dominates;
* ``ring_orientation`` — the azimuthal profile I(χ) of one ring, folded to
  |χ|, with the measured χ coverage, its maxima and the Hermans orientation
  factor f = (3⟨cos²χ⟩ − 1)/2 relative to the surface normal (sin χ weighted,
  i.e. assuming a fibre texture about the normal).  f = 1: along the normal,
  0: random, −0.5: in plane.  The texture sentence compares f with f_iso, the f
  of a random ring over the same measured |χ| (``TEXTURE_SIGMAS``), as a separate coverage diagnostic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np

from ....shared.numbers import format_number
from ....shared.ring_profile import SHADOW_FRACTION, fold_ring, measured_hermans
from .profiles import Profile, estimate_background

DOMINANCE_RATIO = 2.0
MIN_COVERAGE = 0.8
"""Fraction of the sin χ weight of |χ| 0–90° that must be measured for the Hermans f.

f weights each |χ| by sin χ, so the part near the sample plane counts most: an isotropic
ring measured only over |χ| < 60° already gives f ≈ 0.4, while a missing wedge below 10°
hardly matters."""
SIMILAR_MAXIMUM = 0.85
"""A second maximum at least this high (relative to the first) and 30° away: no single orientation."""
WEAK_HERMAN = 0.1
TEXTURE_SIGMAS = 3.0
"""The texture rule: f is compared with f_iso, the f of a random ring over the same measured |χ| (not with 0, which
only a fully measured ring gives). |f − f_iso| below max(WEAK_HERMAN, TEXTURE_SIGMAS·σ_f) does not support "along
the normal" or "in plane", whatever the strongest |χ|."""
NORMAL_CHI, PLANE_CHI = 20.0, 70.0
"""The I(χ) maximum names "along the surface normal" below NORMAL_CHI and "in the sample plane" above PLANE_CHI (each
only with the margin of the texture rule and an f − f_iso of the same sign); between them the ring is called tilted
at its maximum, without that margin."""


def _texture_margin(herman_err: Optional[float]) -> float:
    """The margin |f − f_iso| must reach before a sentence names an orientation."""
    return max(WEAK_HERMAN, TEXTURE_SIGMAS * float(herman_err or 0.0))


def herman_decimals(sigma: Optional[float]) -> int:
    """The decimals f, f_iso and f − f_iso are printed with: those that show σ_f to two significant digits
    (σ_f = 0.0027: 4); 2 when σ_f is not known."""
    if sigma is None or not math.isfinite(float(sigma)) or float(sigma) <= 0:
        return 2
    return int(min(6, max(2, 1 - math.floor(math.log10(float(sigma))))))


def signed(value: float, decimals: int, *, plus: bool = False) -> str:
    """``value`` with ``decimals`` decimals and the Unicode minus (−0.0055, not -0.0055); ``plus`` writes + before a
    positive value. A value that rounds to zero has no minus (0.000, +0.000)."""
    value = float(value)
    if round(value, decimals) == 0:
        value = 0.0
    return (f"{value:+.{decimals}f}" if plus else f"{value:.{decimals}f}").replace("-", "−")


@dataclass(frozen=True)
class SectorComparison:
    q: float
    fwhm: float
    in_plane: Optional[float]
    """Mean net counts per pixel around the peak (``None``: no pixels there)."""
    in_plane_err: Optional[float]
    out_of_plane: Optional[float]
    out_of_plane_err: Optional[float]
    ratio: Optional[float]
    """out-of-plane / in-plane when both are significant."""
    ratio_err: Optional[float]
    preference: str
    note: str


def _net_near(profile: Profile, window_points: int, q: float, half_width: float):
    """(net intensity, its error, the sector's own background) around ``q``."""
    data = profile.measured()
    if data.size < 8:
        return None
    baseline = estimate_background(data.y, data.sigma, window_points)
    net = data.y - baseline
    inside = np.abs(data.x - q) <= half_width
    if not inside.any():
        return None
    count = int(inside.sum())
    return float(net[inside].mean()), float(np.sqrt((data.sigma[inside] ** 2).sum()) / count), float(baseline[inside].mean())


def _shadowed(sector, reference: Optional[float]) -> bool:
    """A sector whose diffuse background is far below the whole ring's at this q sees a shadow, not the sample."""
    return sector is not None and reference is not None and reference > 0 and sector[2] < SHADOW_FRACTION * reference


def compare_sectors(
    peaks: Sequence[tuple[float, float]],
    in_plane: Profile,
    out_of_plane: Profile,
    *,
    background_window: float,
    window_fwhm: float = 1.0,
    min_snr: float = 3.0,
    reference_background: Optional[Callable[[float], Optional[float]]] = None,
) -> tuple[SectorComparison, ...]:
    """``peaks`` are ``(q, fwhm)`` pairs, usually from the full I(q).

    ``reference_background(q)`` is the diffuse background of the whole ring (the full I(q)):
    a sector far below it is shadowed and is not compared.
    """
    step = max(in_plane.measured().step(), out_of_plane.measured().step(), 1e-9)
    window_points = int(max(3, round(background_window / step)))
    results = []
    for q, fwhm in peaks:
        half = max(0.5 * window_fwhm * fwhm, 1.5 * step)
        ip = _net_near(in_plane, window_points, q, half)
        oop = _net_near(out_of_plane, window_points, q, half)
        seen_ip = ip is not None and ip[1] > 0 and ip[0] / ip[1] >= min_snr
        seen_oop = oop is not None and oop[1] > 0 and oop[0] / oop[1] >= min_snr
        ratio = ratio_err = None
        notes = []
        if ip is None:
            notes.append("no in-plane pixels at this q")
        if oop is None:
            notes.append("no out-of-plane pixels at this q (outside the detector or in the missing wedge)")
        level = reference_background(float(q)) if reference_background is not None else None
        dark = [name for name, sector in (("in-plane", ip), ("out-of-plane", oop)) if _shadowed(sector, level)]
        for name, sector in (("in-plane", ip), ("out-of-plane", oop)):
            if name in dark:
                notes.append(
                    f"the {name} sector is shadowed here: its background is {sector[2] / level:.0%} of the whole "
                    "ring's (a shadow, absorber or insensitive detector area), so it is not compared"
                )
        if dark:
            usable = [
                (name, seen) for name, sector, seen in (("in-plane", ip, seen_ip), ("out-of-plane", oop, seen_oop))
                if sector is not None and name not in dark
            ]
            if usable:
                name, seen = usable[0]
                preference = (
                    f"only the {name} sector is usable here: the other is shadowed" + (" (peak seen)" if seen else " (no peak)")
                )
            else:
                preference = "not comparable: " + " and ".join(
                    f"the {name} sector is {'shadowed' if name in dark else 'not measured'}"
                    for name in ("in-plane", "out-of-plane")
                )
        elif seen_ip and seen_oop:
            ratio = oop[0] / ip[0]
            ratio_err = abs(ratio) * float(np.hypot(oop[1] / oop[0], ip[1] / ip[0]))
            if ratio >= DOMINANCE_RATIO:
                preference = "mainly out-of-plane"
            elif ratio <= 1.0 / DOMINANCE_RATIO:
                preference = "mainly in-plane"
            else:
                preference = "both sectors (no strong preference)"
        elif ip is None and oop is None:
            preference = "not measured: neither sector reaches this q on the detector"
        elif oop is None:
            preference = "only the in-plane sector is measured here" + (" (peak seen)" if seen_ip else " (no peak)")
        elif ip is None:
            preference = "only the out-of-plane sector is measured here" + (" (peak seen)" if seen_oop else " (no peak)")
        elif seen_oop:
            preference = "out-of-plane only"
        elif seen_ip:
            preference = "in-plane only"
        else:
            preference = "not detected in either sector"
        results.append(SectorComparison(
            q=float(q), fwhm=float(fwhm),
            in_plane=None if ip is None else ip[0], in_plane_err=None if ip is None else ip[1],
            out_of_plane=None if oop is None else oop[0], out_of_plane_err=None if oop is None else oop[1],
            ratio=ratio, ratio_err=ratio_err, preference=preference, note="; ".join(notes),
        ))
    return tuple(results)


@dataclass(frozen=True)
class RingOrientation:
    q_window: tuple[float, float]
    background: float
    coverage: float
    """Fraction of |χ| = 0…90° with measured pixels."""
    missing: tuple[tuple[float, float], ...]
    """Unmeasured |χ| ranges of at least 2°."""
    herman: Optional[float]
    herman_err: Optional[float]
    cos2: Optional[float]
    maxima: tuple[dict, ...]
    """``{"chi": |χ| of the maximum, "fwhm": degrees, "height": net counts/pixel}``."""
    anisotropy: Optional[float]
    texture: str
    reason: str
    notes: tuple[str, ...]
    weighted_coverage: Optional[float] = None
    """Measured fraction of the sin χ weight the Hermans f gives to |χ| 0–90°."""
    herman_isotropic: Optional[float] = None
    """f a random (isotropic) ring would give over the same measured |χ|: the reference to compare f with."""
    shadowed: tuple[tuple[float, float], ...] = ()
    """|χ| ranges with pixels far below the diffuse background (a shadow); they count as unmeasured."""
    herman_excess: Optional[float] = None
    """f − f_iso: what the texture sentence is based on."""
    texture_margin: Optional[float] = None
    """max(0.1, 3σ_f): |f − f_iso| below it names no orientation."""
    horizon_excess: Optional[float] = None
    """Legacy report field. Exit-angle sensitivity is recorded in the systematic range."""

    orientation_settings: dict = field(default_factory=dict)
    pole_figure: dict = field(default_factory=dict)


def _runs(mask: np.ndarray, minimum: int = 2) -> tuple[tuple[float, float], ...]:
    runs, start = [], None
    for index, flag in enumerate(list(mask) + [False]):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            if index - start >= minimum:
                runs.append((float(start), float(index)))
            start = None
    return tuple(runs)


def ring_orientation(
    profile: Profile,
    *,
    q_window: tuple[float, float],
    background: float = 0.0,
    min_snr: float = 3.0,
    seed: int = 0,
    horizon_chi_deg: Optional[float] = None,
) -> RingOrientation:
    """Orientation from binned mean pixel intensities before the detector Lorentz correction."""
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks

    data = profile.measured((-90.0, 90.0))
    notes = [
        "Hermans f is sin χ weighted: it assumes the film is isotropic in its plane (fibre texture).",
        f"The radial background under the ring ({background:.4g} counts/pixel) was subtracted.",
    ]
    if data.size < 10:
        # The coverage and the unmeasured |χ| of the few bins there are, not those of a ring with none.
        seen = np.zeros(90, dtype=bool)
        seen[np.clip(np.floor(np.abs(data.x)), 0, 89).astype(int)] = True
        return RingOrientation(
            tuple(q_window), background, float(seen.mean()), _runs(~seen), None, None, None, (), None, "",
            f"Only {data.size} measured χ bins in this ring.", tuple(notes),
        )
    chi, folded, error, measured, shadow, _ = fold_ring(data.x, data.y, data.sigma, background,
                                                        horizon_chi_deg=horizon_chi_deg)
    shadowed = _runs(shadow, minimum=1)
    if shadowed:
        where = ", ".join(f"{a:.0f}–{b:.0f}°" for a, b in shadowed)
        notes.append(
            f"|χ| {where} is shadowed: the intensity there is below {SHADOW_FRACTION:.0%} of the diffuse "
            "background at this q (a shadow, absorber or insensitive detector area), so it counts as unmeasured."
        )
    coverage = float(measured.mean())
    missing = _runs(~measured)
    if missing and missing[0][0] == 0.0:
        notes.append(
            f"The missing wedge covers |χ| < {missing[0][1]:.0f}°. The measured f excludes these orientations."
        )
    significance = folded[measured] / error[measured]
    if significance.size == 0 or float(np.nanmax(significance)) < min_snr:
        best = float(np.nanmax(significance)) if significance.size else 0.0
        return RingOrientation(
            tuple(q_window), background, coverage, missing, None, None, None, (), None, "",
            f"The ring is not above the radial background: at most {best:.1f}σ in any 1° χ bin.",
            tuple(notes), shadowed=shadowed,
        )
    values = folded[measured]
    angles = chi[measured]
    sine = np.sin(np.radians(chi))
    weighted = float(sine[measured].sum() / sine.sum())
    if weighted < MIN_COVERAGE:
        where = ", ".join(f"{a:.0f}–{b:.0f}°" for a, b in _runs(measured)) or "none"
        top = float(angles[int(np.argmax(values))])
        return RingOrientation(
            tuple(float(value) for value in q_window), float(background), coverage, missing, None, None, None,
            (), None, "",
            f"Only {weighted:.0%} of the orientation range Hermans f weighs (sin χ) is measured at this q "
            f"(|χ| {where}); f and the orientation distribution need most of it, above all near the sample "
            f"plane. Within the measured part the ring is strongest at |χ| ≈ {top:.0f}°.",
            tuple(notes), weighted, shadowed=shadowed,
        )
    measure = sine[measured]
    herman_isotropic = measured_hermans(angles, measure)[0]
    herman, herman_err, cos2 = measured_hermans(angles, values * measure, error[measured] * measure, seed=seed)
    if herman is None:
        return RingOrientation(
            tuple(q_window), background, coverage, missing, None, None, None, (), None, "",
            "The total sin χ weighted intensity after background subtraction is nonpositive.",
            tuple(notes), weighted, herman_isotropic, shadowed,
        )
    if herman_isotropic is not None and weighted < 0.999:
        reference = format_number(herman_isotropic).replace("-", "−")
        notes.append(f"A random (isotropic) ring measured over the same |χ| would give f = "
                     f"{reference}: compare f with that, not with 0.")
    excess = None if herman is None else float(herman - (herman_isotropic or 0.0))
    margin = _texture_margin(herman_err)
    filled = np.interp(chi, angles, values)
    smooth = gaussian_filter1d(filled, 1.5, mode="nearest")
    typical_error = float(np.nanmedian(error[measured]))
    # A finite floor on both sides lets a maximum at |χ| = 0° or 90° (a fold edge) count.
    floor = float(smooth.min() - 10.0 * (np.abs(smooth).max() + 1.0))
    # Unmeasured |χ| sit at the floor: a maximum is only ever reported where the ring was
    # measured, and one next to an unmeasured stretch says so.
    curve = np.where(measured, smooth, floor)
    padded = np.r_[floor, curve, floor]
    indices, _ = find_peaks(padded, prominence=min_snr * typical_error)
    indices = np.array([index - 1 for index in indices if measured[index - 1]], dtype=int)
    maxima = []
    if indices.size:
        # Half maximum above the smallest measured value, not above the artificial floor.
        lowest = float(smooth[measured].min())
        levels = smooth[indices] - 0.5 * (smooth[indices] - lowest)
        widths = []
        for index, level in zip(indices, levels):
            above = curve >= level
            left = index
            while left > 0 and above[left - 1]:
                left -= 1
            right = index
            while right < smooth.size - 1 and above[right + 1]:
                right += 1
            widths.append(float(right - left + 1))
        for index, width in sorted(zip(indices, widths), key=lambda item: -smooth[item[0]])[:3]:
            edge = index in (0, 89)
            gap = (index > 0 and not measured[index - 1]) or (index < 89 and not measured[index + 1])
            maximum = {
                "chi": float(chi[index]),
                "fwhm": float(2.0 * width if edge else width),
                "height": float(smooth[index]),
            }
            if gap:
                maximum["next_to_unmeasured"] = True
            maxima.append(maximum)
    observed = smooth[measured]
    low = float(np.min(observed))
    high = float(np.max(observed))
    anisotropy = high / low if low > 0 else None
    if high - low < min_snr * typical_error:
        texture = "isotropic within the noise (random orientation)"
    else:
        top_index = int(np.flatnonzero(measured)[int(np.argmax(observed))])
        top = float(chi[top_index])
        next_to_gap = (top_index > 0 and not measured[top_index - 1]) or (
            top_index < 89 and not measured[top_index + 1]
        )
        rival = next(
            (
                item for item in maxima
                if abs(item["chi"] - top) >= 30.0 and item["height"] >= SIMILAR_MAXIMUM * high
            ),
            None,
        )
        if rival is not None:
            texture = (
                f"no single preferred orientation: maxima of similar height at |χ| ≈ {top:.0f}° and "
                f"{rival['chi']:.0f}°"
            )
        elif excess is not None and abs(excess) < margin and (top < NORMAL_CHI or top > PLANE_CHI):
            texture = f"no clear preferred orientation; the strongest measured bin is at |χ| ≈ {top:.0f}°"
        elif excess is not None and ((top < NORMAL_CHI and excess < 0) or (top > PLANE_CHI and excess > 0)):
            texture = f"no single preferred orientation: the maximum at |χ| ≈ {top:.0f}° and the measured moment disagree"
        elif top < NORMAL_CHI:
            texture = "oriented along the surface normal (out-of-plane, χ ≈ 0°)"
        elif top > PLANE_CHI:
            texture = "oriented in the sample plane (in-plane, χ ≈ ±90°)"
        else:  # named from the maximum: the margin of the texture rule does not apply here
            texture = f"tilted: maximum at χ ≈ ±{top:.0f}°"
        if next_to_gap:
            texture += (
                f" — but the strongest measured |χ| ({top:.0f}°) borders an unmeasured range, so the true "
                "maximum may lie inside it"
            )
    return RingOrientation(
        tuple(float(value) for value in q_window), float(background), coverage, missing,
        None if herman is None else float(herman), herman_err, None if cos2 is None else float(cos2),
        tuple(maxima), anisotropy, texture, "", tuple(notes), weighted, herman_isotropic, shadowed,
        excess, None if excess is None else margin,
    )


__all__ = [
    "RingOrientation", "SHADOW_FRACTION", "SectorComparison", "compare_sectors", "herman_decimals", "ring_orientation",
    "signed",
]
