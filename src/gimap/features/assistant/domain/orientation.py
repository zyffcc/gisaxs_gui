"""Crystal orientation from GIWAXS profiles.

χ follows Analyze: 0° along the surface normal (out of plane, qz) and ±90° in
the sample plane (q∥).  Two views:

* ``compare_sectors`` — at each peak, the net intensity per pixel in the
  in-plane and out-of-plane sectors, their ratio and which dominates;
* ``ring_orientation`` — the azimuthal profile I(χ) of one ring, folded to
  |χ|, with the measured χ coverage, its maxima and Herman's orientation
  factor f = (3⟨cos²χ⟩ − 1)/2 relative to the surface normal (sin χ weighted,
  i.e. assuming a fibre texture about the normal).  f = 1: along the normal,
  0: random, −0.5: in plane.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np

from .profiles import Profile, estimate_background

DOMINANCE_RATIO = 2.0
SHADOW_FRACTION = 0.2
"""Intensity below this fraction of the diffuse background at the same q cannot come from the
sample: a shadow, an absorber or an insensitive detector area.  Such pixels count as unmeasured."""
MIN_COVERAGE = 0.8
"""Fraction of the sin χ weight of |χ| 0–90° that must be measured for Herman's f.

f weights each |χ| by sin χ, so the part near the sample plane counts most: an isotropic
ring measured only over |χ| < 59° already gives f ≈ 0.4, while a missing wedge below 13°
hardly matters."""
MONTE_CARLO_SAMPLES = 200
SIMILAR_MAXIMUM = 0.85
"""A second maximum at least this high (relative to the first) and 30° away: no single orientation."""
WEAK_HERMAN = 0.1
"""|f| below this does not support "along the normal" or "in plane", whatever the strongest |χ|."""


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
    """Measured fraction of the sin χ weight Herman's f gives to |χ| 0–90°."""
    herman_isotropic: Optional[float] = None
    """f a random (isotropic) ring would give over the same measured |χ|: the reference to compare f with."""
    shadowed: tuple[tuple[float, float], ...] = ()
    """|χ| ranges with pixels far below the diffuse background (a shadow); they count as unmeasured."""


def _fold(chi: np.ndarray, net: np.ndarray, sigma: np.ndarray):
    """Inverse-variance mean of ±χ into 1° bins of |χ| (0…90°)."""
    bins = np.clip(np.floor(np.abs(chi)), 0, 89).astype(int)
    weights = 1.0 / sigma ** 2
    total = np.bincount(bins, weights=weights, minlength=90)
    value = np.bincount(bins, weights=weights * net, minlength=90)
    measured = total > 0
    folded = np.where(measured, value / np.where(measured, total, 1.0), np.nan)
    error = np.where(measured, 1.0 / np.sqrt(np.where(measured, total, 1.0)), np.nan)
    return np.arange(90) + 0.5, folded, error, measured


def _herman(chi_deg: np.ndarray, intensity: np.ndarray) -> Optional[tuple[float, float]]:
    weight = np.clip(intensity, 0.0, None) * np.sin(np.radians(chi_deg))
    total = float(weight.sum())
    if total <= 0:
        return None
    cos2 = float((weight * np.cos(np.radians(chi_deg)) ** 2).sum() / total)
    return 1.5 * cos2 - 0.5, cos2


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
) -> RingOrientation:
    """Orientation of one ring from its I(χ) (x in degrees, −90…90)."""
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks

    data = profile.measured((-90.0, 90.0))
    notes = [
        "Herman's f is sin χ weighted: it assumes the film is isotropic in its plane (fibre texture).",
        f"The radial background under the ring ({background:.4g} counts/pixel) was subtracted.",
    ]
    empty = RingOrientation(tuple(q_window), background, 0.0, ((0.0, 90.0),), None, None, None, (), None, "", "", tuple(notes))
    if data.size < 10:
        return RingOrientation(**{**empty.__dict__, "reason": f"Only {data.size} measured χ bins in this ring."})
    chi, folded, error, measured = _fold(data.x, data.y - background, data.sigma)
    shadowed: tuple[tuple[float, float], ...] = ()
    if background > 0:
        # Far below the diffuse background nothing comes from the sample: a shadow (with its edge bins).
        core = measured & (folded + background < SHADOW_FRACTION * background)
        if core.any():
            shadow = measured & (np.convolve(core.astype(int), [1, 1, 1], mode="same") > 0)
            shadowed = _runs(shadow, minimum=1)
            where = ", ".join(f"{a:.0f}–{b:.0f}°" for a, b in shadowed)
            notes.append(
                f"|χ| {where} is shadowed: the intensity there is below {SHADOW_FRACTION:.0%} of the diffuse "
                "background at this q (a shadow, absorber or insensitive detector area), so it counts as unmeasured."
            )
            measured = measured & ~shadow
    coverage = float(measured.mean())
    missing = _runs(~measured)
    if missing and missing[0][0] == 0.0:
        notes.append(
            f"|χ| < {missing[0][1]:.0f}° is not measured (missing wedge); f then leaves out the "
            "orientations closest to the surface normal and is biased low."
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
            f"Only {weighted:.0%} of the orientation range Herman's f weighs (sin χ) is measured at this q "
            f"(|χ| {where}); f and the orientation distribution need most of it, above all near the sample "
            f"plane. Within the measured part the ring is strongest at |χ| ≈ {top:.0f}°.",
            tuple(notes), weighted, shadowed=shadowed,
        )
    isotropic = _herman(angles, np.ones_like(angles))
    herman_isotropic = None if isotropic is None else float(isotropic[0])
    if herman_isotropic is not None and weighted < 0.999:
        notes.append(
            f"A random (isotropic) ring measured over the same |χ| would give f = {herman_isotropic:.2f}: "
            "compare f with that, not with 0."
        )
    result = _herman(angles, values)
    herman, cos2 = (result if result is not None else (None, None))
    herman_err = None
    if result is not None:
        rng = np.random.default_rng(seed)
        samples = []
        for _ in range(MONTE_CARLO_SAMPLES):
            trial = _herman(angles, values + rng.normal(0.0, error[measured]))
            if trial is not None:
                samples.append(trial[0])
        herman_err = float(np.std(samples)) if len(samples) > 10 else None
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
        elif herman is not None and abs(herman) < WEAK_HERMAN and (top < 20.0 or top > 70.0):
            texture = (
                f"weak or no preferred orientation (f = {herman:.2f}): the ring is strongest at "
                f"|χ| ≈ {top:.0f}° but nearly as intense elsewhere"
            )
        elif top < 20.0:
            texture = "oriented along the surface normal (out-of-plane, χ ≈ 0°)"
        elif top > 70.0:
            texture = "oriented in the sample plane (in-plane, χ ≈ ±90°)"
        else:
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
    )


__all__ = ["RingOrientation", "SHADOW_FRACTION", "SectorComparison", "compare_sectors", "ring_orientation"]
