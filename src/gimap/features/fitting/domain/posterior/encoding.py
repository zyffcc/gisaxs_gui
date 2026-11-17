"""How a 1D cut and a population's parameters are given to the proposal network, and taken back (numpy).

Everything is dimensionless: q̃ = q/q_max, lengths × q_max (the forward model depends on q·R only, given q_z/q_max and
σ_q/q_max), so one network serves cuts of any distance and energy. A curve is seen on ``POINTS`` points spaced
logarithmically from 1/ρ to 1, ρ = q_max/q_min in [``RATIO_MIN``, ``RATIO_MAX``].

Parameters of one population and the beam tail, in this order (``NAMES``), with the bounds of the prior (``bounds``):

    ln R̃       R̃ = R q_max in [0.5, 5ρ]: from what q_max resolves to what q_min still sees
    σR/R       [0.02, 0.5]
    ln(D/2R)   D/2R in [1, 20] (particles of the mean radius do not overlap)
    σnn/D      [0.02, 0.5]
    ln w̃       the beam tail's width, w̃ = w/q_max in [0.1/ρ, 1/3]
    ν          the beam tail's exponent, [1, 6]

The network works on z = logit((v − lo)/(hi − lo)) of each v; ``to_z`` and ``from_z`` convert, with the log-Jacobian
that turns a density of z into one of v. D and σnn/D are inactive (masked) without a structure factor.
"""

from __future__ import annotations

import numpy as np

NAMES = ("ln_R", "sigma_R", "ln_D_over_2R", "sigma_nn", "ln_tail_width", "tail_exponent")
STRUCTURE_DIMS = (2, 3)
POINTS = 128
RATIO_MIN, RATIO_MAX = 10.0, 300.0
FAMILIES = ("sphere", "vertical_cylinder")
CHANNELS = 4
CONDITIONS = len(FAMILIES) + 4


def bounds(ratio) -> tuple[np.ndarray, np.ndarray]:
    """(lo, hi) of the six values for windows of q_max/q_min = ``ratio`` (B,): arrays (B, 6)."""
    ratio = np.atleast_1d(np.asarray(ratio, float))
    ones = np.ones_like(ratio)
    lo = np.stack([np.log(0.5) * ones, 0.02 * ones, 0.0 * ones, 0.02 * ones, np.log(0.1 / ratio), 1.0 * ones], axis=1)
    hi = np.stack([np.log(5.0 * ratio), 0.5 * ones, np.log(20.0) * ones, 0.5 * ones, np.log(1 / 3) * ones, 6.0 * ones],
                  axis=1)
    return lo, hi


def to_z(values, ratio) -> np.ndarray:
    """z of the values (B, 6)."""
    lo, hi = bounds(ratio)
    unit = np.clip((np.asarray(values, float) - lo) / (hi - lo), 1e-9, 1 - 1e-9)
    return np.log(unit) - np.log1p(-unit)


def from_z(z, ratio) -> tuple[np.ndarray, np.ndarray]:
    """(values (B, 6), log|dz/dv| (B, 6)): the values of z, and what a log-density of z gains to be one of v."""
    lo, hi = bounds(ratio)
    z = np.asarray(z, float)
    unit = 0.5 * (1.0 + np.tanh(0.5 * z))  # the logistic function, stable for large |z|
    log_jacobian = -np.log(np.maximum(unit * (1 - unit), 1e-300)) - np.log(hi - lo)
    return lo + unit * (hi - lo), log_jacobian


def grid(ratio: float, points: int = POINTS) -> np.ndarray:
    """q̃ of the network's points for a window of q_max/q_min = ``ratio``."""
    return np.geomspace(1.0 / float(ratio), 1.0, points)


def features(intensity, sigma, mask) -> np.ndarray:
    """The network's input (B, CHANNELS, POINTS) from intensities, their σ and a mask of measured points (B, POINTS).

    0: ln I − median over the measured points, /5 (I at least 0.1 σ: a point at 0 counts is not −∞);
    1: ln(σ/I), clipped to [−12, 3], /5;  2: the mask;  3: the position on the log grid, −1 … 0."""
    intensity = np.asarray(intensity, float)
    sigma = np.asarray(sigma, float)
    mask = np.asarray(mask, bool)
    level = np.maximum(intensity, 0.1 * sigma)
    log_i = np.log(np.maximum(level, 1e-300))
    masked = np.where(mask, log_i, np.nan)
    median = np.nanmedian(masked, axis=1, keepdims=True)
    median = np.where(np.isfinite(median), median, 0.0)
    out = np.zeros((intensity.shape[0], CHANNELS, intensity.shape[1]), np.float32)
    out[:, 0] = np.where(mask, (log_i - median) / 5.0, 0.0)
    out[:, 1] = np.where(mask, np.clip(np.log(sigma / level), -12.0, 3.0) / 5.0, 0.0)
    out[:, 2] = mask
    out[:, 3] = np.linspace(-1.0, 0.0, intensity.shape[1])[None, :]
    return out


def conditions(family, structure, ratio, qz=0.0, q_resolution=0.0) -> np.ndarray:
    """The network's conditions (B, CONDITIONS): the family (one-hot of ``FAMILIES``), a structure factor or not,
    ln ρ scaled, q_z/q_max and σ_q/q_max of the cut."""
    family = np.atleast_1d(np.asarray(family))
    rows = family.size
    out = np.zeros((rows, CONDITIONS), np.float32)
    if not np.all(np.isin(family, FAMILIES)):
        raise ValueError(f"The network knows the families {', '.join(FAMILIES)}")
    for index, name in enumerate(FAMILIES):
        out[:, index] = family == name
    n = len(FAMILIES)
    out[:, n] = np.broadcast_to(np.asarray(structure, float), (rows,))
    out[:, n + 1] = np.log(np.broadcast_to(np.asarray(ratio, float), (rows,))) / np.log(RATIO_MAX)
    out[:, n + 2] = np.broadcast_to(np.asarray(qz, float), (rows,))
    out[:, n + 3] = 100.0 * np.broadcast_to(np.asarray(q_resolution, float), (rows,))
    return out


def active(structure) -> np.ndarray:
    """Which of the six values a row has (B, 6): D and σnn/D only with a structure factor."""
    structure = np.atleast_1d(np.asarray(structure, bool))
    out = np.ones((structure.size, len(NAMES)), bool)
    out[:, list(STRUCTURE_DIMS)] = structure[:, None]
    return out


def bin_curve(q, intensity, sigma, q_max: float, ratio: float, points: int = POINTS):
    """A measured curve on the network's grid: (I, σ, mask) of ``points`` log bins from q_max/ratio to q_max, each the
    σ-weighted mean of the measured points in it (σ combined); a bin without points is masked."""
    q, intensity, sigma = (np.asarray(a, float) for a in (q, intensity, sigma))
    centres = q_max * grid(ratio, points)
    edges = np.sqrt(centres[1:] * centres[:-1])
    edges = np.concatenate([[centres[0] ** 2 / edges[0]], edges, [centres[-1] ** 2 / edges[-1]]])
    which = np.digitize(q, edges) - 1
    out_i, out_s, mask = np.zeros(points), np.ones(points), np.zeros(points, bool)
    good = (which >= 0) & (which < points) & np.isfinite(intensity) & (sigma > 0)
    for b in np.unique(which[good]):
        sel = good & (which == b)
        w = 1.0 / sigma[sel] ** 2
        out_i[b] = np.sum(w * intensity[sel]) / w.sum()
        out_s[b] = 1.0 / np.sqrt(w.sum())
        mask[b] = True
    return out_i, out_s, mask


__all__ = ["CHANNELS", "CONDITIONS", "FAMILIES", "NAMES", "POINTS", "RATIO_MAX", "RATIO_MIN", "STRUCTURE_DIMS", "active",
           "bin_curve", "bounds", "conditions", "features", "from_z", "grid", "to_z"]
