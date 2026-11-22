"""How a 1D cut, its composition and its parameters are given to the proposal network, and taken back (numpy).

Everything is dimensionless: q̃ = q/q_max, lengths × q_max (the forward model depends on q·R only, given q_z/q_max and
σ_q/q_max), so one network serves cuts of any distance and energy. A curve is seen on ``POINTS`` points spaced
logarithmically from 1/ρ to 1, ρ = q_max/q_min in [``RATIO_MIN``, ``RATIO_MAX``] (``window``, ``bin_curve``).

A composition is one to ``SLOTS`` populations, each of a type (family, with or without a paracrystal), listed by
increasing mean radius (``COMPOSITIONS``; slot 1 is the smallest). Every population has six values, used or not:

    R̃ = R q_max            from 0.5 to 5ρ, and each population's above the previous one's
    σR/R                   [0.02, 0.5]
    h/2R                   [0.1, 10]   random cylinders only, and h̃ = h q_max ≤ 200
    σh/h                   [0.02, 0.5] random cylinders only
    D/2R                   [1, 20]     with a paracrystal only
    σnn/D                  [0.02, 0.5] with a paracrystal only

and six values belong to the cut: the beam tail's width w̃ = w/q_max in [0.3/ρ, ``TAIL_WIDTH_MAX``/ρ] and exponent ν
in ``TAIL_EXPONENT`` (the Lorentzian wings of a beam profile, ∝ q⁻²: a steeper "tail" with a free exponent takes over the
particles' own scattering at low q, and a narrow one cannot describe a measured low-q rise), the
Gaussian resolution σ_q/q_max in [10⁻⁴, 10⁻²], the amplitude τ in [0.003, 0.15] of a smooth multiplicative model
error exp(Σ b_k B_k), b_k ~ N(0, τ²) over the ``ERROR_BASIS`` cubic B-splines of ``error_basis``, and the real and
imaginary part of the effective reflection coefficient r of the exit-reflected path of spheres, each in [−1, 1]
(``cut_model.geometry``, ``two_path``): they matter only when the cut's two paths are known (``conditions``), and
are drawn from the prior otherwise.

The prior is uniform in the unit box v of these values: ln for R̃, h/2R, D/2R, w̃, σ_q and τ, linear for the others,
and for R̃ of slot j > 1 the fraction of the way from the previous slot's ln R̃ to the top (so the slots stay ordered).
A random cylinder has R̃ ≤ ``CYLINDER_RADIUS`` (so the slots before it too) and h̃ ≤ ``CYLINDER_HEIGHT``: rods larger
than that against the window are left to Search and Refine, and the tops above are those of their slots.
The network works on z = logit(v); the prior density of z is logistic (``log_prior``). Values a composition does not
have are inactive (``active``): NaN as values, 0 in z, not seen by the flow.
"""

from __future__ import annotations

import itertools

import numpy as np

FAMILIES = ("sphere", "vertical_cylinder", "cylinder")
SLOTS = 3
TYPES = tuple((family, structure) for family in FAMILIES for structure in (True, False))
COMPOSITIONS = tuple(itertools.chain.from_iterable(
    itertools.product(range(len(TYPES)), repeat=k) for k in range(1, SLOTS + 1)))
SLOT_NAMES = ("R", "sigma_R", "h_over_2R", "sigma_h", "D_over_2R", "sigma_nn")
GLOBAL_NAMES = ("tail_width", "tail_exponent", "q_resolution", "model_error", "reflection_re", "reflection_im")
NAMES = tuple(f"{name}_{slot + 1}" for slot in range(SLOTS) for name in SLOT_NAMES) + GLOBAL_NAMES
PER_SLOT = len(SLOT_NAMES)
DIMS = len(NAMES)
GLOBAL = slice(SLOTS * PER_SLOT, DIMS)
LOGARITHMIC = np.array([n.rsplit("_", 1)[0] in ("R", "h_over_2R", "D_over_2R") for n in NAMES[:GLOBAL.start]]
                       + [True, False, True, True, False, False])
POINTS = 128
RATIO_MIN, RATIO_MAX = 10.0, 300.0
CHANNELS = 4
CONDITIONS = 4
SLOT_CODE = 2 + len(FAMILIES)
"""Per slot of ``composition_code``: in use, the family (one-hot), a paracrystal."""
ERROR_BASIS = 8
TAIL_WIDTH_MAX = 30.0
TAIL_EXPONENT = (1.8, 2.2)
VERSION = 4
"""Of this encoding, recorded with a network (``flow.Network`` refuses another): 4 since spheres can have the
exit-reflected path (r and the cut's k_i − k_f); 3: the tail's exponent is ``TAIL_EXPONENT`` and its width at most
``TAIL_WIDTH_MAX`` q_min (2: ν in [1, 6], w at most 3 q_min; 1: at most 0.1 q_max)."""
CYLINDER_RADIUS, CYLINDER_HEIGHT = 50.0, 200.0


def bounds(ratio) -> tuple[np.ndarray, np.ndarray]:
    """(lo, hi) of every value's coordinate (ln for the logarithmic ones) for windows of q_max/q_min = ``ratio``."""
    ratio = np.atleast_1d(np.asarray(ratio, float))[:, None]
    slot_lo = [np.log(0.5), 0.02, np.log(0.1), 0.02, 0.0, 0.02]
    slot_hi = [np.nan, 0.5, np.log(10.0), 0.5, np.log(20.0), 0.5]
    lo = np.concatenate([np.tile(slot_lo, SLOTS), [0.0, TAIL_EXPONENT[0], np.log(1e-4), np.log(3e-3), -1.0, -1.0]])[None, :]         * np.ones_like(ratio)
    hi = np.concatenate([np.tile(slot_hi, SLOTS), [0.0, TAIL_EXPONENT[1], np.log(1e-2), np.log(0.15), 1.0, 1.0]])[None, :]         * np.ones_like(ratio)
    lo[:, GLOBAL.start] = np.log(0.3 / ratio[:, 0])
    hi[:, GLOBAL.start] = np.log(TAIL_WIDTH_MAX / ratio[:, 0])
    hi[:, 0:GLOBAL.start:PER_SLOT] = np.log(5.0 * ratio)
    return lo, hi


def composition_types(index: int) -> tuple:
    """The (family, paracrystal) of every slot of composition ``index``."""
    return tuple(TYPES[t] for t in COMPOSITIONS[index])


def composition_index(types) -> int:
    """The index of a composition given as (family, paracrystal) pairs by increasing radius."""
    return COMPOSITIONS.index(tuple(TYPES.index((family, bool(structure))) for family, structure in types))


def composition_name(index: int) -> str:
    names = {"sphere": "sphere", "vertical_cylinder": "upright cylinder", "cylinder": "random cylinder"}
    return " + ".join(names[f] + (" (S)" if s else "") for f, s in composition_types(index))


def select(families=FAMILIES, max_populations: int = SLOTS, structure=None) -> list[int]:
    """Compositions of up to ``max_populations`` populations of the given families; ``structure`` True (every one
    with a paracrystal), False (none) or None (either)."""
    out = []
    for c, slots in enumerate(COMPOSITIONS):
        types = composition_types(c)
        if len(slots) <= max_populations and all(f in families for f, _s in types) and                 (structure is None or all(s == structure for _f, s in types)):
            out.append(c)
    return out


def active(index) -> np.ndarray:
    """Which values compositions ``index`` (B,) have: (B, DIMS)."""
    index = np.atleast_1d(np.asarray(index, int))
    table = np.zeros((len(COMPOSITIONS), DIMS), bool)
    for c, slots in enumerate(COMPOSITIONS):
        for slot, t in enumerate(slots):
            family, structure = TYPES[t]
            row = table[c, slot * PER_SLOT:(slot + 1) * PER_SLOT]
            row[:2] = True
            row[2:4] = family == "cylinder"
            row[4:6] = structure
        table[c, GLOBAL] = True
    return table[index]


def composition_code(index) -> np.ndarray:
    """What the flow is told of compositions ``index`` (B,): per slot in use, family one-hot, paracrystal."""
    index = np.atleast_1d(np.asarray(index, int))
    table = np.zeros((len(COMPOSITIONS), SLOTS * SLOT_CODE), np.float32)
    for c, slots in enumerate(COMPOSITIONS):
        for slot, t in enumerate(slots):
            family, structure = TYPES[t]
            base = slot * SLOT_CODE
            table[c, base] = 1.0
            table[c, base + 1 + FAMILIES.index(family)] = 1.0
            table[c, base + SLOT_CODE - 1] = float(structure)
    return table[index]


def _coordinates(values):
    values = np.asarray(values, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(LOGARITHMIC, np.log(values), values)


def _capped_table() -> np.ndarray:
    """Per composition and slot: a random cylinder in this slot or a later one (its radius cap applies)."""
    table = np.zeros((len(COMPOSITIONS), SLOTS), bool)
    for c, slots in enumerate(COMPOSITIONS):
        for slot in range(len(slots)):
            table[c, slot] = any(TYPES[t][0] == "cylinder" for t in slots[slot:])
    return table


_CAPPED = _capped_table()


def _slot_tops(lo, hi, index, slot, base):
    """(bottom, top) of slot's ln R̃ and the top of its ln(h/2R) given ln R̃ (a function)."""
    j = slot * PER_SLOT
    top = np.where(_CAPPED[index, slot], np.minimum(hi[:, j], np.log(CYLINDER_RADIUS)), hi[:, j])
    return (lo[:, j] if base is None else base), top


def _height_top(hi, j, log_radius):
    return np.minimum(hi[:, j + 2], np.log(CYLINDER_HEIGHT / 2.0) - log_radius)


def to_unit(values, ratio, index) -> np.ndarray:
    """The unit-box position v of values (B, DIMS) of compositions ``index`` (NaN where inactive stays NaN)."""
    c = _coordinates(values)
    index = np.broadcast_to(np.asarray(index, int), (c.shape[0],))
    lo, hi = bounds(ratio)
    v = (c - lo) / (hi - lo)
    base = None
    for slot in range(SLOTS):
        j = slot * PER_SLOT
        bottom, top = _slot_tops(lo, hi, index, slot, base)
        v[:, j] = (c[:, j] - bottom) / (top - bottom)
        v[:, j + 2] = (c[:, j + 2] - lo[:, j + 2]) / (_height_top(hi, j, c[:, j]) - lo[:, j + 2])
        base = np.where(np.isfinite(c[:, j]), c[:, j], bottom)
    return v


def from_unit(v, ratio, index) -> np.ndarray:
    """Values (B, DIMS) of unit-box positions v of compositions ``index``; NaN where inactive."""
    v = np.asarray(v, float)
    index = np.broadcast_to(np.asarray(index, int), (v.shape[0],))
    lo, hi = bounds(ratio)
    c = lo + v * (hi - lo)
    base = None
    for slot in range(SLOTS):
        j = slot * PER_SLOT
        bottom, top = _slot_tops(lo, hi, index, slot, base)
        c[:, j] = bottom + v[:, j] * (top - bottom)
        c[:, j + 2] = lo[:, j + 2] + v[:, j + 2] * (_height_top(hi, j, c[:, j]) - lo[:, j + 2])
        base = c[:, j]
    values = np.where(LOGARITHMIC, np.exp(c), c)
    return np.where(active(index), values, np.nan)


def to_z(values, ratio, index) -> np.ndarray:
    """z of values (B, DIMS) of compositions ``index``; 0 where inactive."""
    v = np.clip(to_unit(values, ratio, index), 1e-9, 1 - 1e-9)
    z = np.log(v) - np.log1p(-v)
    return np.where(active(np.broadcast_to(index, (z.shape[0],))), np.nan_to_num(z), 0.0)


def from_z(z, ratio, index) -> np.ndarray:
    """Values (B, DIMS) of z; NaN where inactive."""
    return from_unit(0.5 * (1.0 + np.tanh(0.5 * np.asarray(z, float))), ratio, index)


def log_prior(z, index, dims=None) -> np.ndarray:
    """log p(z) of each row under the prior (uniform in v), over the active values of compositions ``index``."""
    z = np.asarray(z, float)
    terms = -np.logaddexp(0.0, -z) - np.logaddexp(0.0, z)
    if dims is not None:
        return np.sum(terms[:, dims], axis=1)
    return np.sum(np.where(active(np.broadcast_to(index, (z.shape[0],))), terms, 0.0), axis=1)


def grid(ratio: float, points: int = POINTS) -> np.ndarray:
    """q̃ of the network's points for a window of q_max/q_min = ``ratio``."""
    return np.geomspace(1.0 / float(ratio), 1.0, points)


def window(q) -> tuple[float, float]:
    """(q_max, ρ) of the network's window for a curve measured at |q| (any unit): from its largest q down to its
    smallest, at most ``RATIO_MAX`` below; a curve spanning less than ``RATIO_MIN`` is refused."""
    q = np.abs(np.asarray(q, float))
    q = q[np.isfinite(q) & (q > 0)]
    if q.size < 2 or q.max() / q.min() < RATIO_MIN:
        raise ValueError(f"The proposal network needs a cut spanning at least a factor {RATIO_MIN:g} in q")
    return float(q.max()), float(min(q.max() / q.min(), RATIO_MAX))


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
    any_point = mask.any(axis=1, keepdims=True)
    median = np.nanmedian(np.where(any_point, masked, 0.0), axis=1, keepdims=True)
    out = np.zeros((intensity.shape[0], CHANNELS, intensity.shape[1]), np.float32)
    out[:, 0] = np.where(mask, (log_i - median) / 5.0, 0.0)
    out[:, 1] = np.where(mask, np.clip(np.log(np.maximum(sigma, 1e-300) / level), -12.0, 3.0) / 5.0, 0.0)
    out[:, 2] = mask
    out[:, 3] = np.linspace(-1.0, 0.0, intensity.shape[1])[None, :]
    return out


def conditions(ratio, qz=0.0, qz_minus=np.nan) -> np.ndarray:
    """What the network is told of the cut (B, CONDITIONS): ln ρ / ln RATIO_MAX, q₊/q_max = (k_i + k_f)/q_max (0: the
    effective interface), q₋/q_max = (k_i − k_f)/q_max and 1 when spheres have the exit-reflected path (two paths:
    ``qz_minus`` given, finite, with q₊ > 0); else 0 and 0 (the Born term at q₊ alone)."""
    ratio = np.atleast_1d(np.asarray(ratio, float))
    qz = np.broadcast_to(np.asarray(qz, float), ratio.shape)
    minus = np.broadcast_to(np.asarray(qz_minus, float), ratio.shape)
    reflected = np.isfinite(minus) & (qz > 0)
    out = np.zeros((ratio.size, CONDITIONS), np.float32)
    out[:, 0] = np.log(ratio) / np.log(RATIO_MAX)
    out[:, 1] = qz
    out[:, 2] = np.where(reflected, minus, 0.0)
    out[:, 3] = reflected
    return out


def error_basis(position, count: int = ERROR_BASIS) -> np.ndarray:
    """The ``count`` clamped cubic B-splines on [0, 1] at ``position`` (any shape): array (count, *shape)."""
    x = np.clip(np.asarray(position, float), 0.0, 1.0)
    inner = np.linspace(0.0, 1.0, count - 2)
    knots = np.concatenate([[0.0] * 3, inner, [1.0] * 3])
    # degree 0, with the last interval closed so that x = 1 has a basis function
    basis = [((knots[i] <= x) & (x < knots[i + 1])) | ((x == 1.0) & (knots[i] < 1.0) & (knots[i + 1] == 1.0))
             for i in range(len(knots) - 1)]
    basis = [b.astype(float) for b in basis]
    for degree in range(1, 4):
        new = []
        for i in range(len(knots) - degree - 1):
            left_den = knots[i + degree] - knots[i]
            right_den = knots[i + degree + 1] - knots[i + 1]
            left = (x - knots[i]) / left_den * basis[i] if left_den > 0 else 0.0
            right = (knots[i + degree + 1] - x) / right_den * basis[i + 1] if right_den > 0 else 0.0
            new.append(left + right)
        basis = new
    return np.stack(basis)


def bin_curve(q, intensity, sigma, q_max: float, ratio: float, points: int = POINTS):
    """A measured curve on the network's grid: (I, σ, mask) of ``points`` log bins from q_max/ratio to q_max, each the
    σ-weighted mean of the measured points in it (σ combined); a bin without points is masked."""
    q, intensity, sigma = (np.asarray(a, float) for a in (q, intensity, sigma))
    centres = q_max * grid(ratio, points)
    edges = np.sqrt(centres[1:] * centres[:-1])
    edges = np.concatenate([[centres[0] ** 2 / edges[0]], edges, [centres[-1] ** 2 / edges[-1]]])
    which = np.digitize(np.abs(q), edges) - 1
    good = (which >= 0) & (which < points) & np.isfinite(intensity) & (sigma > 0)
    weight = np.where(good, 1.0 / np.where(good, sigma, 1.0) ** 2, 0.0)
    index = np.where(good, which, 0)
    total = np.bincount(index, weight, points)
    mean = np.bincount(index, weight * np.where(good, intensity, 0.0), points)
    mask = total > 0
    out_i = np.where(mask, mean / np.where(mask, total, 1.0), 0.0)
    out_s = np.where(mask, 1.0 / np.sqrt(np.where(mask, total, 1.0)), 1.0)
    return out_i, out_s, mask


__all__ = ["CHANNELS", "COMPOSITIONS", "CONDITIONS", "CYLINDER_HEIGHT", "CYLINDER_RADIUS", "DIMS", "TAIL_EXPONENT", "TAIL_WIDTH_MAX", "VERSION", "ERROR_BASIS", "FAMILIES", "GLOBAL", "NAMES", "PER_SLOT",
           "POINTS", "RATIO_MAX", "RATIO_MIN", "SLOTS", "SLOT_CODE", "TYPES", "active", "bin_curve", "bounds",
           "composition_code", "composition_index", "composition_name", "composition_types", "conditions",
           "error_basis", "features", "from_unit", "from_z", "grid", "log_prior", "select", "to_unit", "to_z", "window"]
