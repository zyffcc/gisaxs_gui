"""Random synthetic sample generation."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from TrainSetBuild import constraints, schema
from TrainSetBuild.noise import add_noise
from TrainSetBuild.physics_adapter import component_array_to_dict, evaluate_clean, global_array_to_dict

SAMPLING_MODE_GLOBAL = -1
SAMPLING_MODE_OBSERVABLE = 0
SAMPLING_MODE_EDGE = 1
SAMPLING_MODE_OUT_OF_WINDOW = 2
SAMPLING_MODE_NAMES = {
    SAMPLING_MODE_GLOBAL: "global",
    SAMPLING_MODE_OBSERVABLE: "observable",
    SAMPLING_MODE_EDGE: "edge",
    SAMPLING_MODE_OUT_OF_WINDOW: "out_of_window",
}

SPHERE_R_COEFF = 4.49
VERTICAL_CYLINDER_R_COEFF = 3.83
CYLINDER_R_COEFF = 3.83
CYLINDER_R_MARGIN_MULTIPLIER = 2.0
STRUCTURE_PERIOD_COEFF = 2.0 * np.pi


def log_uniform(rng: np.random.Generator, low: float, high: float) -> float:
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))


def sample_q_grid(rng: np.random.Generator, max_points: int = schema.MAX_POINTS) -> np.ndarray:
    n = int(rng.integers(128, max_points + 1))
    q_low = log_uniform(rng, schema.Q_MIN_GLOBAL, 0.08)
    q_high_low = max(0.5, 5.0 * q_low)
    q_high = log_uniform(rng, q_high_low, 10.0)
    q_high = max(q_high, q_low * 5.01)
    return np.linspace(q_low, q_high, n).astype(np.float64)


def sample_number_of_components(
    rng: np.random.Generator,
    k_values: np.ndarray | None = None,
    k_probs: np.ndarray | None = None,
) -> int:
    if k_values is None:
        k_values = np.array([1, 2, 3, 4], dtype=np.int32)
        k_probs = np.array([0.40, 0.40, 0.15, 0.05], dtype=np.float64)
    else:
        k_values = np.asarray(k_values, dtype=np.int32)
        if k_probs is None:
            k_probs = np.ones(len(k_values), dtype=np.float64)
        else:
            k_probs = np.asarray(k_probs, dtype=np.float64)
    k_probs = k_probs / np.sum(k_probs)
    return int(rng.choice(k_values, p=k_probs))


def sample_component_type(rng: np.random.Generator) -> int:
    tids = np.array(list(schema.TYPE_PROBS.keys()), dtype=np.int32)
    probs = np.array(list(schema.TYPE_PROBS.values()), dtype=np.float64)
    return int(rng.choice(tids, p=probs / probs.sum()))


def visible_range_from_q(
    q_min: float,
    q_max: float,
    coeff: float,
    global_min: float,
    global_max: float,
    margin: float,
) -> Tuple[float, float]:
    margin = max(float(margin), 1.0)
    q_min_eff = max(float(q_min) / margin, 1e-12)
    q_max_eff = max(float(q_max) * margin, q_min_eff * (1.0 + 1e-9))
    low = max(float(global_min), float(coeff) / q_max_eff)
    high = min(float(global_max), float(coeff) / q_min_eff)
    if not np.isfinite(low) or not np.isfinite(high) or low > high:
        return float(global_min), float(global_max)
    return float(low), float(high)


def _log_interval_width(low: float, high: float) -> float:
    if high <= low or low <= 0:
        return 0.0
    return float(np.log(high) - np.log(low))


def sample_out_of_window_length(
    rng: np.random.Generator,
    q_min: float,
    q_max: float,
    coeff: float,
    global_min: float,
    global_max: float,
) -> float:
    visible_low, visible_high = visible_range_from_q(
        q_min,
        q_max,
        coeff,
        global_min,
        global_max,
        margin=1.0,
    )
    segments = []
    if global_min < visible_low:
        segments.append((float(global_min), float(visible_low)))
    if visible_high < global_max:
        segments.append((float(visible_high), float(global_max)))
    widths = np.array([_log_interval_width(low, high) for low, high in segments], dtype=np.float64)
    if not segments or widths.sum() <= 0:
        return log_uniform(rng, global_min, global_max)
    idx = int(rng.choice(np.arange(len(segments)), p=widths / widths.sum()))
    low, high = segments[idx]
    return log_uniform(rng, low, high)


def sample_visible_length(
    rng: np.random.Generator,
    q_min: float,
    q_max: float,
    coeff: float,
    range_name: str,
    sampling_mode: int,
    margin: float,
) -> float:
    spec = schema.PARAM_RANGES[range_name]
    if sampling_mode == SAMPLING_MODE_OUT_OF_WINDOW:
        return sample_out_of_window_length(rng, q_min, q_max, coeff, spec.low, spec.high)
    low, high = visible_range_from_q(q_min, q_max, coeff, spec.low, spec.high, margin=margin)
    return log_uniform(rng, low, high)


def sample_sampling_mode(
    rng: np.random.Generator,
    q_conditioned_sampling: bool = True,
    visible_fraction: float = 0.70,
    edge_fraction: float = 0.20,
    out_of_window_fraction: float = 0.10,
) -> int:
    if not q_conditioned_sampling:
        return SAMPLING_MODE_GLOBAL
    probs = np.array([visible_fraction, edge_fraction, out_of_window_fraction], dtype=np.float64)
    probs = probs / np.sum(probs)
    modes = np.array([SAMPLING_MODE_OBSERVABLE, SAMPLING_MODE_EDGE, SAMPLING_MODE_OUT_OF_WINDOW], dtype=np.int32)
    return int(rng.choice(modes, p=probs))


def sample_component_params(
    type_id: int,
    rng: np.random.Generator,
    q_min: float | None = None,
    q_max: float | None = None,
    sampling_mode: int = SAMPLING_MODE_GLOBAL,
) -> np.ndarray:
    p = np.zeros(schema.P_MAX, dtype=np.float32)
    q_conditioned = sampling_mode != SAMPLING_MODE_GLOBAL and q_min is not None and q_max is not None
    margin = 2.0 if sampling_mode == SAMPLING_MODE_OBSERVABLE else 5.0
    if q_conditioned:
        if type_id == schema.TYPE_SPHERE:
            r_coeff = SPHERE_R_COEFF
            r_margin = margin
        elif type_id == schema.TYPE_VERTICAL_CYLINDER:
            r_coeff = VERTICAL_CYLINDER_R_COEFF
            r_margin = margin
        else:
            r_coeff = CYLINDER_R_COEFF
            r_margin = margin * CYLINDER_R_MARGIN_MULTIPLIER
        r = sample_visible_length(rng, q_min, q_max, r_coeff, "R", sampling_mode, r_margin)
    else:
        r = log_uniform(rng, schema.PARAM_RANGES["R"].low, schema.PARAM_RANGES["R"].high)
    p[0] = r
    if type_id == schema.TYPE_VERTICAL_CYLINDER:
        p[1] = rng.uniform(schema.PARAM_RANGES["vertical_sigma_R"].low, schema.PARAM_RANGES["vertical_sigma_R"].high)
    else:
        p[1] = rng.uniform(schema.PARAM_RANGES["sigma_R_frac"].low, schema.PARAM_RANGES["sigma_R_frac"].high) * r

    if type_id == schema.TYPE_CYLINDER:
        if q_conditioned:
            h = sample_visible_length(rng, q_min, q_max, STRUCTURE_PERIOD_COEFF, "h", sampling_mode, margin)
        else:
            h = log_uniform(rng, schema.PARAM_RANGES["h"].low, schema.PARAM_RANGES["h"].high)
        p[2] = h
        p[3] = rng.uniform(schema.PARAM_RANGES["sigma_h_frac"].low, schema.PARAM_RANGES["sigma_h_frac"].high) * h

    if rng.random() < 0.25:
        p[4] = 0.0
        p[5] = 0.0
    else:
        if q_conditioned:
            d = sample_visible_length(rng, q_min, q_max, STRUCTURE_PERIOD_COEFF, "D", sampling_mode, margin)
        else:
            d = log_uniform(rng, schema.PARAM_RANGES["D"].low, schema.PARAM_RANGES["D"].high)
        p[4] = d
        p[5] = rng.uniform(schema.PARAM_RANGES["sigma_D_frac"].low, schema.PARAM_RANGES["sigma_D_frac"].high) * d
    return p


def sample_component_weights(k: int, rng: np.random.Generator) -> np.ndarray:
    alpha = float(rng.uniform(0.7, 2.0))
    return rng.dirichlet(np.full(k, alpha)).astype(np.float32)


def _components_unscaled_curve(q: np.ndarray, components: List[Dict]) -> np.ndarray:
    global_ones = {"BG": 0.0, "sigma_Res": 0.01, "nu_Res": 2.0, "int_Res": 0.0, "k": 1.0}
    return evaluate_clean(q, components, global_ones)


def sample_global_params(q: np.ndarray, components: List[Dict], rng: np.random.Generator) -> np.ndarray:
    i_mix = np.maximum(_components_unscaled_curve(q, components), 1e-30)
    k = log_uniform(rng, 1e-2, 1e6)
    bg_frac = log_uniform(rng, 1e-6, 1e-1)
    bg = max(bg_frac * float(np.median(k * i_mix)), 1e-18)
    sigma_res = log_uniform(rng, 0.002, 0.3)
    nu_res = float(rng.uniform(1.0, 8.0))
    if rng.random() < 0.25:
        int_res = 0.0
    else:
        int_res = max(log_uniform(rng, 1e-6, 5e-1) * float(np.max(i_mix)), 1e-18)
    return np.array([bg, sigma_res, nu_res, int_res, k], dtype=np.float32)


def random_missing_q_windows(q, I, sigma, I_clean, rng: np.random.Generator):
    n = len(q)
    keep = np.ones(n, dtype=bool)
    left = int(rng.uniform(0, 0.10) * n)
    right = int(rng.uniform(0, 0.10) * n)
    if left > 0:
        keep[:left] = False
    if right > 0:
        keep[n - right :] = False

    r = rng.random()
    n_windows = 0 if r < 0.50 else (1 if r < 0.85 else 2)
    for _ in range(n_windows):
        width = int(rng.uniform(0.05, 0.20) * n)
        if width <= 0 or n - width <= 2:
            continue
        start = int(rng.integers(max(left, 1), max(max(left + 1, n - right - width), 2)))
        keep[start : start + width] = False

    if keep.sum() < 64:
        keep[:] = True
    return q[keep], I[keep], sigma[keep], I_clean[keep]


def drop_noisy_floor_points(q, I, sigma, I_clean, intensity_floor: float = 1e-20):
    if intensity_floor <= 0.0:
        return q, I, sigma, I_clean, 0
    keep = (
        np.isfinite(q)
        & np.isfinite(I)
        & np.isfinite(sigma)
        & np.isfinite(I_clean)
        & (q > 0)
        & (I > intensity_floor)
        & (sigma > 0)
        & (I_clean > 0)
    )
    removed = int(np.sum(~keep))
    return q[keep], I[keep], sigma[keep], I_clean[keep], removed


def random_intensity_drop_gaps(
    I: np.ndarray,
    rng: np.random.Generator,
    min_regions: int = 1,
    max_regions: int = 3,
    min_width: int = 1,
    max_width: int = 10,
    drop_min: float = 0.01,
    drop_max: float = 0.70,
    max_fraction: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """Drop a few short intensity regions to simulate detector gaps."""
    I_aug = np.asarray(I, dtype=np.float64).copy()
    n = len(I_aug)
    gap_mask = np.zeros(n, dtype=bool)
    max_dropped = int(np.floor(max_fraction * n))
    if n == 0 or max_dropped < min_width:
        return I_aug.astype(np.asarray(I).dtype, copy=False), gap_mask

    max_regions_eff = min(max_regions, max(1, max_dropped // min_width))
    min_regions_eff = min(min_regions, max_regions_eff)
    n_regions = int(rng.integers(min_regions_eff, max_regions_eff + 1))

    for _ in range(n_regions):
        remaining = max_dropped - int(gap_mask.sum())
        width_max = min(max_width, remaining, n)
        if width_max < min_width:
            break
        width = int(rng.integers(min_width, width_max + 1))

        start = None
        region_unmasked = None
        for _attempt in range(20):
            candidate_start = int(rng.integers(0, n - width + 1))
            candidate_region = slice(candidate_start, candidate_start + width)
            candidate_unmasked = ~gap_mask[candidate_region]
            if np.any(candidate_unmasked):
                start = candidate_start
                region_unmasked = candidate_unmasked
                break
        if start is None or region_unmasked is None:
            break

        region_idx = np.arange(start, start + width)[region_unmasked]
        factor = float(rng.uniform(drop_min, drop_max))
        I_aug[region_idx] = np.maximum(I_aug[region_idx] * factor, 1e-30)
        gap_mask[region_idx] = True

    return I_aug.astype(np.asarray(I).dtype, copy=False), gap_mask


def preprocess_curve(q: np.ndarray, I: np.ndarray, sigma_arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    logq_norm = schema.normalize_logq(q)
    log_i = np.log(np.maximum(I, 1e-30))
    log_sigma = np.log(np.maximum(sigma_arr, 1e-30))
    global_features, offset, scale = schema.global_feature_from_curve(q, log_i)
    log_i_norm = (log_i - offset) / scale
    log_sigma_norm = (log_sigma - offset) / scale
    x = np.stack([logq_norm, log_i_norm, log_sigma_norm], axis=-1).astype(np.float32)
    return x, global_features


def pad_1d(arr: np.ndarray, max_points: int, fill: float = 0.0) -> np.ndarray:
    out = np.full((max_points,), fill, dtype=np.asarray(arr).dtype)
    n = min(len(arr), max_points)
    out[:n] = arr[:n]
    return out


def pad_2d(arr: np.ndarray, max_points: int, width: int, fill: float = 0.0) -> np.ndarray:
    out = np.full((max_points, width), fill, dtype=np.float32)
    n = min(len(arr), max_points)
    out[:n] = arr[:n]
    return out


def generate_sample(
    rng: np.random.Generator,
    max_points: int = schema.MAX_POINTS,
    poisson_scale_min: float = 200.0,
    poisson_scale_max: float = 3000.0,
    rel_noise_min: float = 0.001,
    rel_noise_max: float = 0.02,
    drop_noisy_floor: float = 1e-20,
    q_conditioned_sampling: bool = True,
    visible_fraction: float = 0.70,
    edge_fraction: float = 0.20,
    out_of_window_fraction: float = 0.10,
    k_values: np.ndarray | None = None,
    k_probs: np.ndarray | None = None,
    gap_drop_prob: float = 1.0,
    gap_drop_max_fraction: float = 0.05,
    max_attempts: int = 100,
) -> Dict[str, np.ndarray]:
    for _ in range(max_attempts):
        q = sample_q_grid(rng, max_points=max_points)
        q_min = float(np.min(q))
        q_max = float(np.max(q))
        sampling_mode = sample_sampling_mode(
            rng,
            q_conditioned_sampling=q_conditioned_sampling,
            visible_fraction=visible_fraction,
            edge_fraction=edge_fraction,
            out_of_window_fraction=out_of_window_fraction,
        )
        k_active = sample_number_of_components(rng, k_values=k_values, k_probs=k_probs)
        weights = sample_component_weights(k_active, rng)
        components = []
        slot_type = np.zeros(schema.MAX_SLOTS, dtype=np.int32)
        slot_exist = np.zeros(schema.MAX_SLOTS, dtype=np.float32)
        slot_params_phys = np.zeros((schema.MAX_SLOTS, schema.P_MAX), dtype=np.float32)
        slot_params_norm = np.zeros((schema.MAX_SLOTS, schema.P_MAX), dtype=np.float32)
        slot_param_mask = np.zeros((schema.MAX_SLOTS, schema.P_MAX), dtype=np.float32)
        slot_weight = np.zeros(schema.MAX_SLOTS, dtype=np.float32)

        for j in range(k_active):
            tid = sample_component_type(rng)
            params = sample_component_params(
                tid,
                rng,
                q_min=q_min,
                q_max=q_max,
                sampling_mode=sampling_mode,
            )
            slot_type[j] = tid
            slot_exist[j] = 1.0
            slot_params_phys[j] = params
            slot_params_norm[j] = schema.normalize_params(params, tid)
            slot_param_mask[j] = schema.effective_param_mask(tid, params)
            slot_weight[j] = weights[j]
            components.append(component_array_to_dict(tid, params, float(weights[j])))

        try:
            global_phys = sample_global_params(q, components, rng)
            I_clean = evaluate_clean(q, components, global_array_to_dict(global_phys))
        except Exception:
            continue

        if not np.all(np.isfinite(I_clean)) or np.all(I_clean <= 0):
            continue
        dynamic = np.nanmax(I_clean) / max(np.nanmin(I_clean), 1e-30)
        if not np.isfinite(dynamic) or dynamic < 1e-8:
            continue

        I_noisy, sigma_arr = add_noise(
            I_clean,
            rng,
            poisson_scale_min=poisson_scale_min,
            poisson_scale_max=poisson_scale_max,
            rel_noise_min=rel_noise_min,
            rel_noise_max=rel_noise_max,
        )
        q, I_noisy, sigma_arr, I_clean, _ = drop_noisy_floor_points(
            q,
            I_noisy,
            sigma_arr,
            I_clean,
            intensity_floor=drop_noisy_floor,
        )
        if len(q) < 64:
            continue
        q2, I_noisy2, sigma2, I_clean2 = random_missing_q_windows(q, I_noisy, sigma_arr, I_clean, rng)
        if len(q2) < 64:
            continue
        if gap_drop_prob > 0.0 and rng.random() < gap_drop_prob:
            I_noisy2, _ = random_intensity_drop_gaps(
                I_noisy2,
                rng,
                max_fraction=gap_drop_max_fraction,
            )
        x, global_features = preprocess_curve(q2, I_noisy2, sigma2)
        point_mask = np.zeros(max_points, dtype=bool)
        point_mask[: min(len(q2), max_points)] = True

        sample = {
            "x": pad_2d(x, max_points, 3),
            "point_mask": point_mask,
            "q": pad_1d(q2.astype(np.float32), max_points),
            "I_noisy": pad_1d(I_noisy2.astype(np.float32), max_points),
            "sigma": pad_1d(sigma2.astype(np.float32), max_points),
            "I_clean": pad_1d(I_clean2.astype(np.float32), max_points),
            "global_features": global_features.astype(np.float32),
            "slot_type": slot_type,
            "slot_exist": slot_exist,
            "slot_params_phys": slot_params_phys,
            "slot_params_norm": slot_params_norm,
            "slot_param_mask": slot_param_mask,
            "slot_weight": slot_weight,
            "global_params_phys": global_phys.astype(np.float32),
            "global_params_norm": schema.normalize_global(global_phys).astype(np.float32),
            "sampling_mode": np.array(sampling_mode, dtype=np.int32),
        }
        sample.update(constraints.augment_constraints(sample, rng))
        if all(np.all(np.isfinite(v)) for v in sample.values() if np.asarray(v).dtype.kind in "fc"):
            return sample
    raise RuntimeError(f"Could not generate a valid sample after {max_attempts} attempts.")


def stack_samples(samples: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    keys = samples[0].keys()
    return {k: np.stack([s[k] for s in samples], axis=0) for k in keys}
