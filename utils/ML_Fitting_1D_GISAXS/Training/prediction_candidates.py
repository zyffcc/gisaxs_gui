"""Focused helpers for TOP-K prediction: prediction candidates."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from TrainSetBuild import schema
from TrainSetBuild.physics_adapter import component_array_to_dict
from TrainSetBuild.sampling import D_HARD_CORE_MARGIN, d_spacing_threshold

from Training.prediction_preprocessing import effective_range, sigmoid_stable, softmax


def sorted_components(components):
    return sorted(
        components,
        key=lambda c: (
            int(c["type_id"]),
            float(np.asarray(c["params_phys"])[0]),
            float(c["weight"]),
        ),
    )


def candidate_mode_vector(item):
    """Fixed-scale vector for clustering parameter modes within one type combination."""
    values = []
    comps = sorted(
        item["components"],
        key=lambda c: (
            int(c["type_id"]),
            float(np.asarray(c["params_phys"])[0]),
            float(c["weight"]),
        ),
    )
    for comp in comps:
        tid = int(comp["type_id"])
        params = np.asarray(comp["params_phys"], dtype=np.float64)
        effective_mask = schema.effective_param_mask(tid, params)
        for pidx, name in enumerate(schema.PARAM_NAMES):
            values.append(
                schema.normalize_value(float(params[pidx]), schema.PARAM_NORM_RANGES[name])
                if effective_mask[pidx] > 0.5
                else 0.0
            )
        values.append(float(effective_mask[4] > 0.5))
        values.append(float(comp["weight"]))
    values.extend(
        schema.normalize_global(np.asarray(item["global_phys"], dtype=np.float64)).tolist()
    )
    return np.asarray(values, dtype=np.float64)


def enforce_size_distribution_constraints(params_phys, type_id):
    """Project widths into the physical domain used by training.

    Vertical-cylinder sigma_R is fractional.  All other active widths are
    absolute and may not exceed 90% of their corresponding mean.  The trained
    no-D representation is exactly D=sigma_D=0.
    """
    params = np.asarray(params_phys, dtype=np.float64).copy()
    tiny = np.finfo(np.float32).eps
    if int(type_id) == schema.TYPE_VERTICAL_CYLINDER:
        params[1] = np.clip(params[1], tiny, 0.9)
    else:
        params[1] = np.clip(params[1], tiny, 0.9 * max(params[0], tiny))
    if int(type_id) == schema.TYPE_CYLINDER:
        params[3] = np.clip(params[3], tiny, 0.9 * max(params[2], tiny))
    if params[4] <= 0.0:
        params[4:6] = 0.0
    else:
        params[5] = np.clip(params[5], tiny, 0.9 * params[4])
    return params


def posterior_stats(items):
    if not items:
        return {}
    best_sorted = sorted_components(items[0]["components"])
    comp_stats = []
    for comp_idx, comp in enumerate(best_sorted):
        tid = int(comp["type_id"])
        names = (
            ["R", "sigma_R", "D", "sigma_D"] if tid != schema.TYPE_CYLINDER else schema.PARAM_NAMES
        )
        idxs = [0, 1, 4, 5] if tid != schema.TYPE_CYLINDER else list(range(schema.P_MAX))
        same_position = [
            sorted_components(item["components"])[comp_idx]
            for item in items
            if len(item["components"]) > comp_idx
        ]
        stats = {
            "type": comp["type_name"],
            "weight": quantile_summary([c["weight"] for c in same_position]),
            "params": {},
        }
        for name, pidx in zip(names, idxs):
            stats["params"][name] = quantile_summary(
                [np.asarray(c["params_phys"])[pidx] for c in same_position]
            )
        comp_stats.append(stats)
    global_stats = {
        name: quantile_summary([item["global_phys"][i] for item in items])
        for i, name in enumerate(schema.GLOBAL_PARAM_NAMES)
    }
    return {"components": comp_stats, "global_params": global_stats}


def sample_candidate(
    pred,
    cons,
    rng,
    sampling_std=0.03,
    use_predicted_logstd=False,
    exact_nonempty=None,
    return_reason=False,
    d_hard_core_margin=D_HARD_CORE_MARGIN,
):
    exist_prob = sigmoid_stable(pred["exist_logit"][0])
    type_logits = pred["type_logits"][0]
    param_mu = pred["param_mu_norm"][0]
    param_logstd = np.clip(pred["param_logstd_raw"][0], -5.0, 1.0)
    weight_logits = pred["weight_logit"][0]
    global_mu = pred["global_mu_norm"][0]
    global_logstd = np.clip(pred["global_logstd_raw"][0], -5.0, 1.0)

    active = []
    for j in range(schema.MAX_SLOTS):
        force = float(cons["force_exist"][j])
        if force == 1.0:
            exists = True
        elif force == 0.0:
            exists = False
        else:
            exists = bool(rng.random() < exist_prob[j])
        if not exists:
            continue
        probs = softmax(type_logits[j])
        probs[schema.TYPE_EMPTY] = 0.0
        probs = probs / np.maximum(probs.sum(), 1e-300)
        if not np.isfinite(probs).all() or probs.sum() <= 0:
            continue
        tid = int(rng.choice(np.arange(schema.NUM_TYPES), p=probs))
        if tid == schema.TYPE_EMPTY:
            continue
        mu = param_mu[j, tid]
        std = (
            np.exp(param_logstd[j, tid]) if use_predicted_logstd else np.full_like(mu, sampling_std)
        )
        low_eff, high_eff = effective_range(
            cons["param_low_norm"][j, tid],
            cons["param_high_norm"][j, tid],
            cons["param_range_mask"][j, tid],
        )
        params_norm = np.clip(rng.normal(mu, std), low_eff, high_eff)
        params_phys = schema.denormalize_params(params_norm, tid)
        params_phys = schema.apply_type_param_mask(params_phys, tid)
        params_phys = enforce_size_distribution_constraints(params_phys, tid)
        active.append({"slot": j, "type_id": tid, "params_phys": params_phys})

    if not active:
        return (None, "empty") if return_reason else None
    if exact_nonempty is not None:
        target = int(exact_nonempty)
        if len(active) != target:
            return (None, "exact_nonempty") if return_reason else None
    if not enforce_d_constraints(
        active, pred, cons, rng=rng, d_hard_core_margin=d_hard_core_margin
    ):
        return (None, "d_spacing") if return_reason else None
    # Weight softmax is evaluated only for slots that sampled a non-empty component.
    active_logits = np.array([weight_logits[a["slot"]] for a in active], dtype=np.float64)
    weights = softmax(active_logits)
    components = [
        component_array_to_dict(a["type_id"], a["params_phys"], float(weights[i]))
        for i, a in enumerate(active)
    ]
    global_std = (
        np.exp(global_logstd) if use_predicted_logstd else np.full_like(global_mu, sampling_std)
    )
    global_low_eff, global_high_eff = effective_range(
        cons["global_low_norm"], cons["global_high_norm"], cons["global_range_mask"]
    )
    global_norm = np.clip(rng.normal(global_mu, global_std), global_low_eff, global_high_eff)
    global_phys = schema.denormalize_global_with_optional_zero(global_norm)
    candidate = (components, global_phys)
    return (candidate, None) if return_reason else candidate


def enforce_d_constraints(active, pred, cons, rng=None, d_hard_core_margin=D_HARD_CORE_MARGIN):
    """Choose optional D presence and hard-project active D above the requested spacing bound."""
    d_logits = pred.get("d_present_logit")
    for item in active:
        slot = item["slot"]
        absent_allowed, present_allowed = np.asarray(cons["d_allowed"][slot], dtype=float)
        if absent_allowed < 0.5:
            present = True
        elif present_allowed < 0.5:
            present = False
        elif d_logits is None:
            present = bool(item["params_phys"][4] > 0.0)
        else:
            probability = float(sigmoid_stable(d_logits[0, slot]))
            present = probability >= 0.5 if rng is None else bool(rng.random() < probability)
        item["d_present"] = present
        if not present:
            item["params_phys"][4:6] = 0.0

    types = np.array([item["type_id"] for item in active], dtype=np.int32)
    params = np.stack([item["params_phys"] for item in active], axis=0)
    exists = np.ones(len(active), dtype=np.float32)
    rule_id = int(np.argmax(cons["d_spacing_rule"]))
    threshold = d_spacing_threshold(types, params, exists, rule_id)
    required_d = threshold * float(d_hard_core_margin)
    for item in active:
        if not item["d_present"]:
            continue
        slot, tid = item["slot"], item["type_id"]
        _, high_eff = effective_range(
            cons["param_low_norm"][slot, tid],
            cons["param_high_norm"][slot, tid],
            cons["param_range_mask"][slot, tid],
        )
        d_high = schema.denormalize_value(float(high_eff[4]), schema.PARAM_NORM_RANGES["D"])
        if required_d >= d_high:
            return False
        if item["params_phys"][4] <= required_d:
            item["params_phys"][4] = np.nextafter(required_d, np.inf)
    return True


def mean_candidate(
    pred,
    cons,
    exact_nonempty=None,
    return_reason=False,
    d_hard_core_margin=D_HARD_CORE_MARGIN,
):
    exist_prob = sigmoid_stable(pred["exist_logit"][0])
    type_logits = pred["type_logits"][0]
    param_mu = pred["param_mu_norm"][0]
    weight_logits = pred["weight_logit"][0]
    global_mu = pred["global_mu_norm"][0]

    active = []
    for j in range(schema.MAX_SLOTS):
        force = float(cons["force_exist"][j])
        if force == 1.0:
            exists = True
        elif force == 0.0:
            exists = False
        else:
            exists = bool(exist_prob[j] >= 0.5)
        if not exists:
            continue

        scores = np.asarray(type_logits[j], dtype=np.float64).copy()
        scores[schema.TYPE_EMPTY] = -np.inf
        tid = int(np.argmax(scores))
        if tid == schema.TYPE_EMPTY or not np.isfinite(scores[tid]):
            continue

        params_norm = np.asarray(param_mu[j, tid], dtype=np.float64)
        params_phys = schema.denormalize_params(params_norm, tid)
        params_phys = schema.apply_type_param_mask(params_phys, tid)
        params_phys = enforce_size_distribution_constraints(params_phys, tid)
        active.append({"slot": j, "type_id": tid, "params_phys": params_phys})

    if not active:
        return (None, "empty") if return_reason else None
    if exact_nonempty is not None:
        target = int(exact_nonempty)
        if len(active) != target:
            return (None, "exact_nonempty") if return_reason else None

    if not enforce_d_constraints(
        active, pred, cons, rng=None, d_hard_core_margin=d_hard_core_margin
    ):
        return (None, "d_spacing") if return_reason else None
    active_logits = np.array([weight_logits[a["slot"]] for a in active], dtype=np.float64)
    weights = softmax(active_logits)
    components = [
        component_array_to_dict(a["type_id"], a["params_phys"], float(weights[i]))
        for i, a in enumerate(active)
    ]
    global_phys = schema.denormalize_global_with_optional_zero(global_mu)
    candidate = (components, global_phys)
    return (candidate, None) if return_reason else candidate


def combination_key(components):
    return "+".join(sorted([c["type_name"] for c in components]))


def quantile_summary(values):
    arr = np.asarray(values, dtype=np.float64)
    return {
        "median": float(np.median(arr)),
        "p16": float(np.percentile(arr, 16)),
        "p84": float(np.percentile(arr, 84)),
    }


def cluster_parameter_modes(items, radius=0.10):
    """Greedy score-ordered clustering; each returned mode may enter TOP-K separately."""
    radius = max(float(radius), 1e-6)
    modes = []
    for item in sorted(items, key=lambda it: it["score"]):
        vector = candidate_mode_vector(item)
        best_mode = None
        best_distance = np.inf
        for mode in modes:
            distance = float(np.sqrt(np.mean(np.square(vector - mode["centroid"]))))
            if distance <= radius and distance < best_distance:
                best_mode = mode
                best_distance = distance
        if best_mode is None:
            modes.append({"centroid": vector.copy(), "items": [item]})
        else:
            best_mode["items"].append(item)
            n = len(best_mode["items"])
            best_mode["centroid"] += (vector - best_mode["centroid"]) / float(n)
    return [mode["items"] for mode in modes]


def component_params_json(comp):
    tid = int(comp["type_id"])
    p = np.asarray(comp["params_phys"], dtype=float)
    names = ["R", "sigma_R", "D", "sigma_D"] if tid != schema.TYPE_CYLINDER else schema.PARAM_NAMES
    idxs = [0, 1, 4, 5] if tid != schema.TYPE_CYLINDER else list(range(schema.P_MAX))
    return {name: float(p[i]) for name, i in zip(names, idxs)}
