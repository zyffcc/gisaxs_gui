"""Constraint tensor construction for training and inference."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from TrainSetBuild import schema


def unconstrained() -> Dict[str, np.ndarray]:
    return {
        "type_allowed": np.ones((schema.MAX_SLOTS, schema.NUM_TYPES), dtype=np.float32),
        "param_low_norm": np.zeros((schema.MAX_SLOTS, schema.NUM_TYPES, schema.P_MAX), dtype=np.float32),
        "param_high_norm": np.ones((schema.MAX_SLOTS, schema.NUM_TYPES, schema.P_MAX), dtype=np.float32),
        "param_range_mask": np.zeros((schema.MAX_SLOTS, schema.NUM_TYPES, schema.P_MAX), dtype=np.float32),
        "force_exist": np.full((schema.MAX_SLOTS,), -1.0, dtype=np.float32),
        "global_low_norm": np.zeros((schema.G_MAX,), dtype=np.float32),
        "global_high_norm": np.ones((schema.G_MAX,), dtype=np.float32),
        "global_range_mask": np.zeros((schema.G_MAX,), dtype=np.float32),
    }


def fixed_components(slot_type: np.ndarray, slot_exist: np.ndarray) -> Dict[str, np.ndarray]:
    cons = unconstrained()
    cons["type_allowed"][:] = 0.0
    for j in range(schema.MAX_SLOTS):
        t = int(slot_type[j])
        cons["type_allowed"][j, t] = 1.0
        cons["force_exist"][j] = 1.0 if slot_exist[j] > 0 else 0.0
    return cons


def augment_constraints(sample: Dict[str, np.ndarray], rng: np.random.Generator) -> Dict[str, np.ndarray]:
    mode = float(rng.random())
    if mode < 0.40:
        return unconstrained()
    if mode < 0.65:
        return fixed_components(sample["slot_type"], sample["slot_exist"])

    cons = unconstrained()
    if mode < 0.85:
        for j in range(schema.MAX_SLOTS):
            true_t = int(sample["slot_type"][j])
            if true_t == schema.TYPE_EMPTY:
                continue
            allowed = {true_t, schema.TYPE_EMPTY}
            for tid in (schema.TYPE_SPHERE, schema.TYPE_CYLINDER, schema.TYPE_VERTICAL_CYLINDER):
                if rng.random() < 0.35:
                    allowed.add(tid)
            cons["type_allowed"][j, :] = 0.0
            cons["type_allowed"][j, list(allowed)] = 1.0
        return cons

    active_slots = np.where(sample["slot_exist"] > 0.5)[0]
    if len(active_slots) == 0:
        return cons
    j = int(rng.choice(active_slots))
    t = int(sample["slot_type"][j])
    valid = np.where(sample["slot_param_mask"][j] > 0.5)[0]
    if len(valid) == 0:
        return cons
    pidx = int(rng.choice(valid))
    true_val = float(sample["slot_params_norm"][j, pidx])
    width = float(rng.uniform(0.04, 0.18))
    center = true_val
    low = np.clip(center - width, 0.0, 1.0)
    high = np.clip(center + width, 0.0, 1.0)
    cons["param_low_norm"][j, t, pidx] = low
    cons["param_high_norm"][j, t, pidx] = high
    cons["param_range_mask"][j, t, pidx] = 1.0
    return cons


def from_json_dict(config: Optional[dict]) -> Dict[str, np.ndarray]:
    cons = unconstrained()
    if not config or config.get("mode", "free") == "free":
        component_names = config.get("components") if config else None
        if not component_names:
            return cons
    else:
        component_names = config.get("components", [])

    if component_names:
        cons["type_allowed"][:] = 0.0
        cons["force_exist"][:] = 0.0
        for j, name in enumerate(component_names[: schema.MAX_SLOTS]):
            tid = schema.NAME_TO_TYPE[name]
            cons["type_allowed"][j, tid] = 1.0
            cons["force_exist"][j] = 1.0
        for j in range(len(component_names), schema.MAX_SLOTS):
            cons["type_allowed"][j, schema.TYPE_EMPTY] = 1.0

    parameter_ranges = (config or {}).get("parameter_ranges", {})
    for slot_key, ranges in parameter_ranges.items():
        j = int(slot_key.split("_")[-1])
        for pname, bounds in ranges.items():
            pidx = schema.PARAM_NAMES.index(pname)
            for tid in range(schema.NUM_TYPES):
                if tid == schema.TYPE_EMPTY:
                    continue
                spec = schema.PARAM_NORM_RANGES[pname]
                cons["param_low_norm"][j, tid, pidx] = schema.normalize_value(bounds[0], spec)
                cons["param_high_norm"][j, tid, pidx] = schema.normalize_value(bounds[1], spec)
                cons["param_range_mask"][j, tid, pidx] = 1.0

    type_parameter_ranges = (config or {}).get("type_parameter_ranges", {})
    for type_name, ranges in type_parameter_ranges.items():
        tid = schema.NAME_TO_TYPE[type_name]
        for pname, bounds in ranges.items():
            pidx = schema.PARAM_NAMES.index(pname)
            spec = schema.PARAM_NORM_RANGES[pname]
            cons["param_low_norm"][:, tid, pidx] = schema.normalize_value(bounds[0], spec)
            cons["param_high_norm"][:, tid, pidx] = schema.normalize_value(bounds[1], spec)
            cons["param_range_mask"][:, tid, pidx] = 1.0

    for gname, bounds in (config or {}).get("global_ranges", {}).items():
        gidx = schema.GLOBAL_PARAM_NAMES.index(gname)
        spec = schema.GLOBAL_NORM_RANGES[gname]
        cons["global_low_norm"][gidx] = schema.normalize_value(bounds[0], spec)
        cons["global_high_norm"][gidx] = schema.normalize_value(bounds[1], spec)
        cons["global_range_mask"][gidx] = 1.0
    return cons
