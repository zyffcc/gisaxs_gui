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
        # Per slot: [D absent allowed, D present allowed].
        "d_allowed": np.ones((schema.MAX_SLOTS, 2), dtype=np.float32),
        "d_spacing_rule": np.eye(schema.NUM_D_RULES, dtype=np.float32)[schema.D_RULE_FREE],
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
        cons = unconstrained()
        return _attach_d_constraints(cons, sample, rng)
    if mode < 0.65:
        cons = fixed_components(sample["slot_type"], sample["slot_exist"])
        return _attach_d_constraints(cons, sample, rng)

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
        return _attach_d_constraints(cons, sample, rng)

    active_slots = np.where(sample["slot_exist"] > 0.5)[0]
    if len(active_slots) == 0:
        return _attach_d_constraints(cons, sample, rng)
    j = int(rng.choice(active_slots))
    t = int(sample["slot_type"][j])
    valid = np.where(sample["slot_param_mask"][j] > 0.5)[0]
    if len(valid) == 0:
        return _attach_d_constraints(cons, sample, rng)
    pidx = int(rng.choice(valid))
    true_val = float(sample["slot_params_norm"][j, pidx])
    width = float(rng.uniform(0.04, 0.18))
    center = true_val
    low = np.clip(center - width, 0.0, 1.0)
    high = np.clip(center + width, 0.0, 1.0)
    cons["param_low_norm"][j, t, pidx] = low
    cons["param_high_norm"][j, t, pidx] = high
    cons["param_range_mask"][j, t, pidx] = 1.0
    return _attach_d_constraints(cons, sample, rng)


def _attach_d_constraints(cons: Dict[str, np.ndarray], sample: Dict[str, np.ndarray], rng: np.random.Generator):
    """Attach a valid relational D rule and occasionally a hard presence choice."""
    cons["d_spacing_rule"] = np.asarray(sample["d_spacing_rule"], dtype=np.float32).copy()
    if rng.random() < 0.30:
        for j in np.where(sample["slot_exist"] > 0.5)[0]:
            present = sample["slot_param_mask"][j, 4] > 0.5
            cons["d_allowed"][j] = (0.0, 1.0) if present else (1.0, 0.0)
    return cons


def from_json_dict(config: Optional[dict]) -> Dict[str, np.ndarray]:
    cons = unconstrained()
    if not config or config.get("mode", "free") == "free":
        component_names = config.get("components") if config else None
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

    d_config = (config or {}).get("d_constraint", {})
    rule_name = d_config.get("spacing_rule", "free")
    if rule_name not in schema.NAME_TO_D_RULE:
        raise ValueError(f"Unknown D spacing rule {rule_name!r}; expected one of {sorted(schema.NAME_TO_D_RULE)}")
    cons["d_spacing_rule"][:] = 0.0
    cons["d_spacing_rule"][schema.NAME_TO_D_RULE[rule_name]] = 1.0

    presence_map = {
        "optional": (1.0, 1.0),
        "absent": (1.0, 0.0),
        "required": (0.0, 1.0),
    }
    presence = d_config.get("presence", "optional")
    if presence not in presence_map:
        raise ValueError(f"Unknown D presence {presence!r}; expected optional, absent, or required")
    cons["d_allowed"][:] = presence_map[presence]
    for slot_key, slot_presence in d_config.get("slot_presence", {}).items():
        j = int(slot_key.split("_")[-1])
        if not 0 <= j < schema.MAX_SLOTS or slot_presence not in presence_map:
            raise ValueError(f"Invalid D slot presence constraint: {slot_key}={slot_presence!r}")
        cons["d_allowed"][j] = presence_map[slot_presence]
    return cons
