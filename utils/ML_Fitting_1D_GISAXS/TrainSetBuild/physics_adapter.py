"""Adapter from slot dictionaries to the existing NumPy GISAXS model."""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np

try:
    from utils.fitting import make_mixed_model, params_template
except ImportError:  # pragma: no cover - useful when imported from package context
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from utils.fitting import make_mixed_model, params_template

from TrainSetBuild.schema import (
    GLOBAL_PARAM_NAMES,
    PARAM_NAMES,
    TYPE_EMPTY,
    TYPE_NAMES,
    TYPE_CYLINDER,
    TYPE_SPHERE,
    TYPE_VERTICAL_CYLINDER,
)


def components_to_spec_params(components: Iterable[Dict], global_params: Dict[str, float]) -> Tuple[List[str], List[float]]:
    spec: List[str] = []
    params: List[float] = []
    for comp in components:
        type_id = int(comp["type_id"])
        if type_id == TYPE_EMPTY:
            continue
        type_name = TYPE_NAMES[type_id]
        p = np.asarray(comp["params_phys"], dtype=float)
        weight = float(comp.get("weight", 1.0))
        spec.append(type_name)
        if type_id == TYPE_SPHERE:
            params.extend([weight, p[0], p[1], p[4], p[5]])
        elif type_id == TYPE_CYLINDER:
            params.extend([weight, p[0], p[1], p[2], p[3], p[4], p[5]])
        elif type_id == TYPE_VERTICAL_CYLINDER:
            params.extend([weight, p[0], p[1], p[4], p[5]])
        else:
            raise ValueError(f"Unsupported type_id: {type_id}")
    if not spec:
        raise ValueError("At least one non-empty component is required for the physics model.")
    params.extend([float(global_params[name]) for name in GLOBAL_PARAM_NAMES])
    return spec, params


def evaluate_clean(q: np.ndarray, components: Iterable[Dict], global_params: Dict[str, float]) -> np.ndarray:
    spec, params = components_to_spec_params(components, global_params)
    model = make_mixed_model(spec)
    expected = params_template(spec)
    if len(params) != len(expected):
        raise ValueError(f"Internal parameter mismatch: expected {expected}, got {len(params)} values")
    out = model(np.asarray(q, dtype=float), *params)
    out = np.asarray(out, dtype=np.float64)
    return np.clip(out, 1e-30, np.inf)


def global_array_to_dict(global_params_phys: np.ndarray) -> Dict[str, float]:
    return {name: float(global_params_phys[i]) for i, name in enumerate(GLOBAL_PARAM_NAMES)}


def component_array_to_dict(type_id: int, params_phys: np.ndarray, weight: float) -> Dict:
    return {
        "type_id": int(type_id),
        "type_name": TYPE_NAMES[int(type_id)],
        "params_phys": np.asarray(params_phys, dtype=np.float64),
        "weight": float(weight),
        "param_names": PARAM_NAMES,
    }

