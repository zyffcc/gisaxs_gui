"""AI candidate 的 verification、ranking 和参数映射。"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .physical_constraints import ConstraintSet


@dataclass(frozen=True)
class CandidateComponentParameters:
    shape: str
    weight: float
    parameters: dict[str, float]


@dataclass(frozen=True)
class CandidateParameterMapping:
    components: tuple[CandidateComponentParameters, ...]
    global_parameters: dict[str, float]


def verify_and_rank_candidates(
    rows: Sequence[Mapping[str, Any]],
    constraint_options: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], ...]:
    constraints = ConstraintSet.from_dict(constraint_options)
    reviewed: list[tuple[int, dict[str, Any]]] = []
    for index, source in enumerate(rows):
        row = deepcopy(dict(source))
        violations = constraints.validate_components(row.get("components") or [])
        row["constraint_violations"] = [item.message for item in violations]
        reviewed.append((index, row))

    def rank_key(item):
        index, row = item
        try:
            explicit_rank = int(row.get("rank"))
        except (TypeError, ValueError):
            explicit_rank = 10**9
        try:
            score = float(row.get("best_log_rmse"))
        except (TypeError, ValueError):
            score = float("inf")
        try:
            chi2 = float(row.get("best_chi2_weighted"))
        except (TypeError, ValueError):
            chi2 = float("inf")
        return explicit_rank, score, chi2, index

    reviewed.sort(key=rank_key)
    return tuple(row for _index, row in reviewed)


def candidate_parameter_mapping(row: Mapping[str, Any]) -> CandidateParameterMapping:
    components = row.get("components") or []
    if not isinstance(components, list) or not components:
        raise ValueError("Selected candidate has no component parameters")
    shape_map = {
        "sphere": "Sphere",
        "cylinder": "Cylinder",
        "vertical_cylinder": "Vertical Cylinder",
        "vertical cylinder": "Vertical Cylinder",
        "verticalcylinder": "Vertical Cylinder",
    }
    parameter_map = {
        "R": "radius",
        "sigma_R": "sigma_radius",
        "h": "height",
        "sigma_h": "sigma_height",
        "D": "diameter",
        "sigma_D": "sigma_diameter",
    }
    mapped_components = []
    for component in components:
        raw_type = str(component.get("type", "")).strip()
        shape = shape_map.get(raw_type.lower().replace("-", "_"), raw_type or "Sphere")
        values = {"intensity": float(component.get("weight", 1.0))}
        source_parameters = component.get("params") or {}
        for source_key, target_key in parameter_map.items():
            if source_key in source_parameters:
                values[target_key] = float(source_parameters[source_key])
        mapped_components.append(
            CandidateComponentParameters(
                shape=shape,
                weight=values["intensity"],
                parameters=values,
            )
        )

    global_map = {
        "background": "background",
        "bg": "background",
        "sigma_res": "sigma_res",
        "nu_res": "nu_res",
        "int_res": "int_res",
        "k": "k_value",
        "k_value": "k_value",
    }
    mapped_globals = {}
    for key, value in (row.get("global_params") or {}).items():
        target = global_map.get(str(key).lower())
        if target is not None:
            mapped_globals[target] = float(value)
    return CandidateParameterMapping(tuple(mapped_components), mapped_globals)
