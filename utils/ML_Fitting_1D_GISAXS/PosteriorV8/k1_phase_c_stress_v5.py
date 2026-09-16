"""Truth-independent range and acquisition forcing for K1 Phase-C parents."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from typing import Mapping

from .amplitude_query_sampling_v5 import V5_AMPLITUDE_RANGE_REGIMES
from .bounds_query_v5 import AXIS_RANGE_PLACEMENTS, AXIS_RANGE_REGIMES
from .contract import CYLINDER
from .k1_branch_forcing_v5 import V5K1ForcedSobolCoordinates
from .k1_phase_c_contract_v5 import (
    K1_PHASE_C_OBSERVATION_EFFECTS,
    K1_PHASE_C_OBSERVATION_STRESS_STRATA,
    K1_PHASE_C_RANGE_STRESS_STRATA,
)
from .simulation import GridProvenance, NoiseProvenance
from .sobol_recipe_coordinates_v5 import (
    V5_SOBOL_RECIPE_COORDINATE_INDEX,
    validate_v5_sobol_recipe_coordinates,
)


V5_K1_PHASE_C_STRESS_SCHEMA = "gisaxs.posterior_v8.k1_phase_c_parent_stress/v1"
V5_K1_PHASE_C_STRESS_VERSION = (
    "posterior_v8_v5_2_truth_independent_axis_and_one_factor_acquisition_stress_v1"
)
V5_K1_PHASE_C_OBSERVATION_DESIGN_SCHEMA = (
    "gisaxs.posterior_v8.k1_phase_c_observation_design/v1"
)
V5_K1_PHASE_C_OBSERVATION_DESIGN_VERSION = (
    "posterior_v8_v5_2_precurve_one_factor_observation_design_v1"
)

_BASE_GRID = GridProvenance(kind="geometric", q_min=1.0e-3, q_max=5.0, n_points=256)
_CLEAN_NOISE = NoiseProvenance(
    poisson_count_scale=None,
    relative_sigma=0.0,
    sigma_floor_fraction=1.0e-6,
)
_NOISY = NoiseProvenance(
    poisson_count_scale=2.0e3,
    relative_sigma=0.03,
    sigma_floor_fraction=1.0e-6,
)


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _bucket_center(value: str, values: tuple[str, ...]) -> float:
    return (values.index(value) + 0.5) / len(values)


def _geometry_prefixes(forcing: V5K1ForcedSobolCoordinates) -> tuple[str, ...]:
    branch = forcing.branch
    values = ["geometry.slot_1.R", "geometry.slot_1.sigma_R_fraction"]
    if branch.shape == CYLINDER:
        values.extend(("geometry.slot_1.h", "geometry.slot_1.sigma_h_fraction"))
    if branch.d_present:
        values.extend(("geometry.slot_1.D", "geometry.slot_1.sigma_D_fraction"))
    if branch.resolution_present:
        values.extend(("geometry.resolution.sigma_res", "geometry.resolution.nu_res"))
    return tuple(values)


def _amplitude_prefixes(forcing: V5K1ForcedSobolCoordinates) -> tuple[str, ...]:
    values = ["amplitude.query.BG", "amplitude.query.k", "amplitude.query.Int_1"]
    if forcing.branch.resolution_present:
        values.append("amplitude.query.int_Res")
    return tuple(values)


def _stress_assignment(
    range_stress_stratum: str,
    geometry_count: int,
    amplitude_count: int,
) -> tuple[tuple[tuple[str, str], ...], tuple[str, ...]]:
    if range_stress_stratum == "full":
        geometry = (("full", "interior"),) * geometry_count
        amplitude = ("full",) * amplitude_count
    elif range_stress_stratum == "narrow":
        geometry = (("narrow", "interior"),) * geometry_count
        amplitude = ("narrow",) * amplitude_count
    elif range_stress_stratum == "fixed":
        geometry = (("fixed", "interior"),) * geometry_count
        amplitude = ("fixed",) * amplitude_count
    elif range_stress_stratum == "edge":
        geometry = tuple(
            ("narrow", "edge_low" if index % 2 == 0 else "edge_high")
            for index in range(geometry_count)
        )
        amplitude = tuple(
            "edge_low" if index % 2 == 0 else "edge_high"
            for index in range(amplitude_count)
        )
    elif range_stress_stratum == "mixed":
        geometry_choices = (("full", "interior"), ("narrow", "interior"), ("fixed", "interior"))
        amplitude_choices = ("full", "narrow", "fixed", "edge_low", "edge_high")
        geometry = tuple(
            geometry_choices[index % len(geometry_choices)]
            for index in range(geometry_count)
        )
        amplitude = tuple(
            amplitude_choices[(index + 1) % len(amplitude_choices)]
            for index in range(amplitude_count)
        )
    else:  # pragma: no cover - guarded by the public entry point
        raise ValueError("unsupported K1 Phase-C range stress stratum")
    return geometry, amplitude


@dataclass(frozen=True)
class V5K1PhaseCStressCoordinates:
    branch_forcing_sha256: str
    range_stress_stratum: str
    branch_forced_coordinates: tuple[float, ...]
    stressed_coordinates: tuple[float, ...]
    canonical_json: str
    sha256: str

    def audit_payload(self) -> dict[str, object]:
        return json.loads(self.canonical_json)


def force_v5_k1_phase_c_range_stress(
    forcing: V5K1ForcedSobolCoordinates,
    *,
    range_stress_stratum: str,
) -> V5K1PhaseCStressCoordinates:
    """Force active range-decision coordinates without reading truth or curves."""

    if not isinstance(forcing, V5K1ForcedSobolCoordinates):
        raise TypeError("forcing must be V5K1ForcedSobolCoordinates")
    if range_stress_stratum not in K1_PHASE_C_RANGE_STRESS_STRATA:
        raise ValueError("range_stress_stratum is not frozen by Phase-C")
    geometry_prefixes = _geometry_prefixes(forcing)
    amplitude_prefixes = _amplitude_prefixes(forcing)
    geometry, amplitude = _stress_assignment(
        range_stress_stratum,
        len(geometry_prefixes),
        len(amplitude_prefixes),
    )
    values = list(forcing.forced_coordinates)
    replacements: list[dict[str, object]] = []

    def replace(name: str, value: float, axis_key: str) -> None:
        index = V5_SOBOL_RECIPE_COORDINATE_INDEX[name]
        replacements.append(
            {
                "axis_key": axis_key,
                "coordinate_name": name,
                "coordinate_index": index,
                "original_value": values[index],
                "forced_value": value,
            }
        )
        values[index] = value

    for prefix, (regime, placement) in zip(geometry_prefixes, geometry, strict=True):
        replace(
            f"{prefix}.regime",
            _bucket_center(regime, tuple(AXIS_RANGE_REGIMES)),
            prefix,
        )
        replace(
            f"{prefix}.placement",
            _bucket_center(placement, tuple(AXIS_RANGE_PLACEMENTS)),
            prefix,
        )
    for prefix, regime in zip(amplitude_prefixes, amplitude, strict=True):
        replace(
            f"{prefix}.regime",
            _bucket_center(regime, tuple(V5_AMPLITUDE_RANGE_REGIMES)),
            prefix,
        )
    stressed = validate_v5_sobol_recipe_coordinates(values)
    core = {
        "schema": V5_K1_PHASE_C_STRESS_SCHEMA,
        "version": V5_K1_PHASE_C_STRESS_VERSION,
        "branch_forcing_sha256": forcing.sha256,
        "range_stress_stratum": range_stress_stratum,
        "generation_order": "sobol_then_branch_forcing_then_range_stress_then_truth",
        "geometry_axis_assignments": [
            {"axis_key": prefix, "regime": regime, "placement": placement}
            for prefix, (regime, placement) in zip(geometry_prefixes, geometry, strict=True)
        ],
        "amplitude_axis_assignments": [
            {"axis_key": prefix, "regime": regime}
            for prefix, regime in zip(amplitude_prefixes, amplitude, strict=True)
        ],
        "replacements": replacements,
        "branch_forced_coordinates_sha256": sha256(
            _canonical_json(list(forcing.forced_coordinates)).encode()
        ).hexdigest(),
        "stressed_coordinates_sha256": sha256(
            _canonical_json(list(stressed)).encode()
        ).hexdigest(),
    }
    encoded = _canonical_json(core)
    return V5K1PhaseCStressCoordinates(
        branch_forcing_sha256=forcing.sha256,
        range_stress_stratum=range_stress_stratum,
        branch_forced_coordinates=forcing.forced_coordinates,
        stressed_coordinates=stressed,
        canonical_json=encoded,
        sha256=sha256(encoded.encode()).hexdigest(),
    )


def v5_k1_phase_c_observation_contract() -> dict[str, object]:
    core = {
        "schema": V5_K1_PHASE_C_OBSERVATION_DESIGN_SCHEMA,
        "version": V5_K1_PHASE_C_OBSERVATION_DESIGN_VERSION,
        "base_grid": asdict(_BASE_GRID),
        "base_noise": asdict(_CLEAN_NOISE),
        "base_point_keep_probability": 1.0,
        "base_crop_fraction": [0.0, 0.0],
        "one_factor_changes": {
            "clean_control": {},
            "noisy": {"noise": asdict(_NOISY)},
            "masked": {"point_keep_probability": 0.82},
            "cropped": {"crop_fraction": [0.08, 0.08]},
            "q_grid": {"grid_kind": "linear"},
        },
        "choices_precede_curve_evaluation": True,
    }
    return {**core, "contract_sha256": sha256(_canonical_json(core).encode()).hexdigest()}


@dataclass(frozen=True)
class V5K1PhaseCObservationDesign:
    clean_group_id: str
    stress_stratum: str
    observation_seed: int
    grid: GridProvenance
    noise: NoiseProvenance
    point_keep_probability: float
    crop_fraction: tuple[float, float]
    canonical_json: str
    sha256: str

    def audit_payload(self) -> dict[str, object]:
        return json.loads(self.canonical_json)


def build_v5_k1_phase_c_observation_design(
    *, clean_group_id: str, observation_stress_stratum: str
) -> V5K1PhaseCObservationDesign:
    """Build one explicit one-factor acquisition design before curve evaluation."""

    if not isinstance(clean_group_id, str) or len(clean_group_id) != 64:
        raise ValueError("clean_group_id must be a SHA-256")
    try:
        raw = bytes.fromhex(clean_group_id)
    except ValueError as exc:
        raise ValueError("clean_group_id must be a SHA-256") from exc
    if observation_stress_stratum not in K1_PHASE_C_OBSERVATION_STRESS_STRATA:
        raise ValueError("observation_stress_stratum is not frozen by Phase-C")
    grid = _BASE_GRID
    noise = _CLEAN_NOISE
    keep = 1.0
    crop = (0.0, 0.0)
    if observation_stress_stratum == "noisy":
        noise = _NOISY
    elif observation_stress_stratum == "masked":
        keep = 0.82
    elif observation_stress_stratum == "cropped":
        crop = (0.08, 0.08)
    elif observation_stress_stratum == "q_grid":
        grid = GridProvenance(kind="linear", q_min=1.0e-3, q_max=5.0, n_points=256)
    seed = int.from_bytes(raw[:8], "big")
    core = {
        "schema": V5_K1_PHASE_C_OBSERVATION_DESIGN_SCHEMA,
        "version": V5_K1_PHASE_C_OBSERVATION_DESIGN_VERSION,
        "contract_sha256": v5_k1_phase_c_observation_contract()["contract_sha256"],
        "clean_group_id": clean_group_id,
        "stress_stratum": observation_stress_stratum,
        "effects": list(K1_PHASE_C_OBSERVATION_EFFECTS[observation_stress_stratum]),
        "observation_seed": seed,
        "grid": asdict(grid),
        "noise": asdict(noise),
        "point_keep_probability": keep,
        "crop_fraction": list(crop),
        "mask_seed_namespace": "sha256_clean_group_prefix64_xor_0x4d41534b",
        "generated_before_curve": True,
    }
    encoded = _canonical_json(core)
    return V5K1PhaseCObservationDesign(
        clean_group_id=clean_group_id,
        stress_stratum=observation_stress_stratum,
        observation_seed=seed,
        grid=grid,
        noise=noise,
        point_keep_probability=keep,
        crop_fraction=crop,
        canonical_json=encoded,
        sha256=sha256(encoded.encode()).hexdigest(),
    )


def validate_v5_k1_phase_c_stress_semantics(
    *,
    range_stress_stratum: str,
    geometry_axis_regimes: tuple[str, ...],
    geometry_axis_placements: tuple[str, ...],
    amplitude_axis_regimes: tuple[str, ...],
) -> None:
    """Fail closed if materialized query ranges do not realize their labelled stratum."""

    if not geometry_axis_regimes or not amplitude_axis_regimes:
        raise ValueError("Phase-C needs active geometry and amplitude axes")
    families = tuple(
        "edge" if placement in {"edge_low", "edge_high"} else regime
        for regime, placement in zip(
            geometry_axis_regimes, geometry_axis_placements, strict=True
        )
    ) + tuple(
        "edge" if regime in {"edge_low", "edge_high"} else regime
        for regime in amplitude_axis_regimes
    )
    valid = {
        "full": all(value == "full" for value in geometry_axis_regimes + amplitude_axis_regimes)
        and all(value == "interior" for value in geometry_axis_placements),
        "narrow": all(value == "narrow" for value in geometry_axis_regimes + amplitude_axis_regimes)
        and all(value not in {"edge_low", "edge_high"} for value in geometry_axis_placements),
        "fixed": all(value == "fixed" for value in geometry_axis_regimes + amplitude_axis_regimes)
        and all(value not in {"edge_low", "edge_high"} for value in geometry_axis_placements),
        "edge": "edge" in families,
        "mixed": len(set(families)) >= 2,
    }
    if range_stress_stratum not in valid or not valid[range_stress_stratum]:
        raise ValueError("materialized query does not realize its Phase-C range stratum")


__all__ = [
    "V5_K1_PHASE_C_OBSERVATION_DESIGN_SCHEMA",
    "V5_K1_PHASE_C_OBSERVATION_DESIGN_VERSION",
    "V5_K1_PHASE_C_STRESS_SCHEMA",
    "V5_K1_PHASE_C_STRESS_VERSION",
    "V5K1PhaseCObservationDesign",
    "V5K1PhaseCStressCoordinates",
    "build_v5_k1_phase_c_observation_design",
    "force_v5_k1_phase_c_range_stress",
    "v5_k1_phase_c_observation_contract",
    "validate_v5_k1_phase_c_stress_semantics",
]
