"""Compact bounds-first/local-coordinate Posterior V8 pilot generator."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from numbers import Integral
from pathlib import Path
from typing import Sequence

import numpy as np

from .bounds_first_contract import (
    BOUND_PLACEMENTS,
    BOUNDS_EMBEDDING_DIM,
    BOUNDS_EMBEDDING_VERSION,
    BOUNDS_FIRST_SCHEMA_VERSION,
    LOCAL_TARGET_SEMANTICS,
    RANGE_REGIMES,
    BoundsFirstLabel,
    BoundsProvenance,
    local_varying_mask,
)
from .branch_catalog import branch_pattern_id
from .branch_codec import INACTIVE_UNIT_VALUE, ProfiledBranchCodec, ResolutionBounds
from .canonical_branch_catalog import canonicalize_d_flags
from .canonical_component_slots import (
    CANONICAL_COMPONENT_SLOTS_VERSION,
    canonicalize_component_slots,
)
from .contract import (
    CYLINDER,
    D_DOMAIN,
    D_WIDTH_FRACTION_DOMAIN,
    H_DOMAIN,
    NUM_TOPOLOGIES,
    R_DOMAIN,
    RESOLUTION_NU_DOMAIN,
    RESOLUTION_SIGMA_DOMAIN,
    SIZE_WIDTH_FRACTION_DOMAIN,
    TOPOLOGIES,
    VERTICAL_CYLINDER,
    ClosedInterval,
    GuiComponentBounds,
    full_component_bounds,
    topology_id_for,
)
from .preprocessing import DEFAULT_CONTRACT, preprocess_curve
from .simulation import (
    MIN_EFFECTIVE_AMPLITUDE_FRACTION,
    GridProvenance,
    NoiseProvenance,
    SimulationRecipe,
    simulate_recipe,
)


BOUNDS_GENERATOR_VERSION = "posterior_v8_bounds_first_exact_endpoint_physical_ranges_v3"
PILOT_GENERATOR_VERSION = "posterior_v8_bounds_first_compact_pilot_v5"
LOCAL_TARGET_SAMPLING_VERSION = (
    "posterior_v8_open_uniform_then_exact_endpoint_decode_canonical_component_slots_v4"
)
# Match the support used by the logistic-normal training objective. Exact
# closed-boundary values remain an optimizer stress test rather than an atom
# that a continuous density cannot represent.
LOCAL_TARGET_OPEN_EPSILON = 1.0e-5
RANGE_CODE = {value: index for index, value in enumerate(RANGE_REGIMES)}
PLACEMENT_CODE = {value: index for index, value in enumerate(BOUND_PLACEMENTS)}
TASK_CODE = {"in_domain_solution": 0, "no_solution": 1, "ood": 2}
ARRAY_ORDER = (
    "x",
    "point_mask",
    "global_features",
    "topology_id",
    "branch_pattern_id",
    "task_kind",
    "truth_available",
    "target_local_unit",
    "global_reference_unit",
    "active_dimension_mask",
    "local_varying_mask",
    "bounds_embedding",
    "bounds_sha256",
    "range_regime",
    "bound_placement",
    "bounds_seed",
    "local_target_seed",
    "observation_seed",
)


def _integer(value, name, *, positive=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < (1 if positive else 0):
        raise ValueError(f"{name} must be {'positive' if positive else 'non-negative'}")
    return result


@dataclass(frozen=True)
class BoundsFirstPilotConfig:
    master_seed: int = 20260902
    sample_count: int = 6
    topology_ids: tuple[int, ...] = (0, 1, 2, 3)
    points: int = 64
    noise_mode: str = "clean"

    def __post_init__(self) -> None:
        object.__setattr__(self, "master_seed", _integer(self.master_seed, "master_seed"))
        count = _integer(self.sample_count, "sample_count", positive=True)
        points = _integer(self.points, "points", positive=True)
        topology_ids = tuple(self.topology_ids)
        if not topology_ids or any(
            isinstance(value, bool)
            or not isinstance(value, Integral)
            or not 0 <= int(value) < NUM_TOPOLOGIES
            for value in topology_ids
        ):
            raise ValueError("topology_ids must contain valid Posterior V8 topology IDs")
        if not 64 <= points <= 1000:
            raise ValueError("points must be in [64, 1000]")
        if self.noise_mode not in {"clean", "default"}:
            raise ValueError("noise_mode must be clean or default")
        object.__setattr__(self, "sample_count", count)
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "topology_ids", tuple(int(value) for value in topology_ids))


@dataclass(frozen=True)
class BoundsFirstPilotRecord:
    record_index: int
    label: BoundsFirstLabel
    local_target_seed: int
    observation_seed: int
    effective_amplitudes: tuple[float, ...]
    background: float
    resolution_effective_amplitude: float
    x: np.ndarray
    point_mask: np.ndarray
    global_features: np.ndarray


@dataclass(frozen=True)
class BoundsFirstPilotArtifact:
    npz_path: Path
    metadata_path: Path
    sample_count: int
    npz_sha256: str


def _derived_seed(master_seed: int, sample_index: int, namespace: int) -> int:
    state = np.random.SeedSequence(
        [int(master_seed), int(sample_index), int(namespace)]
    ).generate_state(2, dtype=np.uint32)
    return int(state[0]) | (int(state[1]) << 32)


def _interval(rng, domain, regime, placement, *, log_space=True, fixed=False):
    if regime == "full" and not fixed:
        return domain
    low = np.log(domain.low) if log_space else domain.low
    high = np.log(domain.high) if log_space else domain.high
    if fixed:
        coordinate = {
            "edge_low": 0.0,
            "edge_high": 1.0,
        }.get(placement, float(rng.uniform(0.1, 0.9)))
        start = stop = low + coordinate * (high - low)
    else:
        width_fraction = (
            float(rng.uniform(0.45, 0.78))
            if regime == "wide"
            else float(rng.uniform(0.04, 0.18))
        )
        available = 1.0 - width_fraction
        if placement == "edge_low":
            start_fraction = 0.0
        elif placement == "edge_high":
            start_fraction = available
        elif placement == "asymmetric_low":
            start_fraction = float(rng.uniform(0.0, 0.20 * available))
        elif placement == "asymmetric_high":
            start_fraction = float(rng.uniform(0.80 * available, available))
        else:
            start_fraction = float(rng.uniform(0.0, available))
        start = low + start_fraction * (high - low)
        stop = start + width_fraction * (high - low)
    if log_space:
        if fixed and placement == "edge_low":
            start = stop = domain.low
        elif fixed and placement == "edge_high":
            start = stop = domain.high
        else:
            start = domain.low if placement == "edge_low" else float(np.exp(start))
            stop = domain.high if placement == "edge_high" else float(np.exp(stop))
    start = min(max(float(start), domain.low), domain.high)
    stop = min(max(float(stop), domain.low), domain.high)
    return ClosedInterval(start, stop)


def _component_bounds(rng, shape, d_present, regime, placement, *, fixed_r=False):
    if regime == "full" and not fixed_r:
        return full_component_bounds(
            shape, d_policy="required" if d_present else "absent"
        )
    r = _interval(
        rng, R_DOMAIN, regime, placement, fixed=fixed_r, log_space=True
    )
    r_fraction = _interval(
        rng, SIZE_WIDTH_FRACTION_DOMAIN, regime, placement, log_space=False
    )
    sigma_r = (
        r_fraction
        if shape == VERTICAL_CYLINDER
        else ClosedInterval(r.low * r_fraction.low, r.high * r_fraction.high)
    )
    h = sigma_h = None
    if shape == CYLINDER:
        h = _interval(rng, H_DOMAIN, regime, placement, log_space=True)
        h_fraction = _interval(
            rng, SIZE_WIDTH_FRACTION_DOMAIN, regime, placement, log_space=False
        )
        sigma_h = ClosedInterval(h.low * h_fraction.low, h.high * h_fraction.high)
    d = sigma_d = None
    if d_present:
        d = _interval(rng, D_DOMAIN, regime, placement, log_space=True)
        d_fraction = _interval(
            rng, D_WIDTH_FRACTION_DOMAIN, regime, placement, log_space=False
        )
        sigma_d = ClosedInterval(d.low * d_fraction.low, d.high * d_fraction.high)
    return GuiComponentBounds(
        shape,
        R=r,
        sigma_R=sigma_r,
        h=h,
        sigma_h=sigma_h,
        D=d,
        sigma_D=sigma_d,
    )


def sample_user_bounds(
    topology: Sequence[str],
    d_present: Sequence[bool],
    resolution_present: bool,
    *,
    regime: str,
    placement: str,
    bounds_seed: int,
    max_attempts: int = 256,
) -> BoundsProvenance:
    """Sample valid GUI bounds without accepting or inspecting a truth value."""

    topology = tuple(topology)
    flags = tuple(bool(value) for value in d_present)
    if topology not in TOPOLOGIES:
        raise ValueError("topology must be a canonical Posterior V8 topology")
    if len(flags) != len(topology):
        raise ValueError("d_present length must match topology")
    padded_flags = flags + (False,) * (4 - len(flags))
    if canonicalize_d_flags(topology_id_for(topology), padded_flags) != padded_flags:
        raise ValueError(
            "d_present must use the canonical absent-before-present order "
            "within equal-shape component groups"
        )
    if regime not in RANGE_REGIMES or placement not in BOUND_PLACEMENTS:
        raise ValueError("unknown range regime or bound placement")
    seed = _integer(bounds_seed, "bounds_seed")
    attempts = _integer(max_attempts, "max_attempts", positive=True)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0xB0A4D5]))
    last_error = None
    for attempt in range(attempts):
        try:
            bounds = tuple(
                _component_bounds(
                    rng,
                    shape,
                    present,
                    regime,
                    placement,
                    fixed_r=placement == "partial_fixed" and slot == 0,
                )
                for slot, (shape, present) in enumerate(zip(topology, flags))
            )
            resolution = None
            if resolution_present:
                resolution = ResolutionBounds(
                    _interval(
                        rng,
                        RESOLUTION_SIGMA_DOMAIN,
                        regime,
                        placement,
                        log_space=True,
                    ),
                    _interval(
                        rng,
                        RESOLUTION_NU_DOMAIN,
                        regime,
                        placement,
                        log_space=False,
                    ),
                )
            ProfiledBranchCodec.build(
                topology, bounds, flags, resolution_bounds=resolution
            )
            return BoundsProvenance.create(
                bounds_seed=seed,
                generation_attempt=attempt,
                range_regime=regime,
                placement=placement,
                component_bounds=bounds,
                d_present=flags,
                resolution_bounds=resolution,
            )
        except (RuntimeError, TypeError, ValueError) as exc:
            last_error = exc
    raise RuntimeError(
        f"could not sample feasible physical user bounds after {attempts} attempts"
    ) from last_error


def sample_solution_label(
    bounds: BoundsProvenance,
    *,
    local_target_seed: int,
) -> BoundsFirstLabel:
    """Sample local truth only after the immutable bounds provenance exists."""

    if not isinstance(bounds, BoundsProvenance):
        raise TypeError("bounds must be BoundsProvenance")
    seed = _integer(local_target_seed, "local_target_seed")
    codec = bounds.local_codec()
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0x10CA1]))
    active_values = rng.uniform(
        LOCAL_TARGET_OPEN_EPSILON,
        1.0 - LOCAL_TARGET_OPEN_EPSILON,
        len(codec.active_indices),
    )
    latent, _, resolution = codec.decode_active(active_values)
    canonical = canonicalize_component_slots(codec, latent, resolution)
    latent = canonical.components
    gui = codec.latent_components_to_gui(latent)
    local = canonical.coordinates
    global_coordinates = bounds.global_reference_codec().encode(latent, resolution)
    return BoundsFirstLabel(
        task_kind="in_domain_solution",
        bounds=bounds,
        local_target_unit=local.unit_cube,
        global_reference_unit=global_coordinates.unit_cube,
        truth_components=gui,
        truth_resolution=resolution,
    )


def _record(config: BoundsFirstPilotConfig, index: int) -> BoundsFirstPilotRecord:
    topology_id = config.topology_ids[index % len(config.topology_ids)]
    topology = TOPOLOGIES[topology_id]
    regime = RANGE_REGIMES[index % len(RANGE_REGIMES)]
    placement = BOUND_PLACEMENTS[index % len(BOUND_PLACEMENTS)]
    candidate_flags = tuple((index + slot) % 2 == 0 for slot in range(len(topology)))
    flags = canonicalize_d_flags(
        topology_id,
        candidate_flags + (False,) * (4 - len(candidate_flags)),
    )[: len(topology)]
    resolution_present = index % 2 == 0
    bounds_seed = _derived_seed(config.master_seed, index, 0xB0A4D5)
    local_seed = _derived_seed(config.master_seed, index, 0x10CA1)
    observation_seed = _derived_seed(config.master_seed, index, 0x0B5E)
    bounds = sample_user_bounds(
        topology,
        flags,
        resolution_present,
        regime=regime,
        placement=placement,
        bounds_seed=bounds_seed,
    )
    label = sample_solution_label(bounds, local_target_seed=local_seed)
    rng = np.random.default_rng(np.random.SeedSequence([observation_seed, 0xA6]))
    count = len(topology)
    remaining = 1.0 - count * MIN_EFFECTIVE_AMPLITUDE_FRACTION
    fractions = MIN_EFFECTIVE_AMPLITUDE_FRACTION + remaining * rng.dirichlet(
        np.full(count, 2.0)
    )
    total = float(np.exp(rng.uniform(np.log(1.0e2), np.log(1.0e4))))
    amplitudes = tuple(float(total * value) for value in fractions)
    background = total * float(np.exp(rng.uniform(np.log(1e-6), np.log(1e-3))))
    resolution_amplitude = (
        0.0
        if label.truth_resolution is None
        else total * float(np.exp(rng.uniform(np.log(0.01), np.log(0.20))))
    )
    noise = (
        NoiseProvenance(None, 0.0)
        if config.noise_mode == "clean"
        else NoiseProvenance()
    )
    recipe = SimulationRecipe(
        topology_id=topology_id,
        components=label.truth_components,
        effective_amplitudes=amplitudes,
        resolution=label.truth_resolution,
        resolution_effective_amplitude=resolution_amplitude,
        background=background,
        grid=GridProvenance(n_points=config.points),
        noise=noise,
        seed=observation_seed,
        require_characteristic_coverage=False,
    )
    simulated = simulate_recipe(recipe)
    curve = preprocess_curve(simulated.q, simulated.intensity, simulated.sigma)
    return BoundsFirstPilotRecord(
        index,
        label,
        local_seed,
        observation_seed,
        amplitudes,
        float(background),
        float(resolution_amplitude),
        curve.x,
        curve.point_mask,
        curve.global_features,
    )


def _records(config):
    return tuple(_record(config, index) for index in range(config.sample_count))


def _arrays(records):
    return {
        "x": np.stack([item.x for item in records]),
        "point_mask": np.stack([item.point_mask for item in records]),
        "global_features": np.stack([item.global_features for item in records]),
        "topology_id": np.asarray(
            [item.label.bounds.local_codec().topology_id for item in records], dtype=np.int32
        ),
        "branch_pattern_id": np.asarray(
            [
                branch_pattern_id(
                    item.label.bounds.d_present
                    + (False,) * (4 - len(item.label.bounds.d_present)),
                    item.label.truth_resolution is not None,
                )
                for item in records
            ],
            dtype=np.int32,
        ),
        "task_kind": np.full(len(records), TASK_CODE["in_domain_solution"], dtype=np.uint8),
        "truth_available": np.ones(len(records), dtype=np.bool_),
        "target_local_unit": np.asarray(
            [item.label.local_target_unit for item in records], dtype=np.float32
        ),
        "global_reference_unit": np.asarray(
            [item.label.global_reference_unit for item in records], dtype=np.float32
        ),
        "active_dimension_mask": np.asarray(
            [item.label.bounds.local_codec().active_mask for item in records], dtype=np.bool_
        ),
        "local_varying_mask": np.asarray(
            [local_varying_mask(item.label.bounds) for item in records], dtype=np.bool_
        ),
        "bounds_embedding": np.asarray(
            [item.label.bounds.embedding for item in records], dtype=np.float32
        ),
        "bounds_sha256": np.asarray(
            [item.label.bounds.sha256.encode("ascii") for item in records], dtype="S64"
        ),
        "range_regime": np.asarray(
            [RANGE_CODE[item.label.bounds.range_regime] for item in records], dtype=np.uint8
        ),
        "bound_placement": np.asarray(
            [PLACEMENT_CODE[item.label.bounds.placement] for item in records], dtype=np.uint8
        ),
        "bounds_seed": np.asarray(
            [item.label.bounds.bounds_seed for item in records], dtype=np.uint64
        ),
        "local_target_seed": np.asarray(
            [item.local_target_seed for item in records], dtype=np.uint64
        ),
        "observation_seed": np.asarray(
            [item.observation_seed for item in records], dtype=np.uint64
        ),
    }


def _record_payload(item):
    label = item.label
    return {
        "record_index": item.record_index,
        "task_kind": label.task_kind,
        "bounds": json.loads(label.bounds.canonical_json),
        "bounds_sha256": label.bounds.sha256,
        "local_target_seed": item.local_target_seed,
        "observation_seed": item.observation_seed,
        "target_local_unit": label.local_target_unit,
        "global_reference_unit": label.global_reference_unit,
        "truth_components": [asdict(value) for value in label.truth_components],
        "truth_resolution": (
            None if label.truth_resolution is None else asdict(label.truth_resolution)
        ),
        "effective_amplitudes": item.effective_amplitudes,
        "background": item.background,
        "resolution_effective_amplitude": item.resolution_effective_amplitude,
    }


def generate_compact_pilot(config: BoundsFirstPilotConfig):
    """Return deterministic training arrays plus complete physical provenance."""

    if not isinstance(config, BoundsFirstPilotConfig):
        raise TypeError("config must be BoundsFirstPilotConfig")
    records = _records(config)
    arrays = _arrays(records)
    metadata = {
        "dataset_schema_version": BOUNDS_FIRST_SCHEMA_VERSION,
        "dataset_generator_version": PILOT_GENERATOR_VERSION,
        "bounds_generator_version": BOUNDS_GENERATOR_VERSION,
        "bounds_embedding_version": BOUNDS_EMBEDDING_VERSION,
        "local_target_semantics": LOCAL_TARGET_SEMANTICS,
        "local_target_sampling_version": LOCAL_TARGET_SAMPLING_VERSION,
        "canonical_component_slots_version": CANONICAL_COMPONENT_SLOTS_VERSION,
        "local_target_open_epsilon": LOCAL_TARGET_OPEN_EPSILON,
        "generation_order": [
            "sample_gui_physical_bounds",
            "build_bounds_specific_profiled_branch_codec",
            "sample_local_unit_truth",
            "decode_physical_truth",
            "canonicalize_user_bounds_feasible_component_slots",
            "encode_canonical_local_and_global_coordinates",
            "simulate_observation",
        ],
        "forbidden_shortcuts": [
            "truth_conditioned_or_truth_containing_range_generation",
            "global_26d_axis_box_presented_as_gui_physical_bounds",
            "negative_or_ood_row_with_fabricated_inverse_truth",
        ],
        "task_codes": TASK_CODE,
        "range_codes": RANGE_CODE,
        "placement_codes": PLACEMENT_CODE,
        "bounds_embedding_dimension": BOUNDS_EMBEDDING_DIM,
        "array_order": ARRAY_ORDER,
        "config": asdict(config),
        "records": [_record_payload(item) for item in records],
    }
    validate_compact_pilot(config, arrays, metadata, regenerate=False)
    return arrays, metadata


def validate_compact_pilot(config, arrays, metadata, *, regenerate=True):
    if tuple(arrays) != ARRAY_ORDER:
        raise ValueError("bounds-first pilot arrays have the wrong schema/order")
    if metadata.get("dataset_schema_version") != BOUNDS_FIRST_SCHEMA_VERSION:
        raise ValueError("bounds-first pilot metadata has the wrong schema")
    if (
        metadata.get("canonical_component_slots_version")
        != CANONICAL_COMPONENT_SLOTS_VERSION
    ):
        raise ValueError("bounds-first pilot component-slot contract is missing")
    if len(metadata.get("records", ())) != config.sample_count:
        raise ValueError("bounds-first pilot metadata record count is inconsistent")
    if arrays["bounds_embedding"].shape != (
        config.sample_count,
        BOUNDS_EMBEDDING_DIM,
    ):
        raise ValueError("bounds embedding array has the wrong shape")
    active = arrays["active_dimension_mask"]
    varying = arrays["local_varying_mask"]
    target = arrays["target_local_unit"]
    if np.any(varying & ~active):
        raise ValueError("local varying mask cannot activate an inactive branch coordinate")
    if np.any(target[varying] <= 0.0) or np.any(target[varying] >= 1.0):
        raise ValueError("density targets must lie inside the open local unit cube")
    fixed_or_inactive = ~varying
    if not np.all(target[fixed_or_inactive] == np.float32(INACTIVE_UNIT_VALUE)):
        raise ValueError("fixed/inactive local target coordinates must equal 0.5")
    if not np.all(arrays["truth_available"]):
        raise ValueError("compact solution pilot must not mix uncertified negative/OOD rows")
    if regenerate:
        expected_arrays, expected_metadata = generate_compact_pilot(config)
        for name in ARRAY_ORDER:
            if not np.array_equal(arrays[name], expected_arrays[name]):
                raise ValueError(f"bounds-first pilot array {name} is not reproducible")
        if metadata != expected_metadata:
            raise ValueError("bounds-first pilot metadata is not reproducible")


def build_compact_pilot(output_dir: str | Path, config: BoundsFirstPilotConfig):
    destination = Path(output_dir)
    npz_path = destination / "bounds_first_pilot_v5.npz"
    metadata_path = destination / "bounds_first_pilot_v5.json"
    if npz_path.exists() or metadata_path.exists():
        raise FileExistsError("refusing to overwrite an existing bounds-first pilot")
    destination.mkdir(parents=True, exist_ok=True)
    arrays, metadata = generate_compact_pilot(config)
    np.savez_compressed(npz_path, **arrays)
    digest = sha256(npz_path.read_bytes()).hexdigest()
    metadata = {**metadata, "npz_sha256": digest}
    metadata_path.write_text(
        json.dumps(metadata, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return BoundsFirstPilotArtifact(npz_path, metadata_path, config.sample_count, digest)


__all__ = [
    "ARRAY_ORDER",
    "BOUNDS_GENERATOR_VERSION",
    "LOCAL_TARGET_OPEN_EPSILON",
    "LOCAL_TARGET_SAMPLING_VERSION",
    "PILOT_GENERATOR_VERSION",
    "PLACEMENT_CODE",
    "RANGE_CODE",
    "TASK_CODE",
    "BoundsFirstPilotArtifact",
    "BoundsFirstPilotConfig",
    "BoundsFirstPilotRecord",
    "build_compact_pilot",
    "generate_compact_pilot",
    "sample_solution_label",
    "sample_user_bounds",
    "validate_compact_pilot",
]
