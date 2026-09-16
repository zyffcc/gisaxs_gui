"""Deterministic, compact simulation primitives for Posterior V8.

Recipes contain nonlinear geometry and effective linear amplitudes.  Clean
curves are always evaluated by the authoritative GUI forward model; this file
does not carry a second scattering implementation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
from numbers import Integral

import numpy as np

from src.gimap.features.fitting.domain.physical_constraints import exclusion_size
from src.gimap.features.fitting.domain.scattering_model import make_mixed_model

from .contract import (
    CYLINDER,
    D_DOMAIN,
    D_WIDTH_FRACTION_DOMAIN,
    H_DOMAIN,
    NUM_TOPOLOGIES,
    R_DOMAIN,
    SIZE_WIDTH_FRACTION_DOMAIN,
    VERTICAL_CYLINDER,
    GuiComponentParameters,
    topology_from_id,
)
from .profiled_forward import ResolutionShape


SIMULATION_VERSION = "posterior_v8_phase2_candidate_simulation_v4"
OBSERVATION_VIEW_VERSION = "posterior_v8_multiview_observation_v1"
OBSERVATION_STRATUM_VERSION = "posterior_v8_observation_stratum_v1"
OBSERVATION_SEED_DERIVATION = "posterior_v8_seedsequence_clean_recipe_seed_view_index_namespace_v1"
NOISE_APPLICATION_VERSION = (
    "posterior_v8_observation_noise_poisson_lognormal_overflow_safe_rss_v2"
)
MIN_EFFECTIVE_AMPLITUDE_FRACTION = 0.15
HARD_CORE_SPACING_MARGIN = 1.001
MIN_GRID_POINTS = 64
MAX_GRID_POINTS = 1000
OBSERVATION_GRID_KINDS = ("geometric", "linear", "hybrid")
OBSERVATION_Q_WINDOWS = (
    (9.0e-5, 0.8),
    (3.0e-4, 2.0),
    (1.0e-3, 5.0),
    (3.0e-3, 6.0),
)
OBSERVATION_NOISE_PROFILES = (
    (2.0e3, 0.03),
    (1.0e4, 0.015),
    (5.0e4, 0.006),
    (2.0e5, 0.002),
    (1.0e6, 0.0),
)
OBSERVATION_COUNT_SCALES = tuple(value[0] for value in OBSERVATION_NOISE_PROFILES)
OBSERVATION_RELATIVE_SIGMAS = tuple(value[1] for value in OBSERVATION_NOISE_PROFILES)
OBSERVATION_KEEP_PROBABILITIES = (1.0, 0.97, 0.90, 0.82)
OBSERVATION_CROP_PROFILES = (
    (0.0, 0.0),
    (0.02, 0.02),
    (0.05, 0.05),
    (0.0, 0.08),
    (0.08, 0.0),
)
OBSERVATION_POLICY_PHASE_PERIOD = 60
OBSERVATION_DESIGN_STRATUM_UNIVERSE_VERSION = (
    "posterior_v8_reachable_coupled_phase_design_strata_v1"
)
OBSERVATION_DESIGN_STRATUM_FIELDS = (
    "design_point_count_cycle_slot",
    "q_window_id",
    "noise_id",
)
OBSERVATION_DESIGN_STRATUM_UNIVERSE_SEMANTICS = (
    "reachable_categorical_design_strata_under_the_frozen_60_phase_policy;"
    "numeric_grid_and_full_nuisance_policy_are_acquisition_policy_id_scoped"
)


def _finite_float(value: float, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _seed(value: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError("seed must be a non-negative integer")
    result = int(value)
    if result < 0:
        raise ValueError("seed must be a non-negative integer")
    return result


@dataclass(frozen=True, order=True)
class ObservationDesignStratumCoordinate:
    """Categorical design coordinate independent of a configured numeric grid size."""

    design_point_count_cycle_slot: int
    q_window_id: int
    noise_id: int

    def __post_init__(self) -> None:
        limits = {
            "design_point_count_cycle_slot": 4,
            "q_window_id": len(OBSERVATION_Q_WINDOWS),
            "noise_id": len(OBSERVATION_NOISE_PROFILES),
        }
        for name, limit in limits.items():
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be an integer")
            normalized = int(value)
            if not 0 <= normalized < limit:
                raise ValueError(f"{name} must be in [0, {limit})")
            object.__setattr__(self, name, normalized)


def _observation_policy_base_phase(clean_recipe_seed: int) -> int:
    policy_state = np.random.SeedSequence([_seed(clean_recipe_seed), 0x504F4C49]).generate_state(
        1, dtype=np.uint32
    )
    return int(policy_state[0] % OBSERVATION_POLICY_PHASE_PERIOD)


def _observation_design_stratum_from_base_phase(
    base_phase: int,
    view_index: int,
) -> ObservationDesignStratumCoordinate:
    phase = int(base_phase) + _seed(view_index)
    return ObservationDesignStratumCoordinate(
        design_point_count_cycle_slot=phase % 4,
        q_window_id=(phase // len(OBSERVATION_GRID_KINDS)) % len(OBSERVATION_Q_WINDOWS),
        noise_id=(phase + view_index) % len(OBSERVATION_NOISE_PROFILES),
    )


def sample_observation_design_stratum(
    clean_recipe_seed: int,
    view_index: int,
) -> ObservationDesignStratumCoordinate:
    """Return the lightweight categorical stratum selected before any physics."""

    seed = _seed(clean_recipe_seed)
    index = _seed(view_index)
    return _observation_design_stratum_from_base_phase(
        _observation_policy_base_phase(seed),
        index,
    )


def reachable_observation_design_strata() -> tuple[ObservationDesignStratumCoordinate, ...]:
    """Enumerate the exact categorical universe over one complete policy period."""

    return tuple(
        sorted(
            {
                _observation_design_stratum_from_base_phase(base_phase, view_index)
                for base_phase in range(OBSERVATION_POLICY_PHASE_PERIOD)
                for view_index in range(OBSERVATION_POLICY_PHASE_PERIOD)
            }
        )
    )


OBSERVATION_DESIGN_STRATUM_UNIVERSE = reachable_observation_design_strata()
OBSERVATION_DESIGN_STRATUM_COUNT = len(OBSERVATION_DESIGN_STRATUM_UNIVERSE)
if OBSERVATION_DESIGN_STRATUM_COUNT != 60:  # pragma: no cover - import-time contract guard
    raise RuntimeError("the frozen observation policy must expose exactly 60 design strata")
_OBSERVATION_DESIGN_STRATUM_UNIVERSE_PAYLOAD = {
    "version": OBSERVATION_DESIGN_STRATUM_UNIVERSE_VERSION,
    "fields": OBSERVATION_DESIGN_STRATUM_FIELDS,
    "semantics": OBSERVATION_DESIGN_STRATUM_UNIVERSE_SEMANTICS,
    "phase_period": OBSERVATION_POLICY_PHASE_PERIOD,
    "strata": tuple(asdict(value) for value in OBSERVATION_DESIGN_STRATUM_UNIVERSE),
}
OBSERVATION_DESIGN_STRATUM_UNIVERSE_SHA256 = sha256(
    json.dumps(
        _OBSERVATION_DESIGN_STRATUM_UNIVERSE_PAYLOAD,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
).hexdigest()


@dataclass(frozen=True)
class GridProvenance:
    kind: str = "geometric"
    q_min: float = 1.0e-3
    q_max: float = 5.0
    n_points: int = 256

    def __post_init__(self) -> None:
        kind = str(self.kind).strip().lower()
        if kind not in {"geometric", "linear", "hybrid"}:
            raise ValueError("grid kind must be geometric, linear, or hybrid")
        q_min = _finite_float(self.q_min, "q_min")
        q_max = _finite_float(self.q_max, "q_max")
        if q_min <= 0.0 or q_max <= q_min:
            raise ValueError("grid bounds must satisfy 0 < q_min < q_max")
        if isinstance(self.n_points, (bool, np.bool_)) or not isinstance(self.n_points, Integral):
            raise TypeError("n_points must be an integer")
        n_points = int(self.n_points)
        if not MIN_GRID_POINTS <= n_points <= MAX_GRID_POINTS:
            raise ValueError(f"n_points must be between {MIN_GRID_POINTS} and {MAX_GRID_POINTS}")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "q_min", q_min)
        object.__setattr__(self, "q_max", q_max)
        object.__setattr__(self, "n_points", n_points)

    def values(self) -> np.ndarray:
        if self.kind == "geometric":
            return np.geomspace(self.q_min, self.q_max, self.n_points, dtype=np.float64)
        if self.kind == "linear":
            return np.linspace(self.q_min, self.q_max, self.n_points, dtype=np.float64)
        split = max(2, self.n_points // 2)
        pivot = float(np.sqrt(self.q_min * self.q_max))
        low = np.geomspace(self.q_min, pivot, split, endpoint=False, dtype=np.float64)
        high = np.linspace(pivot, self.q_max, self.n_points - split, dtype=np.float64)
        return np.concatenate((low, high))


@dataclass(frozen=True)
class NoiseProvenance:
    """Noise parameters; ``poisson_count_scale=None`` disables counting noise."""

    poisson_count_scale: float | None = 1.0e5
    relative_sigma: float = 0.003
    sigma_floor_fraction: float = 1.0e-6

    def __post_init__(self) -> None:
        if self.poisson_count_scale is not None:
            count_scale = _finite_float(self.poisson_count_scale, "poisson_count_scale")
            if count_scale <= 0.0:
                raise ValueError("poisson_count_scale must be positive or None")
            object.__setattr__(self, "poisson_count_scale", count_scale)
        relative = _finite_float(self.relative_sigma, "relative_sigma")
        floor = _finite_float(self.sigma_floor_fraction, "sigma_floor_fraction")
        if not 0.0 <= relative <= 0.20:
            raise ValueError("relative_sigma must be in [0, 0.20]")
        if floor <= 0.0:
            raise ValueError("sigma_floor_fraction must be strictly positive")
        object.__setattr__(self, "relative_sigma", relative)
        object.__setattr__(self, "sigma_floor_fraction", floor)


@dataclass(frozen=True)
class SimulationRecipe:
    topology_id: int
    components: tuple[GuiComponentParameters, ...]
    effective_amplitudes: tuple[float, ...]
    resolution: ResolutionShape | None
    resolution_effective_amplitude: float
    background: float
    grid: GridProvenance
    noise: NoiseProvenance
    seed: int
    generator_version: str = SIMULATION_VERSION
    hard_core_spacing_margin: float = HARD_CORE_SPACING_MARGIN
    require_characteristic_coverage: bool = True

    def __post_init__(self) -> None:
        topology = topology_from_id(self.topology_id)
        hard_core_spacing_margin = _finite_float(
            self.hard_core_spacing_margin, "hard_core_spacing_margin"
        )
        if hard_core_spacing_margin != HARD_CORE_SPACING_MARGIN:
            raise ValueError(
                "hard_core_spacing_margin is fixed by this simulation version at "
                f"{HARD_CORE_SPACING_MARGIN}"
            )
        components = tuple(self.components)
        if not all(isinstance(item, GuiComponentParameters) for item in components):
            raise TypeError("components must contain only GuiComponentParameters")
        if tuple(item.shape for item in components) != topology:
            raise ValueError("component shapes/order do not match the canonical topology_id")
        for index, component in enumerate(components):
            if component.D is None:
                continue
            required = _required_spacing(component.shape, component.R, component.h)
            if component.D <= required:
                raise ValueError(
                    f"components[{index}].D violates hard-core spacing: "
                    f"D={component.D:g} must be > {required:g}"
                )
        amplitudes = tuple(
            _finite_float(value, f"effective_amplitudes[{index}]")
            for index, value in enumerate(self.effective_amplitudes)
        )
        if len(amplitudes) != len(components) or any(value <= 0.0 for value in amplitudes):
            raise ValueError("one strictly positive effective amplitude is required per component")
        total = float(np.sum(amplitudes))
        fractions = np.asarray(amplitudes, dtype=np.float64) / total
        if np.any(fractions < MIN_EFFECTIVE_AMPLITUDE_FRACTION - 1e-12):
            raise ValueError(
                "each particle effective amplitude fraction must be at least "
                f"{MIN_EFFECTIVE_AMPLITUDE_FRACTION}"
            )
        if self.resolution is not None and not isinstance(self.resolution, ResolutionShape):
            raise TypeError("resolution must be None or ResolutionShape")
        resolution_amplitude = _finite_float(
            self.resolution_effective_amplitude, "resolution_effective_amplitude"
        )
        if self.resolution is None and resolution_amplitude != 0.0:
            raise ValueError("resolution amplitude must be zero when resolution is absent")
        if self.resolution is not None and resolution_amplitude <= 0.0:
            raise ValueError("present resolution requires a positive effective amplitude")
        background = _finite_float(self.background, "background")
        if background <= 0.0:
            raise ValueError("background must be strictly positive")
        if not isinstance(self.grid, GridProvenance):
            raise TypeError("grid must be GridProvenance")
        if not isinstance(self.noise, NoiseProvenance):
            raise TypeError("noise must be NoiseProvenance")
        if self.generator_version != SIMULATION_VERSION:
            raise ValueError(
                f"unsupported generator_version {self.generator_version!r}; "
                f"expected {SIMULATION_VERSION!r}"
            )
        if not isinstance(self.require_characteristic_coverage, (bool, np.bool_)):
            raise TypeError("require_characteristic_coverage must be boolean")
        if self.require_characteristic_coverage:
            _validate_grid_coverage(components, self.grid)
        object.__setattr__(self, "components", components)
        object.__setattr__(self, "effective_amplitudes", amplitudes)
        object.__setattr__(self, "resolution_effective_amplitude", resolution_amplitude)
        object.__setattr__(self, "background", background)
        object.__setattr__(self, "hard_core_spacing_margin", hard_core_spacing_margin)
        object.__setattr__(self, "topology_id", int(self.topology_id))
        object.__setattr__(self, "seed", _seed(self.seed))

    @property
    def component_count(self) -> int:
        return len(self.components)

    @property
    def effective_amplitude_fractions(self) -> np.ndarray:
        values = np.asarray(self.effective_amplitudes, dtype=np.float64)
        result = values / np.sum(values)
        result.setflags(write=False)
        return result


@dataclass(frozen=True)
class SimulatedCurve:
    recipe: SimulationRecipe
    q: np.ndarray
    clean_intensity: np.ndarray
    intensity: np.ndarray
    sigma: np.ndarray

    def __post_init__(self) -> None:
        arrays = []
        for name in ("q", "clean_intensity", "intensity", "sigma"):
            value = np.asarray(getattr(self, name), dtype=np.float64)
            if value.ndim != 1 or value.size != self.recipe.grid.n_points:
                raise ValueError(f"{name} has an invalid simulation shape")
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{name} contains non-finite values")
            if np.any(value <= 0.0):
                raise ValueError(f"{name} must be strictly positive")
            value.setflags(write=False)
            arrays.append((name, value))
        for name, value in arrays:
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class ObservationView:
    """One deterministic synthetic observation of a shared clean recipe."""

    view_index: int
    observation_seed: int
    q_window_id: int
    noise_id: int
    mask_id: int
    crop_id: int
    grid: GridProvenance
    noise: NoiseProvenance
    preprocess_q_range: tuple[float, float]
    point_keep_probability: float
    version: str = OBSERVATION_VIEW_VERSION

    def __post_init__(self) -> None:
        view_index = _seed(self.view_index)
        observation_seed = _seed(self.observation_seed)
        q_window_id = _seed(self.q_window_id)
        noise_id = _seed(self.noise_id)
        mask_id = _seed(self.mask_id)
        crop_id = _seed(self.crop_id)
        if q_window_id >= len(OBSERVATION_Q_WINDOWS):
            raise ValueError("q_window_id is outside the versioned policy")
        if noise_id >= len(OBSERVATION_NOISE_PROFILES):
            raise ValueError("noise_id is outside the versioned policy")
        if mask_id >= len(OBSERVATION_KEEP_PROBABILITIES):
            raise ValueError("mask_id is outside the versioned policy")
        if crop_id >= len(OBSERVATION_CROP_PROFILES):
            raise ValueError("crop_id is outside the versioned policy")
        if not isinstance(self.grid, GridProvenance):
            raise TypeError("grid must be GridProvenance")
        if not isinstance(self.noise, NoiseProvenance):
            raise TypeError("noise must be NoiseProvenance")
        if (self.grid.q_min, self.grid.q_max) != OBSERVATION_Q_WINDOWS[q_window_id]:
            raise ValueError("grid bounds do not match q_window_id")
        expected_count_scale, expected_relative = OBSERVATION_NOISE_PROFILES[noise_id]
        if (
            self.noise.poisson_count_scale != expected_count_scale
            or self.noise.relative_sigma != expected_relative
        ):
            raise ValueError("noise parameters do not match noise_id")
        try:
            q_range = tuple(float(value) for value in self.preprocess_q_range)
        except (TypeError, ValueError) as exc:
            raise ValueError("preprocess_q_range must contain two finite bounds") from exc
        if (
            len(q_range) != 2
            or not np.all(np.isfinite(q_range))
            or q_range[0] < self.grid.q_min
            or q_range[1] > self.grid.q_max
            or q_range[1] <= q_range[0]
        ):
            raise ValueError("preprocess_q_range must lie inside the observation grid")
        keep = _finite_float(self.point_keep_probability, "point_keep_probability")
        if not 0.75 <= keep <= 1.0:
            raise ValueError("point_keep_probability must be in [0.75, 1]")
        if keep != OBSERVATION_KEEP_PROBABILITIES[mask_id]:
            raise ValueError("point_keep_probability does not match mask_id")
        if q_range != _crop_bounds(self.grid, crop_id):
            raise ValueError("preprocess_q_range does not match crop_id")
        if self.version != OBSERVATION_VIEW_VERSION:
            raise ValueError("unsupported observation-view version")
        eligible = (self.grid.values() >= q_range[0]) & (self.grid.values() <= q_range[1])
        if np.count_nonzero(eligible) < 16:
            raise ValueError("observation crop must retain at least 16 grid points")
        object.__setattr__(self, "view_index", view_index)
        object.__setattr__(self, "observation_seed", observation_seed)
        object.__setattr__(self, "q_window_id", q_window_id)
        object.__setattr__(self, "noise_id", noise_id)
        object.__setattr__(self, "mask_id", mask_id)
        object.__setattr__(self, "crop_id", crop_id)
        object.__setattr__(self, "preprocess_q_range", q_range)
        object.__setattr__(self, "point_keep_probability", keep)

    def simulation_recipe(self, clean_recipe: SimulationRecipe) -> SimulationRecipe:
        """Attach this view without changing the clean physical parameters."""

        if not isinstance(clean_recipe, SimulationRecipe):
            raise TypeError("clean_recipe must be a SimulationRecipe")
        if (
            clean_recipe.noise.poisson_count_scale is not None
            or clean_recipe.noise.relative_sigma != 0.0
        ):
            raise ValueError("observation views require a noise-free clean recipe")
        return replace(
            clean_recipe,
            grid=self.grid,
            noise=self.noise,
            seed=self.observation_seed,
            require_characteristic_coverage=False,
        )

    def selection_mask(self, q: np.ndarray) -> np.ndarray:
        """Return deterministic point dropout; crop remains a separate provenance field."""

        values = np.asarray(q, dtype=np.float64)
        if values.ndim != 1 or values.size != self.grid.n_points:
            raise ValueError("q must match this observation grid")
        if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("q must contain finite positive values")
        eligible = (values >= self.preprocess_q_range[0]) & (values <= self.preprocess_q_range[1])
        rng = np.random.default_rng(np.random.SeedSequence([self.observation_seed, 0x4D41534B]))
        mask = rng.random(values.size) < self.point_keep_probability
        eligible_indices = np.flatnonzero(eligible)
        mask[eligible_indices[[0, -1]]] = True
        if np.count_nonzero(mask & eligible) < 16:
            forced = np.rint(np.linspace(0, eligible_indices.size - 1, 16)).astype(int)
            mask[eligible_indices[forced]] = True
        mask.setflags(write=False)
        return mask


def _validate_grid_coverage(
    components: tuple[GuiComponentParameters, ...], grid: GridProvenance
) -> None:
    characteristic_q = _characteristic_q(components)
    if any(not grid.q_min <= value <= grid.q_max for value in characteristic_q):
        raise ValueError("q grid does not cover every component characteristic scale")


def _characteristic_q(
    components: tuple[GuiComponentParameters, ...],
) -> tuple[float, ...]:
    values = []
    for component in components:
        radial_coefficient = 4.49 if component.shape == "sphere" else 3.83
        values.append(radial_coefficient / component.R)
        if component.shape == CYLINDER:
            values.append(2.0 * np.pi / component.h)
        if component.D is not None:
            values.append(2.0 * np.pi / component.D)
    return tuple(float(value) for value in values)


def _derived_observation_seed(clean_seed: int, view_index: int) -> int:
    state = np.random.SeedSequence(
        [_seed(clean_seed), _seed(view_index), 0x4F425356]
    ).generate_state(2, dtype=np.uint32)
    return int(state[0]) | (int(state[1]) << 32)


def _crop_bounds(grid: GridProvenance, crop_id: int) -> tuple[float, float]:
    left_fraction, right_fraction = OBSERVATION_CROP_PROFILES[crop_id]
    values = grid.values()
    left = min(int(np.floor(left_fraction * values.size)), values.size - 16)
    right = min(int(np.floor(right_fraction * values.size)), values.size - left - 16)
    return float(values[left]), float(values[values.size - right - 1])


def sample_observation_view(
    clean_recipe_seed: int,
    view_index: int,
    *,
    max_points: int = MAX_GRID_POINTS,
) -> ObservationView:
    """Derive a view from only a clean seed; unknown physics is never inspected."""

    clean_recipe_seed = _seed(clean_recipe_seed)
    view_index = _seed(view_index)
    if isinstance(max_points, (bool, np.bool_)) or not isinstance(max_points, Integral):
        raise TypeError("max_points must be an integer")
    max_points = int(max_points)
    if not MIN_GRID_POINTS <= max_points <= MAX_GRID_POINTS:
        raise ValueError(f"max_points must be between {MIN_GRID_POINTS} and {MAX_GRID_POINTS}")
    observation_seed = _derived_observation_seed(clean_recipe_seed, view_index)
    base_phase = _observation_policy_base_phase(clean_recipe_seed)
    phase = base_phase + view_index
    design_stratum = _observation_design_stratum_from_base_phase(base_phase, view_index)
    kind = OBSERVATION_GRID_KINDS[phase % len(OBSERVATION_GRID_KINDS)]
    point_options = tuple(
        sorted(
            {
                MIN_GRID_POINTS,
                max(MIN_GRID_POINTS, max_points // 4),
                max(MIN_GRID_POINTS, max_points // 2),
                max_points,
            }
        )
    )
    n_points = point_options[phase % len(point_options)]
    q_window_id = design_stratum.q_window_id
    q_low, q_high = OBSERVATION_Q_WINDOWS[q_window_id]
    grid = GridProvenance(kind=kind, q_min=q_low, q_max=q_high, n_points=n_points)

    crop_id = (phase // 2 + view_index) % len(OBSERVATION_CROP_PROFILES)
    crop = _crop_bounds(grid, crop_id)
    noise_id = design_stratum.noise_id
    count_scale, relative_sigma = OBSERVATION_NOISE_PROFILES[noise_id]
    mask_id = (phase + view_index) % len(OBSERVATION_KEEP_PROBABILITIES)
    keep_probability = OBSERVATION_KEEP_PROBABILITIES[mask_id]
    return ObservationView(
        view_index=view_index,
        observation_seed=observation_seed,
        q_window_id=q_window_id,
        noise_id=noise_id,
        mask_id=mask_id,
        crop_id=crop_id,
        grid=grid,
        noise=NoiseProvenance(
            poisson_count_scale=count_scale,
            relative_sigma=relative_sigma,
        ),
        preprocess_q_range=crop,
        point_keep_probability=keep_probability,
    )


def _authoritative_gui_parameters(recipe: SimulationRecipe) -> list[float]:
    total_amplitude = float(np.sum(recipe.effective_amplitudes))
    weights = np.asarray(recipe.effective_amplitudes, dtype=np.float64) / total_amplitude
    parameters: list[float] = []
    for component, weight in zip(recipe.components, weights):
        d = 0.0 if component.D is None else component.D
        sigma_d = 0.0 if component.sigma_D is None else component.sigma_D
        if component.shape == CYLINDER:
            parameters.extend(
                [weight, component.R, component.sigma_R, component.h, component.sigma_h, d, sigma_d]
            )
        else:
            parameters.extend([weight, component.R, component.sigma_R, d, sigma_d])
    if recipe.resolution is None:
        sigma_res, nu_res, int_res = 0.0, 0.0, 0.0
    else:
        sigma_res = recipe.resolution.sigma_res
        nu_res = recipe.resolution.nu_res
        int_res = recipe.resolution_effective_amplitude / total_amplitude
    parameters.extend([recipe.background, sigma_res, nu_res, int_res, total_amplitude])
    return parameters


def apply_observation_noise(
    clean_intensity: np.ndarray,
    noise: NoiseProvenance,
    *,
    observation_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the versioned observation-noise law to one exact clean curve.

    This is the single public implementation used by both the legacy
    ``SimulationRecipe`` path and V5 observation views.  The seed namespace
    and operation order are frozen to preserve the pre-extraction numerical
    sequence exactly.
    """

    clean = np.asarray(clean_intensity, dtype=np.float64)
    if clean.ndim != 1 or clean.size == 0:
        raise ValueError("clean_intensity must be a non-empty vector")
    if not np.all(np.isfinite(clean)) or np.any(clean <= 0.0):
        raise ValueError("clean_intensity must contain finite positive values")
    if not isinstance(noise, NoiseProvenance):
        raise TypeError("noise must be NoiseProvenance")
    seed = _seed(observation_seed)

    rng = np.random.default_rng(np.random.SeedSequence([seed, 0x5638]))
    reference = max(float(np.median(clean)), np.finfo(np.float64).tiny)
    if noise.poisson_count_scale is None:
        intensity = clean.copy()
        sigma_poisson = np.zeros_like(clean)
    else:
        scale = noise.poisson_count_scale
        expected_counts = np.clip(clean / reference * scale, 0.0, 1.0e9)
        counts = rng.poisson(expected_counts)
        intensity = counts.astype(np.float64) / scale * reference
        sigma_poisson = np.sqrt(np.maximum(counts, 1.0)) / scale * reference
    if noise.relative_sigma > 0.0:
        intensity *= np.exp(rng.normal(0.0, noise.relative_sigma, size=intensity.shape))
    floor = max(noise.sigma_floor_fraction * reference, np.finfo(np.float64).tiny)
    intensity = np.maximum(intensity, floor)
    # ``sqrt(a**2 + b**2 + c**2)`` underflows to zero for scientifically legal
    # subnormal-scale curves (and may overflow at the opposite extreme).  The
    # chained hypot is the same root-sum-square law with binary64 scaling.
    sigma = np.hypot(
        np.hypot(sigma_poisson, noise.relative_sigma * intensity),
        floor,
    )
    intensity.setflags(write=False)
    sigma.setflags(write=False)
    return intensity, sigma


def simulate_recipe(recipe: SimulationRecipe) -> SimulatedCurve:
    """Evaluate one recipe exactly, then add deterministic configured noise."""
    if not isinstance(recipe, SimulationRecipe):
        raise TypeError("recipe must be a SimulationRecipe")
    q = recipe.grid.values()
    model = make_mixed_model([component.shape for component in recipe.components])
    clean = np.asarray(model(q, *_authoritative_gui_parameters(recipe)), dtype=np.float64)
    if clean.shape != q.shape or not np.all(np.isfinite(clean)) or np.any(clean <= 0.0):
        raise ValueError("authoritative forward produced a non-positive or non-finite curve")

    intensity, sigma = apply_observation_noise(
        clean,
        recipe.noise,
        observation_seed=recipe.seed,
    )
    return SimulatedCurve(recipe, q, clean, intensity, sigma)


def _log_uniform(rng: np.random.Generator, low: float, high: float) -> float:
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))


def _required_spacing(shape: str, radius: float, height: float | None) -> float:
    """Return the strict V8 D threshold using the production exclusion rule."""
    params = {"R": radius}
    if height is not None:
        params["h"] = height
    threshold = exclusion_size(shape, params)
    if threshold is None or not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError(f"cannot determine hard-core exclusion size for {shape!r}")
    return HARD_CORE_SPACING_MARGIN * float(threshold)


def _stratified_size(
    rng: np.random.Generator,
    domain_low: float,
    domain_high: float,
    index: int,
    count: int,
) -> float:
    # Disjoint log cells are used only within repeated instances of the same
    # shape. Applying cells by global component slot would leak the canonical
    # shape order into size and create an artificial topology shortcut.
    cell_low = index / count
    cell_high = (index + 1) / count
    position = rng.uniform(cell_low + 0.12 / count, cell_high - 0.12 / count)
    return float(np.exp(np.log(domain_low) + position * np.log(domain_high / domain_low)))


def sample_identifiable_recipe(
    seed: int,
    topology_id: int | None = None,
    max_points: int = 256,
    *,
    noise: NoiseProvenance | None = None,
) -> SimulationRecipe:
    """Sample one deterministic candidate for later identifiability screening.

    The local visibility rules intentionally do not claim global
    identifiability. Competing topology/branch refits decide whether a recipe
    belongs to the identifiable core or the ambiguity bank.
    """
    seed = _seed(seed)
    if isinstance(max_points, (bool, np.bool_)) or not isinstance(max_points, Integral):
        raise TypeError("max_points must be an integer")
    grid = GridProvenance(n_points=int(max_points))
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0x51A]))
    if topology_id is None:
        topology_id = int(rng.integers(0, NUM_TOPOLOGIES))
    topology = topology_from_id(topology_id)
    count = len(topology)

    shape_counts = {shape: topology.count(shape) for shape in set(topology)}
    shape_occurrences = {shape: 0 for shape in shape_counts}
    components = []
    for shape in topology:
        occurrence = shape_occurrences[shape]
        repetitions = shape_counts[shape]
        shape_occurrences[shape] += 1
        if repetitions == 1:
            r = _log_uniform(rng, R_DOMAIN.low, R_DOMAIN.high)
        else:
            r = _stratified_size(rng, R_DOMAIN.low, R_DOMAIN.high, occurrence, repetitions)
        sigma_r_fraction = float(rng.uniform(0.05, 0.28))
        sigma_r = sigma_r_fraction if shape == VERTICAL_CYLINDER else r * sigma_r_fraction
        if shape == CYLINDER:
            h = _log_uniform(rng, H_DOMAIN.low, H_DOMAIN.high)
            sigma_h = h * float(rng.uniform(0.05, 0.28))
        else:
            h = sigma_h = None
        required_d = _required_spacing(shape, r, h)
        feasible_d_low = float(np.nextafter(max(D_DOMAIN.low, required_d), np.inf))
        if rng.random() < 0.50 and feasible_d_low <= D_DOMAIN.high:
            d = _log_uniform(rng, feasible_d_low, D_DOMAIN.high)
            sigma_d = d * float(rng.uniform(max(0.08, D_WIDTH_FRACTION_DOMAIN.low), 0.28))
        else:
            d = sigma_d = None
        components.append(
            GuiComponentParameters(shape, r, sigma_r, h=h, sigma_h=sigma_h, D=d, sigma_D=sigma_d)
        )

    remaining = 1.0 - count * MIN_EFFECTIVE_AMPLITUDE_FRACTION
    if remaining < -1e-12:  # pragma: no cover - protected by K<=4 contract
        raise RuntimeError("amplitude floor is infeasible for this topology")
    extra = rng.dirichlet(np.full(count, 2.0)) * max(remaining, 0.0)
    fractions = MIN_EFFECTIVE_AMPLITUDE_FRACTION + extra
    total_amplitude = _log_uniform(rng, 1.0e2, 1.0e4)
    amplitudes = tuple(float(total_amplitude * value) for value in fractions)

    if rng.random() < 0.50:
        resolution = ResolutionShape(
            sigma_res=_log_uniform(rng, 0.004, 0.04),
            nu_res=float(rng.uniform(3.0, 10.0)),
        )
        resolution_amplitude = total_amplitude * _log_uniform(rng, 0.01, 0.20)
    else:
        resolution = None
        resolution_amplitude = 0.0
    background = total_amplitude * _log_uniform(rng, 1.0e-6, 1.0e-3)
    return SimulationRecipe(
        topology_id=topology_id,
        components=tuple(components),
        effective_amplitudes=amplitudes,
        resolution=resolution,
        resolution_effective_amplitude=resolution_amplitude,
        background=background,
        grid=grid,
        noise=NoiseProvenance() if noise is None else noise,
        seed=seed,
    )
