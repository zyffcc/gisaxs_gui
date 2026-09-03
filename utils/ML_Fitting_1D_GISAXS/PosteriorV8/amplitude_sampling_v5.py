"""Versioned coefficient-composition sampling for V5 synthetic recipes.

These strata deliberately include small particle coefficients and strong
background/legacy-Resolution confounding.  A coefficient stratum is only a
data-generation provenance label: actual visibility is decided later by the
exact delete/refit observability workflow.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np

from .gui_amplitude_constraints import GuiAmplitudeConstraint


V5_AMPLITUDE_SAMPLING_SCHEMA = "gisaxs.posterior_v8.amplitude_composition/v2"
V5_AMPLITUDE_SAMPLING_VERSION = (
    "posterior_v8_independent_gui_k_int_amplitude_composition_v3"
)
V5_AMPLITUDE_REGIMES = (
    "balanced_particles",
    "weak_particle_coefficient",
    "dominant_particle_coefficient",
    "background_confounded",
    "resolution_weak_coefficient",
    "resolution_confounded",
)


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be positive")
    return result


def _non_negative_integer(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _log_uniform(rng: np.random.Generator, low: float, high: float) -> float:
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))


@dataclass(frozen=True)
class V5AmplitudeComposition:
    """GUI amplitudes and their effective forward coefficients."""

    component_count: int
    resolution_present: bool
    regime: str
    seed: int
    background: float
    k: float
    particle_amplitudes: tuple[float, ...]
    resolution_amplitude: float
    selected_particle_slot: int | None
    schema_version: str = V5_AMPLITUDE_SAMPLING_SCHEMA
    generator_version: str = V5_AMPLITUDE_SAMPLING_VERSION

    def __post_init__(self) -> None:
        count = _positive_integer(self.component_count, "component_count")
        if count > 4:
            raise ValueError("component_count must be at most four")
        if not isinstance(self.resolution_present, (bool, np.bool_)):
            raise TypeError("resolution_present must be boolean")
        if self.regime not in V5_AMPLITUDE_REGIMES:
            raise ValueError(f"regime must be one of {V5_AMPLITUDE_REGIMES}")
        if self.regime.startswith("resolution_") and not self.resolution_present:
            raise ValueError("Resolution coefficient regimes require Resolution presence")
        _non_negative_integer(self.seed, "seed")
        particles = tuple(float(value) for value in self.particle_amplitudes)
        values = np.asarray(
            [self.background, self.k, *particles, self.resolution_amplitude],
            dtype=np.float64,
        )
        if len(particles) != count:
            raise ValueError("one particle amplitude is required per component")
        if not np.all(np.isfinite(values)):
            raise ValueError("amplitude composition must contain finite values")
        if self.background < 0.0 or self.k <= 0.0 or any(value < 0.0 for value in particles):
            raise ValueError("background and particle amplitudes must be non-negative")
        if not any(value > 0.0 for value in particles):
            raise ValueError("explicit GUI k requires at least one positive particle amplitude")
        if self.resolution_amplitude < 0.0 or (
            not self.resolution_present and self.resolution_amplitude != 0.0
        ):
            raise ValueError(
                "Resolution amplitude must be non-negative and zero when Resolution is absent"
            )
        slot = self.selected_particle_slot
        if slot is not None and (
            isinstance(slot, (bool, np.bool_))
            or not isinstance(slot, Integral)
            or not 0 <= int(slot) < count
        ):
            raise ValueError("selected_particle_slot must identify a component or be None")
        if self.schema_version != V5_AMPLITUDE_SAMPLING_SCHEMA:
            raise ValueError("unsupported V5 amplitude composition schema")
        if self.generator_version != V5_AMPLITUDE_SAMPLING_VERSION:
            raise ValueError("unsupported V5 amplitude generator version")
        object.__setattr__(self, "component_count", count)
        object.__setattr__(self, "resolution_present", bool(self.resolution_present))
        object.__setattr__(self, "background", float(self.background))
        object.__setattr__(self, "k", float(self.k))
        object.__setattr__(self, "particle_amplitudes", particles)
        object.__setattr__(self, "resolution_amplitude", float(self.resolution_amplitude))
        object.__setattr__(
            self,
            "selected_particle_slot",
            None if slot is None else int(slot),
        )

    @property
    def particle_weights(self) -> tuple[float, ...]:
        """Compatibility name for the independent GUI ``Int_i`` values."""
        return tuple(float(value / self.k) for value in self.particle_amplitudes)

    @property
    def effective_particle_fractions(self) -> tuple[float, ...]:
        total = float(np.sum(self.particle_amplitudes))
        return tuple(float(value / total) for value in self.particle_amplitudes)

    @property
    def int_res(self) -> float:
        return 0.0 if not self.resolution_present else self.resolution_amplitude / self.k

    @property
    def coefficient_vector(self) -> tuple[float, ...]:
        values = (self.background, *self.particle_amplitudes)
        if self.resolution_present:
            values += (self.resolution_amplitude,)
        return tuple(float(value) for value in values)

    @property
    def coefficient_fractions(self) -> tuple[float, ...]:
        values = np.asarray(self.coefficient_vector, dtype=np.float64)
        return tuple(float(value) for value in values / np.sum(values))

    def audit_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "generator_version": self.generator_version,
            "regime": self.regime,
            "seed": self.seed,
            "component_count": self.component_count,
            "resolution_present": self.resolution_present,
            "background": self.background,
            "k": self.k,
            "particle_amplitudes": list(self.particle_amplitudes),
            "resolution_amplitude": self.resolution_amplitude,
            "component_intensities": list(self.particle_weights),
            "int_res": self.int_res,
            "effective_particle_fractions": list(self.effective_particle_fractions),
            "selected_particle_slot": self.selected_particle_slot,
            "coefficient_fraction_is_observability": False,
        }


def _balanced_weights(rng: np.random.Generator, count: int) -> np.ndarray:
    floor = 0.02
    remaining = 1.0 - count * floor
    return floor + remaining * rng.dirichlet(np.full(count, 1.5))


def _weak_weights(rng: np.random.Generator, count: int) -> tuple[np.ndarray, int | None]:
    if count == 1:
        return np.ones(1, dtype=np.float64), 0
    slot = int(rng.integers(0, count))
    weak = _log_uniform(rng, 1.0e-5, 3.0e-2)
    remaining = rng.dirichlet(np.full(count - 1, 1.2)) * (1.0 - weak)
    weights = np.empty(count, dtype=np.float64)
    weights[slot] = weak
    weights[np.arange(count) != slot] = remaining
    return weights, slot


def _dominant_weights(rng: np.random.Generator, count: int) -> tuple[np.ndarray, int | None]:
    if count == 1:
        return np.ones(1, dtype=np.float64), 0
    slot = int(rng.integers(0, count))
    dominant = float(rng.uniform(0.80, 0.995))
    remaining = rng.dirichlet(np.ones(count - 1)) * (1.0 - dominant)
    weights = np.empty(count, dtype=np.float64)
    weights[slot] = dominant
    weights[np.arange(count) != slot] = remaining
    return weights, slot


def sample_v5_amplitude_composition(
    component_count: int,
    *,
    resolution_present: bool,
    seed: int,
    regime: str | None = None,
) -> V5AmplitudeComposition:
    """Draw one deterministic composition without imposing a 15% floor."""

    count = _positive_integer(component_count, "component_count")
    if count > 4:
        raise ValueError("component_count must be at most four")
    if not isinstance(resolution_present, (bool, np.bool_)):
        raise TypeError("resolution_present must be boolean")
    selected_seed = _non_negative_integer(seed, "seed")
    allowed = tuple(
        value
        for value in V5_AMPLITUDE_REGIMES
        if resolution_present or not value.startswith("resolution_")
    )
    rng = np.random.default_rng(
        np.random.SeedSequence([selected_seed, count, int(resolution_present), 0x5635414D])
    )
    selected_regime = allowed[selected_seed % len(allowed)] if regime is None else regime
    if selected_regime not in V5_AMPLITUDE_REGIMES:
        raise ValueError(f"regime must be one of {allowed} for this branch")
    if selected_regime not in allowed:
        raise ValueError(f"regime must be one of {allowed} for this branch")

    selected_slot: int | None = None
    if selected_regime == "weak_particle_coefficient":
        weights, selected_slot = _weak_weights(rng, count)
    elif selected_regime == "dominant_particle_coefficient":
        weights, selected_slot = _dominant_weights(rng, count)
    else:
        weights = _balanced_weights(rng, count)

    # A separate total-Int scale prevents the unconstrained engineering
    # sampler from silently teaching only the unit-sum gauge.
    intensity_scale = _log_uniform(rng, 5.0e-2, 2.0e1)
    intensities = weights * intensity_scale
    k = _log_uniform(rng, 1.0e1, 1.0e7)
    particles = tuple(float(k * value) for value in intensities)
    if selected_regime == "background_confounded":
        background_ratio = _log_uniform(rng, 5.0e-2, 2.0e1)
    elif selected_regime == "weak_particle_coefficient" and count == 1:
        background_ratio = _log_uniform(rng, 2.0, 1.0e2)
    else:
        background_ratio = _log_uniform(rng, 1.0e-8, 1.0e-2)
    background = k * background_ratio

    if not resolution_present:
        resolution_amplitude = 0.0
    elif selected_regime == "resolution_weak_coefficient":
        resolution_amplitude = k * _log_uniform(rng, 1.0e-6, 1.0e-3)
    elif selected_regime == "resolution_confounded":
        resolution_amplitude = k * _log_uniform(rng, 1.0e-1, 1.0e1)
    else:
        resolution_amplitude = k * _log_uniform(rng, 1.0e-4, 3.0e-1)

    return V5AmplitudeComposition(
        component_count=count,
        resolution_present=bool(resolution_present),
        regime=selected_regime,
        seed=selected_seed,
        background=background,
        k=k,
        particle_amplitudes=particles,
        resolution_amplitude=resolution_amplitude,
        selected_particle_slot=selected_slot,
    )


def _sample_interval(
    rng: np.random.Generator,
    low: float,
    high: float,
    *,
    allow_zero: bool,
) -> float:
    if high < low or low < 0.0 or not np.all(np.isfinite((low, high))):
        raise ValueError("sampling interval must be finite and non-negative")
    if high == low:
        return float(low)
    if allow_zero and low == 0.0 and rng.random() < 0.05:
        return 0.0
    positive_low = max(low, high * 1.0e-12, np.finfo(np.float64).tiny)
    if positive_low >= high:
        return float(rng.uniform(low, high))
    return _log_uniform(rng, positive_low, high)


def _fraction_extrema(low: np.ndarray, high: np.ndarray, index: int) -> tuple[float, float]:
    rest = np.arange(low.size) != index
    minimum_denominator = float(low[index] + np.sum(high[rest]))
    maximum_denominator = float(high[index] + np.sum(low[rest]))
    minimum = 0.0 if minimum_denominator == 0.0 else float(low[index] / minimum_denominator)
    maximum = 0.0 if maximum_denominator == 0.0 else float(high[index] / maximum_denominator)
    return minimum, maximum


def _feasible_constrained_regimes(
    constraint: GuiAmplitudeConstraint,
) -> tuple[tuple[str, ...], dict[str, tuple[int, ...]]]:
    low = np.asarray([value.low for value in constraint.component_intensities])
    high = np.asarray([value.high for value in constraint.component_intensities])
    weak, dominant = [], []
    for index in range(low.size):
        minimum, maximum = _fraction_extrema(low, high, index)
        if minimum < 0.03 and float(np.sum(high) - high[index]) > 0.0:
            weak.append(index)
        if maximum > 0.8:
            dominant.append(index)
    feasible = ["balanced_particles"]
    if weak:
        feasible.append("weak_particle_coefficient")
    if dominant:
        feasible.append("dominant_particle_coefficient")
    if constraint.background.high > 0.0 and (
        max(constraint.k.low, constraint.background.low / 20.0)
        <= min(constraint.k.high, constraint.background.high / 0.05)
    ):
        feasible.append("background_confounded")
    if constraint.resolution_present:
        assert constraint.int_res is not None
        if max(constraint.int_res.low, 1.0e-6) <= min(constraint.int_res.high, 1.0e-3):
            feasible.append("resolution_weak_coefficient")
        if max(constraint.int_res.low, 1.0e-1) <= min(constraint.int_res.high, 1.0e1):
            feasible.append("resolution_confounded")
    return (
        tuple(value for value in V5_AMPLITUDE_REGIMES if value in feasible),
        {"weak": tuple(weak), "dominant": tuple(dominant)},
    )


def _sample_independent_intensities(
    rng: np.random.Generator,
    constraint: GuiAmplitudeConstraint,
    regime: str,
    feasible_slots: dict[str, tuple[int, ...]],
) -> tuple[np.ndarray, int | None]:
    low = np.asarray([value.low for value in constraint.component_intensities], dtype=np.float64)
    high = np.asarray([value.high for value in constraint.component_intensities], dtype=np.float64)
    count = low.size
    selected: int | None = None
    if regime in {"weak_particle_coefficient", "dominant_particle_coefficient"}:
        key = "weak" if regime.startswith("weak") else "dominant"
        slots = feasible_slots[key]
        if not slots:
            raise ValueError(f"{key}-particle regime is infeasible in the supplied Int ranges")
        selected = slots[int(rng.integers(0, len(slots)))]
    intensities = np.empty(count, dtype=np.float64)
    for index in range(count):
        if index != selected:
            intensities[index] = _sample_interval(
                rng, low[index], high[index], allow_zero=True
            )
    if selected is not None:
        others = float(np.sum(intensities[np.arange(count) != selected]))
        if regime == "weak_particle_coefficient":
            selected_low = low[selected]
            selected_high = min(
                high[selected],
                float(np.nextafter(0.03, 0.0)) * others / (1.0 - 0.03),
            )
        else:
            selected_low = max(
                low[selected],
                float(np.nextafter(0.8, 1.0)) * others / (1.0 - 0.8),
            )
            selected_high = high[selected]
        if selected_high < selected_low:
            # Re-sampling the other independent axes at their favorable edge
            # deterministically preserves the requested diagnostic stratum.
            rest = np.arange(count) != selected
            intensities[rest] = high[rest] if regime.startswith("weak") else low[rest]
            others = float(np.sum(intensities[rest]))
            if regime.startswith("weak"):
                selected_high = min(
                    high[selected],
                    float(np.nextafter(0.03, 0.0)) * others / (1.0 - 0.03),
                )
            else:
                selected_low = max(
                    low[selected],
                    float(np.nextafter(0.8, 1.0)) * others / (1.0 - 0.8),
                )
        if selected_high < selected_low:
            raise RuntimeError("independent Int sampler lost its feasible diagnostic stratum")
        intensities[selected] = _sample_interval(
            rng, selected_low, selected_high, allow_zero=True
        )
    if float(np.sum(intensities)) <= 0.0:
        anchor = int(np.argmax(high))
        intensities[anchor] = max(low[anchor], min(high[anchor], np.finfo(float).tiny))
    if np.any(intensities < low) or np.any(intensities > high):
        raise RuntimeError("independent Int sampler escaped its query box")
    return intensities, selected


def sample_v5_constrained_amplitude_composition(
    constraint: GuiAmplitudeConstraint,
    *,
    seed: int,
    regime: str | None = None,
) -> V5AmplitudeComposition:
    """Draw one composition strictly inside a pre-existing physical query.

    The query must already exist; this function never constructs ranges around
    the sampled composition.  Requested diagnostic strata fail explicitly
    when the user's ranges make that stratum impossible.
    """

    if not isinstance(constraint, GuiAmplitudeConstraint):
        raise TypeError("constraint must be a GuiAmplitudeConstraint")
    selected_seed = _non_negative_integer(seed, "seed")
    allowed, feasible_slots = _feasible_constrained_regimes(constraint)
    selected_regime = allowed[selected_seed % len(allowed)] if regime is None else regime
    if selected_regime not in V5_AMPLITUDE_REGIMES:
        raise ValueError(f"regime must be one of {allowed} for this branch")
    if selected_regime not in allowed:
        raise ValueError(f"requested {selected_regime} regime is infeasible in the supplied ranges")
    rng = np.random.default_rng(
        np.random.SeedSequence(
            [selected_seed, constraint.particle_count, int(constraint.resolution_present), 0x56354341]
        )
    )
    intensities, selected_slot = _sample_independent_intensities(
        rng, constraint, selected_regime, feasible_slots
    )

    k = _sample_interval(rng, constraint.k.low, constraint.k.high, allow_zero=False)
    if k <= 0.0:
        raise RuntimeError("constrained sampler selected invalid shared GUI k=0")
    particles = tuple(float(k * value) for value in intensities)

    if selected_regime == "background_confounded":
        bg_low = max(constraint.background.low, 0.05 * k)
        bg_high = min(constraint.background.high, 20.0 * k)
        if bg_high < bg_low:
            raise ValueError("background-confounded regime is infeasible in the supplied ranges")
        background = _sample_interval(rng, bg_low, bg_high, allow_zero=False)
    else:
        background = _sample_interval(
            rng,
            constraint.background.low,
            constraint.background.high,
            allow_zero=True,
        )

    if not constraint.resolution_present:
        resolution_amplitude = 0.0
    else:
        assert constraint.int_res is not None
        int_low, int_high = constraint.int_res.low, constraint.int_res.high
        if selected_regime == "resolution_weak_coefficient":
            int_low, int_high = max(int_low, 1.0e-6), min(int_high, 1.0e-3)
        elif selected_regime == "resolution_confounded":
            int_low, int_high = max(int_low, 1.0e-1), min(int_high, 1.0e1)
        if int_high < int_low:
            raise ValueError("requested Resolution regime is infeasible in the supplied ranges")
        int_res = _sample_interval(rng, int_low, int_high, allow_zero=True)
        resolution_amplitude = k * int_res

    result = V5AmplitudeComposition(
        component_count=constraint.particle_count,
        resolution_present=constraint.resolution_present,
        regime=selected_regime,
        seed=selected_seed,
        background=background,
        k=k,
        particle_amplitudes=particles,
        resolution_amplitude=resolution_amplitude,
        selected_particle_slot=selected_slot,
    )
    if not constraint.contains(result.coefficient_vector, k=result.k, atol=2.0e-9):
        raise RuntimeError("constrained amplitude composition escaped the physical query")
    return result


__all__ = [
    "V5_AMPLITUDE_REGIMES",
    "V5_AMPLITUDE_SAMPLING_SCHEMA",
    "V5_AMPLITUDE_SAMPLING_VERSION",
    "V5AmplitudeComposition",
    "sample_v5_amplitude_composition",
    "sample_v5_constrained_amplitude_composition",
]
