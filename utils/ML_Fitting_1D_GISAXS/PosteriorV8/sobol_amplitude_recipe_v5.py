"""Direct amplitude-query and constrained-composition Sobol transforms for V5.2."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np

from .amplitude_query_sampling_v5 import (
    V5_AMPLITUDE_RANGE_REGIMES,
    V5_BACKGROUND_DOMAIN,
    V5_COMPONENT_INTENSITY_DOMAIN,
    V5_INT_RES_DOMAIN,
    V5_K_DOMAIN,
    V5AmplitudeRangeRegimes,
)
from .amplitude_query_v5 import V5AmplitudeQuery
from .amplitude_sampling_v5 import V5_AMPLITUDE_REGIMES
from .bounds_query_v5 import V5BoundsQuery
from .contract import MAX_COMPONENTS, ClosedInterval
from .gui_amplitude_constraints import GuiAmplitudeConstraint
from .sobol_recipe_coordinates_v5 import V5SobolCoordinateReader
from .sobol_numeric_canonicalization_v5 import (
    V5_DETERMINISTIC_NUMERIC_POLICY_VERSION,
    v5_numeric_ops,
)


V5_DIRECT_AMPLITUDE_SCHEMA = "gisaxs.posterior_v8.direct_sobol_amplitude_composition/v6"
V5_DIRECT_AMPLITUDE_VERSION = (
    "posterior_v8_numeric_contract_per_axis_query_contained_gui_amplitude_sobol_map_v6"
)
_POSITIVE_AMPLITUDE_FLOOR = 1.0e-18
_DIRECT_NUMERIC = v5_numeric_ops(V5_DETERMINISTIC_NUMERIC_POLICY_VERSION)


def _nonnegative_integer(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _choose(unit: float, values: tuple[str, ...]) -> str:
    return values[min(int(unit * len(values)), len(values) - 1)]


def _interpolate(low: float, high: float, unit: float) -> float:
    if low == high:
        return float(low)
    return float(low + unit * (high - low))


def _log_interpolate(low: float, high: float, unit: float) -> float:
    if low <= 0.0 or high <= 0.0:
        raise ValueError("log interpolation requires positive endpoints")
    if low == high:
        return float(low)
    value = _DIRECT_NUMERIC.exp(
        _interpolate(_DIRECT_NUMERIC.log(low), _DIRECT_NUMERIC.log(high), unit)
    )
    return min(high, max(low, value))


def _amplitude_interval(
    reader: V5SobolCoordinateReader,
    prefix: str,
    domain: ClosedInterval,
    regime: str,
) -> ClosedInterval:
    if regime == "full":
        return domain
    positive_low = _POSITIVE_AMPLITUDE_FLOOR if domain.low == 0.0 else domain.low
    log_low = _DIRECT_NUMERIC.log(positive_low)
    log_high = _DIRECT_NUMERIC.log(domain.high)
    if regime == "fixed":
        position = 0.02 + 0.96 * reader.take(f"{prefix}.position")
        start = stop = position
    else:
        width_unit = reader.take(f"{prefix}.width")
        width = 0.40 + 0.40 * width_unit if regime == "wide" else 0.03 + 0.15 * width_unit
        available = 1.0 - width
        if regime == "edge_low":
            start = 0.0
        elif regime == "edge_high":
            start = available
        else:
            start = available * reader.take(f"{prefix}.position")
        stop = start + width
    low = _DIRECT_NUMERIC.exp(_interpolate(log_low, log_high, start))
    high = _DIRECT_NUMERIC.exp(_interpolate(log_low, log_high, stop))
    if regime == "edge_low":
        low = domain.low
    if regime == "edge_high":
        high = domain.high
    return ClosedInterval(max(domain.low, low), min(domain.high, high))


def _direct_range_assignment(
    reader: V5SobolCoordinateReader,
    query: V5BoundsQuery,
) -> V5AmplitudeRangeRegimes:
    count = len(query.topology)
    policy = query.resolution_presence_policy

    def active_regime(axis: str) -> str:
        return _choose(
            reader.take(f"amplitude.query.{axis}.regime"),
            V5_AMPLITUDE_RANGE_REGIMES,
        )

    return V5AmplitudeRangeRegimes.create(
        count,
        resolution_presence_policy=policy,
        background=active_regime("BG"),
        k=active_regime("k"),
        component_intensities=tuple(
            active_regime(f"Int_{slot + 1}") for slot in range(count)
        ),
        int_res=None if policy == "absent" else active_regime("int_Res"),
    )


def direct_v5_amplitude_query(
    reader: V5SobolCoordinateReader,
    query: V5BoundsQuery,
) -> tuple[V5AmplitudeQuery, V5AmplitudeRangeRegimes]:
    """Create the complete amplitude query before selecting a branch."""

    regimes = _direct_range_assignment(reader, query)
    policy = query.resolution_presence_policy
    return (
        V5AmplitudeQuery.create(
            background=_amplitude_interval(
                reader,
                "amplitude.query.BG",
                V5_BACKGROUND_DOMAIN,
                regimes.background,
            ),
            k=_amplitude_interval(
                reader,
                "amplitude.query.k",
                V5_K_DOMAIN,
                regimes.k,
            ),
            component_intensities=tuple(
                _amplitude_interval(
                    reader,
                    f"amplitude.query.Int_{slot + 1}",
                    V5_COMPONENT_INTENSITY_DOMAIN,
                    regime,
                )
                for slot, regime in enumerate(regimes.active_component_intensities)
            ),
            resolution_presence_policy=policy,
            int_res=(
                None
                if policy == "absent"
                else _amplitude_interval(
                    reader,
                    "amplitude.query.int_Res",
                    V5_INT_RES_DOMAIN,
                    regimes.int_res,
                )
            ),
            numeric_policy_version=V5_DETERMINISTIC_NUMERIC_POLICY_VERSION,
        ),
        regimes,
    )


def _fraction_extrema(
    low: np.ndarray,
    high: np.ndarray,
    index: int,
) -> tuple[float, float]:
    rest = np.arange(low.size) != index
    minimum_denominator = float(low[index] + np.sum(high[rest]))
    maximum_denominator = float(high[index] + np.sum(low[rest]))
    return (
        0.0 if minimum_denominator == 0.0 else float(low[index] / minimum_denominator),
        0.0 if maximum_denominator == 0.0 else float(high[index] / maximum_denominator),
    )


def _background_k_limits(constraint: GuiAmplitudeConstraint) -> tuple[float, float] | None:
    if constraint.background.high <= 0.0:
        return None
    low = max(constraint.k.low, constraint.background.low / 20.0)
    high = min(constraint.k.high, constraint.background.high / 0.05)
    return None if high < low else (low, high)


def _feasible_amplitude_regimes(
    constraint: GuiAmplitudeConstraint,
) -> tuple[tuple[str, ...], dict[str, tuple[int, ...]]]:
    low = np.asarray([value.low for value in constraint.component_intensities])
    high = np.asarray([value.high for value in constraint.component_intensities])
    weak = []
    dominant = []
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
    if _background_k_limits(constraint) is not None:
        feasible.append("background_confounded")
    if constraint.resolution_present:
        assert constraint.int_res is not None
        if max(constraint.int_res.low, 1.0e-6) <= min(constraint.int_res.high, 1.0e-3):
            feasible.append("resolution_weak_coefficient")
        if max(constraint.int_res.low, 1.0e-1) <= min(constraint.int_res.high, 1.0e1):
            feasible.append("resolution_confounded")
    ordered = tuple(value for value in V5_AMPLITUDE_REGIMES if value in feasible)
    return ordered, {"weak": tuple(weak), "dominant": tuple(dominant)}


def _nonnegative_log_sample(low: float, high: float, unit: float) -> float:
    if high < low or low < 0.0:
        raise ValueError("non-negative sampling interval is invalid")
    if high == low:
        return float(low)
    if low == 0.0:
        if unit < 0.05:
            return 0.0
        unit = (unit - 0.05) / 0.95
        low = max(_POSITIVE_AMPLITUDE_FLOOR, high * 1.0e-12)
    # ``exp(log(high))`` can exceed ``high`` by one ulp near the half-open
    # Sobol boundary.  Clipping preserves the exact closed GUI interval.
    return min(high, max(low, _log_interpolate(low, high, unit)))


def _direct_independent_intensities(
    reader: V5SobolCoordinateReader,
    constraint: GuiAmplitudeConstraint,
    regime: str,
    feasible_slots: dict[str, tuple[int, ...]],
) -> tuple[np.ndarray, int | None]:
    low = np.asarray([value.low for value in constraint.component_intensities], dtype=np.float64)
    high = np.asarray([value.high for value in constraint.component_intensities], dtype=np.float64)
    selected = None
    if regime in {"weak_particle_coefficient", "dominant_particle_coefficient"}:
        key = "weak" if regime.startswith("weak") else "dominant"
        slots = feasible_slots[key]
        selected = slots[
            min(
                int(
                    reader.take("amplitude.composition.selected_particle_within_feasible_set")
                    * len(slots)
                ),
                len(slots) - 1,
            )
        ]

    count = constraint.particle_count
    rotation = min(int(reader.take("amplitude.composition.intensity_order") * count), count - 1)
    order = list(range(rotation, count)) + list(range(rotation))
    if selected is not None:
        order.remove(selected)
        order.append(selected)
    intensities = np.zeros(count, dtype=np.float64)
    for position, index in enumerate(order):
        unit = reader.take(f"amplitude.composition.Int_fraction_{position + 1}")
        if index != selected:
            value = _nonnegative_log_sample(low[index], high[index], unit)
        else:
            others = float(np.sum(intensities))
            if regime == "weak_particle_coefficient":
                minimum = low[index]
                maximum = min(
                    high[index],
                    float(np.nextafter(0.03, 0.0)) * others / (1.0 - 0.03),
                )
            else:
                minimum = max(
                    low[index],
                    float(np.nextafter(0.8, 1.0)) * others / (1.0 - 0.8),
                )
                maximum = high[index]
            if maximum < minimum:
                rest = np.arange(count) != index
                intensities[rest] = high[rest] if regime.startswith("weak") else low[rest]
                others = float(np.sum(intensities[rest]))
                if regime.startswith("weak"):
                    maximum = min(
                        high[index],
                        float(np.nextafter(0.03, 0.0)) * others / (1.0 - 0.03),
                    )
                else:
                    minimum = max(
                        low[index],
                        float(np.nextafter(0.8, 1.0)) * others / (1.0 - 0.8),
                    )
            if maximum < minimum:
                raise RuntimeError("direct Int coordinate lost its feasible diagnostic stratum")
            value = _nonnegative_log_sample(minimum, maximum, unit)
        intensities[index] = value
    if float(np.sum(intensities)) <= 0.0:
        anchor = int(np.argmax(high))
        intensities[anchor] = max(low[anchor], min(high[anchor], np.finfo(float).tiny))
    if np.any(intensities < low) or np.any(intensities > high):
        raise RuntimeError("direct independent Int coordinate escaped its query box")
    return intensities, selected


@dataclass(frozen=True)
class V5DirectAmplitudeComposition:
    component_count: int
    resolution_present: bool
    regime: str
    background: float
    k: float
    component_intensities: tuple[float, ...]
    resolution_intensity: float
    selected_particle_slot: int | None
    schema_version: str = V5_DIRECT_AMPLITUDE_SCHEMA
    generator_version: str = V5_DIRECT_AMPLITUDE_VERSION

    def __post_init__(self) -> None:
        count = _nonnegative_integer(self.component_count, "component_count")
        if not 1 <= count <= MAX_COMPONENTS:
            raise ValueError("component_count must be in [1, 4]")
        if type(self.resolution_present) is not bool:
            raise TypeError("resolution_present must be a bool")
        if self.regime not in V5_AMPLITUDE_REGIMES:
            raise ValueError("unsupported amplitude regime")
        intensities = tuple(float(value) for value in self.component_intensities)
        values = np.asarray(
            [self.background, self.k, *intensities, self.resolution_intensity],
            dtype=np.float64,
        )
        if (
            len(intensities) != count
            or not np.all(np.isfinite(values))
            or np.any(values < 0.0)
            or self.k <= 0.0
        ):
            raise ValueError("direct GUI amplitude parameters are invalid")
        if not any(value > 0.0 for value in intensities):
            raise ValueError("at least one component intensity must be positive")
        if not self.resolution_present and self.resolution_intensity != 0.0:
            raise ValueError("Resolution-absent composition must have zero int_Res")
        if self.regime.startswith("resolution_") and not self.resolution_present:
            raise ValueError("Resolution amplitude regime requires Resolution presence")
        slot = self.selected_particle_slot
        if slot is not None and (
            isinstance(slot, (bool, np.bool_))
            or not isinstance(slot, Integral)
            or not 0 <= int(slot) < count
        ):
            raise ValueError("selected_particle_slot is invalid")
        if self.schema_version != V5_DIRECT_AMPLITUDE_SCHEMA:
            raise ValueError("unsupported direct amplitude schema")
        if self.generator_version != V5_DIRECT_AMPLITUDE_VERSION:
            raise ValueError("unsupported direct amplitude generator")
        object.__setattr__(self, "component_count", count)
        object.__setattr__(self, "component_intensities", intensities)
        object.__setattr__(self, "background", float(self.background))
        object.__setattr__(self, "k", float(self.k))
        object.__setattr__(self, "resolution_intensity", float(self.resolution_intensity))
        object.__setattr__(self, "selected_particle_slot", None if slot is None else int(slot))

    @property
    def particle_weights(self) -> tuple[float, ...]:
        """Compatibility name for independent GUI ``Int_i`` values."""
        return self.component_intensities

    @property
    def particle_amplitudes(self) -> tuple[float, ...]:
        return tuple(float(self.k * value) for value in self.component_intensities)

    @property
    def effective_particle_fractions(self) -> tuple[float, ...]:
        total = float(sum(self.component_intensities))
        return tuple(float(value / total) for value in self.component_intensities)

    @property
    def int_res(self) -> float:
        return 0.0 if not self.resolution_present else self.resolution_intensity

    @property
    def resolution_amplitude(self) -> float:
        return float(self.k * self.int_res)

    @property
    def coefficient_vector(self) -> tuple[float, ...]:
        values = (self.background, *self.particle_amplitudes)
        if self.resolution_present:
            values += (self.resolution_amplitude,)
        return tuple(float(value) for value in values)

    def audit_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "generator_version": self.generator_version,
            "coordinate_consumption": "direct_no_PRNG",
            "regime": self.regime,
            "component_count": self.component_count,
            "resolution_present": self.resolution_present,
            "background": self.background,
            "k": self.k,
            "particle_amplitudes": list(self.particle_amplitudes),
            "resolution_amplitude": self.resolution_amplitude,
            "component_intensities": list(self.component_intensities),
            "int_res": self.int_res,
            "effective_particle_fractions": list(self.effective_particle_fractions),
            "selected_particle_slot": self.selected_particle_slot,
            "coefficient_fraction_is_observability": False,
        }


def direct_v5_amplitude_composition(
    reader: V5SobolCoordinateReader,
    constraint: GuiAmplitudeConstraint,
) -> tuple[V5DirectAmplitudeComposition, tuple[str, ...]]:
    """Select a feasible stratum, then map coordinates inside the exact polytope."""

    feasible_regimes, feasible_slots = _feasible_amplitude_regimes(constraint)
    regime = _choose(
        reader.take("amplitude.composition.regime_within_feasible_set"), feasible_regimes
    )
    intensities, selected = _direct_independent_intensities(
        reader, constraint, regime, feasible_slots
    )
    if regime == "background_confounded":
        k_limits = _background_k_limits(constraint)
        assert k_limits is not None
    else:
        k_limits = (constraint.k.low, constraint.k.high)
    k = _log_interpolate(k_limits[0], k_limits[1], reader.take("amplitude.composition.k"))
    if regime == "background_confounded":
        bg_low = max(constraint.background.low, 0.05 * k)
        bg_high = min(constraint.background.high, 20.0 * k)
    else:
        bg_low, bg_high = constraint.background.low, constraint.background.high
    background = _nonnegative_log_sample(bg_low, bg_high, reader.take("amplitude.composition.BG"))
    if constraint.resolution_present:
        assert constraint.int_res is not None
        int_low, int_high = constraint.int_res.low, constraint.int_res.high
        if regime == "resolution_weak_coefficient":
            int_low, int_high = max(int_low, 1.0e-6), min(int_high, 1.0e-3)
        elif regime == "resolution_confounded":
            int_low, int_high = max(int_low, 1.0e-1), min(int_high, 1.0e1)
        int_res = _nonnegative_log_sample(
            int_low,
            int_high,
            reader.take("amplitude.composition.int_Res"),
        )
    else:
        int_res = 0.0
    result = V5DirectAmplitudeComposition(
        component_count=constraint.particle_count,
        resolution_present=constraint.resolution_present,
        regime=regime,
        background=background,
        k=k,
        # GUI Int_i/int_Res are authoritative.  Effective coefficients remain
        # derived so fixed one-point ranges never depend on a multiply/divide
        # round trip.
        component_intensities=tuple(float(value) for value in intensities),
        resolution_intensity=float(int_res),
        selected_particle_slot=selected,
    )
    if not constraint.background.low <= result.background <= constraint.background.high:
        raise RuntimeError("direct background escaped its query interval")
    if not constraint.k.low <= result.k <= constraint.k.high:
        raise RuntimeError("direct k escaped its query interval")
    if any(
        not interval.low <= value <= interval.high
        for interval, value in zip(
            constraint.component_intensities,
            result.component_intensities,
        )
    ):
        raise RuntimeError("direct component intensity escaped its query interval")
    if constraint.resolution_present:
        assert constraint.int_res is not None
        if not constraint.int_res.low <= result.int_res <= constraint.int_res.high:
            raise RuntimeError("direct int_Res escaped its query interval")
    if not constraint.contains(result.coefficient_vector, k=result.k, atol=2.0e-9):
        raise RuntimeError("direct amplitude composition escaped its query constraint")
    return result, feasible_regimes


__all__ = [
    "V5_DIRECT_AMPLITUDE_SCHEMA",
    "V5_DIRECT_AMPLITUDE_VERSION",
    "V5DirectAmplitudeComposition",
    "direct_v5_amplitude_composition",
    "direct_v5_amplitude_query",
]
