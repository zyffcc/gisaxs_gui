"""Exact projection of independent GUI amplitude ranges.

For fixed nonlinear geometry the forward is linear in effective coefficients
``[BG, a_1, ..., a_K, optional a_res]``.  The GUI parameters are not a
simplex: for one shared positive auxiliary scale ``kappa`` they satisfy
``a_i = kappa * Int_i`` and ``a_res = kappa * int_Res``.  This module keeps
the *existence* of such a shared ``kappa`` when the linear solver works only
with effective coefficients, and reconstructs a legal GUI parameterization
afterwards.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from numbers import Integral
from typing import Sequence

import numpy as np

from .contract import MAX_COMPONENTS, ClosedInterval


GUI_AMPLITUDE_CONSTRAINT_SCHEMA = "gisaxs.posterior_v8.gui_amplitude_constraint/v2"
COEFFICIENT_POLYTOPE_SCHEMA = "gisaxs.posterior_v8.gui_amplitude_coefficient_polytope/v2"
GUI_AMPLITUDE_CONSTRAINT_VERSION = (
    "posterior_v8_independent_gui_k_int_projected_auxiliary_kappa_polytope_v2"
)
CANONICAL_AMPLITUDE_GAUGE = (
    "exists kappa in requested positive k interval; a_i=kappa*Int_i; "
    "a_res=kappa*int_Res; BG=coefficient_BG; output uses a deterministic "
    "feasible kappa witness"
)


def _nonnegative_tolerance(value: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("atol must be finite and non-negative") from exc
    if not np.isfinite(result) or result < 0.0:
        raise ValueError("atol must be finite and non-negative")
    return result


def _finite_nonnegative(value: float, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite and non-negative") from exc
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _roundoff_tolerance(*values: float, atol: float) -> float:
    """Add only floating-point roundoff relative to the compared scale."""

    scale = max(1.0, *(abs(float(value)) for value in values))
    return max(float(atol), 64.0 * np.finfo(np.float64).eps * scale)


def _interval(value: ClosedInterval, name: str) -> ClosedInterval:
    if not isinstance(value, ClosedInterval):
        raise TypeError(f"{name} must be a ClosedInterval")
    if value.low < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return value


@dataclass(frozen=True)
class GuiRangeCheck:
    name: str
    value: float
    low: float
    high: float
    satisfied: bool


@dataclass(frozen=True)
class GuiAmplitudeConstraintAudit:
    """Per-range audit of one coefficient vector."""

    schema: str
    version: str
    canonical_gauge: str
    coefficient_order: tuple[str, ...]
    coefficients: tuple[float, ...]
    auxiliary_k: float
    auxiliary_k_feasible_interval: tuple[float, float] | None
    auxiliary_k_is_witness: bool
    range_checks: tuple[GuiRangeCheck, ...]
    coefficient_polytope_satisfied: bool
    particle_total_positive: bool
    all_constraints_satisfied: bool

    def to_audit_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class CoefficientPolytope:
    """Exact coefficient projection of a shared-``kappa`` GUI box.

    ``inequality_matrix`` contains the Fourier--Motzkin projection of the
    auxiliary constraints.  ``feasible_k_interval`` reconstructs the retained
    auxiliary variable for any coefficient vector in that projection.
    """

    particle_count: int
    resolution_present: bool
    coefficient_order: tuple[str, ...]
    axis_lower: tuple[float, ...]
    axis_upper: tuple[float, ...]
    inequality_labels: tuple[str, ...]
    inequality_matrix: tuple[tuple[float, ...], ...]
    inequality_upper: tuple[float, ...]
    feasible_coefficients: tuple[float, ...]
    auxiliary_k_lower: float
    auxiliary_k_upper: float
    amplitude_ratio_names: tuple[str, ...]
    amplitude_ratio_lower: tuple[float, ...]
    amplitude_ratio_upper: tuple[float, ...]
    feasible_auxiliary_k: float
    schema: str = COEFFICIENT_POLYTOPE_SCHEMA
    version: str = GUI_AMPLITUDE_CONSTRAINT_VERSION
    canonical_gauge: str = CANONICAL_AMPLITUDE_GAUGE

    def __post_init__(self) -> None:
        if self.schema != COEFFICIENT_POLYTOPE_SCHEMA:
            raise ValueError("unsupported coefficient-polytope schema")
        if self.version != GUI_AMPLITUDE_CONSTRAINT_VERSION:
            raise ValueError("unsupported GUI amplitude constraint version")
        if self.canonical_gauge != CANONICAL_AMPLITUDE_GAUGE:
            raise ValueError("unexpected canonical amplitude gauge")
        if isinstance(self.particle_count, (bool, np.bool_)) or not isinstance(
            self.particle_count, Integral
        ):
            raise TypeError("particle_count must be an integer")
        count = int(self.particle_count)
        if not 1 <= count <= MAX_COMPONENTS:
            raise ValueError(f"particle_count must be in [1, {MAX_COMPONENTS}]")
        if type(self.resolution_present) is not bool:
            raise TypeError("resolution_present must be a bool")
        expected_order = ("BG",) + tuple(f"a_{index}" for index in range(1, count + 1))
        if self.resolution_present:
            expected_order += ("a_res",)
        order = tuple(self.coefficient_order)
        if order != expected_order:
            raise ValueError("coefficient_order does not match the canonical branch gauge")
        lower = np.asarray(self.axis_lower, dtype=np.float64)
        upper = np.asarray(self.axis_upper, dtype=np.float64)
        if lower.shape != (len(order),) or upper.shape != lower.shape:
            raise ValueError("coefficient axis bounds have the wrong shape")
        if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
            raise ValueError("coefficient axis bounds must be finite")
        if np.any(lower < 0.0) or np.any(upper < lower):
            raise ValueError("coefficient axis bounds are invalid")
        matrix = np.asarray(self.inequality_matrix, dtype=np.float64)
        if matrix.size == 0:
            matrix = np.empty((0, len(order)), dtype=np.float64)
        rhs = np.asarray(self.inequality_upper, dtype=np.float64)
        labels = tuple(str(value) for value in self.inequality_labels)
        if matrix.ndim != 2 or matrix.shape[1] != len(order):
            raise ValueError("coefficient inequality matrix has the wrong shape")
        if rhs.shape != (matrix.shape[0],) or len(labels) != matrix.shape[0]:
            raise ValueError("coefficient inequality labels/RHS have the wrong length")
        if any(not value for value in labels) or len(set(labels)) != len(labels):
            raise ValueError("coefficient inequality labels must be unique and non-empty")
        if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(rhs)):
            raise ValueError("coefficient inequalities must be finite")
        k_low = _finite_nonnegative(self.auxiliary_k_lower, "auxiliary_k_lower")
        k_high = _finite_nonnegative(self.auxiliary_k_upper, "auxiliary_k_upper")
        if k_high < k_low or k_high <= 0.0:
            raise ValueError("auxiliary k interval must contain a positive value")
        ratio_names = tuple(str(value) for value in self.amplitude_ratio_names)
        expected_ratio_names = tuple(f"Int_{index}" for index in range(1, count + 1))
        if self.resolution_present:
            expected_ratio_names += ("int_Res",)
        if ratio_names != expected_ratio_names:
            raise ValueError("amplitude ratio names do not match the branch")
        ratio_low = np.asarray(self.amplitude_ratio_lower, dtype=np.float64)
        ratio_high = np.asarray(self.amplitude_ratio_upper, dtype=np.float64)
        if ratio_low.shape != (len(ratio_names),) or ratio_high.shape != ratio_low.shape:
            raise ValueError("amplitude ratio bounds have the wrong shape")
        if (
            not np.all(np.isfinite(ratio_low))
            or not np.all(np.isfinite(ratio_high))
            or np.any(ratio_low < 0.0)
            or np.any(ratio_high < ratio_low)
        ):
            raise ValueError("amplitude ratio bounds are invalid")
        if not np.any(ratio_high[:count] > 0.0):
            raise ValueError("at least one particle Int range must contain a positive value")
        feasible_k = _finite_nonnegative(
            self.feasible_auxiliary_k, "feasible_auxiliary_k"
        )
        if feasible_k <= 0.0:
            raise ValueError("feasible auxiliary k must be strictly positive")
        object.__setattr__(self, "particle_count", count)
        object.__setattr__(self, "coefficient_order", order)
        object.__setattr__(self, "axis_lower", tuple(float(value) for value in lower))
        object.__setattr__(self, "axis_upper", tuple(float(value) for value in upper))
        object.__setattr__(self, "inequality_labels", labels)
        object.__setattr__(
            self,
            "inequality_matrix",
            tuple(tuple(float(value) for value in row) for row in matrix),
        )
        object.__setattr__(self, "inequality_upper", tuple(float(value) for value in rhs))
        object.__setattr__(self, "auxiliary_k_lower", k_low)
        object.__setattr__(self, "auxiliary_k_upper", k_high)
        object.__setattr__(self, "amplitude_ratio_names", ratio_names)
        object.__setattr__(
            self, "amplitude_ratio_lower", tuple(float(value) for value in ratio_low)
        )
        object.__setattr__(
            self, "amplitude_ratio_upper", tuple(float(value) for value in ratio_high)
        )
        object.__setattr__(self, "feasible_auxiliary_k", feasible_k)
        feasible = tuple(float(value) for value in self.feasible_coefficients)
        object.__setattr__(self, "feasible_coefficients", feasible)
        if not self.contains(feasible) or not self.contains_with_k(feasible, feasible_k):
            raise ValueError("declared coefficient polytope has no valid feasible witness")

    @property
    def coefficient_count(self) -> int:
        return len(self.coefficient_order)

    def contains(self, coefficients: Sequence[float], *, atol: float = 1.0e-10) -> bool:
        tolerance = _nonnegative_tolerance(atol)
        try:
            values = np.asarray(coefficients, dtype=np.float64)
        except (TypeError, ValueError):
            return False
        if values.shape != (self.coefficient_count,) or not np.all(np.isfinite(values)):
            return False
        lower = np.asarray(self.axis_lower)
        upper = np.asarray(self.axis_upper)
        matrix = np.asarray(self.inequality_matrix)
        if matrix.size == 0:
            matrix = np.empty((0, self.coefficient_count), dtype=np.float64)
        rhs = np.asarray(self.inequality_upper)
        particle_total = float(np.sum(values[1 : self.particle_count + 1]))
        axis_tolerance = np.asarray(
            [
                _roundoff_tolerance(value, low, high, atol=tolerance)
                for value, low, high in zip(values, lower, upper)
            ]
        )
        left = matrix @ values
        row_scale = np.sum(np.abs(matrix) * np.abs(values)[np.newaxis, :], axis=1)
        inequality_tolerance = np.maximum(
            tolerance,
            64.0
            * np.finfo(np.float64).eps
            * np.maximum(1.0, np.maximum(row_scale, np.abs(rhs))),
        )
        return bool(
            particle_total > 0.0
            and np.all(values >= lower - axis_tolerance)
            and np.all(values <= upper + axis_tolerance)
            and np.all(left <= rhs + inequality_tolerance)
        )

    def feasible_k_interval(
        self,
        coefficients: Sequence[float],
        *,
        atol: float = 1.0e-10,
    ) -> tuple[float, float] | None:
        """Return all shared GUI ``k`` witnesses for effective coefficients."""

        tolerance = _nonnegative_tolerance(atol)
        try:
            values = np.asarray(coefficients, dtype=np.float64)
        except (TypeError, ValueError):
            return None
        if not self.contains(values, atol=tolerance):
            return None
        amplitudes = values[1:]
        low = float(self.auxiliary_k_lower)
        high = float(self.auxiliary_k_upper)
        for amplitude, ratio_low, ratio_high in zip(
            amplitudes,
            self.amplitude_ratio_lower,
            self.amplitude_ratio_upper,
        ):
            if ratio_high == 0.0:
                if amplitude > _roundoff_tolerance(amplitude, atol=tolerance):
                    return None
            else:
                low = max(low, float(amplitude / ratio_high))
            if ratio_low > 0.0:
                high = min(high, float(amplitude / ratio_low))
        k_tolerance = _roundoff_tolerance(
            low,
            high,
            self.auxiliary_k_lower,
            self.auxiliary_k_upper,
            atol=tolerance,
        )
        if low > high + k_tolerance:
            return None
        if low > high:
            midpoint = 0.5 * (low + high)
            low = high = midpoint
        return float(low), float(high)

    def select_k(
        self,
        coefficients: Sequence[float],
        *,
        preferred: float | None = None,
        atol: float = 1.0e-10,
    ) -> float:
        """Choose a deterministic legal GUI ``k`` without changing the curve."""

        interval = self.feasible_k_interval(coefficients, atol=atol)
        if interval is None:
            raise ValueError("coefficients have no shared GUI k witness")
        low, high = interval
        if preferred is None:
            selected = low + 0.5 * (high - low)
        else:
            value = _finite_nonnegative(preferred, "preferred k")
            selected = min(max(value, low), high)
        if selected <= 0.0:
            selected = max(low, float(np.nextafter(0.0, 1.0)))
        if not self.contains_with_k(coefficients, selected, atol=atol):
            # At a floating-point intersection, one endpoint is more stable
            # than an arithmetic midpoint reconstructed from ratios.
            for candidate in (low, high):
                if candidate > 0.0 and self.contains_with_k(
                    coefficients, candidate, atol=atol
                ):
                    return float(candidate)
            raise RuntimeError("failed to reconstruct a valid shared GUI k witness")
        return float(selected)

    def contains_with_k(
        self,
        coefficients: Sequence[float],
        k: float,
        *,
        atol: float = 1.0e-10,
    ) -> bool:
        """Check one explicit GUI parameterization, not just its projection."""

        tolerance = _nonnegative_tolerance(atol)
        try:
            values = np.asarray(coefficients, dtype=np.float64)
            k_value = float(k)
        except (TypeError, ValueError):
            return False
        if not self.contains(values, atol=tolerance) or not np.isfinite(k_value) or k_value <= 0.0:
            return False
        k_tolerance = _roundoff_tolerance(
            k_value,
            self.auxiliary_k_lower,
            self.auxiliary_k_upper,
            atol=tolerance,
        )
        if not (
            self.auxiliary_k_lower - k_tolerance
            <= k_value
            <= self.auxiliary_k_upper + k_tolerance
        ):
            return False
        for amplitude, low, high in zip(
            values[1:], self.amplitude_ratio_lower, self.amplitude_ratio_upper
        ):
            ratio = amplitude / k_value
            ratio_tolerance = _roundoff_tolerance(ratio, low, high, atol=tolerance)
            if not low - ratio_tolerance <= ratio <= high + ratio_tolerance:
                return False
        return True

    def scaled_system(
        self, coefficient_scale: Sequence[float]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return bounds and ``A z <= b`` for ``z = scale * coefficients``."""

        scale = np.asarray(coefficient_scale, dtype=np.float64)
        if scale.shape != (self.coefficient_count,):
            raise ValueError("coefficient_scale has the wrong shape")
        if not np.all(np.isfinite(scale)) or np.any(scale <= 0.0):
            raise ValueError("coefficient_scale must be finite and strictly positive")
        matrix = np.asarray(self.inequality_matrix)
        if matrix.size == 0:
            matrix = np.empty((0, self.coefficient_count), dtype=np.float64)
        return (
            np.asarray(self.axis_lower) * scale,
            np.asarray(self.axis_upper) * scale,
            matrix / scale[np.newaxis, :],
            np.asarray(self.inequality_upper),
        )

    def to_audit_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "version": self.version,
            "canonical_gauge": self.canonical_gauge,
            "particle_count": self.particle_count,
            "resolution_present": self.resolution_present,
            "coefficient_order": list(self.coefficient_order),
            "axis_lower": list(self.axis_lower),
            "axis_upper": list(self.axis_upper),
            "linear_inequalities": [
                {
                    "label": label,
                    "coefficients": list(row),
                    "upper": upper,
                }
                for label, row, upper in zip(
                    self.inequality_labels,
                    self.inequality_matrix,
                    self.inequality_upper,
                )
            ],
            "feasible_coefficients": list(self.feasible_coefficients),
            "auxiliary_k": {
                "name": "kappa",
                "low": self.auxiliary_k_lower,
                "high": self.auxiliary_k_upper,
                "feasible_witness": self.feasible_auxiliary_k,
                "ratio_names": list(self.amplitude_ratio_names),
                "ratio_lower": list(self.amplitude_ratio_lower),
                "ratio_upper": list(self.amplitude_ratio_upper),
                "constraints": [
                    {
                        "ratio": name,
                        "lower": f"{low} * kappa <= {coefficient}",
                        "upper": f"{coefficient} <= {high} * kappa",
                    }
                    for name, coefficient, low, high in zip(
                        self.amplitude_ratio_names,
                        self.coefficient_order[1:],
                        self.amplitude_ratio_lower,
                        self.amplitude_ratio_upper,
                    )
                ],
            },
        }


@dataclass(frozen=True, kw_only=True)
class GuiAmplitudeConstraint:
    """Versioned GUI ranges converted exactly to a coefficient polytope."""

    background: ClosedInterval
    component_intensities: tuple[ClosedInterval, ...]
    k: ClosedInterval
    resolution_present: bool
    int_res: ClosedInterval | None = None
    schema: str = GUI_AMPLITUDE_CONSTRAINT_SCHEMA
    version: str = GUI_AMPLITUDE_CONSTRAINT_VERSION
    canonical_gauge: str = CANONICAL_AMPLITUDE_GAUGE

    def __post_init__(self) -> None:
        if self.schema != GUI_AMPLITUDE_CONSTRAINT_SCHEMA:
            raise ValueError("unsupported GUI amplitude constraint schema")
        if self.version != GUI_AMPLITUDE_CONSTRAINT_VERSION:
            raise ValueError("unsupported GUI amplitude constraint version")
        if self.canonical_gauge != CANONICAL_AMPLITUDE_GAUGE:
            raise ValueError("unexpected canonical amplitude gauge")
        background = _interval(self.background, "background")
        k = _interval(self.k, "k")
        if k.high <= 0.0:
            raise ValueError("k range must contain a strictly positive value")
        intensities = tuple(self.component_intensities)
        if not 1 <= len(intensities) <= MAX_COMPONENTS:
            raise ValueError(f"component_intensities must contain 1..{MAX_COMPONENTS} ranges")
        for index, value in enumerate(intensities, 1):
            _interval(value, f"component_intensities[{index}]")
        if type(self.resolution_present) is not bool:
            raise TypeError("resolution_present must be a bool")
        if self.resolution_present:
            if self.int_res is None:
                raise ValueError("int_res range is required when Resolution is present")
            int_res = _interval(self.int_res, "int_res")
        else:
            if self.int_res is not None:
                raise ValueError("int_res range must be omitted when Resolution is absent")
            int_res = None
        if not any(value.high > 0.0 for value in intensities):
            raise ValueError("at least one component Int range must contain a positive value")
        object.__setattr__(self, "background", background)
        object.__setattr__(self, "component_intensities", intensities)
        object.__setattr__(self, "k", k)
        object.__setattr__(self, "int_res", int_res)
        # Building the polytope validates a concrete feasibility witness and
        # catches future conversion regressions at construction time.
        self.coefficient_polytope()

    @property
    def particle_count(self) -> int:
        return len(self.component_intensities)

    def validate_branch(self, particle_count: int, resolution_present: bool) -> None:
        if isinstance(particle_count, (bool, np.bool_)) or not isinstance(particle_count, Integral):
            raise TypeError("particle_count must be an integer")
        if int(particle_count) != self.particle_count:
            raise ValueError("GUI amplitude constraint particle count does not match the branch")
        if type(resolution_present) is not bool:
            raise TypeError("resolution_present must be a bool")
        if resolution_present != self.resolution_present:
            raise ValueError(
                "GUI amplitude constraint Resolution presence does not match the branch"
            )

    def coefficient_polytope(self) -> CoefficientPolytope:
        count = self.particle_count
        size = count + 1 + int(self.resolution_present)
        order = ("BG",) + tuple(f"a_{index}" for index in range(1, count + 1))
        if self.resolution_present:
            order += ("a_res",)
        ratio_names = tuple(f"Int_{index}" for index in range(1, count + 1))
        ratio_intervals = self.component_intensities
        if self.resolution_present:
            assert self.int_res is not None
            ratio_names += ("int_Res",)
            ratio_intervals += (self.int_res,)
        lower = [
            self.background.low,
            *(self.k.low * value.low for value in ratio_intervals),
        ]
        upper = [
            self.background.high,
            *(self.k.high * value.high for value in ratio_intervals),
        ]
        rows: list[np.ndarray] = []
        rhs: list[float] = []
        labels: list[str] = []
        # Eliminate kappa exactly.  Every lower bound on kappa (a_j/u_j)
        # must not exceed every upper bound (a_m/l_m).  The k interval itself
        # is already represented by the exact marginal axis bounds above.
        for upper_index, upper_interval in enumerate(ratio_intervals):
            if upper_interval.high == 0.0:
                continue
            for lower_index, lower_interval in enumerate(ratio_intervals):
                if upper_index == lower_index or lower_interval.low == 0.0:
                    continue
                row = np.zeros(size, dtype=np.float64)
                row[upper_index + 1] = lower_interval.low
                row[lower_index + 1] -= upper_interval.high
                rows.append(row)
                rhs.append(0.0)
                labels.append(
                    f"shared_k_{ratio_names[upper_index]}_upper_vs_"
                    f"{ratio_names[lower_index]}_lower"
                )

        k_value = self.k.low + 0.5 * (self.k.high - self.k.low)
        if k_value <= 0.0:
            k_value = 0.5 * self.k.high
        ratios = np.asarray(
            [value.low + 0.5 * (value.high - value.low) for value in ratio_intervals],
            dtype=np.float64,
        )
        feasible = [
            self.background.low + 0.5 * (self.background.high - self.background.low),
            *(k_value * ratios),
        ]
        return CoefficientPolytope(
            particle_count=count,
            resolution_present=self.resolution_present,
            coefficient_order=order,
            axis_lower=tuple(lower),
            axis_upper=tuple(upper),
            inequality_labels=tuple(labels),
            inequality_matrix=tuple(tuple(float(value) for value in row) for row in rows),
            inequality_upper=tuple(rhs),
            feasible_coefficients=tuple(float(value) for value in feasible),
            auxiliary_k_lower=self.k.low,
            auxiliary_k_upper=self.k.high,
            amplitude_ratio_names=ratio_names,
            amplitude_ratio_lower=tuple(value.low for value in ratio_intervals),
            amplitude_ratio_upper=tuple(value.high for value in ratio_intervals),
            feasible_auxiliary_k=k_value,
        )

    def contains(
        self,
        coefficients: Sequence[float],
        *,
        k: float | None = None,
        atol: float = 1.0e-10,
    ) -> bool:
        polytope = self.coefficient_polytope()
        if k is None:
            return polytope.contains(coefficients, atol=atol)
        return polytope.contains_with_k(coefficients, k, atol=atol)

    def assess(
        self,
        coefficients: Sequence[float],
        *,
        k: float | None = None,
        atol: float = 1.0e-10,
    ) -> GuiAmplitudeConstraintAudit:
        tolerance = _nonnegative_tolerance(atol)
        values = np.asarray(coefficients, dtype=np.float64)
        polytope = self.coefficient_polytope()
        if values.shape != (polytope.coefficient_count,) or not np.all(np.isfinite(values)):
            raise ValueError("coefficients have the wrong shape or contain non-finite values")
        particle_total = float(np.sum(values[1 : self.particle_count + 1]))
        positive = particle_total > 0.0
        feasible_k_interval = polytope.feasible_k_interval(values, atol=tolerance)
        if k is None:
            selected_k = (
                float("nan")
                if feasible_k_interval is None
                else polytope.select_k(values, atol=tolerance)
            )
        else:
            selected_k = _finite_nonnegative(k, "k")
        k_is_witness = polytope.contains_with_k(values, selected_k, atol=tolerance)
        intensities = (
            np.full(self.particle_count, np.nan)
            if not k_is_witness
            else values[1 : self.particle_count + 1] / selected_k
        )
        int_res = (
            float("nan")
            if not k_is_witness
            else float(values[-1] / selected_k)
            if self.resolution_present
            else 0.0
        )
        named = [("BG", float(values[0]), self.background), ("k", selected_k, self.k)]
        named.extend(
            (f"Int_{index}", float(value), interval)
            for index, (value, interval) in enumerate(
                zip(intensities, self.component_intensities), 1
            )
        )
        if self.resolution_present:
            assert self.int_res is not None
            named.append(("int_Res", int_res, self.int_res))
        checks = tuple(
            GuiRangeCheck(
                name=name,
                value=value,
                low=interval.low,
                high=interval.high,
                satisfied=interval.contains(
                    value,
                    atol=_roundoff_tolerance(
                        value,
                        interval.low,
                        interval.high,
                        atol=tolerance,
                    ),
                ),
            )
            for name, value, interval in named
        )
        polytope_satisfied = polytope.contains(values, atol=tolerance)
        all_satisfied = (
            positive
            and polytope_satisfied
            and k_is_witness
            and all(item.satisfied for item in checks)
        )
        return GuiAmplitudeConstraintAudit(
            schema=GUI_AMPLITUDE_CONSTRAINT_SCHEMA,
            version=self.version,
            canonical_gauge=self.canonical_gauge,
            coefficient_order=polytope.coefficient_order,
            coefficients=tuple(float(value) for value in values),
            auxiliary_k=selected_k,
            auxiliary_k_feasible_interval=feasible_k_interval,
            auxiliary_k_is_witness=k_is_witness,
            range_checks=checks,
            coefficient_polytope_satisfied=polytope_satisfied,
            particle_total_positive=positive,
            all_constraints_satisfied=all_satisfied,
        )

    def to_audit_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "version": self.version,
            "canonical_gauge": self.canonical_gauge,
            "resolution_present": self.resolution_present,
            "background": asdict(self.background),
            "component_intensities": [asdict(value) for value in self.component_intensities],
            "k": asdict(self.k),
            "int_res": None if self.int_res is None else asdict(self.int_res),
            "coefficient_polytope": self.coefficient_polytope().to_audit_dict(),
        }


__all__ = [
    "CANONICAL_AMPLITUDE_GAUGE",
    "COEFFICIENT_POLYTOPE_SCHEMA",
    "GUI_AMPLITUDE_CONSTRAINT_SCHEMA",
    "GUI_AMPLITUDE_CONSTRAINT_VERSION",
    "CoefficientPolytope",
    "GuiAmplitudeConstraint",
    "GuiAmplitudeConstraintAudit",
    "GuiRangeCheck",
]
