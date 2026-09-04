"""Versioned scientific contract for the Posterior V8 proposal model.

This module deliberately contains no forward implementation.  It names the
existing production forward contract and provides the lossless boundary
between GUI parameters and the model's latent geometry representation.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations_with_replacement
from numbers import Integral
from typing import Iterable, Sequence

import numpy as np

from .sobol_numeric_canonicalization_v5 import (
    V5_DETERMINISTIC_NUMERIC_POLICY_VERSION,
    V5_FAST_NUMERIC_POLICY_VERSION,
    v5_numeric_ops,
    validate_v5_numeric_policy,
)


CONTRACT_VERSION = "posterior_v8_contract_v1"
CODEC_VERSION = "posterior_v8_policy_aware_log_size_fraction_width_v3"
FORWARD_MODEL_VERSION = "gimap_fitting_scattering_model_2026_08_20"

SPHERE = "sphere"
CYLINDER = "cylinder"
VERTICAL_CYLINDER = "vertical_cylinder"
SHAPES = (SPHERE, CYLINDER, VERTICAL_CYLINDER)
MIN_COMPONENTS = 1
MAX_COMPONENTS = 4

_SHAPE_INDEX = {shape: index for index, shape in enumerate(SHAPES)}
_SHAPE_ALIASES = {
    "sphere": SPHERE,
    "cylinder": CYLINDER,
    "vertical_cylinder": VERTICAL_CYLINDER,
    "verticalcylinder": VERTICAL_CYLINDER,
}


def normalize_shape(shape: str) -> str:
    """Return one canonical shape name or reject an unknown GUI spelling."""
    key = str(shape).strip().lower().replace("-", "_").replace(" ", "_")
    try:
        return _SHAPE_ALIASES[key]
    except KeyError as exc:
        raise ValueError(f"unknown shape {shape!r}; expected one of {SHAPES}") from exc


def canonical_topology(shapes: Iterable[str]) -> tuple[str, ...]:
    """Canonicalize an unordered K=1..4 multiset of particle shapes."""
    result = tuple(normalize_shape(shape) for shape in shapes)
    if not MIN_COMPONENTS <= len(result) <= MAX_COMPONENTS:
        raise ValueError("a topology must contain between one and four components")
    return tuple(sorted(result, key=_SHAPE_INDEX.__getitem__))


TOPOLOGIES: tuple[tuple[str, ...], ...] = tuple(
    topology
    for k in range(MIN_COMPONENTS, MAX_COMPONENTS + 1)
    for topology in combinations_with_replacement(SHAPES, k)
)
TOPOLOGY_TO_ID = {topology: topology_id for topology_id, topology in enumerate(TOPOLOGIES)}
NUM_TOPOLOGIES = len(TOPOLOGIES)

if NUM_TOPOLOGIES != 34:  # pragma: no cover - import-time invariant
    raise RuntimeError(f"Posterior V8 topology catalog must contain 34 classes, got {NUM_TOPOLOGIES}")


def topology_id_for(shapes: Iterable[str]) -> int:
    return TOPOLOGY_TO_ID[canonical_topology(shapes)]


def topology_from_id(topology_id: int) -> tuple[str, ...]:
    if isinstance(topology_id, (bool, np.bool_)) or not isinstance(topology_id, Integral):
        raise TypeError("topology_id must be an integer, not bool or float")
    index = int(topology_id)
    if not 0 <= index < NUM_TOPOLOGIES:
        raise ValueError(f"topology_id must be in [0, {NUM_TOPOLOGIES - 1}]")
    return TOPOLOGIES[index]


@dataclass(frozen=True)
class ClosedInterval:
    low: float
    high: float

    def __post_init__(self) -> None:
        low, high = float(self.low), float(self.high)
        if not np.isfinite(low) or not np.isfinite(high):
            raise ValueError("interval endpoints must be finite")
        if low > high:
            raise ValueError(f"interval low must not exceed high: [{low}, {high}]")
        object.__setattr__(self, "low", low)
        object.__setattr__(self, "high", high)

    def contains(self, value: float, *, atol: float = 1e-12) -> bool:
        value = float(value)
        return np.isfinite(value) and self.low - atol <= value <= self.high + atol


R_DOMAIN = ClosedInterval(1.0, 100.0)
H_DOMAIN = ClosedInterval(2.0, 500.0)
D_DOMAIN = ClosedInterval(3.0, 500.0)
SIZE_WIDTH_FRACTION_DOMAIN = ClosedInterval(0.02, 0.90)
D_WIDTH_FRACTION_DOMAIN = ClosedInterval(0.05, 0.90)
RESOLUTION_SIGMA_DOMAIN = ClosedInterval(0.002, 0.30)
RESOLUTION_NU_DOMAIN = ClosedInterval(1.0, 10.0)


def _require_subset(value: ClosedInterval, domain: ClosedInterval, name: str) -> None:
    if value.low < domain.low or value.high > domain.high:
        raise ValueError(
            f"{name}=[{value.low}, {value.high}] is outside "
            f"[{domain.low}, {domain.high}]"
        )


def _roundoff_only_clamp(
    value: float,
    interval: ClosedInterval,
    label: str,
) -> float:
    """Normalize at most eight binary64 ULPs to a closed physical interval.

    ``exp(log(endpoint))`` is not guaranteed to reproduce the endpoint exactly.
    This is a versioned inverse-codec normalization, not a general-purpose clip:
    values farther than the declared roundoff envelope still fail closed.
    """

    numeric = float(value)
    if not np.isfinite(numeric):
        raise RuntimeError(f"{label} inverse-codec value is not finite")
    scale = max(abs(numeric), abs(interval.low), abs(interval.high))
    tolerance = 8.0 * abs(float(np.spacing(scale)))
    if numeric < interval.low:
        if interval.low - numeric > tolerance:
            raise RuntimeError(f"{label} escaped its physical interval beyond roundoff")
        return interval.low
    if numeric > interval.high:
        if numeric - interval.high > tolerance:
            raise RuntimeError(f"{label} escaped its physical interval beyond roundoff")
        return interval.high
    return numeric


def _exp_closed_interval_endpoint(
    log_value: float,
    interval: ClosedInterval,
    label: str,
    numeric,
) -> float:
    """Invert one log coordinate while preserving exact closed endpoints."""

    encoded = float(log_value)
    if encoded == numeric.log(interval.low):
        return interval.low
    if encoded == numeric.log(interval.high):
        return interval.high
    return _roundoff_only_clamp(numeric.exp(encoded), interval, label)


def _positive(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _optional_d_pair(d: float | None, sigma_d: float | None) -> tuple[float | None, float | None]:
    if d == 0.0 and sigma_d == 0.0:
        return None, None
    if (d is None) != (sigma_d is None):
        raise ValueError("D and sigma_D must either both be present or both be absent")
    return d, sigma_d


@dataclass(frozen=True)
class GuiComponentParameters:
    """One component in current GUI units.

    Sphere/cylinder ``sigma_R`` and cylinder ``sigma_h`` are absolute.  The
    historical vertical-cylinder ``sigma_R`` is already fractional.  A missing
    D term is represented canonically as ``D=sigma_D=None``; GUI ``0, 0`` is
    accepted at this boundary and normalized to ``None, None``.
    """

    shape: str
    R: float
    sigma_R: float
    h: float | None = None
    sigma_h: float | None = None
    D: float | None = None
    sigma_D: float | None = None

    def __post_init__(self) -> None:
        shape = normalize_shape(self.shape)
        object.__setattr__(self, "shape", shape)
        r = _positive(self.R, "R")
        sigma_r = _positive(self.sigma_R, "sigma_R")
        _require_subset(ClosedInterval(r, r), R_DOMAIN, "R")
        sigma_r_fraction = sigma_r if shape == VERTICAL_CYLINDER else sigma_r / r
        _require_subset(
            ClosedInterval(sigma_r_fraction, sigma_r_fraction),
            SIZE_WIDTH_FRACTION_DOMAIN,
            "sigma_R fraction",
        )
        object.__setattr__(self, "R", r)
        object.__setattr__(self, "sigma_R", sigma_r)

        if shape == CYLINDER:
            if self.h is None or self.sigma_h is None:
                raise ValueError("cylinder requires h and sigma_h")
            h = _positive(self.h, "h")
            sigma_h = _positive(self.sigma_h, "sigma_h")
            _require_subset(ClosedInterval(h, h), H_DOMAIN, "h")
            _require_subset(
                ClosedInterval(sigma_h / h, sigma_h / h),
                SIZE_WIDTH_FRACTION_DOMAIN,
                "sigma_h fraction",
            )
            object.__setattr__(self, "h", h)
            object.__setattr__(self, "sigma_h", sigma_h)
        elif self.h is not None or self.sigma_h is not None:
            raise ValueError(f"{shape} must not define h or sigma_h")

        d, sigma_d = _optional_d_pair(self.D, self.sigma_D)
        if d is not None:
            d = _positive(d, "D")
            sigma_d = _positive(sigma_d, "sigma_D")
            _require_subset(ClosedInterval(d, d), D_DOMAIN, "D")
            _require_subset(
                ClosedInterval(sigma_d / d, sigma_d / d),
                D_WIDTH_FRACTION_DOMAIN,
                "sigma_D fraction",
            )
        object.__setattr__(self, "D", d)
        object.__setattr__(self, "sigma_D", sigma_d)


@dataclass(frozen=True)
class LatentComponentParameters:
    """Nonlinear V8 geometry: log sizes and dimensionless width fractions."""

    shape: str
    log_R: float
    sigma_R_fraction: float
    log_h: float | None = None
    sigma_h_fraction: float | None = None
    log_D: float | None = None
    sigma_D_fraction: float | None = None

    def __post_init__(self) -> None:
        shape = normalize_shape(self.shape)
        object.__setattr__(self, "shape", shape)
        _validate_log_value(self.log_R, R_DOMAIN, "log_R")
        _require_subset(
            ClosedInterval(self.sigma_R_fraction, self.sigma_R_fraction),
            SIZE_WIDTH_FRACTION_DOMAIN,
            "sigma_R_fraction",
        )
        if shape == CYLINDER:
            if self.log_h is None or self.sigma_h_fraction is None:
                raise ValueError("cylinder latent parameters require log_h and sigma_h_fraction")
            _validate_log_value(self.log_h, H_DOMAIN, "log_h")
            _require_subset(
                ClosedInterval(self.sigma_h_fraction, self.sigma_h_fraction),
                SIZE_WIDTH_FRACTION_DOMAIN,
                "sigma_h_fraction",
            )
        elif self.log_h is not None or self.sigma_h_fraction is not None:
            raise ValueError(f"{shape} latent parameters must not define h")
        if (self.log_D is None) != (self.sigma_D_fraction is None):
            raise ValueError("log_D and sigma_D_fraction must both be present or absent")
        if self.log_D is not None:
            _validate_log_value(self.log_D, D_DOMAIN, "log_D")
            _require_subset(
                ClosedInterval(self.sigma_D_fraction, self.sigma_D_fraction),
                D_WIDTH_FRACTION_DOMAIN,
                "sigma_D_fraction",
            )


def _validate_log_value(value: float, domain: ClosedInterval, name: str) -> None:
    value = float(value)
    deterministic = v5_numeric_ops(V5_DETERMINISTIC_NUMERIC_POLICY_VERSION)
    low = deterministic.log(domain.low)
    high = deterministic.log(domain.high)
    # Native libm endpoints may differ from the deterministic representative
    # by an adjacent binary64 value.  Validation is deliberately a few ULPs
    # outward, while every policy-bound codec enforces its own exact limits.
    for _ in range(4):
        low = float(np.nextafter(low, -np.inf))
        high = float(np.nextafter(high, np.inf))
    if not np.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{name} is outside the versioned latent domain")


def gui_component_to_latent(
    component: GuiComponentParameters,
    *,
    numeric_policy_version: str = V5_FAST_NUMERIC_POLICY_VERSION,
) -> LatentComponentParameters:
    numeric = v5_numeric_ops(validate_v5_numeric_policy(numeric_policy_version))
    shape = component.shape
    return LatentComponentParameters(
        shape=shape,
        log_R=numeric.log(component.R),
        sigma_R_fraction=(
            component.sigma_R if shape == VERTICAL_CYLINDER else component.sigma_R / component.R
        ),
        log_h=numeric.log(component.h) if shape == CYLINDER else None,
        sigma_h_fraction=(component.sigma_h / component.h) if shape == CYLINDER else None,
        log_D=numeric.log(component.D) if component.D is not None else None,
        sigma_D_fraction=(component.sigma_D / component.D) if component.D is not None else None,
    )


def latent_component_to_gui(
    component: LatentComponentParameters,
    *,
    numeric_policy_version: str = V5_FAST_NUMERIC_POLICY_VERSION,
) -> GuiComponentParameters:
    numeric = v5_numeric_ops(validate_v5_numeric_policy(numeric_policy_version))
    r = _exp_closed_interval_endpoint(component.log_R, R_DOMAIN, "R", numeric)
    h = (
        _exp_closed_interval_endpoint(component.log_h, H_DOMAIN, "h", numeric)
        if component.log_h is not None
        else None
    )
    d = (
        _exp_closed_interval_endpoint(component.log_D, D_DOMAIN, "D", numeric)
        if component.log_D is not None
        else None
    )
    return GuiComponentParameters(
        shape=component.shape,
        R=r,
        sigma_R=(
            component.sigma_R_fraction
            if component.shape == VERTICAL_CYLINDER
            else r * component.sigma_R_fraction
        ),
        h=h,
        sigma_h=(h * component.sigma_h_fraction) if h is not None else None,
        D=d,
        sigma_D=(d * component.sigma_D_fraction) if d is not None else None,
    )


@dataclass(frozen=True)
class GuiComponentBounds:
    """Closed user ranges for one component, expressed in current GUI units."""

    shape: str
    R: ClosedInterval
    sigma_R: ClosedInterval
    h: ClosedInterval | None = None
    sigma_h: ClosedInterval | None = None
    D: ClosedInterval | None = None
    sigma_D: ClosedInterval | None = None
    allow_D_absent: bool = False

    def __post_init__(self) -> None:
        shape = normalize_shape(self.shape)
        object.__setattr__(self, "shape", shape)
        _require_subset(self.R, R_DOMAIN, "R")
        if shape == VERTICAL_CYLINDER:
            _require_subset(self.sigma_R, SIZE_WIDTH_FRACTION_DOMAIN, "sigma_R fraction")
        else:
            _validate_fraction_feasibility(self.sigma_R, self.R, SIZE_WIDTH_FRACTION_DOMAIN, "sigma_R")
        if shape == CYLINDER:
            if self.h is None or self.sigma_h is None:
                raise ValueError("cylinder bounds require h and sigma_h")
            _require_subset(self.h, H_DOMAIN, "h")
            _validate_fraction_feasibility(self.sigma_h, self.h, SIZE_WIDTH_FRACTION_DOMAIN, "sigma_h")
        elif self.h is not None or self.sigma_h is not None:
            raise ValueError(f"{shape} bounds must not define h or sigma_h")
        if (self.D is None) != (self.sigma_D is None):
            raise ValueError("D and sigma_D bounds must both be present or absent")
        if self.D is None:
            if self.allow_D_absent:
                raise ValueError("allow_D_absent requires D and sigma_D ranges")
        else:
            _require_subset(self.D, D_DOMAIN, "D")
            _validate_fraction_feasibility(self.sigma_D, self.D, D_WIDTH_FRACTION_DOMAIN, "sigma_D")


def _validate_fraction_feasibility(
    width: ClosedInterval,
    mean: ClosedInterval,
    domain: ClosedInterval,
    name: str,
) -> None:
    if width.low <= 0.0:
        raise ValueError(f"{name} range must be strictly positive")
    raw_low = width.low / mean.high
    raw_high = width.high / mean.low
    if max(raw_low, domain.low) > min(raw_high, domain.high):
        raise ValueError(f"{name} bounds have no feasible width fraction in the V8 domain")


@dataclass(frozen=True)
class LogBounds:
    """Lossless physical interval with its nonlinear log-space view."""

    physical: ClosedInterval

    def __post_init__(self) -> None:
        if self.physical.low <= 0.0:
            raise ValueError("log-space physical bounds must be strictly positive")

    @property
    def low(self) -> float:
        return float(np.log(self.physical.low))

    @property
    def high(self) -> float:
        return float(np.log(self.physical.high))

    def limits(
        self,
        numeric_policy_version: str = V5_FAST_NUMERIC_POLICY_VERSION,
    ) -> tuple[float, float]:
        numeric = v5_numeric_ops(validate_v5_numeric_policy(numeric_policy_version))
        return numeric.log(self.physical.low), numeric.log(self.physical.high)

    def contains(
        self,
        log_value: float,
        numeric_policy_version: str = V5_FAST_NUMERIC_POLICY_VERSION,
    ) -> bool:
        low, high = self.limits(numeric_policy_version)
        return ClosedInterval(low, high).contains(log_value)


@dataclass(frozen=True)
class CoupledFractionBounds:
    """Exact width constraint plus its axis-aligned latent envelope.

    For absolute GUI widths the constraint remains coupled to its mean size.
    Keeping that coupling avoids silently accepting an invalid corner of the
    rectangular user range.  ``mean_log=None`` denotes an already-fractional
    GUI width (vertical-cylinder sigma_R).
    """

    gui_width: ClosedInterval
    allowed_fraction: ClosedInterval
    mean_log: LogBounds | None = None

    def __post_init__(self) -> None:
        if self.gui_width.low <= 0.0:
            raise ValueError("width bounds must be strictly positive")
        if self.allowed_fraction.low <= 0.0:
            raise ValueError("allowed fractions must be strictly positive")
        if self.mean_log is None:
            _require_subset(self.gui_width, self.allowed_fraction, "direct fractional width")
        else:
            _validate_fraction_feasibility(
                self.gui_width,
                self.mean_log.physical,
                self.allowed_fraction,
                "coupled width",
            )

    @property
    def envelope(self) -> ClosedInterval:
        if self.mean_log is None:
            return self.gui_width
        low = max(self.gui_width.low / self.mean_log.physical.high, self.allowed_fraction.low)
        high = min(self.gui_width.high / self.mean_log.physical.low, self.allowed_fraction.high)
        return ClosedInterval(low, high)

    def contains(
        self,
        fraction: float,
        log_mean: float | None = None,
        numeric_policy_version: str = V5_FAST_NUMERIC_POLICY_VERSION,
    ) -> bool:
        if not self.allowed_fraction.contains(fraction):
            return False
        if self.mean_log is None:
            return self.gui_width.contains(fraction)
        if log_mean is None or not self.mean_log.contains(
            log_mean,
            numeric_policy_version,
        ):
            return False
        numeric = v5_numeric_ops(validate_v5_numeric_policy(numeric_policy_version))
        return self.gui_width.contains(numeric.exp(log_mean) * float(fraction))


@dataclass(frozen=True)
class LatentComponentBounds:
    shape: str
    log_R: LogBounds
    sigma_R_fraction: CoupledFractionBounds
    log_h: LogBounds | None = None
    sigma_h_fraction: CoupledFractionBounds | None = None
    d_policy: str = "absent"
    log_D: LogBounds | None = None
    sigma_D_fraction: CoupledFractionBounds | None = None

    def __post_init__(self) -> None:
        shape = normalize_shape(self.shape)
        object.__setattr__(self, "shape", shape)
        if self.d_policy not in {"absent", "required", "optional"}:
            raise ValueError("d_policy must be absent, required, or optional")
        _require_subset(self.log_R.physical, R_DOMAIN, "R")
        expected_r_mean = None if shape == VERTICAL_CYLINDER else self.log_R
        if self.sigma_R_fraction.mean_log != expected_r_mean:
            raise ValueError("sigma_R fractional bounds use the wrong GUI width semantics")
        if shape == CYLINDER:
            if self.log_h is None or self.sigma_h_fraction is None:
                raise ValueError("cylinder latent bounds require h and sigma_h")
            _require_subset(self.log_h.physical, H_DOMAIN, "h")
            if self.sigma_h_fraction.mean_log != self.log_h:
                raise ValueError("sigma_h fractional bounds must be coupled to h")
        elif self.log_h is not None or self.sigma_h_fraction is not None:
            raise ValueError(f"{shape} latent bounds must not define h or sigma_h")
        has_d_bounds = self.log_D is not None or self.sigma_D_fraction is not None
        if (self.log_D is None) != (self.sigma_D_fraction is None):
            raise ValueError("log_D and sigma_D_fraction bounds must both be present or absent")
        if self.d_policy == "absent" and has_d_bounds:
            raise ValueError("absent d_policy must not define D bounds")
        if self.d_policy != "absent" and not has_d_bounds:
            raise ValueError(f"{self.d_policy} d_policy requires D bounds")
        if self.log_D is not None:
            _require_subset(self.log_D.physical, D_DOMAIN, "D")
            if self.sigma_D_fraction.mean_log != self.log_D:
                raise ValueError("sigma_D fractional bounds must be coupled to D")

    def contains(
        self,
        component: LatentComponentParameters,
        numeric_policy_version: str = V5_FAST_NUMERIC_POLICY_VERSION,
    ) -> bool:
        policy = validate_v5_numeric_policy(numeric_policy_version)
        if component.shape != self.shape or not self.log_R.contains(
            component.log_R,
            policy,
        ):
            return False
        if not self.sigma_R_fraction.contains(
            component.sigma_R_fraction,
            component.log_R,
            policy,
        ):
            return False
        if self.shape == CYLINDER:
            if component.log_h is None or not self.log_h.contains(component.log_h, policy):
                return False
            if not self.sigma_h_fraction.contains(
                component.sigma_h_fraction,
                component.log_h,
                policy,
            ):
                return False
        present = component.log_D is not None
        if self.d_policy == "absent":
            return not present
        if not present:
            return self.d_policy == "optional"
        return self.log_D.contains(component.log_D, policy) and self.sigma_D_fraction.contains(
            component.sigma_D_fraction,
            component.log_D,
            policy,
        )

    def envelope(self) -> dict[str, ClosedInterval | str]:
        result: dict[str, ClosedInterval | str] = {
            "log_R": ClosedInterval(self.log_R.low, self.log_R.high),
            "sigma_R_fraction": self.sigma_R_fraction.envelope,
            "d_policy": self.d_policy,
        }
        if self.log_h is not None:
            result["log_h"] = ClosedInterval(self.log_h.low, self.log_h.high)
            result["sigma_h_fraction"] = self.sigma_h_fraction.envelope
        if self.log_D is not None:
            result["log_D"] = ClosedInterval(self.log_D.low, self.log_D.high)
            result["sigma_D_fraction"] = self.sigma_D_fraction.envelope
        return result


def gui_bounds_to_latent(bounds: GuiComponentBounds) -> LatentComponentBounds:
    log_r = LogBounds(bounds.R)
    sigma_r = CoupledFractionBounds(
        bounds.sigma_R,
        SIZE_WIDTH_FRACTION_DOMAIN,
        None if bounds.shape == VERTICAL_CYLINDER else log_r,
    )
    log_h = LogBounds(bounds.h) if bounds.h is not None else None
    sigma_h = (
        CoupledFractionBounds(bounds.sigma_h, SIZE_WIDTH_FRACTION_DOMAIN, log_h)
        if bounds.sigma_h is not None
        else None
    )
    log_d = LogBounds(bounds.D) if bounds.D is not None else None
    sigma_d = (
        CoupledFractionBounds(bounds.sigma_D, D_WIDTH_FRACTION_DOMAIN, log_d)
        if bounds.sigma_D is not None
        else None
    )
    d_policy = "absent" if log_d is None else ("optional" if bounds.allow_D_absent else "required")
    return LatentComponentBounds(
        shape=bounds.shape,
        log_R=log_r,
        sigma_R_fraction=sigma_r,
        log_h=log_h,
        sigma_h_fraction=sigma_h,
        d_policy=d_policy,
        log_D=log_d,
        sigma_D_fraction=sigma_d,
    )


def latent_bounds_to_gui(bounds: LatentComponentBounds) -> GuiComponentBounds:
    return GuiComponentBounds(
        shape=bounds.shape,
        R=bounds.log_R.physical,
        sigma_R=bounds.sigma_R_fraction.gui_width,
        h=bounds.log_h.physical if bounds.log_h is not None else None,
        sigma_h=(bounds.sigma_h_fraction.gui_width if bounds.sigma_h_fraction is not None else None),
        D=bounds.log_D.physical if bounds.log_D is not None else None,
        sigma_D=(bounds.sigma_D_fraction.gui_width if bounds.sigma_D_fraction is not None else None),
        allow_D_absent=bounds.d_policy == "optional",
    )


def topology_bounds_to_latent(
    bounds: Sequence[GuiComponentBounds],
) -> tuple[LatentComponentBounds, ...]:
    """Canonicalize and convert independently specified ranges for K components."""
    if not MIN_COMPONENTS <= len(bounds) <= MAX_COMPONENTS:
        raise ValueError("component bounds must contain between one and four components")
    ordered = sorted(bounds, key=lambda item: _SHAPE_INDEX[item.shape])
    return tuple(gui_bounds_to_latent(item) for item in ordered)


def topology_bounds_to_gui(
    bounds: Sequence[LatentComponentBounds],
) -> tuple[GuiComponentBounds, ...]:
    if not MIN_COMPONENTS <= len(bounds) <= MAX_COMPONENTS:
        raise ValueError("component bounds must contain between one and four components")
    ordered = sorted(bounds, key=lambda item: _SHAPE_INDEX[item.shape])
    return tuple(latent_bounds_to_gui(item) for item in ordered)


def full_component_bounds(
    shape: str,
    *,
    d_policy: str = "optional",
) -> GuiComponentBounds:
    """Return the complete versioned V8 search domain in current GUI units."""

    shape = normalize_shape(shape)
    if d_policy not in {"absent", "required", "optional"}:
        raise ValueError("d_policy must be absent, required, or optional")
    values = {
        "shape": shape,
        "R": R_DOMAIN,
        "sigma_R": (
            SIZE_WIDTH_FRACTION_DOMAIN
            if shape == VERTICAL_CYLINDER
            else ClosedInterval(
                R_DOMAIN.low * SIZE_WIDTH_FRACTION_DOMAIN.low,
                R_DOMAIN.high * SIZE_WIDTH_FRACTION_DOMAIN.high,
            )
        ),
        "D": None if d_policy == "absent" else D_DOMAIN,
        "sigma_D": (
            None
            if d_policy == "absent"
            else ClosedInterval(
                D_DOMAIN.low * D_WIDTH_FRACTION_DOMAIN.low,
                D_DOMAIN.high * D_WIDTH_FRACTION_DOMAIN.high,
            )
        ),
        "allow_D_absent": d_policy == "optional",
    }
    if shape == CYLINDER:
        values.update(
            {
                "h": H_DOMAIN,
                "sigma_h": ClosedInterval(
                    H_DOMAIN.low * SIZE_WIDTH_FRACTION_DOMAIN.low,
                    H_DOMAIN.high * SIZE_WIDTH_FRACTION_DOMAIN.high,
                ),
            }
        )
    return GuiComponentBounds(**values)
