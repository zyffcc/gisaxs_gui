"""Shared fixed-width codec for Posterior V8 discrete physics branches.

Training labels, learned unit-cube samples, Sobol proposals, and nonlinear
refinement all pass through this module.  It is the single owner of conditional
width coupling and the strict production hard-core feasible domain.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Sequence

import numpy as np

from src.gimap.features.fitting.domain.physical_constraints import constraint_registry

from .contract import (
    CYLINDER,
    MAX_COMPONENTS,
    RESOLUTION_NU_DOMAIN,
    RESOLUTION_SIGMA_DOMAIN,
    VERTICAL_CYLINDER,
    ClosedInterval,
    CoupledFractionBounds,
    GuiComponentBounds,
    GuiComponentParameters,
    LatentComponentBounds,
    LatentComponentParameters,
    LogBounds,
    canonical_topology,
    gui_bounds_to_latent,
    latent_component_to_gui,
    normalize_shape,
    topology_id_for,
)
from .profiled_forward import ResolutionShape
from .sobol_numeric_canonicalization_v5 import (
    V5_FAST_NUMERIC_POLICY_VERSION,
    V5NumericOps,
    v5_numeric_ops,
    validate_v5_numeric_policy,
)


BRANCH_CODEC_VERSION = "posterior_v8_policy_bound_branch_unit_cube_varying_axes_v3"
COMPONENT_UNIT_AXES = (
    "log_R",
    "sigma_R_fraction",
    "log_h",
    "sigma_h_fraction",
    "log_D",
    "sigma_D_fraction",
)
RESOLUTION_UNIT_AXES = ("log_sigma_res", "nu_res")
COMPONENT_STRIDE = len(COMPONENT_UNIT_AXES)
RESOLUTION_OFFSET = MAX_COMPONENTS * COMPONENT_STRIDE
UNIT_CUBE_DIMENSIONS = RESOLUTION_OFFSET + len(RESOLUTION_UNIT_AXES)
INACTIVE_UNIT_VALUE = 0.5
HARD_CORE_SPACING_MARGIN = float(constraint_registry.get("hard_core_spacing").default_margin)
STRICT_LOG_MAX_CORRECTIONS = 8

if MAX_COMPONENTS != 4 or UNIT_CUBE_DIMENSIONS != 26:  # pragma: no cover
    raise RuntimeError("Posterior V8 branch layout must remain fixed at 26 dimensions")


@dataclass(frozen=True)
class ResolutionBounds:
    """Closed user ranges for the hard resolution-present branch."""

    sigma_res: ClosedInterval
    nu_res: ClosedInterval

    def __post_init__(self) -> None:
        if not isinstance(self.sigma_res, ClosedInterval) or not isinstance(
            self.nu_res, ClosedInterval
        ):
            raise TypeError("resolution bounds must be ClosedInterval values")
        if (
            self.sigma_res.low < RESOLUTION_SIGMA_DOMAIN.low
            or self.sigma_res.high > RESOLUTION_SIGMA_DOMAIN.high
            or self.nu_res.low < RESOLUTION_NU_DOMAIN.low
            or self.nu_res.high > RESOLUTION_NU_DOMAIN.high
        ):
            raise ValueError("resolution bounds exceed the versioned V8 resolution domain")

    def contains(self, value: ResolutionShape) -> bool:
        return (
            isinstance(value, ResolutionShape)
            and self.sigma_res.contains(value.sigma_res)
            and self.nu_res.contains(value.nu_res)
        )


def _unit_values(values: Sequence[float]) -> tuple[float, ...]:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("unit_cube must contain numeric coordinates") from exc
    if array.shape != (UNIT_CUBE_DIMENSIONS,):
        raise ValueError(f"unit_cube must have shape ({UNIT_CUBE_DIMENSIONS},)")
    if not np.all(np.isfinite(array)) or np.any(array < 0.0) or np.any(array > 1.0):
        raise ValueError("unit_cube coordinates must be finite and lie in [0, 1]")
    return tuple(float(value) for value in array)


def _active_values(values: Sequence[bool]) -> tuple[bool, ...]:
    try:
        result = tuple(values)
    except TypeError as exc:
        raise TypeError("active_mask must be a boolean sequence") from exc
    if len(result) != UNIT_CUBE_DIMENSIONS or not all(
        isinstance(value, (bool, np.bool_)) for value in result
    ):
        raise ValueError(f"active_mask must contain {UNIT_CUBE_DIMENSIONS} boolean values")
    return tuple(bool(value) for value in result)


def _roundoff_only_clamp(value: float, interval: ClosedInterval, label: str) -> float:
    """Clamp at most eight binary64 ULPs to an authoritative closed GUI range."""

    numeric = float(value)
    scale = max(abs(numeric), abs(interval.low), abs(interval.high))
    tolerance = 8.0 * abs(float(np.spacing(scale)))
    if numeric < interval.low:
        if interval.low - numeric > tolerance:
            raise RuntimeError(f"{label} escaped its GUI interval beyond roundoff")
        return interval.low
    if numeric > interval.high:
        if numeric - interval.high > tolerance:
            raise RuntimeError(f"{label} escaped its GUI interval beyond roundoff")
        return interval.high
    return numeric


@dataclass(frozen=True)
class BranchCoordinates:
    """Immutable fixed-width unit-cube value and semantic active mask."""

    unit_cube: tuple[float, ...]
    active_mask: tuple[bool, ...]

    def __post_init__(self) -> None:
        unit_cube = _unit_values(self.unit_cube)
        active_mask = _active_values(self.active_mask)
        if any(
            value != INACTIVE_UNIT_VALUE
            for value, active in zip(unit_cube, active_mask)
            if not active
        ):
            raise ValueError(f"inactive unit-cube coordinates must equal {INACTIVE_UNIT_VALUE}")
        object.__setattr__(self, "unit_cube", unit_cube)
        object.__setattr__(self, "active_mask", active_mask)


@dataclass(frozen=True)
class _Axis:
    low: float
    high: float
    label: str

    def __post_init__(self) -> None:
        low, high = float(self.low), float(self.high)
        if not np.isfinite(low) or not np.isfinite(high) or low > high:
            raise ValueError(f"{self.label} interval is empty or non-finite")
        object.__setattr__(self, "low", low)
        object.__setattr__(self, "high", high)

    def decode(self, unit: float) -> float:
        unit = float(unit)
        if not np.isfinite(unit) or not 0.0 <= unit <= 1.0:
            raise ValueError(f"{self.label} unit coordinate must lie in [0, 1]")
        if self.low == self.high:
            return self.low
        result = float(self.low + unit * (self.high - self.low))
        if not self.low <= result <= self.high:  # pragma: no cover
            raise RuntimeError(f"{self.label} interpolation escaped its interval")
        return result

    def encode(self, value: float) -> float:
        value = float(value)
        if not np.isfinite(value) or not self.low <= value <= self.high:
            raise ValueError(f"{self.label}={value!r} is outside [{self.low}, {self.high}]")
        if self.low == self.high:
            return INACTIVE_UNIT_VALUE
        result = float((value - self.low) / (self.high - self.low))
        if not 0.0 <= result <= 1.0:  # pragma: no cover
            raise RuntimeError(f"{self.label} inverse escaped the unit cube")
        return result


def _feasible_log_mean(
    bounds: CoupledFractionBounds,
    numeric: V5NumericOps,
) -> tuple[float, float]:
    if bounds.mean_log is None:
        raise ValueError("direct fractional bounds have no coupled mean")
    physical, fraction = bounds.mean_log.physical, bounds.allowed_fraction
    low = max(physical.low, bounds.gui_width.low / fraction.high)
    high = min(physical.high, bounds.gui_width.high / fraction.low)
    if low > high:
        raise ValueError("coupled width bounds have no feasible mean interval")
    return numeric.log(low), numeric.log(high)


def _fraction_limits(
    bounds: CoupledFractionBounds,
    log_mean: float | None,
    numeric: V5NumericOps,
) -> tuple[float, float]:
    if bounds.mean_log is None:
        return bounds.envelope.low, bounds.envelope.high
    if log_mean is None:
        raise ValueError("coupled fractional bounds require their log mean")
    mean = numeric.exp(log_mean)
    low = max(bounds.allowed_fraction.low, bounds.gui_width.low / mean)
    high = min(bounds.allowed_fraction.high, bounds.gui_width.high / mean)
    if low > high:
        raise ValueError("mean has no feasible coupled-width fraction")
    return float(low), float(high)


def _strict_log_above(value: float, numeric: V5NumericOps) -> float:
    physical_probe = float(np.nextafter(float(value), np.inf))
    result = numeric.log(physical_probe)
    for _ in range(STRICT_LOG_MAX_CORRECTIONS):
        if numeric.exp(result) > value:
            return result
        result = float(np.nextafter(result, np.inf))
    raise RuntimeError("could not construct a bounded strict log above physical value")


def _strict_log_below(value: float, numeric: V5NumericOps) -> float:
    physical_probe = float(np.nextafter(float(value), -np.inf))
    if physical_probe <= 0.0:
        raise ValueError("strict log below requires a positive predecessor")
    result = numeric.log(physical_probe)
    for _ in range(STRICT_LOG_MAX_CORRECTIONS):
        if numeric.exp(result) < value:
            return result
        result = float(np.nextafter(result, -np.inf))
    raise RuntimeError("could not construct a bounded strict log below physical value")


def _cap_log_high_strict(
    current_high: float,
    physical_limit: float,
    numeric: V5NumericOps,
) -> float:
    if not np.isfinite(physical_limit) or physical_limit <= 0.0:
        raise ValueError("hard-core spacing leaves no physically feasible geometry")
    return (
        current_high
        if numeric.exp(current_high) < physical_limit
        else _strict_log_below(physical_limit, numeric)
    )


def _required_spacing(
    shape: str,
    log_r: float,
    log_h: float | None,
    numeric: V5NumericOps,
) -> float:
    radius = numeric.exp(log_r)
    if shape == CYLINDER:
        if log_h is None:
            raise ValueError("cylinder hard-core spacing requires h")
        size = numeric.hypot(2.0 * radius, numeric.exp(log_h))
    else:
        size = 2.0 * radius
    if not np.isfinite(size) or size <= 0.0:
        raise ValueError("could not evaluate the authoritative exclusion size")
    return HARD_CORE_SPACING_MARGIN * float(size)


@dataclass(frozen=True)
class _ComponentCodec:
    bounds: LatentComponentBounds
    d_present: bool
    numeric_policy_version: str

    @classmethod
    def build(
        cls,
        gui_bounds: GuiComponentBounds,
        d_present: bool,
        numeric_policy_version: str,
    ) -> "_ComponentCodec":
        bounds = gui_bounds_to_latent(gui_bounds)
        if d_present and bounds.d_policy == "absent":
            raise ValueError(f"{bounds.shape} D-present branch has no D user ranges")
        if not d_present and bounds.d_policy == "required":
            raise ValueError(f"{bounds.shape} D-absent branch conflicts with required D ranges")
        result = cls(bounds, d_present, validate_v5_numeric_policy(numeric_policy_version))
        result._r_log_limits()
        return result

    @property
    def numeric(self) -> V5NumericOps:
        return v5_numeric_ops(self.numeric_policy_version)

    @staticmethod
    def _physical_log_limits(bounds: LogBounds, numeric: V5NumericOps) -> tuple[float, float]:
        return numeric.log(bounds.physical.low), numeric.log(bounds.physical.high)

    def _base_r_log_limits(self) -> tuple[float, float]:
        if self.bounds.shape == VERTICAL_CYLINDER:
            return self._physical_log_limits(self.bounds.log_R, self.numeric)
        return _feasible_log_mean(self.bounds.sigma_R_fraction, self.numeric)

    def _base_h_log_limits(self) -> tuple[float, float]:
        if self.bounds.sigma_h_fraction is None:
            raise ValueError("cylinder branch is missing h bounds")
        return _feasible_log_mean(self.bounds.sigma_h_fraction, self.numeric)

    def _base_d_log_limits(self) -> tuple[float, float]:
        if self.bounds.sigma_D_fraction is None:
            raise ValueError("D-present branch is missing D bounds")
        return _feasible_log_mean(self.bounds.sigma_D_fraction, self.numeric)

    def _r_log_limits(self) -> tuple[float, float]:
        low, high = self._base_r_log_limits()
        if not self.d_present:
            return low, high
        _, d_high = self._base_d_log_limits()
        max_exclusion = self.numeric.exp(d_high) / HARD_CORE_SPACING_MARGIN
        if self.bounds.shape == CYLINDER:
            h_low, _ = self._base_h_log_limits()
            square = max_exclusion**2 - self.numeric.exp(h_low) ** 2
            radius_limit = 0.5 * self.numeric.sqrt(square) if square > 0.0 else 0.0
        else:
            radius_limit = 0.5 * max_exclusion
        high = _cap_log_high_strict(high, radius_limit, self.numeric)
        if low > high:
            raise ValueError("D bounds leave no geometry satisfying hard-core spacing")
        return low, high

    def _h_log_limits(self, log_r: float) -> tuple[float, float]:
        low, high = self._base_h_log_limits()
        if not self.d_present:
            return low, high
        _, d_high = self._base_d_log_limits()
        max_exclusion = self.numeric.exp(d_high) / HARD_CORE_SPACING_MARGIN
        square = max_exclusion**2 - (2.0 * self.numeric.exp(log_r)) ** 2
        height_limit = self.numeric.sqrt(square) if square > 0.0 else 0.0
        high = _cap_log_high_strict(high, height_limit, self.numeric)
        if low > high:
            raise ValueError("D bounds leave no cylinder height satisfying hard-core spacing")
        return low, high

    def _d_log_limits(self, log_r: float, log_h: float | None) -> tuple[float, float]:
        low, high = self._base_d_log_limits()
        low = max(
            low,
            _strict_log_above(
                _required_spacing(self.bounds.shape, log_r, log_h, self.numeric),
                self.numeric,
            ),
        )
        if low > high:
            raise ValueError("decoded geometry leaves no D satisfying hard-core spacing")
        return low, high

    def _validate(self, component: LatentComponentParameters) -> None:
        if not isinstance(component, LatentComponentParameters):
            raise TypeError("seed components must be LatentComponentParameters")
        if component.shape != self.bounds.shape:
            raise ValueError(
                f"seed shape {component.shape!r} does not match hard topology {self.bounds.shape!r}"
            )
        if (component.log_D is not None) != self.d_present:
            raise ValueError("latent seed D presence does not match the hard branch")
        if not self._contains(component):
            raise ValueError("latent component seed is outside its GUI/coupled bounds")
        if self.d_present:
            assert component.log_D is not None
            required = _required_spacing(
                component.shape,
                component.log_R,
                component.log_h,
                self.numeric,
            )
            if self.numeric.exp(component.log_D) <= required:
                raise ValueError(
                    f"latent component seed violates hard-core spacing: D must be > {required:g}"
                )

    def _coupled_contains(
        self,
        bounds: CoupledFractionBounds,
        fraction: float,
        log_mean: float | None,
    ) -> bool:
        if not bounds.allowed_fraction.contains(fraction):
            return False
        if bounds.mean_log is None:
            return bounds.gui_width.contains(fraction)
        if log_mean is None:
            return False
        low, high = self._physical_log_limits(bounds.mean_log, self.numeric)
        return (
            low - 1.0e-12 <= log_mean <= high + 1.0e-12
            and bounds.gui_width.contains(self.numeric.exp(log_mean) * float(fraction))
        )

    def _contains(self, component: LatentComponentParameters) -> bool:
        if component.shape != self.bounds.shape:
            return False
        r_low, r_high = self._physical_log_limits(self.bounds.log_R, self.numeric)
        if not r_low - 1.0e-12 <= component.log_R <= r_high + 1.0e-12:
            return False
        if not self._coupled_contains(
            self.bounds.sigma_R_fraction,
            component.sigma_R_fraction,
            component.log_R,
        ):
            return False
        if self.bounds.shape == CYLINDER:
            if (
                component.log_h is None
                or component.sigma_h_fraction is None
                or self.bounds.log_h is None
                or self.bounds.sigma_h_fraction is None
            ):
                return False
            h_low, h_high = self._physical_log_limits(self.bounds.log_h, self.numeric)
            if not h_low - 1.0e-12 <= component.log_h <= h_high + 1.0e-12:
                return False
            if not self._coupled_contains(
                self.bounds.sigma_h_fraction,
                component.sigma_h_fraction,
                component.log_h,
            ):
                return False
        present = component.log_D is not None
        if self.bounds.d_policy == "absent":
            return not present
        if not present:
            return self.bounds.d_policy == "optional"
        if (
            component.sigma_D_fraction is None
            or self.bounds.log_D is None
            or self.bounds.sigma_D_fraction is None
        ):
            return False
        d_low, d_high = self._physical_log_limits(self.bounds.log_D, self.numeric)
        return (
            d_low - 1.0e-12 <= component.log_D <= d_high + 1.0e-12
            and self._coupled_contains(
                self.bounds.sigma_D_fraction,
                component.sigma_D_fraction,
                component.log_D,
            )
        )

    def to_gui(self, component: LatentComponentParameters) -> GuiComponentParameters:
        self._validate(component)
        radius = _roundoff_only_clamp(
            self.numeric.exp(component.log_R),
            self.bounds.log_R.physical,
            "R",
        )
        sigma_r = (
            component.sigma_R_fraction
            if self.bounds.shape == VERTICAL_CYLINDER
            else radius * component.sigma_R_fraction
        )
        sigma_r = _roundoff_only_clamp(
            sigma_r,
            self.bounds.sigma_R_fraction.gui_width,
            "sigma_R",
        )
        height = sigma_h = None
        if self.bounds.shape == CYLINDER:
            assert component.log_h is not None
            assert component.sigma_h_fraction is not None
            assert self.bounds.log_h is not None
            assert self.bounds.sigma_h_fraction is not None
            height = _roundoff_only_clamp(
                self.numeric.exp(component.log_h),
                self.bounds.log_h.physical,
                "h",
            )
            sigma_h = _roundoff_only_clamp(
                height * component.sigma_h_fraction,
                self.bounds.sigma_h_fraction.gui_width,
                "sigma_h",
            )
        spacing = sigma_d = None
        if self.d_present:
            assert component.log_D is not None
            assert component.sigma_D_fraction is not None
            assert self.bounds.log_D is not None
            assert self.bounds.sigma_D_fraction is not None
            spacing = _roundoff_only_clamp(
                self.numeric.exp(component.log_D),
                self.bounds.log_D.physical,
                "D",
            )
            sigma_d = _roundoff_only_clamp(
                spacing * component.sigma_D_fraction,
                self.bounds.sigma_D_fraction.gui_width,
                "sigma_D",
            )
        return GuiComponentParameters(
            shape=self.bounds.shape,
            R=radius,
            sigma_R=sigma_r,
            h=height,
            sigma_h=sigma_h,
            D=spacing,
            sigma_D=sigma_d,
        )

    def encode(
        self, component: LatentComponentParameters, values: list[float], offset: int
    ) -> None:
        self._validate(component)
        values[offset] = _Axis(*self._r_log_limits(), "log_R").encode(component.log_R)
        values[offset + 1] = _Axis(
            *_fraction_limits(
                self.bounds.sigma_R_fraction,
                component.log_R,
                self.numeric,
            ),
            "sigma_R_fraction",
        ).encode(component.sigma_R_fraction)
        if self.bounds.shape == CYLINDER:
            assert component.log_h is not None and component.sigma_h_fraction is not None
            assert self.bounds.sigma_h_fraction is not None
            values[offset + 2] = _Axis(*self._h_log_limits(component.log_R), "log_h").encode(
                component.log_h
            )
            values[offset + 3] = _Axis(
                *_fraction_limits(
                    self.bounds.sigma_h_fraction,
                    component.log_h,
                    self.numeric,
                ),
                "sigma_h_fraction",
            ).encode(component.sigma_h_fraction)
        if self.d_present:
            assert component.log_D is not None and component.sigma_D_fraction is not None
            assert self.bounds.sigma_D_fraction is not None
            values[offset + 4] = _Axis(
                *self._d_log_limits(component.log_R, component.log_h), "log_D"
            ).encode(component.log_D)
            values[offset + 5] = _Axis(
                *_fraction_limits(
                    self.bounds.sigma_D_fraction,
                    component.log_D,
                    self.numeric,
                ),
                "sigma_D_fraction",
            ).encode(component.sigma_D_fraction)

    def decode(self, values: tuple[float, ...], offset: int) -> LatentComponentParameters:
        log_r = _Axis(*self._r_log_limits(), "log_R").decode(values[offset])
        sigma_r = _Axis(
            *_fraction_limits(self.bounds.sigma_R_fraction, log_r, self.numeric),
            "sigma_R_fraction",
        ).decode(values[offset + 1])
        log_h = sigma_h = None
        if self.bounds.shape == CYLINDER:
            assert self.bounds.sigma_h_fraction is not None
            log_h = _Axis(*self._h_log_limits(log_r), "log_h").decode(values[offset + 2])
            sigma_h = _Axis(
                *_fraction_limits(self.bounds.sigma_h_fraction, log_h, self.numeric),
                "sigma_h_fraction",
            ).decode(values[offset + 3])
        log_d = sigma_d = None
        if self.d_present:
            assert self.bounds.sigma_D_fraction is not None
            log_d = _Axis(*self._d_log_limits(log_r, log_h), "log_D").decode(values[offset + 4])
            sigma_d = _Axis(
                *_fraction_limits(self.bounds.sigma_D_fraction, log_d, self.numeric),
                "sigma_D_fraction",
            ).decode(values[offset + 5])
        component = LatentComponentParameters(
            self.bounds.shape, log_r, sigma_r, log_h, sigma_h, log_d, sigma_d
        )
        self._validate(component)
        return component


def _canonical_inputs(
    topology: Sequence[str], component_bounds: Sequence[GuiComponentBounds]
) -> tuple[tuple[str, ...], tuple[GuiComponentBounds, ...], int]:
    if isinstance(topology, (str, bytes)):
        raise TypeError("topology must be a sequence of canonical shape names")
    try:
        raw = tuple(topology)
        bounds = tuple(component_bounds)
    except TypeError as exc:
        raise TypeError("topology and component_bounds must be sequences") from exc
    if not raw or not all(isinstance(item, str) for item in raw):
        raise TypeError("topology must contain canonical shape-name strings")
    normalized = tuple(normalize_shape(item) for item in raw)
    canonical = canonical_topology(normalized)
    if raw != canonical:
        raise ValueError(f"topology must use canonical catalog names/order; expected {canonical!r}")
    if len(bounds) != len(canonical):
        raise ValueError("component_bounds length must match the canonical topology")
    if not all(isinstance(item, GuiComponentBounds) for item in bounds):
        raise TypeError("component_bounds must contain only GuiComponentBounds")
    shapes = tuple(item.shape for item in bounds)
    if shapes != canonical:
        raise ValueError(
            "component_bounds must use the same canonical topology order; "
            f"expected {canonical!r}, got {shapes!r}"
        )
    return canonical, bounds, topology_id_for(canonical)


def _d_flags(value: bool | Sequence[bool], count: int) -> tuple[bool, ...]:
    if isinstance(value, (bool, np.bool_)):
        if count != 1:
            raise ValueError("multi-component topologies require one d_present flag per component")
        return (bool(value),)
    if isinstance(value, (str, bytes)):
        raise TypeError("d_present must contain boolean values")
    try:
        flags = tuple(value)
    except TypeError as exc:
        raise TypeError("d_present must be a boolean sequence") from exc
    if len(flags) != count:
        raise ValueError("d_present length must match the canonical topology")
    if not all(isinstance(item, (bool, np.bool_)) for item in flags):
        raise TypeError("d_present must contain only boolean values")
    return tuple(bool(item) for item in flags)


@dataclass(frozen=True)
class ProfiledBranchCodec:
    """Authoritative bidirectional codec for one fixed discrete branch."""

    topology_id: int
    topology: tuple[str, ...]
    d_present: tuple[bool, ...]
    component_bounds: tuple[GuiComponentBounds, ...]
    resolution_bounds: ResolutionBounds | None
    numeric_policy_version: str
    _components: tuple[_ComponentCodec, ...]

    def __post_init__(self) -> None:
        policy = validate_v5_numeric_policy(self.numeric_policy_version)
        if (
            len(self.topology) != len(self.d_present)
            or len(self.topology) != len(self.component_bounds)
            or len(self.topology) != len(self._components)
        ):
            raise ValueError("branch codec fields have inconsistent component counts")
        for shape, present, gui_bounds, component in zip(
            self.topology,
            self.d_present,
            self.component_bounds,
            self._components,
        ):
            if (
                shape != gui_bounds.shape
                or present != component.d_present
                or component.numeric_policy_version != policy
                or component.bounds != gui_bounds_to_latent(gui_bounds)
            ):
                raise ValueError("branch codec component does not reproduce its bound policy")

    @classmethod
    def build(
        cls,
        topology: Sequence[str],
        component_bounds: Sequence[GuiComponentBounds],
        d_present: bool | Sequence[bool],
        *,
        resolution_bounds: ResolutionBounds | None = None,
        numeric_policy_version: str = V5_FAST_NUMERIC_POLICY_VERSION,
    ) -> "ProfiledBranchCodec":
        canonical, bounds, topology_id = _canonical_inputs(topology, component_bounds)
        flags = _d_flags(d_present, len(canonical))
        numeric_policy = validate_v5_numeric_policy(numeric_policy_version)
        if resolution_bounds is not None and not isinstance(resolution_bounds, ResolutionBounds):
            raise TypeError("resolution_bounds must be a ResolutionBounds or None")
        components = tuple(
            _ComponentCodec.build(item, present, numeric_policy)
            for item, present in zip(bounds, flags)
        )
        return cls(
            topology_id,
            canonical,
            flags,
            bounds,
            resolution_bounds,
            numeric_policy,
            components,
        )

    @property
    def numeric(self) -> V5NumericOps:
        return v5_numeric_ops(self.numeric_policy_version)

    def latent_components_to_gui(
        self,
        components: Sequence[LatentComponentParameters],
    ) -> tuple[GuiComponentParameters, ...]:
        values = tuple(components)
        if len(values) != len(self._components):
            raise ValueError("components length must match this branch codec")
        return tuple(
            codec.to_gui(value)
            for codec, value in zip(self._components, values)
        )

    @property
    def active_mask(self) -> tuple[bool, ...]:
        mask = [False] * UNIT_CUBE_DIMENSIONS
        for slot, component in enumerate(self._components):
            offset = slot * COMPONENT_STRIDE
            mask[offset : offset + 2] = (True, True)
            if component.bounds.shape == CYLINDER:
                mask[offset + 2 : offset + 4] = (True, True)
            if component.d_present:
                mask[offset + 4 : offset + 6] = (True, True)
        if self.resolution_bounds is not None:
            mask[RESOLUTION_OFFSET:] = (True, True)
        return tuple(mask)

    @property
    def active_indices(self) -> tuple[int, ...]:
        return tuple(index for index, active in enumerate(self.active_mask) if active)

    @cached_property
    def varying_mask(self) -> tuple[bool, ...]:
        """Axes with non-zero freedom under this exact user query and branch.

        ``active_mask`` describes which parameters exist in the hard branch;
        it intentionally includes parameters whose user interval is fixed.
        Refinement must instead use this effective mask.  Round-tripping the
        two open-cube endpoints through the triangular codec also accounts for
        fixed absolute widths represented as mean-coupled fractions.
        """

        midpoint = np.full(UNIT_CUBE_DIMENSIONS, INACTIVE_UNIT_VALUE, dtype=np.float64)
        result = np.zeros(UNIT_CUBE_DIMENSIONS, dtype=bool)
        for index in self.active_indices:
            endpoints: list[float] = []
            for endpoint in (1.0e-12, 1.0 - 1.0e-12):
                probe = midpoint.copy()
                probe[index] = endpoint
                components, resolution = self.decode(probe)
                endpoints.append(self.encode(components, resolution).unit_cube[index])
            result[index] = not np.isclose(endpoints[0], endpoints[1], rtol=0.0, atol=1.0e-12)
        mask = tuple(bool(value) for value in result)
        if any(varying and not active for varying, active in zip(mask, self.active_mask)):
            raise RuntimeError("codec varying mask escaped its active axes")
        return mask

    @cached_property
    def varying_indices(self) -> tuple[int, ...]:
        """Fixed-width coordinate indices that may vary in this user query."""

        return tuple(index for index, varying in enumerate(self.varying_mask) if varying)

    @property
    def latent_bounds(self) -> tuple[LatentComponentBounds, ...]:
        return tuple(component.bounds for component in self._components)

    def canonical_coordinates(self, values: Sequence[float]) -> BranchCoordinates:
        unit_cube = list(_unit_values(values))
        mask = self.active_mask
        for index, active in enumerate(mask):
            if not active:
                unit_cube[index] = INACTIVE_UNIT_VALUE
        return BranchCoordinates(tuple(unit_cube), mask)

    def encode(
        self,
        seed_components: Sequence[LatentComponentParameters],
        resolution_seed: ResolutionShape | None = None,
    ) -> BranchCoordinates:
        seeds = tuple(seed_components)
        if len(seeds) != len(self._components):
            raise ValueError("seed_components length must match this branch topology")
        values = [INACTIVE_UNIT_VALUE] * UNIT_CUBE_DIMENSIONS
        for slot, (codec, seed) in enumerate(zip(self._components, seeds)):
            codec.encode(seed, values, slot * COMPONENT_STRIDE)
        if self.resolution_bounds is None:
            if resolution_seed is not None:
                raise ValueError("resolution-absent branch must not provide a resolution seed")
        else:
            if not isinstance(resolution_seed, ResolutionShape):
                raise TypeError("resolution-present branch requires a ResolutionShape seed")
            if not self.resolution_bounds.contains(resolution_seed):
                raise ValueError("resolution seed is outside its bounds")
            values[RESOLUTION_OFFSET] = _Axis(
                self.numeric.log(self.resolution_bounds.sigma_res.low),
                self.numeric.log(self.resolution_bounds.sigma_res.high),
                "log_sigma_res",
            ).encode(self.numeric.log(resolution_seed.sigma_res))
            values[RESOLUTION_OFFSET + 1] = _Axis(
                self.resolution_bounds.nu_res.low,
                self.resolution_bounds.nu_res.high,
                "nu_res",
            ).encode(resolution_seed.nu_res)
        return BranchCoordinates(tuple(values), self.active_mask)

    def decode(
        self, coordinates: BranchCoordinates | Sequence[float]
    ) -> tuple[tuple[LatentComponentParameters, ...], ResolutionShape | None]:
        if isinstance(coordinates, BranchCoordinates):
            if coordinates.active_mask != self.active_mask:
                raise ValueError("coordinate active_mask does not match this branch codec")
            encoded = coordinates
        else:
            encoded = self.canonical_coordinates(coordinates)
        components = tuple(
            codec.decode(encoded.unit_cube, slot * COMPONENT_STRIDE)
            for slot, codec in enumerate(self._components)
        )
        resolution = None
        if self.resolution_bounds is not None:
            sigma_res = _roundoff_only_clamp(
                self.numeric.exp(
                _Axis(
                    self.numeric.log(self.resolution_bounds.sigma_res.low),
                    self.numeric.log(self.resolution_bounds.sigma_res.high),
                    "log_sigma_res",
                ).decode(encoded.unit_cube[RESOLUTION_OFFSET])
                ),
                self.resolution_bounds.sigma_res,
                "sigma_res",
            )
            nu_res = _Axis(
                self.resolution_bounds.nu_res.low,
                self.resolution_bounds.nu_res.high,
                "nu_res",
            ).decode(encoded.unit_cube[RESOLUTION_OFFSET + 1])
            resolution = ResolutionShape(sigma_res, nu_res)
            if not self.resolution_bounds.contains(resolution):
                raise RuntimeError("decoded resolution escaped its bounds")
        return components, resolution

    def encode_active(
        self,
        seed_components: Sequence[LatentComponentParameters],
        resolution_seed: ResolutionShape | None = None,
    ) -> np.ndarray:
        coordinates = self.encode(seed_components, resolution_seed)
        return np.asarray(
            [coordinates.unit_cube[index] for index in self.active_indices],
            dtype=np.float64,
        )

    def decode_active(
        self, values: Sequence[float]
    ) -> tuple[
        tuple[LatentComponentParameters, ...],
        tuple[GuiComponentParameters, ...],
        ResolutionShape | None,
    ]:
        array = np.asarray(values, dtype=np.float64)
        if array.shape != (len(self.active_indices),):
            raise ValueError("refinement vector has the wrong shape")
        full = [INACTIVE_UNIT_VALUE] * UNIT_CUBE_DIMENSIONS
        for index, value in zip(self.active_indices, array):
            full[index] = float(value)
        components, resolution = self.decode(full)
        gui = self.latent_components_to_gui(components)
        return components, gui, resolution

    def encode_varying(
        self,
        seed_components: Sequence[LatentComponentParameters],
        resolution_seed: ResolutionShape | None = None,
    ) -> np.ndarray:
        """Encode only dimensions with effective freedom under the user query."""

        coordinates = self.encode(seed_components, resolution_seed)
        return np.asarray(
            [coordinates.unit_cube[index] for index in self.varying_indices],
            dtype=np.float64,
        )

    def decode_varying(
        self,
        values: Sequence[float],
        template_coordinates: BranchCoordinates | Sequence[float],
    ) -> tuple[
        tuple[LatentComponentParameters, ...],
        tuple[GuiComponentParameters, ...],
        ResolutionShape | None,
    ]:
        """Decode a varying-only vector while preserving its seed template.

        The template is authoritative for every inactive or query-fixed axis.
        Usually fixed physical intervals encode to the canonical midpoint, but
        retaining the full seed makes that invariant explicit and prevents a
        future coupled-coordinate change from moving fixed values silently.
        """

        array = np.asarray(values, dtype=np.float64)
        if array.shape != (len(self.varying_indices),):
            raise ValueError("varying refinement vector has the wrong shape")
        if not np.all(np.isfinite(array)) or np.any(array < 0.0) or np.any(array > 1.0):
            raise ValueError("varying refinement vector must be finite and lie in [0, 1]")
        if isinstance(template_coordinates, BranchCoordinates):
            if template_coordinates.active_mask != self.active_mask:
                raise ValueError("template active_mask does not match this branch codec")
            template = template_coordinates
        else:
            template = self.canonical_coordinates(template_coordinates)
        full = list(template.unit_cube)
        for index, value in zip(self.varying_indices, array):
            full[index] = float(value)
        components, resolution = self.decode(full)
        gui = self.latent_components_to_gui(components)
        return components, gui, resolution


__all__ = [
    "BRANCH_CODEC_VERSION",
    "BranchCoordinates",
    "COMPONENT_STRIDE",
    "COMPONENT_UNIT_AXES",
    "HARD_CORE_SPACING_MARGIN",
    "INACTIVE_UNIT_VALUE",
    "ProfiledBranchCodec",
    "RESOLUTION_OFFSET",
    "RESOLUTION_UNIT_AXES",
    "ResolutionBounds",
    "UNIT_CUBE_DIMENSIONS",
]
