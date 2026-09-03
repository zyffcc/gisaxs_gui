"""Query-first GUI amplitude ranges for the Posterior V8 V5.1 model.

The physical model is ``BG + k * (sum(Int_i * component_i) + int_Res * R)``.
The ``Int_i`` values and ``int_Res`` are independent GUI parameters sharing
only the multiplicative ``k > 0``.  This module preserves those exact ranges
while providing a fixed-width, observation-scale-equivariant model embedding
before a Resolution branch is selected.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from typing import Sequence

import numpy as np

from .contextual_branch_catalog import PRESENCE_POLICIES, PresencePolicy
from .contract import MAX_COMPONENTS, ClosedInterval
from .gui_amplitude_constraints import (
    CANONICAL_AMPLITUDE_GAUGE,
    GUI_AMPLITUDE_CONSTRAINT_VERSION,
    GuiAmplitudeConstraint,
)
from .sobol_numeric_canonicalization_v5 import (
    V5_FAST_NUMERIC_POLICY_VERSION,
    v5_numeric_policy_sha256,
    v5_numeric_ops,
    validate_v5_numeric_policy,
)


V5_AMPLITUDE_QUERY_SCHEMA = "gisaxs.posterior_v8.amplitude_query/v4"
V5_AMPLITUDE_QUERY_VERSION = (
    "posterior_v8_numeric_contract_bound_query_first_independent_gui_amplitude_ranges_v4"
)
V5_AMPLITUDE_EMBEDDING_VERSION = (
    "posterior_v8_numeric_contract_bound_observation_relative_gui_amplitude_bounds_21d/v4"
)

AMPLITUDE_AXIS_KEYS = (
    "BG",
    "k",
    "Int_1",
    "Int_2",
    "Int_3",
    "Int_4",
    "int_Res",
)
AMPLITUDE_AXIS_STRIDE = 3
AMPLITUDE_AXIS_FIELDS = ("low", "high", "present")
AMPLITUDE_QUERY_PADDING_VALUE = 0.5
AMPLITUDE_QUERY_EMBEDDING_DIM = len(AMPLITUDE_AXIS_KEYS) * AMPLITUDE_AXIS_STRIDE


def _presence_policy(value: object) -> PresencePolicy:
    if not isinstance(value, str):
        raise TypeError("resolution_presence_policy must be a string")
    if value not in PRESENCE_POLICIES:
        raise ValueError("resolution_presence_policy must be absent, optional, or required")
    return value


def _positive_finite(value: float, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite and strictly positive") from exc
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and strictly positive")
    return result


def _nonnegative_interval(value: ClosedInterval, name: str) -> ClosedInterval:
    if not isinstance(value, ClosedInterval):
        raise TypeError(f"{name} must be a ClosedInterval")
    if value.low < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _component_intervals(
    values: Sequence[ClosedInterval],
) -> tuple[ClosedInterval, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError("component_intensities must be a sequence of ClosedInterval values")
    try:
        result = tuple(values)
    except TypeError as exc:
        raise TypeError(
            "component_intensities must be a sequence of ClosedInterval values"
        ) from exc
    if not 1 <= len(result) <= MAX_COMPONENTS:
        raise ValueError(f"component_intensities must contain 1..{MAX_COMPONENTS} ranges")
    for index, value in enumerate(result, 1):
        _nonnegative_interval(value, f"component_intensities[{index}]")
    return result


def _relative_intensity_coordinate(
    value: float,
    intensity_reference: float,
    numeric_policy_version: str,
) -> float:
    """Map a non-negative intensity ratio to [0, 1) without overflow.

    Zero maps to zero.  Positive values use an arctangent of the log ratio,
    retaining useful resolution across many decades.  Jointly scaling
    ``value`` and ``intensity_reference`` leaves this coordinate unchanged.
    """

    if value == 0.0:
        return 0.0
    numeric = v5_numeric_ops(numeric_policy_version)
    log_ratio = numeric.log(value) - numeric.log(intensity_reference)
    return 0.5 + numeric.atan(log_ratio) / math.pi


def _nonnegative_ratio_coordinate(value: float, numeric_policy_version: str) -> float:
    """Map a dimensionless non-negative value to [0, 1) across decades."""

    if value == 0.0:
        return 0.0
    numeric = v5_numeric_ops(numeric_policy_version)
    return 0.5 + numeric.atan(numeric.log(value)) / math.pi


def _axis_triplet(low: float, high: float, *, present: bool) -> tuple[float, float, float]:
    if not present:
        return (AMPLITUDE_QUERY_PADDING_VALUE, AMPLITUDE_QUERY_PADDING_VALUE, 0.0)
    return (float(low), float(high), 1.0)


def _embedding(
    *,
    background: ClosedInterval,
    k: ClosedInterval,
    component_intensities: tuple[ClosedInterval, ...],
    int_res: ClosedInterval | None,
    intensity_reference: float,
    numeric_policy_version: str,
) -> tuple[float, ...]:
    values: list[float] = []
    values.extend(
        _axis_triplet(
            _relative_intensity_coordinate(
                background.low,
                intensity_reference,
                numeric_policy_version,
            ),
            _relative_intensity_coordinate(
                background.high,
                intensity_reference,
                numeric_policy_version,
            ),
            present=True,
        )
    )
    values.extend(
        _axis_triplet(
            _relative_intensity_coordinate(k.low, intensity_reference, numeric_policy_version),
            _relative_intensity_coordinate(k.high, intensity_reference, numeric_policy_version),
            present=True,
        )
    )
    for slot in range(MAX_COMPONENTS):
        if slot < len(component_intensities):
            interval = component_intensities[slot]
            values.extend(
                _axis_triplet(
                    _nonnegative_ratio_coordinate(interval.low, numeric_policy_version),
                    _nonnegative_ratio_coordinate(interval.high, numeric_policy_version),
                    present=True,
                )
            )
        else:
            values.extend(_axis_triplet(0.0, 0.0, present=False))
    if int_res is None:
        values.extend(_axis_triplet(0.0, 0.0, present=False))
    else:
        values.extend(
            _axis_triplet(
                _nonnegative_ratio_coordinate(int_res.low, numeric_policy_version),
                _nonnegative_ratio_coordinate(int_res.high, numeric_policy_version),
                present=True,
            )
        )
    result = tuple(values)
    if len(result) != AMPLITUDE_QUERY_EMBEDDING_DIM or not all(
        np.isfinite(value) and 0.0 <= value <= 1.0 for value in result
    ):
        raise RuntimeError("amplitude query embedding escaped its fixed unit-cube contract")
    return result


def _axis_presence(component_count: int, int_res: ClosedInterval | None) -> tuple[bool, ...]:
    return (
        True,
        True,
        *(slot < component_count for slot in range(MAX_COMPONENTS)),
        int_res is not None,
    )


def _payload(
    *,
    background: ClosedInterval,
    k: ClosedInterval,
    component_intensities: tuple[ClosedInterval, ...],
    resolution_presence_policy: PresencePolicy,
    int_res: ClosedInterval | None,
    numeric_policy_version: str,
) -> dict[str, object]:
    return {
        "schema": V5_AMPLITUDE_QUERY_SCHEMA,
        "version": V5_AMPLITUDE_QUERY_VERSION,
        "embedding_version": V5_AMPLITUDE_EMBEDDING_VERSION,
        "gui_amplitude_constraint_version": GUI_AMPLITUDE_CONSTRAINT_VERSION,
        "canonical_gauge": CANONICAL_AMPLITUDE_GAUGE,
        "numeric_policy_version": numeric_policy_version,
        "numeric_policy_sha256": v5_numeric_policy_sha256(numeric_policy_version),
        "background": asdict(background),
        "k": asdict(k),
        "component_intensities": [asdict(value) for value in component_intensities],
        "resolution_presence_policy": resolution_presence_policy,
        "int_res": None if int_res is None else asdict(int_res),
    }


def _resolution_states(policy: PresencePolicy) -> tuple[bool, ...]:
    if policy == "absent":
        return (False,)
    if policy == "required":
        return (True,)
    return (False, True)


def _constraint(
    *,
    background: ClosedInterval,
    k: ClosedInterval,
    component_intensities: tuple[ClosedInterval, ...],
    int_res: ClosedInterval | None,
    resolution_present: bool,
) -> GuiAmplitudeConstraint:
    return GuiAmplitudeConstraint(
        background=background,
        component_intensities=component_intensities,
        k=k,
        resolution_present=resolution_present,
        int_res=int_res if resolution_present else None,
    )


def _validated_inputs(
    *,
    background: ClosedInterval,
    k: ClosedInterval,
    component_intensities: Sequence[ClosedInterval],
    resolution_presence_policy: PresencePolicy,
    int_res: ClosedInterval | None,
) -> tuple[
    ClosedInterval,
    ClosedInterval,
    tuple[ClosedInterval, ...],
    PresencePolicy,
    ClosedInterval | None,
]:
    bg = _nonnegative_interval(background, "background")
    scale = _nonnegative_interval(k, "k")
    intensities = _component_intervals(component_intensities)
    policy = _presence_policy(resolution_presence_policy)
    if policy == "absent":
        if int_res is not None:
            raise ValueError("Resolution-absent amplitude query must omit int_res")
        resolution_intensity = None
    else:
        resolution_intensity = _nonnegative_interval(int_res, "int_res")
    for resolution_present in _resolution_states(policy):
        _constraint(
            background=bg,
            k=scale,
            component_intensities=intensities,
            int_res=resolution_intensity,
            resolution_present=resolution_present,
        )
    return bg, scale, intensities, policy, resolution_intensity


def _derived_contract(
    *,
    background: ClosedInterval,
    k: ClosedInterval,
    component_intensities: tuple[ClosedInterval, ...],
    resolution_presence_policy: PresencePolicy,
    int_res: ClosedInterval | None,
    numeric_policy_version: str,
) -> tuple[tuple[bool, ...], str, str]:
    payload = _payload(
        background=background,
        k=k,
        component_intensities=component_intensities,
        resolution_presence_policy=resolution_presence_policy,
        int_res=int_res,
        numeric_policy_version=numeric_policy_version,
    )
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return (
        _axis_presence(len(component_intensities), int_res),
        canonical,
        sha256(canonical.encode("utf-8")).hexdigest(),
    )


@dataclass(frozen=True)
class V5AmplitudeQuery:
    """Immutable physical amplitude query created before branch selection."""

    background: ClosedInterval
    k: ClosedInterval
    component_intensities: tuple[ClosedInterval, ...]
    resolution_presence_policy: PresencePolicy
    int_res: ClosedInterval | None
    numeric_policy_version: str
    axis_presence_mask: tuple[bool, ...]
    canonical_json: str
    sha256: str

    @classmethod
    def create(
        cls,
        *,
        background: ClosedInterval,
        k: ClosedInterval,
        component_intensities: Sequence[ClosedInterval],
        resolution_presence_policy: PresencePolicy,
        int_res: ClosedInterval | None,
        numeric_policy_version: str = V5_FAST_NUMERIC_POLICY_VERSION,
    ) -> "V5AmplitudeQuery":
        bg, scale, intensities, policy, resolution_intensity = _validated_inputs(
            background=background,
            k=k,
            component_intensities=component_intensities,
            resolution_presence_policy=resolution_presence_policy,
            int_res=int_res,
        )
        numeric_policy = validate_v5_numeric_policy(numeric_policy_version)
        presence, canonical, digest = _derived_contract(
            background=bg,
            k=scale,
            component_intensities=intensities,
            resolution_presence_policy=policy,
            int_res=resolution_intensity,
            numeric_policy_version=numeric_policy,
        )
        return cls(
            bg,
            scale,
            intensities,
            policy,
            resolution_intensity,
            numeric_policy,
            presence,
            canonical,
            digest,
        )

    def __post_init__(self) -> None:
        bg, scale, intensities, policy, resolution_intensity = _validated_inputs(
            background=self.background,
            k=self.k,
            component_intensities=self.component_intensities,
            resolution_presence_policy=self.resolution_presence_policy,
            int_res=self.int_res,
        )
        numeric_policy = validate_v5_numeric_policy(self.numeric_policy_version)
        presence, canonical, digest = _derived_contract(
            background=bg,
            k=scale,
            component_intensities=intensities,
            resolution_presence_policy=policy,
            int_res=resolution_intensity,
            numeric_policy_version=numeric_policy,
        )
        actual = (
            self.background,
            self.k,
            self.component_intensities,
            self.resolution_presence_policy,
            self.int_res,
            self.numeric_policy_version,
            self.axis_presence_mask,
            self.canonical_json,
            self.sha256,
        )
        expected = (
            bg,
            scale,
            intensities,
            policy,
            resolution_intensity,
            numeric_policy,
            presence,
            canonical,
            digest,
        )
        if actual != expected:
            raise ValueError("V5 amplitude query does not reproduce its derived contract")

    @property
    def particle_count(self) -> int:
        return len(self.component_intensities)

    @property
    def allowed_resolution_states(self) -> tuple[bool, ...]:
        return _resolution_states(self.resolution_presence_policy)

    def constraint_for_branch(self, *, resolution_present: bool) -> GuiAmplitudeConstraint:
        """Build the exact coefficient-polytope constraint for one allowed branch."""

        if type(resolution_present) is not bool:
            raise TypeError("resolution_present must be a bool")
        if resolution_present not in self.allowed_resolution_states:
            raise ValueError("Resolution branch is not allowed by this amplitude query")
        return _constraint(
            background=self.background,
            k=self.k,
            component_intensities=self.component_intensities,
            int_res=self.int_res,
            resolution_present=resolution_present,
        )

    def model_embedding(self, intensity_reference: float) -> tuple[float, ...]:
        """Return the 21D observation-relative model condition.

        The reference belongs to an observation view, not this clean physical
        query, so noisy/masked/cropped sibling views retain one query hash.
        """

        reference = _positive_finite(intensity_reference, "intensity_reference")
        return _embedding(
            background=self.background,
            k=self.k,
            component_intensities=self.component_intensities,
            int_res=self.int_res,
            intensity_reference=reference,
            numeric_policy_version=self.numeric_policy_version,
        )

    def rescaled_intensity(self, scale: float) -> "V5AmplitudeQuery":
        """Return the same dimensionless query under a physical intensity rescaling."""

        factor = _positive_finite(scale, "scale")
        return type(self).create(
            background=ClosedInterval(
                self.background.low * factor,
                self.background.high * factor,
            ),
            k=ClosedInterval(self.k.low * factor, self.k.high * factor),
            component_intensities=self.component_intensities,
            resolution_presence_policy=self.resolution_presence_policy,
            int_res=self.int_res,
            numeric_policy_version=self.numeric_policy_version,
        )

    def to_audit_dict(self) -> dict[str, object]:
        payload = json.loads(self.canonical_json)
        payload.update(
            {
                "axis_keys": list(AMPLITUDE_AXIS_KEYS),
                "axis_fields": list(AMPLITUDE_AXIS_FIELDS),
                "axis_presence_mask": list(self.axis_presence_mask),
                "sha256": self.sha256,
            }
        )
        return payload


__all__ = [
    "AMPLITUDE_AXIS_FIELDS",
    "AMPLITUDE_AXIS_KEYS",
    "AMPLITUDE_AXIS_STRIDE",
    "AMPLITUDE_QUERY_EMBEDDING_DIM",
    "AMPLITUDE_QUERY_PADDING_VALUE",
    "V5_AMPLITUDE_EMBEDDING_VERSION",
    "V5_AMPLITUDE_QUERY_SCHEMA",
    "V5_AMPLITUDE_QUERY_VERSION",
    "V5AmplitudeQuery",
]
