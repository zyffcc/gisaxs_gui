"""Truth-independent physical amplitude-range samplers for V5.2.

These utilities choose the complete GUI amplitude query before any generating
coefficient is drawn.  Every active amplitude axis owns an independent range
regime.  These are deterministic engineering samplers; the paper dataset
builder must feed its named Sobol coordinates directly rather than turning a
Sobol point into only a pseudorandom seed.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from numbers import Integral
from typing import Sequence

import numpy as np

from .amplitude_query_v5 import V5AmplitudeQuery
from .contextual_branch_catalog import PRESENCE_POLICIES, PresencePolicy
from .contract import MAX_COMPONENTS, ClosedInterval


V5_AMPLITUDE_QUERY_SAMPLER_VERSION = (
    "posterior_v8_truth_independent_per_axis_gui_amplitude_range_sampler_v3"
)
V5_AMPLITUDE_RANGE_ASSIGNMENT_SCHEMA = (
    "gisaxs.posterior_v8.amplitude_axis_range_regime_assignment/v1"
)
V5_AMPLITUDE_RANGE_ASSIGNMENT_VERSION = (
    "posterior_v8_bg_k_each_int_and_int_res_independent_range_regimes_v1"
)
V5_AMPLITUDE_RANGE_REGIMES = (
    "full",
    "wide",
    "narrow",
    "fixed",
    "edge_low",
    "edge_high",
)

# These are the non-negative ranges exposed by the current advanced fitting
# constraints, extended to include the exact canonical zero boundary.
V5_BACKGROUND_DOMAIN = ClosedInterval(0.0, 1.0e8)
V5_K_DOMAIN = ClosedInterval(1.0e-2, 1.0e8)
V5_COMPONENT_INTENSITY_DOMAIN = ClosedInterval(0.0, 1.0e8)
V5_INT_RES_DOMAIN = ClosedInterval(0.0, 1.0e8)
_POSITIVE_AMPLITUDE_FLOOR = 1.0e-18
_AMPLITUDE_AXIS_NAMES = (
    "BG",
    "k",
    *(f"Int_{slot + 1}" for slot in range(MAX_COMPONENTS)),
    "int_Res",
)


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _count(value: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError("component_count must be an integer")
    result = int(value)
    if not 1 <= result <= MAX_COMPONENTS:
        raise ValueError(f"component_count must be in [1, {MAX_COMPONENTS}]")
    return result


def _seed(value: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError("query_seed must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError("query_seed must be non-negative")
    return result


def _policy(value: PresencePolicy) -> PresencePolicy:
    if not isinstance(value, str):
        raise TypeError("resolution_presence_policy must be a string")
    if value not in PRESENCE_POLICIES:
        raise ValueError("resolution_presence_policy must be absent, optional, or required")
    return value


def amplitude_range_regime_for(value: str | None, *, query_seed: int) -> str:
    """Resolve one explicitly broadcast range regime.

    This compatibility helper remains useful for deliberately homogeneous
    engineering fixtures.  Production query sampling uses
    :func:`amplitude_axis_range_regimes_for` so no correlation is imposed
    between otherwise independent GUI axes.
    """

    seed = _seed(query_seed)
    selected = (
        V5_AMPLITUDE_RANGE_REGIMES[seed % len(V5_AMPLITUDE_RANGE_REGIMES)]
        if value is None
        else value
    )
    if selected not in V5_AMPLITUDE_RANGE_REGIMES:
        raise ValueError(f"range_regime must be one of {V5_AMPLITUDE_RANGE_REGIMES}")
    return selected


def _range_regime(value: object, name: str) -> str:
    if not isinstance(value, str) or value not in V5_AMPLITUDE_RANGE_REGIMES:
        raise ValueError(f"{name} must be one of {V5_AMPLITUDE_RANGE_REGIMES}")
    return value


@dataclass(frozen=True)
class V5AmplitudeRangeRegimes:
    """Complete active/inactive assignment for the seven GUI amplitude axes.

    Component slots that are absent from the selected topology and an absent
    Resolution range are represented by ``None``.  This makes inactivity
    explicit without erasing those axes from the frozen Sobol dictionary.
    """

    background: str
    k: str
    component_intensities: tuple[str | None, ...]
    int_res: str | None
    schema_version: str = V5_AMPLITUDE_RANGE_ASSIGNMENT_SCHEMA
    version: str = V5_AMPLITUDE_RANGE_ASSIGNMENT_VERSION

    @classmethod
    def create(
        cls,
        component_count: int,
        *,
        resolution_presence_policy: PresencePolicy,
        background: str,
        k: str,
        component_intensities: Sequence[str],
        int_res: str | None,
    ) -> "V5AmplitudeRangeRegimes":
        count = _count(component_count)
        policy = _policy(resolution_presence_policy)
        if isinstance(component_intensities, (str, bytes)):
            raise TypeError("component_intensities regimes must be a sequence")
        try:
            supplied = tuple(component_intensities)
        except TypeError as exc:
            raise TypeError("component_intensities regimes must be a sequence") from exc
        if len(supplied) != count:
            raise ValueError("component_intensities regimes must match component_count")
        if policy == "absent":
            if int_res is not None:
                raise ValueError("Resolution-absent range assignment must omit int_res")
            resolution_regime = None
        else:
            resolution_regime = _range_regime(int_res, "int_res")
        padded = tuple(
            _range_regime(value, f"component_intensities[{index}]")
            for index, value in enumerate(supplied, 1)
        ) + (None,) * (MAX_COMPONENTS - count)
        return cls(
            background=_range_regime(background, "background"),
            k=_range_regime(k, "k"),
            component_intensities=padded,
            int_res=resolution_regime,
        )

    def __post_init__(self) -> None:
        _range_regime(self.background, "background")
        _range_regime(self.k, "k")
        slots = tuple(self.component_intensities)
        if len(slots) != MAX_COMPONENTS:
            raise ValueError(
                f"component_intensities must retain exactly {MAX_COMPONENTS} slot assignments"
            )
        seen_inactive = False
        active_count = 0
        for index, value in enumerate(slots, 1):
            if value is None:
                seen_inactive = True
                continue
            if seen_inactive:
                raise ValueError("active component range regimes must form a leading prefix")
            _range_regime(value, f"component_intensities[{index}]")
            active_count += 1
        _count(active_count)
        if self.int_res is not None:
            _range_regime(self.int_res, "int_res")
        if self.schema_version != V5_AMPLITUDE_RANGE_ASSIGNMENT_SCHEMA:
            raise ValueError("unsupported amplitude range-assignment schema")
        if self.version != V5_AMPLITUDE_RANGE_ASSIGNMENT_VERSION:
            raise ValueError("unsupported amplitude range-assignment version")
        object.__setattr__(self, "component_intensities", slots)

    @property
    def component_count(self) -> int:
        return sum(value is not None for value in self.component_intensities)

    @property
    def active_component_intensities(self) -> tuple[str, ...]:
        return tuple(
            value for value in self.component_intensities if value is not None
        )

    @property
    def axis_regimes(self) -> tuple[str | None, ...]:
        return (
            self.background,
            self.k,
            *self.component_intensities,
            self.int_res,
        )

    @property
    def summary(self) -> str:
        active = tuple(value for value in self.axis_regimes if value is not None)
        return active[0] if len(set(active)) == 1 else "mixed"

    @property
    def active_axis_names(self) -> tuple[str, ...]:
        return tuple(
            name
            for name, regime in zip(_AMPLITUDE_AXIS_NAMES, self.axis_regimes, strict=True)
            if regime is not None
        )

    @property
    def inactive_axis_names(self) -> tuple[str, ...]:
        return tuple(
            name
            for name, regime in zip(_AMPLITUDE_AXIS_NAMES, self.axis_regimes, strict=True)
            if regime is None
        )

    def audit_payload(self) -> dict[str, object]:
        axis_regimes = dict(
            zip(_AMPLITUDE_AXIS_NAMES, self.axis_regimes, strict=True)
        )
        return {
            "schema": self.schema_version,
            "version": self.version,
            "policy": "independent_per_active_axis",
            "summary": self.summary,
            "axis_regimes": axis_regimes,
            "active_axis_names": list(self.active_axis_names),
            "inactive_axis_names": list(self.inactive_axis_names),
            "component_count": self.component_count,
            "normalization": "none",
        }

    @property
    def canonical_json(self) -> str:
        return _canonical_json(self.audit_payload())

    @property
    def sha256(self) -> str:
        return sha256(self.canonical_json.encode("utf-8")).hexdigest()


def _axis_rng(query_seed: int, axis_index: int, purpose: int) -> np.random.Generator:
    return np.random.default_rng(
        np.random.SeedSequence([query_seed, axis_index, purpose, 0x56354151])
    )


def amplitude_axis_range_regimes_for(
    component_count: int,
    *,
    resolution_presence_policy: PresencePolicy,
    query_seed: int,
    range_regime: str | None = None,
    range_regimes: V5AmplitudeRangeRegimes | None = None,
) -> V5AmplitudeRangeRegimes:
    """Resolve replayable independent regimes for every active amplitude axis.

    ``range_regime`` is an explicit broadcast override for fixtures.  Supplying
    neither override samples each active axis from its own deterministic random
    stream.  A complete ``range_regimes`` assignment is the unambiguous mixed
    query interface.
    """

    count = _count(component_count)
    policy = _policy(resolution_presence_policy)
    seed = _seed(query_seed)
    if range_regime is not None and range_regimes is not None:
        raise ValueError("range_regime and range_regimes are mutually exclusive")
    if range_regimes is not None:
        if not isinstance(range_regimes, V5AmplitudeRangeRegimes):
            raise TypeError("range_regimes must be a V5AmplitudeRangeRegimes")
        if range_regimes.component_count != count:
            raise ValueError("range_regimes component count does not match component_count")
        if (range_regimes.int_res is None) != (policy == "absent"):
            raise ValueError("range_regimes Resolution activity disagrees with policy")
        return range_regimes
    if range_regime is not None:
        selected = amplitude_range_regime_for(range_regime, query_seed=seed)
        values = (selected,) * len(_AMPLITUDE_AXIS_NAMES)
    else:
        values = tuple(
            V5_AMPLITUDE_RANGE_REGIMES[
                int(
                    _axis_rng(seed, axis_index, 0x52454749).integers(
                        len(V5_AMPLITUDE_RANGE_REGIMES)
                    )
                )
            ]
            for axis_index in range(len(_AMPLITUDE_AXIS_NAMES))
        )
    return V5AmplitudeRangeRegimes.create(
        count,
        resolution_presence_policy=policy,
        background=values[0],
        k=values[1],
        component_intensities=values[2 : 2 + count],
        int_res=None if policy == "absent" else values[-1],
    )


def _log_interval(
    rng: np.random.Generator,
    domain: ClosedInterval,
    regime: str,
) -> ClosedInterval:
    positive_low = _POSITIVE_AMPLITUDE_FLOOR if domain.low == 0.0 else domain.low
    log_low, log_high = np.log(positive_low), np.log(domain.high)
    if regime == "full":
        return domain
    if regime == "fixed":
        position = float(rng.uniform(0.02, 0.98))
        start = stop = position
    else:
        width = (
            float(rng.uniform(0.40, 0.80))
            if regime == "wide"
            else float(rng.uniform(0.03, 0.18))
        )
        available = 1.0 - width
        if regime == "edge_low":
            start = 0.0
        elif regime == "edge_high":
            start = available
        else:
            start = float(rng.uniform(0.0, available))
        stop = start + width
    low = float(np.exp(log_low + start * (log_high - log_low)))
    high = float(np.exp(log_low + stop * (log_high - log_low)))
    if regime == "edge_low":
        low = domain.low
    if regime == "edge_high":
        high = domain.high
    return ClosedInterval(max(domain.low, low), min(domain.high, high))


def sample_v5_amplitude_query(
    component_count: int,
    *,
    resolution_presence_policy: PresencePolicy,
    query_seed: int,
    range_regime: str | None = None,
    range_regimes: V5AmplitudeRangeRegimes | None = None,
) -> V5AmplitudeQuery:
    """Choose ranges without reading a curve, branch, or generating truth."""

    count = _count(component_count)
    seed = _seed(query_seed)
    policy = _policy(resolution_presence_policy)
    selected = amplitude_axis_range_regimes_for(
        count,
        resolution_presence_policy=policy,
        query_seed=seed,
        range_regime=range_regime,
        range_regimes=range_regimes,
    )
    background = _log_interval(
        _axis_rng(seed, 0, 0x494E5456),
        V5_BACKGROUND_DOMAIN,
        selected.background,
    )
    k = _log_interval(
        _axis_rng(seed, 1, 0x494E5456),
        V5_K_DOMAIN,
        selected.k,
    )
    intensities = tuple(
        _log_interval(
            _axis_rng(seed, slot + 2, 0x494E5456),
            V5_COMPONENT_INTENSITY_DOMAIN,
            regime,
        )
        for slot, regime in enumerate(selected.active_component_intensities)
    )
    int_res = (
        None
        if policy == "absent"
        else _log_interval(
            _axis_rng(seed, len(_AMPLITUDE_AXIS_NAMES) - 1, 0x494E5456),
            V5_INT_RES_DOMAIN,
            selected.int_res,
        )
    )
    return V5AmplitudeQuery.create(
        background=background,
        k=k,
        component_intensities=intensities,
        resolution_presence_policy=policy,
        int_res=int_res,
    )


def full_range_v5_amplitude_query(
    component_count: int,
    *,
    resolution_presence_policy: PresencePolicy = "optional",
) -> V5AmplitudeQuery:
    """Return the complete versioned amplitude domain for product fallback."""

    count = _count(component_count)
    policy = _policy(resolution_presence_policy)
    return V5AmplitudeQuery.create(
        background=V5_BACKGROUND_DOMAIN,
        k=V5_K_DOMAIN,
        component_intensities=(V5_COMPONENT_INTENSITY_DOMAIN,) * count,
        resolution_presence_policy=policy,
        int_res=None if policy == "absent" else V5_INT_RES_DOMAIN,
    )


__all__ = [
    "V5_AMPLITUDE_QUERY_SAMPLER_VERSION",
    "V5_AMPLITUDE_RANGE_ASSIGNMENT_SCHEMA",
    "V5_AMPLITUDE_RANGE_ASSIGNMENT_VERSION",
    "V5_AMPLITUDE_RANGE_REGIMES",
    "V5_BACKGROUND_DOMAIN",
    "V5_COMPONENT_INTENSITY_DOMAIN",
    "V5_INT_RES_DOMAIN",
    "V5_K_DOMAIN",
    "V5AmplitudeRangeRegimes",
    "amplitude_axis_range_regimes_for",
    "amplitude_range_regime_for",
    "full_range_v5_amplitude_query",
    "sample_v5_amplitude_query",
]
