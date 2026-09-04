"""Versioned bounds-first/local-coordinate dataset contract for Posterior V8."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from typing import Sequence

import numpy as np

from .branch_codec import (
    BRANCH_CODEC_VERSION,
    INACTIVE_UNIT_VALUE,
    ProfiledBranchCodec,
    ResolutionBounds,
    UNIT_CUBE_DIMENSIONS,
)
from .canonical_branch_catalog import canonicalize_d_flags
from .canonical_component_slots import canonicalize_component_slots
from .contract import (
    CYLINDER,
    D_DOMAIN,
    D_WIDTH_FRACTION_DOMAIN,
    H_DOMAIN,
    MAX_COMPONENTS,
    R_DOMAIN,
    RESOLUTION_NU_DOMAIN,
    RESOLUTION_SIGMA_DOMAIN,
    SIZE_WIDTH_FRACTION_DOMAIN,
    VERTICAL_CYLINDER,
    ClosedInterval,
    GuiComponentBounds,
    GuiComponentParameters,
    canonical_topology,
    full_component_bounds,
    topology_id_for,
)
from .profiled_forward import ResolutionShape
from .sobol_numeric_canonicalization_v5 import (
    V5_FAST_NUMERIC_POLICY_VERSION,
    v5_numeric_ops,
    validate_v5_numeric_policy,
)


BOUNDS_FIRST_SCHEMA_VERSION = "gisaxs.posterior_v8.bounds_first_local/v7"
BOUNDS_EMBEDDING_VERSION = "posterior_v8_policy_bound_gui_physical_bounds_78d/v2"
LOCAL_TARGET_SEMANTICS = (
    "exact_endpoint_decode_representative_of_local_unit_codec_from_presampled_gui_bounds/v3"
)
RANGE_REGIMES = ("full", "wide", "narrow")
BOUND_PLACEMENTS = (
    "interior",
    "asymmetric_low",
    "asymmetric_high",
    "edge_low",
    "edge_high",
    "partial_fixed",
)
TASK_KINDS = ("in_domain_solution", "no_solution", "ood")
COMPONENT_BOUND_AXES = ("R", "sigma_R", "h", "sigma_h", "D", "sigma_D")
BOUNDS_EMBEDDING_DIM = MAX_COMPONENTS * len(COMPONENT_BOUND_AXES) * 3 + 2 * 3

if BOUNDS_EMBEDDING_DIM != 78:  # pragma: no cover
    raise RuntimeError("bounds-first embedding layout must remain 78-dimensional")


def _log_normalized(
    value: float,
    domain: ClosedInterval,
    numeric_policy_version: str,
) -> float:
    numeric = v5_numeric_ops(numeric_policy_version)
    return float(
        (numeric.log(value) - numeric.log(domain.low))
        / (numeric.log(domain.high) - numeric.log(domain.low))
    )


def _linear_normalized(value: float, domain: ClosedInterval) -> float:
    return float((value - domain.low) / (domain.high - domain.low))


def _width_domain(shape: str, axis: str) -> ClosedInterval:
    if axis == "sigma_R":
        if shape == VERTICAL_CYLINDER:
            return SIZE_WIDTH_FRACTION_DOMAIN
        mean = R_DOMAIN
        fraction = SIZE_WIDTH_FRACTION_DOMAIN
    elif axis == "sigma_h":
        mean, fraction = H_DOMAIN, SIZE_WIDTH_FRACTION_DOMAIN
    elif axis == "sigma_D":
        mean, fraction = D_DOMAIN, D_WIDTH_FRACTION_DOMAIN
    else:  # pragma: no cover
        raise ValueError(f"unknown GUI width axis {axis}")
    return ClosedInterval(mean.low * fraction.low, mean.high * fraction.high)


def _axis_domain(shape: str, axis: str) -> tuple[ClosedInterval, str]:
    if axis == "R":
        return R_DOMAIN, "log"
    if axis == "h":
        return H_DOMAIN, "log"
    if axis == "D":
        return D_DOMAIN, "log"
    return _width_domain(shape, axis), "log"


def bounds_embedding(
    component_bounds: Sequence[GuiComponentBounds],
    resolution_bounds: ResolutionBounds | None,
    *,
    numeric_policy_version: str = V5_FAST_NUMERIC_POLICY_VERSION,
) -> tuple[float, ...]:
    """Encode actual GUI physical intervals; absent axes use ``(.5,.5,0)``."""

    bounds = tuple(component_bounds)
    numeric_policy = validate_v5_numeric_policy(numeric_policy_version)
    values: list[float] = []
    for slot in range(MAX_COMPONENTS):
        if slot >= len(bounds):
            values.extend((INACTIVE_UNIT_VALUE, INACTIVE_UNIT_VALUE, 0.0) * 6)
            continue
        item = bounds[slot]
        for axis in COMPONENT_BOUND_AXES:
            interval = getattr(item, axis)
            if interval is None:
                values.extend((INACTIVE_UNIT_VALUE, INACTIVE_UNIT_VALUE, 0.0))
                continue
            domain, transform = _axis_domain(item.shape, axis)
            if transform == "log":
                low = _log_normalized(interval.low, domain, numeric_policy)
                high = _log_normalized(interval.high, domain, numeric_policy)
            else:
                low = _linear_normalized(interval.low, domain)
                high = _linear_normalized(interval.high, domain)
            values.extend((low, high, 1.0))
    for interval, transform in (
        (
            None if resolution_bounds is None else resolution_bounds.sigma_res,
            "log",
        ),
        (None if resolution_bounds is None else resolution_bounds.nu_res, "linear"),
    ):
        if interval is None:
            values.extend((INACTIVE_UNIT_VALUE, INACTIVE_UNIT_VALUE, 0.0))
        else:
            domain = (
                RESOLUTION_SIGMA_DOMAIN if transform == "log" else RESOLUTION_NU_DOMAIN
            )
            if transform == "log":
                low = _log_normalized(interval.low, domain, numeric_policy)
                high = _log_normalized(interval.high, domain, numeric_policy)
            else:
                low = _linear_normalized(interval.low, domain)
                high = _linear_normalized(interval.high, domain)
            values.extend((low, high, 1.0))
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (BOUNDS_EMBEDDING_DIM,) or not np.all(np.isfinite(array)):
        raise RuntimeError("physical bounds embedding has an invalid shape or value")
    if np.any(array < -1e-12) or np.any(array > 1.0 + 1e-12):
        raise RuntimeError("physical bounds embedding escaped its normalized domains")
    return tuple(float(value) for value in np.clip(array, 0.0, 1.0))


def _payload(
    component_bounds,
    resolution_bounds,
    d_present,
    regime,
    placement,
    bounds_seed,
    generation_attempt,
):
    return {
        "schema": BOUNDS_FIRST_SCHEMA_VERSION,
        "embedding_version": BOUNDS_EMBEDDING_VERSION,
        "branch_codec_version": BRANCH_CODEC_VERSION,
        "bounds_seed": int(bounds_seed),
        "generation_attempt": int(generation_attempt),
        "range_regime": regime,
        "placement": placement,
        "topology": [item.shape for item in component_bounds],
        "d_present": [bool(value) for value in d_present],
        "component_bounds": [asdict(item) for item in component_bounds],
        "resolution_bounds": (
            None if resolution_bounds is None else asdict(resolution_bounds)
        ),
    }


@dataclass(frozen=True)
class BoundsProvenance:
    """Reproducible physical user bounds sampled before any truth coordinate."""

    bounds_seed: int
    generation_attempt: int
    range_regime: str
    placement: str
    component_bounds: tuple[GuiComponentBounds, ...]
    d_present: tuple[bool, ...]
    resolution_bounds: ResolutionBounds | None
    embedding: tuple[float, ...]
    canonical_json: str
    sha256: str

    @classmethod
    def create(
        cls,
        *,
        bounds_seed: int,
        generation_attempt: int,
        range_regime: str,
        placement: str,
        component_bounds: Sequence[GuiComponentBounds],
        d_present: Sequence[bool],
        resolution_bounds: ResolutionBounds | None,
    ) -> "BoundsProvenance":
        bounds = tuple(component_bounds)
        flags = tuple(bool(value) for value in d_present)
        payload = _payload(
            bounds,
            resolution_bounds,
            flags,
            range_regime,
            placement,
            bounds_seed,
            generation_attempt,
        )
        canonical = json.dumps(
            payload, allow_nan=False, separators=(",", ":"), sort_keys=True
        )
        return cls(
            int(bounds_seed),
            int(generation_attempt),
            range_regime,
            placement,
            bounds,
            flags,
            resolution_bounds,
            bounds_embedding(bounds, resolution_bounds),
            canonical,
            sha256(canonical.encode("utf-8")).hexdigest(),
        )

    def __post_init__(self) -> None:
        if self.range_regime not in RANGE_REGIMES:
            raise ValueError(f"range_regime must be one of {RANGE_REGIMES}")
        if self.placement not in BOUND_PLACEMENTS:
            raise ValueError(f"placement must be one of {BOUND_PLACEMENTS}")
        topology = tuple(item.shape for item in self.component_bounds)
        if topology != canonical_topology(topology):
            raise ValueError("component bounds must use canonical topology order")
        if len(self.d_present) != len(topology):
            raise ValueError("d_present length must match component bounds")
        padded_flags = self.d_present + (False,) * (MAX_COMPONENTS - len(topology))
        if canonicalize_d_flags(topology_id_for(topology), padded_flags) != padded_flags:
            raise ValueError(
                "d_present must use the canonical absent-before-present order "
                "within equal-shape component groups"
            )
        codec = self.local_codec()
        del codec
        payload = _payload(
            self.component_bounds,
            self.resolution_bounds,
            self.d_present,
            self.range_regime,
            self.placement,
            self.bounds_seed,
            self.generation_attempt,
        )
        canonical = json.dumps(
            payload, allow_nan=False, separators=(",", ":"), sort_keys=True
        )
        embedding = bounds_embedding(self.component_bounds, self.resolution_bounds)
        digest = sha256(canonical.encode("utf-8")).hexdigest()
        if (
            self.embedding != embedding
            or self.canonical_json != canonical
            or self.sha256 != digest
        ):
            raise ValueError("bounds provenance does not reproduce its embedding/hash")

    def local_codec(self) -> ProfiledBranchCodec:
        return ProfiledBranchCodec.build(
            tuple(item.shape for item in self.component_bounds),
            self.component_bounds,
            self.d_present,
            resolution_bounds=self.resolution_bounds,
        )

    def global_reference_codec(self) -> ProfiledBranchCodec:
        bounds = tuple(
            full_component_bounds(
                item.shape,
                d_policy="required" if present else "absent",
            )
            for item, present in zip(self.component_bounds, self.d_present)
        )
        resolution = (
            None
            if self.resolution_bounds is None
            else ResolutionBounds(RESOLUTION_SIGMA_DOMAIN, RESOLUTION_NU_DOMAIN)
        )
        return ProfiledBranchCodec.build(
            tuple(item.shape for item in bounds),
            bounds,
            self.d_present,
            resolution_bounds=resolution,
        )


def local_varying_mask(provenance: BoundsProvenance) -> tuple[bool, ...]:
    """Return codec-effective degrees of freedom, distinct from branch presence."""

    codec = provenance.local_codec()
    mask = [False] * UNIT_CUBE_DIMENSIONS
    midpoint = np.full(UNIT_CUBE_DIMENSIONS, INACTIVE_UNIT_VALUE, dtype=np.float64)
    for coordinate_index in codec.active_indices:
        canonical_endpoints = []
        for endpoint in (1.0e-12, 1.0 - 1.0e-12):
            probe = midpoint.copy()
            probe[coordinate_index] = endpoint
            latent, resolution = codec.decode(probe)
            canonical = codec.encode(latent, resolution)
            canonical_endpoints.append(canonical.unit_cube[coordinate_index])
        mask[coordinate_index] = not np.isclose(
            canonical_endpoints[0], canonical_endpoints[1], rtol=0.0, atol=1e-12
        )
    return tuple(mask)


@dataclass(frozen=True)
class BoundsFirstLabel:
    """Solution target or certified negative/OOD annotation, never both."""

    task_kind: str
    bounds: BoundsProvenance
    local_target_unit: tuple[float, ...] | None = None
    global_reference_unit: tuple[float, ...] | None = None
    truth_components: tuple[GuiComponentParameters, ...] | None = None
    truth_resolution: ResolutionShape | None = None
    annotation_reason: str | None = None
    annotation_certificate: str | None = None

    def __post_init__(self) -> None:
        if self.task_kind not in TASK_KINDS:
            raise ValueError(f"task_kind must be one of {TASK_KINDS}")
        truth_fields = (
            self.local_target_unit,
            self.global_reference_unit,
            self.truth_components,
        )
        if self.task_kind != "in_domain_solution":
            if any(value is not None for value in truth_fields) or self.truth_resolution is not None:
                raise ValueError("no-solution/OOD annotations must not carry inverse truth")
            if not self.annotation_reason or not self.annotation_certificate:
                raise ValueError("no-solution/OOD labels require reason and certificate")
            return
        if any(value is None for value in truth_fields):
            raise ValueError("in-domain solutions require local/global/physical truth")
        if self.annotation_reason is not None or self.annotation_certificate is not None:
            raise ValueError("solution targets must not carry negative/OOD annotations")
        components = tuple(self.truth_components)
        local_codec = self.bounds.local_codec()
        latent, decoded_resolution = local_codec.decode(self.local_target_unit)
        decoded_gui = local_codec.latent_components_to_gui(latent)
        if decoded_gui != components or decoded_resolution != self.truth_resolution:
            raise ValueError(
                "solution physical truth must be the exact stored local target coordinate decode"
            )
        canonical = canonicalize_component_slots(
            local_codec,
            latent,
            self.truth_resolution,
        )
        if canonical.components != latent:
            raise ValueError(
                "solution truth components must use canonical user-bounds-feasible slots"
            )
        local = canonical.coordinates
        global_coordinates = self.bounds.global_reference_codec().encode(
            latent, self.truth_resolution
        )
        if not np.allclose(local.unit_cube, self.local_target_unit, rtol=0.0, atol=1e-12):
            raise ValueError("local target is not encoded by its presampled user bounds")
        if not np.allclose(
            global_coordinates.unit_cube,
            self.global_reference_unit,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError("global reference coordinate disagrees with physical truth")


__all__ = [
    "BOUND_PLACEMENTS",
    "BOUNDS_EMBEDDING_DIM",
    "BOUNDS_EMBEDDING_VERSION",
    "BOUNDS_FIRST_SCHEMA_VERSION",
    "COMPONENT_BOUND_AXES",
    "LOCAL_TARGET_SEMANTICS",
    "RANGE_REGIMES",
    "TASK_KINDS",
    "BoundsFirstLabel",
    "BoundsProvenance",
    "bounds_embedding",
    "local_varying_mask",
]
