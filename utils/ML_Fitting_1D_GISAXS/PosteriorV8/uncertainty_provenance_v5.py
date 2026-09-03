"""Separate encoder uncertainty features from acceptance uncertainty in V5."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


V5_UNCERTAINTY_SCHEMA = "gisaxs.posterior_v8.uncertainty_provenance/v1"
V5_UNCERTAINTY_VERSION = "posterior_v8_measurement_simulation_proxy_separation_v1"
UNCERTAINTY_KINDS = (
    "measured_sigma",
    "simulated_sigma",
    "encoder_proxy_missing_sigma",
)
UNCERTAINTY_FEATURE_DIM = len(UNCERTAINTY_KINDS)


@dataclass(frozen=True)
class V5UncertaintyProvenance:
    """One auditable source for the encoder's sigma feature."""

    kind: str
    encoder_relative_sigma_proxy: float | None = None
    schema_version: str = V5_UNCERTAINTY_SCHEMA
    version: str = V5_UNCERTAINTY_VERSION

    def __post_init__(self) -> None:
        if self.kind not in UNCERTAINTY_KINDS:
            raise ValueError(f"kind must be one of {UNCERTAINTY_KINDS}")
        proxy = self.encoder_relative_sigma_proxy
        if self.kind == "encoder_proxy_missing_sigma":
            if proxy is None or not np.isfinite(float(proxy)) or not 0.0 < float(proxy) <= 1.0:
                raise ValueError("missing-sigma encoder proxy must lie in (0, 1]")
            object.__setattr__(self, "encoder_relative_sigma_proxy", float(proxy))
        elif proxy is not None:
            raise ValueError("measured/simulated sigma provenance must not define a proxy")
        if self.schema_version != V5_UNCERTAINTY_SCHEMA:
            raise ValueError("unsupported V5 uncertainty schema")
        if self.version != V5_UNCERTAINTY_VERSION:
            raise ValueError("unsupported V5 uncertainty provenance version")

    @property
    def measurement_sigma_available(self) -> bool:
        return self.kind != "encoder_proxy_missing_sigma"

    @property
    def feature_vector(self) -> tuple[float, ...]:
        return tuple(float(value == self.kind) for value in UNCERTAINTY_KINDS)

    def audit_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "version": self.version,
            "kind": self.kind,
            "encoder_relative_sigma_proxy": self.encoder_relative_sigma_proxy,
            "measurement_sigma_available": self.measurement_sigma_available,
            "compatibility_coverage_claim_allowed": self.measurement_sigma_available,
        }


def prepare_encoder_sigma(
    intensity,
    measurement_sigma,
    provenance: V5UncertaintyProvenance,
) -> np.ndarray:
    """Return positive encoder sigma without fabricating measurement evidence."""

    if not isinstance(provenance, V5UncertaintyProvenance):
        raise TypeError("provenance must be V5UncertaintyProvenance")
    observed = np.asarray(intensity, dtype=np.float64)
    if observed.ndim != 1 or observed.size == 0:
        raise ValueError("intensity must be a non-empty vector")
    if not np.all(np.isfinite(observed)) or np.any(observed <= 0.0):
        raise ValueError("intensity must contain finite positive values")
    if provenance.measurement_sigma_available:
        if measurement_sigma is None:
            raise ValueError("measured/simulated uncertainty requires measurement_sigma")
        sigma = np.asarray(measurement_sigma, dtype=np.float64)
        if sigma.shape != observed.shape:
            raise ValueError("measurement_sigma must match intensity")
        if not np.all(np.isfinite(sigma)) or np.any(sigma <= 0.0):
            raise ValueError("measurement_sigma must contain finite positive values")
    else:
        if measurement_sigma is not None:
            raise ValueError("missing-sigma proxy provenance must not receive measurement_sigma")
        sigma = observed * float(provenance.encoder_relative_sigma_proxy)
        sigma = np.maximum(sigma, np.finfo(np.float64).tiny)
    result = np.array(sigma, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


def acceptance_sigma_log(
    intensity,
    measurement_sigma,
    provenance: V5UncertaintyProvenance,
) -> np.ndarray | None:
    """Return measurement sigma in log space, or ``None`` for encoder proxies."""

    if not isinstance(provenance, V5UncertaintyProvenance):
        raise TypeError("provenance must be V5UncertaintyProvenance")
    if not provenance.measurement_sigma_available:
        prepare_encoder_sigma(intensity, measurement_sigma, provenance)
        return None
    observed = np.asarray(intensity, dtype=np.float64)
    sigma = prepare_encoder_sigma(observed, measurement_sigma, provenance)
    result = np.asarray(sigma / observed, dtype=np.float64)
    if not np.all(np.isfinite(result)) or np.any(result <= 0.0):
        raise ValueError("derived sigma_log is invalid")
    result.setflags(write=False)
    return result


__all__ = [
    "UNCERTAINTY_FEATURE_DIM",
    "UNCERTAINTY_KINDS",
    "V5_UNCERTAINTY_SCHEMA",
    "V5_UNCERTAINTY_VERSION",
    "V5UncertaintyProvenance",
    "acceptance_sigma_log",
    "prepare_encoder_sigma",
]
