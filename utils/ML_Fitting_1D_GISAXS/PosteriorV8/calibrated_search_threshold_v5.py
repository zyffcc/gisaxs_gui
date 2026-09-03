"""Checked per-observation compatibility thresholds for formal V5 search.

The formal search never accepts a scalar compatibility threshold from a
command line.  It binds one immutable calibration artifact, derives the
acquisition-only stratum from metadata fixed before fitting, and performs an
exact stratum lookup only when the complete acquisition-policy ID was observed
by calibration. Missing measurement sigma, unseen strata, and unseen nuisance
policies fail closed. The pooled calibration summary is intentionally
inaccessible here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from numbers import Integral, Real
from pathlib import Path
from typing import Mapping

import numpy as np

from .compatibility_calibration import (
    CALIBRATION_SCHEMA,
    CALIBRATION_VERSION,
    COMPATIBILITY_STRATUM_FIELDS,
    COMPATIBILITY_STRATUM_VERSION,
    DESIGN_STRATUM_UNIVERSE_FIELDS,
    DESIGN_STRATUM_UNIVERSE_SEMANTICS,
    DESIGN_STRATUM_UNIVERSE_SHA256,
    DESIGN_STRATUM_UNIVERSE_VERSION,
    MEASUREMENT_SIGMA_POLICY,
    PREREGISTERED_DESIGN_STRATUM_COUNT,
    RESERVED_CALIBRATION_SPLIT_ID,
    SCORE_SEMANTICS,
    STRATIFICATION_SEMANTICS,
    CompatibilityCalibrationArtifact,
    CompatibilityStratum,
    StratumCalibration,
    acquisition_policy_payload,
    load_compatibility_calibration,
)
from .evaluation import STANDARDIZED_LOG_RMSE_METRIC
from .grouped_artifact_v5 import canonical_json
from .observation_v5 import V5ObservationDataView


V5_CALIBRATION_IDENTITY_SCHEMA = "gisaxs.posterior_v8.checked_compatibility_calibration_identity/v2"
V5_CALIBRATION_IDENTITY_VERSION = (
    "posterior_v8_reserved_split_universe_logical_and_file_digest_binding_v2"
)
V5_OBSERVATION_THRESHOLD_SCHEMA = "gisaxs.posterior_v8.calibrated_observation_threshold/v2"
V5_OBSERVATION_THRESHOLD_VERSION = (
    "posterior_v8_prefit_full_acquisition_policy_supported_exact_lookup_v2"
)
V5_CALIBRATED_THRESHOLD_NAME = "reserved_split_conformal_standardized_log_rmse_acquisition_stratum"

_IDENTITY_FIELDS = frozenset(
    {
        "artifact_sha256",
        "file_sha256",
        "input_sha256",
        "dataset_manifest_sha256",
        "calibration_split_id",
        "calibration_split_sha256",
        "calibration_schema",
        "calibration_version",
        "target_coverage",
        "compatibility_stratum_version",
        "compatibility_stratum_fields",
        "preregistered_design_stratum_count",
        "design_stratum_universe_version",
        "design_stratum_universe_fields",
        "design_stratum_universe_semantics",
        "design_stratum_universe_sha256",
        "stratification_semantics",
        "score_semantics",
        "measurement_sigma_policy",
        "schema",
        "version",
    }
)
_THRESHOLD_FIELDS = frozenset(
    {
        "calibration_identity",
        "stratum",
        "acquisition_policy_id_sha256",
        "stratum_calibration_sha256",
        "metric_name",
        "threshold_name",
        "threshold_value",
        "sample_count",
        "order_statistic",
        "guaranteed_coverage",
        "threshold_source_id",
        "schema",
        "version",
    }
)


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _digest(value: object, name: str) -> str:
    result = _text(value, name).lower()
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return result


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be positive")
    return result


def _finite(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be finite")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _exact_mapping(value: object, fields: frozenset[str], name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or frozenset(value) != fields:
        raise ValueError(f"{name} has unsupported fields")
    return value


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True, kw_only=True)
class V5CalibrationArtifactIdentity:
    """Portable identity of the exact calibration artifact used by a run."""

    artifact_sha256: str
    file_sha256: str
    input_sha256: str
    dataset_manifest_sha256: str
    calibration_split_id: str
    calibration_split_sha256: str
    calibration_schema: str
    calibration_version: str
    target_coverage: float
    compatibility_stratum_version: str
    compatibility_stratum_fields: tuple[str, ...]
    stratification_semantics: str
    score_semantics: str
    measurement_sigma_policy: str
    preregistered_design_stratum_count: int = PREREGISTERED_DESIGN_STRATUM_COUNT
    design_stratum_universe_version: str = DESIGN_STRATUM_UNIVERSE_VERSION
    design_stratum_universe_fields: tuple[str, ...] = DESIGN_STRATUM_UNIVERSE_FIELDS
    design_stratum_universe_semantics: str = DESIGN_STRATUM_UNIVERSE_SEMANTICS
    design_stratum_universe_sha256: str = DESIGN_STRATUM_UNIVERSE_SHA256
    schema: str = V5_CALIBRATION_IDENTITY_SCHEMA
    version: str = V5_CALIBRATION_IDENTITY_VERSION

    def __post_init__(self) -> None:
        for name in (
            "artifact_sha256",
            "file_sha256",
            "input_sha256",
            "dataset_manifest_sha256",
            "calibration_split_sha256",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        split_id = _text(self.calibration_split_id, "calibration_split_id")
        if split_id != RESERVED_CALIBRATION_SPLIT_ID:
            raise ValueError("formal calibration must bind the reserved calibration split")
        coverage = _finite(self.target_coverage, "target_coverage")
        if not 0.0 < coverage < 1.0:
            raise ValueError("target_coverage must be in (0, 1)")
        fields = tuple(self.compatibility_stratum_fields)
        universe_count = _positive_integer(
            self.preregistered_design_stratum_count,
            "preregistered_design_stratum_count",
        )
        universe_fields = tuple(self.design_stratum_universe_fields)
        universe_sha256 = _digest(
            self.design_stratum_universe_sha256,
            "design_stratum_universe_sha256",
        )
        expected = {
            "calibration_schema": CALIBRATION_SCHEMA,
            "calibration_version": CALIBRATION_VERSION,
            "compatibility_stratum_version": COMPATIBILITY_STRATUM_VERSION,
            "compatibility_stratum_fields": COMPATIBILITY_STRATUM_FIELDS,
            "preregistered_design_stratum_count": PREREGISTERED_DESIGN_STRATUM_COUNT,
            "design_stratum_universe_version": DESIGN_STRATUM_UNIVERSE_VERSION,
            "design_stratum_universe_fields": DESIGN_STRATUM_UNIVERSE_FIELDS,
            "design_stratum_universe_semantics": DESIGN_STRATUM_UNIVERSE_SEMANTICS,
            "design_stratum_universe_sha256": DESIGN_STRATUM_UNIVERSE_SHA256,
            "stratification_semantics": STRATIFICATION_SEMANTICS,
            "score_semantics": SCORE_SEMANTICS,
            "measurement_sigma_policy": MEASUREMENT_SIGMA_POLICY,
            "schema": V5_CALIBRATION_IDENTITY_SCHEMA,
            "version": V5_CALIBRATION_IDENTITY_VERSION,
        }
        observed = {
            **self.__dict__,
            "compatibility_stratum_fields": fields,
            "preregistered_design_stratum_count": universe_count,
            "design_stratum_universe_fields": universe_fields,
            "design_stratum_universe_sha256": universe_sha256,
        }
        if any(observed[name] != value for name, value in expected.items()):
            raise ValueError("calibration identity has an incompatible scientific contract")
        object.__setattr__(self, "calibration_split_id", split_id)
        object.__setattr__(self, "target_coverage", coverage)
        object.__setattr__(self, "compatibility_stratum_fields", fields)
        object.__setattr__(self, "preregistered_design_stratum_count", universe_count)
        object.__setattr__(self, "design_stratum_universe_fields", universe_fields)
        object.__setattr__(self, "design_stratum_universe_sha256", universe_sha256)

    @classmethod
    def from_payload(cls, value: object) -> "V5CalibrationArtifactIdentity":
        return cls(**dict(_exact_mapping(value, _IDENTITY_FIELDS, "calibration identity")))

    def audit_payload(self) -> dict[str, object]:
        return asdict(self)

    @property
    def sha256(self) -> str:
        return sha256(canonical_json(self.audit_payload()).encode("utf-8")).hexdigest()


@dataclass(frozen=True, kw_only=True)
class V5CheckedCompatibilityCalibration:
    """In-memory artifact paired with its checked logical and file identity."""

    artifact: CompatibilityCalibrationArtifact
    identity: V5CalibrationArtifactIdentity

    def __post_init__(self) -> None:
        if not isinstance(self.artifact, CompatibilityCalibrationArtifact):
            raise TypeError("artifact must be a CompatibilityCalibrationArtifact")
        if not isinstance(self.identity, V5CalibrationArtifactIdentity):
            raise TypeError("identity must be a V5CalibrationArtifactIdentity")
        expected = _identity_from_artifact(self.artifact, file_sha256=self.identity.file_sha256)
        if self.identity != expected:
            raise ValueError("calibration artifact does not reproduce its checked identity")


def _identity_from_artifact(
    artifact: CompatibilityCalibrationArtifact,
    *,
    file_sha256: str,
) -> V5CalibrationArtifactIdentity:
    return V5CalibrationArtifactIdentity(
        artifact_sha256=artifact.sha256,
        file_sha256=file_sha256,
        input_sha256=artifact.input_sha256,
        dataset_manifest_sha256=artifact.dataset_manifest_sha256,
        calibration_split_id=artifact.calibration_split_id,
        calibration_split_sha256=artifact.calibration_split_sha256,
        calibration_schema=artifact.schema,
        calibration_version=artifact.calibration_version,
        target_coverage=artifact.target_coverage,
        compatibility_stratum_version=artifact.compatibility_stratum_version,
        compatibility_stratum_fields=artifact.compatibility_stratum_fields,
        preregistered_design_stratum_count=artifact.preregistered_design_stratum_count,
        design_stratum_universe_version=artifact.design_stratum_universe_version,
        design_stratum_universe_fields=artifact.design_stratum_universe_fields,
        design_stratum_universe_semantics=artifact.design_stratum_universe_semantics,
        design_stratum_universe_sha256=artifact.design_stratum_universe_sha256,
        stratification_semantics=artifact.stratification_semantics,
        score_semantics=artifact.score_semantics,
        measurement_sigma_policy=artifact.measurement_sigma_policy,
    )


def inspect_v5_compatibility_calibration(
    path: str | Path,
) -> V5CheckedCompatibilityCalibration:
    """Strictly read an artifact and compute identities for launch planning."""

    selected = Path(path)
    before = _file_sha256(selected)
    artifact = load_compatibility_calibration(selected)
    after = _file_sha256(selected)
    if before != after:
        raise RuntimeError("calibration artifact changed while it was being inspected")
    identity = _identity_from_artifact(artifact, file_sha256=after)
    return V5CheckedCompatibilityCalibration(artifact=artifact, identity=identity)


def read_v5_checked_compatibility_calibration(
    path: str | Path,
    *,
    expected_artifact_sha256: str,
    expected_file_sha256: str,
) -> V5CheckedCompatibilityCalibration:
    """Read a launch-bound artifact and reject any logical or byte drift."""

    checked = inspect_v5_compatibility_calibration(path)
    if checked.identity.artifact_sha256 != _digest(
        expected_artifact_sha256, "expected_artifact_sha256"
    ):
        raise ValueError("calibration logical artifact SHA-256 changed after planning")
    if checked.identity.file_sha256 != _digest(expected_file_sha256, "expected_file_sha256"):
        raise ValueError("calibration file SHA-256 changed after planning")
    return checked


def compatibility_stratum_from_v5_observation(
    view: V5ObservationDataView,
) -> CompatibilityStratum:
    """Derive the primary stratum only from acquisition metadata fixed pre-fit."""

    if not isinstance(view, V5ObservationDataView):
        raise TypeError("view must be a V5ObservationDataView")
    policy = acquisition_policy_payload(view.acquisition_policy_id)
    grid = policy["grid"]
    sigma = policy["sigma"]
    if grid["design_point_count"] != view.design_point_count:
        raise ValueError("observation acquisition policy changed its design point count")
    return CompatibilityStratum(
        point_count=int(grid["design_point_count"]),
        noise_id=str(sigma["noise_id"]),
        q_window_id=str(grid["q_window_id"]),
    )


def _stratum_calibration_sha256(value: StratumCalibration) -> str:
    return sha256(canonical_json(asdict(value)).encode("utf-8")).hexdigest()


@dataclass(frozen=True, kw_only=True)
class V5CalibratedObservationThreshold:
    """One exact per-observation lookup, including all replay provenance."""

    calibration_identity: V5CalibrationArtifactIdentity
    stratum: CompatibilityStratum
    acquisition_policy_id_sha256: str
    stratum_calibration_sha256: str
    metric_name: str
    threshold_name: str
    threshold_value: float
    sample_count: int
    order_statistic: int
    guaranteed_coverage: float
    threshold_source_id: str
    schema: str = V5_OBSERVATION_THRESHOLD_SCHEMA
    version: str = V5_OBSERVATION_THRESHOLD_VERSION

    def __post_init__(self) -> None:
        identity = self.calibration_identity
        if isinstance(identity, Mapping):
            identity = V5CalibrationArtifactIdentity.from_payload(identity)
        if not isinstance(identity, V5CalibrationArtifactIdentity):
            raise TypeError("calibration_identity has an invalid type")
        stratum = self.stratum
        if isinstance(stratum, Mapping):
            stratum = CompatibilityStratum(**dict(stratum))
        if not isinstance(stratum, CompatibilityStratum):
            raise TypeError("stratum has an invalid type")
        acquisition_sha = _digest(self.acquisition_policy_id_sha256, "acquisition_policy_id_sha256")
        stratum_sha = _digest(self.stratum_calibration_sha256, "stratum_calibration_sha256")
        threshold = _finite(self.threshold_value, "threshold_value")
        coverage = _finite(self.guaranteed_coverage, "guaranteed_coverage")
        count = _positive_integer(self.sample_count, "sample_count")
        order = _positive_integer(self.order_statistic, "order_statistic")
        if (
            threshold < 0.0
            or order > count
            or coverage != order / float(count + 1)
            or coverage < identity.target_coverage
        ):
            raise ValueError("calibrated observation threshold statistics are inconsistent")
        if self.metric_name != STANDARDIZED_LOG_RMSE_METRIC:
            raise ValueError("formal compatibility requires standardized log RMSE")
        if self.threshold_name != V5_CALIBRATED_THRESHOLD_NAME:
            raise ValueError("formal compatibility threshold name is incompatible")
        source = (
            f"compatibility-calibration/{identity.artifact_sha256}/"
            f"stratum/{sha256(canonical_json(asdict(stratum)).encode('utf-8')).hexdigest()}"
        )
        if self.threshold_source_id != source:
            raise ValueError("threshold source ID does not reproduce artifact/stratum identity")
        if self.schema != V5_OBSERVATION_THRESHOLD_SCHEMA or self.version != (
            V5_OBSERVATION_THRESHOLD_VERSION
        ):
            raise ValueError("unsupported calibrated observation-threshold contract")
        object.__setattr__(self, "calibration_identity", identity)
        object.__setattr__(self, "stratum", stratum)
        object.__setattr__(self, "acquisition_policy_id_sha256", acquisition_sha)
        object.__setattr__(self, "stratum_calibration_sha256", stratum_sha)
        object.__setattr__(self, "threshold_value", threshold)
        object.__setattr__(self, "sample_count", count)
        object.__setattr__(self, "order_statistic", order)
        object.__setattr__(self, "guaranteed_coverage", coverage)

    @classmethod
    def from_payload(cls, value: object) -> "V5CalibratedObservationThreshold":
        payload = dict(_exact_mapping(value, _THRESHOLD_FIELDS, "observation threshold"))
        payload["calibration_identity"] = V5CalibrationArtifactIdentity.from_payload(
            payload["calibration_identity"]
        )
        payload["stratum"] = CompatibilityStratum(**dict(payload["stratum"]))
        return cls(**payload)

    def audit_payload(self) -> dict[str, object]:
        return asdict(self)

    @property
    def sha256(self) -> str:
        return sha256(canonical_json(self.audit_payload()).encode("utf-8")).hexdigest()


def bind_v5_calibrated_observation_threshold(
    calibration: V5CheckedCompatibilityCalibration,
    view: V5ObservationDataView,
) -> V5CalibratedObservationThreshold:
    """Bind an observed full policy within one exact stratum; never use fallback."""

    if not isinstance(calibration, V5CheckedCompatibilityCalibration):
        raise TypeError("calibration must be a V5CheckedCompatibilityCalibration")
    if not isinstance(view, V5ObservationDataView):
        raise TypeError("view must be a V5ObservationDataView")
    if view.acceptance_sigma_log is None:
        raise ValueError("formal calibrated search requires measured or simulated acceptance sigma")
    stratum = compatibility_stratum_from_v5_observation(view)
    calibrated = calibration.artifact.threshold_for_acquisition_policy(
        stratum,
        view.acquisition_policy_id,
    )
    entry = next(value for value in calibration.artifact.strata if value.stratum == stratum)
    stratum_identity = sha256(canonical_json(asdict(stratum)).encode("utf-8")).hexdigest()
    return V5CalibratedObservationThreshold(
        calibration_identity=calibration.identity,
        stratum=stratum,
        acquisition_policy_id_sha256=sha256(view.acquisition_policy_id.encode("utf-8")).hexdigest(),
        stratum_calibration_sha256=_stratum_calibration_sha256(entry),
        metric_name=STANDARDIZED_LOG_RMSE_METRIC,
        threshold_name=V5_CALIBRATED_THRESHOLD_NAME,
        threshold_value=calibrated.threshold,
        sample_count=calibrated.sample_count,
        order_statistic=calibrated.order_statistic,
        guaranteed_coverage=calibrated.guaranteed_coverage,
        threshold_source_id=(
            f"compatibility-calibration/{calibration.identity.artifact_sha256}/"
            f"stratum/{stratum_identity}"
        ),
    )


__all__ = [
    "V5CalibrationArtifactIdentity",
    "V5CalibratedObservationThreshold",
    "V5CheckedCompatibilityCalibration",
    "V5_CALIBRATED_THRESHOLD_NAME",
    "V5_CALIBRATION_IDENTITY_SCHEMA",
    "V5_CALIBRATION_IDENTITY_VERSION",
    "V5_OBSERVATION_THRESHOLD_SCHEMA",
    "V5_OBSERVATION_THRESHOLD_VERSION",
    "bind_v5_calibrated_observation_threshold",
    "compatibility_stratum_from_v5_observation",
    "inspect_v5_compatibility_calibration",
    "read_v5_checked_compatibility_calibration",
]
