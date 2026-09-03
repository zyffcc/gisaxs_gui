"""Finite-sample compatibility-score calibration for Posterior V8.

The calibrated quantity is a standardized weighted log-residual compatibility
score. It is a pseudo-likelihood diagnostic, not a strict likelihood or a
posterior probability.  The primary estimand admits exactly one deterministic
observation view for each clean-recipe/acquisition-stratum pair; repeated views
of the same pair fail closed instead of being combined. Thresholds then use the
one-sided split-conformal order statistic ceil((n + 1) * target_coverage).

The primary paper policy deliberately conditions only on acquisition metadata
available before fitting. Its coverage statement is therefore marginal over
the generating model class/K inside each acquisition stratum and does not
calibrate candidate-topology precision or cross-K discovery yield.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from collections import Counter
import hashlib
import json
from math import ceil
from numbers import Integral, Real
import os
from pathlib import Path
import tempfile
from typing import Mapping, Sequence

import numpy as np

from .simulation import (
    OBSERVATION_DESIGN_STRATUM_COUNT,
    OBSERVATION_DESIGN_STRATUM_FIELDS,
    OBSERVATION_DESIGN_STRATUM_UNIVERSE_SEMANTICS,
    OBSERVATION_DESIGN_STRATUM_UNIVERSE_SHA256,
    OBSERVATION_DESIGN_STRATUM_UNIVERSE_VERSION,
)


CALIBRATION_SCHEMA = "gisaxs.posterior_v8.compatibility_calibration/v6"
CALIBRATION_VERSION = (
    "posterior_v8_reserved_split_single_view_marginal_acquisition_compatibility_v6"
)
RESERVED_CALIBRATION_SPLIT_ID = "calibration"
COMPATIBILITY_STRATUM_VERSION = "posterior_v8_observable_acquisition_compatibility_stratum_v1"
COMPATIBILITY_STRATUM_FIELDS = ("point_count", "noise_id", "q_window_id")
STRATIFICATION_SEMANTICS = (
    "marginal_acquisition_conditional_candidate_independent_pre_fit_metadata_only"
)
SCORE_SEMANTICS = "standardized_weighted_log_residual_compatibility_score_pseudo_likelihood"
CALIBRATION_UNIT = "one_clean_recipe_one_deterministic_observation_per_acquisition_stratum"
GROUP_AGGREGATION = "none_duplicate_recipe_stratum_observation_fails_closed"
OBSERVATION_ESTIMAND = (
    "single_deterministic_observation_view_per_clean_recipe_per_acquisition_design_stratum"
)
DUPLICATE_OBSERVATION_POLICY = (
    "reject_repeated_clean_recipe_acquisition_stratum_without_registered_whole_acquisition_estimand"
)
PREREGISTERED_DESIGN_STRATUM_COUNT = OBSERVATION_DESIGN_STRATUM_COUNT
DESIGN_STRATUM_UNIVERSE_VERSION = OBSERVATION_DESIGN_STRATUM_UNIVERSE_VERSION
DESIGN_STRATUM_UNIVERSE_FIELDS = OBSERVATION_DESIGN_STRATUM_FIELDS
DESIGN_STRATUM_UNIVERSE_SEMANTICS = OBSERVATION_DESIGN_STRATUM_UNIVERSE_SEMANTICS
DESIGN_STRATUM_UNIVERSE_SHA256 = OBSERVATION_DESIGN_STRATUM_UNIVERSE_SHA256
DESIGN_POINT_COUNT_SEMANTICS = (
    "CompatibilityStratum.point_count_is_pre_mask_pre_crop_acquisition_design_n"
)
EFFECTIVE_VALID_POINT_COUNT_SEMANTICS = (
    "number_of_finite_selected_points_actually_used_in_the_standardized_score"
)
ACQUISITION_POLICY_ID_VERSION = "posterior_v8_complete_acquisition_policy_id_v1"
ACQUISITION_POLICY_REQUIRED_COMPONENTS = ("grid", "mask", "crop", "view", "sigma")
MEASUREMENT_SIGMA_POLICY = (
    "measured_or_simulated_measurement_sigma_required_no_encoder_proxy_or_missing_sigma"
)
FALLBACK_SEMANTICS = "pooled_empirical_description_only_no_coverage_claim_for_unseen_stratum"
MAX_ARTIFACT_BYTES = 8 * 1024 * 1024

_ACQUISITION_POLICY_PREFIX = f"{ACQUISITION_POLICY_ID_VERSION}:"
_ACQUISITION_POLICY_FIELDS = frozenset({"version", *ACQUISITION_POLICY_REQUIRED_COMPONENTS})
_ACQUISITION_GRID_FIELDS = frozenset(
    {"kind", "design_point_count", "q_min", "q_max", "q_window_id"}
)
_ACQUISITION_MASK_FIELDS = frozenset({"mask_id", "point_keep_probability"})
_ACQUISITION_CROP_FIELDS = frozenset({"crop_id", "q_min", "q_max"})
_ACQUISITION_VIEW_FIELDS = frozenset(
    {"observation_view_version", "view_index", "observation_seed_derivation"}
)
_ACQUISITION_SIGMA_FIELDS = frozenset(
    {
        "noise_id",
        "poisson_count_scale",
        "relative_sigma",
        "sigma_floor_fraction",
        "sigma_log_source",
    }
)


class CompatibilityCalibrationError(ValueError):
    """Raised when calibration input or an artifact fails closed validation."""


class UnseenCompatibilityStratumError(CompatibilityCalibrationError):
    """The requested observation stratum has no conditional calibration."""


class UnseenAcquisitionPolicyError(CompatibilityCalibrationError):
    """The requested full acquisition policy was absent from calibration."""


def _finite(value: float, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise CompatibilityCalibrationError(f"{name} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise CompatibilityCalibrationError(f"{name} must be finite") from exc
    if not np.isfinite(result):
        raise CompatibilityCalibrationError(f"{name} must be finite")
    return result


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise CompatibilityCalibrationError(f"{name} must be a positive integer")
    result = int(value)
    if result < 1:
        raise CompatibilityCalibrationError(f"{name} must be a positive integer")
    return result


def _nonempty_text(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CompatibilityCalibrationError(f"{name} must be a non-empty string")
    result = value.strip()
    if len(result) > 256:
        raise CompatibilityCalibrationError(f"{name} is too long")
    return result


def _nonnegative_integer(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise CompatibilityCalibrationError(f"{name} must be a non-negative integer")
    result = int(value)
    if result < 0:
        raise CompatibilityCalibrationError(f"{name} must be a non-negative integer")
    return result


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def _reject_constant(value: str):
    raise CompatibilityCalibrationError(f"non-finite JSON number is forbidden: {value}")


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise CompatibilityCalibrationError(f"duplicate JSON field is forbidden: {key}")
        result[key] = value
    return result


def _policy_mapping(value: object, expected: frozenset[str], name: str) -> dict[str, object]:
    if not isinstance(value, dict) or frozenset(value) != expected:
        raise CompatibilityCalibrationError(
            f"acquisition_policy_id {name} must contain exactly {sorted(expected)}"
        )
    return value


def _validate_acquisition_policy_payload(value: object) -> dict[str, object]:
    payload = _policy_mapping(value, _ACQUISITION_POLICY_FIELDS, "payload")
    if payload["version"] != ACQUISITION_POLICY_ID_VERSION:
        raise CompatibilityCalibrationError("acquisition_policy_id has an incompatible version")
    grid = _policy_mapping(payload["grid"], _ACQUISITION_GRID_FIELDS, "grid")
    mask = _policy_mapping(payload["mask"], _ACQUISITION_MASK_FIELDS, "mask")
    crop = _policy_mapping(payload["crop"], _ACQUISITION_CROP_FIELDS, "crop")
    view = _policy_mapping(payload["view"], _ACQUISITION_VIEW_FIELDS, "view")
    sigma = _policy_mapping(payload["sigma"], _ACQUISITION_SIGMA_FIELDS, "sigma")

    _nonempty_text(grid["kind"], "acquisition_policy_id.grid.kind")
    design_n = _positive_integer(
        grid["design_point_count"], "acquisition_policy_id.grid.design_point_count"
    )
    q_min = _finite(grid["q_min"], "acquisition_policy_id.grid.q_min")
    q_max = _finite(grid["q_max"], "acquisition_policy_id.grid.q_max")
    if q_min <= 0.0 or q_max <= q_min:
        raise CompatibilityCalibrationError("acquisition_policy_id grid q bounds are invalid")
    _nonempty_text(grid["q_window_id"], "acquisition_policy_id.grid.q_window_id")

    _nonempty_text(mask["mask_id"], "acquisition_policy_id.mask.mask_id")
    keep = _finite(
        mask["point_keep_probability"],
        "acquisition_policy_id.mask.point_keep_probability",
    )
    if not 0.0 < keep <= 1.0:
        raise CompatibilityCalibrationError(
            "acquisition_policy_id mask probability must be in (0, 1]"
        )

    _nonempty_text(crop["crop_id"], "acquisition_policy_id.crop.crop_id")
    crop_min = _finite(crop["q_min"], "acquisition_policy_id.crop.q_min")
    crop_max = _finite(crop["q_max"], "acquisition_policy_id.crop.q_max")
    if crop_min < q_min or crop_max > q_max or crop_max <= crop_min:
        raise CompatibilityCalibrationError(
            "acquisition_policy_id crop bounds must lie inside the grid"
        )

    _nonempty_text(
        view["observation_view_version"],
        "acquisition_policy_id.view.observation_view_version",
    )
    _nonnegative_integer(view["view_index"], "acquisition_policy_id.view.view_index")
    _nonempty_text(
        view["observation_seed_derivation"],
        "acquisition_policy_id.view.observation_seed_derivation",
    )

    _nonempty_text(sigma["noise_id"], "acquisition_policy_id.sigma.noise_id")
    count_scale = sigma["poisson_count_scale"]
    if (
        count_scale is not None
        and _finite(count_scale, "acquisition_policy_id.sigma.poisson_count_scale") <= 0.0
    ):
        raise CompatibilityCalibrationError(
            "acquisition_policy_id poisson_count_scale must be positive or null"
        )
    relative_sigma = _finite(sigma["relative_sigma"], "acquisition_policy_id.sigma.relative_sigma")
    sigma_floor = _finite(
        sigma["sigma_floor_fraction"],
        "acquisition_policy_id.sigma.sigma_floor_fraction",
    )
    if relative_sigma < 0.0 or sigma_floor <= 0.0:
        raise CompatibilityCalibrationError("acquisition_policy_id sigma policy is invalid")
    _nonempty_text(sigma["sigma_log_source"], "acquisition_policy_id.sigma.sigma_log_source")
    if design_n < 1:  # pragma: no cover - protected by _positive_integer
        raise CompatibilityCalibrationError("acquisition design point count is invalid")
    return payload


def make_acquisition_policy_id(
    *,
    grid: Mapping[str, object],
    mask: Mapping[str, object],
    crop: Mapping[str, object],
    view: Mapping[str, object],
    sigma: Mapping[str, object],
) -> str:
    """Encode complete acquisition provenance as a canonical, self-describing ID."""

    payload = _validate_acquisition_policy_payload(
        {
            "version": ACQUISITION_POLICY_ID_VERSION,
            "grid": dict(grid),
            "mask": dict(mask),
            "crop": dict(crop),
            "view": dict(view),
            "sigma": dict(sigma),
        }
    )
    return _ACQUISITION_POLICY_PREFIX + _canonical_json_bytes(payload).decode("utf-8")


def acquisition_policy_payload(acquisition_policy_id: str) -> dict[str, object]:
    """Decode and strictly validate one canonical acquisition-policy ID."""

    if not isinstance(acquisition_policy_id, str) or not acquisition_policy_id.startswith(
        _ACQUISITION_POLICY_PREFIX
    ):
        raise CompatibilityCalibrationError(
            "acquisition_policy_id must use the current complete provenance version"
        )
    if len(acquisition_policy_id) > 4096:
        raise CompatibilityCalibrationError("acquisition_policy_id is too long")
    encoded = acquisition_policy_id[len(_ACQUISITION_POLICY_PREFIX) :]
    try:
        raw = json.loads(
            encoded,
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (TypeError, json.JSONDecodeError) as exc:
        raise CompatibilityCalibrationError("acquisition_policy_id is not valid JSON") from exc
    payload = _validate_acquisition_policy_payload(raw)
    if encoded.encode("utf-8") != _canonical_json_bytes(payload):
        raise CompatibilityCalibrationError("acquisition_policy_id JSON is not canonical")
    return payload


@dataclass(frozen=True, order=True, kw_only=True)
class CompatibilityStratum:
    """Pre-fit design stratum, deliberately independent of model-side K.

    ``point_count`` is the acquisition design N before deterministic crop and
    mask policies.  It is intentionally not the realized number of valid
    points entering the score.
    """

    point_count: int
    noise_id: str
    q_window_id: str

    def __post_init__(self) -> None:
        point_count = _positive_integer(self.point_count, "point_count")
        object.__setattr__(self, "point_count", point_count)
        object.__setattr__(self, "noise_id", _nonempty_text(self.noise_id, "noise_id"))
        object.__setattr__(self, "q_window_id", _nonempty_text(self.q_window_id, "q_window_id"))

    @property
    def design_point_count(self) -> int:
        """Explicit name for the versioned ``point_count`` wire field."""

        return self.point_count


@dataclass(frozen=True, kw_only=True)
class CompatibilityCalibrationSample:
    sample_id: str
    independent_group_id: str
    stratum: CompatibilityStratum
    score: float
    effective_valid_point_count: int
    acquisition_policy_id: str
    measurement_sigma_available: bool

    def __post_init__(self) -> None:
        if not isinstance(self.stratum, CompatibilityStratum):
            raise TypeError("stratum must be a CompatibilityStratum")
        score = _finite(self.score, "score")
        if score < 0.0:
            raise CompatibilityCalibrationError("score must be non-negative")
        effective_n = _positive_integer(
            self.effective_valid_point_count, "effective_valid_point_count"
        )
        if effective_n > self.stratum.design_point_count:
            raise CompatibilityCalibrationError(
                "effective_valid_point_count cannot exceed acquisition design point_count"
            )
        if type(self.measurement_sigma_available) is not bool:
            raise CompatibilityCalibrationError("measurement_sigma_available must be boolean")
        if not self.measurement_sigma_available:
            raise CompatibilityCalibrationError(
                "measurement sigma is required for compatibility calibration"
            )
        policy = acquisition_policy_payload(self.acquisition_policy_id)
        grid = policy["grid"]
        sigma = policy["sigma"]
        if (
            grid["design_point_count"] != self.stratum.design_point_count
            or grid["q_window_id"] != self.stratum.q_window_id
            or sigma["noise_id"] != self.stratum.noise_id
        ):
            raise CompatibilityCalibrationError(
                "acquisition_policy_id does not match the sample design stratum"
            )
        object.__setattr__(self, "sample_id", _nonempty_text(self.sample_id, "sample_id"))
        object.__setattr__(
            self,
            "independent_group_id",
            _nonempty_text(self.independent_group_id, "independent_group_id"),
        )
        object.__setattr__(self, "score", score)
        object.__setattr__(self, "effective_valid_point_count", effective_n)


@dataclass(frozen=True, order=True, kw_only=True)
class EffectiveValidPointCountFrequency:
    effective_valid_point_count: int
    observation_count: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "effective_valid_point_count",
            _positive_integer(self.effective_valid_point_count, "effective_valid_point_count"),
        )
        object.__setattr__(
            self,
            "observation_count",
            _positive_integer(self.observation_count, "observation_count"),
        )


@dataclass(frozen=True, kw_only=True)
class CalibratedThreshold:
    sample_count: int
    target_coverage: float
    order_statistic: int
    guaranteed_coverage: float
    threshold: float

    def __post_init__(self) -> None:
        count = _positive_integer(self.sample_count, "sample_count")
        order = _positive_integer(self.order_statistic, "order_statistic")
        coverage = _finite(self.target_coverage, "target_coverage")
        guaranteed = _finite(self.guaranteed_coverage, "guaranteed_coverage")
        threshold = _finite(self.threshold, "threshold")
        if not 0.0 < coverage < 1.0:
            raise CompatibilityCalibrationError("target_coverage must be in (0, 1)")
        if order > count:
            raise CompatibilityCalibrationError(
                "order_statistic exceeds the finite calibration sample count"
            )
        expected_order = int(ceil((count + 1) * coverage))
        if order != expected_order:
            raise CompatibilityCalibrationError("order_statistic is inconsistent with coverage")
        expected_guaranteed = order / float(count + 1)
        if guaranteed != expected_guaranteed:
            raise CompatibilityCalibrationError(
                "guaranteed_coverage is inconsistent with the order statistic"
            )
        if guaranteed < coverage or threshold < 0.0:
            raise CompatibilityCalibrationError("calibrated threshold fields are inconsistent")
        object.__setattr__(self, "sample_count", count)
        object.__setattr__(self, "target_coverage", coverage)
        object.__setattr__(self, "order_statistic", order)
        object.__setattr__(self, "guaranteed_coverage", guaranteed)
        object.__setattr__(self, "threshold", threshold)


@dataclass(frozen=True, kw_only=True)
class StratumCalibration:
    stratum: CompatibilityStratum
    observation_count: int
    effective_valid_point_count_histogram: tuple[EffectiveValidPointCountFrequency, ...]
    acquisition_policy_ids: tuple[str, ...]
    calibration: CalibratedThreshold

    def __post_init__(self) -> None:
        if not isinstance(self.stratum, CompatibilityStratum):
            raise TypeError("stratum must be a CompatibilityStratum")
        observations = _positive_integer(self.observation_count, "observation_count")
        if not isinstance(self.calibration, CalibratedThreshold):
            raise TypeError("calibration must be a CalibratedThreshold")
        if observations != self.calibration.sample_count:
            raise CompatibilityCalibrationError(
                "single-view observation_count must equal the calibration sample count"
            )
        histogram = tuple(self.effective_valid_point_count_histogram)
        if not histogram or not all(
            isinstance(value, EffectiveValidPointCountFrequency) for value in histogram
        ):
            raise CompatibilityCalibrationError(
                "effective_valid_point_count_histogram must contain frequencies"
            )
        effective_counts = [value.effective_valid_point_count for value in histogram]
        if effective_counts != sorted(set(effective_counts)):
            raise CompatibilityCalibrationError(
                "effective valid point-count histogram must be unique and sorted"
            )
        if any(value > self.stratum.design_point_count for value in effective_counts):
            raise CompatibilityCalibrationError(
                "effective valid point count exceeds the design point count"
            )
        if sum(value.observation_count for value in histogram) != observations:
            raise CompatibilityCalibrationError(
                "effective valid point-count histogram does not cover all observations"
            )
        policy_ids = tuple(self.acquisition_policy_ids)
        if not policy_ids or list(policy_ids) != sorted(set(policy_ids)):
            raise CompatibilityCalibrationError(
                "acquisition_policy_ids must be non-empty, unique, and sorted"
            )
        for policy_id in policy_ids:
            policy = acquisition_policy_payload(policy_id)
            if (
                policy["grid"]["design_point_count"] != self.stratum.design_point_count
                or policy["grid"]["q_window_id"] != self.stratum.q_window_id
                or policy["sigma"]["noise_id"] != self.stratum.noise_id
            ):
                raise CompatibilityCalibrationError(
                    "stratum acquisition policy does not match its design fields"
                )
        object.__setattr__(self, "observation_count", observations)
        object.__setattr__(self, "effective_valid_point_count_histogram", histogram)
        object.__setattr__(self, "acquisition_policy_ids", policy_ids)


@dataclass(frozen=True, kw_only=True)
class DescriptiveThreshold:
    """Pooled empirical reference with deliberately no coverage guarantee."""

    sample_count: int
    target_quantile: float
    order_statistic: int
    threshold: float
    coverage_bound: None = None
    scope: str = FALLBACK_SEMANTICS

    def __post_init__(self) -> None:
        count = _positive_integer(self.sample_count, "sample_count")
        order = _positive_integer(self.order_statistic, "order_statistic")
        quantile = _finite(self.target_quantile, "target_quantile")
        threshold = _finite(self.threshold, "threshold")
        if not 0.0 < quantile < 1.0 or order > count or threshold < 0.0:
            raise CompatibilityCalibrationError("descriptive threshold fields are inconsistent")
        expected = min(count, int(ceil((count + 1) * quantile)))
        if order != expected:
            raise CompatibilityCalibrationError(
                "descriptive order_statistic is inconsistent with target_quantile"
            )
        if self.coverage_bound is not None or self.scope != FALLBACK_SEMANTICS:
            raise CompatibilityCalibrationError(
                "descriptive fallback must not carry a coverage guarantee"
            )
        object.__setattr__(self, "sample_count", count)
        object.__setattr__(self, "target_quantile", quantile)
        object.__setattr__(self, "order_statistic", order)
        object.__setattr__(self, "threshold", threshold)


@dataclass(frozen=True, kw_only=True)
class CalibrationInputSummary:
    observation_count: int
    independent_group_count: int
    recipe_stratum_count: int
    stratum_count: int
    acquisition_policy_count: int
    effective_valid_point_count_min: int
    effective_valid_point_count_max: int
    score_min: float
    score_median: float
    score_max: float

    def __post_init__(self) -> None:
        observations = _positive_integer(self.observation_count, "observation_count")
        groups = _positive_integer(self.independent_group_count, "independent_group_count")
        recipe_strata = _positive_integer(self.recipe_stratum_count, "recipe_stratum_count")
        strata = _positive_integer(self.stratum_count, "stratum_count")
        policies = _positive_integer(self.acquisition_policy_count, "acquisition_policy_count")
        effective_min = _positive_integer(
            self.effective_valid_point_count_min, "effective_valid_point_count_min"
        )
        effective_max = _positive_integer(
            self.effective_valid_point_count_max, "effective_valid_point_count_max"
        )
        low = _finite(self.score_min, "score_min")
        median = _finite(self.score_median, "score_median")
        high = _finite(self.score_max, "score_max")
        if low < 0.0 or not low <= median <= high:
            raise CompatibilityCalibrationError("input score summary is inconsistent")
        if (
            groups > observations
            or recipe_strata != observations
            or policies > observations
            or effective_min > effective_max
        ):
            raise CompatibilityCalibrationError("input group counts are inconsistent")
        object.__setattr__(self, "observation_count", observations)
        object.__setattr__(self, "independent_group_count", groups)
        object.__setattr__(self, "recipe_stratum_count", recipe_strata)
        object.__setattr__(self, "stratum_count", strata)
        object.__setattr__(self, "acquisition_policy_count", policies)
        object.__setattr__(self, "effective_valid_point_count_min", effective_min)
        object.__setattr__(self, "effective_valid_point_count_max", effective_max)
        object.__setattr__(self, "score_min", low)
        object.__setattr__(self, "score_median", median)
        object.__setattr__(self, "score_max", high)


@dataclass(frozen=True, kw_only=True)
class CompatibilityCalibrationArtifact:
    target_coverage: float
    minimum_samples_per_stratum: int
    input_summary: CalibrationInputSummary
    input_sha256: str
    dataset_manifest_sha256: str
    calibration_split_id: str
    calibration_split_sha256: str
    global_fallback: DescriptiveThreshold
    strata: tuple[StratumCalibration, ...]
    schema: str = CALIBRATION_SCHEMA
    calibration_version: str = CALIBRATION_VERSION
    score_semantics: str = SCORE_SEMANTICS
    calibration_unit: str = CALIBRATION_UNIT
    group_aggregation: str = GROUP_AGGREGATION
    observation_estimand: str = OBSERVATION_ESTIMAND
    duplicate_observation_policy: str = DUPLICATE_OBSERVATION_POLICY
    preregistered_design_stratum_count: int = PREREGISTERED_DESIGN_STRATUM_COUNT
    design_stratum_universe_version: str = DESIGN_STRATUM_UNIVERSE_VERSION
    design_stratum_universe_fields: tuple[str, ...] = DESIGN_STRATUM_UNIVERSE_FIELDS
    design_stratum_universe_semantics: str = DESIGN_STRATUM_UNIVERSE_SEMANTICS
    design_stratum_universe_sha256: str = DESIGN_STRATUM_UNIVERSE_SHA256
    design_point_count_semantics: str = DESIGN_POINT_COUNT_SEMANTICS
    effective_valid_point_count_semantics: str = EFFECTIVE_VALID_POINT_COUNT_SEMANTICS
    acquisition_policy_id_version: str = ACQUISITION_POLICY_ID_VERSION
    acquisition_policy_required_components: tuple[str, ...] = ACQUISITION_POLICY_REQUIRED_COMPONENTS
    measurement_sigma_policy: str = MEASUREMENT_SIGMA_POLICY
    fallback_semantics: str = FALLBACK_SEMANTICS
    compatibility_stratum_version: str = COMPATIBILITY_STRATUM_VERSION
    compatibility_stratum_fields: tuple[str, ...] = COMPATIBILITY_STRATUM_FIELDS
    stratification_semantics: str = STRATIFICATION_SEMANTICS

    def __post_init__(self) -> None:
        coverage = _finite(self.target_coverage, "target_coverage")
        minimum = _positive_integer(self.minimum_samples_per_stratum, "minimum_samples_per_stratum")
        if not 0.0 < coverage < 1.0:
            raise CompatibilityCalibrationError("target_coverage must be in (0, 1)")
        if self.schema != CALIBRATION_SCHEMA:
            raise CompatibilityCalibrationError("incompatible calibration schema")
        if self.calibration_version != CALIBRATION_VERSION:
            raise CompatibilityCalibrationError("incompatible calibration version")
        if self.score_semantics != SCORE_SEMANTICS:
            raise CompatibilityCalibrationError("incompatible compatibility-score semantics")
        if self.calibration_unit != CALIBRATION_UNIT:
            raise CompatibilityCalibrationError("incompatible calibration unit")
        if self.group_aggregation != GROUP_AGGREGATION:
            raise CompatibilityCalibrationError("incompatible group aggregation")
        if self.observation_estimand != OBSERVATION_ESTIMAND:
            raise CompatibilityCalibrationError("incompatible calibration observation estimand")
        if self.duplicate_observation_policy != DUPLICATE_OBSERVATION_POLICY:
            raise CompatibilityCalibrationError("incompatible duplicate-observation policy")
        registered_strata = _positive_integer(
            self.preregistered_design_stratum_count,
            "preregistered_design_stratum_count",
        )
        if registered_strata != PREREGISTERED_DESIGN_STRATUM_COUNT:
            raise CompatibilityCalibrationError("incompatible preregistered design-stratum count")
        if self.design_stratum_universe_version != DESIGN_STRATUM_UNIVERSE_VERSION:
            raise CompatibilityCalibrationError("incompatible design-stratum universe version")
        universe_fields = tuple(self.design_stratum_universe_fields)
        if universe_fields != DESIGN_STRATUM_UNIVERSE_FIELDS:
            raise CompatibilityCalibrationError("incompatible design-stratum universe fields")
        if self.design_stratum_universe_semantics != DESIGN_STRATUM_UNIVERSE_SEMANTICS:
            raise CompatibilityCalibrationError("incompatible design-stratum universe semantics")
        if self.design_stratum_universe_sha256 != DESIGN_STRATUM_UNIVERSE_SHA256:
            raise CompatibilityCalibrationError("incompatible design-stratum universe SHA-256")
        if self.design_point_count_semantics != DESIGN_POINT_COUNT_SEMANTICS:
            raise CompatibilityCalibrationError("incompatible design point-count semantics")
        if self.effective_valid_point_count_semantics != EFFECTIVE_VALID_POINT_COUNT_SEMANTICS:
            raise CompatibilityCalibrationError("incompatible effective point-count semantics")
        if self.acquisition_policy_id_version != ACQUISITION_POLICY_ID_VERSION:
            raise CompatibilityCalibrationError("incompatible acquisition-policy ID version")
        required_components = tuple(self.acquisition_policy_required_components)
        if required_components != ACQUISITION_POLICY_REQUIRED_COMPONENTS:
            raise CompatibilityCalibrationError(
                "incompatible acquisition-policy provenance components"
            )
        if self.measurement_sigma_policy != MEASUREMENT_SIGMA_POLICY:
            raise CompatibilityCalibrationError("incompatible measurement-sigma policy")
        if self.fallback_semantics != FALLBACK_SEMANTICS:
            raise CompatibilityCalibrationError("incompatible fallback semantics")
        if self.compatibility_stratum_version != COMPATIBILITY_STRATUM_VERSION:
            raise CompatibilityCalibrationError("incompatible compatibility-stratum version")
        stratum_fields = tuple(self.compatibility_stratum_fields)
        if stratum_fields != COMPATIBILITY_STRATUM_FIELDS:
            raise CompatibilityCalibrationError("incompatible compatibility-stratum fields")
        if self.stratification_semantics != STRATIFICATION_SEMANTICS:
            raise CompatibilityCalibrationError("incompatible stratification semantics")
        split_id = _nonempty_text(self.calibration_split_id, "calibration_split_id")
        if split_id != RESERVED_CALIBRATION_SPLIT_ID:
            raise CompatibilityCalibrationError(
                "compatibility calibration must come from the reserved calibration split"
            )
        if not isinstance(self.input_summary, CalibrationInputSummary):
            raise TypeError("input_summary must be a CalibrationInputSummary")
        if not isinstance(self.global_fallback, DescriptiveThreshold):
            raise TypeError("global_fallback must be a DescriptiveThreshold")
        calibrations = tuple(self.strata)
        if not calibrations or not all(
            isinstance(value, StratumCalibration) for value in calibrations
        ):
            raise CompatibilityCalibrationError("strata must contain calibrations")
        keys = [value.stratum for value in calibrations]
        if len(set(keys)) != len(keys) or keys != sorted(keys):
            raise CompatibilityCalibrationError("strata must be unique and canonically sorted")
        if any(value.calibration.sample_count < minimum for value in calibrations):
            raise CompatibilityCalibrationError("a stratum has insufficient calibration samples")
        if any(value.calibration.target_coverage != coverage for value in calibrations):
            raise CompatibilityCalibrationError("stratum target coverage is inconsistent")
        if self.global_fallback.target_quantile != coverage:
            raise CompatibilityCalibrationError("global fallback quantile is inconsistent")
        total_observations = sum(value.observation_count for value in calibrations)
        total_recipe_strata = sum(value.calibration.sample_count for value in calibrations)
        policy_ids = {
            policy_id for value in calibrations for policy_id in value.acquisition_policy_ids
        }
        effective_counts = [
            frequency.effective_valid_point_count
            for value in calibrations
            for frequency in value.effective_valid_point_count_histogram
        ]
        if (
            total_observations != self.input_summary.observation_count
            or total_recipe_strata != self.input_summary.recipe_stratum_count
            or self.global_fallback.sample_count != self.input_summary.observation_count
            or len(policy_ids) != self.input_summary.acquisition_policy_count
            or min(effective_counts) != self.input_summary.effective_valid_point_count_min
            or max(effective_counts) != self.input_summary.effective_valid_point_count_max
        ):
            raise CompatibilityCalibrationError("artifact sample counts are inconsistent")
        if len(calibrations) != self.input_summary.stratum_count:
            raise CompatibilityCalibrationError("artifact stratum count is inconsistent")
        if len(calibrations) > registered_strata:
            raise CompatibilityCalibrationError(
                "artifact expands beyond the preregistered design-stratum count"
            )
        digests = {
            "input_sha256": self.input_sha256,
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
            "calibration_split_sha256": self.calibration_split_sha256,
            "design_stratum_universe_sha256": self.design_stratum_universe_sha256,
        }
        for name, digest in digests.items():
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise CompatibilityCalibrationError(f"{name} must be lowercase SHA-256 hex")
        object.__setattr__(self, "target_coverage", coverage)
        object.__setattr__(self, "minimum_samples_per_stratum", minimum)
        object.__setattr__(self, "calibration_split_id", split_id)
        object.__setattr__(self, "strata", calibrations)
        object.__setattr__(self, "preregistered_design_stratum_count", registered_strata)
        object.__setattr__(self, "design_stratum_universe_fields", universe_fields)
        object.__setattr__(self, "compatibility_stratum_fields", stratum_fields)
        object.__setattr__(self, "acquisition_policy_required_components", required_components)
        for name, digest in digests.items():
            object.__setattr__(self, name, digest)

    def threshold_for(
        self,
        stratum: CompatibilityStratum,
        *,
        allow_descriptive_fallback: bool = False,
    ) -> tuple[CalibratedThreshold | DescriptiveThreshold, bool]:
        """Return an exact threshold; unseen strata fail closed by default."""
        if not isinstance(stratum, CompatibilityStratum):
            raise TypeError("stratum must be a CompatibilityStratum")
        if type(allow_descriptive_fallback) is not bool:
            raise TypeError("allow_descriptive_fallback must be boolean")
        for value in self.strata:
            if value.stratum == stratum:
                return value.calibration, False
        if not allow_descriptive_fallback:
            raise UnseenCompatibilityStratumError(
                "no conditional calibration exists for the requested observation stratum"
            )
        return self.global_fallback, True

    def threshold_for_acquisition_policy(
        self,
        stratum: CompatibilityStratum,
        acquisition_policy_id: str,
    ) -> CalibratedThreshold:
        """Return an exact threshold only for a calibrated full acquisition policy."""

        if not isinstance(stratum, CompatibilityStratum):
            raise TypeError("stratum must be a CompatibilityStratum")
        policy = acquisition_policy_payload(acquisition_policy_id)
        if (
            policy["grid"]["design_point_count"] != stratum.design_point_count
            or policy["grid"]["q_window_id"] != stratum.q_window_id
            or policy["sigma"]["noise_id"] != stratum.noise_id
        ):
            raise CompatibilityCalibrationError(
                "acquisition_policy_id does not match the requested design stratum"
            )
        for value in self.strata:
            if value.stratum != stratum:
                continue
            if acquisition_policy_id not in value.acquisition_policy_ids:
                raise UnseenAcquisitionPolicyError(
                    "no calibration support exists for the requested full acquisition policy"
                )
            return value.calibration
        raise UnseenCompatibilityStratumError(
            "no conditional calibration exists for the requested observation stratum"
        )

    def to_payload(self) -> dict[str, object]:
        return asdict(self)

    @property
    def sha256(self) -> str:
        """Logical digest of the strictly validated canonical artifact payload."""

        return hashlib.sha256(_canonical_json_bytes(self.to_payload())).hexdigest()


def _calibrate(values: Sequence[float], coverage: float) -> CalibratedThreshold:
    scores = np.sort(np.asarray(values, dtype=np.float64))
    count = int(scores.size)
    order = int(ceil((count + 1) * coverage))
    if order > count:
        raise CompatibilityCalibrationError(
            "sample count is insufficient for a finite split-conformal threshold "
            f"at target coverage {coverage:g}"
        )
    return CalibratedThreshold(
        sample_count=count,
        target_coverage=coverage,
        order_statistic=order,
        guaranteed_coverage=order / float(count + 1),
        threshold=float(scores[order - 1]),
    )


def _describe(values: Sequence[float], quantile: float) -> DescriptiveThreshold:
    scores = np.sort(np.asarray(values, dtype=np.float64))
    count = int(scores.size)
    order = min(count, int(ceil((count + 1) * quantile)))
    return DescriptiveThreshold(
        sample_count=count,
        target_quantile=quantile,
        order_statistic=order,
        threshold=float(scores[order - 1]),
    )


def fit_compatibility_calibration(
    samples: Sequence[CompatibilityCalibrationSample],
    *,
    dataset_manifest_sha256: str,
    calibration_split_sha256: str,
    calibration_split_id: str = RESERVED_CALIBRATION_SPLIT_ID,
    target_coverage: float = 0.95,
    minimum_samples_per_stratum: int = 20,
) -> CompatibilityCalibrationArtifact:
    """Fit held-out standardized-score thresholds for the single-view estimand.

    A clean recipe may occur in multiple acquisition design strata, but it may
    contribute exactly one deterministic observation to any one stratum.
    Repeated recipe/stratum observations are rejected; no max-view reduction is
    performed. The pooled fallback describes all admitted observations and has
    no conditional or finite-sample coverage claim.
    """
    values = tuple(samples)
    if not values or not all(isinstance(value, CompatibilityCalibrationSample) for value in values):
        raise CompatibilityCalibrationError(
            "samples must contain CompatibilityCalibrationSample values"
        )
    coverage = _finite(target_coverage, "target_coverage")
    if not 0.0 < coverage < 1.0:
        raise CompatibilityCalibrationError("target_coverage must be in (0, 1)")
    minimum = _positive_integer(minimum_samples_per_stratum, "minimum_samples_per_stratum")
    identifiers = [value.sample_id for value in values]
    if len(set(identifiers)) != len(identifiers):
        raise CompatibilityCalibrationError("sample_id values must be unique")
    if any(value.measurement_sigma_available is not True for value in values):
        raise CompatibilityCalibrationError(
            "measurement sigma is required for every calibration observation"
        )
    recipe_stratum_keys = [(value.independent_group_id, value.stratum) for value in values]
    if len(set(recipe_stratum_keys)) != len(recipe_stratum_keys):
        raise CompatibilityCalibrationError(
            "duplicate clean-recipe/acquisition-stratum observation is forbidden; "
            "the primary single-view estimand does not aggregate repeated views"
        )

    grouped: dict[CompatibilityStratum, list[CompatibilityCalibrationSample]] = {}
    for value in values:
        grouped.setdefault(value.stratum, []).append(value)
    sparse = [len(observations) for observations in grouped.values() if len(observations) < minimum]
    if sparse:
        raise CompatibilityCalibrationError(
            "stratum has insufficient calibration samples: "
            f"minimum={minimum}, observed_recipe_stratum_counts={sorted(sparse)}"
        )
    calibrations = tuple(
        StratumCalibration(
            stratum=key,
            observation_count=len(grouped[key]),
            effective_valid_point_count_histogram=tuple(
                EffectiveValidPointCountFrequency(
                    effective_valid_point_count=effective_count,
                    observation_count=count,
                )
                for effective_count, count in sorted(
                    Counter(value.effective_valid_point_count for value in grouped[key]).items()
                )
            ),
            acquisition_policy_ids=tuple(
                sorted({value.acquisition_policy_id for value in grouped[key]})
            ),
            calibration=_calibrate(tuple(value.score for value in grouped[key]), coverage),
        )
        for key in sorted(grouped)
    )
    scores = np.asarray([value.score for value in values], dtype=np.float64)
    effective_counts = [value.effective_valid_point_count for value in values]
    canonical_input = [
        {
            "sample_id": value.sample_id,
            "independent_group_id": value.independent_group_id,
            "score": value.score,
            "stratum": asdict(value.stratum),
            "effective_valid_point_count": value.effective_valid_point_count,
            "acquisition_policy_id": value.acquisition_policy_id,
            "measurement_sigma_available": value.measurement_sigma_available,
        }
        for value in sorted(values, key=lambda item: item.sample_id)
    ]
    digest = hashlib.sha256(_canonical_json_bytes(canonical_input)).hexdigest()
    return CompatibilityCalibrationArtifact(
        target_coverage=coverage,
        minimum_samples_per_stratum=minimum,
        input_summary=CalibrationInputSummary(
            observation_count=len(values),
            independent_group_count=len({value.independent_group_id for value in values}),
            recipe_stratum_count=len(recipe_stratum_keys),
            stratum_count=len(grouped),
            acquisition_policy_count=len({value.acquisition_policy_id for value in values}),
            effective_valid_point_count_min=min(effective_counts),
            effective_valid_point_count_max=max(effective_counts),
            score_min=float(np.min(scores)),
            score_median=float(np.median(scores)),
            score_max=float(np.max(scores)),
        ),
        input_sha256=digest,
        dataset_manifest_sha256=dataset_manifest_sha256,
        calibration_split_id=calibration_split_id,
        calibration_split_sha256=calibration_split_sha256,
        global_fallback=_describe(tuple(value.score for value in values), coverage),
        strata=calibrations,
    )


_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema",
        "calibration_version",
        "score_semantics",
        "calibration_unit",
        "group_aggregation",
        "observation_estimand",
        "duplicate_observation_policy",
        "preregistered_design_stratum_count",
        "design_stratum_universe_version",
        "design_stratum_universe_fields",
        "design_stratum_universe_semantics",
        "design_stratum_universe_sha256",
        "design_point_count_semantics",
        "effective_valid_point_count_semantics",
        "acquisition_policy_id_version",
        "acquisition_policy_required_components",
        "measurement_sigma_policy",
        "fallback_semantics",
        "compatibility_stratum_version",
        "compatibility_stratum_fields",
        "stratification_semantics",
        "target_coverage",
        "minimum_samples_per_stratum",
        "input_summary",
        "input_sha256",
        "dataset_manifest_sha256",
        "calibration_split_id",
        "calibration_split_sha256",
        "global_fallback",
        "strata",
    }
)
_SUMMARY_FIELDS = frozenset(
    {
        "observation_count",
        "independent_group_count",
        "recipe_stratum_count",
        "stratum_count",
        "acquisition_policy_count",
        "effective_valid_point_count_min",
        "effective_valid_point_count_max",
        "score_min",
        "score_median",
        "score_max",
    }
)
_THRESHOLD_FIELDS = frozenset(
    {"sample_count", "target_coverage", "order_statistic", "guaranteed_coverage", "threshold"}
)
_DESCRIPTIVE_THRESHOLD_FIELDS = frozenset(
    {
        "sample_count",
        "target_quantile",
        "order_statistic",
        "threshold",
        "coverage_bound",
        "scope",
    }
)
_STRATUM_FIELDS = frozenset(COMPATIBILITY_STRATUM_FIELDS)
_STRATUM_CALIBRATION_FIELDS = frozenset(
    {
        "stratum",
        "observation_count",
        "effective_valid_point_count_histogram",
        "acquisition_policy_ids",
        "calibration",
    }
)
_EFFECTIVE_COUNT_FREQUENCY_FIELDS = frozenset({"effective_valid_point_count", "observation_count"})


def _exact_fields(value: object, expected: frozenset[str], name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise CompatibilityCalibrationError(f"{name} must be a JSON object")
    actual = frozenset(value)
    missing, unknown = expected - actual, actual - expected
    if missing or unknown:
        raise CompatibilityCalibrationError(
            f"{name} fields are invalid: missing={sorted(missing)}, unknown={sorted(unknown)}"
        )
    return value


def _threshold_from_payload(value: object, name: str) -> CalibratedThreshold:
    payload = _exact_fields(value, _THRESHOLD_FIELDS, name)
    return CalibratedThreshold(**payload)


def _descriptive_threshold_from_payload(value: object, name: str) -> DescriptiveThreshold:
    payload = _exact_fields(value, _DESCRIPTIVE_THRESHOLD_FIELDS, name)
    return DescriptiveThreshold(**payload)


def _artifact_from_payload(value: object) -> CompatibilityCalibrationArtifact:
    payload = _exact_fields(value, _TOP_LEVEL_FIELDS, "artifact")
    summary = CalibrationInputSummary(
        **_exact_fields(payload["input_summary"], _SUMMARY_FIELDS, "input_summary")
    )
    strata_value = payload["strata"]
    if not isinstance(strata_value, list):
        raise CompatibilityCalibrationError("strata must be a JSON array")
    strata = []
    for index, item in enumerate(strata_value):
        entry = _exact_fields(item, _STRATUM_CALIBRATION_FIELDS, f"strata[{index}]")
        stratum = CompatibilityStratum(
            **_exact_fields(entry["stratum"], _STRATUM_FIELDS, f"strata[{index}].stratum")
        )
        calibration = _threshold_from_payload(entry["calibration"], f"strata[{index}].calibration")
        histogram_value = entry["effective_valid_point_count_histogram"]
        if not isinstance(histogram_value, list):
            raise CompatibilityCalibrationError(
                f"strata[{index}].effective_valid_point_count_histogram must be a JSON array"
            )
        policy_ids = entry["acquisition_policy_ids"]
        if not isinstance(policy_ids, list):
            raise CompatibilityCalibrationError(
                f"strata[{index}].acquisition_policy_ids must be a JSON array"
            )
        strata.append(
            StratumCalibration(
                stratum=stratum,
                observation_count=entry["observation_count"],
                effective_valid_point_count_histogram=tuple(
                    EffectiveValidPointCountFrequency(
                        **_exact_fields(
                            frequency,
                            _EFFECTIVE_COUNT_FREQUENCY_FIELDS,
                            (
                                f"strata[{index}].effective_valid_point_count_histogram"
                                f"[{frequency_index}]"
                            ),
                        )
                    )
                    for frequency_index, frequency in enumerate(histogram_value)
                ),
                acquisition_policy_ids=tuple(policy_ids),
                calibration=calibration,
            )
        )
    return CompatibilityCalibrationArtifact(
        schema=payload["schema"],
        calibration_version=payload["calibration_version"],
        score_semantics=payload["score_semantics"],
        calibration_unit=payload["calibration_unit"],
        group_aggregation=payload["group_aggregation"],
        observation_estimand=payload["observation_estimand"],
        duplicate_observation_policy=payload["duplicate_observation_policy"],
        preregistered_design_stratum_count=payload["preregistered_design_stratum_count"],
        design_stratum_universe_version=payload["design_stratum_universe_version"],
        design_stratum_universe_fields=payload["design_stratum_universe_fields"],
        design_stratum_universe_semantics=payload["design_stratum_universe_semantics"],
        design_stratum_universe_sha256=payload["design_stratum_universe_sha256"],
        design_point_count_semantics=payload["design_point_count_semantics"],
        effective_valid_point_count_semantics=payload["effective_valid_point_count_semantics"],
        acquisition_policy_id_version=payload["acquisition_policy_id_version"],
        acquisition_policy_required_components=payload["acquisition_policy_required_components"],
        measurement_sigma_policy=payload["measurement_sigma_policy"],
        fallback_semantics=payload["fallback_semantics"],
        compatibility_stratum_version=payload["compatibility_stratum_version"],
        compatibility_stratum_fields=payload["compatibility_stratum_fields"],
        stratification_semantics=payload["stratification_semantics"],
        target_coverage=payload["target_coverage"],
        minimum_samples_per_stratum=payload["minimum_samples_per_stratum"],
        input_summary=summary,
        input_sha256=payload["input_sha256"],
        dataset_manifest_sha256=payload["dataset_manifest_sha256"],
        calibration_split_id=payload["calibration_split_id"],
        calibration_split_sha256=payload["calibration_split_sha256"],
        global_fallback=_descriptive_threshold_from_payload(
            payload["global_fallback"], "global_fallback"
        ),
        strata=tuple(strata),
    )


def load_compatibility_calibration(
    path: Path | str,
) -> CompatibilityCalibrationArtifact:
    """Load and strictly validate one compatibility-calibration artifact."""
    artifact_path = Path(path)
    if artifact_path.is_symlink():
        raise CompatibilityCalibrationError("calibration artifact must not be a symbolic link")
    try:
        size = artifact_path.stat().st_size
    except OSError as exc:
        raise CompatibilityCalibrationError(
            "calibration artifact is missing or unreadable"
        ) from exc
    if size < 2 or size > MAX_ARTIFACT_BYTES:
        raise CompatibilityCalibrationError("calibration artifact size is invalid")
    try:
        payload = json.loads(
            artifact_path.read_text(encoding="utf-8"),
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
        return _artifact_from_payload(payload)
    except CompatibilityCalibrationError:
        raise
    except (OSError, TypeError, UnicodeError, json.JSONDecodeError) as exc:
        raise CompatibilityCalibrationError("invalid compatibility-calibration artifact") from exc


def write_compatibility_calibration_atomic(
    path: Path | str, artifact: CompatibilityCalibrationArtifact
) -> None:
    """Publish canonical JSON without replacing an existing path."""
    if not isinstance(artifact, CompatibilityCalibrationArtifact):
        raise TypeError("artifact must be a CompatibilityCalibrationArtifact")
    destination = Path(path)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite calibration artifact: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical_json_bytes(artifact.to_payload()) + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError:
            raise FileExistsError(
                f"refusing to overwrite calibration artifact: {destination}"
            ) from None
        try:
            directory_descriptor = os.open(destination.parent, os.O_RDONLY)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
        except OSError:
            pass
    finally:
        temporary.unlink(missing_ok=True)


__all__ = [
    "ACQUISITION_POLICY_ID_VERSION",
    "ACQUISITION_POLICY_REQUIRED_COMPONENTS",
    "CALIBRATION_UNIT",
    "CALIBRATION_SCHEMA",
    "CALIBRATION_VERSION",
    "COMPATIBILITY_STRATUM_FIELDS",
    "COMPATIBILITY_STRATUM_VERSION",
    "DESIGN_POINT_COUNT_SEMANTICS",
    "DESIGN_STRATUM_UNIVERSE_FIELDS",
    "DESIGN_STRATUM_UNIVERSE_SEMANTICS",
    "DESIGN_STRATUM_UNIVERSE_SHA256",
    "DESIGN_STRATUM_UNIVERSE_VERSION",
    "DUPLICATE_OBSERVATION_POLICY",
    "EFFECTIVE_VALID_POINT_COUNT_SEMANTICS",
    "FALLBACK_SEMANTICS",
    "GROUP_AGGREGATION",
    "MEASUREMENT_SIGMA_POLICY",
    "OBSERVATION_ESTIMAND",
    "PREREGISTERED_DESIGN_STRATUM_COUNT",
    "RESERVED_CALIBRATION_SPLIT_ID",
    "SCORE_SEMANTICS",
    "STRATIFICATION_SEMANTICS",
    "CalibratedThreshold",
    "CalibrationInputSummary",
    "CompatibilityCalibrationArtifact",
    "CompatibilityCalibrationError",
    "CompatibilityCalibrationSample",
    "CompatibilityStratum",
    "DescriptiveThreshold",
    "EffectiveValidPointCountFrequency",
    "StratumCalibration",
    "UnseenAcquisitionPolicyError",
    "UnseenCompatibilityStratumError",
    "acquisition_policy_payload",
    "fit_compatibility_calibration",
    "load_compatibility_calibration",
    "make_acquisition_policy_id",
    "write_compatibility_calibration_atomic",
]
