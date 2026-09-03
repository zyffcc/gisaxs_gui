"""Versioned NumPy preprocessing shared by Posterior V8 training and inference.

The functions in this module deliberately accept raw paired arrays.  Callers
must not pre-sort, independently filter, or normalize any of the three input
columns before calling :func:`preprocess_curve`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from numbers import Integral
from typing import Any, Mapping, Sequence

import numpy as np


PREPROCESSING_VERSION = "posterior_v8_scale_invariant_p99_logq_v2"


@dataclass(frozen=True)
class PreprocessingContract:
    """Stable numerical contract embedded in every Posterior V8 artifact."""

    q_min: float = 9.0e-5
    q_max: float = 6.0
    max_points: int = 1000
    min_valid_points: int = 16
    intensity_reference_percentile: float = 99.0
    q_unit: str = "nm^-1"
    version: str = PREPROCESSING_VERSION

    def __post_init__(self) -> None:
        try:
            q_min = float(self.q_min)
            q_max = float(self.q_max)
            percentile = float(self.intensity_reference_percentile)
        except (TypeError, ValueError) as exc:
            raise ValueError("contract bounds and percentile must be numeric") from exc
        if not np.isfinite(q_min) or q_min <= 0.0:
            raise ValueError("q_min must be finite and strictly positive")
        if not np.isfinite(q_max) or q_max <= q_min:
            raise ValueError("q_max must be finite and greater than q_min")
        if isinstance(self.max_points, (bool, np.bool_)) or not isinstance(
            self.max_points, Integral
        ) or self.max_points < 1:
            raise ValueError("max_points must be a positive integer")
        if isinstance(self.min_valid_points, (bool, np.bool_)) or not isinstance(
            self.min_valid_points, Integral
        ) or self.min_valid_points < 1:
            raise ValueError("min_valid_points must be a positive integer")
        if self.min_valid_points > self.max_points:
            raise ValueError("min_valid_points cannot exceed max_points")
        if not np.isfinite(percentile) or not 50.0 < percentile <= 100.0:
            raise ValueError("intensity_reference_percentile must be in (50, 100]")
        if self.q_unit != "nm^-1":
            raise ValueError("Posterior V8 q values must use the fixed unit 'nm^-1'")
        if not isinstance(self.version, str) or not self.version.strip():
            raise ValueError("version must be non-empty")
        object.__setattr__(self, "q_min", q_min)
        object.__setattr__(self, "q_max", q_max)
        object.__setattr__(self, "max_points", int(self.max_points))
        object.__setattr__(self, "min_valid_points", int(self.min_valid_points))
        object.__setattr__(self, "intensity_reference_percentile", percentile)


DEFAULT_CONTRACT = PreprocessingContract()


@dataclass(frozen=True)
class PreprocessedCurve:
    """Padded model tensors plus the paired physical samples that produced them."""

    x: np.ndarray
    point_mask: np.ndarray
    global_features: np.ndarray
    q: np.ndarray
    intensity: np.ndarray
    sigma: np.ndarray
    source_indices: np.ndarray
    stats: Mapping[str, Any]

    @property
    def valid_count(self) -> int:
        return int(np.count_nonzero(self.point_mask))

    def valid_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return aligned, sorted, unpadded q/intensity/sigma views."""

        return (
            self.q[self.point_mask],
            self.intensity[self.point_mask],
            self.sigma[self.point_mask],
        )

    def model_inputs(self, *, add_batch_axis: bool = False) -> dict[str, np.ndarray]:
        """Return only the tensors consumed by the Posterior V8 encoder."""

        values = {
            "x": self.x,
            "point_mask": self.point_mask,
            "global_features": self.global_features,
        }
        if add_batch_axis:
            return {name: value[np.newaxis, ...] for name, value in values.items()}
        return values


def _as_paired_vector(name: str, values: Sequence[float] | np.ndarray) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric values") from exc
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional; got shape {array.shape}")
    if array.size == 0:
        raise ValueError(f"{name} cannot be empty")
    return array


def _selection_mask(mask: Sequence[bool] | np.ndarray | None, size: int) -> np.ndarray:
    if mask is None:
        return np.ones(size, dtype=bool)
    array = np.asarray(mask)
    if array.ndim != 1 or array.size != size:
        raise ValueError(f"mask must be one-dimensional with length {size}; got {array.shape}")
    if array.dtype.kind != "b":
        if array.dtype.kind not in "iuf" or not np.all(np.isfinite(array)):
            raise ValueError("mask must contain only boolean or finite 0/1 values")
        if not np.all((array == 0) | (array == 1)):
            raise ValueError("mask must contain only boolean or finite 0/1 values")
    return array.astype(bool, copy=False)


def _validate_q_range(q_range: Sequence[float] | None) -> tuple[float, float] | None:
    if q_range is None:
        return None
    try:
        values = tuple(float(value) for value in q_range)
    except (TypeError, ValueError) as exc:
        raise ValueError("q_range must contain exactly two finite numeric bounds") from exc
    if len(values) != 2 or not np.all(np.isfinite(values)):
        raise ValueError("q_range must contain exactly two finite numeric bounds")
    low, high = values
    if low < 0.0 or high <= low:
        raise ValueError("q_range must satisfy 0 <= low < high")
    return low, high


def _linear_percentile(values: np.ndarray, percentile: float) -> float:
    """Version-independent NumPy linear percentile for a one-dimensional array."""

    ordered = np.sort(np.asarray(values, dtype=np.float64))
    position = (ordered.size - 1) * float(percentile) / 100.0
    lower = int(np.floor(position))
    upper = int(np.ceil(position))
    fraction = position - lower
    return float(ordered[lower] + fraction * (ordered[upper] - ordered[lower]))


def _readonly(array: np.ndarray) -> np.ndarray:
    array.setflags(write=False)
    return array


def preprocess_curve(
    q: Sequence[float] | np.ndarray,
    intensity: Sequence[float] | np.ndarray,
    sigma: Sequence[float] | np.ndarray,
    *,
    mask: Sequence[bool] | np.ndarray | None = None,
    q_range: Sequence[float] | None = None,
    contract: PreprocessingContract = DEFAULT_CONTRACT,
) -> PreprocessedCurve:
    """Validate, filter, pair-sort, normalize, and pad one measured curve.

    Filtering stages are deliberately sequential so every removed point has one
    traceable reason.  Contract q bounds are always enforced; ``q_range`` may
    select a narrower inclusive interval.  Curves longer than ``max_points``
    are deterministically sampled at evenly spaced source positions while
    preserving both endpoints and q/I/sigma pairing.
    """

    if not isinstance(contract, PreprocessingContract):
        raise TypeError("contract must be a PreprocessingContract")
    q_array = _as_paired_vector("q", q)
    intensity_array = _as_paired_vector("intensity", intensity)
    sigma_array = _as_paired_vector("sigma", sigma)
    if not (q_array.size == intensity_array.size == sigma_array.size):
        raise ValueError(
            "q, intensity, and sigma must have identical lengths; "
            f"got {q_array.size}, {intensity_array.size}, and {sigma_array.size}"
        )

    requested_range = _validate_q_range(q_range)
    selected = _selection_mask(mask, q_array.size).copy()
    input_count = int(q_array.size)
    masked_out_count = int(input_count - np.count_nonzero(selected))

    finite = np.isfinite(q_array) & np.isfinite(intensity_array) & np.isfinite(sigma_array)
    nonfinite_count = int(np.count_nonzero(selected & ~finite))
    selected &= finite

    positive = (q_array > 0.0) & (intensity_array > 0.0) & (sigma_array > 0.0)
    nonpositive_count = int(np.count_nonzero(selected & ~positive))
    selected &= positive

    if requested_range is None:
        requested_range_count = 0
    else:
        low, high = requested_range
        in_requested_range = (q_array >= low) & (q_array <= high)
        requested_range_count = int(np.count_nonzero(selected & ~in_requested_range))
        selected &= in_requested_range

    in_contract_range = (q_array >= contract.q_min) & (q_array <= contract.q_max)
    contract_range_count = int(np.count_nonzero(selected & ~in_contract_range))
    selected &= in_contract_range

    source_indices = np.flatnonzero(selected)
    if source_indices.size < contract.min_valid_points:
        raise ValueError(
            "too few valid paired points after filtering: "
            f"{source_indices.size} < {contract.min_valid_points}"
        )

    # Stable sorting makes tied-q behaviour deterministic and preserves the
    # original order among equal q values.
    order = np.argsort(q_array[source_indices], kind="stable")
    source_indices = source_indices[order]
    q_valid = q_array[source_indices]
    intensity_valid = intensity_array[source_indices]
    sigma_valid = sigma_array[source_indices]
    valid_before_downsampling = int(q_valid.size)

    intensity_reference = _linear_percentile(
        intensity_valid, contract.intensity_reference_percentile
    )
    if not np.isfinite(intensity_reference) or intensity_reference <= 0.0:
        raise ValueError("intensity percentile reference must be finite and positive")
    log_intensity_reference = float(np.log(intensity_reference))
    low_percentile = 100.0 - contract.intensity_reference_percentile
    intensity_low_reference = _linear_percentile(intensity_valid, low_percentile)
    if not np.isfinite(intensity_low_reference) or intensity_low_reference <= 0.0:
        raise ValueError("intensity low-percentile reference must be finite and positive")
    log_dynamic_range = float(np.log(intensity_reference / intensity_low_reference))
    median_log_relative_sigma = float(np.median(np.log(sigma_valid / intensity_valid)))

    if q_valid.size > contract.max_points:
        sample_positions = np.rint(
            np.linspace(0, q_valid.size - 1, contract.max_points, dtype=np.float64)
        ).astype(np.int64)
        q_valid = q_valid[sample_positions]
        intensity_valid = intensity_valid[sample_positions]
        sigma_valid = sigma_valid[sample_positions]
        source_indices = source_indices[sample_positions]

    valid_count = int(q_valid.size)
    contract_log_q_min = float(np.log(contract.q_min))
    log_q_denominator = float(np.log(contract.q_max) - contract_log_q_min)
    log_q = (np.log(q_valid) - contract_log_q_min) / log_q_denominator
    log_intensity = np.log(intensity_valid) - log_intensity_reference
    log_sigma = np.log(sigma_valid) - log_intensity_reference
    valid_x = np.stack((log_q, log_intensity, log_sigma), axis=-1).astype(np.float32)

    x = np.zeros((contract.max_points, 3), dtype=np.float32)
    point_mask = np.zeros(contract.max_points, dtype=bool)
    q_padded = np.zeros(contract.max_points, dtype=np.float64)
    intensity_padded = np.zeros(contract.max_points, dtype=np.float64)
    sigma_padded = np.zeros(contract.max_points, dtype=np.float64)
    source_indices_padded = np.full(contract.max_points, -1, dtype=np.int64)
    x[:valid_count] = valid_x
    point_mask[:valid_count] = True
    q_padded[:valid_count] = q_valid
    intensity_padded[:valid_count] = intensity_valid
    sigma_padded[:valid_count] = sigma_valid
    source_indices_padded[:valid_count] = source_indices

    global_features = np.asarray(
        [
            float(log_q[0]),
            float(log_q[-1]),
            valid_count / float(contract.max_points),
            log_dynamic_range,
            median_log_relative_sigma,
        ],
        dtype=np.float32,
    )
    stats = {
        "contract": asdict(contract),
        "input_count": input_count,
        "masked_out_count": masked_out_count,
        "nonfinite_count": nonfinite_count,
        "nonpositive_count": nonpositive_count,
        "outside_requested_q_range_count": requested_range_count,
        "outside_contract_q_range_count": contract_range_count,
        "valid_before_downsampling": valid_before_downsampling,
        "downsampled_count": valid_before_downsampling - valid_count,
        "valid_count": valid_count,
        "requested_q_range": requested_range,
        "q_unit": contract.q_unit,
        "q_min": float(q_valid[0]),
        "q_max": float(q_valid[-1]),
        "intensity_reference": intensity_reference,
        "intensity_low_reference": intensity_low_reference,
        "log_intensity_reference": log_intensity_reference,
        "log_dynamic_range": log_dynamic_range,
        "median_log_relative_sigma": median_log_relative_sigma,
        "source_indices": tuple(int(index) for index in source_indices),
    }

    return PreprocessedCurve(
        x=_readonly(x),
        point_mask=_readonly(point_mask),
        global_features=_readonly(global_features),
        q=_readonly(q_padded),
        intensity=_readonly(intensity_padded),
        sigma=_readonly(sigma_padded),
        source_indices=_readonly(source_indices_padded),
        stats=stats,
    )


__all__ = [
    "DEFAULT_CONTRACT",
    "PREPROCESSING_VERSION",
    "PreprocessedCurve",
    "PreprocessingContract",
    "preprocess_curve",
]
