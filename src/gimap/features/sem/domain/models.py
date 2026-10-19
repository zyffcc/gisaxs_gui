"""Scientific values for calibrated, reviewable SEM measurements."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np


def frozen_array(value, dtype=float):
    array = np.array(value, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class SemImage:
    image: np.ndarray
    source_path: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        image = frozen_array(self.image)
        if image.ndim != 2 or image.size == 0:
            raise ValueError("SEM image must be a non-empty grayscale image")
        object.__setattr__(self, "image", image)
        object.__setattr__(self, "metadata", deepcopy(dict(self.metadata)))


@dataclass(frozen=True)
class SemExtractionSettings:
    nm_per_pixel: float
    roi_xyxy: tuple[int, int, int, int] | None = None
    threshold: float | None = None
    bright_particles: bool = True
    min_area_px: int = 9
    max_area_px: int | None = None
    min_circularity: float = 0.0
    exclude_border: bool = True
    split_touching: bool = False
    peak_min_distance_px: int = 3
    method: str = "threshold"
    python_executable: str = ""
    model_directory: str = ""
    label_path: str = ""
    ai_scale: float = 1.0
    probability_threshold: float | None = None
    overlap_threshold: float | None = None
    normalize_low: float = 1.0
    normalize_high: float = 99.8


@dataclass(frozen=True)
class SemInstanceSegmentation:
    labels: np.ndarray
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        labels = np.asarray(self.labels)
        if labels.ndim != 2 or labels.size == 0:
            raise ValueError("SEM instance labels must be a non-empty 2D array")
        if not np.issubdtype(labels.dtype, np.integer):
            raise ValueError("SEM instance labels must use an integer dtype")
        if np.any(labels < 0) or np.any(labels > np.iinfo(np.int32).max):
            raise ValueError("SEM instance labels must be non-negative and within the int32 range")
        object.__setattr__(self, "labels", frozen_array(labels, np.int32))
        object.__setattr__(self, "metadata", deepcopy(dict(self.metadata)))


@dataclass(frozen=True)
class SemCsvSettings:
    value_column: str
    quantity: str = "radius"
    unit: str = "nm"
    nm_per_pixel: float | None = None
    skip_rows: int = 0
    delimiter: str = ","


@dataclass(frozen=True)
class SemParticle:
    label: int
    radius_nm: float | None
    diameter_nm: float | None
    area_px: int | None = None
    area_nm2: float | None = None
    centroid_x_px: float | None = None
    centroid_y_px: float | None = None
    circularity: float | None = None
    accepted: bool = True
    rejection_reason: str = ""


@dataclass(frozen=True)
class SemMeasurements:
    radius_nm: np.ndarray
    particles: tuple[SemParticle, ...]
    source: Mapping[str, Any]
    settings: Mapping[str, Any]
    summary: Mapping[str, Any]
    labels: np.ndarray | None = None
    image: np.ndarray | None = None

    def __post_init__(self):
        values = frozen_array(self.radius_nm)
        if values.ndim != 1 or not np.all(np.isfinite(values) & (values > 0)):
            raise ValueError("SEM radii must be finite positive values in nm")
        object.__setattr__(self, "radius_nm", values)
        object.__setattr__(self, "particles", tuple(self.particles))
        for key in ("source", "settings", "summary"):
            object.__setattr__(self, key, deepcopy(dict(getattr(self, key))))
        if self.labels is not None:
            object.__setattr__(self, "labels", frozen_array(self.labels, np.int32))
        if self.image is not None:
            object.__setattr__(self, "image", frozen_array(self.image))


@dataclass(frozen=True)
class SemPrediction:
    radius_nm: np.ndarray
    probability: np.ndarray
    source: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        axis = frozen_array(self.radius_nm)
        probability = frozen_array(self.probability)
        if (axis.ndim != 1 or axis.size < 2 or probability.shape != axis.shape
                or not np.all(np.isfinite(axis) & (axis >= 0))
                or not np.all(np.diff(axis) > 0)):
            raise ValueError("Prediction requires an increasing radius axis in nm")
        if not np.all(np.isfinite(probability) & (probability >= 0)) or probability.sum() <= 0:
            raise ValueError("Prediction probability must be finite, non-negative and non-zero")
        object.__setattr__(self, "radius_nm", axis)
        object.__setattr__(self, "probability", probability)
        object.__setattr__(self, "metadata", deepcopy(dict(self.metadata)))


@dataclass(frozen=True)
class SemComparisonSettings:
    bins: int = 30
    radius_scale: float = 1.0
    display_mode: str = "probability"
    radius_min_nm: float | None = None
    radius_max_nm: float | None = None


@dataclass(frozen=True)
class SemComparison:
    bin_edges_nm: np.ndarray
    sem_probability: np.ndarray
    prediction_probability: np.ndarray | None
    display_sem_edges_nm: np.ndarray
    display_sem_values: np.ndarray
    display_prediction_radius_nm: np.ndarray | None
    display_prediction_values: np.ndarray | None
    metadata: Mapping[str, Any]

    def __post_init__(self):
        for key in ("bin_edges_nm", "sem_probability", "prediction_probability",
                    "display_sem_edges_nm", "display_sem_values",
                    "display_prediction_radius_nm", "display_prediction_values"):
            value = getattr(self, key)
            if value is not None:
                object.__setattr__(self, key, frozen_array(value))
        object.__setattr__(self, "metadata", deepcopy(dict(self.metadata)))
