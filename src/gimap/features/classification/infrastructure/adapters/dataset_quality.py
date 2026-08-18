"""Focused data-service behavior for classification."""

from __future__ import annotations

import os
from collections import Counter, defaultdict
from typing import Callable, Optional

import numpy as np

from ...domain import (
    ClassificationSample,
    DataQualityIssue,
    DatasetSummary,
    FeatureMatrix,
)

ProgressCallback = Optional[Callable[[int, int, str], None]]
CancelCallback = Optional[Callable[[], bool]]


class ClassificationDatasetQualityMixin:
    """Own one cohesive part of classification dataset handling."""

    def validate_dataset(self, samples: list[ClassificationSample]) -> DatasetSummary:
        """Return aggregate QC and update per-sample QC statuses."""

        for sample in samples:
            self._update_sample_qc(sample)

        included = [sample for sample in samples if sample.included]
        valid = [
            sample
            for sample in included
            if sample.load_status == "loaded" and sample.qc_status in {"ready", "warning"}
        ]
        counts = Counter(sample.label for sample in included)
        valid_counts = Counter(sample.label for sample in valid)
        data_types = sorted({sample.data_type for sample in valid if sample.data_type})
        shapes = sorted({sample.raw_shape for sample in valid if sample.raw_shape})
        issues: list[DataQualityIssue] = []

        if len(valid_counts) < 2:
            issues.append(
                DataQualityIssue(
                    "error",
                    "At least two classes are required.",
                    "Add or include another labeled class.",
                )
            )

        if len(data_types) > 1:
            issues.append(
                DataQualityIssue(
                    "error",
                    "1D and 2D samples are mixed.",
                    "Use one data type per comparison session.",
                )
            )

        if len(shapes) > 1:
            issues.append(
                DataQualityIssue(
                    "warning",
                    "Sample shapes are not identical.",
                    "Use interpolation, center crop, or resize preprocessing before training.",
                )
            )

        if valid_counts:
            min_count = min(valid_counts.values())
            max_count = max(valid_counts.values())
            if min_count < 2:
                issues.append(
                    DataQualityIssue(
                        "error",
                        "A class has fewer than two valid samples.",
                        "Add more samples or exclude that class before cross-validation.",
                    )
                )
            elif max_count >= 3 * max(1, min_count):
                issues.append(
                    DataQualityIssue(
                        "warning",
                        "Class balance is uneven.",
                        "Prefer macro metrics and consider adding samples to the smaller class.",
                    )
                )

        duplicate_paths = [
            path
            for path, count in Counter(
                os.path.abspath(sample.file_path).lower() for sample in samples
            ).items()
            if count > 1
        ]
        if duplicate_paths:
            issues.append(
                DataQualityIssue(
                    "warning",
                    f"{len(duplicate_paths)} duplicate file path(s) detected.",
                    "Remove duplicate rows or keep only one copy included.",
                )
            )

        for sample in samples:
            if sample.load_status == "failed":
                issues.append(
                    DataQualityIssue(
                        "error",
                        f"{sample.file_name} failed to load.",
                        "Open the file or remove it from the dataset.",
                        sample.sample_id,
                    )
                )
            elif sample.qc_status == "error":
                issues.append(
                    DataQualityIssue(
                        "error",
                        f"{sample.file_name} has invalid values.",
                        "Fix NaN/Inf/empty data or exclude the sample.",
                        sample.sample_id,
                    )
                )

        return DatasetSummary(
            classes=len(counts),
            total_samples=len(samples),
            valid_samples=len(valid),
            invalid_samples=len(
                [
                    sample
                    for sample in samples
                    if sample.load_status == "failed" or sample.qc_status == "error"
                ]
            ),
            included_samples=len(included),
            loaded_samples=len([sample for sample in samples if sample.load_status == "loaded"]),
            class_counts=dict(counts),
            valid_class_counts=dict(valid_counts),
            data_types=data_types,
            shapes=shapes,
            issues=issues,
        )

    def summarize_by_label(
        self, samples: list[ClassificationSample]
    ) -> dict[str, dict[str, object]]:
        grouped: dict[str, list[ClassificationSample]] = defaultdict(list)
        for sample in samples:
            grouped[sample.label].append(sample)

        summary: dict[str, dict[str, object]] = {}
        for label, label_samples in grouped.items():
            loaded = [sample for sample in label_samples if sample.load_status == "loaded"]
            failed = [sample for sample in label_samples if sample.load_status == "failed"]
            shapes = sorted({sample.raw_shape for sample in loaded if sample.raw_shape})
            data_types = sorted({sample.data_type for sample in label_samples if sample.data_type})
            summary[label] = {
                "files": len(label_samples),
                "loaded": len(loaded),
                "failed": len(failed),
                "data_type": "/".join(data_types) if data_types else "-",
                "shape": ", ".join(str(shape) for shape in shapes[:3]) if shapes else "-",
                "status": self._label_status(label_samples),
            }
        return summary

    def estimate_feature_memory(self, summary: FeatureMatrix) -> str:
        bytes_used = int(summary.X.shape[0] * summary.X.shape[1] * 8)
        if bytes_used < 1024:
            return f"{bytes_used} B"
        if bytes_used < 1024**2:
            return f"{bytes_used / 1024:.1f} KB"
        return f"{bytes_used / 1024**2:.1f} MB"

    def _update_sample_qc(self, sample: ClassificationSample) -> None:
        if sample.load_status not in {"loaded", "failed"}:
            return
        if sample.load_status == "failed":
            sample.qc_status = "error"
            return
        data = sample.raw_data
        messages: list[str] = []
        status = "ready"
        if data is None or data.size == 0:
            status = "error"
            messages.append("Empty data.")
        else:
            if not np.all(np.isfinite(data)):
                status = "error"
                messages.append("Contains NaN or Inf values.")
            if sample.data_type == "1D" and data.ndim != 2:
                status = "warning"
                messages.append("1D data is not two-column; values will be flattened.")
            if sample.data_type == "2D" and data.ndim != 2:
                status = "warning"
                messages.append("2D data was reduced to a single image plane.")
        sample.qc_status = status
        sample.qc_messages = messages

    def _label_status(self, samples: list[ClassificationSample]) -> str:
        if not samples:
            return "Empty"
        if any(sample.load_status == "failed" or sample.qc_status == "error" for sample in samples):
            return "Error"
        if any(sample.load_status == "pending" for sample in samples):
            return "Scanned"
        if any(sample.qc_status == "warning" for sample in samples):
            return "Warning"
        return "Ready"
