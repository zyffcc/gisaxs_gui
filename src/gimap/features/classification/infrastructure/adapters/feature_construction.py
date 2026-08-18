"""Focused data-service behavior for classification."""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from ...domain import (
    ClassificationSample,
    FeatureMatrix,
    PreprocessingConfig,
)

ProgressCallback = Optional[Callable[[int, int, str], None]]
CancelCallback = Optional[Callable[[], bool]]


class ClassificationFeatureConstructionMixin:
    """Own one cohesive part of classification dataset handling."""

    def build_feature_matrix(
        self,
        samples: list[ClassificationSample],
        config: PreprocessingConfig,
        require_labels: bool = True,
    ) -> FeatureMatrix:
        """Convert included loaded samples into a rectangular feature matrix."""

        selected = [
            sample
            for sample in samples
            if sample.included
            and sample.load_status == "loaded"
            and sample.raw_data is not None
            and (not require_labels or bool(sample.label))
        ]
        if not selected:
            raise ValueError("No included loaded samples are available.")

        data_types = {sample.data_type for sample in selected}
        if len(data_types) > 1:
            raise ValueError("Cannot train on mixed 1D and 2D data in one run.")
        data_type = next(iter(data_types))

        warnings: list[str] = []
        vectors: list[np.ndarray] = []
        if data_type == "1D":
            vectors, warnings = self._build_1d_vectors(selected, config)
        else:
            vectors, warnings = self._build_2d_vectors(selected, config)

        if not vectors:
            raise ValueError("Feature construction produced no samples.")

        min_len = min(len(vector) for vector in vectors)
        if any(len(vector) != min_len for vector in vectors):
            warnings.append(
                "Feature vectors had different lengths; they were trimmed to the shortest vector."
            )

        X = np.vstack([np.asarray(vector[:min_len], dtype=np.float64) for vector in vectors])
        y = (
            np.asarray([sample.label for sample in selected], dtype=object)
            if require_labels
            else None
        )
        feature_names = [f"f{i}" for i in range(X.shape[1])]
        input_shape = (int(X.shape[0]), int(X.shape[1]))
        return FeatureMatrix(
            X=X,
            y=y,
            samples=selected,
            feature_names=feature_names,
            data_type=data_type,
            input_shape=input_shape,
            warnings=warnings,
        )

    def _build_1d_vectors(
        self,
        samples: list[ClassificationSample],
        config: PreprocessingConfig,
    ) -> tuple[list[np.ndarray], list[str]]:
        arrays: list[np.ndarray] = []
        warnings: list[str] = []
        for sample in samples:
            data = np.asarray(sample.raw_data, dtype=np.float64)
            if data.ndim == 2 and data.shape[1] >= 2:
                arr = data[:, :2]
            else:
                y = data.ravel()
                arr = np.column_stack([np.arange(len(y), dtype=np.float64), y])
            if config.crop_min is not None or config.crop_max is not None:
                low = config.crop_min if config.crop_min is not None else float(np.min(arr[:, 0]))
                high = config.crop_max if config.crop_max is not None else float(np.max(arr[:, 0]))
                mask = (arr[:, 0] >= low) & (arr[:, 0] <= high)
                arr = arr[mask]
            arrays.append(arr)
        if any(len(arr) == 0 for arr in arrays):
            raise ValueError("1D preprocessing left at least one sample empty.")

        if config.one_d_method == "Interpolate to common grid" and arrays:
            if config.one_d_grid:
                grid = np.asarray(config.one_d_grid, dtype=np.float64)
                vectors = [np.interp(grid, arr[:, 0], arr[:, 1]) for arr in arrays]
            else:
                low = max(float(np.min(arr[:, 0])) for arr in arrays if len(arr))
                high = min(float(np.max(arr[:, 0])) for arr in arrays if len(arr))
                n_points = min(len(arr) for arr in arrays if len(arr))
                if n_points < 2 or high <= low:
                    warnings.append(
                        "Could not build a common 1D grid; raw vector lengths were trimmed."
                    )
                    vectors = [arr[:, 1] for arr in arrays]
                else:
                    grid = np.linspace(low, high, n_points)
                    config.one_d_grid = [float(value) for value in grid]
                    vectors = [np.interp(grid, arr[:, 0], arr[:, 1]) for arr in arrays]
        else:
            vectors = [arr[:, 1] for arr in arrays]

        processed = [self._preprocess_vector(vector, config) for vector in vectors]
        for sample, vector in zip(samples, processed):
            sample.processed_data = np.asarray(vector)
        return processed, warnings

    def _build_2d_vectors(
        self,
        samples: list[ClassificationSample],
        config: PreprocessingConfig,
    ) -> tuple[list[np.ndarray], list[str]]:
        images = [np.asarray(sample.raw_data, dtype=np.float64) for sample in samples]
        warnings: list[str] = []
        target_shape = config.resize_shape
        if target_shape is None:
            min_h = min(image.shape[0] for image in images)
            min_w = min(image.shape[1] for image in images)
            target_shape = (int(min_h), int(min_w))
            config.resize_shape = target_shape
        vectors: list[np.ndarray] = []
        for sample, image in zip(samples, images):
            processed = image
            if config.two_d_method in {"Center crop", "None"}:
                processed = self._center_crop(processed, target_shape)
            elif config.two_d_method == "Resize":
                processed = self._resize_image(processed, target_shape)
            elif config.two_d_method == "Mask invalid pixels":
                processed = np.nan_to_num(processed, nan=0.0, posinf=0.0, neginf=0.0)
                processed = self._center_crop(processed, target_shape)
            else:
                processed = self._center_crop(processed, target_shape)
            vector = self._preprocess_vector(processed.ravel(), config)
            sample.processed_data = np.asarray(processed)
            vectors.append(vector)
        if len({tuple(image.shape[:2]) for image in images}) > 1:
            warnings.append(f"2D images were aligned to {target_shape[0]}x{target_shape[1]}.")
        return vectors, warnings

    def _preprocess_vector(self, vector: np.ndarray, config: PreprocessingConfig) -> np.ndarray:
        values = np.asarray(vector, dtype=np.float64)
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        if config.log_transform:
            values = np.log1p(np.maximum(values, 0.0))
        if config.smoothing_window and config.smoothing_window > 1:
            window = int(config.smoothing_window)
            kernel = np.ones(window, dtype=np.float64) / window
            values = np.convolve(values, kernel, mode="same")
        if config.normalize == "max":
            denom = float(np.max(np.abs(values))) if values.size else 0.0
            if denom > 0:
                values = values / denom
        elif config.normalize == "area":
            denom = float(np.sum(np.abs(values))) if values.size else 0.0
            if denom > 0:
                values = values / denom
        return values.astype(np.float64)

    def _center_crop(self, image: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
        h, w = image.shape[:2]
        target_h = max(1, min(int(target_shape[0]), h))
        target_w = max(1, min(int(target_shape[1]), w))
        y0 = max(0, (h - target_h) // 2)
        x0 = max(0, (w - target_w) // 2)
        return image[y0 : y0 + target_h, x0 : x0 + target_w]

    def _resize_image(self, image: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
        try:
            import cv2

            return cv2.resize(
                image,
                (int(target_shape[1]), int(target_shape[0])),
                interpolation=cv2.INTER_AREA,
            )
        except Exception:
            cropped = self._center_crop(image, target_shape)
            return cropped
