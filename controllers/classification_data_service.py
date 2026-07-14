"""Data scanning, loading, QC, preprocessing, and feature construction."""

from __future__ import annotations

import fnmatch
import hashlib
import os
from collections import Counter, defaultdict
from typing import Callable, Iterable, Optional

import h5py
import numpy as np

from controllers.classification_models import (
    ClassificationSample,
    DataQualityIssue,
    DatasetSource,
    DatasetSummary,
    FeatureMatrix,
    PreprocessingConfig,
)
from utils.path_utils import normalize_path


ProgressCallback = Optional[Callable[[int, int, str], None]]
CancelCallback = Optional[Callable[[], bool]]


class ClassificationDataService:
    """Read and validate labeled 1D/2D classification data."""

    ONE_D_EXTENSIONS = {".dat", ".txt", ".csv", ".xy", ".chi"}
    TWO_D_EXTENSIONS = {".edf", ".tif", ".tiff", ".cbf", ".png", ".jpg", ".jpeg", ".bmp"}
    ARRAY_EXTENSIONS = {".npy"}
    HDF5_EXTENSIONS = {".h5", ".hdf5"}

    def supported_extensions(self) -> list[str]:
        return sorted(
            self.ONE_D_EXTENSIONS
            | self.TWO_D_EXTENSIONS
            | self.ARRAY_EXTENSIONS
            | self.HDF5_EXTENSIONS
        )

    def detect_data_type_for_path(self, path: str) -> Optional[str]:
        ext = os.path.splitext(path)[1].lower()
        if ext in self.ONE_D_EXTENSIONS:
            return "1D"
        if ext in self.TWO_D_EXTENSIONS:
            return "2D"
        if ext in self.ARRAY_EXTENSIONS or ext in self.HDF5_EXTENSIONS:
            return "auto"
        return None

    def scan_source(self, source: DatasetSource) -> list[ClassificationSample]:
        """Scan one labeled source and return de-duplicated sample records."""

        seen: set[str] = set()
        files: list[str] = []
        pattern = source.file_pattern or "*"

        for raw_path in source.paths:
            path = normalize_path(raw_path)
            if not path:
                continue
            if os.path.isdir(path):
                walker = os.walk(path) if source.recursive else [(path, [], os.listdir(path))]
                for root, _, names in walker:
                    for name in names:
                        if not self._name_matches(name, pattern):
                            continue
                        full_path = os.path.join(root, name)
                        if self.detect_data_type_for_path(full_path) is None:
                            continue
                        normalized = os.path.abspath(full_path)
                        if normalized not in seen:
                            seen.add(normalized)
                            files.append(normalized)
            elif os.path.isfile(path):
                name = os.path.basename(path)
                if self.detect_data_type_for_path(path) is None:
                    continue
                if self._name_matches(name, pattern) or pattern in ("", "*"):
                    normalized = os.path.abspath(path)
                    if normalized not in seen:
                        seen.add(normalized)
                        files.append(normalized)

        files.sort(key=lambda item: item.lower())
        return [self._sample_from_file(file_path, source.label) for file_path in files]

    def scan_sources(self, sources: Iterable[DatasetSource]) -> list[ClassificationSample]:
        """Scan all sources while avoiding duplicate files across classes."""

        samples: list[ClassificationSample] = []
        seen_paths: set[str] = set()
        for source in sources:
            for sample in self.scan_source(source):
                key = os.path.abspath(sample.file_path).lower()
                if key in seen_paths:
                    sample.qc_status = "warning"
                    sample.qc_messages.append("Duplicate file path; only the first occurrence is used.")
                    continue
                seen_paths.add(key)
                samples.append(sample)
        return samples

    def samples_from_paths(self, paths: Iterable[str], label: str = "Unknown") -> list[ClassificationSample]:
        samples: list[ClassificationSample] = []
        for raw_path in paths:
            path = normalize_path(raw_path)
            if os.path.isdir(path):
                source = DatasetSource(label=label, source_type="folder", paths=[path], file_pattern="*")
                samples.extend(self.scan_source(source))
            elif os.path.isfile(path) and self.detect_data_type_for_path(path) is not None:
                samples.append(self._sample_from_file(os.path.abspath(path), label))
        return samples

    def load_samples(
        self,
        samples: list[ClassificationSample],
        progress: ProgressCallback = None,
        is_cancelled: CancelCallback = None,
    ) -> list[ClassificationSample]:
        """Load every sample that has not already been loaded."""

        total = len(samples)
        for index, sample in enumerate(samples, start=1):
            if is_cancelled and is_cancelled():
                break
            if sample.raw_data is None:
                self.load_sample(sample)
            if progress:
                progress(index, total, sample.file_name)
        return samples

    def load_sample(self, sample: ClassificationSample) -> ClassificationSample:
        """Read one sample and update its load/QC status in place."""

        sample.load_status = "loading"
        sample.qc_messages.clear()
        try:
            data = self.read_data(sample.file_path)
            if data is None:
                raise ValueError("No numeric data found.")
            data = np.asarray(data, dtype=np.float64)
            data = np.squeeze(data)
            if data.size == 0:
                raise ValueError("The file contains an empty array.")
            if data.ndim == 1:
                sample.data_type = "1D"
            elif data.ndim == 2:
                if sample.data_type == "auto":
                    sample.data_type = "2D"
            elif data.ndim > 2:
                data = self._reduce_image_channels(data)
                sample.data_type = "2D"
            sample.raw_data = data
            sample.raw_shape = tuple(int(v) for v in data.shape)
            sample.processed_data = None
            sample.load_status = "loaded"
            self._update_sample_qc(sample)
        except Exception as exc:
            sample.raw_data = None
            sample.processed_data = None
            sample.raw_shape = None
            sample.load_status = "failed"
            sample.qc_status = "error"
            sample.qc_messages = [str(exc)]
        return sample

    def read_data(self, path: str) -> Optional[np.ndarray]:
        ext = os.path.splitext(path)[1].lower()
        if ext in self.ONE_D_EXTENSIONS:
            return self._read_1d_text(path)
        if ext in {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}:
            return self._read_image(path)
        if ext in {".cbf", ".edf"}:
            return self._read_fabio(path)
        if ext in self.HDF5_EXTENSIONS:
            return self._read_hdf5(path)
        if ext == ".npy":
            return np.load(path, allow_pickle=False)
        return None

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
            issues.append(DataQualityIssue("error", "At least two classes are required.", "Add or include another labeled class."))

        if len(data_types) > 1:
            issues.append(DataQualityIssue("error", "1D and 2D samples are mixed.", "Use one data type per comparison session."))

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
            for path, count in Counter(os.path.abspath(sample.file_path).lower() for sample in samples).items()
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
            invalid_samples=len([sample for sample in samples if sample.load_status == "failed" or sample.qc_status == "error"]),
            included_samples=len(included),
            loaded_samples=len([sample for sample in samples if sample.load_status == "loaded"]),
            class_counts=dict(counts),
            valid_class_counts=dict(valid_counts),
            data_types=data_types,
            shapes=shapes,
            issues=issues,
        )

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
            warnings.append("Feature vectors had different lengths; they were trimmed to the shortest vector.")

        X = np.vstack([np.asarray(vector[:min_len], dtype=np.float64) for vector in vectors])
        y = np.asarray([sample.label for sample in selected], dtype=object) if require_labels else None
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

    def summarize_by_label(self, samples: list[ClassificationSample]) -> dict[str, dict[str, object]]:
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

    def _sample_from_file(self, path: str, label: str) -> ClassificationSample:
        data_type = self.detect_data_type_for_path(path) or "unknown"
        digest = hashlib.sha1(f"{label}|{os.path.abspath(path).lower()}".encode("utf-8")).hexdigest()[:16]
        return ClassificationSample(
            sample_id=digest,
            file_path=path,
            file_name=os.path.basename(path),
            label=label,
            data_type=data_type,
            load_status="pending",
            qc_status="pending",
        )

    def _name_matches(self, name: str, pattern: str) -> bool:
        if not pattern or pattern == "*":
            return True
        if "*" in pattern or "?" in pattern:
            return fnmatch.fnmatch(name, pattern)
        return pattern.lower() in name.lower()

    def _read_1d_text(self, path: str) -> Optional[np.ndarray]:
        rows: list[tuple[float, float]] = []
        with open(path, "r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped or stripped.startswith(("#", "%", ";")):
                    continue
                parts = stripped.replace(",", " ").split()
                if len(parts) < 2:
                    continue
                try:
                    rows.append((float(parts[0]), float(parts[1])))
                except ValueError:
                    continue
        if not rows:
            return None
        return np.asarray(rows, dtype=np.float64)

    def _read_image(self, path: str) -> np.ndarray:
        try:
            import imageio.v2 as imageio
        except Exception:
            import imageio
        image = imageio.imread(path)
        return self._reduce_image_channels(np.asarray(image, dtype=np.float64))

    def _read_fabio(self, path: str) -> Optional[np.ndarray]:
        try:
            import fabio
        except Exception as exc:
            raise RuntimeError("fabio is required for EDF/CBF files.") from exc
        image = fabio.open(path)
        data = getattr(image, "data", None)
        if data is None:
            return None
        return np.asarray(data, dtype=np.float64)

    def _read_hdf5(self, path: str) -> Optional[np.ndarray]:
        with h5py.File(path, "r") as handle:
            dataset = self._first_numeric_hdf5_dataset(handle)
            if dataset is None:
                return None
            return np.asarray(dataset)

    def _first_numeric_hdf5_dataset(self, node) -> Optional[np.ndarray]:
        if isinstance(node, h5py.Dataset):
            if np.issubdtype(node.dtype, np.number):
                return np.asarray(node)
            return None
        for key in node.keys():
            result = self._first_numeric_hdf5_dataset(node[key])
            if result is not None:
                return result
        return None

    def _reduce_image_channels(self, data: np.ndarray) -> np.ndarray:
        if data.ndim == 3:
            if data.shape[-1] in (3, 4):
                data = data[..., :3].mean(axis=-1)
            else:
                data = data[0]
        return np.asarray(data, dtype=np.float64)

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
                    warnings.append("Could not build a common 1D grid; raw vector lengths were trimmed.")
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
