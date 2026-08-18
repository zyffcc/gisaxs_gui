"""Focused data-service behavior for classification."""

from __future__ import annotations

import fnmatch
import hashlib
import os
from typing import Callable, Iterable, Optional

import h5py
import numpy as np

from ...domain import (
    ClassificationSample,
    DatasetSource,
)
from src.gimap.shared.file_paths import normalize_path

ProgressCallback = Optional[Callable[[int, int, str], None]]
CancelCallback = Optional[Callable[[], bool]]


class ClassificationDatasetDiscoveryMixin:
    """Own one cohesive part of classification dataset handling."""

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
                    sample.qc_messages.append(
                        "Duplicate file path; only the first occurrence is used."
                    )
                    continue
                seen_paths.add(key)
                samples.append(sample)
        return samples

    def samples_from_paths(
        self, paths: Iterable[str], label: str = "Unknown"
    ) -> list[ClassificationSample]:
        samples: list[ClassificationSample] = []
        for raw_path in paths:
            path = normalize_path(raw_path)
            if os.path.isdir(path):
                source = DatasetSource(
                    label=label, source_type="folder", paths=[path], file_pattern="*"
                )
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

    def _sample_from_file(self, path: str, label: str) -> ClassificationSample:
        data_type = self.detect_data_type_for_path(path) or "unknown"
        digest = hashlib.sha1(
            f"{label}|{os.path.abspath(path).lower()}".encode("utf-8")
        ).hexdigest()[:16]
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
