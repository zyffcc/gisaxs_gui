"""Local NXS/CBF series repository backed by shared detector loading."""

from __future__ import annotations

import re
from pathlib import Path

import h5py
import numpy as np

from ...application import XrrDetectorFrame, XrrFrameRef, XrrSeriesSpec


def _natural_key(path: Path) -> tuple:
    return tuple(
        int(token) if token.isdigit() else token.casefold()
        for token in re.split(r"(\d+)", path.name)
    )


class LocalXrrSeriesRepository:
    """Discover lightweight references and load only the requested frame."""

    def discover(self, spec: XrrSeriesSpec) -> tuple[XrrFrameRef, ...]:
        source = Path(spec.source_path).expanduser().resolve()
        kind = self._kind(source, spec.source_kind)
        if kind == "nxs":
            return self._discover_nxs(source, spec)
        return self._discover_cbf(source, spec.pattern)

    def load_frame(self, frame: XrrFrameRef) -> XrrDetectorFrame:
        from src.gimap.shared.detector_io import load_detector_image

        loaded = load_detector_image(frame.path, frame_idx=frame.frame_index)
        return XrrDetectorFrame(
            data=np.asarray(loaded.data, dtype=np.float32),
            invalid_mask=(
                None if loaded.mask is None else np.asarray(loaded.mask, dtype=bool)
            ),
            metadata=dict(loaded.metadata),
        )

    @staticmethod
    def _kind(source: Path, requested: str) -> str:
        if requested in {"nxs", "cbf"}:
            return requested
        if source.suffix.casefold() == ".nxs":
            return "nxs"
        if source.is_dir() or source.suffix.casefold() == ".cbf":
            return "cbf"
        raise ValueError("Select an NXS module file or a folder containing CBF files.")

    def _discover_nxs(self, source: Path, spec: XrrSeriesSpec) -> tuple[XrrFrameRef, ...]:
        if not source.is_file():
            raise FileNotFoundError(f"NXS source was not found: {source}")
        from src.gimap.shared.detector_io import detect_nxs_frame_count, nxs_series_paths

        module_paths = nxs_series_paths(source)
        canonical = module_paths[0]
        counts = [int(detect_nxs_frame_count(path)) for path in module_paths]
        if len(set(counts)) != 1:
            summary = ", ".join(
                f"{path.name}: {count}" for path, count in zip(module_paths, counts)
            )
            raise ValueError(f"NXS detector modules have different frame counts ({summary}).")
        count = counts[0]
        angles = self._nxs_angles(canonical, spec.angle_dataset_path, count)
        return tuple(
            XrrFrameRef(
                path=canonical,
                frame_index=index,
                sequence_index=index,
                theta_deg=None if angles is None else float(angles[index]),
            )
            for index in range(count)
        )

    @staticmethod
    def _nxs_angles(source: Path, dataset_path: str, count: int) -> np.ndarray | None:
        if not dataset_path:
            return None
        with h5py.File(str(source), "r") as handle:
            if dataset_path not in handle:
                raise ValueError(f"NXS angle dataset was not found: {dataset_path}")
            dataset = handle[dataset_path]
            values = np.asarray(dataset[()], dtype=float).reshape(-1)
            raw_units = dataset.attrs.get("units", "deg")
            if isinstance(raw_units, bytes):
                raw_units = raw_units.decode("utf-8", errors="replace")
            units = str(raw_units).casefold()
        if values.size != count:
            raise ValueError(
                f"NXS angle dataset contains {values.size} values but the series has {count} frames."
            )
        if "mrad" in units:
            values = np.degrees(values / 1000.0)
        elif units in {"rad", "radian", "radians"}:
            values = np.degrees(values)
        return values

    @staticmethod
    def _discover_cbf(source: Path, pattern: str) -> tuple[XrrFrameRef, ...]:
        folder = source if source.is_dir() else source.parent
        if not folder.is_dir():
            raise FileNotFoundError(f"CBF series folder was not found: {folder}")
        selected_pattern = pattern.strip() or "*.cbf"
        paths = sorted(
            (
                path
                for path in folder.glob(selected_pattern)
                if path.is_file() and path.suffix.casefold() == ".cbf"
            ),
            key=_natural_key,
        )
        if not paths:
            raise ValueError(f"No CBF files match {selected_pattern!r} in {folder}.")
        return tuple(
            XrrFrameRef(path=path, frame_index=0, sequence_index=index)
            for index, path in enumerate(paths)
        )


__all__ = ["LocalXrrSeriesRepository"]
