"""Shared detector-file loading primitives used by multiple features."""

from .loading import (
    AmbiguousDatasetError,
    _dataset_candidates,
    detect_nxs_frame_count,
    dump_metadata,
    load_detector_image,
    nxs_invalid_pixel_mask,
    nxs_series_paths,
    select_nxs_dataset,
)
from .models import DetectorImage

__all__ = [
    "AmbiguousDatasetError",
    "DetectorImage",
    "_dataset_candidates",
    "detect_nxs_frame_count",
    "dump_metadata",
    "load_detector_image",
    "nxs_invalid_pixel_mask",
    "nxs_series_paths",
    "select_nxs_dataset",
]
