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
from .writing import FRAME_FORMATS, frame_suffix, write_frame

__all__ = [
    "FRAME_FORMATS",
    "frame_suffix",
    "write_frame",
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
