"""Shared detector-file loading primitives used by multiple features."""

from .loading import (
    DETECTOR_SUFFIXES,
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
from .nexus_layout import (
    HDF5_SUFFIXES,
    BeingWritten,
    LeftUnfinished,
    NexusLayout,
    cached_layout,
    forget_layout,
    group_sources,
    inspect_nexus,
    is_hdf5,
    last_known_layout,
    open_nexus,
    scan_stem,
)
from .nexus_metadata import extract_nexus_metadata
from .nexus_reading import read_nexus_frame
from .writing import FRAME_FORMATS, frame_suffix, write_frame

__all__ = [
    "DETECTOR_SUFFIXES",
    "FRAME_FORMATS",
    "HDF5_SUFFIXES",
    "frame_suffix",
    "write_frame",
    "AmbiguousDatasetError",
    "BeingWritten",
    "DetectorImage",
    "LeftUnfinished",
    "NexusLayout",
    "_dataset_candidates",
    "cached_layout",
    "detect_nxs_frame_count",
    "dump_metadata",
    "extract_nexus_metadata",
    "forget_layout",
    "group_sources",
    "inspect_nexus",
    "is_hdf5",
    "last_known_layout",
    "load_detector_image",
    "nxs_invalid_pixel_mask",
    "nxs_series_paths",
    "open_nexus",
    "read_nexus_frame",
    "scan_stem",
    "select_nxs_dataset",
]
