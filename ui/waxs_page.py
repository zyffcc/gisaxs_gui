"""Legacy imports for the feature-owned WAXS presentation and loading API."""

from src.gimap.features.waxs.infrastructure import (
    detect_nxs_frame_count,
    load_image_matrix,
    load_tiff_matrix,
)
from src.gimap.features.waxs.presentation.page import (
    BatchWorker,
    ImageLoadResult,
    ImageLoadWorker,
    InSituProcessingWidget,
    SCATTERING_FILTER,
    SUPPORTED_EXTENSIONS,
    ScatteringImageViewer,
    make_double_spin,
)

__all__ = [
    "BatchWorker",
    "ImageLoadResult",
    "ImageLoadWorker",
    "InSituProcessingWidget",
    "SCATTERING_FILTER",
    "SUPPORTED_EXTENSIONS",
    "ScatteringImageViewer",
    "detect_nxs_frame_count",
    "load_image_matrix",
    "load_tiff_matrix",
    "make_double_spin",
]
