"""WAXS infrastructure adapters。"""

from .detector_images import CalibrationWaxsImageRepository
from .job_runner_batch import JobRunnerWaxsBatchAdapter
from .legacy_loading import detect_nxs_frame_count, load_image_matrix, load_tiff_matrix
from .local_files import LocalWaxsExportAdapter, LocalWaxsFileCatalog
from .local_paths import LocalWaxsPathAdapter

__all__ = [
    "CalibrationWaxsImageRepository",
    "JobRunnerWaxsBatchAdapter",
    "LocalWaxsPathAdapter",
    "LocalWaxsExportAdapter",
    "LocalWaxsFileCatalog",
    "detect_nxs_frame_count",
    "load_image_matrix",
    "load_tiff_matrix",
]
