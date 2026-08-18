"""Stable Trainset generation API composed from focused adapters."""

from .dataset_generation import DatasetGenerator, PreviewResult
from .detector_images import crop_roi, load_scattering_image
from .detector_masks import (
    build_fixed_mask,
    build_random_mask,
    build_reference_threshold_mask,
    build_roi_shape_mask,
    build_threshold_mask,
    merge_threshold_mask,
)
from .physical_background import generate_physical_background
from .preprocessing_pipeline import apply_preprocessing

__all__ = [
    "DatasetGenerator",
    "PreviewResult",
    "apply_preprocessing",
    "build_fixed_mask",
    "build_random_mask",
    "build_reference_threshold_mask",
    "build_roi_shape_mask",
    "build_threshold_mask",
    "crop_roi",
    "generate_physical_background",
    "load_scattering_image",
    "merge_threshold_mask",
]
