"""Stable classification dataset service composed from focused adapters."""

from .dataset_discovery import ClassificationDatasetDiscoveryMixin
from .dataset_quality import ClassificationDatasetQualityMixin
from .feature_construction import ClassificationFeatureConstructionMixin


class ClassificationDataService(
    ClassificationDatasetDiscoveryMixin,
    ClassificationDatasetQualityMixin,
    ClassificationFeatureConstructionMixin,
):
    """Read, validate, preprocess and vectorize labeled 1D/2D data."""

    ONE_D_EXTENSIONS = {".dat", ".txt", ".csv", ".xy", ".chi"}
    TWO_D_EXTENSIONS = {".edf", ".tif", ".tiff", ".cbf", ".png", ".jpg", ".jpeg", ".bmp"}
    ARRAY_EXTENSIONS = {".npy"}
    HDF5_EXTENSIONS = {".h5", ".hdf5"}
