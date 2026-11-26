"""Prediction adapters。"""

from .detector_images import DetectorPredictionImageRepository, ProfileFrameContexts
from .fabio_images import FabioPredictionImageRepository, LocalPredictionFileCatalog
from .module_preprocessing import ModuleEntryPreprocessor
from .model_paths import InMemoryModelPathStore, PreferencesModelPathStore
from .yaml_modules import YamlModuleRepository, module_to_legacy_dict
from .local_exports import LocalPredictionExportRepository
from .numpy_masks import NumpyPredictionMaskRepository

__all__ = [
    "DetectorPredictionImageRepository",
    "FabioPredictionImageRepository",
    "InMemoryModelPathStore",
    "LocalPredictionFileCatalog",
    "LocalPredictionExportRepository",
    "ModuleEntryPreprocessor",
    "NumpyPredictionMaskRepository",
    "PreferencesModelPathStore",
    "ProfileFrameContexts",
    "YamlModuleRepository",
    "module_to_legacy_dict",
]
