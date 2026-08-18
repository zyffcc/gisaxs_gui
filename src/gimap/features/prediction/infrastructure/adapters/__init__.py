"""Prediction adapters。"""

from .fabio_images import FabioPredictionImageRepository, LocalPredictionFileCatalog
from .module_preprocessing import ModuleEntryPreprocessor
from .yaml_modules import YamlModuleRepository, module_to_legacy_dict
from .local_exports import LocalPredictionExportRepository
from .numpy_masks import NumpyPredictionMaskRepository

__all__ = [
    "FabioPredictionImageRepository",
    "LocalPredictionFileCatalog",
    "LocalPredictionExportRepository",
    "ModuleEntryPreprocessor",
    "NumpyPredictionMaskRepository",
    "YamlModuleRepository",
    "module_to_legacy_dict",
]
