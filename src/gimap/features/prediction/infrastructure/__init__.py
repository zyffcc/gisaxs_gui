"""Prediction infrastructure public API。"""

from .adapters import (
    FabioPredictionImageRepository,
    LocalPredictionFileCatalog,
    LocalPredictionExportRepository,
    ModuleEntryPreprocessor,
    NumpyPredictionMaskRepository,
    YamlModuleRepository,
    module_to_legacy_dict,
)

__all__ = [
    "FabioPredictionImageRepository",
    "LocalPredictionFileCatalog",
    "LocalPredictionExportRepository",
    "ModuleEntryPreprocessor",
    "NumpyPredictionMaskRepository",
    "YamlModuleRepository",
    "module_to_legacy_dict",
]
