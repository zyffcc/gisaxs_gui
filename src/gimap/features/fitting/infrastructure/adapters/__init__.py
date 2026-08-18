"""Fitting infrastructure adapters。"""

from .local_files import (
    LocalCurveRepository,
    LocalFitResultRepository,
    LocalScatteringFileRepository,
)
from .legacy_model import LegacyMixedModelAdapter
from .ai_pipeline import AiPipelinePredictor
from .local_candidates import JsonCandidateRepository
from .remote_cache import LocalRemoteFileCacheAdapter
from .local_insitu_records import LocalInSituRecordRepository
from .local_parameter_files import LocalFittingParameterFileRepository
from .local_ai_artifacts import LocalAiFittingArtifactRepository
from .local_logs import LocalFittingLogRepository
from .importlib_dependencies import ImportlibFittingDependencyAvailabilityAdapter
from .legacy_q_space import LegacyQSpaceAdapter
from .legacy_model_parameters import LegacyFittingModelParametersAdapter
from .legacy_ai_catalog import LegacyAiFittingCatalogAdapter

__all__ = [
    "LocalCurveRepository",
    "LocalFitResultRepository",
    "LocalScatteringFileRepository",
    "LegacyMixedModelAdapter",
    "AiPipelinePredictor",
    "JsonCandidateRepository",
    "LocalRemoteFileCacheAdapter",
    "LocalInSituRecordRepository",
    "LocalFittingParameterFileRepository",
    "LocalAiFittingArtifactRepository",
    "LocalFittingLogRepository",
    "ImportlibFittingDependencyAvailabilityAdapter",
    "LegacyQSpaceAdapter",
    "LegacyFittingModelParametersAdapter",
    "LegacyAiFittingCatalogAdapter",
]
