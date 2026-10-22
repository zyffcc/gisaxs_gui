"""Trainset adapter exports。

``QtTrainsetLocalProcessAdapter`` (a QObject) is imported from ``qt_local_process`` itself: the
model-contract job worker imports this package, and PyQt5 must stay out of worker processes.
"""

from .keras_modeling import (
    build_keras_model,
    build_optimizer,
    normalized_layers,
    resolve_keras_api,
    static_contract,
)
from .legacy_generation import LegacyDatasetGenerationAdapter
from .project_config import LocalTrainsetConfigRepository
from .preview import TrainsetPreviewAdapter
from .job_packages import PortableTrainsetJobPackageAdapter
from .model_registration import LocalTrainsetModelRegistrationAdapter
from .remote_jobs import (
    LocalTrainsetMetricsRepository,
    SlurmTrainsetRemoteJobAdapter,
)
from .design import TrainsetDesignAdapter
from .configuration_policy import TrainsetConfigurationAdapter
from .model_contract import TensorFlowModelContractAdapter

__all__ = [
    "build_keras_model",
    "build_optimizer",
    "normalized_layers",
    "resolve_keras_api",
    "static_contract",
    "LegacyDatasetGenerationAdapter",
    "LocalTrainsetConfigRepository",
    "TrainsetPreviewAdapter",
    "PortableTrainsetJobPackageAdapter",
    "LocalTrainsetModelRegistrationAdapter",
    "LocalTrainsetMetricsRepository",
    "SlurmTrainsetRemoteJobAdapter",
    "TrainsetDesignAdapter",
    "TrainsetConfigurationAdapter",
    "TensorFlowModelContractAdapter",
]
