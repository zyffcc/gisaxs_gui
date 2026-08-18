"""Trainset feature composition helpers。"""

from __future__ import annotations

from ...app import AppContext
from .application import (
    GenerateTrainsetPreview,
    LoadTrainsetProject,
    SaveTrainsetProject,
    PrepareTrainsetJobPackage,
    ManageTrainsetLocalProcess,
    FindTrainedTrainsetModel,
    RegisterTrainsetPredictionModule,
    ManageTrainsetRemoteJobs,
    LoadTrainsetMetrics,
    PrepareTrainsetDesign,
    ManageTrainsetConfiguration,
    SimulateTrainsetWhatIf,
    ValidateModelContract,
    TrainsetUiCatalog,
)
from .application.ports import SimulationPort
from .infrastructure.adapters import (
    TensorFlowModelContractAdapter,
    TrainsetPreviewAdapter,
    LocalTrainsetConfigRepository,
    PortableTrainsetJobPackageAdapter,
    QtTrainsetLocalProcessAdapter,
    LocalTrainsetModelRegistrationAdapter,
    LocalTrainsetMetricsRepository,
    SlurmTrainsetRemoteJobAdapter,
    TrainsetDesignAdapter,
    TrainsetConfigurationAdapter,
)
from .presentation.view_model import TrainsetViewModel


def create_model_contract_validator(context: AppContext) -> ValidateModelContract:
    if context.jobs is None:
        raise ValueError("Trainset model validation requires a JobRunner")
    return ValidateModelContract(TensorFlowModelContractAdapter(context.jobs))


def create_trainset_view_model(
    context: AppContext, simulation_port: SimulationPort
) -> TrainsetViewModel:
    preview = TrainsetPreviewAdapter(simulation_port)
    projects = LocalTrainsetConfigRepository()
    packages = PortableTrainsetJobPackageAdapter()
    local_processes = QtTrainsetLocalProcessAdapter()
    registration = LocalTrainsetModelRegistrationAdapter()
    remote_jobs = SlurmTrainsetRemoteJobAdapter()
    metrics = LocalTrainsetMetricsRepository()
    design = TrainsetDesignAdapter()
    configuration = TrainsetConfigurationAdapter()
    return TrainsetViewModel(
        context=context,
        generate_preview=GenerateTrainsetPreview(preview),
        simulate_what_if=SimulateTrainsetWhatIf(preview),
        validate_model_contract=create_model_contract_validator(context),
        load_project=LoadTrainsetProject(projects),
        save_project=SaveTrainsetProject(projects),
        prepare_job_package=PrepareTrainsetJobPackage(packages),
        local_processes=ManageTrainsetLocalProcess(local_processes),
        find_trained_model=FindTrainedTrainsetModel(registration),
        register_prediction_module=RegisterTrainsetPredictionModule(registration),
        remote_jobs=ManageTrainsetRemoteJobs(remote_jobs),
        load_metrics=LoadTrainsetMetrics(metrics),
        prepare_design=PrepareTrainsetDesign(design),
        configuration=ManageTrainsetConfiguration(configuration),
        catalog=TrainsetUiCatalog(),
    )
