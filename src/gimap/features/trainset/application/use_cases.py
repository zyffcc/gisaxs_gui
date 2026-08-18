"""Trainset framework-neutral use cases。"""

from __future__ import annotations

from pathlib import Path

from .models import (
    GenerateTrainsetRequest,
    ModelContractRequest,
    TrainsetPreviewRequest,
    TrainsetWhatIfRequest,
    PrepareTrainsetJobRequest,
    TrainsetLocalProcessRequest,
    RegisterTrainsetModelRequest,
)
from .ports import (
    DatasetGenerationPort,
    ModelContractPort,
    TrainsetConfigRepository,
    TrainsetPreviewPort,
    TrainsetJobPackagePort,
    TrainsetLocalProcessPort,
    TrainsetModelRegistrationPort,
    TrainsetMetricsRepository,
    TrainsetRemoteJobPort,
    TrainsetDesignPort,
    TrainsetConfigurationPort,
)
from ..domain import roi_to_spherical_ranges


class GenerateTrainset:
    def __init__(self, generator: DatasetGenerationPort):
        self._generator = generator

    def execute(self, request: GenerateTrainsetRequest, *, on_progress=None, pause=None):
        if request.sample_count <= 0:
            raise ValueError("Trainset sample_count must be positive")
        if request.mode not in {"preview", "demo", "dry", "full"}:
            raise ValueError(f"Unsupported trainset generation mode: {request.mode}")
        return self._generator.generate(
            request, on_progress=on_progress, pause=pause
        )


class LoadTrainsetProject:
    def __init__(self, repository: TrainsetConfigRepository):
        self._repository = repository

    def execute(self, path: Path):
        return self._repository.load(Path(path))


class SaveTrainsetProject:
    def __init__(self, repository: TrainsetConfigRepository):
        self._repository = repository

    def execute(self, config, path: Path):
        return self._repository.save(config, Path(path))


class ValidateModelContract:
    def __init__(self, validator: ModelContractPort):
        self._validator = validator

    def execute(self, request: ModelContractRequest):
        if len(request.input_shape) != 3 or any(value <= 0 for value in request.input_shape):
            raise ValueError("Model input dimensions must be positive.")
        if request.output_size < 1:
            raise ValueError("At least one physics parameter needs a non-zero range.")
        return self._validator.validate(request)


class GenerateTrainsetPreview:
    def __init__(self, preview: TrainsetPreviewPort):
        self._preview = preview

    def execute(self, request: TrainsetPreviewRequest, *, on_progress=None) -> dict:
        if request.preview_count <= 0:
            raise ValueError("Preview sample count must be positive")
        if request.maximum < request.minimum:
            raise ValueError("Preview range maximum must be greater than or equal to minimum")
        return self._preview.generate_preview(request, on_progress=on_progress)


class SimulateTrainsetWhatIf:
    def __init__(self, preview: TrainsetPreviewPort):
        self._preview = preview

    def execute(self, request: TrainsetWhatIfRequest) -> dict:
        return self._preview.simulate_what_if(request)


class PrepareTrainsetJobPackage:
    def __init__(self, packages: TrainsetJobPackagePort):
        self._packages = packages

    def execute(self, request: PrepareTrainsetJobRequest) -> Path:
        return self._packages.prepare(request)


class ManageTrainsetLocalProcess:
    def __init__(self, processes: TrainsetLocalProcessPort):
        self._processes = processes

    def is_running(self) -> bool:
        return self._processes.is_running()

    def start(self, request: TrainsetLocalProcessRequest, **callbacks) -> None:
        if self._processes.is_running():
            raise RuntimeError("A local generation/training process is already running.")
        if not request.arguments:
            raise ValueError("A local process requires a command")
        self._processes.start(request, **callbacks)

    def set_paused(self, paused: bool) -> bool:
        return self._processes.set_paused(bool(paused))

    def cancel(self) -> bool:
        return self._processes.cancel()


class FindTrainedTrainsetModel:
    def __init__(self, registration: TrainsetModelRegistrationPort):
        self._registration = registration

    def execute(self, roots: tuple[Path, ...]) -> Path | None:
        return self._registration.find_model(tuple(Path(root) for root in roots))


class RegisterTrainsetPredictionModule:
    def __init__(self, registration: TrainsetModelRegistrationPort):
        self._registration = registration

    def execute(self, request: RegisterTrainsetModelRequest):
        return self._registration.register(request)


def _require_remote_config(config: dict) -> None:
    hpc = config.get("hpc", {})
    if not hpc.get("user") or not hpc.get("remote_path"):
        raise ValueError("Configure Maxwell user and remote project path first.")


class ManageTrainsetRemoteJobs:
    def __init__(self, remote: TrainsetRemoteJobPort):
        self._remote = remote

    def connection_check(self, config: dict) -> str:
        _require_remote_config(config)
        return self._remote.connection_check(config)

    def upload_and_submit(self, config: dict, package_dir: Path) -> dict[str, str]:
        _require_remote_config(config)
        return self._remote.upload_and_submit(config, Path(package_dir))

    def query(self, config: dict, job_id: str):
        _require_remote_config(config)
        return self._remote.query(config, str(job_id))

    def download_results(self, config: dict, destination: Path) -> str:
        _require_remote_config(config)
        return self._remote.download_results(config, Path(destination))


class LoadTrainsetMetrics:
    def __init__(self, metrics: TrainsetMetricsRepository):
        self._metrics = metrics

    def execute(self, path: Path):
        return self._metrics.load(Path(path))


class PrepareTrainsetDesign:
    def __init__(self, design: TrainsetDesignPort):
        self._design = design

    def load_reference(self, path: Path):
        return self._design.load_reference(Path(path))

    def crop(self, image, roi):
        return self._design.crop(image, dict(roi))

    def threshold_summary(
        self,
        image,
        roi,
        threshold,
        *,
        automatic_upper,
        lower,
        upper,
    ):
        return self._design.threshold_summary(
            image,
            dict(roi),
            dict(threshold),
            automatic_upper=bool(automatic_upper),
            lower=float(lower),
            upper=float(upper),
        )

    def overlay(self, image, roi, config, random_mask=None):
        return self._design.overlay(
            image, dict(roi), config, random_mask
        )

    def random_mask(self, shape, config):
        return self._design.random_mask(tuple(shape), config)

    def geometry_ranges(self, config):
        return roi_to_spherical_ranges(config)


class ManageTrainsetConfiguration:
    def __init__(self, configuration: TrainsetConfigurationPort):
        self._configuration = configuration

    def default(self):
        return self._configuration.default()

    def merge_with_defaults(self, values):
        return self._configuration.synchronize(
            self._configuration.merge(self._configuration.default(), values)
        )

    def synchronize(self, config):
        return self._configuration.synchronize(config)

    def validate(self, config, **options):
        return self._configuration.validate(config, **options)
