"""Qt-free Trainset ViewModel."""

from __future__ import annotations

from dataclasses import replace

from src.gimap.app import AppContext

from ..application import (
    ModelContractRequest,
    TrainsetPreviewRequest,
    TrainsetWhatIfRequest,
)
from .state import TrainsetState


class TrainsetViewModel:
    def __init__(
        self,
        *,
        context: AppContext,
        generate_preview,
        simulate_what_if,
        validate_model_contract,
        load_project,
        save_project,
        prepare_job_package,
        local_processes,
        find_trained_model,
        register_prediction_module,
        remote_jobs,
        load_metrics,
        prepare_design,
        configuration,
        catalog,
    ):
        self.context = context
        self._generate_preview = generate_preview
        self._simulate_what_if = simulate_what_if
        self._validate_model_contract = validate_model_contract
        self._load_project = load_project
        self._save_project = save_project
        self._prepare_job_package = prepare_job_package
        self._local_processes = local_processes
        self._find_trained_model = find_trained_model
        self._register_prediction_module = register_prediction_module
        self._remote_jobs = remote_jobs
        self._load_metrics = load_metrics
        self._prepare_design = prepare_design
        self._configuration = configuration
        self.catalog = catalog
        self.state = TrainsetState()

    def load_settings(self, *, reload: bool = False) -> dict[str, object]:
        if reload:
            self.context.settings.reload()
        return dict(self.context.settings.get_section("trainset"))

    def save_settings(self, values: dict[str, object]) -> None:
        self.context.settings.update_section("trainset", dict(values))
        self.context.settings.save()

    def generate_preview(
        self, request: TrainsetPreviewRequest, *, on_progress=None
    ) -> dict | None:
        self.state = replace(
            self.state,
            preview_status="running",
            error_message=None,
            status_message="Generating preview",
        )
        try:
            result = self._generate_preview.execute(
                request, on_progress=on_progress
            )
        except Exception as exc:
            self.state = replace(
                self.state,
                preview_status="error",
                error_message=str(exc),
                status_message=str(exc),
            )
            return None
        self.state = replace(
            self.state,
            preview_status="ready",
            error_message=None,
            status_message="Preview ready",
        )
        return result

    def simulate_what_if(self, request: TrainsetWhatIfRequest) -> dict | None:
        self.state = replace(
            self.state, what_if_status="running", error_message=None
        )
        try:
            result = self._simulate_what_if.execute(request)
        except Exception as exc:
            self.state = replace(
                self.state,
                what_if_status="error",
                error_message=str(exc),
                status_message=str(exc),
            )
            return None
        self.state = replace(
            self.state, what_if_status="ready", error_message=None
        )
        return result

    def validate_model_contract(self, request: ModelContractRequest):
        try:
            return self._validate_model_contract.execute(request)
        except Exception as exc:
            self.state = replace(
                self.state, error_message=str(exc), status_message=str(exc)
            )
            return None

    def load_project(self, path):
        try:
            return self._load_project.execute(path)
        except Exception as exc:
            self.state = replace(
                self.state, error_message=str(exc), status_message=str(exc)
            )
            raise

    def save_project(self, config, path):
        try:
            return self._save_project.execute(config, path)
        except Exception as exc:
            self.state = replace(
                self.state, error_message=str(exc), status_message=str(exc)
            )
            raise

    def prepare_job_package(self, request):
        try:
            return self._prepare_job_package.execute(request)
        except Exception as exc:
            self.state = replace(
                self.state, error_message=str(exc), status_message=str(exc)
            )
            return None

    def local_process_running(self) -> bool:
        return self._local_processes.is_running()

    def start_local_process(self, request, **callbacks) -> bool:
        try:
            self._local_processes.start(request, **callbacks)
        except Exception as exc:
            self.state = replace(
                self.state, error_message=str(exc), status_message=str(exc)
            )
            return False
        return True

    def set_local_process_paused(self, paused: bool) -> bool:
        return self._local_processes.set_paused(paused)

    def cancel_local_process(self) -> bool:
        return self._local_processes.cancel()

    def find_trained_model(self, roots):
        try:
            return self._find_trained_model.execute(tuple(roots))
        except Exception as exc:
            self.state = replace(
                self.state, error_message=str(exc), status_message=str(exc)
            )
            return None

    def register_prediction_module(self, request):
        try:
            return self._register_prediction_module.execute(request)
        except Exception as exc:
            self.state = replace(
                self.state, error_message=str(exc), status_message=str(exc)
            )
            return None

    def check_remote_connection(self, config):
        return self._remote_call(self._remote_jobs.connection_check, config)

    def submit_remote_job(self, config, package_dir):
        return self._remote_call(
            self._remote_jobs.upload_and_submit, config, package_dir
        )

    def query_remote_job(self, config, job_id):
        return self._remote_call(self._remote_jobs.query, config, job_id)

    def download_remote_results(self, config, destination):
        return self._remote_call(
            self._remote_jobs.download_results, config, destination
        )

    def load_metrics(self, path):
        try:
            return self._load_metrics.execute(path)
        except Exception as exc:
            self.state = replace(
                self.state, error_message=str(exc), status_message=str(exc)
            )
            return ()

    def _remote_call(self, command, *args):
        try:
            return command(*args)
        except Exception as exc:
            self.state = replace(
                self.state, error_message=str(exc), status_message=str(exc)
            )
            raise

    def load_reference(self, path):
        return self._design_call(self._prepare_design.load_reference, path)

    def crop_reference(self, image, roi):
        return self._design_call(self._prepare_design.crop, image, roi)

    def threshold_summary(self, image, roi, threshold, **values):
        return self._design_call(
            self._prepare_design.threshold_summary,
            image,
            roi,
            threshold,
            **values,
        )

    def design_overlay(self, image, roi, config, random_mask=None):
        return self._design_call(
            self._prepare_design.overlay,
            image,
            roi,
            config,
            random_mask,
        )

    def generate_random_mask(self, shape, config):
        return self._design_call(
            self._prepare_design.random_mask, shape, config
        )

    def geometry_ranges(self, config):
        return self._design_call(self._prepare_design.geometry_ranges, config)

    def _design_call(self, command, *args, **kwargs):
        try:
            return command(*args, **kwargs)
        except Exception as exc:
            self.state = replace(
                self.state, error_message=str(exc), status_message=str(exc)
            )
            return None

    def default_config(self):
        return self._configuration.default()

    def merge_config_with_defaults(self, values):
        return self._configuration.merge_with_defaults(values)

    def synchronize_config(self, config):
        return self._configuration.synchronize(config)

    def validate_config(self, config, **options):
        return self._configuration.validate(config, **options)
