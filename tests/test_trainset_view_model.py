"""Trainset ViewModel tests without QApplication or scientific runtimes."""

from __future__ import annotations

from types import SimpleNamespace

from src.gimap.app import AppContext
from src.gimap.features.trainset.application import (
    ModelContractRequest,
    TrainsetPreviewRequest,
    TrainsetWhatIfRequest,
    TrainsetUiCatalog,
)
from src.gimap.features.trainset.presentation import TrainsetViewModel
from src.gimap.integrations.state import (
    InMemorySessionRepository,
    InMemorySettingsRepository,
    InMemoryUserPreferencesRepository,
)


class _Call:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.calls = []

    def execute(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        if self.error:
            raise self.error
        return self.result

    def is_running(self):
        self.calls.append((("is_running",), {}))
        return False

    def start(self, *args, **kwargs):
        self.calls.append((args, kwargs))

    def set_paused(self, paused):
        self.calls.append((("set_paused", paused), {}))
        return True

    def cancel(self):
        self.calls.append((("cancel",), {}))
        return True


def _view_model(**overrides):
    calls = {
        "generate_preview": _Call({"comparison_images": {}}),
        "simulate_what_if": _Call({"image": "array"}),
        "validate_model_contract": _Call(SimpleNamespace(output_shape=(1, 2))),
        "load_project": _Call({"schema_version": 2}),
        "save_project": _Call("project.yaml"),
        "prepare_job_package": _Call("package"),
        "local_processes": _Call(),
        "find_trained_model": _Call("best_model.keras"),
        "register_prediction_module": _Call(
            SimpleNamespace(module_name="Model", module_dir="modules/model")
        ),
        "remote_jobs": SimpleNamespace(
            connection_check=lambda config: "ok",
            upload_and_submit=lambda config, package: {"train_job_id": "2"},
            query=lambda config, job_id: ("status", "log"),
            download_results=lambda config, destination: "downloaded",
        ),
        "load_metrics": _Call(({"epoch": 1},)),
        "prepare_design": SimpleNamespace(
            load_reference=lambda path: "image",
            crop=lambda image, roi: "crop",
            threshold_summary=lambda *args, **kwargs: {"masked": 1},
            overlay=lambda *args, **kwargs: {"mask": "mask"},
            random_mask=lambda shape, config: "random",
            geometry_ranges=lambda config: {"phi_min_deg": -1.0},
        ),
        "configuration": SimpleNamespace(
            default=lambda: {"schema_version": 2},
            merge_with_defaults=lambda values: {"schema_version": 2, **values},
            synchronize=lambda config: config,
            validate=lambda config, **options: (True, [], []),
        ),
        "catalog": TrainsetUiCatalog(),
    }
    calls.update(overrides)
    context = AppContext(
        settings=InMemorySettingsRepository(),
        session=InMemorySessionRepository(),
        preferences=InMemoryUserPreferencesRepository(),
    )
    return TrainsetViewModel(context=context, **calls), calls


def test_trainset_view_model_delegates_preview_and_what_if_commands():
    view_model, calls = _view_model()
    preview_request = TrainsetPreviewRequest(
        {}, "physics", "radius_nm", 1.0, 2.0, "Radius", 3, 0
    )
    what_if_request = TrainsetWhatIfRequest({}, {"radius_nm": 1.5}, 0)

    assert view_model.generate_preview(preview_request) is not None
    assert view_model.simulate_what_if(what_if_request) is not None
    assert calls["generate_preview"].calls[0][0] == (preview_request,)
    assert calls["simulate_what_if"].calls[0][0] == (what_if_request,)
    assert view_model.state.preview_status == "ready"
    assert view_model.state.what_if_status == "ready"


def test_trainset_ui_catalog_exposes_stable_domain_metadata_through_application():
    view_model, _calls = _view_model()

    assert view_model.catalog.plugin("particle", "spherical_segment").label == (
        "Spherical segment"
    )
    assert view_model.catalog.plugin("interference", "none").label == "None"
    assert len(view_model.catalog.background_parameters()) == 18


def test_trainset_view_model_converts_preview_failure_to_state():
    view_model, _calls = _view_model(
        generate_preview=_Call(error=RuntimeError("simulation unavailable"))
    )
    request = TrainsetPreviewRequest(
        {}, "physics", "radius_nm", 1.0, 2.0, "Radius", 3, 0
    )

    assert view_model.generate_preview(request) is None
    assert view_model.state.preview_status == "error"
    assert view_model.state.error_message == "simulation unavailable"


def test_trainset_view_model_validates_model_contract_and_persists_settings():
    view_model, calls = _view_model()
    request = ModelContractRequest((32, 32, 1), 2, {})

    result = view_model.validate_model_contract(request)
    view_model.save_settings({"project": {"seed": 42}})

    assert result.output_shape == (1, 2)
    assert calls["validate_model_contract"].calls[0][0] == (request,)
    assert view_model.load_settings() == {"project": {"seed": 42}}


def test_trainset_view_model_delegates_project_persistence(tmp_path):
    view_model, calls = _view_model()
    config = {"schema_version": 2}
    path = tmp_path / "project.yaml"

    assert view_model.load_project(path) == config
    assert view_model.save_project(config, path) == "project.yaml"
    assert calls["load_project"].calls[0][0] == (path,)
    assert calls["save_project"].calls[0][0] == (config, path)


def test_trainset_view_model_delegates_local_process_lifecycle():
    view_model, calls = _view_model()
    callbacks = {
        "on_started": lambda: None,
        "on_progress": lambda *_args: None,
        "on_log": lambda _message: None,
        "on_finished": lambda _code: None,
        "on_error": lambda _message: None,
    }

    assert view_model.start_local_process("request", **callbacks)
    assert view_model.set_local_process_paused(True)
    assert view_model.cancel_local_process()
    assert calls["local_processes"].calls[0][0] == ("request",)


def test_trainset_view_model_delegates_remote_jobs_and_metrics(tmp_path):
    view_model, _calls = _view_model()
    config = {"hpc": {"user": "scientist"}}

    assert view_model.check_remote_connection(config) == "ok"
    assert view_model.submit_remote_job(config, tmp_path)["train_job_id"] == "2"
    assert view_model.query_remote_job(config, "2") == ("status", "log")
    assert view_model.download_remote_results(config, tmp_path) == "downloaded"
    assert view_model.load_metrics(tmp_path / "metrics.jsonl") == ({"epoch": 1},)
