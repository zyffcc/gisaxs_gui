import ast
import importlib
from pathlib import Path

import pytest

from src.gimap.features.trainset.application import (
    GeneratedTrainset,
    GenerateTrainset,
    GenerateTrainsetRequest,
    GenerateTrainsetPreview,
    LoadTrainsetProject,
    ManageTrainsetLocalProcess,
    ManageTrainsetRemoteJobs,
    LoadTrainsetMetrics,
    ManageTrainsetConfiguration,
    ModelContractRequest,
    ModelContractResult,
    PrepareTrainsetDesign,
    PrepareTrainsetJobPackage,
    PrepareTrainsetJobRequest,
    RegisterTrainsetModelRequest,
    RegisterTrainsetPredictionModule,
    RegisteredTrainsetModel,
    SimulateTrainsetWhatIf,
    SaveTrainsetProject,
    TrainsetPreviewRequest,
    TrainsetWhatIfRequest,
    TrainsetLocalProcessRequest,
    ValidateModelContract,
)


ROOT = Path(__file__).resolve().parents[1]


class _Generator:
    def __init__(self):
        self.request = None

    def generate(self, request, *, on_progress=None, pause=None):
        self.request = request
        return GeneratedTrainset(value={"mode": request.mode})


class _Configs:
    def __init__(self):
        self.saved = None

    def save(self, config, path):
        self.saved = (config, path)
        return path

    def load(self, path):
        assert self.saved[1] == path
        return self.saved[0]


class _ModelContract:
    def validate(self, request):
        return ModelContractResult(
            static_summary=f"Input  {request.input_shape}",
            output_shape=(1, request.output_size),
            trainable_weights=17,
        )


class _Preview:
    def __init__(self):
        self.preview_request = None
        self.what_if_request = None

    def generate_preview(self, request, *, on_progress=None):
        self.preview_request = request
        if on_progress:
            on_progress(1, "started")
        return {"preview": request.compared_text}

    def simulate_what_if(self, request):
        self.what_if_request = request
        return {"values": request.sampled}


class _LocalProcesses:
    def __init__(self):
        self.request = None
        self.paused = None
        self.cancelled = False

    def is_running(self):
        return False

    def start(self, request, **callbacks):
        self.request = request
        callbacks["on_started"]()

    def set_paused(self, paused):
        self.paused = paused
        return True

    def cancel(self):
        self.cancelled = True
        return True


class _RemoteJobs:
    def connection_check(self, config):
        return f"ok:{config['hpc']['user']}"

    def upload_and_submit(self, config, package_dir):
        return {"generate_job_id": "1", "train_job_id": "2"}

    def query(self, config, job_id):
        return job_id, "log"

    def download_results(self, config, destination):
        return str(destination)


class _Metrics:
    def load(self, path):
        return ({"epoch": 1, "path": str(path)},)


class _Packages:
    def __init__(self):
        self.request = None

    def prepare(self, request):
        self.request = request
        return request.workspace / "package"


class _Registration:
    def __init__(self):
        self.request = None

    def find_model(self, roots):
        return roots[0] / "best.keras"

    def register(self, request):
        self.request = request
        return RegisteredTrainsetModel("trained", request.modules_root / "trained")


class _Design:
    def load_reference(self, path):
        return ("image", path)

    def crop(self, image, roi):
        return image, roi

    def threshold_summary(self, image, roi, threshold, **options):
        return image, roi, threshold, options

    def overlay(self, image, roi, config, random_mask):
        return {"image": image, "roi": roi, "random_mask": random_mask}

    def random_mask(self, shape, config):
        return shape, config


class _Configuration:
    def default(self):
        return {"schema_version": 2, "sample": {}}

    def merge(self, base, override):
        return {**base, **override}

    def synchronize(self, config):
        return {**config, "synchronized": True}

    def validate(self, config, **options):
        return bool(config.get("schema_version")), [], list(options)


def _imports(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return names


def test_generation_and_project_use_cases_depend_only_on_ports(tmp_path):
    generator = _Generator()
    output = GenerateTrainset(generator).execute(
        GenerateTrainsetRequest({"project": {}}, 3, mode="demo")
    )
    configs = _Configs()
    path = tmp_path / "project.yaml"
    SaveTrainsetProject(configs).execute({"seed": 42}, path)
    loaded = LoadTrainsetProject(configs).execute(path)

    assert output.value == {"mode": "demo"}
    assert generator.request.sample_count == 3
    assert loaded == {"seed": 42}


def test_generation_use_case_rejects_invalid_request_before_adapter():
    with pytest.raises(ValueError, match="positive"):
        GenerateTrainset(_Generator()).execute(GenerateTrainsetRequest({}, 0))


def test_model_contract_use_case_uses_port_without_tensorflow():
    result = ValidateModelContract(_ModelContract()).execute(
        ModelContractRequest(
            input_shape=(32, 32, 1),
            output_size=3,
            model_config={"layers": [{"type": "flatten"}]},
        )
    )

    assert result.output_shape == (1, 3)
    assert result.trainable_weights == 17


def test_model_contract_use_case_rejects_zero_outputs_before_adapter():
    with pytest.raises(ValueError, match="non-zero range"):
        ValidateModelContract(_ModelContract()).execute(
            ModelContractRequest((32, 32, 1), 0, {})
        )


def test_preview_use_cases_delegate_without_qapplication_or_bornagain():
    preview = _Preview()
    progress = []
    preview_request = TrainsetPreviewRequest(
        config={"dataset": {}},
        plugin="physics",
        key="radius_nm",
        minimum=1.0,
        maximum=2.0,
        compared_text="Radius",
        preview_count=3,
        realization=0,
    )
    what_if_request = TrainsetWhatIfRequest(
        config={"sample": {}}, sampled={"radius_nm": 1.5}, realization=0
    )

    result = GenerateTrainsetPreview(preview).execute(
        preview_request, on_progress=lambda *args: progress.append(args)
    )
    what_if = SimulateTrainsetWhatIf(preview).execute(what_if_request)

    assert result == {"preview": "Radius"}
    assert what_if == {"values": {"radius_nm": 1.5}}
    assert preview.preview_request is preview_request
    assert preview.what_if_request is what_if_request
    assert progress == [(1, "started")]


def test_local_process_use_case_owns_lifecycle_without_qprocess(tmp_path):
    processes = _LocalProcesses()
    use_case = ManageTrainsetLocalProcess(processes)
    started = []
    request = TrainsetLocalProcessRequest(
        tmp_path, Path("/usr/bin/python3"), ("generate_dataset.py",)
    )
    callbacks = {
        "on_started": lambda: started.append(True),
        "on_progress": lambda *_args: None,
        "on_log": lambda _message: None,
        "on_finished": lambda _code: None,
        "on_error": lambda _message: None,
    }

    use_case.start(request, **callbacks)

    assert processes.request is request
    assert started == [True]
    assert use_case.set_paused(True)
    assert processes.paused is True
    assert use_case.cancel()
    assert processes.cancelled


def test_remote_job_and_metric_use_cases_depend_only_on_ports(tmp_path):
    config = {"hpc": {"user": "scientist", "remote_path": "/project"}}
    remote = ManageTrainsetRemoteJobs(_RemoteJobs())

    assert remote.connection_check(config) == "ok:scientist"
    assert remote.upload_and_submit(config, tmp_path)["train_job_id"] == "2"
    assert remote.query(config, "42") == ("42", "log")
    assert remote.download_results(config, tmp_path) == str(tmp_path)
    metrics = LoadTrainsetMetrics(_Metrics()).execute(tmp_path / "metrics.jsonl")
    assert metrics[0]["epoch"] == 1


def test_job_package_and_model_registration_use_cases_depend_only_on_ports(tmp_path):
    packages = _Packages()
    package_request = PrepareTrainsetJobRequest({}, tmp_path, ROOT)
    assert PrepareTrainsetJobPackage(packages).execute(package_request) == (
        tmp_path / "package"
    )
    assert packages.request is package_request

    registration = _Registration()
    register_request = RegisterTrainsetModelRequest(
        {}, tmp_path / "best.keras", tmp_path / "modules"
    )
    registered = RegisterTrainsetPredictionModule(registration).execute(
        register_request
    )
    assert registered.module_name == "trained"
    assert registration.request is register_request


def test_design_and_configuration_use_cases_depend_only_on_ports(tmp_path):
    design = PrepareTrainsetDesign(_Design())
    image = design.load_reference(tmp_path / "reference.cbf")
    assert image == ("image", tmp_path / "reference.cbf")
    assert design.crop("pixels", {"x": 1}) == ("pixels", {"x": 1})
    summary = design.threshold_summary(
        "pixels",
        {"x": 1},
        {"mode": "fixed"},
        automatic_upper=False,
        lower=1,
        upper=2,
    )
    assert summary[-1] == {
        "automatic_upper": False,
        "lower": 1.0,
        "upper": 2.0,
    }
    assert design.random_mask((32, 32), {"mask": {}})[0] == (32, 32)

    configuration = ManageTrainsetConfiguration(_Configuration())
    merged = configuration.merge_with_defaults({"project": {"name": "demo"}})
    assert merged["schema_version"] == 2
    assert merged["synchronized"] is True
    assert configuration.validate(merged, require_hpc=False)[0] is True


def test_legacy_trainset_has_no_concrete_bornagain_import():
    for relative in (
        "trainset/config.py",
        "trainset/generator.py",
        "trainset/grid_cache.py",
        "trainset/simulation.py",
    ):
        imports = _imports(ROOT / relative)
        assert "bornagain" not in imports
        assert not any("integrations.bornagain" in name for name in imports)


def test_trainset_view_binding_receives_simulation_port_from_composition_root():
    binding = (
        ROOT / "src/gimap/features/trainset/presentation/view_binding.py"
    ).read_text(encoding="utf-8")
    main_controller = (ROOT / "src/gimap/app/runtime.py").read_text(
        encoding="utf-8"
    )
    composition_root = (ROOT / "main.py").read_text(encoding="utf-8")

    assert "simulation_port: SimulationPort" in binding
    assert "TrainsetViewModel" in binding
    assert "DatasetGenerator(" not in binding
    assert "simulate_pattern(" not in binding
    assert "BornAgainSimulator(" not in main_controller
    assert "simulation_port=simulation_port" in main_controller
    assert "simulation_port=BornAgainSimulator(" in composition_root


def test_trainset_view_binding_does_not_import_tensorflow_runtime():
    imports = _imports(ROOT / "src/gimap/features/trainset/presentation/view_binding.py")

    assert not any(name.startswith("tensorflow") for name in imports)
    assert "create_trainset_view_model" in (
        ROOT / "src/gimap/app/runtime.py"
    ).read_text(encoding="utf-8")


def test_trainset_view_binding_uses_settings_repository_not_global_singleton():
    binding = (ROOT / "src/gimap/features/trainset/presentation/view_binding.py").read_text(
        encoding="utf-8"
    )

    assert "core.global_params" not in binding
    assert "TrainsetViewModel" in binding
    assert "self.trainset_view_model.load_settings" in binding
    assert "self.trainset_view_model.save_settings" in binding


def test_trainset_presentation_does_not_import_infrastructure_or_manage_files_and_processes():
    binding_path = ROOT / "src/gimap/features/trainset/presentation/view_binding.py"
    source = binding_path.read_text(encoding="utf-8")
    imports = _imports(binding_path)

    assert not any("infrastructure" in name for name in imports)
    assert not any("trainset.domain" in name for name in imports)
    page_imports = _imports(
        ROOT / "src/gimap/features/trainset/presentation/page.py"
    )
    assert not any("trainset.domain" in name for name in page_imports)
    for forbidden in (
        "DatasetGenerator(",
        "QProcess",
        ".write_text(",
        "read_metrics(",
        "SlurmBackend(",
        "load_scattering_image(",
        "self.package_dir = prepare_job_package(",
    ):
        assert forbidden not in source


def test_legacy_trainset_controller_path_is_a_thin_reexport():
    source = (ROOT / "controllers/trainset_controller.py").read_text(encoding="utf-8")

    assert "src.gimap.features.trainset.presentation.legacy_bridge" in source
    assert len(source.splitlines()) <= 8

    feature_legacy = (
        ROOT / "src/gimap/features/trainset/presentation/legacy_bridge.py"
    ).read_text(encoding="utf-8")
    assert "from .view_binding import TrainsetViewBinding" in feature_legacy
    assert len(feature_legacy.splitlines()) <= 8


def test_legacy_trainset_modules_alias_feature_owned_implementations():
    pairs = {
        "trainset.backends": "src.gimap.features.trainset.infrastructure.adapters.job_backends",
        "trainset.config": "src.gimap.features.trainset.infrastructure.adapters.configuration",
        "trainset.generator": "src.gimap.features.trainset.infrastructure.adapters.dataset_generator",
        "trainset.grid_cache": "src.gimap.features.trainset.infrastructure.adapters.grid_cache",
        "trainset.job_package": "src.gimap.features.trainset.infrastructure.adapters.portable_job_package",
        "trainset.simulation": "src.gimap.features.trainset.application.simulation",
    }

    for legacy_name, owner_name in pairs.items():
        assert importlib.import_module(legacy_name) is importlib.import_module(owner_name)
        legacy_source = ROOT.joinpath(*legacy_name.split(".")).with_suffix(".py")
        assert len(legacy_source.read_text(encoding="utf-8").splitlines()) <= 8


def test_domain_modules_do_not_import_runtime_or_io_infrastructure():
    forbidden = ("PyQt", "PySide", "tensorflow", "bornagain", "h5py", "yaml")
    domain = ROOT / "src/gimap/features/trainset/domain"
    for path in domain.glob("*.py"):
        imports = _imports(path)
        assert not any(
            name.startswith(prefix) for name in imports for prefix in forbidden
        )
