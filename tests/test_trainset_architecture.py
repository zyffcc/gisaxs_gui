import ast
from pathlib import Path

import pytest

from src.gimap.features.trainset.application import (
    GeneratedTrainset,
    GenerateTrainset,
    GenerateTrainsetRequest,
    LoadTrainsetProject,
    SaveTrainsetProject,
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


def test_trainset_controller_receives_simulation_port_from_composition_root():
    controller = (ROOT / "controllers/trainset_controller.py").read_text(encoding="utf-8")
    main = (ROOT / "controllers/main_controller.py").read_text(encoding="utf-8")

    assert "simulation_port: SimulationPort" in controller
    assert "simulator=self.simulation_port" in controller
    assert "BornAgainSimulator(" in main
    assert "simulation_port=simulation_port" in main


def test_domain_modules_do_not_import_runtime_or_io_infrastructure():
    forbidden = ("PyQt", "PySide", "tensorflow", "bornagain", "h5py", "yaml")
    domain = ROOT / "src/gimap/features/trainset/domain"
    for path in domain.glob("*.py"):
        imports = _imports(path)
        assert not any(
            name.startswith(prefix) for name in imports for prefix in forbidden
        )
