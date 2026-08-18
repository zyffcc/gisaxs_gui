"""Focused Trainset adapter regression tests without external runtimes."""

from __future__ import annotations

import copy

import numpy as np
import yaml

from src.gimap.features.trainset.application import (
    RegisterTrainsetModelRequest,
    TrainsetPreviewRequest,
    TrainsetWhatIfRequest,
)
from src.gimap.features.trainset.infrastructure.adapters import (
    LocalTrainsetModelRegistrationAdapter,
    TrainsetPreviewAdapter,
)
from src.gimap.features.trainset.infrastructure.adapters.configuration import (
    default_project_config,
)


class _Simulation:
    def is_available(self):
        return True

    def simulate(self, config, sampled):
        height = int(config["roi"]["height"])
        width = int(config["roi"]["width"])
        value = float(sampled.get("radius_nm", 1.0))
        y, x = np.mgrid[:height, :width]
        return np.asarray(value + x + 2.0 * y, dtype=np.float32)


def _preview_config():
    config = default_project_config()
    config["dataset"]["number_of_samples"] = 8
    config["dataset"]["preview_samples"] = 4
    config["roi"].update({"x": 0, "y": 0, "width": 32, "height": 32})
    return config


def test_preview_adapter_preserves_shape_pipeline_and_cache_semantics():
    adapter = TrainsetPreviewAdapter(_Simulation())
    request = TrainsetPreviewRequest(
        config=_preview_config(),
        plugin="physics",
        key="radius_nm",
        minimum=2.0,
        maximum=4.0,
        compared_text="Physics · radius_nm",
        preview_count=4,
        realization=0,
    )

    first = adapter.generate_preview(request)
    second = adapter.generate_preview(request)

    assert first["comparison_images"]["midpoint"].shape == (32, 32)
    assert first["stats"]["tensor_shape"] == [1, 32, 32, 1]
    assert len(first["parameter_samples"]) == 4
    assert first["cache_misses"] == 3
    assert second["cache_hits"] == 3
    np.testing.assert_allclose(
        first["comparison_images"]["midpoint"],
        second["comparison_images"]["midpoint"],
    )


def test_what_if_adapter_keeps_physical_constraint_errors():
    adapter = TrainsetPreviewAdapter(_Simulation())
    config = _preview_config()
    config["sample"]["constraints"]["segment_height_le_2r"] = True

    try:
        adapter.simulate_what_if(
            TrainsetWhatIfRequest(
                config,
                {"radius_nm": 2.0, "height_nm": 5.0},
                0,
            )
        )
    except ValueError as exc:
        assert "h ≤ 2R" in str(exc)
    else:
        raise AssertionError("Expected the existing h ≤ 2R constraint")


def test_model_registration_adapter_preserves_module_yaml_contract(tmp_path):
    config = _preview_config()
    config["project"]["name"] = "Copper GISAXS"
    model = tmp_path / "best_model.keras"
    model.write_bytes(b"fake model")
    modules_root = tmp_path / "modules"
    adapter = LocalTrainsetModelRegistrationAdapter()

    registered = adapter.register(
        RegisterTrainsetModelRequest(copy.deepcopy(config), model, modules_root)
    )

    payload = yaml.safe_load(
        (registered.module_dir / "module.yaml").read_text(encoding="utf-8")
    )
    assert registered.module_name == "Copper GISAXS (trained)"
    assert payload["model"]["format"] == "tensorflow_keras"
    assert payload["io"]["input_shape"] == [1, 32, 32, 1]
    assert payload["outputs"]["parameter_names"]
    assert (registered.module_dir / "preprocessing.py").is_file()
