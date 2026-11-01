"""Labs / 2D Prediction -> SEM comparison: only a radius axis the module declares is a physical scale.

A distribution output whose module declares no axes still gets display axes (0.05-15 nm, the old
fallback), and they used to pass the SEM gate (key ``r``, unit ``nm``). The SEM comparison would then
have rebinned the prediction onto a scale the module never stated. Now every axis says whether the module
declared it, and only a declared radius axis is handed to SEM.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.gimap.features.prediction.application import (
    PredictPreparedInput,
    PredictPreparedInputRequest,
)
from src.gimap.features.prediction.domain import PredictionResult, normalize_prediction_output
from src.gimap.features.prediction.infrastructure import InMemoryModelPathStore, YamlModuleRepository
from src.gimap.features.prediction.presentation.bindings.sem_comparison import (
    RADIUS_AXIS_NOT_DECLARED,
    SemComparisonMixin,
)

MODULES = Path(__file__).resolve().parents[1] / "modules"
FULL_AXIS = {"key": "r", "label": "R", "unit": "nm", "min": 1.0, "max": 4.0}


@pytest.mark.parametrize(
    "row, declared",
    [
        (None, False),  # nothing declared: the 0.05-15 nm placeholder
        ({"key": "r", "unit": "nm"}, False),  # no range
        ({"key": "r", "min": 1.0, "max": 4.0}, False),  # no unit
        ({"unit": "nm", "min": 1.0, "max": 4.0}, False),  # no key
        ({**FULL_AXIS, "min": 4.0, "max": 1.0}, False),  # an impossible range is replaced
        (FULL_AXIS, True),
    ],
)
def test_each_distribution_axis_says_whether_the_module_declared_it(row, declared):
    axes = {} if row is None else {"output_axes": {"row": row}}
    result = normalize_prediction_output(np.ones((3, 2), dtype=np.float32), axes)

    axis = result["distribution_axes"]["row"]
    assert axis["declared"] is declared
    assert axis["key"] == "r" and axis["unit"] == "nm"  # the display contract is unchanged
    if declared:
        np.testing.assert_allclose(result["r_x"], [1.5, 2.5, 3.5])


class _Predictor:
    def __init__(self, outputs):
        self.outputs = outputs

    def predict(self, request):
        return PredictionResult(outputs=self.outputs)


def _bridge(module, outputs):
    bridge = SimpleNamespace(
        _current_image=np.ones((4, 4)),
        _typed_prediction_module=lambda: module,
        current_parameters={"module_model_path": "synthetic_model"},
        _current_input_files=["synthetic_00001.cbf"],
        _current_input_stack=1,
        prediction_results=outputs,
        said=[],
        emitted=[],
    )
    bridge._append_status_message = lambda text, level="INFO": bridge.said.append((level, text))
    bridge._sem_prediction_for_comparison = lambda: SemComparisonMixin._sem_prediction_for_comparison(bridge)
    bridge.sem_comparison_requested = SimpleNamespace(emit=bridge.emitted.append)
    return bridge


@pytest.mark.parametrize(
    "folder, attached",
    [
        ("Ag_Polymer_Silicon_Ale", False),
        ("Ag_Polymer_Silicon_Geo", False),
        ("Cu_Polymer_Silicon_Sputter", False),
        ("Au_Silicon_15nm", True),  # declares R 0.05-15 nm and H 0.1-30 nm in its module.yaml
    ],
)
def test_only_a_module_that_declares_its_radius_axis_hands_the_prediction_to_sem(folder, attached):
    module = YamlModuleRepository(MODULES, InMemoryModelPathStore()).load(MODULES / folder / "module.yaml")
    distribution = np.full((1, 30, 30, 1), 1.0 / 900.0, dtype=np.float32)
    result = PredictPreparedInput(_Predictor(distribution)).execute(
        PredictPreparedInputRequest(np.zeros((1, 256, 256, 1), np.float32), module, Path("synthetic_model"))
    )
    outputs = result.outputs
    assert outputs["distribution_axes"]["row"]["declared"] is attached

    bridge = _bridge(module, outputs)
    SemComparisonMixin._capture_sem_prediction(bridge, outputs)
    SemComparisonMixin._on_sem_comparison_clicked(bridge)

    if attached:
        (snapshot,) = bridge.emitted
        assert snapshot["metadata"]["radius_axis"]["declared"] is True
        assert snapshot["radius_nm"].size == 30
        assert bridge.said == []
    else:
        assert bridge.emitted == [None]  # SEM opens without a prediction
        assert bridge.said == [("WARN", RADIUS_AXIS_NOT_DECLARED)]  # and the page says why


def test_the_refusal_is_not_repeated_for_a_later_result():
    module = YamlModuleRepository(MODULES, InMemoryModelPathStore()).load(
        MODULES / "Cu_Polymer_Silicon_Sputter" / "module.yaml"
    )
    outputs = normalize_prediction_output(np.ones((3, 3), np.float32), {})
    bridge = _bridge(module, outputs)
    SemComparisonMixin._capture_sem_prediction(bridge, outputs)
    bridge.prediction_results = {"scalars": np.ones(2)}  # another result is shown now

    SemComparisonMixin._on_sem_comparison_clicked(bridge)

    assert bridge.emitted == [None] and bridge.said == []
