from pathlib import Path

from utils.ai_fitting_models import ModelRegistry, discover_ai_fitting_models, validate_model_info


ROOT = Path(__file__).resolve().parents[1]
MODEL_BASE = ROOT / "modules" / "Fitting_1D_Model"


def test_new_k1_k4_model_is_discoverable_and_contract_is_complete():
    models = discover_ai_fitting_models([MODEL_BASE])
    matches = [item for item in models if item.model_id == "gisaxs-k1-k4-phys-constraints"]
    assert matches
    keras = next(item for item in matches if item.artifact_type == "keras")
    assert keras.contract.supported_k == (1, 2, 3, 4)
    assert {"d_allowed", "d_spacing_rule"}.issubset(keras.contract.required_inputs)
    assert "d_present_logit" in keras.contract.required_outputs
    assert keras.training_status["state"] == "complete"
    validate_model_info(keras, verify_checksum=True)


def test_registry_is_lazy_until_load_is_requested():
    registry = ModelRegistry([MODEL_BASE])
    models = registry.refresh()
    assert models
    assert registry._loaded == {}
