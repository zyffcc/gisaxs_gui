from src.gimap.features.fitting.application import ManageFittingModelParameters
from src.gimap.features.fitting.infrastructure.adapters import (
    LegacyFittingModelParametersAdapter,
)


def test_model_parameter_repository_preserves_legacy_json_and_particle_api(tmp_path):
    path = tmp_path / "model_parameters.json"
    parameters = ManageFittingModelParameters(
        LegacyFittingModelParametersAdapter(path)
    )

    parameters.replace_section(
        "fitting",
        {
            "global_parameters": {"background": 0.25},
            "particles": {},
        },
    )
    parameters.ensure_particle_entry("fitting", "particle_1", "Sphere")
    assert parameters.set_particle_parameter(
        "fitting", "particle_1", "sphere", "radius", 42.0
    )
    assert parameters.set_global_parameter("fitting", "k_value", 1.5)
    assert parameters.save_parameters()

    restored = ManageFittingModelParameters(
        LegacyFittingModelParametersAdapter(path)
    )
    assert restored.get_particle_shape("fitting", "particle_1") == "Sphere"
    assert restored.get_particle_parameter(
        "fitting", "particle_1", "sphere", "radius"
    ) == 42.0
    assert restored.get_global_parameter("fitting", "background") == 0.25
    assert restored.get_global_parameter("fitting", "k_value") == 1.5
    assert '"global_parameters"' in path.read_text(encoding="utf-8")
