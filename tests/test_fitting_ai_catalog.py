from src.gimap.features.fitting.application import AiFittingCatalog
from src.gimap.features.fitting.infrastructure.adapters import (
    LegacyAiFittingCatalogAdapter,
)


def test_ai_catalog_preserves_profiles_and_model_discovery(tmp_path):
    catalog = AiFittingCatalog(LegacyAiFittingCatalogAdapter())

    assert catalog.profile_names() == ("Fast", "Balanced", "Exhaustive")
    assert catalog.default_profile_name == "Balanced"
    assert catalog.profile("Fast").random_seed == 123

    model_dir = tmp_path / "example_model"
    model_dir.mkdir()
    artifact = model_dir / "model.keras"
    artifact.write_bytes(b"placeholder")

    found = catalog.discover_model(model_dir)
    assert len(found) == 1
    assert found[0].artifact_path == artifact


def test_ai_catalog_default_directories_keep_both_legacy_spellings(tmp_path):
    catalog = AiFittingCatalog(LegacyAiFittingCatalogAdapter())

    assert catalog.default_model_directories(tmp_path) == (
        tmp_path / "modules/Fitting_1D_Model",
        tmp_path / "modules/Fitting_1D_model",
    )
