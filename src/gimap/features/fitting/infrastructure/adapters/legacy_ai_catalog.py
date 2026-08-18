"""Adapter for existing AI fitting registry modules."""

from __future__ import annotations

from pathlib import Path


class LegacyAiFittingCatalogAdapter:
    @property
    def default_profile_name(self) -> str:
        from utils.ai_fitting_profiles import DEFAULT_PROFILE_NAME

        return DEFAULT_PROFILE_NAME

    def profile_names(self):
        from utils.ai_fitting_profiles import profile_registry

        return profile_registry.names()

    def profile(self, name):
        from utils.ai_fitting_profiles import profile_registry

        return profile_registry.get(name)

    def default_model_directories(self, root: Path):
        from utils.ai_fitting_models import default_ai_fitting_model_base_dirs

        return tuple(default_ai_fitting_model_base_dirs(root))

    def discover_models(self, directories):
        from utils.ai_fitting_models import discover_ai_fitting_models

        return tuple(discover_ai_fitting_models(directories))

    def discover_model(self, path):
        from utils.ai_fitting_models import discover_model_in_path

        return tuple(discover_model_in_path(path))
