"""Adapter for feature-owned AI fitting profiles and model discovery."""

from __future__ import annotations

from pathlib import Path

from ...application.fitting_profiles import DEFAULT_PROFILE_NAME, profile_registry
from .ai_model_registry import (
    default_ai_fitting_model_base_dirs,
    discover_ai_fitting_models,
    discover_model_in_path,
)


class AiFittingCatalogAdapter:
    @property
    def default_profile_name(self) -> str:
        return DEFAULT_PROFILE_NAME

    def profile_names(self):
        return profile_registry.names()

    def profile(self, name):
        return profile_registry.get(name)

    def default_model_directories(self, root: Path):
        return tuple(default_ai_fitting_model_base_dirs(root))

    def discover_models(self, directories):
        return tuple(discover_ai_fitting_models(directories))

    def discover_model(self, path):
        return tuple(discover_model_in_path(path))
