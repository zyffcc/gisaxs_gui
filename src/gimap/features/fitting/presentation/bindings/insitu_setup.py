"""Compose focused In-situ page, Recipe and execution settings bindings."""

from .insitu_page_binding import InsituPageBindingMixin
from .insitu_recipe_binding import InsituRecipeBindingMixin
from .insitu_watch_settings import InsituWatchSettingsMixin


class InsituSetupMixin(
    InsituPageBindingMixin,
    InsituRecipeBindingMixin,
    InsituWatchSettingsMixin,
):
    """Compose In-situ setup concerns without a second dialog implementation."""
