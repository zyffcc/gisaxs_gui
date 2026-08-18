"""Compose focused particle editor bindings."""

from .particle_registry import ParticleRegistryMixin
from .particle_widget_editor import ParticleWidgetEditorMixin
from .particle_connections import ParticleConnectionsMixin


class ParticleEditorMixin(
    ParticleRegistryMixin, ParticleWidgetEditorMixin, ParticleConnectionsMixin
):
    """Compatibility composition for focused particle editor bindings."""

    pass
