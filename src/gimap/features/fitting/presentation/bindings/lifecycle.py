"""Compose focused lifecycle bindings."""

from .fitting_ui_lifecycle import FittingUiLifecycleMixin
from .fitting_session_state import FittingSessionStateMixin


class LifecycleMixin(FittingUiLifecycleMixin, FittingSessionStateMixin):
    """Compatibility composition for focused lifecycle bindings."""

    pass
