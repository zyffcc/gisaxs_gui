"""Compose focused ai workspace bindings."""

from .ai_settings import AiSettingsMixin
from .ai_workspace_dialog import AiWorkspaceDialogMixin
from .ai_model_controls import AiModelControlsMixin
from .ai_workspace_state import AiWorkspaceStateMixin


class AiWorkspaceMixin(
    AiSettingsMixin, AiWorkspaceDialogMixin, AiModelControlsMixin, AiWorkspaceStateMixin
):
    """Compatibility composition for focused ai workspace bindings."""

    pass
