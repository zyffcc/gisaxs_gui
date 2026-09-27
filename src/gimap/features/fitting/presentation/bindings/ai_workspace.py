"""Compose focused ai workspace bindings."""

from .ai_settings import AiSettingsMixin
from .workflow_v5_binding import WorkflowV5BindingMixin
from .ai_workspace_dialog import AiWorkspaceDialogMixin
from .ai_model_controls import AiModelControlsMixin
from .ai_workspace_state import AiWorkspaceStateMixin


class AiWorkspaceMixin(
    WorkflowV5BindingMixin, AiSettingsMixin, AiWorkspaceDialogMixin, AiModelControlsMixin, AiWorkspaceStateMixin
):
    """Compatibility composition for focused ai workspace bindings."""

    pass
