"""Assistant presentation: the start dialog, the AI panel, the settings page and the guided analysis page."""

from .choice_dialog import ChoiceDialog, GuiChooser
from .controller import AssistantController
from .gui_bridge import GuiBridge, RunCancelled
from .guided_analysis import GuidedAnalysis
from .guided_results import GuidedResultsPanel, automatic_outcome
from .gui_workbench import GuiConfirmer, GuiWorkbench
from .panel import AssistantPanel
from .report_view import report_html
from .services import AssistantServices
from .settings_page import AssistantSettingsPage
from .start_dialog import AssistantStartDialog

__all__ = [
    "AssistantController",
    "AssistantPanel",
    "AssistantServices",
    "AssistantSettingsPage",
    "AssistantStartDialog",
    "ChoiceDialog",
    "GuiChooser",
    "GuiBridge",
    "GuiConfirmer",
    "GuiWorkbench",
    "GuidedAnalysis",
    "GuidedResultsPanel",
    "automatic_outcome",
    "RunCancelled",
    "report_html",
]
