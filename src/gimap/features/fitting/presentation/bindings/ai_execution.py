"""Compose focused ai execution bindings."""

from .ai_input_data import AiInputDataMixin
from .ai_job_execution import AiJobExecutionMixin
from .ai_candidate_output import AiCandidateOutputMixin


class AiExecutionMixin(AiInputDataMixin, AiJobExecutionMixin, AiCandidateOutputMixin):
    """Compatibility composition for focused ai execution bindings."""

    pass
