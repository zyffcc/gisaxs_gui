"""Compose focused ai review bindings."""

from .ai_constraints import AiConstraintsMixin
from .global_parameter_controls import GlobalParameterControlsMixin


class AiReviewMixin(AiConstraintsMixin, GlobalParameterControlsMixin):
    """Compatibility composition for focused ai review bindings."""

    pass
