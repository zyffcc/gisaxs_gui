"""Compose focused insitu execution bindings."""

from .insitu_curve_processing import InsituCurveProcessingMixin
from .insitu_persistence_preview import InsituPersistencePreviewMixin
from .insitu_refinement_lifecycle import InsituRefinementLifecycleMixin
from .insitu_sequence import InsituSequenceMixin


class InsituExecutionMixin(
    InsituSequenceMixin,
    InsituCurveProcessingMixin,
    InsituRefinementLifecycleMixin,
    InsituPersistencePreviewMixin,
):
    """Queue, load, fit and record the curves of an in-situ series."""
