"""Composition seam for in-situ cut processing and refinement."""

from .insitu_cut_processing import InsituCutProcessingMixin
from .insitu_refinement_lifecycle import InsituRefinementLifecycleMixin


class InsituCutRefinementMixin(
    InsituCutProcessingMixin,
    InsituRefinementLifecycleMixin,
):
    """Compose the focused in-situ fitting presentation concerns."""
