"""Compose focused insitu execution bindings."""

from .insitu_sequence import InsituSequenceMixin
from .insitu_cut_refinement import InsituCutRefinementMixin
from .insitu_persistence_preview import InsituPersistencePreviewMixin


class InsituExecutionMixin(
    InsituSequenceMixin, InsituCutRefinementMixin, InsituPersistencePreviewMixin
):
    """Compatibility composition for focused insitu execution bindings."""

    pass
