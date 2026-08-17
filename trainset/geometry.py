"""Legacy compatibility for trainset domain geometry。"""

from src.gimap.features.trainset.domain.geometry import (
    q_vectors,
    roi_to_spherical_ranges,
)

__all__ = ["q_vectors", "roi_to_spherical_ranges"]
