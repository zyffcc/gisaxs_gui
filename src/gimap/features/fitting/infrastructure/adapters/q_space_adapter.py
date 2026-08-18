"""Adapter exposing feature-owned q-space geometry calculations."""

from __future__ import annotations

from ...domain.q_space_geometry import (
    create_detector_from_image_and_params,
    get_q_axis_labels_and_extents,
)


class QSpaceGeometryAdapter:
    def create_detector(self, **geometry):
        return create_detector_from_image_and_params(**geometry)

    def axis_labels_and_extent(self, detector):
        return get_q_axis_labels_and_extents(detector)
