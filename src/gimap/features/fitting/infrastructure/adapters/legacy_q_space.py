"""Adapter for the existing q-space calculator."""

from __future__ import annotations


class LegacyQSpaceAdapter:
    def create_detector(self, **geometry):
        from utils.q_space_calculator import create_detector_from_image_and_params

        return create_detector_from_image_and_params(**geometry)

    def axis_labels_and_extent(self, detector):
        from utils.q_space_calculator import get_q_axis_labels_and_extents

        return get_q_axis_labels_and_extents(detector)
