"""Detector reference and mask-design adapter."""

from __future__ import annotations

import numpy as np

from .dataset_generator import (
    build_fixed_mask,
    build_random_mask,
    build_roi_shape_mask,
    crop_roi,
    load_scattering_image,
    merge_threshold_mask,
)


class TrainsetDesignAdapter:
    def load_reference(self, path):
        return load_scattering_image(path)

    def crop(self, image, roi):
        return crop_roi(image, roi)

    def threshold_summary(
        self,
        image,
        roi,
        threshold,
        *,
        automatic_upper,
        lower,
        upper,
    ):
        roi_image = crop_roi(image, roi)
        if not roi_image.size:
            raise ValueError("The selected ROI is empty.")
        if automatic_upper:
            quantile = float(threshold.get("upper_quantile", 99.999))
            finite = roi_image[np.isfinite(roi_image)]
            if not finite.size:
                raise ValueError("The ROI has no finite intensity values.")
            upper = float(np.percentile(finite, quantile))
        invalid = ~np.isfinite(roi_image)
        below = np.isfinite(roi_image) & (roi_image < lower)
        above = np.isfinite(roi_image) & (roi_image > upper)
        total = int(roi_image.size)
        return {
            "upper": float(upper),
            "lower": float(lower),
            "total": total,
            "masked": int(np.count_nonzero(invalid | below | above)),
            "below": int(np.count_nonzero(below)),
            "above": int(np.count_nonzero(above)),
            "invalid": int(np.count_nonzero(invalid)),
        }

    def overlay(self, image, roi, config, random_mask):
        roi_image = crop_roi(image, roi)
        if config.get("mask", {}).get("mode") == "random":
            if random_mask is None or random_mask.shape != roi_image.shape:
                random_mask = build_random_mask(
                    roi_image.shape, config, np.random.default_rng()
                )
            mask = merge_threshold_mask(roi_image, random_mask, config)
            mask_label = "random example + threshold (preview only)"
        else:
            random_mask = None
            mask = build_fixed_mask(roi_image, config)
            mask_label = "fixed"
        return {
            "roi_image": roi_image,
            "mask": mask,
            "roi_shape_mask": build_roi_shape_mask(roi_image.shape, config),
            "random_mask": random_mask,
            "mask_label": mask_label,
        }

    def random_mask(self, shape, config):
        return build_random_mask(shape, config, np.random.default_rng())
