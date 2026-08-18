"""Focused Trainset detector-data and generation behavior."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import cv2
import numpy as np


from .physical_background import generate_physical_background


def apply_preprocessing(
    image: np.ndarray,
    config: Dict[str, Any],
    mask: Optional[np.ndarray],
    rng: np.random.Generator,
    overrides: Optional[Dict[str, float]] = None,
    trace: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    overrides = overrides or {}
    current_mask = None if mask is None else np.asarray(mask, dtype=bool).copy()
    stages: List[Dict[str, Any]] = [
        {
            "name": "BornAgain Raw",
            "image": np.asarray(image, dtype=np.float32),
            "mask": current_mask,
        }
    ]
    current = np.asarray(image, dtype=np.float32).copy()
    mask_value = float(config.get("mask", {}).get("mask_value", -1.0))
    for step in config.get("preprocessing", {}).get("steps", []):
        if not step.get("enabled", False):
            continue
        plugin = step.get("plugin")
        if plugin == "physical_background":
            background_trace: Dict[str, float] = {}
            background = generate_physical_background(
                current, config, step, rng, overrides, background_trace
            )
            if trace is not None:
                trace["physical_background"] = background_trace
            current = np.maximum(current + background, 0.0).astype(np.float32)
        elif plugin in {"noise", "gaussian_noise"}:
            if "gaussian_noise.snr_db" in overrides:
                snr = float(overrides["gaussian_noise.snr_db"])
            else:
                snr = float(
                    rng.uniform(
                        float(step.get("snr_min_db", 80.0)), float(step.get("snr_max_db", 110.0))
                    )
                )
            power = float(np.mean(np.square(np.nan_to_num(current))))
            sigma = np.sqrt(power / max(10 ** (snr / 10.0), 1e-12))
            if trace is not None:
                trace["gaussian_noise"] = {"snr_db": float(snr), "sigma": float(sigma)}
            current = np.maximum(current + rng.normal(0.0, sigma, current.shape), 0.0).astype(
                np.float32
            )
        elif plugin == "poisson_noise":
            if "poisson_noise.count_scale" in overrides:
                count_scale = float(overrides["poisson_noise.count_scale"])
            else:
                count_scale = float(
                    rng.uniform(
                        float(step.get("count_scale_min", 1.0)),
                        float(step.get("count_scale_max", 20.0)),
                    )
                )
            count_scale = max(count_scale, 1e-12)
            if trace is not None:
                trace["poisson_noise"] = {"count_scale": float(count_scale)}
            current = (rng.poisson(np.maximum(current, 0.0) * count_scale) / count_scale).astype(
                np.float32
            )
        elif plugin == "mask" and current_mask is not None:
            if trace is not None:
                trace["mask"] = {
                    "masked_fraction": float(current_mask.mean()),
                    "threshold_enabled": bool(
                        config.get("mask", {}).get("threshold", {}).get("enabled", False)
                    ),
                }
            current = current.copy()
            current[current_mask] = mask_value
        elif plugin == "log":
            epsilon = float(step.get("epsilon", 1e-6))
            valid = (
                current != mask_value
                if current_mask is not None
                else np.ones(current.shape, dtype=bool)
            )
            transformed = np.full(current.shape, mask_value, dtype=np.float32)
            transformed[valid] = np.log(np.maximum(current[valid], 0.0) + epsilon)
            current = transformed
        elif plugin == "normalize":
            valid = (
                current != mask_value
                if current_mask is not None
                else np.ones(current.shape, dtype=bool)
            )
            values = current[valid]
            if values.size:
                mode = step.get("mode", "range")
                low, high = float(values.min()), float(values.max())
                if mode == "upper":
                    current[valid] = values / max(abs(high), 1e-12)
                elif mode == "lower":
                    current[valid] = values - low
                else:
                    out_low, out_high = float(step.get("lower", 0.0)), float(step.get("upper", 1.0))
                    current[valid] = out_low + (values - low) * (out_high - out_low) / max(
                        high - low, 1e-12
                    )
        elif plugin == "random_edge_crop":
            maximum = max(0, int(step.get("maximum_px", 0)))
            if maximum:
                output_height, output_width = current.shape[:2]
                top, bottom, left, right = [int(rng.integers(0, maximum + 1)) for _ in range(4)]
                if trace is not None:
                    trace["random_edge_crop"] = {
                        "top_px": top,
                        "bottom_px": bottom,
                        "left_px": left,
                        "right_px": right,
                    }
                cropped = current[
                    top : current.shape[0] - bottom or None, left : current.shape[1] - right or None
                ]
                if cropped.size:
                    current = cv2.resize(
                        cropped, (output_width, output_height), interpolation=cv2.INTER_AREA
                    ).astype(np.float32)
                    if current_mask is not None:
                        cropped_mask = current_mask[
                            top : current_mask.shape[0] - bottom or None,
                            left : current_mask.shape[1] - right or None,
                        ]
                        current_mask = cv2.resize(
                            cropped_mask.astype(np.uint8),
                            (output_width, output_height),
                            interpolation=cv2.INTER_NEAREST,
                        ).astype(bool)
                        current[current_mask] = mask_value
        stage_name = {
            "noise": "Gaussian Noise",
            "gaussian_noise": "Gaussian Noise",
            "poisson_noise": "Poisson Noise",
            "mask": (
                "Threshold + Detector Mask"
                if config.get("mask", {}).get("threshold", {}).get("enabled", False)
                else "Detector Mask"
            ),
        }.get(str(plugin), str(plugin).replace("_", " ").title())
        stages.append({"name": stage_name, "image": current.copy(), "mask": current_mask})
    return stages
