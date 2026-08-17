"""Simulation orchestration depending only on the application port。"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from src.gimap.features.trainset.application.ports import SimulationPort

from .geometry import q_vectors


def _simulate_pattern_once(
    config: Dict[str, Any],
    sampled: Dict[str, float],
    simulator: SimulationPort | None = None,
) -> np.ndarray:
    """兼容旧 helper；调用者必须从 composition root 注入 simulator port。"""
    if simulator is None:
        raise ValueError("Trainset simulation requires an injected SimulationPort")
    return simulator.simulate(config, sampled)


def apply_interference(
    config: Dict[str, Any],
    sampled: Dict[str, Any],
    image: np.ndarray,
) -> np.ndarray:
    """Apply the selected structure factor to a form-factor intensity image."""
    image = np.asarray(image, dtype=np.float32).copy()
    interference = config.get("sample", {}).get("interference", {})
    if interference.get("enabled", False) and interference.get("plugin") == "paracrystal":
        all_q = q_vectors(config)["qy"]
        roi_cfg = config["roi"]
        qy = all_q[
            int(roi_cfg["y"]) : int(roi_cfg["y"]) + int(roi_cfg["height"]),
            int(roi_cfg["x"]) : int(roi_cfg["x"]) + int(roi_cfg["width"]),
        ]
        parameters = interference.get("parameters", {})
        spacing_default = parameters.get("D_nm", {}).get("minimum", 20.0)
        sigma_default = parameters.get("sigma_D_ratio", {}).get("minimum", 0.1)
        spacing = max(
            float(sampled.get("D_nm", sampled.get("spacing_nm", spacing_default))),
            1e-6,
        )
        sigma = float(sampled.get("sigma_D_ratio", sigma_default)) * spacing
        phi_q = np.exp(-np.pi * qy**2 * sigma**2)
        structure_factor = np.abs(
            (1.0 - phi_q**2)
            / np.maximum(
                1.0 + phi_q**2 - 2.0 * phi_q * np.cos(qy * spacing),
                1e-8,
            )
        )
        image *= structure_factor.astype(np.float32)
    return np.asarray(image, dtype=np.float32)


def simulate_pattern(
    config: Dict[str, Any],
    sampled: Dict[str, Any],
    simulator: SimulationPort | None = None,
) -> np.ndarray:
    if simulator is None:
        raise ValueError("Trainset simulation requires an injected SimulationPort")
    components = sampled.get("__mixture_components")
    weights = sampled.get("__mixture_weights")
    if isinstance(components, list) and components:
        if not isinstance(weights, list) or len(weights) != len(components):
            weights = [1.0 / len(components)] * len(components)
        image = np.zeros(
            (int(config["roi"]["height"]), int(config["roi"]["width"])),
            dtype=np.float32,
        )
        for weight, component in zip(weights, components):
            image += float(weight) * simulator.simulate(config, component)
    else:
        image = simulator.simulate(config, sampled)
    return apply_interference(config, sampled, image)


__all__ = [
    "_simulate_pattern_once",
    "apply_interference",
    "simulate_pattern",
]
