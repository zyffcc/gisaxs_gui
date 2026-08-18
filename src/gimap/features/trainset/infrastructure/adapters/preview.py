"""Existing numerical Trainset preview pipeline behind an application port."""

from __future__ import annotations

import copy
import hashlib
import json
import time
from typing import Any

import numpy as np

from ...application.models import TrainsetPreviewRequest, TrainsetWhatIfRequest
from ...application.ports import SimulationPort
from ...application.simulation import simulate_pattern
from ...domain.geometry import roi_to_spherical_ranges
from .dataset_generator import (
    DatasetGenerator,
    apply_preprocessing,
    build_fixed_mask,
    build_random_mask,
    merge_threshold_mask,
)


class TrainsetPreviewAdapter:
    """Own simulation caching and preprocessing outside the Qt presentation."""

    def __init__(self, simulation_port: SimulationPort, *, cache_size: int = 24):
        self.simulation_port = simulation_port
        self.cache_size = max(1, int(cache_size))
        self._simulation_cache: dict[str, np.ndarray] = {}

    def _cached_simulation(
        self, config: dict[str, Any], sampled: dict[str, Any]
    ) -> tuple[np.ndarray, bool]:
        payload = {
            "beam": config.get("beam"),
            "detector": config.get("detector"),
            "roi": config.get("roi"),
            "simulation": {
                key: value
                for key, value in config.get("simulation", {}).items()
                if key != "grid_cache"
            },
            "sample": config.get("sample"),
            "sampled": sampled,
        }
        key = hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        if key in self._simulation_cache:
            return self._simulation_cache[key].copy(), True
        image = simulate_pattern(config, sampled, simulator=self.simulation_port)
        if len(self._simulation_cache) >= self.cache_size:
            self._simulation_cache.pop(next(iter(self._simulation_cache)))
        self._simulation_cache[key] = np.asarray(image, dtype=np.float32).copy()
        return np.asarray(image, dtype=np.float32), False

    def generate_preview(
        self, request: TrainsetPreviewRequest, *, on_progress=None
    ) -> dict[str, Any]:
        progress = on_progress or (lambda _value, _message: None)
        config = request.config
        if request.force:
            self._simulation_cache.clear()
        started = time.perf_counter()
        generator = DatasetGenerator(config, simulation_port=self.simulation_port)
        midpoint = 0.5 * (request.minimum + request.maximum)
        comparison_values = (
            ("minimum", request.minimum),
            ("midpoint", midpoint),
            ("maximum", request.maximum),
        )
        base_sample = {
            name: 0.5
            * (float(spec.get("minimum", 0.0)) + float(spec.get("maximum", 0.0)))
            for name, spec in config.get("parameters", {}).items()
        }
        simulation_seed = int(config.get("project", {}).get("seed", 42))
        realization_seed = simulation_seed + 1009 * request.realization
        comparison_images: dict[str, np.ndarray] = {}
        comparison_labels: dict[str, str] = {}
        comparison_details: dict[str, Any] = {}
        midpoint_stages = []
        cache_hits = 0
        cache_misses = 0
        ranges = roi_to_spherical_ranges(config)
        for index, (position, value) in enumerate(comparison_values):
            progress(
                10 + index * 23,
                f"BornAgain simulation {index + 1}/3: {position} {request.key} = {value:.5g}",
            )
            sampled = dict(base_sample)
            overrides: dict[str, float] = {}
            if request.plugin == "physics":
                sampled[request.key] = value
            else:
                overrides[f"{request.plugin}.{request.key}"] = value
            mixture_generator = DatasetGenerator(
                config, simulation_port=self.simulation_port
            )
            mixture_generator.rng = np.random.default_rng(simulation_seed)
            simulation_values = mixture_generator._mixture_values(sampled)
            raw, cache_hit = self._cached_simulation(config, simulation_values)
            cache_hits += int(cache_hit)
            cache_misses += int(not cache_hit)
            progress(
                20 + index * 23,
                f"Applying enabled preprocessing to {position} image…",
            )
            realization_rng = np.random.default_rng(realization_seed + 17)
            if config.get("mask", {}).get("mode") == "random":
                mask = build_random_mask(raw.shape, config, realization_rng)
                mask = merge_threshold_mask(raw, mask, config)
            else:
                mask = build_fixed_mask(raw, config)
            preprocessing_trace: dict[str, Any] = {}
            stages = apply_preprocessing(
                raw,
                config,
                mask,
                realization_rng,
                overrides=overrides,
                trace=preprocessing_trace,
            )
            final_image = np.asarray(stages[-1]["image"], dtype=np.float32)
            comparison_images[position] = final_image
            comparison_labels[position] = f"{position.title()} · {value:.5g}"
            comparison_details[position] = {
                "comparison": {
                    "parameter": request.compared_text,
                    "value": float(value),
                },
                "editable physics": copy.deepcopy(sampled),
                "physics values": simulation_values,
                "preprocessing realization": preprocessing_trace or "none enabled",
                "beam": config.get("beam", {}),
                "detector": config.get("detector", {}),
                "roi": config.get("roi", {}),
                "angular range": {
                    "phi min deg": ranges["phi_min_deg"],
                    "phi max deg": ranges["phi_max_deg"],
                    "alpha top deg": ranges["alpha_top_deg"],
                    "alpha bottom deg": ranges["alpha_bottom_deg"],
                },
            }
            if position == "midpoint":
                midpoint_stages = stages
        progress(82, "Sampling label coverage and calculating diagnostics…")
        parameter_samples = generator.sample_parameters(request.preview_count)
        total_samples = int(config["dataset"]["number_of_samples"])
        final_image = comparison_images["midpoint"]
        bytes_per_sample = (
            final_image.nbytes
            + final_image.size
            + 4 * len(config.get("parameters", {}))
        )
        valid_values = final_image[np.isfinite(final_image)]
        histogram, edges = (
            np.histogram(valid_values, bins=64)
            if valid_values.size
            else (np.zeros(64, dtype=float), np.arange(65, dtype=float))
        )
        stats = {
            "source": "BornAgain simulation (experimental reference is geometry guidance only)",
            "orientation": "x right, y down, qz/exit angle higher at image top",
            "compared_parameter": request.compared_text,
            "range": f"{request.minimum:.6g} / {midpoint:.6g} / {request.maximum:.6g}",
            "tensor_shape": [1, int(final_image.shape[0]), int(final_image.shape[1]), 1],
            "enabled_pipeline": " → ".join(
                str(stage["name"]) for stage in midpoint_stages
            ),
            "bornagain_cache": f"{cache_hits} hit(s), {cache_misses} recomputed",
            "stochastic_realization": request.realization + 1,
            "estimated_dataset_gib": round(
                total_samples * bytes_per_sample / (1024**3), 3
            ),
            "preview_elapsed_s": round(time.perf_counter() - started, 3),
        }
        if request.warnings:
            stats["warning"] = " · ".join(request.warnings)
        progress(96, "Rendering the comparison in the GUI…")
        return {
            "comparison_images": comparison_images,
            "comparison_labels": comparison_labels,
            "comparison_details": comparison_details,
            "stages": midpoint_stages,
            "stats": stats,
            "spectrum_x": (edges[:-1] + edges[1:]) / 2.0,
            "spectrum_y": histogram,
            "parameter_samples": parameter_samples,
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
            "cache_size": len(self._simulation_cache),
        }

    def simulate_what_if(self, request: TrainsetWhatIfRequest) -> dict[str, Any]:
        config = request.config
        sampled = request.sampled
        constraints = config.get("sample", {}).get("constraints", {})
        if (
            constraints.get("segment_height_le_2r", False)
            and "height_nm" in sampled
            and "radius_nm" in sampled
            and sampled["height_nm"] > 2.0 * sampled["radius_nm"]
        ):
            raise ValueError(
                "Manual simulation violates h ≤ 2R. Adjust height/radius or disable the physical constraint."
            )
        if (
            constraints.get("interparticle_spacing_gt_2r", False)
            and "D_nm" in sampled
            and "radius_nm" in sampled
            and sampled["D_nm"] <= 2.0 * sampled["radius_nm"]
        ):
            raise ValueError(
                "Manual simulation violates D > 2R. Adjust spacing/radius or disable the physical constraint."
            )

        seed = int(config.get("project", {}).get("seed", 42))
        generator = DatasetGenerator(config, simulation_port=self.simulation_port)
        generator.rng = np.random.default_rng(seed)
        simulation_values = generator._mixture_values(sampled)
        raw, cache_hit = self._cached_simulation(config, simulation_values)
        realization_rng = np.random.default_rng(seed + 1009 * request.realization + 17)
        if config.get("mask", {}).get("mode") == "random":
            mask = build_random_mask(raw.shape, config, realization_rng)
            mask = merge_threshold_mask(raw, mask, config)
        else:
            mask = build_fixed_mask(raw, config)
        stages = apply_preprocessing(raw, config, mask, realization_rng, trace={})
        return {
            "image": np.asarray(stages[-1]["image"], dtype=np.float32),
            "cache_hit": cache_hit,
            "values": sampled,
            "pipeline": " → ".join(str(stage["name"]) for stage in stages),
        }
