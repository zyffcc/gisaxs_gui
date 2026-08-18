"""Persistent simulation-grid cache adapter."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np

from ...application.simulation import _simulate_pattern_once, apply_interference
from ...application.ports import SimulationPort


ProgressCallback = Optional[Callable[[int, int, str], None]]
PauseCallback = Optional[Callable[[], None]]


def _cache_config(config: Dict[str, Any]) -> Dict[str, Any]:
    return config.get("simulation", {}).get("grid_cache", {})


def cache_directory(config: Dict[str, Any]) -> Path:
    value = str(_cache_config(config).get("directory", "_bornagain_cache")).strip()
    return Path(value or "_bornagain_cache").expanduser().resolve()


def particle_axes(config: Dict[str, Any]) -> Dict[str, np.ndarray]:
    particle = next(iter(config.get("sample", {}).get("particles", [])), {})
    axes: Dict[str, np.ndarray] = {}
    for name, spec in particle.get("parameters", {}).items():
        low = float(spec.get("minimum", 0.0))
        high = float(spec.get("maximum", low))
        points = max(1, int(spec.get("grid_points", 30)))
        if points == 1 or high <= low:
            axes[str(name)] = np.asarray([(low + high) * 0.5], dtype=np.float32)
        else:
            edges = np.linspace(low, high, points + 1, dtype=np.float32)
            axes[str(name)] = (edges[:-1] + edges[1:]) * 0.5
    return axes


def grid_node_count(config: Dict[str, Any]) -> int:
    axes = particle_axes(config)
    return int(np.prod([len(values) for values in axes.values()], dtype=np.int64)) if axes else 0


def cache_compatible(config: Dict[str, Any]) -> bool:
    """Return whether every BornAgain-affecting non-shape parameter is fixed."""
    for spec in config.get("parameters", {}).values():
        if spec.get("source") in {"form_factor", "structure_factor"}:
            continue
        if float(spec.get("maximum", 0.0)) > float(spec.get("minimum", 0.0)):
            return False
    return True


def _payload(config: Dict[str, Any], axes: Dict[str, np.ndarray]) -> Dict[str, Any]:
    sample = config.get("sample", {})
    return {
        "format": 1,
        "beam": config.get("beam", {}),
        "detector": config.get("detector", {}),
        "roi": config.get("roi", {}),
        "resolution": {
            key: value
            for key, value in config.get("simulation", {}).items()
            if key != "grid_cache"
        },
        "particle": next(iter(sample.get("particles", [])), {}),
        "layers": sample.get("layers", []),
        "substrate": sample.get("substrate", {}),
        "surface_density_per_nm2": sample.get("surface_density_per_nm2", 0.01),
        "axes": {name: values.tolist() for name, values in axes.items()},
        "segment_height_le_2r": sample.get("constraints", {}).get("segment_height_le_2r", False),
    }


def cache_path(config: Dict[str, Any]) -> Path:
    axes = particle_axes(config)
    digest = hashlib.sha256(
        json.dumps(_payload(config, axes), sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:20]
    return cache_directory(config) / f"form_factor_grid_{digest}.npz"


def prune_cache(directory: Path, max_files: int, keep: Optional[Path] = None) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    files = sorted(
        directory.glob("form_factor_grid_*.npz"),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    protected = keep.resolve() if keep is not None and keep.exists() else None
    limit = max(1, int(max_files))
    retained = []
    if protected is not None:
        retained.append(protected)
    for path in files:
        resolved = path.resolve()
        if resolved == protected:
            continue
        if len(retained) < limit:
            retained.append(resolved)
        else:
            path.unlink(missing_ok=True)


def _is_valid_node(config: Dict[str, Any], sampled: Dict[str, float]) -> bool:
    constraints = config.get("sample", {}).get("constraints", {})
    return not (
        constraints.get("segment_height_le_2r", False)
        and "radius_nm" in sampled
        and "height_nm" in sampled
        and sampled["height_nm"] > 2.0 * sampled["radius_nm"]
    )


def load_or_build_grid(
    config: Dict[str, Any],
    simulation_port: SimulationPort | None = None,
    progress: ProgressCallback = None,
    pause: PauseCallback = None,
    force: bool = False,
) -> Dict[str, Any]:
    axes = particle_axes(config)
    if not axes:
        raise ValueError("The selected particle shape has no grid-cache parameters.")
    path = cache_path(config)
    max_files = max(1, int(_cache_config(config).get("max_files", 5)))
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        archive = np.load(path, allow_pickle=False)
        path.touch()
        prune_cache(path.parent, max_files, keep=path)
        return {
            "path": path,
            "axes": {name: np.asarray(archive[f"axis_{name}"], dtype=np.float32) for name in axes},
            "images": np.asarray(archive["images"], dtype=np.float32),
            "valid": np.asarray(archive["valid"], dtype=bool),
            "cache_hit": True,
        }

    names = list(axes)
    shape = tuple(len(axes[name]) for name in names)
    roi = config["roi"]
    images = np.zeros((*shape, int(roi["height"]), int(roi["width"])), dtype=np.float16)
    valid = np.zeros(shape, dtype=np.uint8)
    total = int(np.prod(shape))
    fixed = {
        name: float(spec.get("minimum", 0.0))
        for name, spec in config.get("parameters", {}).items()
        if name not in axes
    }
    for completed, index in enumerate(np.ndindex(shape), start=1):
        if pause is not None:
            pause()
        sampled = dict(fixed)
        sampled.update({name: float(axes[name][index[i]]) for i, name in enumerate(names)})
        if _is_valid_node(config, sampled):
            if simulation_port is None:
                images[index] = np.asarray(
                    _simulate_pattern_once(config, sampled), dtype=np.float16
                )
            else:
                images[index] = np.asarray(
                    _simulate_pattern_once(config, sampled, simulation_port),
                    dtype=np.float16,
                )
            valid[index] = 1
        if progress is not None:
            progress(completed, total, f"BornAgain form-factor grid {completed}/{total}")

    metadata = json.dumps(_payload(config, axes), sort_keys=True, default=str)
    np.savez_compressed(
        path,
        images=images,
        valid=valid,
        metadata=np.asarray(metadata),
        **{f"axis_{name}": values for name, values in axes.items()},
    )
    prune_cache(path.parent, max_files, keep=path)
    return {
        "path": path,
        "axes": axes,
        "images": np.asarray(images, dtype=np.float32),
        "valid": valid.astype(bool),
        "cache_hit": False,
    }


def mix_cached_grid(
    config: Dict[str, Any],
    sampled: Dict[str, Any],
    grid: Dict[str, Any],
    rng: np.random.Generator,
) -> np.ndarray:
    axes: Dict[str, np.ndarray] = grid["axes"]
    names = list(axes)
    mesh = np.meshgrid(*(axes[name] for name in names), indexing="ij")
    mixture_cfg = config.get("sample", {}).get("mixture", {})
    components = sampled.get("__mixture_components")
    component_weights = sampled.get("__mixture_weights")
    if not isinstance(components, list) or not components:
        components = [sampled]
    if not isinstance(component_weights, list) or len(component_weights) != len(components):
        component_weights = [1.0 / len(components)] * len(components)
    sigma_min = max(1e-6, float(mixture_cfg.get("sigma_fraction_min", 0.01)))
    sigma_max = max(sigma_min, float(mixture_cfg.get("sigma_fraction_max", 0.30)))
    weights = np.zeros(tuple(len(axes[name]) for name in names), dtype=np.float64)
    for contribution, component in zip(component_weights, components):
        exponent = np.zeros(weights.shape, dtype=np.float64)
        for axis_index, name in enumerate(names):
            values = axes[name]
            span = max(float(values[-1] - values[0]), 1e-6)
            sigma = float(rng.uniform(sigma_min, sigma_max)) * span
            exponent += ((mesh[axis_index] - float(component.get(name, values.mean()))) / sigma) ** 2
        weights += float(contribution) * np.exp(-0.5 * exponent)
    weights *= np.asarray(grid["valid"], dtype=np.float64)
    total = float(weights.sum())
    if total <= 1e-20:
        weights = np.asarray(grid["valid"], dtype=np.float64)
        total = float(weights.sum())
    weights /= max(total, 1e-20)
    raw = np.tensordot(
        weights,
        np.asarray(grid["images"], dtype=np.float32),
        axes=(tuple(range(weights.ndim)), tuple(range(weights.ndim))),
    )
    return apply_interference(config, sampled, np.asarray(raw, dtype=np.float32))
