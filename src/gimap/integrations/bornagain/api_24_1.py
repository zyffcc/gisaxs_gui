"""现有 trainset scientific semantics 的 BornAgain 24.1 API binding。"""

from __future__ import annotations

from typing import Any

import numpy as np


MATERIALS = {
    "Vacuum": (0.0, 0.0),
    "Gold": (1.7065e-5, 2.0640e-6),
    "Silicon": (2.8402e-6, 2.5265e-8),
    "Copper": (1.2081e-5, 1.0584e-6),
    "Polymer": (1.9503e-6, 1.8413e-9),
    "PEO": (1.9503e-6, 1.8413e-9),
}

MATERIAL_COLORS = {
    "Gold": (0.90, 0.67, 0.12),
    "Silicon": (0.35, 0.55, 0.78),
    "Copper": (0.79, 0.32, 0.17),
    "Polymer": (0.45, 0.72, 0.63),
    "PEO": (0.45, 0.72, 0.63),
}


def _material(ba: Any, name: str) -> Any:
    if name == "Vacuum" and hasattr(ba, "Vacuum"):
        return ba.Vacuum()
    delta, beta = MATERIALS.get(name, MATERIALS["Silicon"])
    try:
        return ba.RefractiveMaterial(
            name,
            MATERIAL_COLORS.get(name, (0.5, 0.6, 0.7)),
            delta,
            beta,
        )
    except TypeError:
        return ba.RefractiveMaterial(name, delta, beta)


def _build_particle_form_factor(ba: Any, plugin: str, sampled: dict[str, float]) -> Any:
    radius = float(sampled.get("radius_nm", 1.0))
    height = float(sampled.get("height_nm", 1.0))
    if plugin == "sphere":
        return ba.Sphere(radius * ba.nm)
    if plugin == "cylinder":
        return ba.Cylinder(radius * ba.nm, height * ba.nm)
    if plugin == "box":
        return ba.Box(
            float(sampled.get("length_x_nm", 2.0 * radius)) * ba.nm,
            float(sampled.get("length_y_nm", 2.0 * radius)) * ba.nm,
            float(sampled.get("length_z_nm", height)) * ba.nm,
        )
    return ba.SphericalSegment(
        radius * ba.nm,
        0.0 * ba.nm,
        max(1e-6, 2.0 * radius - height) * ba.nm,
    )


def _layer_roughness(ba: Any, sigma_nm: float) -> Any:
    if sigma_nm <= 0:
        return None
    autocorrelation = ba.SelfAffineFractalModel(sigma_nm * ba.nm, 0.3, 5.0 * ba.nm)
    return ba.Roughness(
        autocorrelation,
        ba.TanhTransient(),
        ba.CommonDepthCrosscorrelation(10.0 * ba.nm),
    )


def _layer(
    ba: Any,
    material: Any,
    thickness_nm: float | None = None,
    roughness_nm: float = 0.0,
) -> Any:
    roughness = _layer_roughness(ba, roughness_nm)
    if thickness_nm is None:
        return ba.Layer(material, roughness) if roughness is not None else ba.Layer(material)
    if roughness is not None:
        return ba.Layer(material, thickness_nm * ba.nm, roughness)
    return ba.Layer(material, thickness_nm * ba.nm)


def build_sample(ba: Any, config: dict[str, Any], sampled: dict[str, float]) -> Any:
    sample_cfg = config["sample"]
    particle_cfg = next(
        (particle for particle in sample_cfg.get("particles", []) if particle.get("enabled", True)),
        None,
    )
    if particle_cfg is None:
        raise ValueError("At least one particle plugin must be enabled.")
    form_factor = _build_particle_form_factor(
        ba,
        str(particle_cfg.get("plugin", "spherical_segment")),
        sampled,
    )
    particle = ba.Particle(
        _material(ba, str(particle_cfg.get("material", "Copper"))),
        form_factor,
    )
    radius = float(sampled.get("radius_nm", sampled.get("length_x_nm", 1.0) / 2.0))
    footprint = np.pi * max(radius, 1e-6) ** 2
    configured_density = float(
        sampled.get(
            "surface_density_per_nm2",
            sample_cfg.get("surface_density_per_nm2", 0.01),
        )
    )
    effective_density = min(configured_density, 0.35 / max(footprint, 1e-12))

    sample = ba.Sample()
    ambient = ba.Layer(_material(ba, "Vacuum"))
    if hasattr(ba, "ParticleLayout"):
        layout = ba.ParticleLayout()
        layout.addParticle(particle, 1.0)
        layout.setTotalParticleSurfaceDensity(effective_density)
        ambient.addLayout(layout)
    else:
        ambient.deposit2D(ba.Dilute2D(effective_density, particle))
    sample.addLayer(ambient)
    for index, layer in enumerate(sample_cfg.get("layers", [])):
        if not layer.get("enabled", True):
            continue
        thickness_spec = layer.get("thickness_nm", 0.0)
        roughness_spec = layer.get("roughness_nm", 0.0)
        thickness_default = (
            float(thickness_spec.get("minimum", 0.0))
            if isinstance(thickness_spec, dict)
            else float(thickness_spec)
        )
        roughness_default = (
            float(roughness_spec.get("minimum", 0.0))
            if isinstance(roughness_spec, dict)
            else float(roughness_spec)
        )
        thickness = float(sampled.get(f"layer_{index}_thickness_nm", thickness_default))
        if thickness > 0:
            roughness = float(
                sampled.get(f"layer_{index}_roughness_nm", roughness_default)
            )
            sample.addLayer(
                _layer(
                    ba,
                    _material(ba, str(layer.get("material", "Silicon"))),
                    thickness,
                    roughness,
                )
            )
    substrate = sample_cfg.get("substrate", {})
    substrate_roughness = float(
        sampled.get("roughness_nm", substrate.get("roughness_nm", 0.0))
    )
    sample.addLayer(
        _layer(
            ba,
            _material(ba, str(substrate.get("material", "Silicon"))),
            None,
            substrate_roughness,
        )
    )
    return sample


def simulate_raw_24_1(ba: Any, config: dict[str, Any], sampled: dict[str, float]) -> np.ndarray:
    from trainset.geometry import roi_to_spherical_ranges

    ranges = roi_to_spherical_ranges(config)
    roi = config["roi"]
    sample = build_sample(ba, config, sampled)
    beam = ba.Beam(
        1e12,
        float(config["beam"]["wavelength_nm"]) * ba.nm,
        float(config["beam"]["grazing_angle_deg"]) * ba.deg,
    )
    detector = ba.SphericalDetector(
        int(roi["width"]),
        ranges["phi_min_deg"] * ba.deg,
        ranges["phi_max_deg"] * ba.deg,
        int(roi["height"]),
        ranges["alpha_min_deg"] * ba.deg,
        ranges["alpha_max_deg"] * ba.deg,
    )
    simulation_cfg = config.get("simulation", {})
    detector.setResolutionFunction(
        ba.ResolutionFunction2DGaussian(
            float(simulation_cfg.get("resolution_sigma_phi_deg", 0.01)) * ba.deg,
            float(simulation_cfg.get("resolution_sigma_alpha_deg", 0.01)) * ba.deg,
        )
    )
    simulation = ba.ScatteringSimulation(beam, sample, detector)
    simulation.options().setUseAvgMaterials(True)
    result = simulation.simulate()
    if hasattr(result, "intensities"):
        image = np.flipud(np.asarray(result.intensities()))
    else:
        from bornagain.numpyutil import Arrayf64Converter as converter

        image = np.flipud(converter.asNpArray(result.dataArray()))
    return np.asarray(image, dtype=np.float32)
