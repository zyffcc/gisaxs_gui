from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .geometry import q_vectors, roi_to_spherical_ranges


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
        return ba.RefractiveMaterial(name, MATERIAL_COLORS.get(name, (0.5, 0.6, 0.7)), delta, beta)
    except TypeError:
        return ba.RefractiveMaterial(name, delta, beta)


def _build_particle_form_factor(ba: Any, plugin: str, radius: float, height: float) -> Any:
    nm = ba.nm
    if plugin == "sphere":
        return ba.Sphere(radius * nm)
    if plugin == "cylinder":
        return ba.Cylinder(radius * nm, height * nm)
    if plugin == "box":
        return ba.Box(2.0 * radius * nm, 2.0 * radius * nm, height * nm)
    # Preserve the Yuxin server pipeline convention: ``height_nm`` is the
    # truncation amount, so BornAgain receives the remaining segment height.
    return ba.SphericalSegment(radius * nm, 0.0 * nm, max(1e-6, 2.0 * radius - height) * nm)


def _layer_roughness(ba: Any, sigma_nm: float) -> Any:
    if sigma_nm <= 0:
        return None
    autocorrelation = ba.SelfAffineFractalModel(sigma_nm * ba.nm, 0.3, 5.0 * ba.nm)
    return ba.Roughness(autocorrelation, ba.TanhTransient(), ba.CommonDepthCrosscorrelation(10.0 * ba.nm))


def _layer(ba: Any, material: Any, thickness_nm: float | None = None, roughness_nm: float = 0.0) -> Any:
    roughness = _layer_roughness(ba, roughness_nm)
    if thickness_nm is None:
        return ba.Layer(material, roughness) if roughness is not None else ba.Layer(material)
    return ba.Layer(material, thickness_nm * ba.nm, roughness) if roughness is not None else ba.Layer(material, thickness_nm * ba.nm)


def build_sample(ba: Any, config: Dict[str, Any], sampled: Dict[str, float]) -> Any:
    sample_cfg = config["sample"]
    particle_cfg = next((p for p in sample_cfg.get("particles", []) if p.get("enabled", True)), None)
    if particle_cfg is None:
        raise ValueError("At least one particle plugin must be enabled.")
    ff = _build_particle_form_factor(
        ba,
        str(particle_cfg.get("plugin", "spherical_segment")),
        sampled["radius_nm"],
        sampled["height_nm"],
    )
    particle = ba.Particle(_material(ba, str(particle_cfg.get("material", "Copper"))), ff)
    footprint = np.pi * max(sampled["radius_nm"], 1e-6) ** 2
    configured_density = float(sampled.get("surface_density_per_nm2", sample_cfg.get("surface_density_per_nm2", 0.01)))
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
    for layer in sample_cfg.get("layers", []):
        if not layer.get("enabled", True):
            continue
        thickness = float(layer.get("thickness_nm", 0.0))
        if thickness > 0:
            roughness = float(sampled.get("roughness_nm", layer.get("roughness_nm", 0.0)))
            sample.addLayer(_layer(ba, _material(ba, str(layer.get("material", "Silicon"))), thickness, roughness))
    substrate = sample_cfg.get("substrate", {})
    substrate_roughness = float(sampled.get("roughness_nm", substrate.get("roughness_nm", 0.0)))
    sample.addLayer(_layer(ba, _material(ba, str(substrate.get("material", "Silicon"))), None, substrate_roughness))
    return sample


def simulate_pattern(config: Dict[str, Any], sampled: Dict[str, float]) -> np.ndarray:
    import bornagain as ba  # type: ignore

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
        try:
            from bornagain.numpyutil import Arrayf64Converter as dac  # type: ignore

            image = np.flipud(dac.asNpArray(result.dataArray()))
        except Exception:
            image = np.asarray(result.array())
    image = np.asarray(image, dtype=np.float32)

    interference = config.get("sample", {}).get("interference", {})
    if interference.get("enabled", False) and interference.get("plugin") == "paracrystal":
        all_q = q_vectors(config)["qy"]
        roi_cfg = config["roi"]
        qy = all_q[
            int(roi_cfg["y"]) : int(roi_cfg["y"]) + int(roi_cfg["height"]),
            int(roi_cfg["x"]) : int(roi_cfg["x"]) + int(roi_cfg["width"]),
        ]
        spacing = max(sampled.get("spacing_nm", 20.0), 1e-6)
        sigma = float(interference.get("sigma_ratio", 0.1)) * spacing
        phi_q = np.exp(-np.pi * qy**2 * sigma**2)
        structure_factor = np.abs((1.0 - phi_q**2) / np.maximum(1.0 + phi_q**2 - 2.0 * phi_q * np.cos(qy * spacing), 1e-8))
        image *= structure_factor.astype(np.float32)
    return image
