"""Calculate the hexagonal paracrystal structure factor with BornAgain, outside the Qt process."""

import math
import sys

import bornagain as ba
import numpy as np


def simulate(distance, spread, *, paracrystal):
    wavelength, alpha, count = 0.1, 0.2, 60
    particle = ba.Particle(ba.RefractiveMaterial("p", (0.5, 0.5, 0.5), 1e-5, 1e-7), ba.Sphere(0.5 * ba.nm))
    if paracrystal:
        layout = ba.Paracrystal2D(particle, ba.HexagonalLattice2D(distance * ba.nm, 0.0), 0.0, 0.0, 0.0)
        layout.setIntegrationOverXi(True)
        pdf = ba.Profile2DGauss(spread * distance * ba.nm, spread * distance * ba.nm, 0.0)
        layout.setProbabilityDistributions(pdf, pdf)
    else:
        layout = ba.Dilute2D(0.001, particle)
    top = ba.Layer(ba.Vacuum())
    top.deposit2D(layout)
    sample = ba.Sample()
    sample.addLayer(top)
    sample.addLayer(ba.Layer(ba.Vacuum()))
    k = 2 * math.pi / wavelength
    exit_angle = math.radians(alpha)
    phi_max = math.asin(1.5 / (k * math.cos(exit_angle)))
    detector = ba.SphericalDetector(count, 1e-4, phi_max, 1, exit_angle - 1e-7, exit_angle + 1e-7)
    result = ba.ScatteringSimulation(ba.Beam(1.0, wavelength * ba.nm, alpha * ba.deg), sample, detector).simulate()
    values = np.asarray(result.npArray() if hasattr(result, "npArray") else result.intensities(), float).reshape(-1)
    phi = 1e-4 + (np.arange(count) + 0.5) * (phi_max - 1e-4) / count
    q = np.hypot(k * math.cos(exit_angle) * np.sin(phi), k * math.cos(exit_angle) * (np.cos(phi) - 1.0))
    return q, values


if __name__ == "__main__":
    distance, spread = map(float, sys.argv[1:])
    q, structured = simulate(distance, spread, paracrystal=True)
    _, dilute = simulate(distance, spread, paracrystal=False)
    density = 2.0 / (math.sqrt(3.0) * distance**2)
    structure_factor = structured / dilute * 0.001 / density
    np.savetxt(sys.stdout, np.column_stack((q, structure_factor)))
