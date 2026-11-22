"""Single-interface film optics for a GIWAXS ring of a known crystal phase."""
from __future__ import annotations

from dataclasses import dataclass
import math
import re

import numpy as np
from scipy.interpolate import PchipInterpolator

from src.gimap.shared.geometry.q_inverse import exit_sine_of_qz
from src.gimap.shared.ring_profile import fold_ring
from src.gimap.shared.xray_scattering import (
    ENERGY_RANGE_KEV, HC_KEV_NM, R_E_NM, TABULATED_ELEMENTS, scattering_factors,
)

from ..intensity import film_absorption_factor
from .scattering import ATOMIC_NUMBER
from .symmetry import expand_sites


def _formula_counts(formula, units):
    """Flat CIF sum formula, e.g. 'C H6 I3 N Pb'. Require the entire formula and a positive Z."""
    if not formula or units is None or not math.isfinite(units) or units <= 0.:
        return None
    counts = {}
    remaining = formula.strip()
    while remaining:
        match = re.match(r"([A-Z][a-z]?)\s*(\d+(?:\.\d+)?|\.\d+)?\s*", remaining)
        if not match or match[1] not in ATOMIC_NUMBER:
            return None
        amount = float(match[2] or 1.)
        if amount <= 0. or not math.isfinite(amount*units):
            return None
        counts[match[1]] = counts.get(match[1], 0.) + units*amount
        remaining = remaining[match.end():]
    return counts or None


def film_optics(phase, wavelength_angstrom):
    """Index n = 1 − δ + iβ from CIF sum formula × Z, or occupied sites, at the frame wavelength."""
    if not np.isfinite(wavelength_angstrom) or wavelength_angstrom <= 0.:
        raise ValueError("The film refractive index requires a positive wavelength.")
    energy = 10 * HC_KEV_NM / wavelength_angstrom
    site_counts = {}
    positions = expand_sites([atom.xyz for atom in phase.atoms], phase.operations) if phase.atoms else []
    for atom, sites in zip(phase.atoms, positions):
        site_counts[atom.element] = site_counts.get(atom.element, 0.) + atom.occupancy * len(sites)
    bulk = _formula_counts(phase.formula_sum, phase.formula_units_z)
    populations = bulk or site_counts
    composition = "CIF sum formula × Z" if bulk else "symmetry-expanded CIF atom sites"
    record = {"available": False, "source": composition, "composition_source": composition, "atoms_per_cell": populations,
              "formula_sum": phase.formula_sum, "formula_units_z": phase.formula_units_z, "site_atoms_per_cell": site_counts,
              "wavelength_angstrom": wavelength_angstrom, "energy_kev": energy, "cell_volume_a3": phase.cell.volume,
              "electron_density_e_per_a3": None, "delta": None, "beta": None}
    if not bulk and (phase.formula_sum or phase.formula_units_z is not None):
        record["composition_note"] = "A flat elemental sum formula and positive formula units Z are required to use the CIF bulk formula."
    if not populations or any(n < 0. for n in populations.values()) or sum(populations.values()) <= 0.:
        record["reason"] = "No usable atom populations in the phase."
        return record
    volume = phase.cell.volume
    rho_e = sum(n * ATOMIC_NUMBER[element] for element, n in populations.items()) / volume
    factor = 10 * R_E_NM * wavelength_angstrom**2 / (2 * math.pi)
    record.update(electron_density_e_per_a3=rho_e, delta=factor*rho_e)
    missing = sorted(element for element, n in populations.items() if n and element not in TABULATED_ELEMENTS)
    if missing or not ENERGY_RANGE_KEV[0] <= energy <= ENERGY_RANGE_KEV[1]:
        record.update(source=f"neutral-atom Z; {composition}", missing_factors=missing,
                      reason="Absorption factors unavailable for the composition or wavelength.")
        return record
    density = sum(n * complex(*scattering_factors(element, energy)) for element, n in populations.items() if n) / volume
    delta, beta = factor*density.real, factor*density.imag
    record.update(available=True, source=f"Henke f1 and f2; {composition}",
                  delta=delta, beta=beta, critical_angle_deg=math.degrees(math.asin(math.sqrt(2*delta))))
    return record


def exit_transmission(sin_exit, delta, beta):
    """Reciprocal exit-wave field |2s / (s + √(s² − 2δ + 2iβ))|², s = sin αf in vacuum."""
    sine = np.asarray(sin_exit, dtype=float)
    transmitting = np.isfinite(sine) & (sine > 0.) & (sine <= 1.)
    sine = np.where(transmitting, sine, 0.)
    field = np.sqrt(sine**2 - 2*delta + 2j*beta)
    amplitude = np.divide(2*sine, sine + field, where=transmitting, out=np.zeros_like(field))
    return np.abs(amplitude)**2


def ring_optical_factors(optics, geometry, q, chi_pixels, qz_pixels, signed_profile, *, fit_absorption=False):
    """Fold pixel averages of T_f (and, optionally, T_f F_A) with the intensity weights.

    Model-only bins use the ray at the ring centre. The fitted attenuation curve
    divides the folded product by the folded field already present in the texture library.
    """
    sine = exit_sine_of_qz(geometry, qz_pixels)
    centres = np.arange(90) + 0.5
    central_sine = exit_sine_of_qz(geometry, q * np.cos(np.radians(centres)))
    field = exit_transmission(sine, optics["delta"], optics["beta"])
    central_field = exit_transmission(central_sine, optics["delta"], optics["beta"])
    index = np.clip(np.floor(chi_pixels.astype(float) + 90).astype(int), 0, 179)
    counts = np.bincount(index, minlength=180)
    signed_bins = np.floor(signed_profile.x + 90).astype(int)

    def fold(pixel_factor, central_factor):
        totals = np.bincount(index, weights=pixel_factor, minlength=180)
        signed = totals[signed_bins] / counts[signed_bins]
        _, values, _, measured, _, _ = fold_ring(signed_profile.x, signed, signed_profile.sigma, 0.)
        return np.where(measured, values, central_factor)

    transmission = fold(field, central_field)
    if not fit_absorption:
        return transmission, None
    sin_in = math.sin(geometry.incidence_rad)
    if sin_in <= 0.:
        raise ValueError("Film absorption requires a positive incidence angle.")
    # u = 1 − exp(−μd / sin αi) maps both the thin and semi-infinite limits to finite endpoints.
    # Extra knots near zero resolve the short absorption scale of the lowest exit rays.
    knots = np.r_[0., np.geomspace(1e-6, .01, 20, endpoint=False), np.linspace(.01, 1., 100)]
    relative = []
    for u in knots:
        tau = -sin_in * np.log1p(-u) if u < 1. else np.inf
        product = fold(field * np.nan_to_num(film_absorption_factor(sine, sin_in, tau)),
                       central_field * np.nan_to_num(film_absorption_factor(central_sine, sin_in, tau)))
        relative.append(np.divide(product, transmission, out=np.ones_like(product), where=transmission > 0.))
    return transmission, FilmAbsorption(sin_in, PchipInterpolator(knots, relative, axis=0))


@dataclass(frozen=True)
class FilmAbsorption:
    """Relative angular attenuation multiplying the T_f-weighted polar-bin intensities."""

    sin_incidence: float
    curve: PchipInterpolator

    def record(self, u, error_u):
        thin = u < 1e-6
        thick = u > 1. - 1e-6
        return {"parameter": "optical depth d / attenuation length, shared across rings of the phase",
                "formula": "T_f times F_A, with F_A = A(af)/A(ai), A = (1-exp(-mu*d*s))/s, s = csc(ai)+csc(af)",
                "optical_depth": None if thick else float(-self.sin_incidence * np.log1p(-u)),
                "optical_depth_error": (None if error_u is None or thin or thick else
                                        float(error_u * self.sin_incidence / (1.-u))),
                "thin_limit": bool(thin), "thick_limit": bool(thick),
                "incident_path_absorbed_fraction": float(u)}


def ring_transmission(optics, geometry, q, chi_pixels, qz_pixels, signed_profile):
    return ring_optical_factors(optics, geometry, q, chi_pixels, qz_pixels, signed_profile)[0]
