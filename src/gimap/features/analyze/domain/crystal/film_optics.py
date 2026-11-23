"""Single-interface film optics for a GIWAXS ring of a known crystal phase."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.interpolate import PchipInterpolator

from src.gimap.shared.geometry.q_inverse import exit_sine_of_qz, qz_of_exit_sine
from src.gimap.shared.ring_profile import fold_ring
from src.gimap.shared.xray_scattering import (
    ENERGY_RANGE_KEV, HC_KEV_NM, R_E_NM, TABULATED_ELEMENTS, scattering_factors,
)

from ..intensity import film_absorption_factor
from ..refracted_film import REFRACTED_FILM_FORMULA, RefractedFilm, exit_transmission, internal_qz
from .scattering import ATOMIC_NUMBER
from .composition import phase_composition


def film_optics(phase, wavelength_angstrom):
    """Index n = 1 − δ + iβ from CIF sum formula × Z, or occupied sites, at the frame wavelength."""
    if not np.isfinite(wavelength_angstrom) or wavelength_angstrom <= 0.:
        raise ValueError("The film refractive index requires a positive wavelength.")
    energy = 10 * HC_KEV_NM / wavelength_angstrom
    counts = phase_composition(phase)
    bulk = counts["formula_atoms_per_cell"]
    populations = bulk or counts["site_atoms_per_cell"]
    composition = "CIF sum formula × Z" if bulk else "symmetry-expanded CIF atom sites"
    record = {"available": False, "source": composition, "composition_source": composition, "atoms_per_cell": populations,
              **counts,
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


def _central_exit(geometry, q, centres, optics):
    """External ray at an internal polar bin centre and the active map's ring radius.

    This supplies plotting values only for bins without pixels. Along the ring, the
    internal pole angle decreases monotonically with external exit angle.
    """
    def angle(sine):
        qz_map = qz_of_exit_sine(geometry, sine)
        parallel = np.sqrt(np.maximum(0., q**2 - qz_map**2))
        return np.degrees(np.arctan2(parallel, internal_qz(geometry, sine, optics["delta"], optics["beta"])))

    low = np.full_like(centres, math.sin(geometry.critical_rad))
    high = np.ones_like(low)
    accessible = (centres <= angle(low)) & (centres >= angle(high))
    for _ in range(40):
        mid = .5*(low + high)
        below = angle(mid) > centres
        low, high = np.where(below, mid, low), np.where(below, high, mid)
    return np.where(accessible, .5*(low + high), np.nan)


class _RingRays:
    """Per-pixel and central model rays, folded with the measured profile's uncertainty weights."""

    def __init__(self, geometry, q, chi_pixels, qz_pixels, signed_profile, sin_exit, internal_optics):
        self.sine = exit_sine_of_qz(geometry, qz_pixels) if sin_exit is None else sin_exit
        centres = np.arange(90) + .5
        self.central_sine = (exit_sine_of_qz(geometry, q * np.cos(np.radians(centres))) if internal_optics is None
                             else _central_exit(geometry, q, centres, internal_optics))
        self.index = np.clip(np.floor(chi_pixels.astype(float) + 90).astype(int), 0, 179)
        self.counts = np.bincount(self.index, minlength=180)
        self.profile = signed_profile
        self.signed_bins = np.floor(signed_profile.x + 90).astype(int)

    def fold(self, values, central):
        totals = np.bincount(self.index, weights=np.nan_to_num(values), minlength=180)
        bins = self.signed_bins
        signed = np.divide(totals[bins], self.counts[bins], where=self.counts[bins] > 0,
                           out=np.full(len(bins), np.nan))
        _, values, _, measured, _, _ = fold_ring(self.profile.x, signed, self.profile.sigma, 0.)
        return np.where(measured, values, np.nan_to_num(central))


def ring_transmission(optics, geometry, q, chi_pixels, qz_pixels, signed_profile, *, sin_exit=None, internal_bins=False):
    """The exit surface field alone, for the exported angular diagnostics."""
    rays = _RingRays(geometry, q, chi_pixels, qz_pixels, signed_profile, sin_exit, optics if internal_bins else None)
    return rays.fold(*(exit_transmission(s, optics["delta"], optics["beta"])
                       for s in (rays.sine, rays.central_sine)))


def _thickness(u, incident_rate, depth_rate):
    if u >= 1.:
        return np.inf
    # When the incident field has no decay, the evanescent/specular depth ratio
    # falls as 1/d. The rational coordinate keeps this limit smooth at u = 1.
    return (-np.log1p(-u) if incident_rate else u / (1.-u)) / depth_rate


def ring_optical_factors(optics, geometry, q, chi_pixels, qz_pixels, signed_profile, *,
                         fit_absorption=False, thickness_nm=np.inf, geometric_depth=None, sin_exit=None,
                         internal_bins=False):
    """Fold the full depth-integrated wave intensity on the same rays as the observed data.

    A geometrically corrected frame retains the ratio of wave to geometrical factors at its
    known thickness. Unmeasured bins are evaluated at the ring centre for plotting.
    """
    rays = _RingRays(geometry, q, chi_pixels, qz_pixels, signed_profile, sin_exit, optics if internal_bins else None)
    sin_in = math.sin(geometry.incidence_rad)
    waves = [RefractedFilm(s, sin_in, optics["delta"], optics["beta"], geometry.wavelength_angstrom)
             for s in (rays.sine, rays.central_sine)]
    if geometric_depth is not None:
        denominator = [film_absorption_factor(s, sin_in, geometric_depth) for s in (rays.sine, rays.central_sine)]
        return rays.fold(*(w.factor(thickness_nm) / g for w, g in zip(waves, denominator))), None
    if not fit_absorption:
        return rays.fold(*(w.factor(thickness_nm) for w in waves)), None

    # Usually u is the fraction of the incident intensity lost over the film thickness.
    # In a lossless film with a propagating incident field, the exit field can still be
    # evanescent. Parameterise d using its maximum exit decay rate in that case.
    incident_rate = waves[0].incident_decay_nm
    depth_rate = incident_rate or (40 * np.pi / geometry.wavelength_angstrom
                                  * math.sqrt(max(2 * optics["delta"], 0.)))
    thin = rays.fold(*(w.factor(0.) for w in waves))
    if depth_rate == 0.:
        return thin, None
    # Evanescent rays can decay on a much shorter scale than the incident field.
    # Space knots smoothly through the thin end where that loss changes most rapidly.
    grid = np.linspace(0., 1., 181)[1:]**2
    knots = np.r_[0., np.geomspace(1e-8, grid[0], 16, endpoint=False), grid]
    relative = []
    for u in knots:
        d = _thickness(u, incident_rate, depth_rate)
        factor = rays.fold(*(w.factor(d) for w in waves))
        relative.append(np.divide(factor, thin, out=np.ones_like(factor), where=thin > 0.))
    return thin, FilmAbsorption(sin_in, PchipInterpolator(knots, relative, axis=0),
                                waves[0].mu_nm, incident_rate, depth_rate)


@dataclass(frozen=True)
class FilmAbsorption:
    """A fitted depth-integrated intensity, relative to the folded thin-film surface fields."""

    sin_incidence: float
    curve: PchipInterpolator
    mu_nm: float
    incident_decay_nm: float
    depth_rate_nm: float

    def record(self, u, error_u):
        thin = u < 1e-6
        thick = u > 1. - 1e-6
        d = None if thick else float(_thickness(u, self.incident_decay_nm, self.depth_rate_nm))
        slope = (1.-u) if self.incident_decay_nm else (1.-u)**2
        d_error = (None if error_u is None or thin or thick else
                   float(error_u / (slope*self.depth_rate_nm)))
        return {"parameter": "film thickness, shared across rings of the phase",
                "formula": REFRACTED_FILM_FORMULA,
                "thickness_nm": d, "thickness_error_nm": d_error,
                "optical_depth": None if d is None else d * self.mu_nm,
                "optical_depth_error": None if d_error is None else d_error * self.mu_nm,
                "thin_limit": bool(thin), "thick_limit": bool(thick),
                "incident_intensity_decay_fraction": float(u) if self.incident_decay_nm else 0.}
