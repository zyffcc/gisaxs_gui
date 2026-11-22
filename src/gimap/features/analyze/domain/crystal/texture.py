"""Orientation of a phase from all its assigned ring profiles."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

from .....shared.ring_profile import measured_hermans
from .pole_figure import MIN_BINS, PoleProfile
from .pole_models import GAUSSIAN_MOSAIC, hermans
from .reflections import Phase, reflections
from .texture_kernel import BIN_SAMPLES, LOG_WIDTHS, MAX_DEGREE, WIDTH_BOUNDS, PoleLibrary, candidate_axes, ring_families
from .texture_search import BIC_TIE, Hypothesis, RingData, _coefficients, multiple_poles, search


MAX_REDUCED_CHI2 = 5.
MIN_DELTA_BIC = 6.


@dataclass(frozen=True)
class TextureRing:
    ring_id: int
    q: float
    q_min: float
    q_max: float
    profile: PoleProfile
    transmission: np.ndarray | None = None


def _finite(value):
    return None if value is None or not np.isfinite(value) else float(value)


def _error(covariance, i):
    variance = covariance[i, i]
    return _finite(np.sqrt(max(0., variance))) if np.isfinite(variance) else None


def _diagnostics(problem, x):
    """Joint Jacobian incl. the profiled scales/weights, and each ring's residual degrees of freedom."""
    blocks, residuals, amplitudes = [], [], []
    start = len(x)
    for i, ring in enumerate(problem.rings):
        shapes = problem.shapes(x, i) if len(x) else ring.iso[None]
        coeff = _coefficients(shapes, ring.y)
        jac = np.zeros((len(ring.y), problem.parameters))
        for j in range(len(x)):
            up, down = x.copy(), x.copy()
            step = 1e-5
            up[j], down[j] = x[j] + step, x[j] - step
            jac[:, j] = coeff @ (problem.shapes(up, i) - problem.shapes(down, i)) / (2*step)
        jac[:, start:start+len(coeff)] = shapes.T
        start += len(coeff)
        amplitudes.append(coeff)
        blocks.append(jac)
        residuals.append(coeff @ shapes - ring.y)
    jac = np.concatenate(blocks)
    lengths = np.linalg.norm(jac, axis=0)
    scaled = jac / np.where(lengths > 0., lengths, 1.)
    u, s, vt = np.linalg.svd(scaled, full_matrices=False)
    rank = int(np.count_nonzero(s > np.finfo(float).eps * max(scaled.shape) * s[0]))
    leverage = np.sum(u[:, :rank]**2, axis=1)
    dof = len(jac) - problem.parameters
    red = sum(float(r @ r) for r in residuals) / dof
    covariance = np.full((problem.parameters, problem.parameters), np.nan)
    if rank == problem.parameters:
        inverse = (vt.T / s) / lengths[:, None]
        covariance = (inverse @ inverse.T) * max(red, 1.)
    local_red, local_dof = [], []
    for residual, lev in zip(residuals, np.split(leverage, np.cumsum([len(r) for r in residuals[:-1]]))):
        df = float(len(residual) - lev.sum())
        local_dof.append(df)
        local_red.append(float(residual @ residual) / df if df > 0. else None)
    return red, covariance, amplitudes, local_red, local_dof


def _ring_record(data, trial, coeff, red, dof):
    ring, library = data.ring, data.library
    scale = float(coeff.sum())
    relative = coeff / scale if trial.free and scale > 0. else library.fractions
    if trial.indices:
        oriented = library.spline(trial.x[0])[trial.indices[0]]
        if len(trial.indices) == 2:
            second = library.spline(trial.x[2])[trial.indices[1]]
            oriented = trial.x[3]*oriented + (1. - trial.x[3])*second
        shape = (1. - trial.x[1])*oriented + trial.x[1]*library.isotropic
    else:
        shape = library.isotropic[None, :]
    contributions = scale * relative[:, None] * shape
    components = [{"hkl": f.label, "multiplicity": f.multiplicity, "q_hkl": f.q, "sum_f2": f.strength,
                   "kinematic_weight": float(kinematic), "weight": float(weight), "values": curve.tolist()}
                  for f, kinematic, weight, curve in zip(data.families, library.fractions, relative, contributions)]
    profile, use = ring.profile, data.mask
    fitted = contributions.sum(axis=0)
    residual = (profile.value[use] - fitted[use]) / profile.sigma[use]
    worst = int(np.argmax(np.abs(residual)))
    f_ring, f_error, _ = measured_hermans(profile.chi[use], profile.value[use], profile.sigma[use])
    return {
        "ring_id": int(ring.ring_id), "q": float(ring.q), "q_window": [float(ring.q_min), float(ring.q_max)],
        "points": int(use.sum()), "effective_dof": dof, "reduced_chi2": _finite(red), "scale": scale,
        "herman_measured": _finite(f_ring), "herman_measured_err": _finite(f_error),
        "largest_residual": {"chi_deg": float(profile.chi[use][worst]), "observed_minus_fit_sigma": float(residual[worst])},
        "chi_deg": profile.chi.tolist(), "fitted": fitted.tolist(), "components": components,
    }


def _order(trial):
    """Largest fraction first, narrower width first for equal fractions."""
    if not trial.indices:
        return []
    shares = [1.] if len(trial.indices) == 1 else [float(trial.x[3]), float(1. - trial.x[3])]
    return sorted(enumerate(shares), key=lambda item: (-item[1], trial.x[2*item[0]]))


def _directions(trial, axes):
    return [list(axes[trial.indices[i]]) for i, _ in _order(trial)]


def _populations(trial, axes, covariance):
    records = []
    for i, share in _order(trial):
        width = math.exp(trial.x[2*i])
        width_err = _error(covariance, 2*i)
        oriented = 1. - trial.x[1]
        # Propagate the covariance of (1-phi)*share, including their cross term.
        gradient = np.zeros(len(trial.x))
        gradient[1] = -share
        if len(trial.indices) == 2:
            gradient[3] = oriented if i == 0 else -oriented
        variance = gradient @ covariance[:len(gradient), :len(gradient)] @ gradient
        records.append({"axis": list(axes[trial.indices[i]]), "fwhm_deg": width,
                        "fwhm_err_deg": None if width_err is None else width*width_err,
                        "fraction": float(oriented * share),
                        "fraction_err": _finite(np.sqrt(max(variance, 0.)))})
    return records


def fit_crystal_orientation(phase: Phase, rings: Sequence[TextureRing], *, check_cancel=None) -> dict:
    """Fit all supplied profiles of one phase. Each extraction window is in Å⁻¹.

    Relative family weights are compared by BIC; they retain the calculated
    ratios among the symmetry-equivalent poles of each family. Returned ring
    curves include the model over unmeasured bins and use the original folded
    profile's units. A cancellation callback can raise between candidates.
    """
    result = {"phase": phase.name, "ok": False, "determined": False, "reasons": [], "rings": [],
              "model": GAUSSIAN_MOSAIC, "axes": [], "populations": [], "runner_up_axes": None, "delta_bic": None,
              "width_bounds_deg": list(WIDTH_BOUNDS), "max_reduced_chi2": MAX_REDUCED_CHI2,
              "min_delta_bic": MIN_DELTA_BIC, "excluded_rings": []}
    if not rings:
        result["reasons"].append("no assigned rings")
        return result
    axes = candidate_axes(phase)
    found = reflections(phase, max(r.q_max for r in rings))
    result.update(
        approximated_scattering_factors=list(found.approximated),
        structure_factor_source="CIF atom sites" if found.with_atoms else "unit intensity per allowed reflection",
        search_settings={"axis_index_limit": 2, "candidate_axis_count": len(axes), "max_axes_per_model": 2,
                         "width_interpolation": "cubic spline in log FWHM", "width_knots": len(LOG_WIDTHS),
                         "max_legendre_degree": MAX_DEGREE, "bin_width_deg": 1., "bin_samples": BIN_SAMPLES,
                         "selection": "BIC = chi2 + k ln(N)", "bic_tie_tolerance": BIC_TIE,
                         "mosaic_widths": "one per population; repeated axis allowed",
                         "min_distinct_axis_ring_separation_sigma": 1.},
    )
    data = []
    for ring in rings:
        if check_cancel is not None:
            check_cancel()
        families = ring_families(phase, found, ring.q_min, ring.q_max)
        p = ring.profile
        mask = p.measured & np.isfinite(p.value) & np.isfinite(p.sigma) & (p.sigma > 0.)
        if not families or mask.sum() < MIN_BINS:
            reason = "no phase reflections in the extraction window" if not families else "fewer than 8 measured bins"
            result["excluded_rings"].append({"ring_id": int(ring.ring_id), "q": float(ring.q), "reason": reason})
            continue
        optics = {} if ring.transmission is None else {"transmission": ring.transmission}
        data.append(RingData(ring, families, PoleLibrary(phase, axes, families, p.chi, **optics), mask))
    if not data:
        result["reasons"].append("no assigned ring has a usable profile and a reflection in its extraction window")
        return result

    result["search_settings"]["noncollinear_poles_available"] = bool(multiple_poles(data))
    trials = search(data, len(axes), check_cancel)
    best = trials[0]
    runner = trials[1] if len(trials) > 1 else None
    problem = Hypothesis(data, best.indices, best.free)
    red, covariance, amplitudes, local_red, local_dof = _diagnostics(problem, best.x)
    populations = _populations(best, axes, covariance)
    phi = float(best.x[1]) if best.indices else 1.
    phi_err = _error(covariance, 1) if best.indices else None
    delta = float(runner.bic - best.bic) if runner is not None else None
    if delta is not None and abs(delta) <= BIC_TIE:
        delta = 0.
    reasons = result["reasons"]
    if not best.indices:
        reasons.append("the isotropic model has the lowest BIC")
    if red > MAX_REDUCED_CHI2 or any(r is None or r > MAX_REDUCED_CHI2 for r in local_red):
        reasons.append("joint or ring reduced χ² exceeds 5")
    if delta is None or delta < MIN_DELTA_BIC:
        reasons.append("the competing orientation is within 6 BIC units or unavailable")
    if best.indices and (phi_err is None or any(p["fwhm_err_deg"] is None or p["fraction_err"] is None
                                                for p in populations)):
        reasons.append("fit parameters are not identifiable")
    width_at_bound = any(abs(p["fwhm_deg"] - b) < 1e-3 for p in populations for b in WIDTH_BOUNDS)
    if best.indices and (abs(phi-1.) < 1e-5 or width_at_bound):
        reasons.append("the width or isotropic limit is at a bound")
    directions = _directions(best, axes)
    fractions = [share for _, share in _order(best)]
    share_errors = ([_error(covariance, 3)]*2 if len(fractions) == 2
                    else [0.] if len(fractions) == 1 else [])
    # In a mixed texture, f is the alignment of each population's own fibre
    # director with the surface normal, averaged over populations and isotropic material.
    result.update({
        "ok": True, "determined": bool(best.indices and not reasons), "axes": directions,
        "populations": populations, "axis_fractions": fractions, "axis_fractions_err": share_errors,
        "family_weights": "free" if best.free else "kinematic",
        "runner_up_axes": _directions(runner, axes) if runner is not None else None,
        "runner_up_tied_axes": [_directions(t, axes) for t in trials[2:]
                                if runner is not None and abs(t.bic-runner.bic) <= BIC_TIE],
        "runner_up_bic": float(runner.bic) if runner is not None else None, "delta_bic": delta,
        "bic": float(best.bic), "points": problem.points, "parameters": problem.parameters,
        "reduced_chi2": float(red),
        "fwhm_deg": populations[0]["fwhm_deg"] if len(populations) == 1 else None,
        "fwhm_err_deg": populations[0]["fwhm_err_deg"] if len(populations) == 1 else None,
        "fraction_iso": phi, "fraction_iso_err": phi_err, "oriented_fraction": 1. - phi,
        "oriented_fraction_err": phi_err,
        "herman_crystal": sum(p["fraction"] * hermans(GAUSSIAN_MOSAIC, 0., p["fwhm_deg"], 0.)
                              for p in populations),
        "herman_crystal_reference": "each population's own fibre axis relative to the surface normal",
        "rings": [_ring_record(r, best, c, v, df) for r, c, v, df in zip(data, amplitudes, local_red, local_dof)],
    })
    return result
