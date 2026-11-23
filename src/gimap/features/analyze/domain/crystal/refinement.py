"""Weighted refinement of the positions of indexed GIWAXS peaks, in Å⁻¹.

The reciprocal metric is linear in its independent symmetry-constrained entries. The
objective, zero shift and uncertainties are in q, as measured, not in q². The fibre
model instead scales the two orthogonal projections of the reference reciprocal lattice.
"""

from __future__ import annotations

from dataclasses import asdict
import math

import numpy as np
from scipy.optimize import least_squares

from .cell import UnitCell
from .cell_metric import CellMetric, independent_reference_parameters, reference_errors, system_for_cell
from .cell_observations import (
    CellPeak, NO_DIRECTION, POLE_TIE, SECTOR_POLE, Refraction, corrected, finite_number, select_peaks,
)

TOO_FEW = "Fewer usable peak positions than free parameters."
SINGULAR = "The indexed peak positions do not independently determine the free parameters."
UNPHYSICAL = "The fitted parameters do not describe a physical unit cell."
NO_FIT = "The weighted position fit did not converge."
COINCIDENT = "The coincident indices separate when the free cell parameters change."


def _rank(jacobian):
    """Compare singular values after removing parameter units; avoid amplifying numerical zeros."""
    norm = np.linalg.norm(jacobian, axis=0)
    if not norm.size or not norm.max():
        return 0
    active = norm > norm.max() * 1e-12
    singular = np.linalg.svd(jacobian[:, active] / norm[active], compute_uv=False)
    return int(np.count_nonzero(singular > singular[0] * 1e-10))


def _full_rank(jacobian):
    return _rank(jacobian) == jacobian.shape[1]


def _weighted_fit(design, initial, observed, sigma, *, fit_shift=None):
    """Fit sqrt(D·x) plus a constant. The tangent below q²=0 only guides invalid trial steps back."""
    n, p = design.shape
    if n < p:
        raise ValueError(TOO_FEW)
    floor = np.min(observed)**2 * 1e-12

    def positions(x):
        square = design @ x[:p]
        root = np.sqrt(np.maximum(square, floor))
        return root + (np.minimum(square - floor, 0.) / (2 * root))

    def jacobian(x):
        root = np.sqrt(np.maximum(design @ x[:p], floor))
        jac = design / (2 * root[:, None])
        if len(x) > p:
            jac = np.column_stack((jac, np.ones(n)))
        return jac / sigma[:, None]

    def residual(x):
        q = positions(x)
        if len(x) > p:
            q = q + x[-1]
        return (q - observed) / sigma

    if not _full_rank(jacobian(initial)):
        raise ValueError(SINGULAR)
    augmented = np.r_[initial, 0.]
    shift_free = (n >= p + 2 and _full_rank(jacobian(augmented))) if fit_shift is None else fit_shift
    start = augmented if shift_free else initial
    fit = least_squares(residual, start, jac=jacobian, x_scale="jac")
    if not fit.success:
        raise ValueError(NO_FIT)
    if not _full_rank(fit.jac):
        raise ValueError(SINGULAR)
    if np.any(design @ fit.x[:p] <= 0):
        raise ValueError(UNPHYSICAL)
    dof = n - len(start)
    chi2 = float(fit.fun @ fit.fun)
    chi2_red = chi2 / dof if dof else None
    _, singular, vt = np.linalg.svd(fit.jac, full_matrices=False)
    covariance = (vt.T / singular**2) @ vt * max(1., chi2_red or 0.)
    info = {"chi2_red": chi2_red, "chi2": chi2, "dof": dof,
            "zero_shift_fitted": shift_free, "zero_shift": float(fit.x[-1]) if shift_free else 0.,
            "zero_shift_error": float(np.sqrt(covariance[-1, -1])) if shift_free else None}
    return fit.x[:p], covariance, observed + fit.fun * sigma, info


def _pole_projection(peak, direction, reference, normal):
    if direction is None and peak.chi_range_deg is None:
        raise ValueError(NO_DIRECTION)
    indices = np.array(peak.equivalents or (peak.hkl,))
    vectors = reference.q_vectors(indices)
    z = np.abs(vectors @ normal)
    par = np.linalg.norm(vectors - (vectors @ normal)[:, None] * normal, axis=1)
    angles = np.degrees(np.arctan2(par, z))
    if peak.chi_range_deg is not None:
        low, high = peak.chi_range_deg
        inside = np.flatnonzero((angles >= low - 1e-5) & (angles <= high + 1e-5))
        if not inside.size or np.ptp(angles[inside]) > 1e-4:
            raise ValueError(SECTOR_POLE)
        best = inside[0]
    else:
        measured = math.degrees(math.atan2(*direction))
        distances = np.abs(angles - measured)
        best = np.argmin(distances)
        tied = np.isclose(distances, distances[best], atol=1e-6, rtol=0)
        if np.ptp(angles[tied]) > 1e-5:
            raise ValueError(POLE_TIE)
    return [float(par[best]**2), float(z[best]**2)], indices[best].tolist(), float(angles[best])


def _refine(reference, peaks, optics, *, metric=None, axis_hkl=None, system="", cell_errors=()):
    result = {"ok": False, "reason": "", "system": system, "values": {}, "errors": {}, "reference": {},
              "strain": {}, "strain_errors": {}, "chi2_red": None, "residuals": [], "excluded": [],
              "zero_shift_fitted": False, "zero_shift": None, "zero_shift_error": None,
              "zero_shift_rule": "Fit an additive q offset with N >= p + 2 and full augmented Jacobian rank.",
              "uncertainties": "Absolute peak-position errors; covariance multiplied by max(1, reduced chi-square).",
              "refraction": {k: float(v) for k, v in asdict(optics).items()} if optics is not None else None}
    if axis_hkl is not None:
        result["axis_hkl"] = [int(v) for v in axis_hkl]
        normal = reference.plane_normal(axis_hkl)
    n_free = len(metric.start) if metric is not None else 2
    design, rows, directions = [], [], []
    for peak in peaks:
        source_index = None if peak.source_index is None else int(peak.source_index)
        aperture = None if peak.chi_range_deg is None else tuple(map(finite_number, peak.chi_range_deg))
        try:
            q, sigma, direction = corrected(peak, optics)
            pole_angle = None
            if metric is None:
                coefficients, hkl, pole_angle = _pole_projection(peak, direction, reference, normal)
                # For measured spots the data must span both projections, as must their assigned indices.
                # A sector instead supplies the one reference pole angle selected by its aperture.
                directions.append(np.square(direction) if direction is not None else coefficients / np.sum(coefficients))
            else:
                hkl = list(peak.hkl)
                coefficients = (2 * np.pi)**2 * np.einsum("i,nij,j->n", hkl, metric.basis, hkl)
                if peak.equivalents:
                    # Some metric coincidences are grouped by index permutation for a centring-only phase.
                    indices = np.asarray(peak.equivalents)
                    members = (2 * np.pi)**2 * np.einsum("ki,nij,kj->kn", indices, metric.basis, indices)
                    if not np.allclose(members, coefficients, atol=1e-10, rtol=1e-10):
                        raise ValueError(COINCIDENT)
            design.append(coefficients)
            rows.append({"hkl": list(map(int, hkl)), "source_index": source_index, "q": float(peak.q),
                         "sigma": float(peak.sigma), "q_corrected": float(q), "sigma_corrected": float(sigma),
                         "q_parallel": finite_number(peak.q_parallel), "qz": finite_number(peak.qz), "chi_range_deg": aperture,
                         "reference_pole_angle_deg": pole_angle})
        except ValueError as exc:
            result["excluded"].append({"source_index": source_index, "q": finite_number(peak.q), "reason": str(exc),
                                       "chi_range_deg": aperture})
    result.update(metric_rank=_rank(np.asarray(design)) if design else 0, metric_parameters=n_free)
    if len(rows) < n_free:
        result["reason"] = TOO_FEW
        return result
    if metric is None and not _full_rank(np.asarray(directions)):
        result["reason"] = SINGULAR
        return result
    try:
        x, covariance, predicted, info = _weighted_fit(
            np.asarray(design), metric.start if metric is not None else np.ones(2),
            np.array([r["q_corrected"] for r in rows]), np.array([r["sigma_corrected"] for r in rows]))
        if metric is None:
            if np.any(x <= 0):
                raise ValueError(UNPHYSICAL)
            names = ["in_plane", "out_of_plane"]
            values = dict(zip(names, (1 / np.sqrt(x) - 1).tolist()))
            deriv = np.diag(-.5 * x**-1.5)
            ref = dict.fromkeys(names, 0.)
        else:
            names = metric.names
            values, deriv = metric.parameters(x)
            native, _ = metric.parameters(x, native=True)
            result["native_cell"] = list(native.values())
            if metric.rhombohedral:
                ref, _ = metric.parameters(metric.start)
                result["cell_setting"] = "conventional hexagonal (CIF indices in rhombohedral axes)"
            else:
                ref = {name: float(getattr(reference, name)) for name in names}
                ref["V"] = float(reference.volume)
        prop = np.zeros((len(values) + int(info["zero_shift_fitted"]), len(covariance)))
        prop[:len(values), :n_free] = deriv
        order = list(values)
        if info["zero_shift_fitted"]:
            order.append("zero_shift")
            prop[-1, -1] = 1.
        propagated = prop @ covariance @ prop.T
        errors = dict(zip(values, np.sqrt(np.maximum(np.diag(propagated)[:len(values)], 0.)).tolist()))
    except (ValueError, np.linalg.LinAlgError) as exc:
        result["reason"] = str(exc) if isinstance(exc, ValueError) else UNPHYSICAL
        return result
    for row, prediction in zip(rows, predicted):
        row["q_calc"] = float(prediction)
        row["residual"] = float(row["q_corrected"] - prediction)
    if metric is not None:
        ref_errors = result["reference_errors"] = reference_errors(reference, metric, system, cell_errors)
        for name in ("a", "b", "c", "V"):
            if name in values:
                result["strain"][name] = (values[name] - ref[name]) / ref[name]
                result["strain_errors"][name] = math.hypot(errors[name] / ref[name],
                    values[name] * ref_errors.get(name, 0.) / ref[name]**2)
    result.update(info, ok=True, values=values, errors=errors, reference=ref, residuals=rows,
                  parameter_order=order, covariance=propagated.tolist())
    return result


def refine_cell(reference, system, peaks, *, refraction=None, cell_errors=()):
    """Fit the free cell parameters with measured radial errors and propagate errors to volume and strain."""
    metric = CellMetric(reference, system)
    return _refine(reference, peaks, refraction, metric=metric, system=system, cell_errors=cell_errors)


def refine_fibre_strain(reference, axis_hkl, peaks, *, refraction=None, cell_errors=(), system=None):
    """Fit two real-space strains relative to the reference, conditional on the fitted fibre pole.

    For each 2D/sector centre select the family member at the closest polar angle. Fit its
    radial position with q² = q_ref,parallel²/(1+ε_parallel)² + q_ref,z²/(1+ε_z)².
    """
    result = _refine(reference, peaks, refraction, axis_hkl=axis_hkl, system="fibre")
    if not result["ok"] or not any(s is not None for s in cell_errors):
        return result
    system = system or system_for_cell(reference)
    metric = CellMetric(reference, system)
    rows = result["residuals"]
    # The observed radii and selected symmetry-equivalent poles stay fixed. A small
    # change to the CIF affects both the reciprocal vectors and the director.
    indices = [r["hkl"] for r in rows]
    q = np.array([r["q_corrected"] for r in rows])
    sigma = np.array([r["sigma_corrected"] for r in rows])

    def fitted(cell):
        vectors = cell.q_vectors(indices)
        normal = cell.plane_normal(axis_hkl)
        z = vectors @ normal
        parallel = vectors - z[:, None] * normal
        design = np.column_stack((np.sum(parallel**2, axis=1), z*z))
        x, _, _, info = _weighted_fit(design, np.ones(2), q, sigma, fit_shift=result["zero_shift_fitted"])
        values = 1 / np.sqrt(x) - 1
        return np.r_[values, info["zero_shift"]] if result["zero_shift_fitted"] else values

    parameters = np.array(reference.as_tuple(), dtype=float)
    contribution = np.zeros_like(result["covariance"])
    for group, error in independent_reference_parameters(metric, system, cell_errors):
        step = parameters[group[0]] * 1.e-5
        if group[0] >= 3:
            step = min(step, (180. - parameters[group[0]]) * 1.e-5)
        delta = np.zeros(6)
        delta[list(group)] = step
        derivative = (fitted(UnitCell(*(parameters + delta))) - fitted(UnitCell(*(parameters - delta)))) / (2*step)
        contribution += np.outer(derivative, derivative) * error**2
    covariance = np.asarray(result["covariance"]) + contribution
    result.update(reference_covariance=contribution.tolist(), covariance=covariance.tolist(),
                  uncertainties="Peak-position covariance plus independent CIF reference-parameter propagation.")
    for name, error in zip(result["parameter_order"], np.sqrt(np.diag(covariance))):
        if name == "zero_shift":
            result["zero_shift_error"] = float(error)
        else:
            result["errors"][name] = float(error)
    return result


__all__ = ["CellPeak", "Refraction", "refine_cell", "refine_fibre_strain", "select_peaks", "system_for_cell"]
