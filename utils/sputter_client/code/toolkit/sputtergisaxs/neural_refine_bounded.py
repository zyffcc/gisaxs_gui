'Low-angle NR objective with bounded smooth backgrounds.'
from __future__ import annotations
import copy
import time
from functools import lru_cache
import numpy as np
from scipy.optimize import least_squares, lsq_linear
from scipy.stats import chi2
from .neural_refine_input import Q, NATIVE_AF, extract, project_partition_rates
from .neural_refine_refine import (
    selected_partition, _to_internal, _from_internal, _bounds, _unit_signal)
from .neural_refine_chart import decode
from .neural_refine_forward import rates
from .neural_refine_measurement import response
from .neural_refine_likelihood import residual, compare_floors

RECIPE = "H0123_once_reference_powerlaw_bounded_sideband_v1"


def objective_policy(module):
    return dict(module=module, horizontal=[0, 1, 2, 3], vertical=False,
                highq_only=False, normalization="per_cut_bounded_components",
                common_signal_indices=list(range(8 * len(Q))),
                source=RECIPE, D_condition="input_only_no_Dcorr_truncation")


def background_contract(y, e, ry, re, q, groups, observations, reference_observations):
    'Fix monotonic shape and amplitude envelopes before structural fitting.'
    rate = np.divide(y, e, out=np.zeros_like(y), where=e > 0)
    rrate = np.divide(ry, re, out=np.zeros_like(ry), where=re > 0)
    bg = np.zeros_like(y); flat = np.zeros_like(y); records = {}
    sideband = q >= .1
    for obs in (observations, reference_observations):
        peak, width = obs.get("q_star_nm_inv"), obs.get("FWHM_nm_inv")
        if peak is not None and width is not None:
            sideband &= abs(q - peak) > max(.5 * width, .075)
    for group in np.unique(groups):
        ix = groups == group
        ok = ix & sideband & (e > 0) & (re > 0) & (rrate > 0)
        if ok.sum() < 6:

            exponent = 2.; source = "fixed_q^-2_fallback_insufficient_reference_sidebands"
        else:
            slope = np.polyfit(np.log(q[ok] / .5), np.log(rrate[ok]), 1)[0]
            exponent = float(np.clip(-slope, 0., 4.))
            source = "earliest_observed_reference_sideband_log_slope_clipped_0_4"
        shape = (q[ix] / .5) ** -exponent
        sb = sideband[ix] & (e[ix] > 0)
        if sb.sum() < 6:
            raise ValueError("Fewer than six nonpeak sideband cells for " + str(group))
        envelope = float(np.quantile(rate[ix][sb] / shape[sb], .25))
        flat_envelope = float(np.quantile(rate[ix][sb], .25))
        bg[ix] = 1.25 * max(envelope, 1e-12) * shape * e[ix]
        flat[ix] = .10 * max(flat_envelope, 1e-12) * e[ix]
        records[str(group)] = dict(
            exponent=exponent, slope_source=source, sideband_cells=int(sb.sum()),
            amplitude_bounds=[0., 1.25 * max(envelope, 1e-12)],
            flat_rate_bounds=[0., .10 * max(flat_envelope, 1e-12)],
            reference_certified_zero_metal=False,
            constraint_scope="TRAIN-derived working envelope, not measured flux or confidence interval")
    return bg, flat, records, sideband


def profile_components(y, signal, bg, flat, groups, eps):
    'Profile positive signal and bounded background/flat for each cut.'
    prediction = np.zeros_like(y); metal = np.zeros_like(y); nuisance = {}
    for group in np.unique(groups):
        ix = groups == group; a = y[ix]; s = signal[ix]
        normalizer = max(float(np.quantile(s[s > 0], .75)) if np.any(s > 0) else 0., 1e-100)
        scale = max(float(np.quantile(a, .75)), 1.)
        columns = np.column_stack((s / normalizer * scale, bg[ix], flat[ix]))
        lower = np.zeros(3); upper = np.array([1e4, 1., 1.])
        weight = 1. / np.sqrt(np.maximum(a + (eps * a)**2, 1.))
        start = lsq_linear(columns * weight[:, None], a * weight,
                           bounds=(lower, upper), tol=1e-10).x
        fit = least_squares(lambda x: residual(a, columns @ x, eps),
                            np.clip(start, 1e-10, upper - 1e-10),
                            bounds=(lower, upper), max_nfev=60,
                            ftol=1e-10, xtol=1e-10, gtol=1e-9)
        prediction[ix] = columns @ fit.x
        metal[ix] = columns[:, 0] * fit.x[0]
        nuisance[str(group)] = dict(
            signal_gain=float(fit.x[0] * scale / normalizer),
            background_fraction_of_bound=float(fit.x[1]),
            flat_fraction_of_bound=float(fit.x[2]), optimizer_success=bool(fit.success))
    return prediction, metal, nuisance


def scores(y, mu, nparameters):
    out = compare_floors(y, mu, nparameters)
    for value in out:
        value["asymptotic_tail_probability"] = float(chi2.sf(value["deviance"], value["dof"]))
        value["compatible_at_5pct_asymptotic"] = value["asymptotic_tail_probability"] >= .05
        value["compatibility_scope"] = "model-level asymptotic reference; data-selected bounds and parameter recovery not calibrated"
    return out


def refine_bounded(Y, E, proposal, *, energy_ev, alpha_i_deg, metal,
                   reference, module, max_nfev=30, nsize=224, nz=64):
    started = time.perf_counter()
    obs = extract(Y, E, energy_ev, alpha_i_deg, metal=metal)["observations"]
    policy = objective_policy(module)
    part = selected_partition(Y, E, None, policy); keep = part["keep"]
    y, e = part["y"][keep], part["e"][keep]
    groups, q = part["groups"][keep], part["coordinates"][keep]
    rpart = selected_partition(reference["Y"], reference["E"], None, policy)
    bg, flat, contract, sideband = background_contract(
        y, e, rpart["y"][keep], rpart["e"][keep], q, groups, obs, reference["observations"])
    z = np.asarray(proposal["chart_z"], float); active = np.asarray(proposal["active"], bool)
    prototype = copy.deepcopy(proposal["prototype"])
    prototype.update(energy_ev=float(energy_ev), alpha_i_deg=float(alpha_i_deg), metal=metal)
    result = dict(
        mode_id=proposal["mode_id"], branch=proposal.get("branch"), probability=proposal.get("probability"),
        recipe=RECIPE, network_chart_z=z.tolist(), active=active.tolist(),
        network_intervals=proposal.get("intervals"), background_contract=contract,
        background_source=reference["source"] + "; earliest observed, not certified zero metal",
        model_error_floor=.05, input_policy=policy, observed_features=obs,
        observed_partition_counts=y.tolist(), selected_partition_exposure=e.tolist(),
        selected_partition_group=groups.tolist(), selected_partition_kind=part["kinds"][keep].tolist(),
        selected_partition_coordinate=q.tolist(), background_sideband=sideband.tolist(),
        background_upper_counts=bg.tolist(), flat_upper_counts=flat.tolist(),
        structural_validation="pending_new_end_to_end_simulation_TEST",
        refined_interval_scope="network conditional intervals only; refined intervals not calibrated",
        profile_scope="independent cut amplitudes; relative AF loading and height not independently identified",
        coordinate_units="H:q_y nm^-1", D_window=None)
    try:
        initial = decode(prototype, z, active)
        if not initial["valid"]:
            raise ValueError("Invalid original network proposal: " + str(initial["reasons"]))
        notes = []; x0 = _to_internal(z, active, notes); lo, hi = _bounds(prototype, active, None)
        if np.any(x0 < lo) or np.any(x0 > hi):
            raise ValueError("Original network proposal outside physical model domain; retained without clipping")
        x0 = np.clip(x0, lo + 1e-8*(hi-lo), hi - 1e-8*(hi-lo))
        nstruct = len(x0)

        @lru_cache(maxsize=4)
        def raw_signal(key):
            chart = _from_internal(np.asarray(key), z, active)
            decoded = decode(prototype, chart, active)
            if not decoded["valid"]:
                raise ValueError("Invalid coupled physical chart: " + str(decoded["reasons"]))
            return rates(_unit_signal(decoded["parameters"]), nsize=nsize, nz=nz)

        def prediction(x):
            raw = raw_signal(tuple(x[:nstruct]))
            signal = project_partition_rates(response(raw, x[-2], x[-1]), E, None)[keep]
            return profile_components(y, signal, bg, flat, groups, .05)


        lower = np.r_[lo, .002, .002]; upper = np.r_[hi, .035, .025]
        start = np.r_[x0, .015, .008]
        network_fit = least_squares(
            lambda v: residual(y, prediction(np.r_[x0, v])[0], .05),
            start[-2:], bounds=(lower[-2:], upper[-2:]), max_nfev=12,
            x_scale="jac", ftol=2e-6, xtol=2e-6, gtol=1e-5)
        start[-2:] = network_fit.x
        network_mu, network_metal, network_nuisance = prediction(start)
        sol = least_squares(lambda x: residual(y, prediction(x)[0], .05), start,
                            bounds=(lower, upper), max_nfev=max_nfev, x_scale="jac",
                            ftol=2e-6, xtol=2e-6, gtol=1e-5)
        mu, signal, nuisance = prediction(sol.x)
        chart = _from_internal(sol.x[:nstruct], z, active)
        count = len(sol.x) + 3*len(nuisance)
        result.update(
            status="refined" if sol.success else "refinement_iteration_limit",
            refined_chart_z=chart.tolist(), parameters=decode(prototype, chart, active)["parameters"],
            optimizer_success=bool(sol.success), optimizer_nfev=int(sol.nfev),
            numerical_boundary_start_notes=notes, PSF_q_nm_inv=float(sol.x[-2]), AF_blur_deg=float(sol.x[-1]),
            refined_boundary_flags=((sol.x-lower < .002*(upper-lower)) | (upper-sol.x < .002*(upper-lower))).tolist(),
            network_partition_counts=network_mu.tolist(), refined_partition_counts=mu.tolist(),
            network_metal_partition_counts=network_metal.tolist(), refined_metal_partition_counts=signal.tolist(),
            network_nuisance=network_nuisance, profiled_per_cut_gains=nuisance,
            network_initial_scores=scores(y, network_mu, count), refined_scores=scores(y, mu, count),
            local_parameter_count=count, native_orders=dict(nsize=nsize, nz=nz),
            curve_scope="exact fitted H0-H3 counts; no duplicated V or feature penalty",
            peak_signal_fraction=float(signal[~sideband].sum()/max(mu[~sideband].sum(), 1e-100)))
    except (ValueError, FloatingPointError, OverflowError) as exc:
        result.update(status="preserved_network_proposal_refinement_not_completed", reason=str(exc))
    result["wall_seconds"] = time.perf_counter() - started
    return result
