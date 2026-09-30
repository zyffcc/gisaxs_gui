"""Portable V5 workflow. TensorFlow is imported only inside the job worker.

The frozen conditional backend is not modified. Blind discovery is an experimental
front end; its estimated conditions do not inherit the known-condition validation.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np


from ...application.workflow_v5 import bundled_workflow, validate_options


def read_curve(path: Path):
    """Read numeric text with optional header, commas/tabs/spaces, no silent row loss."""
    rows = []
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        try:
            values = [float(x) for x in line.replace(",", " ").split()]
        except ValueError:
            if not rows and (
                any(word in line.lower() for word in ("q", "intensity", "sigma"))
                or line.lower().split() in (["x", "y"], ["x", "y", "err"])
            ):
                continue
            raise ValueError(f"Non-numeric row in {path.name}: {line[:80]}") from None
        if len(values) not in (2, 3, 4):
            raise ValueError("Curve files need q, intensity[, sigma[, pixels]] columns")
        rows.append(values)
    if not rows or len({len(row) for row in rows}) != 1:
        raise ValueError("Empty curve or inconsistent columns")
    a = np.asarray(rows, dtype=float)
    return a[:, 0], a[:, 1], a[:, 2] if a.shape[1] >= 3 else None


def prepare_sides(q, intensity, sigma, options):
    q, y = np.asarray(q, float).reshape(-1), np.asarray(intensity, float).reshape(-1)
    if q.shape != y.shape or not np.isfinite(q).all() or not np.isfinite(y).all():
        raise ValueError("q and intensity must have equal lengths and contain finite values")
    q = q * (10.0 if options["q_unit"] == "A^-1" else 1.0)
    estimated = sigma is None
    if estimated:
        floor = options["absolute_noise"] or max(float(np.max(np.abs(y))) * 0.001, 1e-12)
        sigma = np.hypot(options["relative_noise"] * np.abs(y), floor)
    sigma = np.asarray(sigma, float).reshape(-1)
    if sigma.shape != y.shape or not np.isfinite(sigma).all() or np.any(sigma <= 0):
        raise ValueError("Every supplied sigma must be finite and positive")
    result = []
    for side, sign in (("positive", 1), ("negative", -1)):
        if options["side"] not in ("both", side):
            continue
        indices = np.flatnonzero(q * sign > 0)
        if len(indices) == 0:
            continue
        indices = indices[np.argsort(np.abs(q[indices]), kind="stable")]
        qq = np.abs(q[indices])
        if not 8 <= len(indices) <= 1000:
            raise ValueError(
                f"{side}: need 8–1000 measured points (received {len(indices)}); no automatic interpolation"
            )
        if np.any(np.diff(qq) <= 0):
            raise ValueError(
                f"{side}: duplicate q values; combine replicates with their uncertainties first"
            )
        norm = options["normalizer"] or float(np.max(y[indices]))
        if norm <= 0:
            raise ValueError(f"{side}: at least one positive intensity is required")
        result.append(
            dict(
                side=side,
                sign=sign,
                q=qq,
                observed=y[indices],
                sigma=sigma[indices],
                normalizer=norm,
                indices=indices,
                sigma_estimated=estimated,
            )
        )
    if not result:
        raise ValueError("No nonzero q points in the selected side")
    return result


class WorkflowEngine:
    """Reuse loaded networks for all files and sides within one batch job."""

    def __init__(self, root=None):
        self.root = Path(root or bundled_workflow())
        if not (self.root / "conditional_fast_manifest_v2.json").is_file():
            raise FileNotFoundError(f"V5 workflow assets missing: {self.root}")
        sys.path.insert(0, str(self.root))
        import scipy.special, scipy.optimize  # load the current environment before frozen imports
        from native_observation_adapter import NativeObservationPredictor

        self.adapter = NativeObservationPredictor()

    def fit_side(self, item, options, report, cancelled):
        from native_observation_adapter import summarize_signed
        from conditional_fast_predict_v2 import canonical, COMBOS, KEYS, B
        from portable_predictor import PortablePredictor
        from predict_1d import preprocess
        from gpu_precision import initial_candidates
        from numpy_candidate_forward import direct_forward
        from solution_output import denormalize, PARAMETER_UNITS, GLOBAL_UNITS, WIDTH_DEFINITIONS
        from TrainSetBuild import schema

        q, y, sigma, norm = (item[k] for k in ("q", "observed", "sigma", "normalizer"))
        data = preprocess(q, np.maximum(y, 0.1 * sigma) / norm, sigma / norm)
        model = self.adapter.model
        banks = []
        conditions = []
        fixed = (
            options["components"]
            and options["sigma_res"] is not None
            and options["nu_res"] is not None
        )
        if fixed:
            conditions = [(options["components"], options["sigma_res"], options["nu_res"])]
        else:
            report(0, 1, f"{item['side']}: discovering component and resolution candidates", {})
            if model.proposal is None:
                model.proposal = PortablePredictor(B, data)
            ids = None
            if options["components"]:
                from component_prior import resolve_components

                ids = np.array([resolve_components(options["components"], COMBOS)[0]], "int32")
            seed = canonical(
                initial_candidates(model.proposal, data, ids, options["search_combinations"])
            )
            n = seed["combos"].shape[1]
            seed["curves"] = np.ones((1, n, 1000))
            valid = data["mask"][0] > 0
            for h in range(n):
                seed["curves"][0, h, valid] = direct_forward(
                    seed, 0, h, data["q"][0, valid].astype(float)
                )
            residual = (seed["curves"][0][:, valid] * norm - y) / sigma
            score = np.mean(
                np.where(abs(residual) <= 2, 0.5 * residual**2, 2 * (abs(residual) - 1)), axis=1
            )
            seen = set()
            physical = denormalize(
                seed["globals"], schema.V5_GLOBAL_TARGET_NAMES[:4], schema.V5_GLOBAL_NORM_RANGES
            )
            for head in np.argsort(score, kind="stable"):
                cid = int(seed["combos"][0, head])
                if cid in seen:
                    continue
                seen.add(cid)
                types = COMBOS[cid][COMBOS[cid] > 0].tolist()
                sr = (
                    options["sigma_res"]
                    if options["sigma_res"] is not None
                    else float(physical[0, head, 1])
                )
                nu = (
                    options["nu_res"]
                    if options["nu_res"] is not None
                    else float(physical[0, head, 2])
                )
                conditions.append((types, sr, nu))
                if len(conditions) >= options["condition_combinations"]:
                    break
            # Unconstrained seeds cannot be returned when the user fixes any resolution value.
            if options["sigma_res"] is None and options["nu_res"] is None:
                seed["candidate_stage"] = np.array(["discovery_neural"] * n)
                banks.append(seed)
        for index, (types, sr, nu) in enumerate(conditions):
            if cancelled():
                raise RuntimeError("Prediction cancelled")
            report(
                index,
                len(conditions),
                f"{item['side']}: fitting combination {index + 1}/{len(conditions)}",
                {},
            )
            _, _, raw = self.adapter.predict(
                q,
                y,
                sigma,
                types,
                sr,
                nu,
                normalizer=norm,
                numerical=options["numerical"],
                max_solutions=options["max_solutions"],
                render_points=options["render_points"],
            )
            banks.append(raw)
        merged = {k: np.concatenate([b[k] for b in banks], axis=1) for k in (*KEYS, "curves")}
        merged.update({k: data[k] for k in ("q", "observed", "sigma", "mask")})
        merged["candidate_stage"] = np.concatenate([b["candidate_stage"] for b in banks])
        result, artifact = summarize_signed(
            merged,
            y,
            sigma,
            norm,
            np.linspace(q[0], q[-1], options["render_points"]),
            options["max_solutions"],
        )
        result.update(
            unit_contract=dict(q="nm^-1", component_parameters=PARAMETER_UNITS,
                               relative_width_definitions=WIDTH_DEFINITIONS,
                               global_parameters=GLOBAL_UNITS,
                               intensity="Input intensity units; explicit normalizer recorded",
                               weight="Normalized model mixture weight, not probability/mass/volume fraction",
                               reference="BG and resolution amplitudes fixed at measured q before rendering"),
            side=item["side"],
            conditions_source="user_fixed" if fixed else "estimated_or_partially_fixed",
            searched_conditions=[
                dict(components=t, sigma_res=s, nu_res=n) for t, s, n in conditions
            ],
            sigma_estimated=item["sigma_estimated"],
            omitted_q_zero=True,
            options=options,
            validation_scope="Known complete components and resolution: conditional V2. Automatic discovery: experimental; no all-mode guarantee.",
        )
        return result, artifact, merged


def write_side(output, item, result, artifact, raw):
    output.mkdir(parents=True, exist_ok=True)
    (output / "solutions.json").write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
    )
    np.savez_compressed(output / "candidates.npz", **raw)
    np.savez_compressed(output / "display.npz", **artifact)
    rows = []
    valid = raw["mask"][0] > 0
    for rank, solution in enumerate(result["solutions"], 1):
        head = solution["candidate_index"]
        curve = raw["curves"][0, head, valid] * item["normalizer"]
        components = [
            dict(
                type=c["type"],
                weight=c["weight"],
                params=c["parameters"],
                absolute_widths_nm=c["absolute_widths_nm"],
            )
            for c in solution["components"]
        ]
        rows.append(
            dict(
                rank=rank,
                workflow="native_v5",
                side=item["side"],
                combination=" + ".join(c["type"] for c in components),
                components=components,
                global_params=solution["global_parameters"],
                best_source=solution["candidate_stage"],
                best_log_rmse=solution["positive_observation_logrmse"],
                best_chi2_weighted=solution["signed_weighted_rms"] ** 2,
                signed_huber_delta2=solution["signed_huber_delta2"],
                signed_weighted_rms=solution["signed_weighted_rms"],
                probability_status="Not calibrated",
                conditions_source=result["conditions_source"],
                normalizer=item["normalizer"],
                forward_reference=solution["reference_coefficients_normalized_intensity"],
                native_q=(item["q"] * item["sign"]).tolist(),
                observed=item["observed"].tolist(),
                sigma=item["sigma"].tolist(),
                native_fit=curve.tolist(),
                display_q=(artifact["render_q"] * item["sign"]).tolist(),
                display_fit=artifact["rendered_curves"][rank - 1].tolist(),
            )
        )
    np.savetxt(
        output / "fitting_curves.csv",
        np.column_stack([artifact["render_q"] * item["sign"], artifact["rendered_curves"].T]),
        delimiter=",",
        header="q_nm^-1," + ",".join(f"candidate_{i + 1}" for i in range(len(rows))),
        comments="",
    )
    return rows


def run_workflow_job(payload, report, is_cancelled):
    """One isolated process for a full text batch, with per-file durable results."""
    started = time.perf_counter()
    options = validate_options(payload.get("options", {}))
    output = Path(payload["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "request.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
    )
    experimental = options["method"] == "experimental"
    stable = options["method"] == "stable"
    if stable:
        from .stable_blue import StableEngine
        engine = StableEngine()
    else:
        engine = None if experimental else WorkflowEngine(payload.get("model_path"))
    sources = payload.get("files") or [None]
    records, all_rows = [], []
    for index, source in enumerate(sources):
        if is_cancelled():
            raise RuntimeError("Prediction cancelled")
        tick = time.perf_counter()
        name = Path(source).name if source else "Current curve"
        directory = output / f"{index + 1:04d}_{Path(name).stem}"
        try:
            q, y, sigma = (
                read_curve(Path(source))
                if source
                else (payload["q"], payload["intensity"], payload.get("sigma"))
            )
            sides = prepare_sides(q, y, sigma, options)
            metadata = payload.get("observation_metadata", {}) if source is None else {}
            counts = None
            if metadata.get("source") == "native_cbf_columns" and "valid_pixel_counts" in metadata:
                counts = np.asarray(metadata["valid_pixel_counts"], float)
                if counts.shape != np.asarray(q).shape or not np.isfinite(counts).all() or np.any(counts <= 0):
                    raise ValueError("Native CBF valid_pixel_counts must match q and contain finite positive counts")
            for item in sides:
                item["observation_metadata"] = {k:v for k,v in metadata.items() if k != "valid_pixel_counts"}
                if counts is not None:
                    item["count"] = counts[item["indices"]]
            if payload.get("sigma_estimated"):
                for item in sides:
                    item["sigma_estimated"] = True
            rows = []
            for item in sides:
                if stable:
                    rows.extend(engine.fit_and_write(directory / item["side"],item,options,report,is_cancelled))
                elif experimental:
                    from .experimental_fit import fit_and_write

                    rows.extend(fit_and_write(directory / item["side"], item, options, report, is_cancelled))
                else:
                    result, artifact, raw = engine.fit_side(item, options, report, is_cancelled)
                    rows.extend(write_side(directory / item["side"], item, result, artifact, raw))
            for row in rows:
                row["file"] = name
                row["file_seconds"] = time.perf_counter() - tick
                row["curve_logrmse_target"] = None
                row["curve_quality_passed"] = None
                row["curve_quality_note"] = (
                    "Observed-data residual includes measurement noise. Review peak positions "
                    "and overall shape; no mandatory logRMSE cutoff is applied."
                )
                row["quality_scope"] = "Full measured positive-intensity observations; excludes masked pixels and q=0. Not a composition/parameter accuracy guarantee."
            all_rows.extend(rows)
            records.append(
                dict(
                    file=name,
                    source=source,
                    status="complete",
                    seconds=time.perf_counter() - tick,
                    output_dir=str(directory),
                    candidates=len(rows),
                )
            )
        except Exception as exc:
            if is_cancelled() or not payload.get("files"):
                raise
            records.append(
                dict(
                    file=name,
                    source=source,
                    status="failed",
                    error=str(exc),
                    seconds=time.perf_counter() - tick,
                )
            )
        (output / "batch_summary.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
        (output / "top20_candidates.json").write_text(
            json.dumps(all_rows, indent=2, allow_nan=False), encoding="utf-8"
        )
        report(index + 1, len(sources), f"{index + 1}/{len(sources)} files finished — {name}", {})
    summary = dict(
        profile="Single RC specialist (experimental) + numerical fallback" if stable else ("Experimental physical fit (numerical)" if experimental else ("V5 + four-step correction" if options["numerical"] else "V5 neural only")),
        runtime_seconds=time.perf_counter() - started,
        configured_candidates=len(all_rows),
        best_log_rmse=all_rows[0]["best_log_rmse"] if all_rows else None,
        exit_code=0 if all_rows else 1,
    )
    (output / "fitting_run_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return dict(output_dir=str(output), summary=summary, candidates=all_rows, records=records)
