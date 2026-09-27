"""Guarded local neural branch with explicit calibration and numerical fallback.

No training/research module is imported at runtime. Neither candidate ranking nor
acceptance for fast routing is a calibrated probability or component identity.
"""

import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import nnls
from scipy.special import expit

from ...application.workflow_v5 import bundled_workflow
from ...domain.blue_rc_forward import FORWARD_VERSION, forward, particle


def bundle_path():
    return bundled_workflow().parent / "stable_blue_rc_v1"


def encode_features(q, y, count, edges):
    centers = (edges[:-1] + edges[1:]) / 2
    idx = np.searchsorted(edges, q, side="right") - 1
    valid = np.isfinite(y) & (count > 0) & (idx >= 0) & (idx < len(centers))
    idx, q, y, count = idx[valid], q[valid], y[valid], count[valid]
    n = len(centers)
    exposure = np.bincount(idx, weights=count, minlength=n)
    total = np.bincount(idx, weights=y * count, minlength=n)
    qsum = np.bincount(idx, weights=q * count, minlength=n)
    present = exposure > 0
    means = total / np.maximum(exposure, 1)
    log_y = np.where(present, np.log(np.maximum(means, 0.1 / np.maximum(exposure, 1))), 0.0)
    offset = np.where(present, (qsum / np.maximum(exposure, 1) - centers) / np.diff(edges), 0.0)
    return np.r_[log_y, present, offset, np.log1p(exposure)].astype("float32")


def eligibility(item, options):
    if options["components"] != [2]:
        return (
            "The single-RC specialist requires an explicitly selected single random cylinder; "
            "a small curve residual cannot establish composition"
        )
    if options["sigma_res"] is not None or options["nu_res"] is not None:
        return "Fixed resolution values require constrained numerical fitting"
    metadata = item.get("observation_metadata", {})
    if metadata.get("source") != "native_cbf_columns" or "count" not in item:
        return "A native CBF counting/exposure contract is required for the learned branch"
    if metadata.get("intensity_unit") != "counts_per_pixel":
        return "The intensity unit is not explicitly recorded as counts per pixel"
    if (
        metadata.get("threshold_enabled")
        or metadata.get("mirror_replaced_pixels", 0)
        or metadata.get("stack_count", 1) != 1
    ):
        return "Thresholded, mirrored or stacked observations require numerical fallback"
    if metadata.get("counting_model_valid") is False:
        return "Detector preprocessing does not preserve the validated counting contract"
    q, y, count = item["q"], item["observed"], item["count"]
    if not 450 <= len(q) <= 700 or not 0.0005 <= q[0] <= 0.012 or not 4.0 <= q[-1] <= 4.3:
        return "Measured q coverage is outside the learned branch's validated profile"
    if np.any(y < 0) or not np.all(np.isfinite(count)) or np.any((count < 2.5) | (count > 12)):
        return "Intensity/count range is outside the learned branch's counting profile"
    return None


class BluePredictor:
    def __init__(self, root=None):
        self.root = Path(root or bundle_path())
        self.protocol = json.loads((self.root / "protocol.json").read_text(encoding="utf-8"))
        manifest = json.loads((self.root / "MANIFEST.json").read_text(encoding="utf-8"))
        if manifest["forward_version"] != FORWARD_VERSION:
            raise ValueError("Blue RC forward/model version mismatch")
        for name, checksum in manifest["files"].items():
            path = (self.root / name).resolve()
            if self.root.resolve() not in path.parents:
                raise ValueError("Invalid model manifest path")
            if hashlib.sha256(path.read_bytes()).hexdigest() != checksum:
                raise ValueError(f"Blue RC model checksum mismatch: {name}")
        self.model_id = manifest["model_id"]
        self.amplitude_tolerance = float(manifest["amplitude_relative_tolerance"])
        self.low, self.high = np.array(self.protocol["low"]), np.array(self.protocol["high"])
        self.edges = np.array(self.protocol["feature_edges"])
        with np.load(self.root / "feature_scaling.npz") as scale:
            self.mean, self.scale = scale["mean"], scale["scale"]
        if manifest.get("inference_backend") != "numpy_dense_swish_v1":
            raise ValueError("Unsupported blue model inference backend")
        widths = [len(self.mean), 256, 256, 128, 11]
        with np.load(self.root / "network.npz", allow_pickle=False) as arrays:
            self.layers = []
            for i in range(4):
                kernel, bias = arrays[f"kernel_{i}"], arrays[f"bias_{i}"]
                if kernel.shape != (widths[i], widths[i+1]) or bias.shape != (widths[i+1],):
                    raise ValueError("Neural layer shape mismatch")
                if not np.isfinite(kernel).all() or not np.isfinite(bias).all():
                    raise ValueError("Nonfinite neural weights")
                self.layers.append((kernel, bias))

    def predict(self, item):
        x = encode_features(item["q"], item["observed"], item["count"], self.edges)
        activation = ((x - self.mean) / self.scale)[None]
        for i, (kernel, bias) in enumerate(self.layers):
            activation = activation @ kernel + bias
            activation = activation * expit(activation) if i < 3 else expit(activation)
        u = activation[0]
        if not np.isfinite(u).all():
            raise ValueError("Nonfinite neural parameters")
        z = self.low + u * (self.high - self.low)
        r = float(np.exp(z[0]))
        lower = max(3.0, 2.002 * r)
        p = dict(
            R=r,
            sigma_R=float(z[1]),
            D=float(lower + (5.5 - lower) * z[2]),
            sigma_D=float(z[3]),
            h=float(np.exp(z[4])),
            sigma_h=float(z[5]),
        )
        cs = [
            dict(
                type="random_cylinder",
                type_id=2,
                weight=1.0,
                amplitude=float(np.exp(z[8])),
                params=p,
                structure_factor=True,
            )
        ]
        gs = dict(
            background=float(np.exp(z[9])),
            resolution_amplitude=float(np.exp(z[10])),
            sigma_Res=float(np.exp(z[6])),
            nu_Res=float(z[7]),
        )
        return cs, gs, u


def calibrate_amplitudes(
    q, y, count, components, globals_, relative_tolerance, *, return_prediction=False
):
    """One three-coefficient solve; all neural shape/instrument parameters fixed."""
    form = particle(q, components[0]["params"])
    shape = 1 / (1 + (q / globals_["sigma_Res"]) ** globals_["nu_Res"])
    basis = np.column_stack([form, np.ones_like(q), shape])
    initial = basis @ np.array(
        [components[0]["amplitude"], globals_["background"], globals_["resolution_amplitude"]]
    )
    sigma = np.sqrt(np.maximum(initial / count + (relative_tolerance * initial) ** 2, 1e-12))
    design = basis / sigma[:, None]
    scale = np.maximum(np.linalg.norm(design, axis=0), 1e-30)
    coefficients = nnls(design / scale, y / sigma)[0] / scale
    cs = [{**components[0], "amplitude": float(coefficients[0])}]
    gs = {
        **globals_,
        "background": float(coefficients[1]),
        "resolution_amplitude": float(coefficients[2]),
    }
    return (cs, gs, basis @ coefficients) if return_prediction else (cs, gs)


def route_diagnostics(item, prediction):
    y, q, count = item["observed"], item["q"], item["count"]
    # Routing tolerance, not a user quality-pass threshold or calibrated noise model.
    sigma = np.sqrt(np.maximum(prediction / count + (0.10 * prediction) ** 2, 1e-12))
    rms = float(np.sqrt(np.mean(((prediction - y) / sigma) ** 2)))
    biases = []
    for lo, hi in zip(np.linspace(q[0], q[-1], 9)[:-1], np.linspace(q[0], q[-1], 9)[1:]):
        use = (q >= lo) & (q <= hi)
        if use.sum() >= 8:
            observed = np.sum(y[use] * count[use])
            fitted = np.sum(prediction[use] * count[use])
            if observed <= 0 or fitted <= 0:
                biases.append(float("inf"))
            else:
                biases.append(float(abs(np.log(fitted / observed))))
    worst = max(biases, default=float("inf"))
    accepted = np.isfinite(rms) and rms <= 2.0 and worst <= 0.20
    return dict(
        accepted_for_fast_route=bool(accepted),
        count_relative_weighted_rms=rms,
        max_band_log_bias=worst if np.isfinite(worst) else None,
        policy="Routing only: RMS <= 2 with 10% working tolerance, eight-band absolute log bias <= 0.20; not a scientific quality certificate",
    )


def forward_row(q, row):
    if row.get("forward_version") == FORWARD_VERSION:
        return forward(q, row["components"], row["global_params"])
    from .experimental_fit import forward as legacy_forward

    return legacy_forward(q, row["components"], row["global_params"])


def _row(item, options, components, globals_, source, diagnostics, seconds, model_id, curve):
    q, y, sigma = item["q"], item["observed"], item["sigma"]
    residual = (curve - y) / sigma
    positive = y > 0
    render = np.linspace(q[0], q[-1], options["render_points"])
    nonparticle = globals_["background"] + globals_["resolution_amplitude"] / (
        1 + (q / globals_["sigma_Res"]) ** globals_["nu_Res"]
    )
    particle_fraction = float(np.max(np.maximum(curve - nonparticle, 0) / np.maximum(curve, 1e-30)))
    warnings = []
    if particle_fraction < 1e-6:
        warnings.append(
            "Particle contribution is numerically negligible; the returned RC dimensions "
            "are unconstrained and do not establish the presence of particles."
        )
    return dict(
        rank=1,
        workflow="native_v5",
        physics_backend="stable_blue_rc",
        forward_version=FORWARD_VERSION,
        model_id=model_id,
        side=item["side"],
        combination="random_cylinder",
        components=components,
        global_params=globals_,
        best_source=source,
        best_log_rmse=float(
            np.sqrt(np.mean(np.log(np.maximum(curve[positive], 1e-30) / y[positive]) ** 2))
        )
        if positive.any()
        else None,
        best_chi2_weighted=float(np.mean(residual**2)),
        signed_weighted_rms=float(np.sqrt(np.mean(residual**2))),
        signed_huber_delta2=float(
            np.mean(np.where(abs(residual) <= 2, 0.5 * residual**2, 2 * (abs(residual) - 1)))
        ),
        probability_status="Not calibrated",
        conditions_source="local_single_RC_candidate",
        algorithm="Neural shape and instrument parameters"
        + (
            " + one nonnegative three-amplitude calibration"
            if source == "stable_amplitude_calibrated"
            else ""
        ),
        nonlinear_refinement=False,
        linear_amplitude_calibration=source == "stable_amplitude_calibrated",
        validation_scope="Local single-RC candidate. A good curve does not identify unique components or parameters; no all-mode coverage.",
        unit_contract=dict(
            q="nm^-1",
            R_h_D="nm",
            sigma_R_h_D="relative standard deviation",
            sigma_Res="nm^-1",
            nu_Res="dimensionless",
            amplitudes="input intensity units",
        ),
        warnings=warnings,
        max_particle_intensity_fraction=particle_fraction,
        routing=diagnostics,
        seconds=seconds,
        nfev=0,
        converged=None,
        native_q=(q * item["sign"]).tolist(),
        observed=y.tolist(),
        sigma=sigma.tolist(),
        native_fit=curve.tolist(),
        display_q=(render * item["sign"]).tolist(),
        display_fit=forward(render, components, globals_).tolist(),
    )


class StableEngine:
    def __init__(self, root=None):
        self.root = root
        self.predictor = None
        self.load_failure = None

    def fit_and_write(self, output, item, options, report, cancelled):
        tick = time.perf_counter()
        if cancelled():
            raise RuntimeError("Prediction cancelled")
        reason = eligibility(item, options)
        rows = []
        if reason is None:
            try:
                if self.load_failure:
                    raise RuntimeError(self.load_failure)
                if self.predictor is None:
                    report(0, 1, "Loading fast CBF model…", {})
                    self.predictor = BluePredictor(self.root)
                if cancelled():
                    raise RuntimeError("Prediction cancelled")
                components, globals_, latent = self.predictor.predict(item)
                source = "stable_neural"
                if options["amplitude_calibration"]:
                    components, globals_, curve = calibrate_amplitudes(
                        item["q"],
                        item["observed"],
                        item["count"],
                        components,
                        globals_,
                        self.predictor.amplitude_tolerance,
                        return_prediction=True,
                    )
                    source = "stable_amplitude_calibrated"
                else:
                    curve = forward(item["q"], components, globals_)
                checks = route_diagnostics(item, curve)
                if checks["accepted_for_fast_route"]:
                    rows = [
                        _row(
                            item,
                            options,
                            components,
                            globals_,
                            source,
                            checks,
                            0,
                            self.predictor.model_id,
                            curve,
                        )
                    ]
                    rows[0]["seconds"] = time.perf_counter() - tick
                    rows[0]["normalized_neural_parameters"] = latent.tolist()
                else:
                    reason = "Fast candidate has structured residuals; numerical fallback requested"
            except Exception as exc:
                if cancelled():
                    raise RuntimeError("Prediction cancelled") from exc
                reason = f"Fast model unavailable: {type(exc).__name__}: {exc}"
                if self.predictor is None:
                    self.load_failure = reason
        if cancelled():
            raise RuntimeError("Prediction cancelled")
        if not rows:
            from .experimental_fit import fit_candidates

            report(0, 1, f"{item['side']}: {reason}", {})
            rows = fit_candidates(item, {**options, "method": "experimental"}, report, cancelled)
            for row in rows:
                row.update(
                    best_source="stable_numerical_fallback",
                    forward_version="legacy_v5_quadrature",
                    fallback_reason=reason,
                    nonlinear_refinement=True,
                    linear_amplitude_calibration=True,
                    routing=dict(accepted_for_fast_route=False, reason=reason),
                )
                row["warnings"].append(
                    "Broad fallback uses the legacy numerical forward; inspect residuals and parameter bounds. Local neural validation does not apply."
                )
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        (output / "solutions.json").write_text(
            json.dumps(
                dict(method="stable", options=options, solutions=rows), indent=2, allow_nan=False
            ),
            encoding="utf-8",
        )
        np.savez_compressed(
            output / "display.npz",
            reference_q=item["q"],
            observed=item["observed"],
            sigma=item["sigma"],
            render_q=np.abs(rows[0]["display_q"]),
            rendered_curves=[r["display_fit"] for r in rows],
        )
        np.savetxt(
            output / "fitting_curves.csv",
            np.column_stack([rows[0]["display_q"], *[r["display_fit"] for r in rows]]),
            delimiter=",",
            header="q_nm^-1," + ",".join(f"candidate_{i + 1}" for i in range(len(rows))),
            comments="",
        )
        return rows
