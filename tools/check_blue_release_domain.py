"""Independent, untuned release TEST for the frozen guarded Blue RC branch."""

import os

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.stats import qmc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.gimap.features.fitting.application.workflow_v5 import bundled_workflow
from src.gimap.features.fitting.domain.blue_rc_forward import forward as production_forward
from src.gimap.features.fitting.infrastructure.adapters.stable_blue import (
    BluePredictor,
    calibrate_amplitudes,
    eligibility,
    encode_features,
    route_diagnostics,
)
from tools.blue_curve_distillation import AUDIT, OUT, EDGES, features, forward

DEST = ROOT / "validation/stable_blue_20260922"
SEED = 2026092202
# Current specialist policy requires an explicit single-RC prior. Historical
# release evidence predates this scope correction; it is not a composition test.
OPTIONS = dict(components=[2], sigma_res=None, nu_res=None)
META = dict(
    source="native_cbf_columns",
    intensity_unit="counts_per_pixel",
    threshold_enabled=False,
    mirror_replaced_pixels=0,
    stack_count=1,
    counting_model_valid=True,
)


def error(pred, target):
    return float(np.sqrt(np.mean(np.log(pred / target) ** 2)))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summary(values):
    a = np.array(values)
    return (
        dict(
            n=len(a),
            mean=float(a.mean()),
            median=float(np.median(a)),
            p90=float(np.quantile(a, 0.9)),
            max=float(a.max()),
        )
        if len(a)
        else dict(n=0)
    )


def immutable_weights():
    root = bundled_workflow()
    manifest = json.loads((root / "conditional_fast_manifest_v2.json").read_text())
    mismatches = [
        name for name, expected in manifest["files"].items() if sha(root / name) != expected
    ]
    copied = ROOT / "modules/Fitting_1D_Model/Workflow_v5/stable_blue_rc_v1/curve_best.weights.h5"
    original = OUT / "curve_best.weights.h5"
    return dict(
        legacy_count=len(manifest["files"]),
        legacy_mismatches=mismatches,
        stable_sha=sha(copied),
        pilot_sha=sha(original),
        identical=sha(copied) == sha(original),
    )


def one(predictor, q, y, count):
    item = dict(q=q, observed=y, count=count, observation_metadata=META.copy())
    tick = time.perf_counter()
    cs, gs, u = predictor.predict(item)
    raw = production_forward(q, cs, gs)
    ac, ag = calibrate_amplitudes(q, y, count, cs, gs, 0.10)
    calibrated = production_forward(q, ac, ag)
    seconds = time.perf_counter() - tick
    # Scientific invariants: same particle and resolution shape, changed amplitudes only.
    assert cs[0]["params"] == ac[0]["params"]
    assert gs["sigma_Res"] == ag["sigma_Res"] and gs["nu_Res"] == ag["nu_Res"]
    pilot_feature = features(q, y, count)
    runtime_feature = encode_features(q, y, count, EDGES)
    np.testing.assert_array_equal(runtime_feature, pilot_feature)
    pilot_curve = forward(q, u)
    np.testing.assert_allclose(raw, pilot_curve, rtol=1e-12, atol=1e-10)
    return (
        item,
        cs,
        gs,
        ac,
        ag,
        raw,
        calibrated,
        dict(
            normalized_parameters=u.tolist(),
            warm_seconds=seconds,
            forward_relative_difference=float(np.max(abs(raw / pilot_curve - 1))),
            feature_exact=True,
        ),
    )


def run():
    DEST.mkdir(exist_ok=True, parents=True)
    protocol = dict(
        samples=128,
        seed=SEED,
        noise_seed=SEED + 101,
        declared_before_evaluation=True,
        selection="NONE: weights, 10% amplitude tolerance and routing thresholds frozen before this TEST",
        scoring="Clean lnRMSE on complete native q template, even at randomly masked columns",
        routing="Eligibility and diagnostics use only retained measured points; neither is a parameter-truth certificate",
        stress="12 predetermined modified curves, no threshold selection or tuning",
    )
    (DEST / "release_domain.protocol.json").write_text(json.dumps(protocol, indent=2))
    initial_hashes = immutable_weights()
    assert not initial_hashes["legacy_mismatches"] and initial_hashes["identical"]
    predictor = BluePredictor()
    assert predictor.amplitude_tolerance == 0.10
    templates = json.loads((AUDIT / "inputs.json").read_text())
    latent = qmc.LatinHypercube(11, seed=SEED).random(128)
    rng = np.random.default_rng(SEED + 101)
    records = []
    for i, u in enumerate(latent):
        template = templates[("positive", "negative")[i % 2]]
        q = np.array(template["q"])
        count = np.array(template["count"]) * np.exp(rng.uniform(np.log(0.5), np.log(2)))
        clean = forward(q, u)
        noisy = rng.poisson(clean * count) / count
        use = rng.random(len(q)) > 0.025
        item, cs, gs, ac, ag, raw, calibrated, detail = one(
            predictor, q[use], noisy[use], count[use]
        )
        raw_full = production_forward(q, cs, gs)
        cal_full = production_forward(q, ac, ag)
        diag = route_diagnostics(item, calibrated)
        reason = eligibility(item, OPTIONS)
        records.append(
            dict(
                index=i,
                clean_raw=error(raw_full, clean),
                clean_calibrated=error(cal_full, clean),
                eligible=reason is None,
                eligibility_reason=reason,
                diagnostics=diag,
                routed_fast=reason is None and diag["accepted_for_fast_route"],
                **detail,
            )
        )
        if (i + 1) % 32 == 0:
            print(json.dumps(dict(completed=i + 1)), flush=True)
    accepted = [r for r in records if r["routed_fast"]]
    diagnostics_accepted = [r for r in records if r["diagnostics"]["accepted_for_fast_route"]]
    stats = dict(
        raw=summary([r["clean_raw"] for r in records]),
        calibrated=summary([r["clean_calibrated"] for r in records]),
        eligible_count=sum(r["eligible"] for r in records),
        diagnostics_accepted_count=len(diagnostics_accepted),
        routed_count=len(accepted),
        routed_fraction=len(accepted) / len(records),
        accepted_clean=summary([r["clean_calibrated"] for r in accepted]),
        accepted_error_gt_015=sum(r["clean_calibrated"] > 0.15 for r in accepted),
        accepted_error_gt_020=sum(r["clean_calibrated"] > 0.20 for r in accepted),
        accepted_fraction_error_gt_015=float(
            np.mean([r["clean_calibrated"] > 0.15 for r in accepted])
        )
        if accepted
        else None,
        accepted_fraction_error_gt_020=float(
            np.mean([r["clean_calibrated"] > 0.20 for r in accepted])
        )
        if accepted
        else None,
        calibration_improved_fraction=float(
            np.mean([r["clean_calibrated"] < r["clean_raw"] for r in records])
        ),
        warm_median_seconds=float(np.median([r["warm_seconds"] for r in records[1:]])),
        max_forward_relative_difference=max(r["forward_relative_difference"] for r in records),
    )
    print(json.dumps(stats), flush=True)
    real = []
    references = json.loads((OUT / "real_references.json").read_text())
    old = json.loads((OUT / "evaluation.json").read_text())
    for reference in references:
        q, y, count = [np.array(reference[k]) for k in ("q", "y", "count")]
        item, cs, gs, ac, ag, raw, calibrated, detail = one(predictor, q, y, count)
        earlier = next(
            r
            for r in old["real"]
            if r["frame"] == reference["frame"]
            and r["side"] == reference["side"]
            and r["model"] == "curve_supervised"
        )
        np.testing.assert_allclose(raw, earlier["forward_curve"], rtol=1e-5)
        real.append(
            dict(
                frame=reference["frame"],
                side=reference["side"],
                neural_vs_previous_max_relative_difference=float(
                    np.max(abs(raw / np.array(earlier["forward_curve"]) - 1))
                ),
                calibrated_reference_error=error(calibrated, np.array(reference["reference"])),
                eligible_reason=eligibility(item, OPTIONS),
                diagnostics=route_diagnostics(item, calibrated),
                **detail,
            )
        )
    # Exact eligibility stress, no numerical fallback optimization invoked here.
    base = dict(
        q=np.array(references[0]["q"]),
        observed=np.array(references[0]["y"]),
        count=np.array(references[0]["count"]),
        observation_metadata=META.copy(),
    )
    cases = []
    for label in (
        "q_scale_10x",
        "missing_count",
        "short_q",
        "negative_intensity",
        "threshold",
        "mirror",
        "stack",
        "fixed_components",
        "fixed_sigma_res",
        "fixed_nu_res",
    ):
        item, options = deepcopy(base), deepcopy(OPTIONS)
        if label == "q_scale_10x":
            item["q"] *= 10
        elif label == "missing_count":
            del item["count"]
        elif label == "short_q":
            for key in ("q", "observed", "count"):
                item[key] = item[key][:100]
        elif label == "negative_intensity":
            item["observed"][50] = -1
        elif label == "threshold":
            item["observation_metadata"]["threshold_enabled"] = True
        elif label == "mirror":
            item["observation_metadata"]["mirror_replaced_pixels"] = 1
        elif label == "stack":
            item["observation_metadata"]["stack_count"] = 2
        elif label == "fixed_components":
            options["components"] = [1, 2]
        elif label == "fixed_sigma_res":
            options["sigma_res"] = 0.02
        elif label == "fixed_nu_res":
            options["nu_res"] = 3.0
        reason = eligibility(item, options)
        assert reason is not None, label
        cases.append(dict(case=label, fallback=True, reason=reason))
    stress = []
    q, count = base["q"], base["count"]
    clean = forward(q, np.full(11, 0.5))
    for kind in ("extra_peak", "tilted_tail", "oscillation", "effective_count_reduced"):
        for strength in (0.3, 0.8, 1.5):
            altered, exposure = clean.copy(), count.copy()
            if kind == "extra_peak":
                altered *= 1 + strength * np.exp(-0.5 * ((q - 2.4) / 0.12) ** 2)
            elif kind == "tilted_tail":
                altered *= np.exp(strength * (q / q.max()) ** 2)
            elif kind == "oscillation":
                altered *= np.exp(strength * np.sin(7 * q))
            elif kind == "effective_count_reduced":
                exposure = count / (1 + 10 * strength)
            y = rng.poisson(altered * exposure) / exposure
            supplied_count = count if kind == "effective_count_reduced" else exposure
            item, cs, gs, ac, ag, raw, calibrated, detail = one(predictor, q, y, supplied_count)
            stress.append(
                dict(
                    kind=kind,
                    strength=strength,
                    clean_error=error(calibrated, altered),
                    diagnostics=route_diagnostics(item, calibrated),
                    eligibility_reason=eligibility(item, OPTIONS),
                )
            )
    final_hashes = immutable_weights()
    assert initial_hashes == final_hashes
    result = dict(
        protocol=protocol,
        statistics=stats,
        records=records,
        real=real,
        eligibility_stress=cases,
        shape_noise_stress=stress,
        hashes=final_hashes,
        failures=0,
        limitations="Local synthetic TEST shares declared physics/domain; unknown experimental sources and multi-component truths remain unvalidated; routing does not identify all OOD curves.",
    )
    (DEST / "release_domain.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    report(result)


def report(r):
    s = r["statistics"]
    lines = [
        "# Independent frozen release TEST",
        "",
        "128 new synthetic TEST samples; seed 2026092202. No training, tolerance choice or routing threshold tuning used this set.",
        "",
        "| Prediction | Mean clean lnRMSE | Median | P90 | Maximum |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ("raw", "calibrated", "accepted_clean"):
        d = s[name]
        lines.append(
            f"| {name} | {d['mean']:.5f} | {d['median']:.5f} | {d['p90']:.5f} | {d['max']:.5f} |"
        )
    lines += [
        "",
        f"Eligible {s['eligible_count']}/128; residual diagnostic acceptance {s['diagnostics_accepted_count']}/128; actual combined fast route {s['routed_count']}/128 ({s['routed_fraction']:.1%}).",
        f"Accepted clean error >0.15: {s['accepted_error_gt_015']} ({s['accepted_fraction_error_gt_015']:.1%}); >0.20: {s['accepted_error_gt_020']} ({s['accepted_fraction_error_gt_020']:.1%}).",
        "",
        "Routing diagnostics are working tolerances, not a guarantee about clean curve quality or true components/parameters.",
        "",
        f"Calibration improves {s['calibration_improved_fraction']:.1%} of TEST curves. Runtime/pilot feature arrays identical; max relative forward difference {s['max_forward_relative_difference']:.3g}. Six experimental neural curves reproduce earlier pilot outputs to rtol 1e-5. Frozen legacy manifest and copied neural weight hashes match.",
        "",
        "| Eligibility stress | Fallback |",
        "|---|---|",
    ]
    lines += [f"| {d['case']} | {d['fallback']} |" for d in r["eligibility_stress"]]
    lines += [
        "",
        "| Alteration | Strength | Clean error | Residual gate accepted |",
        "|---|---:|---:|---|",
    ]
    lines += [
        f"| {d['kind']} | {d['strength']} | {d['clean_error']:.5f} | {d['diagnostics']['accepted_for_fast_route']} |"
        for d in r["shape_noise_stress"]
    ]
    lines += [
        "",
        r["limitations"],
        "",
        "Failures: 0. This test does not fit numerical fallback candidates; GUI end-to-end fallback validation is a separate task.",
    ]
    parity_path = DEST / "release_domain.numpy_parity.json"
    if parity_path.exists():
        parity = json.loads(parity_path.read_text())
        worst = max(d["calibrated_relative_difference"] for d in parity["curve_checks"])
        lines += [
            "",
            "## TensorFlow to NumPy inference parity",
            "",
            f"Replayed all 128 identical noisy TEST inputs after inference-only conversion. Maximum normalized-parameter difference {parity['max_u_absolute_difference']:.3g}; four predetermined calibrated-curve relative differences at most {worst:.3g}; their routing decisions unchanged. No TensorFlow import in the replay. The copied H5 and 73 frozen original assets remain identical.",
            "",
            "The release statistics above were collected before conversion; this parity check does not retrain, retune or select samples. Detailed record: release_domain.numpy_parity.json.",
        ]
    (DEST / "release_domain.md").write_text("\n".join(lines), encoding="utf-8")


def numpy_parity():
    """Replay identical TEST observations after inference-only implementation change."""
    from tools.probe_blue_calibration import basis_from_prediction, calibrate

    previous = json.loads((DEST / "release_domain.json").read_text())
    predictor = BluePredictor()
    templates = json.loads((AUDIT / "inputs.json").read_text())
    latent = qmc.LatinHypercube(11, seed=SEED).random(128)
    rng = np.random.default_rng(SEED + 101)
    values, curves = [], []
    for i, u in enumerate(latent):
        template = templates[("positive", "negative")[i % 2]]
        q = np.array(template["q"])
        count = np.array(template["count"]) * np.exp(rng.uniform(np.log(0.5), np.log(2)))
        clean = forward(q, u)
        noisy = rng.poisson(clean * count) / count
        use = rng.random(len(q)) > 0.025
        item = dict(
            q=q[use], observed=noisy[use], count=count[use], observation_metadata=META.copy()
        )
        cs, gs, new_u = predictor.predict(item)
        old_u = np.array(previous["records"][i]["normalized_parameters"])
        difference = float(np.max(abs(new_u - old_u)))
        assert difference < 2e-6, (i, difference)
        values.append(difference)
        if i in (0, 32, 64, 96):
            old_raw, old_basis, _ = basis_from_prediction(q[use], old_u)
            old_cal, _ = calibrate(old_basis, old_raw, noisy[use], count[use], 0.10)
            new_raw = production_forward(q[use], cs, gs)
            ac, ag = calibrate_amplitudes(q[use], noisy[use], count[use], cs, gs, 0.10)
            new_cal = production_forward(q[use], ac, ag)
            relative_raw = float(np.max(abs(new_raw / old_raw - 1)))
            relative_cal = float(np.max(abs(new_cal / old_cal - 1)))
            assert max(relative_raw, relative_cal) < 2e-5
            old_gate = route_diagnostics(item, old_cal)
            new_gate = route_diagnostics(item, new_cal)
            assert old_gate["accepted_for_fast_route"] == new_gate["accepted_for_fast_route"]
            curves.append(
                dict(
                    index=i,
                    raw_relative_difference=relative_raw,
                    calibrated_relative_difference=relative_cal,
                    same_route=True,
                )
            )
    payload = dict(
        samples=128,
        source="Replay of frozen release TEST, no tuning",
        max_u_absolute_difference=max(values),
        median_u_absolute_difference=float(np.median(values)),
        curve_checks=curves,
        hashes=immutable_weights(),
        tensorflow_imported="tensorflow" in sys.modules,
        note="An inference implementation parity check; release TEST statistics remain those recorded before conversion.",
    )
    assert not payload["tensorflow_imported"]
    (DEST / "release_domain.numpy_parity.json").write_text(json.dumps(payload, indent=2))
    report(previous)
    print(json.dumps(payload), flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--numpy-parity", action="store_true")
    if parser.parse_args().numpy_parity:
        numpy_parity()
    else:
        run()
