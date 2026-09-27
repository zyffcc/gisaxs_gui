"""Explicit experimental single-RC predictor, no numerical optimization.

Input NPZ must contain q (nm^-1), intensity (counts / pixel), and count
(effective valid-pixel exposure per point). This is the CBF counting contract;
arbitrary normalized curves and unknown geometry are outside the pilot scope.
"""

import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.blue_curve_distillation import OUT, VERSION, build_model, decode, features, forward


def predict_file(input_path, output_path, model_dir=OUT):
    arrays = np.load(input_path)
    q, y, count = [np.asarray(arrays[k], float) for k in ("q", "intensity", "count")]
    if q.ndim != 1 or q.shape != y.shape or q.shape != count.shape:
        raise ValueError("q, intensity and count must be equal-length 1D arrays")
    if (
        not np.isfinite(q).all()
        or not np.isfinite(y).all()
        or not np.isfinite(count).all()
        or np.any(count <= 0)
    ):
        raise ValueError(
            "Provide only valid measured points with finite values and positive pixel exposures"
        )
    if np.any(y < 0) or np.any(abs(q) > 4.3):
        raise ValueError(
            "This counting-data pilot supports nonnegative intensities and |q| <= 4.3 nm^-1"
        )
    model_dir = Path(model_dir)
    protocol = json.loads((model_dir / "protocol.json").read_text())
    if protocol["version"] != VERSION:
        raise ValueError("Incompatible physics/model version")
    sc = np.load(model_dir / "feature_scaling.npz")
    net = build_model()
    net.load_weights(str(model_dir / "curve_best.weights.h5"))
    results = []
    for side, sign in [("positive", 1), ("negative", -1)]:
        idx = np.flatnonzero(q * sign > 0)
        if not len(idx):
            continue
        idx = idx[np.argsort(abs(q[idx]))]
        qq = abs(q[idx])
        yy = y[idx]
        cc = count[idx]
        if len(idx) < 100 or qq.max() < 3:
            raise ValueError(
                "Pilot requires a broadly sampled CBF cut; sparse/narrow-q inputs are not validated"
            )
        tick = time.perf_counter()
        x = (features(qq, yy, cc) - sc["mean"]) / sc["scale"]
        u = net(x[None], training=False).numpy()[0]
        curve = forward(qq, u)
        positive = yy > 0
        results.append(
            dict(
                side=side,
                parameters=decode(u),
                normalized_parameters=u.tolist(),
                q=qq.tolist(),
                observed=yy.tolist(),
                forward_curve=curve.tolist(),
                observed_logrmse=float(
                    np.sqrt(np.mean(np.log(curve[positive] / yy[positive]) ** 2))
                ),
                prediction_seconds=time.perf_counter() - tick,
                refinement=False,
            )
        )
    payload = dict(
        version=VERSION,
        scope=protocol["limitations"],
        results=results,
        units=dict(
            q="nm^-1",
            R="nm",
            D="nm",
            h="nm",
            sigma_res="nm^-1",
            sigma_R="relative standard deviation",
            sigma_h="relative standard deviation",
            sigma_D="relative standard deviation",
            A="counts per pixel multiplier of normalized particle form",
            B="counts per pixel",
            C="counts per pixel",
        ),
        quality_note="No mandatory observed-data cutoff. Review shape and residuals. One candidate per side; no probability or unique-parameter claim.",
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        json.dumps(dict(output=str(output_path), sides=len(results), refinement=False)), flush=True
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("input", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--model-dir", type=Path, default=OUT)
    a = p.parse_args()
    predict_file(a.input, a.output, a.model_dir)
