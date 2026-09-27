"""Versioned local inverse pilot: physical parameters, clean curve supervision.

This single-random-cylinder branch is NOT the universal multi-component model.
No experimental curve or fitted parameter is a training sample. Its declared
domain was chosen using CBF 00033, which therefore remains a development case.
"""

import os

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "4")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys
import time
import numpy as np
from scipy.stats import qmc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.audit_gauss_forward import component

OUT = ROOT / "validation/blue_curve_distill_20260921"
AUDIT = ROOT / "validation/cause_audit_20260921"
NAMES = [
    "log_R",
    "sigma_R",
    "distance_fraction",
    "sigma_D",
    "log_h",
    "sigma_h",
    "log_sigma_res",
    "nu_res",
    "log_particle_amplitude",
    "log_background",
    "log_resolution_amplitude",
]
LO = np.array(
    [
        np.log(1.1),
        0.12,
        0.0,
        0.075,
        np.log(6),
        0.08,
        np.log(0.009),
        2.0,
        np.log(500),
        np.log(0.15),
        np.log(50000),
    ]
)
HI = np.array(
    [
        np.log(2.2),
        0.4,
        1.0,
        0.2,
        np.log(24),
        0.35,
        np.log(0.026),
        4.0,
        np.log(7000),
        np.log(4),
        np.log(900000),
    ]
)
EDGES = np.r_[0.0, np.geomspace(0.006, 0.12, 21), np.linspace(0.15, 4.3, 139)]
CENTERS = (EDGES[:-1] + EDGES[1:]) / 2
VERSION = "cbf_rc_gauss48_96_independent_amplitudes_v1"


def decode(u):
    z = LO + np.asarray(u) * (HI - LO)
    r = np.exp(z[0])
    lower = max(3.0, 2.002 * r)
    return dict(
        R=r,
        sigma_R=z[1],
        D=lower + (5.5 - lower) * z[2],
        sigma_D=z[3],
        h=np.exp(z[4]),
        sigma_h=z[5],
        sigma_res=np.exp(z[6]),
        nu_res=z[7],
        A=np.exp(z[8]),
        B=np.exp(z[9]),
        C=np.exp(z[10]),
    )


def forward(q, u, order=48, angles=96):
    p = decode(u)
    form = component(
        np.asarray(q),
        2,
        p["R"],
        p["sigma_R"],
        p["h"],
        p["sigma_h"],
        p["D"],
        p["sigma_D"],
        True,
        order=order,
        orientation_order=angles,
    )
    return p["A"] * form + p["B"] + p["C"] / (1 + (q / p["sigma_res"]) ** p["nu_res"])


def features(q, y, count):
    """Native pixel/count weighted bins, with explicit missing-bin indicators."""
    q, y, count = map(np.asarray, (q, y, count))
    idx = np.searchsorted(EDGES, q, side="right") - 1
    valid = np.isfinite(y) & (count > 0) & (idx >= 0) & (idx < len(CENTERS))
    idx, q, y, count = idx[valid], q[valid], y[valid], count[valid]
    n = len(CENTERS)
    exposure = np.bincount(idx, weights=count, minlength=n)
    total = np.bincount(idx, weights=y * count, minlength=n)
    qsum = np.bincount(idx, weights=q * count, minlength=n)
    present = exposure > 0
    means = total / np.maximum(exposure, 1)
    log_y = np.where(present, np.log(np.maximum(means, 0.1 / np.maximum(exposure, 1))), 0.0)
    q_offset = np.where(present, (qsum / np.maximum(exposure, 1) - CENTERS) / np.diff(EDGES), 0.0)
    return np.r_[log_y, present, q_offset, np.log1p(exposure)].astype("float32")


def generate_one(job):
    index, u, template = job
    q, count = map(np.asarray, template)
    clean = forward(q, u)
    rng = np.random.default_rng(88291 + index)
    exposure = count * np.exp(rng.uniform(np.log(0.5), np.log(2)))
    observed = rng.poisson(clean * exposure) / exposure
    # Independent masked columns, never replaced by interpolated observations.
    exposure = exposure * (rng.random(len(q)) > 0.025)
    return features(q, observed, exposure), clean.astype("float32")


def generate(workers, ntrain=4096, nval=512, ntest=512):
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "dataset.npz").exists():
        return
    inputs = json.loads((AUDIT / "inputs.json").read_text())
    templates = [(inputs[s]["q"], inputs[s]["count"]) for s in ("positive", "negative")]
    # Independent Latin hypercube draws for train/validation/test, immutable split.
    u = np.vstack(
        [
            qmc.LatinHypercube(11, seed=seed).random(n)
            for seed, n in [(5201, ntrain), (6201, nval), (7201, ntest)]
        ]
    ).astype("float32")
    jobs = [(i, a, templates[i % 2]) for i, a in enumerate(u)]
    maxn = max(len(t[0]) for t in templates)
    clean = np.ones((len(u), maxn), "float32")
    q = np.zeros_like(clean)
    mask = np.zeros_like(clean)
    xx = []
    tick = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, (feat, yy) in enumerate(pool.map(generate_one, jobs, chunksize=8)):
            xx.append(feat)
            clean[i, : len(yy)] = yy
            qq = templates[i % 2][0]
            q[i, : len(qq)] = qq
            mask[i, : len(qq)] = 1
            if (i + 1) % 512 == 0:
                print(
                    json.dumps(
                        dict(
                            generated=i + 1,
                            total=len(u),
                            seconds=round(time.perf_counter() - tick, 1),
                        )
                    ),
                    flush=True,
                )
    np.savez_compressed(
        OUT / "dataset.npz",
        x=np.array(xx),
        u=u,
        clean=clean,
        q=q,
        mask=mask,
        ntrain=ntrain,
        nval=nval,
        ntest=ntest,
    )
    (OUT / "protocol.json").write_text(
        json.dumps(
            dict(
                version=VERSION,
                parameters=NAMES,
                low=LO.tolist(),
                high=HI.tolist(),
                ntrain=ntrain,
                nval=nval,
                ntest=ntest,
                split_seeds=[5201, 6201, 7201],
                feature_edges=EDGES.tolist(),
                geometry="Same native q templates and masking geometry as CBF 00033; local specialist only.",
                target="Noise-free converged forward curves AND physical parameter labels; raw noise is input augmentation only.",
                development_frame="00033 used to choose domain, never an independent generalization test.",
                heldout_frames=["00005", "00045"],
                limitations="Single random cylinder with active structure factor; no component classification or calibrated posterior.",
                seconds=time.perf_counter() - tick,
            ),
            indent=2,
        ),
        encoding="utf-8",
    )


def tf_forward(q, u, order=48, angles=96):
    import tensorflow as tf

    dtype = u.dtype
    z = tf.constant(LO, dtype=dtype) + u * tf.constant(HI - LO, dtype=dtype)
    r, sr, eta, sd, h, sh, rs, nu, amp, bg, resamp = tf.unstack(z, axis=-1)
    r, h, rs, amp, bg, resamp = [tf.exp(v) for v in (r, h, rs, amp, bg, resamp)]
    lower = tf.maximum(3.0, 2.002 * r)
    dist = lower + (5.5 - lower) * eta
    xn, wn = np.polynomial.legendre.leggauss(order)
    xa, wa = np.polynomial.legendre.leggauss(angles)
    xn, wn = tf.constant(xn, dtype), tf.constant(wn, dtype)
    orientation = tf.constant((xa + 1) / 2, dtype)
    ow = tf.constant(wa / 2, dtype)

    def gaussian(mu, width):
        sigma = mu * width
        lo = tf.maximum(mu - 4 * sigma, 0.0)
        hi = mu + 4 * sigma
        nodes = lo[:, None] + (xn[None, :] + 1) * (hi - lo)[:, None] / 2
        weights = wn[None, :] * tf.exp(-0.5 * ((nodes - mu[:, None]) / sigma[:, None]) ** 2)
        return nodes, weights / tf.reduce_sum(weights, axis=1, keepdims=True)

    rr, wr = gaussian(r, sr)
    hh, wh = gaussian(h, sh)
    xx = (
        rr[:, :, None, None]
        * tf.sqrt(1 - orientation**2)[None, None, :, None]
        * q[:, None, None, :]
    )
    safe = tf.where(abs(xx) < 1e-3, tf.ones_like(xx), xx)
    radial = tf.where(
        abs(xx) < 1e-3, 1 - xx**2 / 8 + xx**4 / 192, 2 * tf.math.special.bessel_j1(safe) / safe
    )
    fm = tf.reduce_sum(wr[:, :, None, None] * radial**2, axis=1)
    xx = hh[:, :, None, None] * orientation[None, None, :, None] * q[:, None, None, :] / 2
    safe = tf.where(abs(xx) < 1e-3, tf.ones_like(xx), xx)
    sinc = tf.where(abs(xx) < 1e-3, 1 - xx**2 / 6 + xx**4 / 120, tf.sin(safe) / safe)
    hm = tf.reduce_sum(wh[:, :, None, None] * sinc**2, axis=1)
    form = tf.reduce_sum(ow[None, :, None] * fm * hm, axis=1)
    lp = -np.pi * q * q * (dist * sd)[:, None] ** 2
    phi = tf.exp(lp)
    sf = -tf.math.expm1(2 * lp) / tf.maximum(
        tf.math.expm1(lp) ** 2 + 4 * phi * tf.sin(0.5 * q * dist[:, None]) ** 2, 1e-15
    )
    return (
        amp[:, None] * form * sf
        + bg[:, None]
        + resamp[:, None] / (1 + (q / rs[:, None]) ** nu[:, None])
    )


def build_model():
    import tensorflow as tf

    return tf.keras.Sequential(
        [
            tf.keras.layers.Input((len(CENTERS) * 4,)),
            tf.keras.layers.Dense(256, activation="swish"),
            tf.keras.layers.Dense(256, activation="swish"),
            tf.keras.layers.Dense(128, activation="swish"),
            tf.keras.layers.Dense(11, activation="sigmoid"),
        ]
    )


def train(steps):
    import tensorflow as tf

    tf.keras.utils.set_random_seed(81021)
    data = np.load(OUT / "dataset.npz")
    n = int(data["ntrain"])
    nv = int(data["nval"])
    mean = data["x"][:n].mean(0)
    scale = np.maximum(data["x"][:n].std(0), 0.05)
    x = (data["x"] - mean) / scale
    net = build_model()
    net.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss="mse")
    tick = time.perf_counter()
    net.fit(
        x[:n],
        data["u"][:n],
        validation_data=(x[n : n + nv], data["u"][n : n + nv]),
        epochs=100,
        batch_size=64,
        verbose=0,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=12, restore_best_weights=True)],
    )
    net.save_weights(str(OUT / "parameter_only.weights.h5"))
    np.savez(OUT / "feature_scaling.npz", mean=mean, scale=scale)
    print(
        json.dumps(
            dict(stage="parameter_pretraining", seconds=round(time.perf_counter() - tick, 2))
        ),
        flush=True,
    )
    optimizer = tf.keras.optimizers.Adam(0.00015)

    @tf.function
    def step(xx, uu, qq, yy):
        with tf.GradientTape() as tape:
            pred = net(xx, training=True)
            curve = tf_forward(qq, pred)
            loss_curve = tf.reduce_mean((tf.math.log(curve) - tf.math.log(yy)) ** 2)
            loss_param = tf.reduce_mean((pred - uu) ** 2)
            loss = loss_curve + 0.002 * loss_param
        grads = tape.gradient(loss, net.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 5.0)
        optimizer.apply_gradients(zip(grads, net.trainable_variables))
        return loss_curve

    rng = np.random.default_rng(9121)
    # Fixed validation subset chosen before training; test and real frames never select weights.
    vi = np.arange(n, n + min(64, nv))
    qi = np.linspace(0, 570, 64, dtype=int)

    def validate():
        uhat = net(x[vi], training=False).numpy()
        errors = []
        for j, i in enumerate(vi):
            y = forward(data["q"][i, qi], uhat[j])
            errors.append(np.sqrt(np.mean(np.log(y / data["clean"][i, qi]) ** 2)))
        return float(np.mean(errors)), float(np.median(errors))

    best, median = validate()
    history = [dict(step=0, validation_mean=best, validation_median=median)]
    net.save_weights(str(OUT / "curve_best.weights.h5"))
    print(json.dumps(history[-1]), flush=True)
    for k in range(1, steps + 1):
        ix = rng.integers(n, size=8)
        # Uniform sampling across native q preserves the full-curve log objective.
        lengths = data["mask"][ix].sum(1).astype(int)
        qi_train = (rng.random((8, 32)) * lengths[:, None]).astype(int)
        qq = data["q"][ix[:, None], qi_train]
        yy = data["clean"][ix[:, None], qi_train]
        loss = float(step(x[ix], data["u"][ix], qq, yy))
        if k % 100 == 0:
            score, median = validate()
            history.append(
                dict(
                    step=k,
                    loss=loss,
                    validation_mean=score,
                    validation_median=median,
                    seconds=time.perf_counter() - tick,
                )
            )
            if score < best:
                best = score
                net.save_weights(str(OUT / "curve_best.weights.h5"))
            (OUT / "training_history.json").write_text(json.dumps(history, indent=2))
            print(json.dumps(history[-1]), flush=True)
    net.save_weights(str(OUT / "curve_last.weights.h5"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("phase", choices=["generate", "train"])
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--steps", type=int, default=1600)
    a = p.parse_args()
    if a.phase == "generate":
        generate(a.workers)
    else:
        train(a.steps)


if __name__ == "__main__":
    main()
