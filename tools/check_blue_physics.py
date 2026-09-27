"""Independent NumPy/TensorFlow forward, gradient and domain quadrature checks."""

import os

os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "4")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.blue_curve_distillation import forward, tf_forward, OUT


def main():
    import tensorflow as tf

    rng = np.random.default_rng(135711)
    u = rng.uniform(0.01, 0.99, (32, 11))
    q = np.r_[np.geomspace(0.001, 0.1, 12), np.linspace(0.11, 4.3, 100)]
    original = np.array([forward(q, a) for a in u])
    fine = np.array([forward(q, a, order=96, angles=192) for a in u])
    convergence = np.sqrt(np.mean(np.log(original / fine) ** 2, axis=1))
    actual = tf_forward(
        tf.constant(np.tile(q, (len(u), 1)), tf.float32), tf.constant(u, tf.float32)
    ).numpy()
    parity = np.sqrt(np.mean(np.log(actual / original) ** 2, axis=1))
    a = tf.Variable(u[:1], dtype=tf.float64)
    qq = tf.constant(q[None, :], dtype=tf.float64)
    with tf.GradientTape() as tape:
        loss = tf.reduce_mean(tf.math.log(tf_forward(qq, a)) ** 2)
    gradient = tape.gradient(loss, a).numpy()[0]
    eps = 1e-5
    finite = []
    for j in range(11):
        plus = u[0].copy()
        minus = u[0].copy()
        plus[j] += eps
        minus[j] -= eps
        finite.append(
            (np.mean(np.log(forward(q, plus)) ** 2) - np.mean(np.log(forward(q, minus)) ** 2))
            / (2 * eps)
        )
    np.testing.assert_allclose(gradient, finite, rtol=2e-4, atol=1e-5)
    assert convergence.max() < 0.002, convergence.max()
    assert parity.max() < 1e-4, parity.max()
    OUT.mkdir(exist_ok=True, parents=True)
    result = dict(
        random_domain_samples=32,
        quadrature_max_logrmse=float(convergence.max()),
        tf_numpy_max_logrmse=float(parity.max()),
        gradient_max_absolute_error=float(np.max(abs(gradient - np.array(finite)))),
        passed=True,
        note="Sampled local domain audit, not proof over the original universal parameter space.",
    )
    (OUT / "physics_checks.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
