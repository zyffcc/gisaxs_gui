"""cond_v2 proposal network (scale-free inputs, composition + mode conditioning, 8 hypotheses).

Inputs (all computed by features() from a curve, identical in training and deployment):
  points[B,Q,4] = (log(q/q_max), q/q_max, asinh(I/m), asinh(sigma/m)), m = median positive I,
                  on the network view (<=192 points, research.conditional_fit.pipeline.network_view)
  point_mask[B,Q], combo[B] (0..33), cond[B,3] = (qz/q_max, gisaxs flag, log10(q_max/q_min)/4)
Output u[B,H,32] in the per-curve codec research.cond_v2.codec.CurveCodec(q_min, q_max).
The TF graph mirrors numpy_net.NumpyCondV2 exactly (no TensorFlow needed for inference).
"""
from __future__ import annotations

import numpy as np

CONV = ((3, 1), (5, 1), (9, 1), (17, 1), (33, 1))


def features(q, observed, sigma, mask, q_min=None, q_max=None):
    """Network features for padded arrays [B,Q]; q_min/q_max default to the view's own range."""
    q, y, s, m = (np.asarray(v) for v in (q, observed, sigma, mask))
    order = np.argsort(~m, axis=1, kind='stable')
    take = lambda a: np.take_along_axis(a, order, axis=1)
    q, y, s, m = take(q), take(y), take(s), take(m)
    qmax = np.where(m, q, 0).max(1, keepdims=True) if q_max is None else np.asarray(q_max, float).reshape(-1, 1)
    safe_q = np.where(m, q, qmax)
    pos = np.where(m & (y > 0), y, np.nan)
    med = np.nanmedian(pos, axis=1, keepdims=True)
    med = np.where(np.isfinite(med) & (med > 0), med, 1.0)
    pts = np.stack([np.log(safe_q/qmax), safe_q/qmax, np.arcsinh(y/med), np.arcsinh(np.where(m, s, 0)/med)], -1)
    pts = np.where(m[..., None], pts, 0.0).astype('float32')
    n = int(m.sum(1).max())
    return pts[:, :n], m[:, :n]


def cond_vector(qz, gisaxs, q_min, q_max):
    qz, gisaxs, q_min, q_max = (np.asarray(v, float) for v in (qz, gisaxs, q_min, q_max))
    return np.stack([qz/q_max, gisaxs.astype(float), np.log10(q_max/q_min)/4], -1).astype('float32')


def build_model(heads=8, width=256, n_combos=34, embed=64, hidden=512, dropout=0.05, l2=1e-6):
    import tensorflow as tf
    from research.multishape_v2.new_model import ApplyPointMask, FeatureScaling, MaskedConv1D, MaskedMeanMax
    points = tf.keras.Input((None, 4), dtype='float32', name='points')
    mask = tf.keras.Input((None,), dtype='bool', name='point_mask')
    combo = tf.keras.Input((), dtype='int32', name='combo')
    cond = tf.keras.Input((3,), dtype='float32', name='cond')
    emb = tf.keras.layers.Embedding(n_combos, embed, name='composition_embedding')(combo)
    context = tf.keras.layers.Concatenate(name='context_in')([emb, cond])
    context = tf.keras.layers.Dense(128, activation='swish', name='context')(context)
    state = ApplyPointMask(name='input_mask')([points, mask])
    state = FeatureScaling(name='feature_scaling')(state)
    state = ApplyPointMask(name='scaled_mask')([state, mask])
    state = tf.keras.layers.Dense(width, activation='swish', name='point_projection')(state)
    for i, (kernel, _) in enumerate(CONV, 1):
        update = MaskedConv1D(width, kernel, l2=l2, name=f'conv_{i}')([state, mask])
        update = tf.keras.layers.LayerNormalization(name=f'norm_{i}')(update)
        film = tf.keras.layers.Dense(2*width, name=f'film_{i}')(context)
        gamma, beta = tf.split(film, 2, axis=-1)
        update = update*(1+gamma[:, None, :])+beta[:, None, :]
        update = tf.keras.layers.Dropout(dropout, name=f'dropout_{i}')(update)
        state = ApplyPointMask(name=f'mask_{i}')([state+update, mask])
    pooled = MaskedMeanMax(name='pool')([state, mask])
    x = tf.keras.layers.Concatenate(name='pooled_and_context')([pooled, context])
    for i in range(3):
        x = tf.keras.layers.Dense(hidden, activation='swish', name=f'head_hidden_{i}',
                                  kernel_regularizer=tf.keras.regularizers.l2(l2))(x)
    u = tf.keras.layers.Dense(heads*32, activation='sigmoid', name='parameter_logits')(x)
    u = tf.keras.layers.Reshape((heads, 32), name='u')(u)
    return tf.keras.Model({'points': points, 'point_mask': mask, 'combo': combo, 'cond': cond}, {'u': u}, name='cond_v2')
