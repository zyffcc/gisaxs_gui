"""TensorFlow-free inference for cond_v2 (mirrors model.build_model).

export_weights() (run where TensorFlow is installed) writes weights_numpy.npz next to
best.weights.h5; NumpyCondV2 then runs anywhere with numpy. verify() compares both.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from research.cond_v2.model import CONV


def _swish(x):
    return x/(1+np.exp(-x))


class NumpyCondV2:
    def __init__(self, model_dir):
        model_dir = Path(model_dir)
        self.config = json.loads((model_dir/'model_config.json').read_text())
        with np.load(model_dir/'weights_numpy.npz') as f:
            # C-contiguous copies: a file written from transposed views (PyTorch export before 2026-10-06) loads
            # as Fortran-ordered arrays, and kernel[j] of such a kernel is a strided view that numpy multiplies
            # without BLAS (measured: 1.25 s instead of 7 ms per convolution)
            self.w = {k: np.ascontiguousarray(f[k], dtype=np.float64) for k in f.files}
        self.heads = int(self.config.get('heads', 8))

    def _get(self, layer, index):
        return self.w[f'{layer}/{index}']

    def _dense(self, x, layer, activation=None):
        y = x@self._get(layer, 0)+self._get(layer, 1)
        if activation == 'swish':
            return _swish(y)
        if activation == 'sigmoid':
            return 1/(1+np.exp(-y))
        return y

    def _masked_conv(self, x, mask, layer):
        a, b = self._get(layer, 0), self._get(layer, 1)
        kernel, bias = (a, b) if a.ndim == 3 else (b, a)
        k = kernel.shape[0]
        left = (k-1)//2
        safe = np.where(mask[..., None], x, 0.0)
        pad = np.pad(safe, ((0, 0), (left, k-1-left), (0, 0)))
        m = np.pad(mask.astype(float), ((0, 0), (left, k-1-left)))
        n = x.shape[1]
        out = np.zeros((x.shape[0], n, kernel.shape[2]))
        support = np.zeros((x.shape[0], n))
        for j in range(k):
            out += pad[:, j:j+n, :]@kernel[j]
            support += m[:, j:j+n]
        values = out*k/np.maximum(support, 1.0)[..., None]+bias
        return np.where(mask[..., None], _swish(values), 0.0)

    def _layer_norm(self, x, layer, eps=1e-3):
        gamma, beta = self._get(layer, 0), self._get(layer, 1)
        return (x-x.mean(-1, keepdims=True))/np.sqrt(x.var(-1, keepdims=True)+eps)*gamma+beta

    def __call__(self, points, mask, combo, cond):
        points, mask = np.asarray(points, float), np.asarray(mask, bool)
        emb = self._get('composition_embedding', 0)[np.asarray(combo, int)]
        context = self._dense(np.concatenate([emb, np.asarray(cond, float)], -1), 'context', 'swish')
        x = np.where(mask[..., None], points, 0.0)
        x = (x-self._get('feature_scaling', 0))/np.maximum(self._get('feature_scaling', 1), 0.05)
        x = np.where(mask[..., None], x, 0.0)
        x = self._dense(x, 'point_projection', 'swish')
        for i in range(1, len(CONV)+1):
            update = self._layer_norm(self._masked_conv(x, mask, f'conv_{i}'), f'norm_{i}')
            gamma, beta = np.split(self._dense(context, f'film_{i}'), 2, axis=-1)
            update = update*(1+gamma[:, None, :])+beta[:, None, :]
            x = np.where(mask[..., None], x+update, 0.0)
        weight = mask[..., None].astype(float)
        mean_pool = (x*weight).sum(1)/np.maximum(weight.sum(1), 1.0)
        max_pool = np.where(mask.any(1)[:, None], np.where(mask[..., None], x, -1e30).max(1), 0.0)
        h = np.concatenate([mean_pool, max_pool, context], -1)
        for i in range(3):
            h = self._dense(h, f'head_hidden_{i}', 'swish')
        return self._dense(h, 'parameter_logits', 'sigmoid').reshape(len(h), self.heads, 32)


def export_weights(model_dir, weights='best.weights.h5'):
    from research.cond_v2 import model as cm
    model_dir = Path(model_dir)
    config = json.loads((model_dir/'model_config.json').read_text())
    net = cm.build_model(**{k: v for k, v in config.items() if k != 'arch'})
    net.load_weights(str(model_dir/weights))
    np.savez_compressed(model_dir/'weights_numpy.npz',
                        **{f'{layer.name}/{i}': w for layer in net.layers for i, w in enumerate(layer.get_weights())})
    return net


def verify(model_dir, shard, n=64):
    from research.cond_v2 import model as cm
    net = export_weights(model_dir)
    d = np.load(shard)
    pts, m = cm.features(d['q'][:n], d['observed'][:n], d['sigma'][:n], d['mask'][:n], q_max=d['q_max'][:n])
    cond = cm.cond_vector(d['qz'][:n], d['gisaxs'][:n], d['q_min'][:n], d['q_max'][:n])
    combo = d['combo'][:n].astype('int32')
    ref = net({'points': pts, 'point_mask': m, 'combo': combo, 'cond': cond}, training=False)['u'].numpy()
    ours = NumpyCondV2(model_dir)(pts, m, combo, cond)
    return dict(max_abs_diff=float(np.abs(ref-ours).max()))


if __name__ == '__main__':
    import sys
    print(json.dumps(verify(sys.argv[1], sys.argv[2])))
