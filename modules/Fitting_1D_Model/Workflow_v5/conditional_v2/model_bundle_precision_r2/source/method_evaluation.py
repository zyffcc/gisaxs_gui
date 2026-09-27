"""Shared, observed-only candidate selection for validation and frozen step-budget tests."""
import argparse
import json
import os
import time
from pathlib import Path
import numpy as np
import tensorflow as tf
from benchmark_model import COMBOS, ProposalModel, TRAIN_KEYS, load_arrays
from evaluate_benchmark import project, logrmse, stats
from fast_refinement import FastRefiner, forward


@tf.function(reduce_retracing=True)
def compiled_forward(q, mask, types, p, w, g, d, res):
    return forward(q, mask, types, p, w, g, tf.where(d > 0, 30., -30.), tf.cast(res > 0, tf.float32))


def load_model(path, data):
    cfg = json.loads((path/'config.json').read_text())
    model = ProposalModel(cfg['architecture'], cfg['hypotheses'])
    model({k: tf.convert_to_tensor(data[k][:2]) for k in TRAIN_KEYS})
    model.load_weights(str(path/'best.weights.h5'))
    return model


def select_indices(cache, split, count, previous=None):
    raw = load_arrays(cache/split, ['types'])
    limit = 2500 if split == 'val' else len(raw['types'])
    k = np.sum(raw['types'][:limit] > 0, axis=1)
    eligible = np.ones(limit, bool)
    if previous is not None:
        eligible[np.load(previous)['source_indices']] = False
    rng = np.random.default_rng(20260915)
    return np.concatenate([rng.permutation(np.flatnonzero((k == j) & eligible))[:count//4] for j in range(1, 5)])


def propose(model, data):
    start = time.monotonic()
    inp = {k: tf.convert_to_tensor(data[k]) for k in ('curve', 'context')}
    z = model.encode(inp)
    prob = tf.nn.softmax(model.classifier(z)).numpy()
    choices = np.argsort(-prob, axis=1)[:, :4]
    out = {k: v.numpy() for k, v in model.decode(tf.repeat(z, 4, axis=0), choices.reshape(-1)).items()}
    n, h = len(choices), model.hypotheses
    candidate = {'params': project(out['params']).reshape(n, 4*h, 4, 6),
        'weights': out['weight_logits'].reshape(n, 4*h, 4), 'globals': out['globals'].reshape(n, 4*h, 4),
        'd': out['d_logits'].reshape(n, 4*h, 4), 'res': out['resolution_logits'].reshape(n, 4*h),
        'combos': np.repeat(choices, h, axis=1), 'probabilities': prob}
    candidate['network_seconds'] = time.monotonic()-start
    q, mask = [np.repeat(data[k], 4*h, axis=0) for k in ('q', 'mask')]
    types = COMBOS[candidate['combos']].reshape(-1, 4)
    p, w, g, d, res = [candidate[k].reshape((-1,)+candidate[k].shape[2:]) for k in ('params','weights','globals','d','res')]
    curves = []
    for i in range(0, len(q), 32):
        sl = slice(i, i+32)
        args = [q[sl], mask[sl], types[sl], p[sl], w[sl], g[sl], d[sl], res[sl]]
        curves.append(compiled_forward(*[tf.convert_to_tensor(a) for a in args]).numpy())
    candidate['curves'] = np.concatenate(curves).reshape(n, 4*h, 1000)
    candidate['seconds'] = time.monotonic()-start
    return candidate


def diagnostics(prediction, data, combos):
    observed = logrmse(prediction, data['observed'], data['mask'])
    clean = logrmse(prediction, data['clean'], data['mask'])
    # An observable fit statistic. Whether it safely predicts clean accuracy is calibrated on validation only.
    residual = (prediction.astype('float64')-data['observed'])/np.maximum(data['sigma'].astype('float64'), 1e-30)
    chi2 = np.sum(residual**2*data['mask'], axis=1)/np.sum(data['mask'], axis=1)
    result = {'observed': stats(observed), 'clean': stats(clean),
              'selected_combination_accuracy': float(np.mean(combos == data['combo']))}
    result['by_K'] = {str(k): {'observed': stats(observed[np.sum(data['types'] > 0, axis=1) == k]),
                             'clean': stats(clean[np.sum(data['types'] > 0, axis=1) == k])} for k in range(1, 5)}
    return result, {'observed_error': observed, 'clean_error': clean, 'chi2': chi2}


def direct_evaluate(model, data):
    cand = propose(model, data)
    errors = logrmse(cand['curves'], data['observed'][:, None], data['mask'][:, None])
    best, rows = np.argmin(errors, axis=1), np.arange(len(errors))
    result, detail = diagnostics(cand['curves'][rows, best], data, cand['combos'][rows, best])
    result.update(network_seconds=cand['network_seconds'], seconds=cand['seconds'],
        combination_top4=float(np.mean(np.any(cand['combos'] == data['combo'][:, None], axis=1))))
    return result, detail, cand


def main(a):
    assert os.environ.get('SLURM_JOB_ID')
    for device in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(device, True)
    a.output.mkdir(parents=True, exist_ok=False)
    ids = select_indices(a.cache, a.split, a.count, a.exclude)
    raw = load_arrays(a.cache/a.split)
    data = {k: np.asarray(v[ids]) for k, v in raw.items()}
    model = load_model(a.model, data)
    # Warm the forward graph on training data, without evaluating test labels.
    warm = load_arrays(a.cache/'train')
    propose(model, {k: np.asarray(v[:2]) for k, v in warm.items()})
    result0, details0, cand = direct_evaluate(model, data)
    n, h = len(ids), model.hypotheses
    errors = logrmse(cand['curves'], data['observed'][:, None], data['mask'][:, None])
    selected = np.argmin(errors.reshape(n, 4, h), axis=2)+h*np.arange(4)[None]
    rows = np.arange(n)[:, None]
    initials = [cand[k][rows, selected].reshape((-1,)+cand[k].shape[2:]) for k in ('params','weights','globals','d','res')]
    combos = cand['combos'][rows, selected]
    q, mask, observed = [np.repeat(data[k], 4, axis=0) for k in ('q','mask','observed')]
    typ = COMBOS[combos].reshape(-1, 4)
    refiner = FastRefiner(32)
    checkpoints = (0, 10, 30, 160)
    # Compile on training-derived teacher-like starts. Compilation excluded from warmed runtime.
    wa = [np.repeat(warm[k][:1], 32, axis=0) for k in ('q', 'mask', 'observed', 'types', 'params')]
    ww = np.log(np.maximum(np.repeat(warm['weights'][:1], 32, axis=0), 1e-10))
    wg = np.repeat(warm['globals'][:1], 32, axis=0)
    wd = np.repeat(warm['d'][:1], 32, axis=0)*2-1
    wr = np.repeat(warm['resolution'][:1], 32, axis=0)*2-1
    refiner.run(*wa, ww, wg, wd, wr, (0, 1))
    snapshots = {s: {k: [] for k in ('curve', 'params', 'weights', 'globals')} for s in checkpoints}
    times = {s: 0. for s in checkpoints}
    for i in range(0, len(q), 32):
        sl = slice(i, i+32)
        out = refiner.run(q[sl], mask[sl], observed[sl], typ[sl], *[x[sl] for x in initials], checkpoints)
        for s in checkpoints:
            for k in snapshots[s]: snapshots[s][k].append(out[s][k])
            times[s] += out[s]['seconds']
        if i % 256 == 0: print('EVALUATED', min(i+32,len(q)), '/', len(q), flush=True)
    results = {'model': str(a.model), 'split': a.split, 'count': n, 'stages': {'0': result0},
               'selection': 'observed native-grid logRMSE, one initial per predicted topology; no truth-based inference',
               'refinement_kernel_cumulative_seconds': times, 'initial_proposal_seconds': cand['seconds']}
    arrays = {'source_indices': ids, 'q': data['q'], 'mask': data['mask'], 'observed': data['observed'],
              'clean': data['clean'], 'true_combo': data['combo'], 'candidate_combos': combos}
    row = np.arange(n)
    firstbest = np.argmin(errors, axis=1)
    arrays['prediction_0'] = cand['curves'][row, firstbest]
    for k, v in details0.items(): arrays[k+'_0'] = v
    for s in checkpoints[1:]:
        curves = np.concatenate(snapshots[s]['curve']).reshape(n, 4, 1000)
        err = logrmse(curves, data['observed'][:, None], data['mask'][:, None])
        best = np.argmin(err, axis=1)
        result, detail = diagnostics(curves[row, best], data, combos[row, best])
        results['stages'][str(s)] = result
        for k,v in detail.items(): arrays[k+'_'+str(s)] = v
        arrays['prediction_'+str(s)] = curves[row, best]
        arrays['candidate_params_'+str(s)] = np.concatenate(snapshots[s]['params']).reshape(n,4,4,6)
    (a.output/'result.json').write_text(json.dumps(results, indent=2))
    np.savez_compressed(a.output/'examples.npz', **arrays)
    print('METHOD_RESULT', json.dumps(results), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('cache','model','output'): p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--split', choices=['val','test'], default='val')
    p.add_argument('--count', type=int, default=128)
    p.add_argument('--exclude', type=Path)
    main(p.parse_args())
