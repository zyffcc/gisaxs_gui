"""Score proposals by observed-curve fit, with separate truth-only diagnostics."""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

import physics_v5
from benchmark_model import COMBOS, ProposalModel, TRAIN_KEYS, load_arrays


def project(params):
    """Apply the dataset's D > 2R support, without changing any fitted amplitudes."""
    p = np.clip(params, 0, 1).copy()
    radius = np.exp(np.log(1.) + p[..., 0] * np.log(100.))
    minimum_d = np.log(2 * radius * 1.001 / 3.) / np.log(500. / 3.)
    p[..., 4] = np.maximum(p[..., 4], minimum_d)
    return np.clip(p, 0, 1)


def predict_curves(q, mask, types, params, weights, glob, d, resolution, batch_size=8):
    curves = []
    for start in range(0, len(q), batch_size):
        sl = slice(start, start+batch_size)
        gp = np.pad(glob[sl], ((0, 0), (0, 1))).astype('float32')
        pred = physics_v5.reconstruct_intensity(
            q[sl], types[sl], (types[sl] > 0).astype('float32'), params[sl],
            weights[sl], gp, np.where(d[sl] > 0, 30., -30.).astype('float32'),
            (resolution[sl] > 0).astype('float32'), point_mask=mask[sl] > .5)
        curves.append(pred.numpy())
    return np.concatenate(curves)


def logrmse(prediction, target, mask):
    residual = np.log(np.maximum(prediction, 1e-30)) - np.log(np.maximum(target, 1e-30))
    return np.sqrt(np.sum(residual**2 * mask, axis=-1) / np.sum(mask, axis=-1))


def stats(values):
    a = np.asarray(values)
    return {'median': float(np.median(a)), 'p90': float(np.quantile(a, .9)),
            'fraction_lt_0.05': float(np.mean(a < .05)),
            'fraction_lt_0.1': float(np.mean(a < .1)),
            'fraction_lt_0.2': float(np.mean(a < .2))}


def main(args):
    assert os.environ.get('SLURM_JOB_ID')
    for device in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(device, True)
    cfg = json.loads((args.model / 'config.json').read_text())
    model = ProposalModel(cfg['architecture'], cfg['hypotheses'])
    raw = load_arrays(args.cache / args.split)
    limit = 2500 if args.split == 'val' else len(raw['combo'])
    rng = np.random.default_rng(20260914)
    all_k = np.sum(raw['types'][:limit] > 0, axis=1)
    chosen = np.concatenate([rng.permutation(np.flatnonzero(all_k == k))[:args.count // 4]
                             for k in range(1, 5)])
    data = {k: np.asarray(v[chosen]) for k, v in raw.items()}
    inputs = {k: tf.convert_to_tensor(data[k]) for k in TRAIN_KEYS}
    model({k: v[:2] for k, v in inputs.items()})
    weights_path = args.model / (args.weights + '.weights.h5')
    model.load_weights(str(weights_path))
    started = time.monotonic()
    feature = model.encode(inputs)
    logits = model.classifier(feature)
    probs = tf.nn.softmax(logits, axis=-1).numpy()
    combo_choices = np.argsort(-probs, axis=1)[:, :args.topk]
    n, h = len(chosen), model.hypotheses
    candidates = args.topk * h
    expanded_feature = tf.repeat(feature, args.topk, axis=0)
    output = {k: v.numpy() for k, v in model.decode(expanded_feature, combo_choices.reshape(-1)).items()}
    params = project(output['params'].reshape(n, candidates, 4, 6))
    w = output['weight_logits'].reshape(n, candidates, 4)
    gp = output['globals'].reshape(n, candidates, 4)
    d = output['d_logits'].reshape(n, candidates, 4)
    resolution = output['resolution_logits'].reshape(n, candidates)
    proposal_probs = tf.nn.softmax(output['mixture_logits'], axis=-1).numpy().reshape(n, args.topk, h)
    proposal_probs *= np.take_along_axis(probs, combo_choices, axis=1)[..., None]
    proposal_probs = proposal_probs.reshape(n, candidates)
    candidate_combos = np.repeat(combo_choices, h, axis=1)
    types = COMBOS[candidate_combos]
    inference_seconds = time.monotonic() - started
    observed_errors = np.zeros((n, candidates))
    clean_errors = np.zeros((n, candidates))
    predicted = np.zeros((n, candidates, 1000), dtype='float32')
    for j in range(candidates):
        curves = predict_curves(data['q'], data['mask'], types[:, j], params[:, j], w[:, j],
                                gp[:, j], d[:, j], resolution[:, j])
        assert np.all(np.isfinite(curves))
        predicted[:, j] = curves
        observed_errors[:, j] = logrmse(curves, data['observed'], data['mask'])
        clean_errors[:, j] = logrmse(curves, data['clean'], data['mask'])
        print('FORWARD', j+1, '/', candidates, flush=True)
    # Selection uses only the measured curve, never clean curves or parameter labels.
    best = np.argmin(observed_errors, axis=1)
    top = np.argmax(proposal_probs, axis=1)
    rows = np.arange(n)
    exact = candidate_combos[rows, best] == data['combo']
    selected_params = params[rows, best].copy()
    selected_types = types[rows, best]
    for i in range(n):
        order = sorted(range(4), key=lambda j: (selected_types[i, j] == 0,
                                                int(selected_types[i, j]), selected_params[i, j, 0]))
        selected_params[i] = selected_params[i, order]
    param_mae = np.sum(np.abs(selected_params - data['params']) * data['param_mask'], axis=(1, 2))
    param_mae /= np.maximum(np.sum(data['param_mask'], axis=(1, 2)), 1)
    result = {'architecture': cfg['architecture'], 'train_count': cfg['train_count'],
              'split': args.split, 'count': n, 'topk': args.topk, 'hypotheses': h,
              'combination_top1': float(np.mean(combo_choices[:, 0] == data['combo'])),
              'combination_topk': float(np.mean(np.any(combo_choices == data['combo'][:, None], axis=1))),
              'selected_combination_accuracy': float(np.mean(exact)),
              'top_score_observed_logrmse': stats(observed_errors[rows, top]),
              'data_selected_observed_logrmse': stats(observed_errors[rows, best]),
              'data_selected_clean_logrmse': stats(clean_errors[rows, best]),
              'normalized_parameter_mae_when_combination_correct': float(np.mean(param_mae[exact])) if np.any(exact) else None,
              'parameter_metric_sample_count': int(np.sum(exact)),
              'network_seconds': inference_seconds, 'total_seconds': time.monotonic()-started,
              'selection': 'minimum native-q observed logRMSE; no optimizer and no truth-based selection',
              'probability_status': 'uncalibrated topology probabilities and proposal scores'}
    result['by_K'] = {}
    for k in range(1, 5):
        use = np.sum(data['types'] > 0, axis=1) == k
        result['by_K'][str(k)] = {
            'count': int(np.sum(use)), 'observed_logrmse': stats(observed_errors[rows, best][use]),
            'clean_logrmse': stats(clean_errors[rows, best][use]),
            'combination_topk': float(np.mean(np.any(combo_choices[use] == data['combo'][use, None], axis=1)))}
    prefix = args.model / ('evaluation_' + args.split + '_' + args.weights)
    prefix.with_suffix('.json').write_text(json.dumps(result, indent=2))
    np.savez_compressed(str(prefix) + '_examples.npz', q=data['q'], mask=data['mask'],
                        observed=data['observed'], clean=data['clean'],
                        prediction=predicted[rows, best], true_combo=data['combo'],
                        selected_combo=candidate_combos[rows, best],
                        selected_params=selected_params, true_params=data['params'],
                        selected_observed_error=observed_errors[rows, best],
                        selected_clean_error=clean_errors[rows, best],
                        source_indices=chosen, combo_probabilities=probs)
    print('RESULT', json.dumps(result), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--split', choices=['val', 'test'], default='val')
    p.add_argument('--count', type=int, default=256)
    p.add_argument('--topk', type=int, default=4)
    p.add_argument('--weights', choices=['best', 'last'], default='best')
    main(p.parse_args())
