"""Time-consistent path through all network explanations of a sputter series (protocol sputter_sequence_v1).

Every window gets every branch x component centre of the polymer (island) and
TS (film) networks as candidates. An exact Viterbi path maximises the sum of
per-window data scores (component weight, agreement of D with the direct
2pi/q*, agreement with the observation-rule stage label) and registered
physical transition scores (sizes and spacing do not shrink quickly; film does
not revert to islands). Each chosen value says whether the data support it
or the constraint imposed it. Costs are fixed by the registered protocol
sputter_sequence_v1 and were not tuned on real scans. Amendment (PROTOCOL):
windows whose stage rule is unresolved are neutral, they add no rule-mismatch
cost; this change was made after real scans showed such windows suppressing
the film, and the amended path was not re-scored in the protocol's simulation
test.

Example:
    from sputtergisaxs.sputter_sequence import sequence_table
    sequence_table("OUT/binned", "OUT", metal="Ag", polymer_expert="experts/polymer", ts_expert="experts/TS")
"""
from __future__ import annotations
from pathlib import Path
import csv, gzip, json, math
import numpy as np

COSTS = dict(D_direct_sd=0.33, rule_mismatch=2.0, R_up=0.15, R_down=0.05, D_up=0.15, D_down=0.07, D_nopeak_weight=0.5,
             L_up=0.15, L_down=0.07, islands_to_film=2.0, window_s=0.5, constraint_driven_nats=2.0)
PROTOCOL = 'sputter_sequence_v1+unresolved_neutral (amendment not re-scored in simulation)'
LATE = {'coalescence_candidate', 'coalescence_or_film_candidate', 'continuous_film_candidate'}
PEAK = ('bilateral', 'one_side')


# ------------------------------------------------------------------ candidates --
def mixture_candidates(expert, encoded):
    """All branch x component centres of one expert for a batch of encoded windows -> list per window."""
    from .neural_refine_training import apply_policy
    from .neural_refine_calibration import extract_mixture
    from .neural_refine_chart import NAMES
    from .neural_refine_infer import physical_values
    mix, active = extract_mixture(expert, apply_policy(encoded, expert.policy))
    bank = expert.declaration['bank']; module = bank['module']
    mean = np.asarray(bank['normalization_mean']); scale = np.asarray(bank['normalization_scale'])
    mask = np.zeros(len(NAMES), bool); mask[active] = True
    out = []
    for n in range(len(encoded)):
        cands = []
        for b in range(mix['logw'].shape[1]):
            for k in range(mix['logw'].shape[2]):
                z = np.zeros(len(NAMES)); z[active] = mix['mu'][n, b, k] * scale[active] + mean[active]
                with np.errstate(over='ignore', invalid='ignore'):
                    v = {key: float(val) for key, val in physical_values(z, mask).items()}
                if not all(math.isfinite(x) and x >= 0 for x in v.values()):
                    continue
                cands.append(dict(module=module, mode=f'{b}:{k}', logw=float(mix['logw'][n, b, k]), values=v))
        out.append(cands)
    return out


# --------------------------------------------------------------------- scoring --
def emission(c, window, costs=COSTS):
    # Log weight relative to the best component of the same network in this window: weights of two
    # different networks are not comparable (protocol amendment 2026-10-22T09:10).
    score = c['logw'] - window['best_logw'][c['module']]
    if c['module'] == 'polymer_embed' and window.get('peak') and window.get('D_corr'):
        D = c['values'].get('D_nm')
        if D and D > 0:
            score += -0.5 * (math.log(D / window['D_corr']) / costs['D_direct_sd']) ** 2
    if window.get('rule_module') and c['module'] != window['rule_module']:
        score -= costs['rule_mismatch']
    return score


def _step(d, up, down):
    return -0.5 * (d / up) ** 2 if d >= 0 else -0.5 * (d / down) ** 2


def transition(a, b, dt, later_has_peak, costs=COSTS):
    f = math.sqrt(max(dt, 1e-3) / costs['window_s'])
    if a['module'] == 'TS_layer' and b['module'] == 'polymer_embed':
        return -math.inf
    if a['module'] != b['module']:
        return -costs['islands_to_film']
    va, vb = a['values'], b['values']
    if b['module'] == 'TS_layer':
        La, Lb = va.get('spectral_domain_nm'), vb.get('spectral_domain_nm')
        return _step(math.log(Lb / La), costs['L_up'] * f, costs['L_down'] * f) if La and Lb else 0.
    s = 0.
    if va.get('R_number_mean_nm') and vb.get('R_number_mean_nm'):
        s += _step(math.log(vb['R_number_mean_nm'] / va['R_number_mean_nm']), costs['R_up'] * f, costs['R_down'] * f)
    if va.get('D_nm') and vb.get('D_nm'):
        w = 1. if later_has_peak else costs['D_nopeak_weight']
        s += w * _step(math.log(vb['D_nm'] / va['D_nm']), costs['D_up'] * f, costs['D_down'] * f)
    return s


def viterbi(windows, costs=COSTS):
    """windows: [{time_s, candidates, D_corr, peak, rule_module}] -> chosen candidate index per window and diagnostics."""
    n = len(windows)
    for w in windows:
        w['best_logw'] = {}
        for c in w['candidates']:
            w['best_logw'][c['module']] = max(w['best_logw'].get(c['module'], -math.inf), c['logw'])
    em = [np.array([emission(c, w, costs) for c in w['candidates']]) for w in windows]
    score = em[0].copy(); back = []
    for t in range(1, n):
        prev, cur = windows[t - 1], windows[t]
        dt = (cur['time_s'] - prev['time_s']) if (cur.get('time_s') is not None and prev.get('time_s') is not None) else costs['window_s']
        trans = np.array([[transition(a, b, dt, cur.get('peak'), costs) for a in prev['candidates']] for b in cur['candidates']])
        total = trans + score[None, :]
        arg = np.argmax(total, axis=1); back.append(arg)
        score = total[np.arange(len(arg)), arg] + em[t]
    path = [int(np.argmax(score))]
    for arg in reversed(back):
        path.append(int(arg[path[-1]]))
    path.reverse()
    diag = []
    for t, (w, i) in enumerate(zip(windows, path)):
        best = int(np.argmax(em[t])); cost = float(em[t][best] - em[t][i])
        diag.append(dict(chosen=i, best=best, data_cost_nats=cost,
                         status='data-supported' if cost <= costs['constraint_driven_nats'] else 'constraint-driven'))
    return path, diag


# --------------------------------------------------------------- whole series --
def rule_module(stage):
    """Module the observation rule votes for; an unresolved (or missing) stage casts no vote."""
    if not stage or stage == 'stage_unresolved':
        return None
    return 'TS_layer' if stage in LATE else 'polymer_embed'


def load_windows(binned):
    binned = Path(binned)
    obs = json.loads(gzip.open(binned / 'observations.json.gz').read())
    index = json.loads((binned / 'features.json').read_text())
    keys, encoded = [], []
    for f in index['files']:
        rows = json.loads(gzip.open(binned / f['file']).read())
        vec = np.load(binned / f['vector_file'])['encoded']
        if len(rows) != len(vec):
            raise ValueError('Feature rows and vectors differ for ' + f['file'])
        keys += [r['key'] for r in rows]; encoded.append(vec)
    encoded = np.concatenate(encoded)
    by_key = {r['key']: r for r in obs}
    windows = []
    for key in keys:
        r = by_key[key]; o = r['observations']
        dc = o.get('D_corr_nm'); dc = dc if isinstance(dc, (int, float)) else None
        stage = (r.get('stage_review') or {}).get('stage')
        windows.append(dict(key=key, time_s=r.get('time_s'), D_corr=dc, peak=o.get('feature_status') in PEAK,
                            feature_status=o.get('feature_status'), stage=stage,
                            rule_module=rule_module(stage)))
    order = sorted(range(len(windows)), key=lambda i: (windows[i]['time_s'] is None, windows[i]['time_s'] or 0., i))
    return [windows[i] for i in order], encoded[order]


def sequence_rows(binned, *, metal, polymer_expert, ts_expert, device='cpu', costs=COSTS):
    from .neural_refine_infer import NRExpert
    windows, encoded = load_windows(binned)
    poly = NRExpert(Path(polymer_expert) / 'model/best.pt', Path(polymer_expert) / 'policy.json', device=device)
    film = NRExpert(Path(ts_expert) / 'model/best.pt', Path(ts_expert) / 'policy.json', device=device)
    pc, tc = mixture_candidates(poly, encoded), mixture_candidates(film, encoded)
    for w, a, b in zip(windows, pc, tc):
        w['candidates'] = a + b
    path, diag = viterbi(windows, costs)
    from .sputter_struct import reliability
    rel = reliability()['parameters']; model_metal = 'Ag' if metal == 'Cu' else metal
    band = rel.get('D_over_direct', {}).get('band_90', [0.587, 1.75])
    rows = []
    for w, i, d in zip(windows, path, diag):
        c = w['candidates'][i]; best = w['candidates'][d['best']]; v = c['values']
        poly_best = max((x for x in w['candidates'] if x['module'] == 'polymer_embed'), key=lambda x: emission(x, w, costs))
        grade = 'S' if d['status'] == 'data-supported' else 'T'
        row = dict(frame=w['key'], time_s=w['time_s'], stage=w['stage'], module=c['module'], feature_status=w['feature_status'],
                   D_corr_nm=w['D_corr'], sequence_choice=f"{c['module']}:{c['mode']}", sequence_component_weight=math.exp(c['logw']),
                   sequence_data_cost_nats=d['data_cost_nats'], sequence_status=d['status'],
                   unconstrained_choice=f"{best['module']}:{best['mode']}",
                   unconstrained_R_nm=poly_best['values'].get('R_number_mean_nm'), unconstrained_D_nm=poly_best['values'].get('D_nm'))
        if c['module'] == 'polymer_embed':
            err = rel['R'][model_metal]['network']
            R = v.get('R_number_mean_nm')
            row.update(R_nm=R, R_source='sequence', R_grade=grade, R_reason='', R_typical_rel_error=err['typical_rel_scatter'],
                       R_lo_nm=R / (1 + err['typical_rel_scatter']), R_hi_nm=R * (1 + err['typical_rel_scatter']),
                       R_P90_rel_error=err['P90_rel_error'], R_order_accuracy=err['order_accuracy_1p2x'],
                       sigma_log=v.get('radius_sigma_log'), spacing_disorder=None,
                       burial_nm_conditional=v.get('burial_base_depth_nm'), thickness_nm_conditional=v.get('polymer_thickness_nm'),
                       L_TS_nm=None, TS_metal_fraction_conditional=None, TS_film_thickness_nm_conditional=None)
            if w['peak'] and v.get('D_nm'):
                e = rel['D'][model_metal]['network']; D = v['D_nm']
                row.update(D_nm=D, D_source='sequence', D_grade=grade, D_reason='', D_typical_rel_error=e['typical_rel_scatter'],
                           D_lo_nm=D / (1 + e['typical_rel_scatter']), D_hi_nm=D * (1 + e['typical_rel_scatter']),
                           D_P90_rel_error=e['P90_rel_error'], D_order_accuracy=e['order_accuracy_1p2x'])
                ratio = D / w['D_corr'] if w['D_corr'] else None
                row.update(D_over_direct=ratio, D_direct_check=None if ratio is None else
                           ('consistent' if band[0] <= ratio <= band[1] else 'outside simulated 90% range of D/(2pi/q*)'))
            else:
                row.update(D_nm=None, D_grade=None, D_reason='no resolved peak: D not reported', D_direct_check=None)
        else:
            row.update(R_nm=None, R_grade=None, R_reason='film (TS) window on the time-consistent path', D_nm=None, D_grade=None,
                       D_reason='film (TS) window', L_TS_nm=v.get('spectral_domain_nm'),
                       TS_metal_fraction_conditional=v.get('metal_fraction'), TS_film_thickness_nm_conditional=v.get('film_thickness_nm'))
        row['metal'] = metal
        row['note'] = 'time-consistent path (sputter_sequence_v1); typical errors are the network values until the sequence test is scored'
        rows.append(row)
    return rows


def sequence_table(binned, out, *, metal, polymer_expert, ts_expert, device='cpu'):
    out = Path(out); rows = sequence_rows(binned, metal=metal, polymer_expert=polymer_expert, ts_expert=ts_expert, device=device)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    path = out / 'structure_sequence.csv'
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    from .sputter_struct import summary, plot_structure
    s = summary(rows, metal)
    s['sequence'] = dict(protocol=PROTOCOL, costs=COSTS,
                         constraint_driven=sum(r['sequence_status'] == 'constraint-driven' for r in rows),
                         film_windows=sum(r['module'] == 'TS_layer' for r in rows),
                         first_film_time_s=next((r['time_s'] for r in rows if r['module'] == 'TS_layer'), None),
                         changed_from_unconstrained=sum(r['sequence_choice'] != r['unconstrained_choice'] for r in rows))
    (out / 'structure_sequence_summary.json').write_text(json.dumps(s, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    plot_structure(rows, out / 'structure_sequence_trend.png', title=Path(out).name + ' (time-consistent)')
    return path
