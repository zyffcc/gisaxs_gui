"""Outputs of a series1d run: series.csv, trend.png, summary.json, summary.html.

series.csv   one line per frame (the valid row of every frame index), columns in the fixed order
             BASE_COLUMNS, then POP_COLUMNS for pop1 .. popK (K = populations of the composition; 4 for a
             folder of curves fitted with an 'auto' composition). Empty = not available (off population,
             skipped frame, no uncertainty). No timing columns: the file of a resumed or re-run batch/review
             run is byte-identical.
trend.png    small multiples over time: R, D, amplitude per population and chi2/point (one axis per panel)
summary.json settings, provenance (as fit1d + series1d sources hash), counts, timing percentiles
summary.html one self-contained page for a phone: key table, population table, embedded trend.png
"""
from __future__ import annotations

import base64
import csv
import hashlib
import html
import json
import time
from pathlib import Path

import numpy as np

from research.fit1d import schema as fit_schema

SUMMARY_SCHEMA = 'gisaxs1d_series_summary_v1'
BASE_COLUMNS = ('index', 'frame', 'file', 'time_s', 'mode', 'state', 'n_points', 'total_counts', 'chi2_per_point',
                'chi2_baseline', 'fallback', 'fallback_accepted', 'chi2_jump', 'signals', 'change_points', 'off',
                'weakly_constrained', 'at_bound', 'qz_nm_inv', 'background_B', 'low_q_C', 'low_q_sigma_res_nm_inv',
                'low_q_nu', 'low_q_powerlaw_n')
POP_COLUMNS = ('type', 'state', 'amplitude', 'R_nm', 'R_nm_err', 'R_nm_smoothed', 'D_nm', 'D_nm_err', 'D_nm_smoothed',
               'sigma_R_rel', 'sigma_R_rel_err', 'w_rel', 'w_rel_err', 'h_nm', 'h_nm_err', 'sigma_h_rel',
               'R_number_mean_nm', 'weak')
# categorical slots 1-4 of the validated reference palette (fixed order: the colour follows the population)
SERIES = ('#2a78d6', '#eb6834', '#1baf7a', '#eda100')
MARKERS = ('o', 's', '^', 'D')
INK, MUTED, GRID, SURFACE = '#0b0b0b', '#898781', '#e1e0d9', '#fcfcfb'
SERIOUS = '#e34948'


def n_populations(recipe, rows):
    from research.series1d.model import ModelSpec
    spec = ModelSpec.from_recipe(recipe, independent=True)
    if spec.auto:
        return 4
    return len(spec.slots)


def csv_columns(k):
    return list(BASE_COLUMNS)+[f'pop{i}_{c}' for i in range(1, k+1) for c in POP_COLUMNS]


def _fmt(v):
    if v is None:
        return ''
    if isinstance(v, bool):
        return 'true' if v else 'false'
    if isinstance(v, float):
        return repr(v)
    if isinstance(v, (list, tuple)):
        return ';'.join(str(x) for x in v)
    return str(v)


def write_csv(path, recipe, rows):
    k = n_populations(recipe, rows)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(csv_columns(k))
        for row in rows:
            flags, low = row['flags'], row.get('low_q') or {}
            line = dict(index=row['index'], frame=row['frame'], file=row['file'], time_s=row['time_s'], mode=row['mode'],
                        state=row['state'], n_points=row['n_points'], total_counts=row.get('total_counts'),
                        chi2_per_point=row['chi2_per_point'], chi2_baseline=row.get('chi2_baseline'),
                        fallback=flags['fallback'], fallback_accepted=flags['fallback_accepted'],
                        chi2_jump=flags.get('chi2_jump', False), signals=flags['signals'],
                        change_points=flags.get('change_points', []), off=flags['off'],
                        weakly_constrained=flags['weakly_constrained'], at_bound=flags['at_bound'],
                        qz_nm_inv=row['qz_nm_inv'], background_B=row['background_B'], low_q_C=low.get('C'),
                        low_q_sigma_res_nm_inv=low.get('sigma_res_nm_inv'), low_q_nu=low.get('nu'),
                        low_q_powerlaw_n=low.get('powerlaw_n'))
            for i, pop in enumerate(row['populations'][:k], 1):
                for c in POP_COLUMNS:
                    line[f'pop{i}_{c}'] = pop.get(c)
            w.writerow([_fmt(line.get(c)) for c in csv_columns(k)])
    return str(path)


# ---------------------------------------------------------------------------- statistics
def _percentiles(values, scale=1.0):
    v = np.asarray([x for x in values if x is not None], float)*scale
    if not len(v):
        return None
    return dict(n=int(len(v)), median=float(np.median(v)), p95=float(np.percentile(v, 95)), max=float(v.max()),
                mean=float(v.mean()))


def series_statistics(rows, sigma_jump=3.0):
    """Stability numbers of a series (PLAN A7 a/c and the M3 quality line), computed from the rows.

    jumps_3sigma: pairs of consecutive fitted frames in which a reportable parameter (R_nm, D_nm) of a
    population that is on in both changes by more than sigma_jump x sqrt(err1^2 + err2^2), with the 1-sigma
    values of the two fits; pairs in which the parameter is weakly constrained or has no error are left out.
    'pairs'/'jumps' count the pairs without a change point flagged for that parameter in the later frame
    (PLAN A7 a); 'unflagged_pairs'/'unflagged_jumps' count only pairs in which neither frame carries a
    chi2_jump flag, a fallback or a change point of that parameter.
    jumps_at_accepted_fallbacks (PLAN A7 c): the same test for every frame in which a fresh solution replaced
    the track (accepted fallback, live backfill, batch basin switch) against the fitted frame before it,
    whether or not a change point is flagged there; 'frames' lists the frame indices that jump."""
    fitted = [r for r in rows if r['mode'] not in ('skipped',)]
    out = dict(frames=len(rows), fitted=len(fitted))
    if not fitted:
        return out
    chi = [r['chi2_per_point'] for r in fitted]
    out['chi2_per_point'] = _percentiles(chi)
    out['chi2_jump_frames'] = int(sum(bool(r['flags'].get('chi2_jump')) for r in fitted))
    out['chi2_jump_fraction'] = out['chi2_jump_frames']/len(fitted)
    labels = [p['pop'] for p in fitted[0]['populations']]
    jumps, swaps, fallback_jumps = {}, 0, {}
    for a, b in zip(fitted[:-1], fitted[1:]):
        pa = {p['pop']: p for p in a['populations']}
        pb = {p['pop']: p for p in b['populations']}
        for label in labels:
            x, y = pa.get(label), pb.get(label)
            if x is None or y is None or x['state'] != 'on' or y['state'] != 'on':
                continue
            for key in ('R_nm', 'D_nm'):
                name = f'{label}.{key}'
                if x.get(key) is None or y.get(key) is None:
                    continue
                e1, e2 = x.get(f'{key}_err'), y.get(f'{key}_err')
                if not e1 or not e2 or key in x.get('weak', ()) or key in y.get('weak', ()):
                    continue
                d = jumps.setdefault(name, dict(pairs=0, jumps=0, at_change_points=0, unflagged_pairs=0,
                                                unflagged_jumps=0))
                big = bool(abs(y[key]-x[key]) > sigma_jump*np.hypot(e1, e2))
                if b['flags']['fallback_accepted']:
                    fb = fallback_jumps.setdefault(name, dict(fallback_frames=0, jumps=0, frames=[]))
                    fb['fallback_frames'] += 1
                    fb['jumps'] += int(big)
                    if big:
                        fb['frames'].append(int(b['index']))
                if name in b['flags'].get('change_points', []):
                    d['at_change_points'] += 1
                    continue
                d['pairs'] += 1
                d['jumps'] += int(big)
                if not any(r['flags'].get('chi2_jump') or r['flags']['fallback'] or r['state'] == 'pending_fallback'
                           or name in r['flags'].get('change_points', []) for r in (a, b)):
                    d['unflagged_pairs'] += 1
                    d['unflagged_jumps'] += int(big)
        # identity swap: two populations of one type exchange their (ln R, ln D) positions between two frames
        on = [l for l in labels if pa.get(l, {}).get('state') == 'on' and pb.get(l, {}).get('state') == 'on']
        for i, l1 in enumerate(on):
            for l2 in on[i+1:]:
                if pa[l1]['type'] != pa[l2]['type']:
                    continue

                def dist(p, q):
                    d = np.log(p['R_nm']/q['R_nm'])**2
                    if p.get('D_nm') and q.get('D_nm'):
                        d += np.log(p['D_nm']/q['D_nm'])**2
                    return d
                # both populations end nearer to where the other one was than to where they were themselves
                if dist(pb[l1], pa[l2]) < dist(pb[l1], pa[l1]) and dist(pb[l2], pa[l1]) < dist(pb[l2], pa[l2]):
                    swaps += 1
    for d in jumps.values():
        d['fraction'] = d['jumps']/d['pairs'] if d['pairs'] else None
        d['unflagged_fraction'] = d['unflagged_jumps']/d['unflagged_pairs'] if d['unflagged_pairs'] else None
    out.update(jumps_3sigma=jumps, identity_swaps=swaps, jumps_at_accepted_fallbacks=fallback_jumps)
    return out


def summarise(recipe, header, rows, info, n_total=None):
    fitted = [r for r in rows if r['mode'] != 'skipped']
    modes = {}
    for r in rows:
        modes[r['mode']] = modes.get(r['mode'], 0)+1
    labels = [p['pop'] for p in fitted[0]['populations']] if fitted else []
    pops = {}
    for label in labels:
        items = [next((p for p in r['populations'] if p['pop'] == label), None) for r in fitted]
        items = [p for p in items if p is not None]
        on = [p for p in items if p['state'] == 'on']
        entry = dict(type=items[0]['type'] if items else None, frames_on=len(on), frames_off=len(items)-len(on),
                     off_entries=0, on_entries=0)
        states = [p['state'] for p in items]
        for a, b in zip(states[:-1], states[1:]):
            entry['off_entries'] += int(a == 'on' and b == 'off')
            entry['on_entries'] += int(a == 'off' and b == 'on')
        for key in ('R_nm', 'D_nm', 'sigma_R_rel', 'w_rel', 'amplitude'):
            v = np.array([p[key] for p in on if p.get(key) is not None], float)
            entry[key] = dict(median=float(np.median(v)), mean=float(v.mean()), std=float(v.std())) if len(v) else None
        pops[label] = entry
    track = [r['timing'].get('total_s') for r in fitted if r['mode'] == 'track' and not r['flags']['fallback']]
    fallbacks = [r for r in fitted if r['flags']['fallback']]
    timing = dict(wall_seconds=info.get('wall_seconds'), processing_seconds=info.get('processing_seconds'),
                  frames_per_second=(len(rows)-info.get('resumed_frames', 0))/info['processing_seconds']
                  if info.get('processing_seconds') else None,
                  tracked_frame_ms=_percentiles(track, 1e3),
                  all_frames_ms=_percentiles([r['timing'].get('total_s') for r in fitted], 1e3),
                  first_frame_s=next((r['timing'].get('total_s') for r in fitted if r['mode'] == 'first'), None),
                  fallback_s=_percentiles([r['timing'].get('fallback_s') or r['timing'].get('backfill_s')
                                           for r in fallbacks]),
                  note='single frame times are wall times of this run on this machine (cut + curve + fit)')
    stats = series_statistics(rows)
    counts = dict(frames=len(rows), frames_in_input=n_total, fitted=len(fitted), modes=modes,
                  fallbacks_run=len(fallbacks),
                  fallbacks_accepted=int(sum(r['flags']['fallback_accepted'] for r in fitted)),
                  pending_fallback=int(sum(r['state'] == 'pending_fallback' for r in rows)),
                  backfilled=int(sum(r['state'] == 'backfilled' for r in rows)),
                  retracked=int(sum('retrack' in r for r in rows)),
                  skipped_unreadable=int(sum(r.get('skip_kind') == 'unreadable' for r in rows)),
                  basin_switches=int(sum(any(e.startswith('basin_switch') for e in r['flags'].get('events', []))
                                         for r in rows)),
                  reseeded_frames=int(sum(any(e.startswith('reseeded_from') for e in r['flags'].get('events', []))
                                          for r in rows)),
                  frames_with_off_population=int(sum(bool(r['flags']['off']) for r in fitted)),
                  change_points={}, signals={},
                  identity_permutations_of_fresh_solutions=int(sum(bool((r['fit'] or {}).get('fallback', {}).get('identity_permutation'))
                                                                    for r in fitted)))
    for r in fitted:
        for name in r['flags'].get('change_points', []):
            counts['change_points'][name] = counts['change_points'].get(name, 0)+1
        for name in r['flags']['signals']:
            counts['signals'][name] = counts['signals'].get(name, 0)+1
    package = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(package.glob('*.py')):
        digest.update(path.name.encode('utf-8')+b'\0'+path.read_bytes())
    provenance = dict(fit_schema.provenance(Path(recipe['model'].get('model_dir') or
                                                 fit_schema.WORK_DIR/'models'/'cond_v2_a').resolve()),
                      series1d_version=header.get('series1d_version'), series1d_sources_sha256=digest.hexdigest(),
                      recipe_sha256=header.get('recipe_sha256'))
    conventions = dict(fit_schema.CONVENTIONS,
                       series=dict(time='time_s = series.time_step_s x (frame number - series.time_origin_frame)',
                                   populations='pop1.. are slots in the fit1d population order (type code, then '
                                               'slot); in the first frame populations of one type and equal '
                                               'constraints are sorted by R; later frames keep the slot',
                                   off="state 'off': the NNLS amplitude is 0, or removing the population raises "
                                       'chi2 by less than 9 (not significant at 3 sigma); the amplitude is then '
                                       "reported as 0, the parameters as null, the frozen values under 'frozen'",
                                   sphere_model=('born: every frame. dwba: the fit1d DWBA stage runs on the first '
                                                 'frame and on accepted fallback frames only (row block dwba); '
                                                 'the tracked columns are Born'),
                                   sigma='max(Poisson, left/right difference) per frame (model.sigma side)',
                                   composition='given by the user; a series does not identify the composition'))
    return dict(schema=SUMMARY_SCHEMA, mode=info['mode'], settings=dict(recipe=recipe, calibration=header['calibration'],
                                                                         source=header['source'], workers=info.get('workers')),
                provenance=provenance, conventions=conventions, counts=counts, statistics=stats, populations=pops,
                timing=timing, live=info.get('live'), batch=info.get('batch'), resumed_frames=info.get('resumed_frames'),
                interrupted=info.get('interrupted'), notes=list(recipe.get('notes', []))+list(info.get('notes', [])),
                created_unix=time.time())


# ---------------------------------------------------------------------------- figure and page
def write_trend(path, rows, title=''):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fitted = [r for r in rows if r['mode'] != 'skipped']
    fig, axes = plt.subplots(4, 1, figsize=(7.2, 9.6), sharex=True, facecolor=SURFACE)
    panels = (('R_nm', 'R (nm)'), ('D_nm', 'D (nm)'), ('amplitude', 'amplitude'), (None, 'chi2 / point'))
    independent = bool(fitted) and all(r['mode'] == 'independent' for r in fitted)
    xs = np.array([r['index'] if independent else r['time_s'] for r in fitted], float)
    labels = [p['pop'] for p in fitted[0]['populations']] if fitted else []
    for ax, (key, name) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, lw=0.6)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        for side in ('left', 'bottom'):
            ax.spines[side].set_color('#c3c2b7')
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.set_ylabel(name, color='#52514e', fontsize=9)
        if key is None:
            ax.plot(xs, [r['chi2_per_point'] for r in fitted], color='#52514e', lw=1.2, label='chi2 / point')
            base = np.array([np.nan if r.get('chi2_baseline') is None else r['chi2_baseline'] for r in fitted], float)
            if np.isfinite(base).any():
                ax.plot(xs, base, color=MUTED, lw=1.0, ls='--', label='baseline (median of last 10)')
            jump = np.array([bool(r['flags'].get('chi2_jump')) for r in fitted])
            if jump.any():
                ax.plot(xs[jump], np.array([r['chi2_per_point'] for r in fitted])[jump], 'x', color=SERIOUS, ms=6,
                        label='chi2 jump')
            fb = np.array([bool(r['flags']['fallback']) for r in fitted])
            if fb.any():
                ax.plot(xs[fb], np.array([r['chi2_per_point'] for r in fitted])[fb], 'v', mfc='none', mec=INK, ms=7,
                        label='fallback ran')
            if fitted:
                ax.set_yscale('log')
                ax.legend(fontsize=7, frameon=False, labelcolor='#52514e', loc='upper right')
            continue
        values_all = []
        for i, label in enumerate(labels):
            v = np.full(len(fitted), np.nan)
            e = np.full(len(fitted), np.nan)
            for k, r in enumerate(fitted):
                pop = next((p for p in r['populations'] if p['pop'] == label), None)
                if pop is None or pop['state'] != 'on' or pop.get(key) is None:
                    continue
                v[k] = pop[key]
                err = pop.get(f'{key}_err')
                e[k] = err if err is not None else np.nan
            colour = SERIES[i % len(SERIES)]
            kind = fitted[0]['populations'][i]['type']
            ok = np.isfinite(e) & np.isfinite(v) & (e < 0.5*np.abs(v))     # weak values: no band
            if ok.any() and key != 'amplitude':
                ax.fill_between(xs, np.where(ok, v-e, np.nan), np.where(ok, v+e, np.nan), color=colour, alpha=0.18, lw=0)
            ax.plot(xs, v, color=colour, lw=1.6 if len(xs) > 40 else 1.2, marker=MARKERS[i % 4],
                    ms=0 if len(xs) > 40 else 4, label=f'{label} ({kind})')
            cps = np.array([f'{label}.{key}' in r['flags'].get('change_points', []) for r in fitted])
            if cps.any():
                ax.plot(xs[cps], v[cps], marker=MARKERS[i % 4], ls='', mfc=SURFACE, mec=colour, mew=1.6, ms=8)
            values_all.append(v[np.isfinite(v) & (v > 0)])
        values_all = np.concatenate(values_all) if values_all else np.array([])
        if len(values_all) and values_all.max() > 5*values_all.min():
            ax.set_yscale('log')
        if key == 'R_nm' and labels:
            ax.legend(fontsize=7, frameon=False, labelcolor='#52514e', loc='best', ncol=min(len(labels), 2))
    axes[-1].set_xlabel('file index' if independent else 'time (s)', color='#52514e', fontsize=9)
    fig.suptitle(title or 'series1d', color=INK, fontsize=10, x=0.02, ha='left')
    fig.text(0.02, 0.005, 'bands: 1-sigma (not drawn above 50 %); open markers: flagged change points; gaps: population off',
             color=MUTED, fontsize=7)
    fig.tight_layout(rect=(0, 0.015, 1, 0.98))
    fig.savefig(path, dpi=110, facecolor=SURFACE)
    plt.close(fig)
    return str(path)


def _num(v, digits=4):
    if v is None:
        return '–'
    if isinstance(v, float):
        return f'{v:.{digits}g}'
    return str(v)


def write_html(path, summary, png_path):
    recipe = summary['settings']['recipe']
    m, c, t, s = recipe['model'], summary['counts'], summary['timing'], summary['statistics']
    png = base64.b64encode(Path(png_path).read_bytes()).decode('ascii') if Path(png_path).exists() else ''
    chi = s.get('chi2_per_point') or {}
    track = t.get('tracked_frame_ms') or {}
    rows = [('mode', summary['mode']), ('composition', m['composition']),
            ('fit mode / qz (nm⁻¹)', f"{m['mode']} / {_num(m['qz']) if m['qz'] is not None else 'vacuum qz of the cut'}"),
            ('sphere model', m['sphere_model']+(' (first/fallback frames only; tracking is Born)' if m['sphere_model'] == 'dwba' else '')),
            ('frames (fitted / skipped)', f"{c['frames']} ({c['fitted']} / {c['frames']-c['fitted']})"),
            ('chi²/point median (p95)', f"{_num(chi.get('median'))} ({_num(chi.get('p95'))})"),
            ('frames with chi² jump', f"{s.get('chi2_jump_frames', 0)} ({100*s.get('chi2_jump_fraction', 0):.1f} %)"),
            ('fallbacks run / accepted', f"{c['fallbacks_run']} / {c['fallbacks_accepted']}"),
            ('frames with a population off', c['frames_with_off_population']),
            ('identity swaps', s.get('identity_swaps', 0)),
            ('tracked frame time median (p95)', f"{_num(track.get('median'))} ms ({_num(track.get('p95'))} ms)"),
            ('throughput', f"{_num(t.get('frames_per_second'))} frames/s")]
    if summary.get('live') and summary['live'].get('latency_p95_s') is not None:
        lv = summary['live']
        rows.append((f"live {lv['replay_hz']:g} Hz: arrival→result p95, max backlog",
                     f"{1e3*lv['latency_p95_s']:.0f} ms, {lv['max_backlog']} frames"))
    pop_rows = []
    for label, p in summary['populations'].items():
        def cell(key):
            v = p.get(key)
            return '–' if v is None else f"{v['mean']:.4g} ± {v['std']:.2g}"
        pop_rows.append(f"<tr><th>{html.escape(label)}</th><td>{html.escape(str(p['type']))}</td><td>{cell('R_nm')}</td>"
                        f"<td>{cell('D_nm')}</td><td>{p['frames_on']} / {p['frames_off']}</td></tr>")
    notes = ''.join(f'<li>{html.escape(n)}</li>' for n in summary.get('notes', []))
    body = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>series1d summary</title>
<style>
:root {{ color-scheme: light; }}
body {{ margin: 0; padding: 16px; background: #f9f9f7; color: #0b0b0b; font: 15px/1.45 -apple-system, "Segoe UI", sans-serif; }}
main {{ max-width: 760px; margin: 0 auto; }}
h1 {{ font-size: 19px; margin: 0 0 4px; }}
h2 {{ font-size: 15px; margin: 22px 0 6px; }}
p.sub {{ margin: 0 0 14px; color: #52514e; font-size: 13px; overflow-wrap: anywhere; }}
table {{ border-collapse: collapse; width: 100%; background: #fcfcfb; }}
th, td {{ text-align: left; padding: 6px 8px; border-bottom: 1px solid #e1e0d9; vertical-align: top; font-size: 14px; }}
th {{ font-weight: 600; color: #52514e; width: 46%; }}
td {{ font-variant-numeric: tabular-nums; overflow-wrap: anywhere; }}
.wrap {{ overflow-x: auto; }}
img {{ width: 100%; height: auto; display: block; border: 1px solid #e1e0d9; }}
ul {{ padding-left: 18px; color: #52514e; font-size: 13px; }}
</style></head><body><main>
<h1>series1d summary</h1>
<p class="sub">{html.escape(str(summary['settings']['source'].get('path') or summary['settings']['source'].get('directory')))}<br>
{html.escape(str(summary['provenance'].get('series1d_version')))} · recipe {html.escape(str(summary['provenance'].get('recipe_sha256'))[:12])}</p>
<table>{''.join(f'<tr><th>{html.escape(str(k))}</th><td>{html.escape(str(v))}</td></tr>' for k, v in rows)}</table>
<h2>Populations (mean ± std over the frames where they are on)</h2>
<div class="wrap"><table><tr><th>population</th><th>type</th><th>R (nm)</th><th>D (nm)</th><th>frames on / off</th></tr>
{''.join(pop_rows)}</table></div>
<h2>Trend</h2>
<img alt="R, D, amplitude per population and chi2 per point against time" src="data:image/png;base64,{png}">
<h2>Conventions</h2>
<ul><li>q in nm⁻¹, lengths in nm; number-weighted size distributions (σ = relative width); Hosemann paracrystal, w = w_rel·D.</li>
<li>Spheres and random cylinders on a Yoneda cut use |q| = √(q∥² + qz²).</li>
<li>The composition is given by the user; a series does not identify it.</li>
<li>Per-frame values with flags are in series.csv and frames.jsonl.</li>{notes}</ul>
</main></body></html>
"""
    Path(path).write_text(body, encoding='utf-8')
    return str(path)


def write_all(out, recipe, header, rows, summary):
    out = Path(out)
    write_csv(out/'series.csv', recipe, rows)
    m = recipe['model']
    title = f"{m['composition']}  ({m['mode']}), {len(rows)} frames"
    write_trend(out/'trend.png', rows, title)
    summary['outputs'] = dict(frames='frames.jsonl', csv='series.csv', trend='trend.png', summary='summary.json',
                              page='summary.html', csv_columns=csv_columns(n_populations(recipe, rows)))
    (out/'summary.json').write_text(json.dumps(fit_schema.jsonable(summary), indent=1, allow_nan=False),
                                    encoding='utf-8')
    write_html(out/'summary.html', summary, out/'trend.png')
    return summary
