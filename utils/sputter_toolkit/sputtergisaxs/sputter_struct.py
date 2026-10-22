"""Size/structure trends with per-value reliability for in-situ sputter GISAXS on polymer.

Pipeline (scan of CBF frames -> structure table):
  intake (native counts per frame) -> exact 0.5 s count bins (0.1 s for the
  first 5 s) -> direct features + stage rule -> polymer / TS networks ->
  optional merge of converged physical refinement -> structure.csv with, for
  every R and D value, its source, typical error and ordering reliability.

Example:
    from sputtergisaxs.sputter_struct import analyze_scan
    analyze_scan("scan_folder", "out", energy_ev=12000, sdd_mm=2500, alpha_i_deg=.4, metal="Ag",
                 polymer_expert="experts/polymer", ts_expert="experts/TS", device="cpu")
Reliability numbers come from configs/sputter_struct_reliability_v1.json.
R is an equivalent-hemisphere (h=R, flat base) radius; burial and polymer
thickness are not reliable and are reported only as conditional values.
"""
from __future__ import annotations
from pathlib import Path
import csv, gzip, json, math, time
import numpy as np

PKG = Path(__file__).resolve().parents[1]
RELIABILITY_FILE = PKG / 'configs' / 'sputter_struct_reliability_v1.json'
PEAK = ('bilateral', 'one_side')


def say(progress, stage, fraction, message):
    """One machine-readable progress line (the GUI parses SPUTTER_PROGRESS)."""
    line = f'SPUTTER_PROGRESS\t{stage}\t{fraction:.3f}\t{message}'
    (progress or print)(line)


def reliability():
    return json.loads(RELIABILITY_FILE.read_text(encoding='utf-8'))


def _num(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


# ----------------------------------------------------------------- analysis --
def analyze_scan(input_path, out, *, energy_ev, sdd_mm, alpha_i_deg, metal, polymer_expert, ts_expert,
                 beam_col=None, beam_row=None, sample_roll_deg=0., mask=None, bin_seconds=.5, workers=4,
                 device='cpu', polymer_type='polymer', refined_dir=None, progress=None, rate_nm_s=None, t0_s=None):
    """Run the whole chain for one scan folder; returns the structure.csv path.

    Also writes structure_physical.csv / structure_physical_summary.json (see
    sputter_physical); rate_nm_s and t0_s give the nominal deposited thickness.
    """
    from .polymer_dev_intake import prepare
    from .polymer_review_data import intake_rows, bin_ids, aggregate, measure_groups
    from .polymer_review_infer import infer_groups
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    if metal not in ('Ag', 'Au', 'Cu'):
        raise ValueError('metal must be Ag, Au or Cu (Cu uses the Ag model)')
    intake = out / 'intake'
    if not (intake / 'run.json').exists():
        say(progress, 'intake', 0., 'reading CBF frames (raw data read-only)')
        prepare(input_path, intake, energy_ev=energy_ev, sdd_mm=sdd_mm, alpha_i_deg=alpha_i_deg, metal=metal,
                beam_col=beam_col, beam_row=beam_row, sample_roll_deg=sample_roll_deg, maskpath=mask,
                workers=workers, system_note=f'{metal}/{polymer_type}')
    excluded = json.loads((intake / 'run.json').read_text(encoding='utf-8')).get('excluded_frames', [])
    say(progress, 'intake', 1., 'frames read' + (f'; {len(excluded)} damaged CBF excluded: '
        + ', '.join(e['frame'] for e in excluded[:5]) + ' (intake/excluded_frames.json)' if excluded else ''))
    binned = out / 'binned'
    if not (binned / 'run.json').exists():
        say(progress, 'binning', 0., f'summing counts into {bin_seconds:g} s windows')
        run, rows = intake_rows(intake)
        ids = bin_ids([r['time_s'] for r in rows], seconds=bin_seconds)
        aggregate(intake, binned, run, rows, ids, nominal_seconds=bin_seconds)
    say(progress, 'binning', 1., 'windows written')
    if not (binned / 'observations.json.gz').exists():
        say(progress, 'features', 0., 'direct features and stage rule per window')
        measure_groups(binned, workers)
    say(progress, 'features', 1., 'features done')
    if not (binned / 'inference.json').exists():
        say(progress, 'network', 0., f'polymer/TS networks on {device}')
        infer_groups(binned, polymer_expert, ts_expert, device=device)
    elif not (binned / 'frames_alt.csv').exists():
        from .polymer_review_infer import infer_alternate
        say(progress, 'network', .5, 'other network on every window (for the physical path)')
        infer_alternate(binned, polymer_expert, ts_expert, device=device)
    say(progress, 'network', 1., 'network done')
    csv_path = write_structure(binned, out, metal=metal, refined_dir=refined_dir)
    from .sputter_physical import write_physical
    write_physical(out, binned=binned, rate_nm_s=rate_nm_s, t0_s=t0_s)
    say(progress, 'done', 1., f'{csv_path} ({time.monotonic() - t0:.0f} s)')
    return csv_path


# ---------------------------------------------------------- structure table --
def _refined(refined_dir):
    out = {}
    if refined_dir is None:
        return out
    for p in sorted(Path(refined_dir).glob('*.json')):
        w = json.loads(p.read_text(encoding='utf-8'))
        if 'key' in w:
            out[w['key']] = w
    return out


def _selected(w):
    if not w or w.get('status') != 'converged':
        return None
    return next((m for m in w.get('modes', []) if m['mode_id'] == w.get('selected')), None)


def structure_rows(frames_csv, *, metal, refined_dir=None):
    rel = reliability()['parameters']
    model_metal = 'Ag' if metal == 'Cu' else metal
    with Path(frames_csv).open(encoding='utf-8') as f:
        frames = list(csv.DictReader(f))
    refined = _refined(refined_dir)
    rows = []
    for fr in frames:
        key = fr['frame']; module = fr.get('module', '')
        peak = fr.get('feature_status') in PEAK
        w = refined.get(key); sel = _selected(w)
        row = dict(frame=key, time_s=_num(fr.get('time_s')), stage=fr.get('stage'), module=module,
                   feature_status=fr.get('feature_status'), q_star_nm_inv=_num(fr.get('q_star_nm_inv')),
                   D_corr_nm=_num(fr.get('D_corr_nm')), network_R_nm=_num(fr.get('mode1_R_number_mean_nm')),
                   network_D_nm=_num(fr.get('mode1_D_nm')), network_mode1_probability=_num(fr.get('mode1_probability')),
                   refine_status=(w or {}).get('status', 'not_run'), refined_R_nm=None, refined_D_nm=None,
                   refined_deviance_per_dof_5pct=None, coequal_alternatives=0)
        if sel is not None:
            row.update(refined_R_nm=sel['values'].get('R_number_mean_nm'), refined_D_nm=sel['values'].get('D_nm'),
                       refined_deviance_per_dof_5pct=sel['scores'][0]['deviance_per_dof'],
                       coequal_alternatives=len(w.get('alternatives', [])))
        for name, net_key, ref_key in (('R', 'network_R_nm', 'refined_R_nm'), ('D', 'network_D_nm', 'refined_D_nm')):
            value = source = grade = None; reason = ''
            if module == 'TS_layer':
                reason = 'late TS window: island R/D not defined (see L_TS_nm)'
            elif name == 'D' and not peak:
                reason = 'no resolved peak: D not reported'
            else:
                ok = sel is not None and sel.get('identified', {}).get(
                    'R_number_mean_nm' if name == 'R' else 'D_nm', False) and sel.get('compatible5')
                if ok and row[ref_key] is not None:
                    value, source = row[ref_key], 'refined'
                    grade = 'A' if row['coequal_alternatives'] == 0 else 'B'
                elif row[net_key] is not None:
                    value, source, grade = row[net_key], 'network', 'C'
                else:
                    reason = 'no estimate'
            r = rel[name][model_metal][source] if source else {}
            err = r.get('typical_rel_scatter')
            row.update({f'{name}_nm': value, f'{name}_source': source, f'{name}_grade': grade, f'{name}_reason': reason,
                        f'{name}_typical_rel_error': err,
                        f'{name}_lo_nm': value / (1 + err) if value is not None and err is not None else None,
                        f'{name}_hi_nm': value * (1 + err) if value is not None and err is not None else None,
                        f'{name}_P90_rel_error': r.get('P90_rel_error'), f'{name}_order_accuracy': r.get('order_accuracy_1p2x')})
        band = rel.get('D_over_direct', {}).get('band_90', [0.587, 1.75])
        if row.get('D_nm') is not None and row['D_corr_nm']:
            ratio = row['D_nm'] / row['D_corr_nm']
            row['D_over_direct'] = ratio
            row['D_direct_check'] = 'consistent' if band[0] <= ratio <= band[1] else 'outside simulated 90% range of D/(2pi/q*)'
        else:
            row['D_over_direct'] = None; row['D_direct_check'] = None
        row['L_TS_nm'] = _num(fr.get('mode1_spectral_domain_nm')) if module == 'TS_layer' else None
        # Shape descriptors for the real-space sketch: same source as R (refined values when R is refined).
        refined_shape = row.get('R_source') == 'refined' and sel is not None
        values = sel['values'] if refined_shape else {}
        row['sigma_log'] = values.get('radius_sigma_log') if refined_shape else _num(fr.get('mode1_radius_sigma_log'))
        row['spacing_disorder'] = values.get('spacing_disorder') if refined_shape else None
        row['burial_nm_conditional'] = values.get('burial_base_depth_nm') if refined_shape else _num(fr.get('mode1_burial_base_depth_nm'))
        row['thickness_nm_conditional'] = values.get('polymer_thickness_nm') if refined_shape else _num(fr.get('mode1_polymer_thickness_nm'))
        row['shape_source'] = 'refined' if refined_shape else ('network' if row.get('R_source') else None)
        if module == 'TS_layer':
            row['TS_metal_fraction_conditional'] = _num(fr.get('mode1_metal_fraction'))
            row['TS_film_thickness_nm_conditional'] = _num(fr.get('mode1_film_thickness_nm'))
        else:
            row['TS_metal_fraction_conditional'] = row['TS_film_thickness_nm_conditional'] = None
        row['metal'] = metal
        row['note'] = 'Cu approximated with the Ag model, for reference only (errors not validated)' if metal == 'Cu' else ''
        rows.append(row)
    return rows


def window_scatter(rows, name):
    """Robust per-window precision from adjacent independent windows (log units)."""
    seq = [(i, math.log(r[f'{name}_nm'])) for i, r in enumerate(rows) if r.get(f'{name}_nm')]
    d = dict(seq); diffs = [d[i + 1] - d[i] for i in d if i + 1 in d]
    if len(diffs) < 5:
        return None
    return float(np.expm1(1.4826 * np.median(np.abs(diffs)) / math.sqrt(2)))


def summary(rows, metal):
    s = dict(windows=len(rows), metal=metal)
    for name in ('R', 'D'):
        g = [r[f'{name}_grade'] for r in rows]
        s[name] = dict(available=sum(x is not None for x in g), refined_A=g.count('A'), refined_B=g.count('B'),
                       network_C=g.count('C'), sequence_S=g.count('S'), sequence_T=g.count('T'),
                       window_to_window_scatter=window_scatter(rows, name))
    s['D']['checked_against_direct'] = sum(r.get('D_direct_check') is not None for r in rows)
    s['D']['outside_direct_band'] = sum(bool(r.get('D_direct_check')) and r['D_direct_check'] != 'consistent' for r in rows)
    s['TS_windows'] = sum(r['module'] == 'TS_layer' for r in rows)
    s['grades'] = {'A': 'converged physical refinement, unique solution', 'B': 'converged refinement with an equally good alternative',
                   'C': 'network estimate only'}
    s['reliability'] = reliability()
    return s


def write_structure(binned, out, *, metal, refined_dir=None):
    out = Path(out); rows = structure_rows(Path(binned) / 'frames.csv', metal=metal, refined_dir=refined_dir)
    path = out / 'structure.csv'
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    (out / 'structure_summary.json').write_text(json.dumps(summary(rows, metal), indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    plot_structure(rows, out / 'structure_trend.png', title=Path(out).name)
    return path


def plot_structure(rows, path, title=''):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    has_ts = any(r['L_TS_nm'] for r in rows)
    fig, axes = plt.subplots(3 if has_ts else 2, 1, figsize=(7.2, 7.8 if has_ts else 5.6), sharex=True, constrained_layout=True)
    colors = {'A': '#1f5fa8', 'B': '#6a8fc4', 'C': '#c26a00', 'S': '#2a8a3e', 'T': '#c0392b'}
    labels = {'A': 'refined, unique', 'B': 'refined, alternative exists', 'C': 'network only',
              'S': 'time-consistent, data-supported', 'T': 'time-consistent, constraint-driven'}
    for ax, name, ylabel in zip(axes, ('R', 'D'), ('R (nm), hemisphere-equivalent', 'D (nm), peak windows')):
        for g in ('C', 'B', 'A', 'S', 'T'):
            sel = [r for r in rows if r[f'{name}_grade'] == g and r['time_s'] is not None]
            if not sel:
                continue
            if name == 'D':
                bad = [r for r in sel if r.get('D_direct_check') not in (None, 'consistent')]
                if bad:
                    ax.plot([r['time_s'] for r in bad], [r['D_nm'] for r in bad], 'o', ms=6, mfc='none', mec='red', mew=.8,
                            label='D vs 2π/q* outside simulated 90% range' if g == 'C' else None)
            t = np.array([r['time_s'] for r in sel]); v = np.array([r[f'{name}_nm'] for r in sel])
            lo = np.array([r[f'{name}_lo_nm'] for r in sel]); hi = np.array([r[f'{name}_hi_nm'] for r in sel])
            ax.errorbar(t, v, yerr=[v - lo, hi - v], fmt='o', ms=2.6, lw=.5, color=colors[g], alpha=.85, label=labels[g])
        ax.set_ylabel(ylabel); ax.grid(alpha=.25)
        if name == 'D':
            dc = [(r['time_s'], r['D_corr_nm']) for r in rows if r['D_corr_nm'] and r['time_s'] is not None]
            if dc:
                ax.plot(*zip(*dc), 'x', ms=2.5, color='0.45', label='2π/q* (direct)')
        ax.legend(fontsize=7, loc='best')
    if has_ts:
        ts = [(r['time_s'], r['L_TS_nm']) for r in rows if r['L_TS_nm'] and r['time_s'] is not None]
        axes[2].plot(*zip(*ts), 'o', ms=2.5, color='#5a7d2a'); axes[2].set_ylabel('TS length L (nm)'); axes[2].grid(alpha=.25)
    axes[-1].set_xlabel('time (s)')
    fig.suptitle(f'{title}: size/structure with typical error bars (simulation-based)', fontsize=9)
    fig.savefig(path, dpi=180); plt.close(fig)
