"""Command line of fit1d (writes result.json, result.csv, fit.png into --out).

    python3 -m research.fit1d INPUT [--composition C] [--mode gisaxs|saxs] [--qz X | --geometry GEOMETRY.json]
        [--policy fast|auto|robust|preview] [--fix pop1.R_nm=2.5] [--range pop2.D_nm=5:8]
        [--lowq lorentz|powerlaw|none] [--structure-factor on|off] [--out DIR] [--workers N]
        [--sphere-model born|dwba [--alpha-f DEG] [--substrate-delta D --substrate-beta B]]

qz (GISAXS): --qz, else --geometry (key qz_vacuum_at_yoneda_nm_inv), else the cut metadata
(qz_at_yoneda_mean of a yoneda_cut npz: the vacuum qz of the rows that were actually cut); otherwise 0
with a note. When --qz/--geometry differs from the cut metadata by more than 0.5 % a note says so.
SAXS mode has no qz: --qz/--geometry are errors there. Default composition: auto:vc:3 (gisaxs),
auto:sphere:3 (saxs). Exit code 0 on success, 2 on any input/usage error (message on stderr).
--workers N > 1 starts a 'spawn' process pool for the robust stages.
--sphere-model dwba (GISAXS, spheres on a substrate): after the fast Born stages the spheres are refitted
with the four-term DWBA model (stage S5). It needs wavelength_nm and alpha_i_deg (cut metadata 'geometry' or
--geometry) and the exit angle of the cut (cut metadata yoneda_alpha_f_deg, GEOMETRY.json yoneda_alpha_f_deg,
or --alpha-f); substrate Si unless --substrate-delta/--substrate-beta are given. qz is then the vacuum qz.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DEFAULT_COMPOSITION = {'gisaxs': 'auto:vc:3', 'saxs': 'auto:sphere:3'}


def build_parser():
    p = argparse.ArgumentParser(prog='python3 -m research.fit1d', description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('input', type=Path, help='yoneda_cut .npz, or CSV/TXT/DPDAK text with q, I[, sigma]')
    p.add_argument('--composition', default=None, help="e.g. 'sphere,vc,vc', 'vc,vc', 'auto:vc:3'")
    p.add_argument('--mode', default='gisaxs', choices=('gisaxs', 'saxs'))
    g = p.add_mutually_exclusive_group()
    g.add_argument('--qz', type=float, default=None, help='out-of-plane momentum (nm^-1) for spheres/random cylinders')
    g.add_argument('--geometry', type=Path, default=None, help='GEOMETRY.json with qz_vacuum_at_yoneda_nm_inv')
    p.add_argument('--policy', default='auto', choices=('fast', 'auto', 'robust', 'preview'))
    p.add_argument('--fix', action='append', default=[], metavar='NAME=VALUE', help='e.g. pop1.R_nm=2.5, nu=2, background=0')
    p.add_argument('--range', action='append', default=[], metavar='NAME=LO:HI', help='e.g. pop2.D_nm=5:8, powerlaw_n=2:4')
    p.add_argument('--lowq', default=None, choices=('lorentz', 'powerlaw', 'none'),
                   help='default: lorentz (gisaxs), none (saxs)')
    p.add_argument('--lowq-nu', default='2', help="Lorentzian exponent (default 2); 'free' fits it in [1, 6]")
    p.add_argument('--structure-factor', default=None, choices=('on', 'off'), help='default: on (gisaxs), off (saxs)')
    p.add_argument('--sphere-model', default='born', choices=('born', 'dwba'),
                   help='dwba: four-term DWBA for spheres on a substrate (stage S5; needs the cut geometry)')
    p.add_argument('--alpha-f', type=float, default=None,
                   help='exit angle of the cut in degrees (dwba; default: cut metadata, else the critical angle of the substrate)')
    p.add_argument('--wavelength-nm', type=float, default=None, help='X-ray wavelength in nm (dwba, when the input has no geometry)')
    p.add_argument('--alpha-i', type=float, default=None, help='incident angle in degrees (dwba, when the input has no geometry)')
    p.add_argument('--substrate-delta', type=float, default=None, help='substrate refractive decrement (default: Si)')
    p.add_argument('--substrate-beta', type=float, default=None, help='substrate absorption index (default: Si)')
    p.add_argument('--sigma-floor', type=float, default=0.0,
                   help='relative systematic error added to sigma in quadrature, e.g. 0.03 (default 0 = counting statistics only)')
    p.add_argument('--heads', type=int, default=2)
    p.add_argument('--max-steps', type=int, default=40)
    p.add_argument('--workers', type=int, default=1)
    p.add_argument('--q-unit', default='nm^-1', choices=('nm^-1', 'A^-1'))
    p.add_argument('--model-dir', type=Path, default=None)
    p.add_argument('--out', type=Path, default=None, help='output directory (default ./fit1d_<input stem>)')
    p.add_argument('--no-plot', action='store_true')
    p.add_argument('--quiet', action='store_true')
    return p


QZ_NOTE_TOLERANCE = 0.005


def resolve_qz(args, meta):
    """(qz, source text, note or None). Raises ValueError for unusable input."""
    if args.mode != 'gisaxs':
        if args.qz is not None or args.geometry is not None:
            raise ValueError('--qz/--geometry apply to --mode gisaxs only (SAXS mode has no qz)')
        return 0.0, 'saxs (no qz)', None
    cut_qz = meta.get('qz_at_yoneda_mean')
    cut_qz = float(cut_qz) if cut_qz is not None else None
    if args.qz is not None:
        qz, source = float(args.qz), '--qz'
    elif args.geometry is not None:
        path = args.geometry.expanduser()
        try:
            geometry = json.loads(path.read_text(encoding='utf-8'))
        except json.JSONDecodeError as exc:
            raise ValueError(f'{path}: not a JSON file ({exc})') from None
        if not isinstance(geometry, dict) or 'qz_vacuum_at_yoneda_nm_inv' not in geometry:
            raise ValueError(f'{path}: key qz_vacuum_at_yoneda_nm_inv missing')
        qz, source = float(geometry['qz_vacuum_at_yoneda_nm_inv']), f'{path} (qz_vacuum_at_yoneda_nm_inv)'
    elif cut_qz is not None:
        return cut_qz, 'cut metadata qz_at_yoneda_mean (vacuum qz)', None
    else:
        return 0.0, 'default 0 (no --qz/--geometry/cut metadata)', None
    note = None
    if cut_qz is not None and abs(qz-cut_qz) > QZ_NOTE_TOLERANCE*abs(cut_qz):
        note = (f'qz = {qz:.4g} from {source.split(" ")[0]} differs from the cut metadata qz_at_yoneda_mean = '
                f'{cut_qz:.4g} (the vacuum qz of the rows actually cut) by {abs(qz/cut_qz-1)*100:.1f} %; '
                'omit --qz/--geometry to use the cut metadata.')
    return qz, source, note


def resolve_dwba(args, meta):
    """dwba dict for fit1d, or None. Raises ValueError when the geometry is incomplete."""
    if args.sphere_model != 'dwba':
        if any(v is not None for v in (args.alpha_f, args.substrate_delta, args.substrate_beta, args.wavelength_nm, args.alpha_i)):
            raise ValueError('--alpha-f/--alpha-i/--wavelength-nm/--substrate-delta/--substrate-beta apply to --sphere-model dwba only')
        return None
    if args.mode != 'gisaxs':
        raise ValueError('--sphere-model dwba applies to --mode gisaxs only')
    geometry = dict(meta.get('geometry') or {})
    alpha_f = meta.get('yoneda_alpha_f_deg')
    if args.geometry is not None:
        extra = json.loads(args.geometry.expanduser().read_text(encoding='utf-8'))
        geometry.update({k: extra[k] for k in ('wavelength_nm', 'alpha_i_deg') if k in extra})
        if alpha_f is None:
            alpha_f = extra.get('yoneda_alpha_f_deg')
    if args.alpha_f is not None:
        alpha_f = args.alpha_f
    if args.wavelength_nm is not None:
        geometry['wavelength_nm'] = args.wavelength_nm
    if args.alpha_i is not None:
        geometry['alpha_i_deg'] = args.alpha_i
    missing = [name for name, value in (('the wavelength (--wavelength-nm)', geometry.get('wavelength_nm')),
                                        ('the incident angle (--alpha-i)', geometry.get('alpha_i_deg'))) if value is None]
    if missing:
        raise ValueError(f'--sphere-model dwba needs {" and ".join(missing)}, or --geometry GEOMETRY.json')
    spec = dict(wavelength_nm=float(geometry['wavelength_nm']), alpha_i_deg=float(geometry['alpha_i_deg']),
                alpha_f_deg=None if alpha_f is None else float(alpha_f))
    if args.substrate_delta is not None:
        spec.update(substrate_delta=args.substrate_delta, substrate_beta=args.substrate_beta or 0.0)
    else:
        spec['substrate'] = 'Si'
    return spec


def summary(result):
    lines = []
    for r in result['results']:
        mark = '*' if r['indistinguishable'] else ' '
        lines.append(f"{mark}{r['rank']:2d} {r['composition_text']:<16} chi2/pt={r['chi2_per_point']:.3f} "
                     f"dBIC={r['delta_bic']:.1f} stop={r['stop_stage']} flags={','.join(r['flags']['raised']) or '-'}")
        for pop in r['populations']:
            def f(key):
                v, e = pop.get(key), pop.get(f'{key}_err')
                if v is None:
                    return '-'
                return f'{v:.4g}' + (f'+-{e:.2g}' if e is not None else '')
            lines.append(f"      {pop['pop']} {pop['type']:<17} R={f('R_nm')} nm  D={f('D_nm')} nm  "
                         f"sR={f('sigma_R_rel')}  w/D={f('w_rel')}  A={pop['amplitude']:.3g}")
    return '\n'.join(lines)


def main(argv=None):
    args = build_parser().parse_args(argv)
    from research.fit1d import constraints as cons
    from research.fit1d.api import fit1d
    from research.fit1d.io import load_input
    from research.fit1d.schema import jsonable, plot, to_csv
    args.input = args.input.expanduser()
    composition = args.composition or DEFAULT_COMPOSITION[args.mode]
    try:
        q, y, s, meta = load_input(args.input, q_unit=args.q_unit)
        dwba = resolve_dwba(args, meta)
        if dwba is not None:
            args.qz_given, args.qz, args.geometry_for_qz = args.qz, None, args.geometry
            qz, qz_source, qz_note = 0.0, 'vacuum qz of the DWBA geometry', None
        else:
            qz, qz_source, qz_note = resolve_qz(args, meta)
        if args.lowq_nu != 'free':
            try:
                float(args.lowq_nu)
            except ValueError:
                raise ValueError(f"--lowq-nu must be a number or 'free', got {args.lowq_nu!r}") from None
        constraints = cons.parse_cli(args.fix, args.range)
        result = fit1d(q, y, s, composition=composition, mode=args.mode, qz=qz,
                       structure_factor=None if args.structure_factor is None else args.structure_factor == 'on',
                       lowq=args.lowq, lowq_nu=None if args.lowq_nu == 'free' else float(args.lowq_nu),
                       constraints=constraints, policy=args.policy, heads=args.heads, max_steps=args.max_steps,
                       workers=args.workers, model_dir=args.model_dir, dwba=dwba, sigma_floor=args.sigma_floor,
                       source=jsonable(dict(file=str(args.input), qz_source=qz_source, reader=meta)))
    except (ValueError, OSError, KeyError) as exc:        # ConstraintError, JSONDecodeError, FileNotFoundError included
        message = exc.args[0] if isinstance(exc, KeyError) and exc.args else exc
        print(f'fit1d: error: {message}', file=sys.stderr)
        return 2
    if qz_note:
        result['notes'].append(qz_note)
    out = (args.out or Path.cwd()/f'fit1d_{args.input.stem}').expanduser()
    try:
        out.mkdir(parents=True, exist_ok=True)
        (out/'result.json').write_text(json.dumps(result, indent=1, allow_nan=False), encoding='utf-8')
        to_csv(result, out/'result.csv')
        if not args.no_plot:
            plot(result, out/'fit.png')
    except OSError as exc:
        print(f'fit1d: error: cannot write the output to {out}: {exc}', file=sys.stderr)
        return 2
    if not args.quiet:
        qz = result['settings']['qz_nm_inv']
        print(f"fit1d {result['settings']['mode']} qz={qz:.4g} ({qz_source}); composition {composition}; "
              f"policy {args.policy}; {result['input']['n_points_used']} points; "
              f"{result['timing']['total_seconds']:.3f} s")
        print(summary(result))
        for note in result['notes']:
            print('note:', note)
        print(f'written: {out}/result.json, result.csv' + ('' if args.no_plot else ', fit.png'))
    return 0


if __name__ == '__main__':
    sys.exit(main())
