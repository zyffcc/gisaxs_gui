"""Sputter size/structure toolkit v1 (polymer substrates): R, D trends with per-value reliability.

analyze  a scan folder of CBF frames -> OUT/structure.csv, structure_summary.json, structure_trend.png
         and the physical path OUT/structure_physical.csv, structure_physical_summary.json
    python scripts/sputter_struct_v1.py analyze --input SCAN_FOLDER --out OUT --energy-ev 12000 --sdd-mm 2500 \
        --alpha-i-deg 0.4 --metal Ag --polymer-expert EXPERTS/polymer --ts-expert EXPERTS/TS [--beam-col C --beam-row R] [--mask MASK.npy]
        [--rate-nm-s RATE --t0-s T0]   (deposition rate, e.g. from a QCM, for the nominal thickness rate*(t-t0))
physical rebuild only the physical-path files of an analysed folder (fast, no frames read); with a rate also the
         literature method and its comparison with published growth (summary key 'literature')
    python scripts/sputter_struct_v1.py physical --folder OUT [--rate-nm-s RATE] [--t0-s T0] [--polymer-type PS]
        [--rate-error 0.1] [--t0-error 2]   (errors of the rate and t0 in the percolation interval)
sequence time-consistent path (all network components, physical transition costs) -> OUT/structure_sequence.csv
    python scripts/sputter_struct_v1.py sequence --binned OUT/binned --out OUT --metal Ag --polymer-expert EXPERTS/polymer --ts-expert EXPERTS/TS
select   learned selector (trained/tested on simulated growth series; D passed its test, R did not) -> OUT/structure_selected.csv
    python scripts/sputter_struct_v1.py select --binned OUT/binned --out OUT --metal Ag --polymer-expert EXPERTS/polymer --ts-expert EXPERTS/TS
table    rebuild the structure table (and the physical-path files) from an existing binned folder, optionally merging refined windows
    python scripts/sputter_struct_v1.py table --binned OUT/binned --out OUT --metal Ag [--refined REFINED/windows] [--rate-nm-s RATE] [--t0-s T0] [--polymer-type PS]
        [--rate-error 0.1] [--t0-error 2]
Progress lines start with SPUTTER_PROGRESS (stage, fraction, message) for the GIMaP Sputter page; reading frames also
prints INTAKE_PLAN <chunks> <frames>, INTAKE_CHUNK <name> <index> <frames> <seconds> and, for each damaged frame that
is left out, INTAKE_EXCLUDED <frame> <reason>.
"""
from __future__ import annotations
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(name, '1')
from pathlib import Path
import argparse, sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _thickness_args(parser, polymer=True):
    parser.add_argument('--rate-nm-s', type=float, help='deposition rate in nm/s (e.g. QCM) for the nominal thickness '
                        'and the literature method')
    parser.add_argument('--t0-s', type=float, help='time (s) at which deposition started, for the nominal thickness (default 0)')
    if polymer:
        parser.add_argument('--polymer-type', help='host for the literature band (default: the polymer of the analysis)')
        parser.add_argument('--rate-error', type=float, help='relative error of the rate in the literature percolation '
                            'interval (default 0.1)')
        parser.add_argument('--t0-error', type=float, help='error of t0 in s in the literature percolation interval '
                            '(default 2 without --t0-s, 0.5 with it)')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    a = sub.add_parser('analyze')
    a.add_argument('--input', type=Path, required=True); a.add_argument('--out', type=Path, required=True)
    g = a.add_mutually_exclusive_group(required=True)
    g.add_argument('--energy-ev', type=float); g.add_argument('--wavelength-nm', type=float)
    a.add_argument('--sdd-mm', type=float, required=True); a.add_argument('--alpha-i-deg', type=float, required=True)
    a.add_argument('--metal', choices=('Ag', 'Au', 'Cu'), required=True); a.add_argument('--polymer-type', default='polymer')
    a.add_argument('--beam-col', type=float); a.add_argument('--beam-row', type=float); a.add_argument('--sample-roll-deg', type=float, default=0.)
    a.add_argument('--mask', type=Path); a.add_argument('--bin-seconds', type=float, default=.5)
    a.add_argument('--polymer-expert', type=Path, required=True); a.add_argument('--ts-expert', type=Path, required=True)
    a.add_argument('--device', default='cpu'); a.add_argument('--workers', type=int, default=4)
    a.add_argument('--refined', type=Path, help='folder of refined window JSON files to merge (optional)')
    _thickness_args(a, polymer=False)
    y = sub.add_parser('physical', help='rebuild structure_physical.csv / _summary.json of an analysed folder')
    y.add_argument('--folder', type=Path, required=True); y.add_argument('--binned', type=Path, help='default: FOLDER/binned')
    _thickness_args(y)
    q = sub.add_parser('sequence', help='time-consistent path through all network explanations (sputter_sequence_v1, amended: unresolved windows neutral)')
    q.add_argument('--binned', type=Path, required=True); q.add_argument('--out', type=Path, required=True)
    q.add_argument('--metal', choices=('Ag', 'Au', 'Cu'), required=True)
    q.add_argument('--polymer-expert', type=Path, required=True); q.add_argument('--ts-expert', type=Path, required=True)
    q.add_argument('--device', default='cpu')
    l = sub.add_parser('select', help='learned selector among the network explanations (sputter_selector_v1)')
    l.add_argument('--binned', type=Path, required=True); l.add_argument('--out', type=Path, required=True)
    l.add_argument('--metal', choices=('Ag', 'Au', 'Cu'), required=True)
    l.add_argument('--polymer-expert', type=Path, required=True); l.add_argument('--ts-expert', type=Path, required=True)
    l.add_argument('--model', type=Path, default=Path(__file__).resolve().parents[1] / 'configs' / 'sputter_selector_v1.json')
    l.add_argument('--device', default='cpu')
    t = sub.add_parser('table')
    t.add_argument('--binned', type=Path, required=True); t.add_argument('--out', type=Path, required=True)
    t.add_argument('--metal', choices=('Ag', 'Au', 'Cu'), required=True); t.add_argument('--refined', type=Path)
    _thickness_args(t)
    args = p.parse_args()
    from sputtergisaxs.sputter_struct import analyze_scan, write_structure, say
    if args.cmd == 'analyze':
        energy = args.energy_ev if args.energy_ev else 1239.841984 / args.wavelength_nm
        analyze_scan(args.input, args.out, energy_ev=energy, sdd_mm=args.sdd_mm, alpha_i_deg=args.alpha_i_deg, metal=args.metal,
                     polymer_expert=args.polymer_expert, ts_expert=args.ts_expert, beam_col=args.beam_col, beam_row=args.beam_row,
                     sample_roll_deg=args.sample_roll_deg, mask=args.mask, bin_seconds=args.bin_seconds, workers=args.workers,
                     device=args.device, polymer_type=args.polymer_type, refined_dir=args.refined, progress=lambda s: print(s, flush=True),
                     rate_nm_s=args.rate_nm_s, t0_s=args.t0_s)
    elif args.cmd == 'physical':
        from sputtergisaxs.sputter_physical import write_physical
        path = write_physical(args.folder, binned=args.binned, rate_nm_s=args.rate_nm_s, t0_s=args.t0_s,
                              polymer_type=args.polymer_type, rate_error=args.rate_error, t0_error_s=args.t0_error)
        say(lambda s: print(s, flush=True), 'done', 1., str(path))
    elif args.cmd == 'select':
        from sputtergisaxs.sputter_selector import selector_table
        args.out.mkdir(parents=True, exist_ok=True)
        path = selector_table(args.binned, args.out, metal=args.metal, polymer_expert=args.polymer_expert,
                              ts_expert=args.ts_expert, model=args.model, device=args.device)
        say(lambda s: print(s, flush=True), 'done', 1., str(path))
    elif args.cmd == 'sequence':
        from sputtergisaxs.sputter_sequence import sequence_table
        args.out.mkdir(parents=True, exist_ok=True)
        path = sequence_table(args.binned, args.out, metal=args.metal, polymer_expert=args.polymer_expert,
                              ts_expert=args.ts_expert, device=args.device)
        say(lambda s: print(s, flush=True), 'done', 1., str(path))
    else:
        args.out.mkdir(parents=True, exist_ok=True)
        path = write_structure(args.binned, args.out, metal=args.metal, refined_dir=args.refined)
        from sputtergisaxs.sputter_physical import write_physical
        write_physical(args.out, binned=args.binned, rate_nm_s=args.rate_nm_s, t0_s=args.t0_s, polymer_type=args.polymer_type,
                       rate_error=args.rate_error, t0_error_s=args.t0_error)
        say(lambda s: print(s, flush=True), 'done', 1., str(path))


if __name__ == '__main__':
    main()
