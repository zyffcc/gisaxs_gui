"""A simulated ALS 7.3.3 spray-deposition run, written frame by frame as the beamline writes it, to test live analysis
without beamtime.

    python tools/beamline_sim.py render --out <folder> [--pulses 3 --sample mix_cnf --detector "Pilatus 2M"]
    python tools/beamline_sim.py replay --src <rendered folder> --dst <folder GIMaP watches> [--speed 1]

render: synthetic GISAXS frames of Au spheres sprayed onto Si or a cellulose nanofibril (CNF) film, as 7.3.3 files:
        Pilatus EDF (int32, module gaps stored as 0, a few bad pixels at -2, header with Date and count_time) and a
        ``.txt`` side file per frame in the layout of the 7.3.3 side files (``Motors:``, ``Sample Alpha Stage``,
        ``Izero``, ``I1``, ``Beam Current``), plus ``manifest.json``: the schedule, the geometry and what each frame
        holds (deposited amount, wetness, spacing). A run: reference frames at every angle, then spray pulses with
        0.1 s frames during spraying and the first drying, 1 s frames afterwards, and frames at the other incidence
        angles between pulses (0.4 deg reference, 0.1, 0.25 and 0.6 deg by default).
replay: puts the frames into a folder at the times of the schedule (``--speed`` faster), each with its header Date and
        file time of that moment, and writes ``spray_log.csv`` (pulse, time, run_time_s) as the pulses happen.

The physics is deliberately simple (Born form factor of spheres with a Gaussian spread of radii, a 1D paracrystal,
Fresnel transmission of the substrate for the Yoneda band, a decaying wet term after each pulse, Debye-Bueche CNF
scattering, Poisson counts). It is a test signal for the software, not a model of any sample; nothing in it comes
from measured data.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PIXEL = 172e-6
ENERGY_KEV = 10.0
WAVELENGTH = 12.398419843320026 / ENERGY_KEV
K = 2 * math.pi / WAVELENGTH
DELTA = {"si": 4.88e-6, "cnf": 3.0e-6}
BETA = {"si": 7.4e-8, "cnf": 2.0e-9}


# ----------------------------------------------------------------------------------- schedule

def schedule(pulses: int, fast_s: float, period_s: float, angles: list[float], reference: float):
    """List of frames: (start_s, count_time_s, alpha_deg, phase, pulse)."""
    frames = []
    t = 0.0
    for _ in range(4):  # bare substrate before spraying, at the reference angle
        frames.append((t, 1.0, reference, "reference", 0)); t += 1.05
    for a in angles:  # and at each extra angle
        t += 0.3
        frames.append((t, 1.0, a, "reference", 0)); t += 1.05
    t += 0.3
    pulse_times = []
    for p in range(1, pulses + 1):
        t0 = t
        pulse_times.append(t0)
        n_fast = int(round(fast_s / 0.1))
        for i in range(n_fast):
            frames.append((t0 + 0.1 * i, 0.097, reference, "spray" if 0.1 * i < 0.2 else "drying-fast", p))
        t = t0 + fast_s
        while t < t0 + period_s - (len(angles) * 1.35 + 1.1):
            frames.append((t, 1.0, reference, "drying", p)); t += 1.05
        for a in angles:
            t += 0.3
            frames.append((t, 1.0, a, "angle", p)); t += 1.05
        t += 0.3
        while t + 1.0 <= t0 + period_s - 0.05:  # (a frame ends before the next pulse)
            frames.append((t, 1.0, reference, "drying", p)); t += 1.05
        t = t0 + period_s
    return frames, pulse_times


# ----------------------------------------------------------------------------------- physics

def transmission(sin_a, delta, beta):
    sin_a = np.asarray(sin_a, dtype=np.complex128)
    kz = np.sqrt(sin_a * sin_a - 2 * delta + 2j * beta)
    t = 2 * sin_a / (sin_a + kz)
    return (np.abs(t) ** 2).real


def sphere_intensity(q, radius, rel_sigma, points=9):
    """<|F|^2> over a Gaussian number distribution of radii (nm), q in 1/nm; normalised to R0^6."""
    xs = np.linspace(-2.0, 2.0, points)
    w = np.exp(-0.5 * xs ** 2); w /= w.sum()
    out = np.zeros(q.shape, dtype=np.float64)
    for x, wi in zip(xs, w):
        r = radius * (1 + rel_sigma * x)
        if r <= 0:
            continue
        qr = np.maximum(q * r, 1e-6)
        amp = 3 * (np.sin(qr) - qr * np.cos(qr)) / qr ** 3
        out += wi * (r / radius) ** 6 * amp * amp
    return out


def paracrystal(q, d, rel_sigma):
    phi = np.exp(-0.5 * (q * d * rel_sigma) ** 2)
    return (1 - phi ** 2) / (1 + phi ** 2 - 2 * phi * np.cos(q * d))


SAMPLES = {
    # populations: (radius nm, rel. size spread, mass share); D1: spacing after one pulse (nm)
    "au10_si": {"substrate": "si", "pops": [(5.0, 0.10, 1.0)], "D1": 60.0, "cnf": 0.0},
    "au50_si": {"substrate": "si", "pops": [(25.0, 0.08, 1.0)], "D1": 260.0, "cnf": 0.0},
    "mix_cnf": {"substrate": "cnf", "pops": [(5.0, 0.10, 0.5), (25.0, 0.08, 0.5)], "D1": 120.0, "cnf": 1.0},
}


# ----------------------------------------------------------------------------------- files

def edf_bytes(counts: np.ndarray, date: datetime, count_time: float) -> bytes:
    lines = [
        "HeaderID       = EH:000001:000000:000000 ;",
        "Image          = 1 ;",
        "VersionNumber  = 0.10 ;",
        "ByteOrder      = LowByteFirst ;",
        "DataType       = SignedInteger ;",
        f"Dim_1          = {counts.shape[1]} ;",
        f"Dim_2          = {counts.shape[0]} ;",
        f"Size           = {counts.size * 4} ;",
        f"Date           = {date:%a %b} {date.day:2d} {date:%H:%M:%S %Y} ;",
        f"count_time     = {count_time:.9f} ;",
        "title          = # Pixel_size 172e-6 m x 172e-6 m ;",
        "run            = 0 ;",
    ]
    body = "{\n" + "\n".join(lines) + "\n"
    header = body + " " * (1024 - len(body) - 2) + "}\n"
    assert len(header) == 1024
    return header.encode("ascii") + counts.astype("<i4").tobytes()


def side_text(i0: float, i1: float, beamstop: float, alpha: float, current: float) -> str:
    """The layout of a 7.3.3 side file (motors, DIOs, AIs), only the values that matter filled in."""
    return "\n".join([
        f"{i0:.3f}", f"{i1:.3f}", f"{beamstop:.3f}",
        "Motors: 6",
        "Sample X Stage: -13.162500", "Sample Y Stage: 23.370579",
        f"Sample Alpha Stage: {alpha:.6f}", "Sample Phi Stage: 0.000000",
        "Detector Horizontal: -247.063886", "Detector Vertical: -3.902140",
        "DIOs: 2", "Beamline Shutter Open: 1.000000", "Beamline Pass Beam: 1.000000",
        "AIs: 5",
        f"Beam Current: {current:.6f}",
        f"Izero: {i0:.6f}", f"I1: {i1:.6f}",
        f"GiSAXS Beamstop Counter: {beamstop:.6f}",
        "Sample Temperature: 80.000000",
        "!0 ",
        "Alpha_scan_positions=", "Alpha_scan_I0_intensities=", "Alpha_scan_I1_intensities=",
        "Alpha_scan_diode_intensities=", "",
    ])


def detector_layout(detector: str, seed: int):
    """Module gaps (stored 0) of a Pilatus of that size, and a few bad pixels (stored -2) inside its modules."""
    from src.gimap.shared.detector_io.module_gaps import GAP, LAYOUTS, MODULE, layout_gaps

    rows, columns = (n * size + (n - 1) * gap for n, size, gap in zip(LAYOUTS[detector], MODULE, GAP))
    _name, gaps = layout_gaps((rows, columns))
    bad = np.zeros((rows, columns), bool)
    inside = np.flatnonzero(~gaps.ravel())
    bad.ravel()[np.random.default_rng(seed + 7).choice(inside, size=min(200, inside.size // 1000), replace=False)] = True
    return np.array(gaps), bad


# ----------------------------------------------------------------------------------- render

def render(args) -> None:
    from src.gimap.shared.geometry import DetectorGeometry
    from src.gimap.shared.geometry.q_mapping import grazing_q_map

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    sample = SAMPLES[args.sample]
    frames, pulse_times = schedule(args.pulses, args.fast_seconds, args.period, args.angles, args.reference)
    gaps, bad = detector_layout(args.detector, args.seed)
    SHAPE = gaps.shape
    rows, cols = np.indices(SHAPE)
    bx = args.beam_x if args.beam_x is not None else round(0.464 * SHAPE[1], 1)
    by = args.beam_y if args.beam_y is not None else float(SHAPE[0] - 29)
    beamstop = (np.abs(cols + 0.5 - bx) < 11) & (rows + 0.5 > by - 14)
    rng = np.random.default_rng(args.seed)
    sub = sample["substrate"]

    # q maps per angle (Å^-1 -> nm^-1 for the form factors)
    maps = {}
    for a in sorted(set([args.reference] + list(args.angles))):
        g = DetectorGeometry(pixel_size_x_m=PIXEL, pixel_size_y_m=PIXEL, distance_m=args.distance,
                             beam_center_x_px=bx, beam_center_y_px=by, wavelength_angstrom=WAVELENGTH,
                             incidence_deg=a)
        qm = grazing_q_map(SHAPE, g)
        qx, qy, qz, sin_f = (np.asarray(v, float) for v in (qm.qx, qm.qy, qm.qz, qm.sin_exit))
        q = np.sqrt(qx * qx + qy * qy + qz * qz) * 10.0
        qpar = np.abs(qy) * 10.0
        above = sin_f > 0
        sin_i = math.sin(math.radians(a))
        tf = np.where(above, transmission(np.where(above, sin_f, 1e-6), DELTA[sub], BETA[sub]), 0.0)
        ti = float(transmission(np.array([sin_i]), DELTA[sub], BETA[sub])[0])
        if sub == "cnf":  # a CNF film on Si: both edges show
            tf = 0.5 * tf + 0.5 * np.where(above, transmission(np.where(above, sin_f, 1e-6), DELTA["si"], BETA["si"]), 0)
            ti = 0.5 * ti + 0.5 * float(transmission(np.array([sin_i]), DELTA["si"], BETA["si"])[0])
        dwba = ti * tf
        pops = [sphere_intensity(q, r, s) for r, s, _m in sample["pops"]]
        cnf = 1.0 / (1 + (q * 15.0) ** 2) ** 2
        # specular spot and transmitted spray mist (around the direct beam)
        spec_row = by - args.distance * math.tan(math.radians(2 * a)) / PIXEL
        spec = np.exp(-0.5 * (((cols + 0.5 - bx) / 2.0) ** 2 + ((rows + 0.5 - spec_row) / 2.0) ** 2))
        r_direct = np.hypot(cols + 0.5 - bx, rows + 0.5 - by) * PIXEL
        q_direct = 2 * K * np.sin(0.5 * np.arctan(r_direct / args.distance)) * 10.0
        mist = 1.0 / (1 + (q_direct / 0.05) ** 4)
        maps[a] = dict(qpar=qpar.astype(np.float32), dwba=dwba.astype(np.float32), above=above,
                       pops=[p.astype(np.float32) for p in pops], cnf=(dwba * cnf).astype(np.float32),
                       spec=spec.astype(np.float32), mist=mist.astype(np.float32))
        print(f"q map for alpha_i={a}: q_par max {qpar.max():.2f} 1/nm", flush=True)

    start = datetime(2027, 1, 20, 14, 0, 0)  # (local time, as the beamline stamps it; replay restamps)
    record = []
    for index, (t, ct, a, phase, pulse) in enumerate(frames):
        m = maps[a]
        since = t - pulse_times[pulse - 1] if pulse else None
        deposited = pulse - (0 if (since is not None and since >= 0.2) else 0.5 if pulse else 0)
        deposited = max(0.0, float(deposited))
        wet = math.exp(-since / args.dry_tau) if since is not None else 0.0
        spraying = phase == "spray"
        d = sample["D1"] / math.sqrt(max(deposited, 1e-9)) * (1.0 + 0.10 * wet) if deposited else 0.0
        sig = 0.25 + 0.20 * wet
        rate = np.zeros(SHAPE, np.float32)
        if deposited:
            s_q = paracrystal(m["qpar"], d, sig).astype(np.float32)
            for (r, s, mass), pop in zip(sample["pops"], m["pops"]):
                n = deposited * mass / r ** 3  # number per area from mass
                rate += m["dwba"] * pop * (n * r ** 6 * args.particle_scale) * s_q
        rate += sample["cnf"] * 3000.0 * m["cnf"]
        rate += 8.0 * wet * m["dwba"] * (1.0 / (1 + (m["qpar"] / 0.08) ** 4))  # wet film / droplets on the surface
        if spraying:
            rate += 400.0 * m["mist"]
        rate += 2e4 * m["spec"] * (0.6 if spraying else 1.0)
        rate *= args.flux
        rate += 0.02  # detector + air background per second
        current = 500.0 + 0.4 * math.sin(t / 37.0) - (0.6 if (t % 30) > 28 else 0.0)
        beam = current / 500.0 * (0.85 if spraying else 1.0)  # mist absorbs a little
        counts = rng.poisson(rate * ct * beam).astype(np.int32)
        counts[beamstop] = rng.poisson(0.02 * ct, size=int(beamstop.sum()))
        counts[gaps] = 0
        counts[bad] = -2
        stem = f"{args.name}_{index:05d}"
        date = start + timedelta(seconds=t + ct)
        (out / f"{stem}.edf").write_bytes(edf_bytes(counts, date, ct))
        i0 = 63805.0 / 5.0 * ct * beam
        (out / f"{stem}.txt").write_text(side_text(i0, i0 * 0.652, 3.4e5 * ct * beam, a + args.alpha_offset, current),
                                         encoding="ascii")
        record.append(dict(index=index, file=f"{stem}.edf", start_s=round(t, 4), count_time_s=ct, alpha_deg=a,
                           phase=phase, pulse=pulse, deposited=deposited, wet=round(wet, 4),
                           spacing_nm=round(d, 2), spacing_rel_sigma=round(sig, 3), total_counts=int(counts[counts > 0].sum())))
        if index % 20 == 0:
            print(f"frame {index}/{len(frames)} t={t:.1f}s alpha={a} {phase} counts={record[-1]['total_counts']}", flush=True)
    manifest = dict(
        note="Synthetic ALS 7.3.3 spray run for software tests; not a model of any measured sample.",
        sample=args.sample, populations=sample["pops"], substrate=sub, energy_kev=ENERGY_KEV,
        wavelength_angstrom=WAVELENGTH, distance_m=args.distance, pixel_m=PIXEL,
        beam_center_px=[bx, by], beam_center_convention="GIMaP canonical (pixel corners on integers, row 0 top)",
        detector=args.detector, shape=list(SHAPE),
        side_file_alpha_offset_deg=args.alpha_offset, pulse_times_s=pulse_times, start=start.isoformat(),
        frames=record)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"wrote {len(frames)} frames to {out}")


# ----------------------------------------------------------------------------------- replay

def replay(args) -> None:
    src, dst = Path(args.src), Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((src / "manifest.json").read_text(encoding="utf-8"))
    frames = manifest["frames"]
    pulses = list(manifest["pulse_times_s"])
    log_rows = []
    t_begin = time.perf_counter()
    log_path = dst / "spray_log.csv"
    replay_log = []
    events = sorted([(f["start_s"] + f["count_time_s"], "frame", f) for f in frames] +
                    [(p, "pulse", i + 1) for i, p in enumerate(pulses)], key=lambda e: e[0])
    for when, kind, item in events:
        target = t_begin + when / args.speed
        delay = target - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        if kind == "pulse":
            log_rows.append((item, datetime.now().isoformat(timespec="milliseconds"), round(when, 3)))
            with log_path.open("w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh); w.writerow(["pulse", "time", "run_time_s"]); w.writerows(log_rows)
            continue
        name = item["file"]
        for ext in (".edf", ".txt"):  # the image first, then its side file (7.3.3 order unknown)
            part = dst / f".{Path(name).stem}{ext}.part"
            final = dst / f"{Path(name).stem}{ext}"
            if args.link:
                os.link(src / final.name, part)
            else:  # a copy whose header Date is the moment it is written, as the detector's would be
                data = (src / final.name).read_bytes()
                if ext == ".edf":
                    now = datetime.now()
                    stamp = f"{now:%a %b} {now.day:2d} {now:%H:%M:%S %Y}".encode("ascii")
                    start = data.index(b"Date           = ") + len(b"Date           = ")
                    data = data[:start] + stamp + data[start + len(stamp):]
                part.write_bytes(data)
            os.replace(part, final)
            os.utime(final, None)  # written now, as the detector would (hard links share the staged file's times)
        replay_log.append(dict(index=item["index"], file=name, written_s=round(time.perf_counter() - t_begin, 4),
                               scheduled_s=round(when / args.speed, 4)))
    (dst.parent / f"{dst.name}_replay_log.json").write_text(json.dumps(replay_log), encoding="utf-8")
    print(f"replayed {len(replay_log)} frames in {time.perf_counter() - t_begin:.1f} s")


def main(argv=None) -> None:
    from src.gimap.shared.detector_io.module_gaps import LAYOUTS

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("render")
    r.add_argument("--out", required=True)
    r.add_argument("--detector", choices=sorted(LAYOUTS), default="Pilatus 2M")
    r.add_argument("--sample", choices=sorted(SAMPLES), default="mix_cnf")
    r.add_argument("--name", default="spray_RUN1")
    r.add_argument("--pulses", type=int, default=3)
    r.add_argument("--fast-seconds", type=float, default=5.0)
    r.add_argument("--period", type=float, default=15.2, help="spray pulse + drying interval (s)")
    r.add_argument("--angles", type=float, nargs="*", default=[0.1, 0.25, 0.6])
    r.add_argument("--reference", type=float, default=0.4)
    r.add_argument("--alpha-offset", type=float, default=0.0, help="motor reading minus true incidence (deg)")
    r.add_argument("--distance", type=float, default=3.5)
    r.add_argument("--beam-x", type=float, help="direct beam column (canonical pixels; default near the middle)")
    r.add_argument("--beam-y", type=float, help="direct beam row (default 29 rows above the bottom)")
    r.add_argument("--flux", type=float, default=1.0)
    r.add_argument("--particle-scale", type=float, default=2.0)
    r.add_argument("--dry-tau", type=float, default=3.0)
    r.add_argument("--seed", type=int, default=1)
    r.set_defaults(func=render)
    y = sub.add_parser("replay")
    y.add_argument("--src", required=True)
    y.add_argument("--dst", required=True)
    y.add_argument("--speed", type=float, default=1.0)
    y.add_argument("--link", action="store_true", help="hard links (no extra disk; header Date as rendered)")
    y.set_defaults(func=replay)
    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
