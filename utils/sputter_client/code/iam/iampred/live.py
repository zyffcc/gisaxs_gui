'Live view of a running analysis for a GUI (2026-11-15): what each step has found so far, while it works.'

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

ENV = "IAMPRED_LIVE"
PREFIX = "@live "
POLL_S = 0.5


def folder():
    'The folder of the live events of this step, or None (not run for a GUI).'
    value = os.environ.get(ENV, "").strip()
    return Path(value) if value else None


def emit(kind, arrays=None, **fields):
    'Write one event (see the module docstring); returns its name, or None when there is no live folder or it'
    where = folder()
    if where is None:
        return None
    try:
        where.mkdir(parents=True, exist_ok=True)
        name = f"{time.time_ns():020d}_{os.getpid()}"
        if arrays:
            tmp = where / f"{name}.tmp.npz"
            np.savez_compressed(tmp, **{k: np.asarray(v) for k, v in arrays.items()})
            os.replace(tmp, where / f"{name}.npz")
        record = dict(kind=kind, **fields)
        if arrays:
            record["arrays_file"] = f"{name}.npz"
        tmp = where / f"{name}.tmp"
        tmp.write_text(json.dumps(record, default=_plain), encoding="utf-8")
        os.replace(tmp, where / f"{name}.json")
        return name
    except Exception:
        return None


PROGRESS_S = 1.0
_LAST_PROGRESS = {}


def progress(phase, done, total, force=False, **fields):
    'A progress event of this process (see the module docstring): written when ``force``, at the first call of a'
    if folder() is None:
        return None
    now = time.monotonic()
    if not (force or done >= total or now - _LAST_PROGRESS.get(phase, -1e9) >= PROGRESS_S):
        return None
    _LAST_PROGRESS[phase] = now
    return emit("progress", phase=phase, done=int(done), total=int(total), worker=os.getpid(), **fields)


def _plain(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def say(kind, step, **fields):
    'An event of iampred.analyze itself, printed at once (it runs the steps and has no live folder of its own).'
    try:
        print(PREFIX + json.dumps(dict(kind=kind, step=step, **fields), separators=(",", ":"), default=_plain), flush=True)
    except (OSError, ValueError):
        pass


def rounded(value, digits=4):
    'A float for an event (None when it is not finite).'
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return round(value, digits) if np.isfinite(value) else None


class Forwarder:
    'Prints the events of a live folder as ``@live {json}`` lines while a step runs (a thread).'

    def __init__(self, where, step, out=None):
        self.where, self.step, self.out = Path(where), step, out or sys.stdout
        self.seen = set()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=10)
        self.sweep()
        return False

    def _loop(self):
        while not self._stop.wait(POLL_S):
            self.sweep()

    def sweep(self):
        with self._lock:
            self._sweep()

    def _sweep(self):
        try:
            names = sorted(p.name for p in self.where.glob("*.json") if p.name not in self.seen)
        except OSError:
            return
        for name in names:
            path = self.where / name
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            self.seen.add(name)
            record["step"] = self.step
            arrays = record.pop("arrays_file", None)
            if arrays:
                record["arrays"] = str((self.where / arrays).resolve())
            try:
                self.out.write(PREFIX + json.dumps(record, separators=(",", ":")) + "\n")
                self.out.flush()
            except (OSError, ValueError):
                return


def run_step(command, log, cwd, where, step):
    'Run one step as its own process (output to ``log``) with its live folder ``where``, forwarding its events;'
    env = dict(os.environ)
    env[ENV] = str(where)
    Path(where).mkdir(parents=True, exist_ok=True)
    with open(log, "w", encoding="utf-8") as fh, Forwarder(where, step):
        return subprocess.run(command, stdout=fh, stderr=subprocess.STDOUT, cwd=str(cwd), env=env).returncode


def clear(where):
    'Remove the events of an earlier run (the folder holds nothing else).'
    where = Path(where)
    if not where.is_dir():
        return
    for path in sorted(where.rglob("*"), reverse=True):
        try:
            path.unlink() if path.is_file() else path.rmdir()
        except OSError:
            pass
    try:
        where.rmdir()
    except OSError:
        pass


def binned_log(total, keep, factor=4):
    '(log10 of the summed counts on blocks of factor x factor pixels - NaN where nothing was counted -, the share of'
    total = np.asarray(total, float)
    keep = np.asarray(keep, bool)
    h, w = total.shape
    H, W = -(-h // factor), -(-w // factor)
    sums, kept = np.zeros((H * factor, W * factor)), np.zeros((H * factor, W * factor))
    sums[:h, :w] = np.where(np.isfinite(total) & (total > 0), total, 0.0)
    kept[:h, :w] = keep
    sums = sums.reshape(H, factor, W, factor).sum(axis=(1, 3))
    kept = kept.reshape(H, factor, W, factor).mean(axis=(1, 3))
    with np.errstate(divide="ignore"):
        image = np.where(sums > 0, np.log10(sums), np.nan)
    return image.astype(np.float16), (1.0 - kept).astype(np.float16)


def native_axes(grid):
    '(q_y with both sides of the beam - the first side mirrored to negative q -, exit angle) of the native grid of an'
    g = np.load(grid)
    q = 0.5 * (g["q_edges"][1:] + g["q_edges"][:-1])
    af = 0.5 * (g["native_edges"][1:] + g["native_edges"][:-1])
    return np.r_[-q[::-1], q].astype(np.float32), af.astype(np.float32)


def native_rate(Y, P, exposure):
    'Count rate per pixel (af x q, both sides as in native_axes) of the frames of one chunk: Y, P (frames x side x af'
    Y = np.asarray(Y, float).sum(0)
    E = (np.asarray(P, float) * np.asarray(exposure, float)[:, None, None, None]).sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = np.where(E > 0, Y / E, np.nan)
    return np.concatenate([rate[0][:, ::-1], rate[1]], axis=1).astype(np.float32)


class ChunkWatcher:
    'While the frames of an import are read (polymer_dev_intake.prepare writes one chunk of 128 frames at a time,'

    def __init__(self, intake_dir, grid, total):
        self.dir, self.grid, self.total = Path(intake_dir), Path(grid), int(total)
        self.done, self.seen, self.waiting, self.t0 = 0, set(), {}, None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def __enter__(self):
        if folder() is not None:
            self._thread.start()
        return self

    def __exit__(self, *exc):
        if self._thread.is_alive():
            self._stop.set()
            self._thread.join(timeout=30)
        return False

    def _loop(self):
        while not self._stop.wait(1.0):
            self.sweep()
        self.sweep()

    def sweep(self):
        from datetime import datetime
        try:
            new = sorted(p for p in self.dir.glob("chunk_*.json") if p.name not in self.seen)
        except OSError:
            return
        for meta in new:
            try:
                doc = json.loads(meta.read_text(encoding="utf-8"))
                rows = doc["rows"]
                index = int(meta.stem.split("_")[-1])
            except (OSError, ValueError, KeyError):
                continue
            self.seen.add(meta.name)
            self.waiting[index] = (meta, rows, doc)
            if index == 0 and rows:
                self.t0 = datetime.fromisoformat(rows[0]["timestamp"]).timestamp()
        if self.t0 is None:
            return
        for index in sorted(self.waiting):
            meta, rows, doc = self.waiting.pop(index)
            self.done += len(rows)
            try:
                self._event(index, meta, rows, doc)
            except Exception:
                pass

    def _event(self, index, meta, rows, doc=None):
        from datetime import datetime
        times = [datetime.fromisoformat(r["timestamp"]).timestamp() - self.t0 for r in rows]
        trace = [[round(t, 3), rounded(r["selected_counts"] / max(r["valid_pixel_count"] * r["exposure_s"], 1e-12), 6)]
                 for t, r in zip(times, rows)]
        arrays = None
        if self.grid.exists():
            with np.load(meta.with_suffix(".npz")) as z:
                rate = native_rate(z["Y"], z["P"], [r["exposure_s"] for r in rows])
            q, af = native_axes(self.grid)
            arrays = dict(q=q, af=af, rate=rate)
        doc = doc or {}


        emit("frames", arrays, chunk=index, done=min(self.done, self.total), total=self.total,
             t_first=rounded(min(times), 3), t_last=rounded(max(times), 3), trace=trace, frames=len(rows),
             read_s=rounded(doc.get("read_seconds"), 3), wall_s=rounded(doc.get("wall_seconds"), 3),
             mb=rounded((doc.get("read_bytes") or 0) / 1e6, 2) if doc.get("read_bytes") is not None else None)


def native_rate_YE(Y, E):
    'Count rate (af x q, both sides as in native_axes) of counts Y and exposures E on the native grid (side x af x q,'
    Y, E = np.asarray(Y, float), np.asarray(E, float)
    if Y.ndim == 4:
        Y, E = Y.sum(0), E.sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = np.where(E > 0, Y / E, np.nan)
    return np.concatenate([rate[0][:, ::-1], rate[1]], axis=1).astype(np.float32)


def static_frame_images(w, mu):
    '(measured, model) count rate of one static frame on its cells (af x q; the window arrays carry a leading axis of'
    ok = np.asarray(w["mask"], bool)
    E = np.where(ok, w["E"], 0.0)
    Es = E.sum(0) if E.ndim == 3 else E
    with np.errstate(invalid="ignore", divide="ignore"):
        Y = np.where(ok, w["Y"], 0.0)
        meas = np.where(Es > 0, (Y.sum(0) if Y.ndim == 3 else Y) / np.maximum(Es, 1e-30), np.nan)
        M = np.asarray(mu, float) * E
        model = np.where(Es > 0, (M.sum(0) if M.ndim == 3 else M) / np.maximum(Es, 1e-30), np.nan)
    return meas.astype(np.float32), model.astype(np.float32)


def static_values(rec):
    'The quantities of the main screen of a static fit record (as iampred.static puts them on its main screen).'
    try:
        R, D, t_b, T = (float(rec[k]) for k in ("R", "d_hex", "t_b", "T"))
        return dict(R_nm=rounded(R), H_nm=rounded(t_b + T), D_nm=rounded(D), gap_nm=rounded(max(D - 2.0 * R, 0.0)),
                    coverage=rounded(rec.get("eta")), amount_nm=rounded(rec.get("h_m_model")), base_nm=rounded(t_b))
    except (KeyError, TypeError, ValueError):
        return {}


def window_images(ctx, mu):
    '(measured, model) count rate on the cells of a window (af x q, NaN where no data), as iampred.present draws'
    ok = ctx.w["mask"]
    E = np.where(ok, ctx.w["E"], 0.0)
    Es = E.sum(0)
    good = Es > 0
    with np.errstate(invalid="ignore", divide="ignore"):
        meas = np.where(good, np.where(ok, ctx.w["Y"], 0.0).sum(0) / np.maximum(Es, 1e-30), np.nan)
        model = np.where(good, (np.where(ok, mu, 0.0) * E).sum(0) / np.maximum(Es, 1e-30), np.nan)
    return meas.astype(np.float32), model.astype(np.float32)
