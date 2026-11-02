"""Client side of the kit: what a GUI process uses. Standard library only (numpy arrays are accepted).

    client = Fit1DClient()                       # nothing starts yet
    res = client.fit(q, intensity, sigma, q_unit='A^-1', composition='sphere,vc', qz=0.5, policy='fast')
    res = client.preview(q, intensity, sigma, composition='sphere,vc')        # network only, a few 10 ms
    out = client.fit_many(curves, composition='sphere,vc', on_progress=callback)
    out = client.series(files, recipe, mode='review', out=folder, on_progress=callback)
    client.cancel()      # from another thread: stops the running call (the worker is killed and restarted lazily)
    client.close()

One worker process (python -m gisaxs1d_kit.worker) serves the calls one after another; it keeps the fitting code,
the network and the tables loaded, so only the first call pays the start-up (about 1-2 s). The calls block: run
them in a worker thread (QThread, concurrent.futures) and never in the GUI thread. A client is thread-safe (calls
are serialised); use several clients for parallel fits.

Errors: Fit1DError (the fit raised: bad input, unknown composition, ...; .kind and .remote_traceback tell more),
Fit1DCancelled (cancel() or close() during a call), Fit1DTimeout (subclass of Fit1DError; the worker is stopped).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from pathlib import Path

from gisaxs1d_kit._paths import kit_home


class Fit1DError(RuntimeError):
    def __init__(self, message, kind='Error', remote_traceback=''):
        super().__init__(message)
        self.kind, self.remote_traceback = kind, remote_traceback


class Fit1DTimeout(Fit1DError):
    pass


class Fit1DCancelled(RuntimeError):
    pass


def _plain(value):
    """numpy arrays / scalars -> lists / numbers (without importing numpy here)."""
    if hasattr(value, 'tolist'):
        return value.tolist()
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    return value


class Fit1DClient:
    def __init__(self, python=None, home=None, env=None, stderr_path=None):
        """python: interpreter of the worker (default: the one running this process; it needs numpy and scipy).
        home: the kit folder (default: the folder that contains this package). env: extra environment variables
        (e.g. GISAXS1D_TABLES). stderr_path: file for the worker's messages (default: discarded)."""
        self.python = str(python or self._default_python())
        self.home = Path(home).resolve() if home else kit_home()
        self.env = dict(env or {})
        self.stderr_path = stderr_path
        self._proc = None
        self._lock = threading.RLock()          # one call at a time
        self._state = threading.Lock()          # protects _proc / _cancelled
        self._cancelled = False
        self._next = 0
        self.info_cache = None

    @staticmethod
    def _default_python():
        exe = Path(sys.executable)
        if exe.name.lower() == 'pythonw.exe' and (exe.with_name('python.exe')).exists():
            return exe.with_name('python.exe')              # a console-less GUI interpreter: use its console twin
        return exe

    # ------------------------------------------------------------------ process
    def _start(self):
        env = dict(os.environ)
        env.update(self.env)
        paths = [str(self.home)]
        if env.get('PYTHONPATH'):
            paths.append(env['PYTHONPATH'])
        env['PYTHONPATH'] = os.pathsep.join(paths)
        env.setdefault('GISAXS1D_KIT_HOME', str(self.home))
        env['PYTHONUTF8'] = '1'
        for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
            env.setdefault(name, '1')
        stderr = open(self.stderr_path, 'ab') if self.stderr_path else subprocess.DEVNULL
        flags = getattr(subprocess, 'CREATE_NO_WINDOW', 0)   # Windows: no console window flashing up
        try:
            proc = subprocess.Popen([self.python, '-u', '-m', 'gisaxs1d_kit.worker'], stdin=subprocess.PIPE,
                                    stdout=subprocess.PIPE, stderr=stderr, cwd=str(self.home), env=env, creationflags=flags)
        finally:
            if self.stderr_path:
                stderr.close()
        line = proc.stdout.readline()
        if not line:
            raise Fit1DError(f'the worker did not start ({self.python} -m gisaxs1d_kit.worker in {self.home}); '
                             'check that this Python has numpy and scipy', 'StartupError')
        hello = json.loads(line)
        if hello.get('event') != 'ready':
            raise Fit1DError(f'unexpected first message of the worker: {hello}', 'StartupError')
        return proc

    def _process(self):
        with self._state:
            if self._proc is None or self._proc.poll() is not None:
                self._proc = self._start()
            return self._proc

    @property
    def running(self):
        with self._state:
            return self._proc is not None and self._proc.poll() is None

    @staticmethod
    def _close_pipes(proc):
        for pipe in (proc.stdin, proc.stdout):
            try:
                if pipe is not None:
                    pipe.close()
            except Exception:                                # noqa: BLE001 - broken pipe of a dead process
                pass

    def _stop(self, proc):
        if proc is None:
            return
        if proc.poll() is None:
            try:
                proc.kill()
                proc.wait(timeout=5)
            except Exception:                                # noqa: BLE001 - already gone
                pass
        self._close_pipes(proc)

    # ------------------------------------------------------------------ calls
    def call(self, op, args=None, on_progress=None, timeout=None):
        """One request. on_progress(completed, total, message, details) is called from this thread."""
        with self._lock:
            with self._state:
                self._cancelled = False
            proc = self._process()
            self._next += 1
            ident = self._next
            timer = None
            timed_out = threading.Event()
            if timeout:
                def expire():
                    timed_out.set()
                    self._stop(proc)
                timer = threading.Timer(float(timeout), expire)
                timer.daemon = True
                timer.start()
            try:
                # NaN / inf in the curve (masked points, gaps) travel as the JSON extensions NaN / Infinity:
                # both ends are Python, and the fit drops non-finite points itself
                payload = (json.dumps(dict(id=ident, op=op, args=_plain(args or {})))+'\n').encode('utf-8')
                try:
                    proc.stdin.write(payload)
                    proc.stdin.flush()
                except (BrokenPipeError, OSError, ValueError) as exc:
                    if self._was_cancelled():
                        raise Fit1DCancelled('cancelled') from None
                    raise Fit1DError(f'the worker is gone: {exc}', 'WorkerDied') from exc
                while True:
                    line = proc.stdout.readline()
                    if not line:
                        if timed_out.is_set():
                            raise Fit1DTimeout(f'no result within {timeout} s; the worker was stopped', 'Timeout')
                        if self._was_cancelled():
                            raise Fit1DCancelled('cancelled')
                        raise Fit1DError('the worker stopped unexpectedly (see stderr_path for its messages)', 'WorkerDied')
                    message = json.loads(line)
                    if message.get('id') != ident:
                        continue
                    if message.get('event') == 'progress':
                        if on_progress is not None:
                            on_progress(message.get('completed'), message.get('total'), message.get('message', ''),
                                        message.get('details') or {})
                        continue
                    if message.get('ok'):
                        return message.get('value')
                    error = message.get('error') or {}
                    raise Fit1DError(error.get('message', 'the worker reported an error'), error.get('type', 'Error'),
                                     error.get('traceback', ''))
            finally:
                if timer is not None:
                    timer.cancel()

    def _was_cancelled(self):
        with self._state:
            return self._cancelled

    def cancel(self):
        """Stop the running call (if any) from another thread. Returns True when a worker was stopped."""
        with self._state:
            proc, self._cancelled = self._proc, True
            self._proc = None
        if proc is None:
            return False
        alive = proc.poll() is None
        self._stop(proc)
        return alive

    def close(self):
        with self._state:
            proc, self._proc = self._proc, None
            self._cancelled = True
        if proc is None:
            return
        if proc.poll() is None:
            try:
                proc.stdin.write(b'{"op": "quit"}\n')
                proc.stdin.flush()
                proc.wait(timeout=2)
            except Exception:                                # noqa: BLE001
                pass
        self._stop(proc)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:                                    # noqa: BLE001
            pass

    # ------------------------------------------------------------------ operations
    def ping(self, timeout=60):
        return self.call('ping', timeout=timeout)

    def info(self, timeout=120):
        """Versions, networks, compositions, policies (also loads the fitting code: a good warm-up call)."""
        self.info_cache = self.call('info', timeout=timeout)
        return self.info_cache

    def load(self, path, *, q_unit='nm^-1', side='both', sigma_mode='side', timeout=120):
        """Read a curve file (yoneda_cut .npz, or text with q, I[, sigma]); returns q in nm^-1, intensity, sigma,
        qz (when the file knows it) and meta."""
        return self.call('load', dict(path=os.fspath(path), q_unit=q_unit, side=side, sigma_mode=sigma_mode), timeout=timeout)

    @staticmethod
    def _curve_args(q, intensity, sigma, settings):
        args = dict(settings)
        args.update(q=q, intensity=intensity, sigma=sigma)
        return args

    def fit(self, q, intensity, sigma=None, *, timeout=600, **settings):
        """Fit one curve. settings: q_unit, fold, q_range, composition, mode, qz, policy, model, sigma_floor,
        constraints, lowq, lowq_nu, structure_factor, dwba, include_raw (see gisaxs1d_kit.core)."""
        return self.call('fit', self._curve_args(q, intensity, sigma, settings), timeout=timeout)

    def preview(self, q, intensity, sigma=None, *, timeout=120, **settings):
        """Network proposals only: an estimate within a few 10 ms (no uncertainties)."""
        return self.call('preview', self._curve_args(q, intensity, sigma, settings), timeout=timeout)

    def fit_many(self, curves, *, on_progress=None, timeout=None, **settings):
        """Independent fits. curves: [{'q': ..., 'intensity': ..., 'sigma': ..., 'label': ...}, ...]."""
        return self.call('fit_many', dict(settings, curves=curves), on_progress=on_progress, timeout=timeout)

    def series(self, files, recipe, *, on_progress=None, timeout=None, **settings):
        """Tracked series fit (research.series1d): settings mode, out, workers, resume, frames, input_format."""
        return self.call('series', dict(settings, files=files, recipe=recipe), on_progress=on_progress, timeout=timeout)
