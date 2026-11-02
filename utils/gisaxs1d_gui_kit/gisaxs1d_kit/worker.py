"""Worker process of the kit: reads one JSON request per line on stdin, writes JSON messages on stdout.

    python -u -m gisaxs1d_kit.worker        (started by gisaxs1d_kit.client.Fit1DClient)

request   {"id": 7, "op": "fit" | "preview" | "fit_many" | "series" | "info" | "ping" | "quit", "args": {...}}
messages  {"id": 7, "event": "progress", "completed": 3, "total": 10, "message": "...", "details": {...}}
          {"id": 7, "event": "result", "ok": true, "value": {...}}
          {"id": 7, "event": "result", "ok": false, "error": {"type": ..., "message": ..., "traceback": ...}}
The first message after start is {"event": "ready", "kit_version": ..., "pid": ...}.

The fitting code, numpy and the tables are loaded once and stay loaded: after the first call a fit costs only its
own time. Everything the fitting code prints goes to stderr (stdout carries the protocol only). One BLAS thread
(a 6-column problem gains nothing from more, and several workers must not fight for the cores).
"""
import os

for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_name, '1')

import json  # noqa: E402
import sys  # noqa: E402
import traceback  # noqa: E402


def _jsonable(obj):
    import numpy as np
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        return value if value == value and abs(value) != float('inf') else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, os.PathLike):
        return os.fspath(obj)
    return obj


def main():
    channel = sys.stdout.buffer                       # the protocol channel
    sys.stdout = sys.stderr                           # anything printed by library code must not reach it

    def send(message):
        channel.write((json.dumps(_jsonable(message), allow_nan=False)+'\n').encode('utf-8'))
        channel.flush()

    from gisaxs1d_kit import _paths
    send(dict(event='ready', kit_version=_paths.KIT_VERSION, pid=os.getpid()))
    core = None
    for raw in sys.stdin.buffer:
        line = raw.decode('utf-8').strip()
        if not line:
            continue
        ident = None
        try:
            request = json.loads(line)
            ident, op, args = request.get('id'), request.get('op'), request.get('args') or {}
            if op == 'quit':
                send(dict(id=ident, event='result', ok=True, value=None))
                break
            if op == 'ping':
                send(dict(id=ident, event='result', ok=True, value=dict(pong=True, pid=os.getpid())))
                continue
            if core is None:
                from gisaxs1d_kit import core as core_module
                core = core_module
            if op not in core.OPERATIONS:
                raise ValueError(f'unknown operation {op!r}; expected one of {sorted(core.OPERATIONS)}')

            def report(completed, total, message='', details=None, _id=ident):
                send(dict(id=_id, event='progress', completed=completed, total=total, message=message, details=details or {}))
            value = core.OPERATIONS[op](args, report, lambda: False)
            send(dict(id=ident, event='result', ok=True, value=value))
        except Exception as exc:                        # noqa: BLE001 - reported to the client, the worker stays alive
            send(dict(id=ident, event='result', ok=False,
                      error=dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())))


if __name__ == '__main__':
    main()
