"""One results database per series folder (SQLite): every frame's reduced curves and fits, and the files a
watch skipped, so that a session can be opened again and series can be compared without their frames.

Analyze records the curve of every frame of a Series map (Build Map, a Batch Export, a watched folder);
Fitting the fit of every frame of an In-situ series; Compare reads the curves back as a series. The file is
``gimap_series.sqlite`` (``STORE_NAME``) in the folder of the series' results (``gimap_analysis/`` next to
the data, or the folder of the fitted curves); one file may hold several series, told apart by name.

::

    store     (key, value)                         schema, software, created
    series    (name, info, updated)                what a series was made with (JSON: curve, settings …)
    curves    (series, curve, file, frame_index)   x, I, σ as little-endian float64; frame, label, time
    fits      (series, kind, file)                 state, χ²ᵣ, values and errors (JSON), flags, start
    skipped   (id, series, time, file, …)          a file a watch could not take, and why

A frame is identified by its file and its frame in that file (``frame_index``, from 0); writing it again
replaces it. The database is written in WAL mode, so it can be read (Compare, another GIMaP) while a watch
writes it. Nothing here imports a feature: the arrays and records are plain values.
"""

from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np

from .version import SOFTWARE

STORE_NAME = "gimap_series.sqlite"
SCHEMA = 1
NOT_A_STORE = "{name} is not a GIMaP results database ({error})."
NEWER_STORE = "{name} was written by a newer GIMaP (database schema {schema}); this version reads schema {known}."
NO_SERIES = "{name} holds no series."
NO_CURVES = "{name} holds no curve “{curve}” of the series “{series}”."

_TABLES = (
    "CREATE TABLE IF NOT EXISTS store (key TEXT PRIMARY KEY, value TEXT)",
    "CREATE TABLE IF NOT EXISTS series (name TEXT PRIMARY KEY, info TEXT, updated TEXT)",
    "CREATE TABLE IF NOT EXISTS curves (series TEXT NOT NULL, curve TEXT NOT NULL, file TEXT NOT NULL,"
    " frame_index INTEGER NOT NULL, frame INTEGER, label TEXT, x_label TEXT, x BLOB, intensity BLOB, sigma BLOB,"
    " points INTEGER, time_s REAL, timestamp TEXT, summed INTEGER, written TEXT,"
    " PRIMARY KEY (series, curve, file, frame_index))",
    "CREATE TABLE IF NOT EXISTS fits (series TEXT NOT NULL, kind TEXT NOT NULL, file TEXT NOT NULL, frame INTEGER,"
    " state TEXT, chi2 REAL, fit_values TEXT, errors TEXT, flags TEXT, start_from TEXT, time_s REAL, record TEXT,"
    " written TEXT, PRIMARY KEY (series, kind, file))",
    "CREATE TABLE IF NOT EXISTS skipped (id INTEGER PRIMARY KEY AUTOINCREMENT, series TEXT, time TEXT, file TEXT,"
    " frame_index INTEGER, what TEXT, reason TEXT)",
)


def store_path(folder) -> Path:
    """The results database of the series whose results are in ``folder``."""
    return Path(folder) / STORE_NAME


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _blob(values) -> Optional[bytes]:
    if values is None:
        return None
    return np.ascontiguousarray(np.asarray(values, dtype="<f8")).tobytes()


def _array(blob) -> Optional[np.ndarray]:
    return None if blob is None else np.frombuffer(blob, dtype="<f8").copy()


def _number(value) -> Optional[float]:
    """A finite float, or ``None`` (SQLite and JSON keep no NaN)."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _plain(value):
    """``value`` as JSON can hold it: NaN and ±inf as ``None``, numpy numbers as Python numbers."""
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return _number(value)
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    return value


def _json(value) -> Optional[str]:
    return None if value is None else json.dumps(_plain(value), ensure_ascii=False, default=str)  # (paths: as text)


def _read_json(text, default=None):
    if not text:
        return default
    try:
        return json.loads(text)
    except ValueError:
        return default


@dataclass(frozen=True)
class StoredCurve:
    """One frame's curve as recorded."""

    series: str
    curve: str
    file: str
    frame_index: int
    frame: int
    """The frame as the series counts it, from 1."""
    x: np.ndarray
    intensity: np.ndarray
    sigma: Optional[np.ndarray] = None
    x_label: str = ""
    label: str = ""
    time_s: Optional[float] = None
    timestamp: str = ""
    summed: int = 1
    written: str = ""


@dataclass(frozen=True)
class StoredFit:
    """One frame's fit as recorded (``values`` and ``errors``: parameter name → number)."""

    series: str
    kind: str
    file: str
    frame: int
    state: str
    chi2: Optional[float] = None
    values: dict = field(default_factory=dict)
    errors: dict = field(default_factory=dict)
    flags: tuple = ()
    start_from: str = ""
    time_s: Optional[float] = None
    record: Optional[dict] = None
    written: str = ""


@dataclass(frozen=True)
class SkippedFile:
    series: str
    time: str
    file: str
    frame_index: int
    what: str
    reason: str


class SeriesStore:
    """The results database at ``path`` (made when written to; ``read_only``: an existing one, never changed).

    Use it from one thread (the window's); every ``put_*`` is one transaction."""

    def __init__(self, path, *, read_only: bool = False):
        self.path = Path(path)
        self.read_only = bool(read_only)
        self._db: Optional[sqlite3.Connection] = None

    # -- the connection -------------------------------------------------------------------------------

    @property
    def db(self) -> sqlite3.Connection:
        if self._db is None:
            self._db = self._open()
        return self._db

    def _open(self) -> sqlite3.Connection:
        name = self.path.name
        db = None
        try:
            if self.read_only:
                if not self.path.is_file():
                    raise sqlite3.OperationalError("no such file")
                db = sqlite3.connect(f"{self.path.resolve().as_uri()}?mode=ro", uri=True, timeout=10.0)
            else:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                db = sqlite3.connect(str(self.path), timeout=10.0)
                db.execute("PRAGMA journal_mode=WAL")
                db.execute("PRAGMA synchronous=NORMAL")
                with db:
                    for statement in _TABLES:
                        db.execute(statement)
                    db.execute("INSERT OR IGNORE INTO store VALUES ('schema', ?)", (str(SCHEMA),))
                    db.execute("INSERT OR IGNORE INTO store VALUES ('software', ?)", (SOFTWARE,))
                    db.execute("INSERT OR IGNORE INTO store VALUES ('created', ?)", (_now(),))
            row = db.execute("SELECT value FROM store WHERE key = 'schema'").fetchone()
        except sqlite3.DatabaseError as exc:
            if db is not None:
                db.close()
            raise ValueError(NOT_A_STORE.format(name=name, error=exc)) from exc
        schema = int(row[0]) if row and str(row[0]).isdigit() else 0
        if schema > SCHEMA:
            db.close()
            raise ValueError(NEWER_STORE.format(name=name, schema=schema, known=SCHEMA))
        return db

    def close(self) -> None:
        if self._db is not None:
            self._db.close()
            self._db = None

    def __enter__(self) -> "SeriesStore":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    # -- writing --------------------------------------------------------------------------------------

    def set_info(self, series: str, info: Mapping[str, Any]) -> None:
        """What ``series`` was made with (merged into what is recorded already)."""
        db = self.db
        known = _read_json((db.execute("SELECT info FROM series WHERE name = ?", (series,)).fetchone() or [None])[0], {})
        known.update(json.loads(_json(dict(info))))
        with db:
            db.execute("INSERT OR REPLACE INTO series VALUES (?, ?, ?)", (series, _json(known), _now()))

    def put_curves(self, series: str, curves: Iterable[Mapping[str, Any]]) -> int:
        """Record frames' curves in one transaction; each mapping holds ``curve``, ``file``, ``x``, ``intensity``
        and optionally ``frame_index``, ``frame``, ``sigma``, ``x_label``, ``label``, ``time_s``, ``timestamp``,
        ``summed``. Returns how many were written."""
        written, rows = _now(), []
        for item in curves:
            x = np.asarray(item["x"], dtype=float)
            rows.append((
                series, str(item["curve"]), str(item["file"]), int(item.get("frame_index", 0) or 0),
                int(item.get("frame", 0) or 0), str(item.get("label") or ""), str(item.get("x_label") or ""),
                _blob(x), _blob(item["intensity"]), _blob(item.get("sigma")), int(x.size), _number(item.get("time_s")),
                str(item.get("timestamp") or ""), int(item.get("summed", 1) or 1), written))
        if not rows:
            return 0
        db = self.db
        with db:
            db.executemany("INSERT OR REPLACE INTO curves VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
            db.execute("INSERT OR IGNORE INTO series VALUES (?, '{}', ?)", (series, written))
            db.execute("UPDATE series SET updated = ? WHERE name = ?", (written, series))
        return len(rows)

    def put_curve(self, series: str, curve: str, file, x, intensity, **more) -> None:
        self.put_curves(series, [dict(curve=curve, file=str(file), x=x, intensity=intensity, **more)])

    def put_fits(self, series: str, kind: str, fits: Iterable[Mapping[str, Any]]) -> int:
        """Record frames' fits in one transaction; each mapping holds ``file`` and ``state`` and optionally
        ``frame``, ``chi2``, ``values``, ``errors``, ``flags``, ``start_from``, ``time_s``, ``record``."""
        written, rows = _now(), []
        for item in fits:
            rows.append((
                series, str(kind), str(item["file"]), int(item.get("frame", 0) or 0), str(item.get("state") or ""),
                _number(item.get("chi2")), _json(dict(item.get("values") or {})), _json(dict(item.get("errors") or {})),
                _json(list(item.get("flags") or ())), str(item.get("start_from") or ""), _number(item.get("time_s")),
                _json(item.get("record")), written))
        if not rows:
            return 0
        db = self.db
        with db:
            db.executemany("INSERT OR REPLACE INTO fits VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
            db.execute("INSERT OR IGNORE INTO series VALUES (?, '{}', ?)", (series, written))
            db.execute("UPDATE series SET updated = ? WHERE name = ?", (written, series))
        return len(rows)

    def log_skipped(self, series: str, file, reason: str, *, frame_index: int = 0, what: str = "skipped",
                    time: Optional[str] = None) -> None:
        """A file a watch could not take (still written, held by its writer, unreadable …), and why."""
        db = self.db
        with db:
            db.execute("INSERT INTO skipped (series, time, file, frame_index, what, reason) VALUES (?, ?, ?, ?, ?, ?)",
                       (series, time or _now(), str(file), int(frame_index), str(what), str(reason)))

    # -- reading --------------------------------------------------------------------------------------

    def series_names(self) -> list[str]:
        """Every series recorded, with curves or fits, in the order they were last written."""
        rows = self.db.execute(
            "SELECT name FROM series WHERE name IN (SELECT series FROM curves UNION SELECT series FROM fits)"
            " ORDER BY updated").fetchall()
        return [str(name) for (name,) in rows]

    def info(self, series: str) -> dict:
        row = self.db.execute("SELECT info FROM series WHERE name = ?", (series,)).fetchone()
        return _read_json(row[0], {}) if row else {}

    def curve_keys(self, series: str) -> list[tuple[str, int, str]]:
        """``(curve, frames, x_label)`` of every curve recorded for ``series``."""
        rows = self.db.execute("SELECT curve, COUNT(*), MAX(x_label) FROM curves WHERE series = ? GROUP BY curve"
                               " ORDER BY MIN(rowid)", (series,)).fetchall()
        return [(str(curve), int(count), str(label or "")) for curve, count, label in rows]

    def curves(self, series: str, curve: str) -> list[StoredCurve]:
        """The curve ``curve`` of every recorded frame of ``series``, in frame order (then file, frame index)."""
        rows = self.db.execute(
            "SELECT file, frame_index, frame, label, x_label, x, intensity, sigma, time_s, timestamp, summed, written"
            " FROM curves WHERE series = ? AND curve = ? ORDER BY frame, file, frame_index", (series, curve)).fetchall()
        return [StoredCurve(series, curve, str(file), int(index), int(frame or 0), _array(x), _array(intensity),
                            _array(sigma), str(x_label or ""), str(label or ""), _number(time_s), str(stamp or ""),
                            int(summed or 1), str(written or ""))
                for file, index, frame, label, x_label, x, intensity, sigma, time_s, stamp, summed, written in rows]

    def fit_kinds(self, series: str) -> list[str]:
        rows = self.db.execute("SELECT DISTINCT kind FROM fits WHERE series = ? ORDER BY kind", (series,)).fetchall()
        return [str(kind) for (kind,) in rows]

    def fits(self, series: str, kind: Optional[str] = None) -> list[StoredFit]:
        """The fits of ``series`` (of one ``kind``, or all), in frame order."""
        query = ("SELECT kind, file, frame, state, chi2, fit_values, errors, flags, start_from, time_s, record, written"
                 " FROM fits WHERE series = ?" + (" AND kind = ?" if kind is not None else "") + " ORDER BY frame, file")
        rows = self.db.execute(query, (series,) if kind is None else (series, kind)).fetchall()
        return [StoredFit(series, str(found), str(file), int(frame or 0), str(state or ""), _number(chi2),
                          _read_json(values, {}), _read_json(errors, {}), tuple(_read_json(flags, []) or ()),
                          str(start or ""), _number(time_s), _read_json(record), str(written or ""))
                for found, file, frame, state, chi2, values, errors, flags, start, time_s, record, written in rows]

    def skipped(self, series: Optional[str] = None) -> list[SkippedFile]:
        query = "SELECT series, time, file, frame_index, what, reason FROM skipped" + (
            " WHERE series = ?" if series is not None else "") + " ORDER BY id"
        rows = self.db.execute(query, () if series is None else (series,)).fetchall()
        return [SkippedFile(str(name or ""), str(time or ""), str(file or ""), int(index or 0), str(what or ""),
                            str(reason or "")) for name, time, file, index, what, reason in rows]

    def summary(self) -> dict:
        """What the database holds, for a record or a question: per series its curves and fits."""
        return {"path": str(self.path), "series": [
            {"name": name, "curves": [{"curve": curve, "frames": count, "x_label": label}
                                      for curve, count, label in self.curve_keys(name)],
             "fits": {kind: len(self.fits(name, kind)) for kind in self.fit_kinds(name)}}
            for name in self.series_names()]}


def read_curves(path, series: Optional[str] = None, curve: Optional[str] = None) -> tuple[str, str, list[StoredCurve]]:
    """``(series, curve, curves)`` of a results database: the series named (else the one recorded last) and the
    curve named (else the first one recorded for it). ``ValueError`` when there is none."""
    with SeriesStore(path, read_only=True) as store:
        names = store.series_names()
        if not names:
            raise ValueError(NO_SERIES.format(name=Path(path).name))
        series = series if series in names else names[-1]
        keys = [key for key, _count, _label in store.curve_keys(series)]
        if curve not in keys:
            if curve is not None or not keys:
                raise ValueError(NO_CURVES.format(name=Path(path).name, curve=curve or "", series=series))
            curve = keys[0]
        return series, curve, store.curves(series, curve)


def stacked(curves: Sequence[StoredCurve]) -> tuple[np.ndarray, np.ndarray]:
    """``(x, image)``: recorded curves on the x of the first one (the others interpolated inside their own
    range, NaN outside it and never across a gap wider than ten of their steps)."""
    if not curves:
        raise ValueError("No curves.")
    first = curves[0]
    order = np.argsort(first.x)
    grid = np.asarray(first.x, dtype=float)[order]
    image = np.full((len(curves), grid.size), np.nan)
    for row, item in enumerate(curves):
        x, y = np.asarray(item.x, dtype=float), np.asarray(item.intensity, dtype=float)
        keep = np.isfinite(x) & np.isfinite(y)
        x, y = x[keep], y[keep]
        if x.size < 2:
            continue
        sort = np.argsort(x)
        x, y = x[sort], y[sort]
        if x.shape == grid.shape and np.allclose(x, grid):
            image[row] = y
            continue
        inside = (grid >= x[0]) & (grid <= x[-1])
        values = np.interp(grid[inside], x, y)
        steps = np.diff(x)
        usual = float(np.median(steps)) if steps.size else 0.0
        after = np.clip(np.searchsorted(x, grid[inside]), 1, x.size - 1)
        values[steps[after - 1] > 10 * usual] = np.nan
        image[row, inside] = values
    return grid, image


__all__ = ["NEWER_STORE", "NOT_A_STORE", "NO_CURVES", "NO_SERIES", "SCHEMA", "STORE_NAME", "SeriesStore", "SkippedFile",
           "StoredCurve", "StoredFit", "read_curves", "stacked", "store_path"]
