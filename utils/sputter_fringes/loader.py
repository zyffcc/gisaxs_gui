"""Read a GIMaP Sputter result folder (read only): the geometry, the time windows and the vertical cuts I(alpha_f) of
a low q_y band, on both sides of the beam.

The toolkit bins every window's photons on its native (alpha_f, |q_y|) grid: ``binned/chunk_NNNNN.npz`` holds ``Y``
(counts) and ``E`` (pixel exposure) of shape (windows, 2 sides, native alpha_f rows, q_y bins), and
``binned/chunk_NNNNN.json`` the windows (``frame``, ``time_s``, ``first_time_s``, ``last_time_s``, ``raw_frames``).
``binned/run.json`` records the geometry and the SHA-256 of every chunk: each file is checked against it before it is
used. The native grid is the toolkit's own (``sputtergisaxs.linecut_v3.NATIVE_EDGES``, ``tool_qinput.Q_EDGES``,
imported read only). Its rows are merged into the regular 0.01 deg cells it is built on (every regular edge is a
native edge), so that a few narrow native rows without a pixel centre do not break the cut.

Windows where the beam may be off the sample (``structure_physical.csv``, column ``beam_off_sample``) and windows with
fewer than ``MIN_WINDOW_COUNTS`` counts in the band (shutter closed) are left out by the analysis.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

CELL_DEG = 0.01
MIN_WINDOW_COUNTS = 1000.0
TOOLKIT_ENV = "GIMAP_SPUTTER_TOOLKIT"


class SourceError(ValueError):
    """The result folder cannot be read as it is (missing or changed files, an unknown grid)."""


@dataclass(frozen=True)
class Grid:
    native_edges: np.ndarray
    q_edges: np.ndarray
    cell_edges: np.ndarray
    cell_of_row: np.ndarray

    @property
    def cell_af(self) -> np.ndarray:
        return (self.cell_edges[:-1] + self.cell_edges[1:]) / 2

    @property
    def rows(self) -> int:
        return len(self.native_edges) - 1


def make_grid(native_edges, q_edges, step: float = CELL_DEG) -> Grid:
    """The regular cells of ``step`` the native alpha_f rows are built on (each row lies in one cell)."""
    native = np.asarray(native_edges, dtype=float)
    regular = np.round(np.arange(native[0], native[-1] + step / 2, step), 10)
    if not np.all(np.min(np.abs(regular[:, None] - native[None, :]), axis=1) < 1e-7) or regular[-1] < native[-1] - 1e-7:
        raise SourceError("the native alpha_f grid is not built on regular 0.01 deg cells")
    centres = (native[:-1] + native[1:]) / 2
    cell = np.searchsorted(regular, centres, side="right") - 1
    return Grid(native, np.asarray(q_edges, dtype=float), regular, cell)


def toolkit_home(toolkit: Optional[str] = None) -> Path:
    if toolkit:
        return Path(toolkit)
    override = os.environ.get(TOOLKIT_ENV, "").strip()
    return Path(override) if override else Path(__file__).resolve().parent.parent / "sputter_toolkit"


def toolkit_grid(toolkit: Optional[str] = None) -> Grid:
    """The native grid of the Sputter toolkit (its constants, imported read only)."""
    home = str(toolkit_home(toolkit))
    sys.path.insert(0, home)
    try:
        from sputtergisaxs.linecut_v3 import NATIVE_EDGES
        from sputtergisaxs.tool_qinput import Q_EDGES
    except ImportError as exc:
        raise SourceError(f"the Sputter toolkit's grid cannot be imported from {home}: {exc}") from exc
    finally:
        if home in sys.path:
            sys.path.remove(home)
    return make_grid(NATIVE_EDGES, Q_EDGES)


@dataclass
class Scan:
    folder: Path
    geometry: dict
    metal: str
    band: tuple
    grid: Grid
    frames: list
    time: np.ndarray
    first: np.ndarray
    last: np.ndarray
    raw_frames: np.ndarray
    counts: np.ndarray
    """(windows, 2 sides, cells): counts of the q_y band."""
    exposure: np.ndarray
    """(windows, 2 sides, cells): pixel exposure of the q_y band."""
    density: np.ndarray
    """(2 sides, cells): exposure per degree of each cell, summed over all windows and the band's q_y bins."""
    beam_off: np.ndarray
    sources: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)

    @property
    def usable(self) -> np.ndarray:
        """Windows the analysis uses: on the sample and with counts in the band."""
        return (~self.beam_off) & (self.counts.sum(axis=(1, 2)) >= MIN_WINDOW_COUNTS)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _checked(path: Path, expected: Optional[str], what: str) -> bytes:
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise SourceError(f"{what} {path.name} cannot be read: {exc}") from exc
    if expected and sha256_bytes(data) != expected:
        raise SourceError(f"{what} {path.name} does not match the SHA-256 recorded in run.json (changed after binning)")
    return data


def _metal(folder: Path, run: dict) -> str:
    metal = run.get("metal")
    if not metal:
        try:
            request = json.loads((folder / "gimap_request.json").read_text(encoding="utf-8")).get("request") or {}
            metal = request.get("metal")
        except (OSError, ValueError, AttributeError):
            metal = None
    return str(metal or "")


def _beam_off(folder: Path, sources: dict, notes: list) -> dict:
    """frame -> True for the windows the toolkit flags as possibly off the sample."""
    path = folder / "structure_physical.csv"
    if not path.is_file():
        notes.append("structure_physical.csv is missing: no window is left out as off the sample")
        return {}
    data = path.read_bytes()
    sources["structure_physical.csv"] = sha256_bytes(data)
    reader = csv.DictReader(io.StringIO(data.decode("utf-8-sig")))
    if "beam_off_sample" not in (reader.fieldnames or []):
        notes.append("structure_physical.csv has no beam_off_sample column: no window is left out as off the sample")
        return {}
    flags = {row.get("frame"): str(row.get("beam_off_sample") or "").strip().lower() in ("true", "1", "yes")
             for row in reader}
    sources["beam_off_frames"] = sorted(str(frame) for frame, off in flags.items() if off)
    return flags


def read_scan(folder, grid: Grid, band: tuple = (0.05, 0.30)) -> Scan:
    folder = Path(folder)
    binned = folder / "binned"
    run_path = binned / "run.json"
    run_bytes = _checked(run_path, None, "the run record")
    try:
        run = json.loads(run_bytes.decode("utf-8"))
    except ValueError as exc:
        raise SourceError(f"binned/run.json is not valid JSON: {exc}") from exc
    geometry = dict(run.get("geometry") or {})
    for key in ("energy_ev", "alpha_i_deg"):
        value = geometry.get(key)
        if not isinstance(value, (int, float)) or not value > 0:
            raise SourceError(f"binned/run.json gives no positive {key}")
    chunks = run.get("chunks") or []
    if not chunks:
        raise SourceError("binned/run.json lists no chunks")
    sources = {"run.json": sha256_bytes(run_bytes), "chunks": {}}
    q_edges = grid.q_edges
    q_select = (q_edges[:-1] >= band[0] - 1e-9) & (q_edges[1:] <= band[1] + 1e-9)
    if not q_select.any():
        raise SourceError(f"no q_y bin lies inside the band {band[0]}-{band[1]} nm^-1")
    to_cells = np.zeros((grid.rows, len(grid.cell_edges) - 1))
    to_cells[np.arange(grid.rows), grid.cell_of_row] = 1.0
    counts, exposure, rows = [], [], []
    total = None
    for chunk in chunks:
        name = str(chunk.get("file") or "")
        data = _checked(binned / name, chunk.get("sha256"), "chunk")
        meta = _checked(binned / (name[:-4] + ".json"), chunk.get("metadata_sha256"), "chunk record")
        sources["chunks"][name] = sha256_bytes(data)
        with np.load(io.BytesIO(data), allow_pickle=False) as z:
            Y, E = z["Y"], z["E"]
        if Y.ndim != 4 or Y.shape[1:] != (2, grid.rows, len(q_edges) - 1) or E.shape != Y.shape:
            raise SourceError(f"{name}: arrays of shape {Y.shape} do not match the toolkit's native grid "
                              f"(windows, 2, {grid.rows}, {len(q_edges) - 1})")
        Yb = Y[..., q_select].astype(float)
        Eb = E[..., q_select].astype(float)
        counts.append(Yb.sum(-1) @ to_cells)
        exposure.append(Eb.sum(-1) @ to_cells)
        exposed = Eb.sum(axis=(0, -1))
        total = exposed if total is None else total + exposed
        chunk_rows = json.loads(meta.decode("utf-8")).get("rows") or []
        if len(chunk_rows) != len(Y):
            raise SourceError(f"{name}: {len(Y)} windows but {len(chunk_rows)} rows in its record")
        rows += chunk_rows
    density = (total @ to_cells) / np.diff(grid.cell_edges)
    notes: list = []
    off = _beam_off(folder, sources, notes)
    order = np.argsort([float(r.get("time_s", 0.0)) for r in rows], kind="stable")
    rows = [rows[i] for i in order]

    def column(key, fallback):
        return np.array([float(r.get(key, r.get(fallback, 0.0)) or 0.0) for r in rows])

    return Scan(folder=folder, geometry=geometry, metal=_metal(folder, run), band=(float(band[0]), float(band[1])),
                grid=grid, frames=[str(r.get("frame", "")) for r in rows], time=column("time_s", "time_s"),
                first=column("first_time_s", "time_s"), last=column("last_time_s", "time_s"),
                raw_frames=np.array([int(r.get("raw_frames") or 0) for r in rows]),
                counts=np.concatenate(counts)[order], exposure=np.concatenate(exposure)[order], density=density,
                beam_off=np.array([bool(off.get(str(r.get("frame")), False)) for r in rows]), sources=sources,
                notes=notes)


__all__ = ["CELL_DEG", "Grid", "MIN_WINDOW_COUNTS", "Scan", "SourceError", "make_grid", "read_scan",
           "sha256_bytes", "toolkit_grid", "toolkit_home"]
