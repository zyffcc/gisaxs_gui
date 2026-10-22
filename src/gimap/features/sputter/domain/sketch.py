"""Real-space sketch of one time window: what the measured sizes look like (pure Python).

Islands: flat-based hemispheres (h = R) with a lognormal number size
distribution (number mean R, width sigma), centre-to-centre spacing from the
model D only when it agrees with the direct peak, otherwise the direct
2π/q*; without a resolved peak the spacing is not measured and the sketch
says so. Late TS windows: a continuous film whose lateral structure has the
characteristic length L. Burial depth, polymer thickness and TS metal
fraction/thickness are unreliable; they are drawn only on request and marked.

The random numbers (positions, sizes, film texture) are fixed per sketcher,
so moving through time changes only the measured values, not the dice.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
from typing import Optional

DEFAULT_DISORDER = 0.25     # relative spacing disorder drawn when no refined value exists (illustration)
UNKNOWN_SPACING_FACTOR = 4.0  # islands drawn 4R apart when no spacing is measured (illustration only)
OVERLAP_COVERAGE = 0.9        # above this projected coverage the drawn islands overlap: R/spacing inconsistent


@dataclass
class WindowSketch:
    time_s: Optional[float]
    kind: str                      # "islands", "film" or "none"
    R: Optional[float] = None
    R_grade: Optional[str] = None
    R_error: Optional[float] = None
    sigma: float = 0.0
    spacing: Optional[float] = None
    spacing_source: str = ""       # "model D (grade X)", "direct 2π/q*", "not measured"
    disorder: float = DEFAULT_DISORDER
    disorder_measured: bool = False
    burial: Optional[float] = None
    thickness: Optional[float] = None
    L: Optional[float] = None
    metal_fraction: Optional[float] = None
    film_thickness: Optional[float] = None
    stage: str = ""
    notes: list = field(default_factory=list)

    @property
    def coverage(self) -> Optional[float]:
        """Projected area fraction of the hemisphere bases on a hexagonal arrangement (model-derived)."""
        if self.kind != "islands" or not self.R or not self.spacing or self.spacing_source == "not measured":
            return None
        mean_r2 = self.R ** 2 * math.exp(self.sigma ** 2)       # number-mean R, lognormal width sigma
        return min(1.0, math.pi * mean_r2 * 2.0 / (math.sqrt(3.0) * self.spacing ** 2))


def window_sketch(row: dict) -> WindowSketch:
    """Pick the values to draw for one structure-table row (same rules as the reliability tab)."""
    t = row.get("time_s"); module = row.get("module") or ""
    if module == "TS_layer" and row.get("L_TS_nm"):
        s = WindowSketch(t, "film", L=row["L_TS_nm"], metal_fraction=row.get("TS_metal_fraction_conditional"),
                         film_thickness=row.get("TS_film_thickness_nm_conditional"), stage=row.get("stage") or "")
        s.spacing, s.spacing_source = row["L_TS_nm"], "TS length L"
        s.notes.append("late window: continuous-film (TS) model; L is a characteristic length, not an island spacing")
        return s
    R = row.get("R_nm")
    if not R:
        return WindowSketch(t, "none", stage=row.get("stage") or "", notes=[row.get("R_reason") or "no size estimate"])
    s = WindowSketch(t, "islands", R=R, R_grade=row.get("R_grade"), R_error=row.get("R_typical_rel_error"),
                     sigma=float(row.get("sigma_log") or 0.0), burial=row.get("burial_nm_conditional"),
                     thickness=row.get("thickness_nm_conditional"), stage=row.get("stage") or "")
    if row.get("spacing_disorder"):
        s.disorder, s.disorder_measured = float(row["spacing_disorder"]), True
    D, check, direct = row.get("D_nm"), row.get("D_direct_check"), row.get("D_corr_nm")
    if D and check == "consistent":
        s.spacing, s.spacing_source = D, f"model D (grade {row.get('D_grade') or '?'})"
    elif direct:
        s.spacing, s.spacing_source = direct, "direct 2π/q*"
        if D:
            s.notes.append(f"model D {D:.3g} nm disagrees with the peak; spacing drawn from 2π/q*")
    else:
        s.spacing, s.spacing_source = UNKNOWN_SPACING_FACTOR * R, "not measured"
        s.notes.append("no resolved peak: spacing not measured, islands drawn 4R apart for illustration only")
    if s.coverage is not None and s.coverage > OVERLAP_COVERAGE:
        s.notes.append(f"islands would overlap at this R and spacing (coverage {100 * s.coverage:.0f}% > "
                       f"{100 * OVERLAP_COVERAGE:.0f}%): R or the spacing is not reliable here")
    if row.get("smoothed_windows"):
        s.notes.append(f"median of {row['smoothed_windows']} neighbouring windows of the same model")
    return s


SMOOTHED_KEYS = ("R_nm", "sigma_log", "D_corr_nm", "L_TS_nm", "spacing_disorder", "burial_nm_conditional",
                 "thickness_nm_conditional", "TS_metal_fraction_conditional", "TS_film_thickness_nm_conditional")


def _median(values):
    v = sorted(values); n = len(v)
    return v[n // 2] if n % 2 else .5 * (v[n // 2 - 1] + v[n // 2])


def smoothed_rows(rows: list, half: int = 5) -> list:
    """Median over ±``half`` neighbouring windows of the same model (suppresses single-window jumps).

    Only values present in the centre window are replaced; a model D is
    replaced by the median of neighbouring D that passed the peak check.
    """
    out = []
    for i, r in enumerate(rows):
        same = [x for x in rows[max(0, i - half):i + half + 1] if x.get("module") == r.get("module")]
        nr = dict(r)
        for key in SMOOTHED_KEYS:
            values = [x[key] for x in same if x.get(key) is not None]
            if r.get(key) is not None and values:
                nr[key] = _median(values)
        consistent = [x["D_nm"] for x in same if x.get("D_nm") and x.get("D_direct_check") == "consistent"]
        if r.get("D_nm") and r.get("D_direct_check") == "consistent" and consistent:
            nr["D_nm"] = _median(consistent)
        nr["smoothed_windows"] = len(same)
        out.append(nr)
    return out


@dataclass
class Particle:
    x: float
    y: float
    r: float


class Sketcher:
    """Fixed dice for one series: side row, top-view lattice jitter, sizes and a film texture."""

    MAX_TOP = 160   # lattice sites per side; enough to fill a 150 nm view at about 1 nm spacing

    def __init__(self, seed: int = 7, side_count: int = 60):
        rng = random.Random(seed)
        self.side_count = side_count
        self.side_z = [rng.gauss(0, 1) for _ in range(side_count)]
        self.side_jitter = [rng.gauss(0, 1) for _ in range(side_count)]
        n = self.MAX_TOP
        self.top_z = [[rng.gauss(0, 1) for _ in range(n)] for _ in range(n)]
        self.top_jitter = [[(rng.gauss(0, 1), rng.gauss(0, 1)) for _ in range(n)] for _ in range(n)]
        self.film_waves = [(rng.uniform(0, 2 * math.pi), rng.uniform(0, 2 * math.pi), rng.uniform(.8, 1.25))
                           for _ in range(24)]

    @staticmethod
    def _radius(R, sigma, z):
        return R * math.exp(sigma * max(-2.5, min(2.5, z)) - .5 * sigma ** 2)

    def side_row(self, s: WindowSketch, width: float) -> list:
        """Islands along a cut, x in [0, width] (nm), base at y = 0."""
        out = []
        if s.kind != "islands":
            return out
        for i in range(self.side_count):
            x = (i + .5) * s.spacing + self.side_jitter[i] * s.disorder * s.spacing * .5
            if x > width + s.spacing:
                break
            out.append(Particle(x, 0.0, self._radius(s.R, s.sigma, self.side_z[i])))
        return out

    def top_view(self, s: WindowSketch, width: float) -> list:
        """Islands on a hexagonal arrangement with jitter, filling [0, width]^2 (nm)."""
        out = []
        if s.kind != "islands":
            return out
        a = s.spacing; h = a * math.sqrt(3) / 2
        rows = min(self.MAX_TOP, int(math.ceil(width / h)) + 2); cols = min(self.MAX_TOP, int(math.ceil(width / a)) + 2)
        for j in range(rows):
            y0 = (j + .5) * h
            for i in range(cols):
                x0 = (i + .5 + (.5 if j % 2 else 0.)) * a
                jx, jy = self.top_jitter[j][i]
                out.append(Particle(x0 + jx * s.disorder * a * .5, y0 + jy * s.disorder * a * .5,
                                    self._radius(s.R, s.sigma, self.top_z[j][i])))
        return out

    def film_field(self, s: WindowSketch, width: float, n: int = 160) -> list:
        """Level-cut texture with characteristic length L: n x n booleans (metal = True)."""
        if s.kind != "film" or not s.L:
            return []
        k0 = 2 * math.pi / s.L
        waves = [(k0 * m * math.cos(a), k0 * m * math.sin(a), p) for a, p, m in self.film_waves]
        values = []
        for j in range(n):
            y = (j + .5) * width / n
            values.append([sum(math.cos(kx * (i + .5) * width / n + ky * y + p) for kx, ky, p in waves) for i in range(n)])
        frac = s.metal_fraction if s.metal_fraction and 0.05 <= s.metal_fraction <= .95 else .6
        flat = sorted(v for row in values for v in row)
        cut = flat[int((1 - frac) * (len(flat) - 1))]
        return [[v >= cut for v in row] for row in values]


def series_extent(sketches: list) -> tuple[float, float]:
    """Common width and island height scale (nm) so all windows share one ruler."""
    spacings = [s.spacing for s in sketches if s.kind in ("islands", "film") and s.spacing]
    radii = [s.R * math.exp(s.sigma) for s in sketches if s.kind == "islands" and s.R]
    width = min(150.0, max(30.0, 6 * (sorted(spacings)[int(.9 * (len(spacings) - 1))] if spacings else 10.0)))
    height = max(3.0, 1.3 * (sorted(radii)[int(.95 * (len(radii) - 1))] if radii else 3.0))
    return width, height


def storyboard_indices(sketches: list, count: int = 6) -> list:
    """Times spread on a logarithmic axis (growth is fastest early), among drawable windows.

    A nearby window whose spacing was measured (peak or TS length) is preferred
    over an illustrative one.
    """
    usable = [i for i, s in enumerate(sketches) if s.kind != "none" and s.time_s is not None and s.time_s > 0]
    if not usable:
        return []
    t1 = sketches[usable[-1]].time_s
    t0 = max(min(sketches[i].time_s for i in usable), min(1.0, t1 / 100.))
    picks = []
    for k in range(count):
        target = math.log(t0) + (math.log(t1) - math.log(t0)) * k / max(1, count - 1)
        best = min(usable, key=lambda i: abs(math.log(max(sketches[i].time_s, 1e-6)) - target)
                   + (.15 if sketches[i].spacing_source == "not measured" else 0.))
        if best not in picks:
            picks.append(best)
    return sorted(picks, key=lambda i: sketches[i].time_s)


__all__ = ["Particle", "Sketcher", "WindowSketch", "series_extent", "smoothed_rows", "storyboard_indices",
           "window_sketch"]
