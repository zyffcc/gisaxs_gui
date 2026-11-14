"""Invented results of the client analysis package and a fake of its process (no package, no real data): shared by
``test_sputter_client.py`` and ``test_sputter_main_screen.py``.

The output folder written here has the layout the package writes: ``analysis.json``, ``main/main.json`` and one
``main/w_<t>.npz`` per time window. Three kinds: a scan on a flat substrate (``write_result``), one on a porous
substrate (``porous_windows`` with ``POROUS_WANTED``) and a series of static samples (``write_static``: one
``main/w_<label>.npz`` per sample). Every number and name is made up.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.gimap.features.sputter.application import ClientPackage, ClientRun, StaticDefaults

WANTED = [{"key": "R_nm", "label": "grain radius R", "unit": "nm"}, {"key": "H_nm", "label": "grain height H", "unit": "nm"},
          {"key": "D_nm", "label": "centre distance D", "unit": "nm"},
          {"key": "gap_nm", "label": "gap between grains D - 2R", "unit": "nm"},
          {"key": "width", "label": "width of the radius distribution (log-normal)", "unit": ""},
          {"key": "coverage", "label": "coverage", "unit": ""}, {"key": "amount_nm", "label": "metal amount", "unit": "nm"}]
POROUS_WANTED = WANTED + [{"key": "f_islands", "label": "share of the metal in resolvable islands", "unit": ""},
                          {"key": "cap_share", "label": "share of the metal in the cap layer", "unit": ""},
                          {"key": "depth_nm", "label": "vertical correlation length", "unit": "nm"},
                          {"key": "xi_nm", "label": "lateral correlation length", "unit": "nm"},
                          {"key": "film_share", "label": "share that leaves through the film", "unit": ""}]
"""The package lists every quantity for every model family: a porous result just has no value for some."""
STATIC_WANTED = [item for item in WANTED if item["key"] != "width"]
STATIC_AXIS = {"key": "h_nom", "label": "nominal thickness", "unit": "nm"}
Q = np.linspace(0.05, 1.5, 12)
AF = np.concatenate([np.linspace(0.02, 0.2, 6), np.linspace(0.25, 1.4, 8)])  # not evenly spaced, as the package's


def entry(label: str, deviation, radius, *, kind: str = "candidate", rank=None, weight=None, **extra) -> dict:
    """One solution: D = 4R, the gap 2R, a height unless ``H_nm=None`` is given."""
    values = {"label": label, "kind": kind, "rank": rank, "weight": weight, "fit_deviation": deviation,
              "R_nm": radius, "H_nm": round(1.5 * radius, 3), "D_nm": 4.0 * radius, "gap_nm": 2.0 * radius,
              "width": 0.21, "coverage": 0.3, "amount_nm": 0.5 * radius}
    values.update(extra)
    return values


def window(t: float, entries: list, *, pick: str = "", smooth: str = "", h_nom=None) -> dict:
    name = f"w_{t:g}"
    return {"t": t, "t_rel": t - 0.5, "h_nom": 0.2 * t if h_nom is None else h_nom, "image": name + ".png",
            "arrays": name + ".npz", "entries": entries, "pick": pick, "path": pick, "smooth": smooth}


def default_windows() -> list:
    """Three windows with solutions and one the package could not use.

    2 s: the smooth evolution takes candidate 2 although the refined one fits best; 4 s: no ``smooth`` (the lowest
    deviation is shown), a candidate without height; 6 s: no solution; 8 s: candidates only, no height at all."""
    return [
        window(2.0, [entry("candidate 1", 3.2, 1.0, rank=1, weight=0.7), entry("candidate 2", 4.4, 1.5, rank=2, weight=0.2),
                     entry("candidate 3", 9.9, 2.5, rank=3, weight=0.1), entry("refined", 2.1, 1.2, kind="refined")],
               pick="refined", smooth="candidate 2"),
        window(4.0, [entry("candidate 1", 5.5, 2.0, rank=1, weight=0.6), entry("candidate 2", 3.3, 2.6, rank=2, weight=0.4,
                                                                               H_nm=None),
                     entry("refined", 2.8, 2.2, kind="refined")], pick="refined"),
        {"t": 6.0, "error": "no frames in this time window"},
        window(8.0, [entry("candidate 1", 6.1, 3.0, rank=1, weight=0.9, H_nm=None, coverage=0.55),
                     entry("candidate 2", 7.7, 3.4, rank=2, weight=0.1, H_nm=None, coverage=None)],
               pick="candidate 1", smooth="candidate 1"),
    ]


def porous_entry(label: str, deviation, radius, *, rank=None, weight=None, **extra) -> dict:
    """One solution on a porous substrate: no height, distance, gap or coverage."""
    values = {"label": label, "kind": "candidate", "rank": rank, "weight": weight, "fit_deviation": deviation,
              "R_nm": radius, "width": 0.17, "f_islands": round(2.4 / radius, 3), "cap_share": 0.25,
              "depth_nm": round(1.5 + 0.1 * radius, 3), "xi_nm": 4.0 * radius, "film_share": 0.04,
              "amount_nm": 0.6 * radius}
    values.update(extra)
    return values


def porous_windows() -> list:
    """Three windows on a porous substrate, two candidates each (3 s: the island share is above 1)."""
    return [window(t, [porous_entry("candidate 1", 4.0 + t, radius, rank=1, weight=0.8),
                       porous_entry("candidate 2", 6.0 + t, radius + 0.5, rank=2, weight=0.2)],
                   pick="candidate 1", smooth="candidate 1")
            for t, radius in ((3.0, 1.2), (6.0, 2.0), (9.0, 3.0))]


def static_sample(position: int, label: str, nominal: float, radius: float, deviation: float, *,
                  substrate: str = "Si", reference: str = "analytic_si", frames: int = 4) -> dict:
    """One sample of a static series: one entry (D = 3R, the gap R) with the range of each value but the gap."""
    values = {"R_nm": radius, "H_nm": round(1.2 * radius, 4), "D_nm": 3.0 * radius, "gap_nm": radius,
              "coverage": 0.4, "amount_nm": round(0.9 * nominal, 4)}
    ranges = {key: [round(0.9 * value, 4), round(1.2 * value, 4)] for key, value in values.items() if key != "gap_nm"}
    found = {"label": "model", "kind": "candidate", "rank": None, "weight": None, "fit_deviation": deviation,
             **values, "ranges": ranges, "source": "model"}
    return {"t": float(position), "t_rel": None, "h_nom": nominal, "label": label, "sample": "the sample " + label,
            "substrate": substrate, "reference": reference, "frames": frames, "image": None,
            "arrays": f"w_{label}.npz", "entries": [found], "pick": "model", "smooth": "model", "path": None}


def static_samples() -> list:
    """Four samples in the order of the series; two share a nominal thickness, the last has a measured substrate."""
    return [static_sample(1, "zz_a", 4.0, 2.0, 3.1), static_sample(2, "zz_b", 4.0, 2.5, 4.2),
            static_sample(3, "zz_c", 7.0, 3.5, 2.6),
            static_sample(4, "zz_d", 10.0, 5.0, 5.4, substrate="bare substrate measured", reference="zz_d_ref",
                          frames=3)]


def patterns(labels: list, seed: int = 0) -> dict:
    """Measured counts with a gap of no data (NaN) and one model per label, each a different multiple of a peak."""
    rng = np.random.default_rng(seed)
    qq, aa = np.meshgrid(Q, AF)
    peak = 5.0 + 200.0 * np.exp(-((qq - 0.6) / 0.25) ** 2) * np.exp(-aa / 0.4)
    measured = peak * (1.0 + 0.05 * rng.standard_normal(peak.shape))
    measured[5, :] = np.nan
    measured[0, 0] = 0.0
    models = np.stack([peak * (1.0 + 0.1 * (index + 1)) for index in range(len(labels))])
    models[:, 5, :] = np.nan
    return {"q": Q.astype(np.float32), "af": AF.astype(np.float32), "measured": measured.astype(np.float32),
            "models": models.astype(np.float32), "labels": np.array(labels)}


def write_result(folder: Path, *, windows=None, status: str = "ok", with_main: bool = True, report: bool = True,
                 scan: str = "zz_invented_scan", metal: str = "Ag", family: str = "zz_family", wanted=None) -> Path:
    """An output folder as the package writes it (``analysis.json``, ``main/``); returns the folder."""
    folder = Path(folder)
    main = folder / "main"
    main.mkdir(parents=True, exist_ok=True)
    windows = default_windows() if windows is None else windows
    files = {key: None for key in ("report_md", "report_png", "report_csv", "windows_csv", "evolution_png",
                                   "patterns_png", "summary_json", "main_json")}
    if report:
        (folder / "REPORT.md").write_text("# an invented report\n", encoding="utf-8")
        files["report_md"] = str(folder / "REPORT.md")
    if with_main:
        for index, item in enumerate(windows):
            if item.get("arrays"):
                np.savez(main / item["arrays"], **patterns([e["label"] for e in item["entries"]], seed=index))
        (main / "main.json").write_text(json.dumps(
            {"scan": scan, "family": family, "metal": metal, "wanted": wanted or WANTED, "note": "an invented note",
             "choice": "an invented choice note", "windows": windows}), encoding="utf-8")
        files["main_json"] = str(main / "main.json")
    steps = [{"step": name, "returncode": 0, "seconds": 1.0, "log": str(folder / f"analyze_{name}.log")}
             for name in ("intake", "predict", "report", "present")]
    if status != "ok":
        failed = status.rsplit(" ", 1)[-1]
        for step in steps:
            if step["step"] == failed:
                step["returncode"] = 1
                Path(step["log"]).write_text("an invented first line\nthe invented reason of the failure\n",
                                             encoding="utf-8")
    (folder / "analysis.json").write_text(json.dumps(
        {"status": status, "label": "gui_" + scan, "metal": metal, "substrate": "Si", "family": family,
         "mode": "quick", "steps": steps, "files": files, "note": "an invented note"}), encoding="utf-8")
    return folder


def write_static(folder: Path, *, samples=None, status: str = "ok", label=None, data_root=None) -> Path:
    """A series of static samples as the package writes it; ``status`` ``failed at <step>`` gives that step a
    note with its reason (and, for ``predict``, the log the note names). Returns the folder."""
    folder = Path(folder)
    main = folder / "main"
    main.mkdir(parents=True, exist_ok=True)
    samples = static_samples() if samples is None else samples
    (folder / "REPORT.md").write_text("# an invented report of a series\n", encoding="utf-8")
    for index, item in enumerate(samples):
        np.savez(main / item["arrays"], **patterns(["model"], seed=index))
        (folder / item["label"]).mkdir(exist_ok=True)
        (folder / item["label"] / "frames.csv").write_text("label,frame\n", encoding="utf-8")
    (main / "main.json").write_text(json.dumps(
        {"scan": folder.name, "kind": "static", "family": "zz_static_family", "metal": "Au", "x": STATIC_AXIS,
         "wanted": STATIC_WANTED, "note": "an invented note", "choice": "one entry per sample", "windows": samples}),
        encoding="utf-8")
    steps = [{"step": name, "returncode": 0, "seconds": 1.0, "note": None} for name in ("intake", "predict", "report")]
    if status != "ok":
        failed = status.rsplit(" ", 1)[-1]
        steps = steps[:[step["step"] for step in steps].index(failed) + 1]
        steps[-1]["returncode"] = 1
        steps[-1]["note"] = "the invented reason of the failure"
        if failed == "predict":
            log = folder / "analyze_predict_zz.log"
            log.write_text("an invented first line\nthe last line of the invented log\n", encoding="utf-8")
            steps[-1]["note"] = f"the invented step failed (see {log})"
    files = {"report_md": str(folder / "REPORT.md"), "report_png": None, "report_csv": None, "patterns_png": None,
             "main_json": str(main / "main.json")}
    (folder / "analysis.json").write_text(json.dumps(
        {"status": status, "kind": "static", "label": label, "samples": [item["label"] for item in samples],
         "mode": "quick", "data_root": str(data_root or folder / "zz_no_such_data"), "steps": steps, "files": files,
         "note": "an invented note"}), encoding="utf-8")
    return folder


def package_folder(folder: Path, *, manifest: bool = True) -> Path:
    """A folder with the parts of an analysis package (empty files: it is never run)."""
    folder = Path(folder)
    for part in ("code/iam/iampred", "code/gimap", "code/toolkit", "models"):
        (folder / part).mkdir(parents=True, exist_ok=True)
    (folder / "code/iam/iampred/analyze.py").write_text("", encoding="utf-8")
    if manifest:
        (folder / "MANIFEST.json").write_text('{"invented": true}', encoding="utf-8")
    return folder


STATIC_SOURCE = """GEOMETRY = dict(energy_ev=9500.0, sdd_mm=4321.0, pixel_mm=0.1, db_row=700.0, db_col=300.0,
                sample_roll_deg=-0.25, alpha_i_deg=0.35)
MODES = dict(quick=dict(refine=0), refine=dict(refine=2), full=dict(refine=5))
"""
"""What a package's ``static`` module assigns (invented values): GIMaP reads its geometry from the source."""


def static_package(folder: Path) -> Path:
    """A package that knows static frames and brings two pixel masks."""
    folder = package_folder(folder)
    (folder / "code/iam/iampred/static.py").write_text(STATIC_SOURCE, encoding="utf-8")
    (folder / "code/assets").mkdir(parents=True, exist_ok=True)
    for name in ("zz_second_mask.npy", "zz_first_mask.npy"):
        np.save(folder / "code/assets" / name, np.ones((2, 2), dtype=bool))
    (folder / "code/assets/zz_not_a_mask.txt").write_text("", encoding="utf-8")
    return folder


STATIC_LINES = ("[intake] ...", "      zz: 4 frames, 4 usable, 1e+06 counts", "      done in 2 s", "[predict] ...",
                "      done in 3 s", "[report] ...", "      done in 1 s")
PROGRESS_LINES = ("[1/4] intake ...", "      done in 3 s", "a plain line of the package", "[2/4] predict ...",
                  "      done in 5 s", "[3/4] report ...", "      done in 1 s", "[4/4] present ...",
                  "      done in 2 s")


class FakeClientRunner:
    """The package's process: says its steps and writes an invented result (or fails, or waits to be stopped)."""

    def __init__(self, *, missing: str = "", status: str = "ok", with_main: bool = True, wait_for_stop: bool = False,
                 code=None, windows=None):
        self.missing, self.status, self.with_main, self.wait_for_stop = missing, status, with_main, wait_for_stop
        self.code, self.windows = code, windows
        self.requests: list = []
        self.folders: list = []
        self.chosen = None
        self.mask_files: list = []

    def package(self) -> ClientPackage:
        if self.missing:
            return ClientPackage(None, "", self.missing)
        return ClientPackage(Path("zz_invented_package"), "chosen", "", "0" * 64)

    def choose(self, folder) -> None:
        self.chosen = folder
        self.missing = ""

    def masks(self) -> list:
        return list(self.mask_files)

    def run(self, request, out_folder, on_line, should_stop) -> ClientRun:
        import time

        self.requests.append(request)
        self.folders.append(Path(out_folder))
        if self.wait_for_stop:
            on_line(PROGRESS_LINES[0])
            end = time.monotonic() + 20
            while not should_stop() and time.monotonic() < end:
                time.sleep(0.01)
            return ClientRun(None, True, (PROGRESS_LINES[0],))
        for line in PROGRESS_LINES:
            on_line(line)
        write_result(out_folder, status=self.status, with_main=self.with_main, windows=self.windows)
        on_line(f"status: {self.status}; report: none")
        code = self.code if self.code is not None else (0 if self.status == "ok" else 1)
        return ClientRun(code, False, (*PROGRESS_LINES, f"status: {self.status}; report: none"))


class FakeStaticRunner:
    """The package's ``static`` command: says its steps and writes the series with one more sample (or fails, or
    waits to be stopped). ``defaults`` are those of ``STATIC_SOURCE``, the beam centre in GIMaP's coordinates."""

    def __init__(self, series_root: Path, *, problem: str = "", status: str = "ok", wait_for_stop: bool = False):
        self.series_root, self.problem, self.status = Path(series_root), problem, status
        self.wait_for_stop = wait_for_stop
        self.requests: list = []
        self.archive: list = []
        self.samples: dict = {}
        """Series folder -> {label: (the sample, the digest of its request, the digest of its import)}."""

    def package(self) -> ClientPackage:
        return ClientPackage(Path("zz_invented_package"), "chosen", "", "0" * 64)

    def defaults(self) -> StaticDefaults:
        return StaticDefaults(9500.0, 4321.0, 300.5, 700.5, -0.25, 0.35, self.problem)

    def default_series(self) -> Path:
        return self.series_root / "series_1"

    def archive_folders(self, archive) -> list:
        return list(self.archive)

    def _kept(self, request) -> tuple:
        return self.samples.get(str(request.series_folder), {}).get(request.label(), (None, "", ""))

    def analysed(self, request) -> bool:
        return self._kept(request)[1] == request.digest("0" * 64)

    def replaces(self, request) -> bool:
        return bool(self._kept(request)[2]) and self._kept(request)[2] != request.intake_digest()

    def run(self, request, on_line, should_stop) -> ClientRun:
        import time

        self.requests.append(request)
        if self.wait_for_stop:
            on_line(STATIC_LINES[0])
            end = time.monotonic() + 20
            while not should_stop() and time.monotonic() < end:
                time.sleep(0.01)
            return ClientRun(None, True, (STATIC_LINES[0],))
        for line in STATIC_LINES:
            on_line(line)
        series = self.samples.setdefault(str(request.series_folder), {})
        if self.status == "ok":
            sample = static_sample(0, request.label(), request.nominal_nm, 0.5 * request.nominal_nm, 3.3)
            series[request.label()] = (sample, request.digest("0" * 64), request.intake_digest())
        ordered = sorted((kept[0] for kept in series.values()), key=lambda item: (item["h_nom"], item["label"]))
        write_static(request.series_folder, status=self.status, label=request.label(),
                     samples=[item | {"t": float(index)} for index, item in enumerate(ordered, 1)])
        on_line(f"status: {self.status}; report: none")
        return ClientRun(0 if self.status == "ok" else 1, False, (*STATIC_LINES, f"status: {self.status}; report: none"))


__all__ = ["AF", "FakeClientRunner", "FakeStaticRunner", "POROUS_WANTED", "PROGRESS_LINES", "Q", "STATIC_AXIS",
           "STATIC_LINES", "STATIC_SOURCE", "STATIC_WANTED", "WANTED", "default_windows", "entry", "package_folder",
           "patterns", "porous_entry", "porous_windows", "static_package", "static_sample", "static_samples", "window",
           "write_result", "write_static"]
