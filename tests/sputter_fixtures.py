"""Invented Sputter results and fakes of the model, PyTorch's installer and the scan header (no network, no
PyTorch, no real data): shared by ``test_sputter_feature.py`` and ``test_sputter_page.py``."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from src.gimap.features.sputter.application import EnvironmentStatus, InstallError, InstallProgress, ScanInfo, Stopped

ROWS = [
    dict(frame="bin_000000", time_s="0.05", stage="pre", module="polymer_embed", feature_status="no_resolved_peak",
         R_nm="2.0", R_grade="C", R_lo_nm="1.5", R_hi_nm="2.6", R_typical_rel_error="0.307", D_nm="", D_grade="",
         D_reason="no resolved peak: D not reported", D_corr_nm="", L_TS_nm="", metal="Ag"),
    dict(frame="bin_000001", time_s="0.6", stage="islands", module="polymer_embed", feature_status="bilateral",
         R_nm="2.4", R_grade="A", R_lo_nm="2.1", R_hi_nm="2.8", R_typical_rel_error="0.157", D_nm="9.0", D_grade="A",
         D_lo_nm="8.0", D_hi_nm="10.1", D_typical_rel_error="0.119", D_corr_nm="8.7", L_TS_nm="", metal="Ag"),
    dict(frame="bin_000002", time_s="1.1", stage="film", module="TS_layer", feature_status="bilateral",
         R_nm="", R_grade="", D_nm="", D_grade="", D_corr_nm="9.5", L_TS_nm="12.0", metal="Ag",
         R_reason="late TS window: island R/D not defined (see L_TS_nm)"),
]
SUMMARY = dict(metal="Ag", TS_windows=1, R=dict(window_to_window_scatter=0.02), D=dict(window_to_window_scatter=None),
               reliability=dict(parameters=dict(R=dict(Ag=dict(network=dict(typical_rel_scatter=.307, order_accuracy_1p2x=.92),
                                                                refined=dict(typical_rel_scatter=.157, order_accuracy_1p2x=.935))),
                                                D=dict(Ag=dict(network=dict(typical_rel_scatter=.26, order_accuracy_1p2x=.90),
                                                                refined=dict(typical_rel_scatter=.119, order_accuracy_1p2x=.918))))))
_PHYSICAL_BASE = dict(module="polymer_embed", module_path="polymer_embed", film="false", path_changed="false",
                      R_source="network", D_source="network", metal="Ag")
PHYSICAL_ROWS = [
    dict(_PHYSICAL_BASE, frame="bin_000000", time_s="0.5", stage="pre", stage_path="pre_nucleation", R_nm="1.8",
         R_grade="P", R_typical_rel_error="0.31", R_trend_nm="1.9", D_nm="", D_grade="", D_reason="no resolved peak",
         t_eq_nm="", nominal_thickness_nm="0.1"),
    dict(_PHYSICAL_BASE, frame="bin_000001", time_s="1.0", stage="islands", stage_path="islands", R_nm="2.4",
         R_grade="P", R_typical_rel_error="0.31", R_trend_nm="2.3", D_nm="9.0", D_grade="P", D_trend_nm="9.1",
         D_corr_nm="8.7", D_direct_check="consistent", t_eq_nm="0.30", t_eq_lo_nm="0.12", t_eq_hi_nm="0.80",
         nominal_thickness_nm="0.2", refined_R_nm="2.6", refined_D_nm="9.4"),
    dict(_PHYSICAL_BASE, frame="bin_000002", time_s="1.5", stage="film", stage_path="islands", path_changed="true",
         R_nm="2.6", R_grade="P", R_typical_rel_error="0.31", R_trend_nm="2.5", D_nm="9.6", D_grade="P",
         D_trend_nm="9.5", D_corr_nm="9.9", D_direct_check="consistent", t_eq_nm="0.36", t_eq_lo_nm="0.15",
         t_eq_hi_nm="0.9", nominal_thickness_nm="0.3"),
    dict(_PHYSICAL_BASE, frame="bin_000003", time_s="2.0", stage="film", stage_path="film", module="TS_layer",
         module_path="TS_layer", film="true", R_nm="", R_grade="", D_nm="", D_grade="", L_TS_nm="40",
         t_eq_nm="", nominal_thickness_nm="0.4", film_thickness_status="not determined"),
]
PHYSICAL_SUMMARY = dict(
    SUMMARY, TS_windows=1, film_onset_s=2.0, film_onset_interval_s=[1.5, 2.5], rotation_period_s=7.0,
    rotation_detection=dict(method="acf", acf_peak=0.42, cycles=9),
    reversals=dict(before=dict(stage=3, module_revert=1, R_zigzag=4, D_zigzag=2),
                   after=dict(stage=0, module_revert=0, R_zigzag=0, D_zigzag=1)),
    thickness=dict(rate_nm_s=0.2, t0_s=0.0, t_eq_note="invented t_eq note", nominal_note="invented nominal note",
                   film_thickness_note="invented film note",
                   mass_continuity=dict(time_s=2.0, t_eq_nm=0.36, nominal_nm=0.4, ratio=0.9)),
    notes=["an invented note of the model"])


def write_table(folder: Path, name: str, rows: list, summary: dict) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with (folder / f"{name}.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    (folder / f"{name}_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return folder


def result_folder(folder: Path, *, physical: bool = False) -> Path:
    write_table(folder, "structure", ROWS, SUMMARY)
    if physical:
        write_table(folder, "structure_physical", PHYSICAL_ROWS, PHYSICAL_SUMMARY)
    return folder


# -- the literature method (invented values, written as the toolkit writes them with a deposition rate) -----------
_LIT = [
    dict(delta_nm="0.1", lit_status="no_peak", lit_reason="no resolved peak: D and R not given"),
    dict(delta_nm="0.2", D_lit_nm="8.7", D_lit_trend_nm="8.7", R_lit_nm="1.84", R_lit_trend_nm="1.84",
         two_R_over_D_lit="0.42", two_R_over_D_lit_trend="0.42", coverage_lit="0.16", two_R_over_D_net="0.53",
         coverage_net="0.26", eta_net="1.5", lit_status="islands", lit_reason=""),
    dict(delta_nm="0.3", D_lit_nm="9.9", D_lit_trend_nm="9.9", R_lit_nm="2.3", R_lit_trend_nm="2.3",
         two_R_over_D_lit="0.46", two_R_over_D_lit_trend="0.46", coverage_lit="0.19", two_R_over_D_net="0.54",
         coverage_net="0.27", eta_net="1.2", lit_status="islands", lit_reason=""),
    dict(delta_nm="0.4", D_corr_nm="10.4", D_lit_nm="10.4", D_lit_trend_nm="9.9", two_R_over_D_lit_trend="1.02",
         lit_status="after_percolation",
         lit_reason="after the percolation of the literature method: D = 2*pi/q* kept, R not given (film)"),
]
LITERATURE_ROWS = [dict(row, **lit) for row, lit in zip(PHYSICAL_ROWS, _LIT)]
LITERATURE = dict(
    status="ok", island_windows=2, message="", metal="Ag", polymer="PS", host_class="Ag/polymers",
    host_note="The band of Ag/polymers includes published values for PS.", rate_nm_s=0.2, t0_s=0.0, t0_assumed=True,
    percolation=dict(state="reached", delta_nm=0.35, time_s=1.75, interval_delta_nm=[0.32, 0.38],
                     interval_time_s=[1.6, 1.9], reached=True, bound=None, max_two_R_over_D_trend=1.02,
                     last_delta_nm=0.4, last_time_s=2.0, rate_error=0.1, t0_error_s=2.0),
    band=dict(host_class="Ag/polymers", low_nm=4.7, high_nm=13.5, rate_scaled_nm=[3.9, 10.2], tolerance=0.1,
              entries=[]),
    reference_curves=[dict(id="Gen19_PS", metal="Ag", host="PS", label="Ag/PS RF", ref="Gen19",
                           points=[[1.0, 11.3], [2.0, 18.0], [5.0, 27.0]])],
    two_R_over_D_bands=[dict(regime="islands and coalescence", metal="Ag", range=[0.66, 0.95], kind="mass balance",
                             refs=["GenTh23"])],
    coverage_bands=[dict(condition="delta = 1 nm", delta_nm=[1.0, 1.0], range=[0.5, 0.65], refs=["Bulut24"])],
    coverage_at_percolation=dict(range=[0.56, 0.9]), eta_warning=0.5,
    checks=[
        dict(id="D_monotone", title="D (= 2*pi/q*) does not decrease with delta", verdict="consistent",
             measured="Spearman rho = 1 over 2 windows; no drop of the binned median D beyond the scatter (largest 0%)",
             expected="D grows or stays with delta after nucleation (delta > 0.4 nm); only during nucleation a slight "
                      "decrease is reported", references=["Gen19"], note="", notes=[]),
        dict(id="percolation", title="Percolation by the literature method against the published band",
             verdict="inconsistent",
             measured="delta = 0.35 nm at t = 1.75 s (68% interval 0.32-0.38 nm with the noise, the rate ±10% and "
                      "t0 ±2 s)",
             expected="percolation at delta = 4.7-13.5 nm for Ag/polymers (published values: 14, 7 of them by GISAXS; "
                      "each with its printed ± or ±10%)",
             references=["Gen19"], notes=["Earlier than every published value of this band."]),
        dict(id="network_two_R_over_D", title="Network 2R/D and coverage against the literature",
             verdict="not testable", measured="fewer than 5 island windows with network R and D at delta >= 1 nm",
             expected="", references=["Bulut24"], notes=[]),
    ],
    notes=["R scales as f^(-1/3) with the shape factor f = H/R (here f = 1): the literature reports flatter Ag on "
           "oxides and f = 2.75 for Cu on PS-b-PEO.",
           "delta = rate*(t - t0) with the given rate (0.2 nm/s), e.g. from a QCM; GISAXS does not measure it. The "
           "percolation interval allows the rate ±10% and t0 ±2 s."],
    references={"Gen19": dict(citation="Gensch M. et al., ACS Appl. Mater. Interfaces 11, 29416 (2019)",
                              doi="10.1021/acsami.9b08594")})
LITERATURE_SUMMARY = dict(PHYSICAL_SUMMARY, literature=LITERATURE)
NEEDS_RATE = dict(status="needs_rate", island_windows=0,
                  message="needs a deposition rate: the literature method computes R from the nominal thickness delta = "
                          "rate*(t - t0); without it only the monotony of D is tested (in time)",
                  checks=[dict(id="percolation", title="Percolation by the literature method against the published band",
                               verdict="not testable", measured="needs a deposition rate", expected="",
                               references=[], notes=[])], notes=[], references={})


def literature_folder(folder: Path, *, rate: bool = True) -> Path:
    """A result folder whose physical files carry the literature method (with a rate) or say it needs one."""
    write_table(folder, "structure", ROWS, SUMMARY)
    if rate:
        write_table(folder, "structure_physical", LITERATURE_ROWS, LITERATURE_SUMMARY)
    else:
        thickness = dict(PHYSICAL_SUMMARY["thickness"], rate_nm_s=None, t0_s=None)
        write_table(folder, "structure_physical", PHYSICAL_ROWS,
                    dict(PHYSICAL_SUMMARY, thickness=thickness, literature=NEEDS_RATE))
    return folder


BEAM_OFF_REASON = ("intensity 28% of the level within one modulation period (7 s) before and after (below 50%): the "
                   "beam is probably partly or fully off the sample (stage translation)")
BEAM_OFF_RECORD = dict(windows=1, evaluated=4, not_evaluated=0, rule="modulation", limit_fraction=0.5, span_s=7.0,
                       quantile=0.9, min_side_windows=3, yoneda_missing=0, yoneda_found_share=None, lowest_ratio=0.28,
                       method="invented method", meaning="invented meaning", scope="invented scope")


def beam_off_folder(folder: Path, *, flagged: bool = True) -> Path:
    """A physical result as the toolkit writes it after a stage translation: the window at 1.5 s dips (the beam may
    be off the sample), and the summary names the modulation by its neutral key (with the old one beside it)."""
    write_table(folder, "structure", ROWS, SUMMARY)
    rows = [dict(r, beam_off_sample="false", beam_off_reason="") for r in PHYSICAL_ROWS]
    if flagged:
        rows[2].update(beam_off_sample="true", beam_off_reason=BEAM_OFF_REASON)
    record = dict(BEAM_OFF_RECORD, windows=int(flagged))
    summary = dict(PHYSICAL_SUMMARY, modulation_period_s=7.0, modulation_detection=PHYSICAL_SUMMARY["rotation_detection"],
                   beam_off_sample=record, beam_off_sample_windows=int(flagged))
    write_table(folder, "structure_physical", rows, summary)
    return folder


def nominal_rows(rows: list, rate, t0, t_end) -> list:
    """``rows`` with the nominal thickness as the toolkit writes it, rate × clip(t − t0, 0, t_end − t0) (as they are
    without a rate)."""
    if rate is None:
        return rows
    start = t0 or 0.0
    stop = math.inf if t_end is None else t_end - start
    return [dict(r, nominal_thickness_nm=f"{round(rate * min(max(0.0, float(r['time_s']) - start), stop), 6):g}")
            for r in rows]


def scan_folder(root: Path, name: str = "zz_invented_scan_00001", frames: int = 3) -> Path:
    """A folder of invented mini-CBF frames (a text header and a few bytes; no real data)."""
    folder = root / name
    folder.mkdir(parents=True, exist_ok=True)
    for index in range(1, frames + 1):
        (folder / f"{name}_{index:05d}.cbf").write_bytes(cbf_header().encode("latin-1") + b"\x0c\x1a\x04\xd5\x00\x00")
    return folder


def cbf_header(*, serial: str = "99-0001", stamp: str = "2030-01-02T03:04:05.678", flat: str = "FF_x99_E9000_T4500.tif",
               extra: str = "") -> str:
    return ("###CBF: VERSION 1.5\n\ndata_invented\n\n_array_data.header_convention \"PILATUS_1.2\"\n"
            "_array_data.header_contents\n;\n"
            f"# Detector: PILATUS 1M, S/N {serial}, Invented\n# {stamp}\n# Pixel_size 172e-6 m x 172e-6 m\n"
            "# Silicon sensor, thickness 0.000450 m\n# Exposure_time 0.0300000 s\n# Exposure_period 0.0400000 s\n"
            "# Tau = 100.0e-09 s\n# Count_cutoff 123456 counts\n# Threshold_setting: 4500 eV\n"
            f"# Flat_field: {flat}\n# Trim_file: x99_E9000_T4500.bin\n{extra};\n\n_array_data.data\n;\n"
            "--CIF-BINARY-FORMAT-SECTION--\n")


class FakeInstaller:
    """PyTorch missing until ``install`` runs (or always ready / always failing)."""

    def __init__(self, *, ready: bool = False, failure: str = "", steps: int = 3):
        self.python = "invented-python"
        self.ready, self.failure, self.steps = ready, failure, steps
        self.installs = 0
        self.repairs = 0
        self.cut_off = False   # an earlier installation was cut off while pip wrote the packages
        self.broken = False    # PyTorch is there but does not load

    def manual_command(self, repair: bool = False) -> str:
        return '"invented-python" -m pip install ' + ("--force-reinstall --no-deps " if repair else "") +             '-r "requirements-sputter.txt"'

    def interrupted(self) -> bool:
        return self.cut_off

    def status(self) -> EnvironmentStatus:
        if self.broken:
            return EnvironmentStatus(self.python, None, problem="import_failed", detail="invented DLL failure")
        return EnvironmentStatus(self.python, "9.9.9" if self.ready else None, problem="" if self.ready else "missing")

    def verify(self) -> EnvironmentStatus:
        return self.status()

    def install(self, on_progress, should_stop, repair: bool = False) -> None:
        self.installs += 1
        self.repairs += bool(repair)
        on_progress(InstallProgress("resolve", None, (("package", "torch"),)))
        for step in range(self.steps):
            if should_stop():
                raise InstallError("cancelled", "Stopped before PyTorch was installed. Press Predict to start again.")
            on_progress(InstallProgress("download_progress", .05 + .8 * (step + 1) / self.steps,
                                        (("package", "torch"), ("percent", str(100 * (step + 1) // self.steps)))))
        if self.failure:
            from src.gimap.features.sputter.application import install_message
            raise InstallError(self.failure, install_message(self.failure, command=self.manual_command()),
                               "invented pip output line")
        on_progress(InstallProgress("install", None, cancellable=False))
        self.ready, self.broken, self.cut_off = True, False, False
        on_progress(InstallProgress("done", 1.0, cancellable=False))


class FakeRunner:
    """Writes an invented result (with the physical files) and reports progress like the toolkit."""

    def __init__(self, *, missing: str = "", wait_for_stop: bool = False, metal: str = "Ag", literature: bool = False):
        self.missing, self.wait_for_stop, self.metal = missing, wait_for_stop, metal
        self.literature = literature  # write the literature method as the toolkit does (with a rate)
        self.model = "invented-model-1"
        self.calls = []

    def problem(self) -> str:
        return self.missing

    def model_id(self) -> str:
        return self.model

    def analyze(self, request, out_folder, on_line, should_stop, environment=None):
        self.calls.append(("analyze", request, Path(out_folder)))
        on_line("SPUTTER_PROGRESS\tintake\t0.000\treading CBF frames")
        on_line("INTAKE_PLAN 2 4")
        on_line("INTAKE_CHUNK intake 0 2 0.1")
        on_line("INTAKE_EXCLUDED zz_invented_00003.cbf truncated_file")
        if self.wait_for_stop:
            import time
            deadline = time.monotonic() + 20
            while not should_stop() and time.monotonic() < deadline:
                time.sleep(0.02)
            raise Stopped("Stopped. Finished steps are kept in the results folder and are reused on the next run.")
        on_line("INTAKE_CHUNK intake 1 2 0.1")
        on_line("SPUTTER_PROGRESS\tnetwork\t1.000\tnetwork done")
        on_line("a plain line of the model")
        result_folder(Path(out_folder), physical=True)
        rate = request.rate_nm_s
        t0 = request.t0_s if request.t0_s is not None else (0. if rate is not None else None)
        thickness = dict(PHYSICAL_SUMMARY["thickness"], rate_nm_s=rate, t0_s=t0, t_end_s=request.t_end_s,
                         t_end_given=request.t_end_s is not None)
        rows = nominal_rows([dict(r, metal=self.metal) for r in PHYSICAL_ROWS], rate, t0, request.t_end_s)
        write_table(Path(out_folder), "structure_physical", rows, dict(PHYSICAL_SUMMARY, metal=self.metal,
                                                                        thickness=thickness))
        record = {"scan_folder": str(request.scan_folder), "request": {"scan_folder": str(request.scan_folder)}}
        (Path(out_folder) / "gimap_request.json").write_text(json.dumps(record), encoding="utf-8")
        on_line("SPUTTER_PROGRESS\tdone\t1.000\tdone")
        return Path(out_folder) / "structure.csv"

    def table(self, out_folder, metal, refined_folder, on_line):
        self.calls.append(("table", Path(out_folder), metal, refined_folder))
        return Path(out_folder) / "structure.csv"

    def physical(self, out_folder, rate_nm_s, t0_s, on_line, *, t_end_s=None):
        self.calls.append(("physical", Path(out_folder), rate_nm_s, t0_s, t_end_s))
        t0 = t0_s if t0_s is not None else (0. if rate_nm_s is not None else None)  # as the toolkit writes it
        summary = dict(PHYSICAL_SUMMARY, thickness=dict(PHYSICAL_SUMMARY["thickness"], rate_nm_s=rate_nm_s, t0_s=t0,
                                                        t_end_s=t_end_s, t_end_given=t_end_s is not None))
        rows = PHYSICAL_ROWS
        if self.literature:
            summary["literature"] = (dict(LITERATURE, rate_nm_s=rate_nm_s, t_end_s=t_end_s) if rate_nm_s is not None
                                     else NEEDS_RATE)
            rows = LITERATURE_ROWS if rate_nm_s is not None else PHYSICAL_ROWS
        write_table(Path(out_folder), "structure_physical", nominal_rows(rows, rate_nm_s, t0, t_end_s), summary)
        return Path(out_folder) / "structure_physical.csv"


class FakeInspector:
    def __init__(self, **values):
        self.values = values
        self.calls = []

    def inspect(self, folder):
        self.calls.append(Path(folder))
        values = dict(dict(frames=3, detector="PILATUS 1M", serial="99-0001", pixel_um=172.0, exposure_s=0.03,
                           period_s=0.04, timestamp="2030-01-02T03:04:05", flat_field="FF_x99_E9000_T4500.tif",
                           energy_kev_hint=9.0, energy_hint_source="header"), **self.values)
        return ScanInfo(Path(folder), **values)
