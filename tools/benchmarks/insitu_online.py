"""Time and memory of Analyze's online mode on a watched folder: frames written in bursts, each reduced, appended to
the Series map (no Build Map, no export) and recorded in the results database; the timeline and the stages of the
grown map. Public data only: crops of the GALAXI frame (tests/data/external) with a moving side peak and counting
noise, written into a temporary folder.

    python tools/benchmarks/insitu_online.py --frames 1000 --burst 50

Prints one JSON object: frames, seconds per frame (reduction and append, at the end), peak resident memory and its
growth per 100 frames, map rows and bytes, database size, timeline and stage times. Runs offscreen with a fresh
GIMAP_HOME; nothing is written outside the temporary folder.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GALAXI = ROOT / "tests" / "data" / "external" / "gisaxs_galaxi" / "galaxi_data.tif"
CROP = (560, 497, 100, 200)
"""First row, first column, rows and columns of the crop: the Yoneda band and the beam column."""


def _environment(home: Path) -> None:
    home.mkdir(parents=True, exist_ok=True)
    (home / "settings.json").write_text('{"schema_version": 1, "settings": {}, "preferences": {}}', encoding="utf-8")
    (home / "instrument_profiles.json").write_text('{"schema_version": 1, "profiles": []}', encoding="utf-8")
    os.environ["GIMAP_HOME"] = str(home)
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    sys.path.insert(0, str(ROOT))


def _frames(folder: Path, first: int, count: int, crop) -> None:
    import numpy as np
    from PIL import Image

    from src.gimap.shared.detector_io import load_detector_image

    row, column, rows, columns = crop
    base = np.asarray(load_detector_image(str(GALAXI)).data, dtype=np.float64)[row:row + rows, column:column + columns]
    r, c = np.indices(base.shape)
    for index in range(first, first + count):
        offset = 30.0 + 0.02 * index + (25.0 if index >= 600 else 0.0)
        peak = sum(400.0 * np.exp(-0.5 * ((c - (597.1 - column + sign * offset)) / 3.0) ** 2) for sign in (-1, 1))
        level = np.clip(base, 0, None) * (1.0 + 1e-4 * index) + peak * np.exp(-0.5 * ((r - (605.0 - row)) / 6.0) ** 2)
        counts = np.random.default_rng(index).poisson(level).astype(np.int32)
        counts[base < 0] = -1
        part = folder / f".galaxi_{index:05d}.tif.part"
        Image.fromarray(counts, mode="I").save(part, format="TIFF")
        part.replace(folder / f"galaxi_{index:05d}.tif")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--frames", type=int, default=1000)
    parser.add_argument("--burst", type=int, default=50, help="frames written between two looks at the folder")
    parser.add_argument("--no-record", action="store_true", help="do not record in the results database")
    args = parser.parse_args(argv)
    work = Path(tempfile.mkdtemp(prefix="gimap_online_"))
    _environment(work / "home")

    import numpy as np
    import psutil
    from PyQt5.QtWidgets import QApplication

    from src.gimap.app import AppContext
    from src.gimap.features.analyze.bootstrap import create_analyze_view_model
    from src.gimap.features.analyze.presentation.page import AnalyzePage
    from src.gimap.integrations.state import (
        InMemoryInstrumentProfileRepository,
        InMemorySessionRepository,
        InMemorySettingsRepository,
        InMemoryUserPreferencesRepository,
    )
    from src.gimap.shared.geometry import DetectorGeometry, InstrumentProfile

    app = QApplication.instance() or QApplication([])
    row, column, rows, columns = CROP
    profile = InstrumentProfile("GALAXI crop", DetectorGeometry(172e-6, 172e-6, 1.73, 597.1 - column, 719.6 - row,
                                                                1.34, 0.463), None, (rows, columns))
    context = AppContext(settings=InMemorySettingsRepository({}), session=InMemorySessionRepository(),
                         preferences=InMemoryUserPreferencesRepository(),
                         instrument_profiles=InMemoryInstrumentProfileRepository([profile]))
    page = AnalyzePage(create_analyze_view_model(context))
    folder = work / "run"
    folder.mkdir()
    _frames(folder, 0, 1, CROP)

    def settle() -> None:
        page.tasks.wait(600)
        app.processEvents()

    page.set_mode_choice("gisaxs")
    page.add_paths([str(folder)])
    settle()
    page.auto_export_check.setChecked(False)
    page.watch_record_action.setChecked(not args.no_record)
    page.start_watch(folder)
    page._watch_timer.stop()
    process = psutil.Process()
    start_rss = process.memory_info().rss
    samples, written, busy = [], 1, 0.0
    began = time.perf_counter()
    while written < args.frames + 1:
        count = min(args.burst, args.frames + 1 - written)
        _frames(folder, written, count, CROP)
        written += count
        tick = time.perf_counter()
        for _look in range(2):  # a new file is taken once it is the same at two looks
            page.poll_watch()
            settle()
        page._watch_series_redraw()  # (the window does this once a second)
        settle()
        busy += time.perf_counter() - tick
        samples.append((len(page._series_rows), process.memory_info().rss))
    elapsed = time.perf_counter() - began
    series = page._series_map
    tick = time.perf_counter()
    page.series_trace_combo.setCurrentIndex(page.series_trace_combo.findData("neighbour"))
    timeline = page._series_timeline()
    timeline_s = time.perf_counter() - tick
    from src.gimap.features.analyze.application import stages_of

    tick = time.perf_counter()
    stages = stages_of(series)
    stages_s = time.perf_counter() - tick
    database = folder / "gimap_analysis" / "gimap_series.sqlite"
    rows_bytes = sum(np.asarray(item[0]).nbytes + np.asarray(item[1]).nbytes for item in page._series_rows)
    growth = [(b - a) / max(1, (n2 - n1)) * 100 for (n1, a), (n2, b) in zip(samples, samples[1:]) if n2 > n1]
    report = {
        "frames": len(page._series_rows), "frame_pixels": rows * columns, "burst": args.burst,
        "seconds_total": round(elapsed, 2), "seconds_per_frame": round(busy / max(1, len(page._series_rows)), 4),
        "rss_start_mb": round(start_rss / 2**20, 1), "rss_end_mb": round(samples[-1][1] / 2**20, 1),
        "rss_growth_mb_per_100_frames_last_half": round(float(np.median(growth[len(growth) // 2:])) / 2**20, 2)
        if growth else None,
        "map_rows": series.rows, "map_points": int(series.x.size), "map_mb": round(series.image.nbytes / 2**20, 2),
        "rows_mb": round(rows_bytes / 2**20, 2),
        "database_mb": round(database.stat().st_size / 2**20, 2) if database.is_file() else None,
        "timeline_s": round(timeline_s, 3), "jumps": [series.number(row) for row in timeline.jumps],
        "stages_s": round(stages_s, 3), "stages": stages.count, "skipped": len(page.skipped_files()),
    }
    page.stop_watch()
    page.dispose()
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
