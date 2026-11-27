"""What a long run costs the window when a watched folder already holds N frames: listing them, starting the watch,
a look that finds ten new frames (the window's part of it, after the look at the folder on the watch's own thread),
the watch's applying of a look to N files listed (in memory: with the look's marks, and looking at every file as
before them), a redraw or zoom step of a Series map of N rows zoomed in on its last 1 000, and a change of a row of
2D Prediction's batch results when they hold N rows. Public data only: eight crops of the GALAXI frame (tests/data/external), linked
N times into a temporary folder (hard links: no disk space), and synthetic map pictures.

    python tools/benchmarks/live_scale.py --frames 50000 --json live_scale.json

Prints one JSON object (seconds). Runs offscreen with a fresh GIMAP_HOME; nothing is written outside the temporary
folder. The numbers depend on the computer and on what else runs on it.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.benchmarks.insitu_online import CROP, _environment, _frames  # noqa: E402

POOL = 8
"""Different frames linked over and over (their contents do not matter here, only their number)."""
LINKS = 1000
"""Links of one frame at most (NTFS takes 1 024 names of a file)."""


def _link(pool: list[Path], folder: Path, first: int, count: int) -> None:
    for index in range(first, first + count):
        os.link(pool[index % len(pool)], folder / f"galaxi_{index:06d}.tif")


def _looks(page, folder: Path, pool: list[Path], first: int, looks: int, settle) -> list[float]:
    """The window's part of looks that each find ten new frames (the scan made beforehand, as on the watch's thread)."""
    times = []
    for look in range(looks):
        _link(pool, folder, first + 10 * look, 10)
        for _again in range(2):  # (a new file is taken once it is the same at two looks)
            scan = page.view_model.watch.scan()
            began = time.perf_counter()
            page._watch_found(page.view_model.take_watch(scan))
            times.append(time.perf_counter() - began)
            settle()
    return times


def _map_zoom(rows: int) -> dict:
    """One redraw of a picture of ``rows`` × 1 000 zoomed in on its last 1 000 rows: given whole, and cut to the view."""
    import numpy as np
    from PyQt5.QtWidgets import QApplication

    from src.gimap.app.presentation.components import image_clip
    from src.gimap.app.presentation.components.detector_view import DetectorView

    app = QApplication.instance()
    picture = np.random.default_rng(1).random((rows, 1000), dtype=np.float32) * 100
    picture[:, 495:505] = np.nan  # (blank columns, as a map of a cut has where no frame has a point)
    result, limit = {}, image_clip.CLIP_CELLS
    for name, cells in (("whole", 10**12), ("cut", limit)):
        image_clip.CLIP_CELLS = cells
        view = DetectorView()
        view.resize(1100, 800)
        view.set_aspect_locked(False)
        view.show()
        view.set_image(picture, rect=(-0.12, 0.0, 0.24, float(rows)), y_down=False)
        view.view_box.setRange(xRange=(-0.12, 0.12), yRange=(rows - 1000, rows), padding=0)
        app.processEvents()
        view._clip.follow()
        item = view.image_item
        times = []
        for _ in range(3):
            item._renderRequired, item.qimage = True, None
            began = time.perf_counter()
            item.render()
            times.append(time.perf_counter() - began)
        result[name] = round(min(times), 4)
        view.dispose()
        view.deleteLater()
        app.processEvents()
    image_clip.CLIP_CELLS = limit
    return result


def _take(files: int) -> dict:
    """``FolderWatch.take`` of looks that each find ten new files among ``files`` listed (a listing in memory):
    the look's marks used, and every file looked at (``look=None``, as before the marks)."""
    import dataclasses

    from src.gimap.features.analyze.presentation.folder_watch import FolderWatch

    with tempfile.TemporaryDirectory(prefix="gimap_live_take_") as temp:
        folder = Path(temp)
        sizes = {f"f_{index:06d}.edf": 1000 for index in range(files)}
        paths: dict[str, Path] = {}
        watch = FolderWatch(lambda _folders: [paths.setdefault(name, folder / name) for name in sorted(sizes)],
                            lambda path: sizes[path.name], lambda path: 7)
        watch.start(folder)
        times = {"marked": [], "every": []}
        for look in range(12):
            sizes.update({f"g_{look:03d}_{index}.edf": 1000 for index in range(10)})
            for _again in range(2):  # (taken once it is the same at two looks)
                scan, kind = watch.scan(), ("marked", "every")[look % 2]
                scan = scan if kind == "marked" else dataclasses.replace(scan, look=None)
                began = time.perf_counter()
                watch.take(scan)
                times[kind].append(time.perf_counter() - began)
    return {kind: round(statistics.median(values), 5) for kind, values in times.items()}


def _prediction_rows(rows: int) -> dict:
    from src.gimap.features.prediction.presentation.result_types import PredictStatus
    from src.gimap.features.prediction.presentation.results_widget import MultiFilePredictResultsWidget

    widget = MultiFilePredictResultsWidget()
    began = time.perf_counter()
    for index in range(rows):
        widget.addPredictResult(f"frame_{index:06d}.edf")
    listed = time.perf_counter() - began
    began = time.perf_counter()
    for index in range(rows - 20, rows):
        widget.updatePredictResult(index, status=PredictStatus.RUNNING)
        widget.updatePredictResult(index, status=PredictStatus.COMPLETED)
    update = (time.perf_counter() - began) / 40
    widget.close()
    return {"list_s": round(listed, 2), "update_s": round(update, 5)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Time the window's work of a long watched run at N frames.")
    parser.add_argument("--frames", type=int, default=20000, help="frames in the folder before the watch starts")
    parser.add_argument("--looks", type=int, default=6)
    parser.add_argument("--json", type=Path, help="also write the result to this JSON file")
    args = parser.parse_args(argv)
    if args.frames < 1:
        parser.error("--frames must be positive")
    with tempfile.TemporaryDirectory(prefix="gimap_live_scale_") as temp:
        work = Path(temp)
        _environment(work / "home")
        from PyQt5.QtWidgets import QApplication

        from src.gimap.app import AppContext
        from src.gimap.features.analyze.bootstrap import create_analyze_view_model
        from src.gimap.features.analyze.presentation.page import AnalyzePage
        from src.gimap.integrations.state import (
            InMemoryInstrumentProfileRepository, InMemorySessionRepository, InMemorySettingsRepository,
            InMemoryUserPreferencesRepository,
        )
        from src.gimap.shared.geometry import DetectorGeometry, InstrumentProfile

        app = QApplication.instance() or QApplication([])
        source = work / "pool"
        source.mkdir()
        _frames(source, 0, max(POOL, (args.frames + 10 * args.looks) // LINKS + 1), CROP)
        pool = sorted(source.glob("*.tif"))
        folder = work / "run"
        folder.mkdir()
        _link(pool, folder, 0, args.frames)
        row, column, rows, columns = CROP
        profile = InstrumentProfile("GALAXI crop", DetectorGeometry(172e-6, 172e-6, 1.73, 597.1 - column, 719.6 - row,
                                                                    1.34, 0.463), None, (rows, columns))
        context = AppContext(settings=InMemorySettingsRepository({}), session=InMemorySessionRepository(),
                             preferences=InMemoryUserPreferencesRepository(),
                             instrument_profiles=InMemoryInstrumentProfileRepository([profile]))
        page = AnalyzePage(create_analyze_view_model(context))
        page.resize(1500, 1000)
        page.show()

        def settle() -> None:
            page.tasks.wait(600)
            app.processEvents()

        page.set_mode_choice("gisaxs")
        began = time.perf_counter()
        page.add_paths([str(folder)])
        settle()
        listed = time.perf_counter() - began
        page.auto_export_check.setChecked(False)
        began = time.perf_counter()
        page.start_watch(folder)
        page._watch_timer.stop()
        started = time.perf_counter() - began
        looks = _looks(page, folder, pool, args.frames, args.looks, settle)
        page.stop_watch()
        page.dispose()
        report = {
            "frames": args.frames, "list_s": round(listed, 2), "start_watch_s": round(started, 2),
            "first_look_s": round(max(looks[:2]), 3), "look_s_median": round(statistics.median(looks[2:] or looks), 3),
            "take_s": _take(args.frames), "map_zoomed_redraw_s": _map_zoom(args.frames),
            "prediction_rows": _prediction_rows(args.frames),
        }
    text = json.dumps(report, indent=2)
    print(text)
    if args.json:
        args.json.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
