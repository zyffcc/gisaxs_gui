"""Analyze ▸ Series: the stages and odd frames of a built map, their export, Batch Export without the
odd frames, and Send to Compare."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tests.test_assistant_calibration import CENTER, DISTANCE_M, PIXEL, SHAPE, WAVELENGTH, save_tiff
from tests.test_series_map import _ring_frame, _wait
from tests.test_analyze_workspace import _app, _context


def _frames(tmp_path: Path) -> list[Path]:
    """24 frames: the ring at 1.1 grows (in log) until frame 12, then a ring at 0.7 grows too; frame 7 is odd."""
    paths = []
    for index in range(24):
        height = 40.0 * 10 ** (min(index, 12) / 6.0)
        image = _ring_frame(height, seed=index)
        if index >= 12:
            image += _ring_frame(40.0 * 10 ** ((index - 12) / 4.0), q0=0.7, seed=100 + index) - _ring_frame(0.0, seed=200 + index)
        if index == 6:
            image += _ring_frame(3000.0, q0=0.5, seed=300) - _ring_frame(0.0, seed=301)
        paths.append(save_tiff(tmp_path / f"run_{index:03d}.tif", np.maximum(image, 0)))
    return paths


def test_a_built_map_shows_its_stages_odd_frames_and_what_changes(tmp_path: Path) -> None:
    from src.gimap.features.analyze.bootstrap import create_analyze_view_model
    from src.gimap.features.analyze.presentation.page import AnalyzePage
    from src.gimap.integrations.state import InMemoryInstrumentProfileRepository
    from src.gimap.shared.geometry import DetectorGeometry, InstrumentProfile

    _app()
    paths = _frames(tmp_path)
    profile = InstrumentProfile("synthetic", DetectorGeometry(PIXEL, PIXEL, DISTANCE_M, CENTER[0], CENTER[1], WAVELENGTH, 0.2),
                                None, SHAPE)
    page = AnalyzePage(create_analyze_view_model(_context(InMemoryInstrumentProfileRepository([profile]))))
    sent = []
    page.set_compare_target(lambda series, name: sent.append((series, name)))
    try:
        page.set_mode_choice("giwaxs")
        page.add_paths([str(path) for path in paths])
        page.tasks.wait(60)
        _app().processEvents()
        assert page.series_stages_row.isHidden() and page.series_compare_button.isHidden()
        page.series_build_button.click()
        _wait(page, lambda: page._series_stages is not None)
        stages = page._series_stages
        assert {frame.row for frame in stages.odd} == {6}
        assert stages.suggested >= 2 and abs(stages.boundaries[stages.suggested][-1] - 12) <= 2
        assert not page.series_stages_row.isHidden() and not page.series_stages_details.isHidden()
        assert page.series_stages_label.text().startswith(f"{stages.suggested} stages: frames 1–")
        assert "odd frame" in page.series_stages_label.text()
        details = page.series_changes_label.text()
        assert "Odd frame 7 (run_006.tif)" in details and "grows most at q 0.7" in details
        assert page._stage_overlay.shown and "stages" in page.series_map_view.marks.present()
        # One stage chosen by hand, then Auto again.
        page.series_stages_combo.setCurrentIndex(1)
        page.series_stages_combo.activated.emit(1)
        assert page._series_stages.count == 1 and page.series_stages_label.text().startswith("One stage")
        page.series_stages_combo.activated.emit(0)
        assert page._series_stages.count == stages.suggested
        # The trace: change along the series, one colour per stage, the odd frame marked.
        page.series_trace_combo.setCurrentIndex(page.series_trace_combo.findData("change"))
        names = [curve[0] for curve in page.series_trace_plot.figure_state()["curves"]]
        assert names[0] == "stage 1" and names[-1] == "odd frames"
        # Export: a row per frame and the record.
        written = page.export_series_stages(tmp_path / "stages.csv")
        rows = [line for line in written.read_text(encoding="utf-8").splitlines() if not line.startswith("#")]
        assert rows[0].startswith("frame,file,stage,odd,why") and len(rows) == 25
        record = json.loads(written.with_suffix(".json").read_text(encoding="utf-8"))
        assert record["odd_frames"][0]["frame"] == 7 and record["stages_suggested"] == stages.suggested
        # Batch Export can leave the odd frame out.
        requests = page.view_model.batch_requests()
        assert len(page._without_odd_frames(requests)) == 24
        page.series_skip_odd_check.setChecked(True)
        kept = page._without_odd_frames(requests)
        assert len(kept) == 23 and all(Path(request.path).name != "run_006.tif" for request in kept)
        # Send to Compare.
        assert not page.series_compare_button.isHidden()
        page.series_compare_button.click()
        assert sent and sent[0][0] is page._series_map and sent[0][1] == "run"
    finally:
        page.tasks.wait(30)
        page.dispose()
