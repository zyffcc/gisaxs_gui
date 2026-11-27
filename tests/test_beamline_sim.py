"""The simulated ALS 7.3.3 spray run (``tools/beamline_sim.py``): frames GIMaP reads as it reads 7.3.3 frames, written
into a folder at the beamline's pace."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from src.gimap.shared.detector_io import load_detector_image
from src.gimap.shared.detector_io.module_gaps import layout_gaps
from tools.beamline_sim import main


def test_a_rendered_run_replays_into_a_watched_folder_as_7_3_3_frames(tmp_path: Path) -> None:
    staged, live = tmp_path / "staged", tmp_path / "live"
    main(["render", "--out", str(staged), "--detector", "Pilatus 300K", "--pulses", "1", "--fast-seconds", "0.3",
          "--period", "6.0", "--distance", "2.0"])
    manifest = json.loads((staged / "manifest.json").read_text(encoding="utf-8"))
    frames = manifest["frames"]
    assert manifest["shape"] == [619, 487] and len(frames) == len(list(staged.glob("*.edf")))
    assert {frame["phase"] for frame in frames} >= {"reference", "spray", "drying-fast", "angle"}
    ends = [frame["start_s"] + frame["count_time_s"] for frame in frames]
    assert ends == sorted(ends)  # a frame ends before the next one does
    main(["replay", "--src", str(staged), "--dst", str(live), "--speed", "40"])
    assert sorted(path.name for path in live.glob("*.edf")) == [frame["file"] for frame in frames]
    with open(live / "spray_log.csv", newline="", encoding="utf-8") as handle:
        assert [row["pulse"] for row in csv.DictReader(handle)] == ["1"]
    angle = next(frame for frame in frames if frame["phase"] == "angle")
    image = load_detector_image(live / angle["file"])
    assert image.metadata["incidence_deg"] == pytest.approx(angle["alpha_deg"])
    assert image.metadata["side_monitors"]["Izero"] > 0 and image.metadata["timestamp"]
    _name, gaps = layout_gaps((619, 487))
    assert np.array_equal(image.mask & gaps, gaps)  # every module gap left out
    staged_date = (staged / angle["file"]).read_bytes()[:1024]
    assert (live / angle["file"]).read_bytes()[:1024] != staged_date  # stamped when it was written
