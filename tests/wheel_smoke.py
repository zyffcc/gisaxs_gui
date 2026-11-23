"""Run inside the fresh wheel venv created by tools/check_wheel.py."""

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import psutil

from src.gimap.app.help_commands import CITATION_FILE
from src.gimap.app.presentation import assets, help_documents
from src.gimap.features.analyze.application import AnalysisRequest, AnalyzeFrame, ExportAnalysis
from src.gimap.features.analyze.domain import GisaxsCutSettings
from src.gimap.features.analyze.infrastructure.adapters import CsvCurveWriter, DetectorIoFrameSource
from src.gimap.features.calibration.infrastructure.adapters.local import JsonDetectorCatalogAdapter
from src.gimap.shared.resources import RESOURCE_ROOT
from src.gimap.shared.revision import source_commit


def main():
    work = Path.cwd()
    assert RESOURCE_ROOT.is_relative_to(Path(sys.prefix))
    assert re.fullmatch(r"[0-9a-f]{40,64}", source_commit() or "")
    assert JsonDetectorCatalogAdapter().load()["Pilatus 1M"]["nbins_x"] == 981
    assert assets.APP_ICON_PATH.is_file() and CITATION_FILE.is_file() and help_documents.THIRD_PARTY_NOTICES.is_file()
    for manual in (help_documents.USER_MANUAL, help_documents.USER_MANUAL_ZH,
                   help_documents.PHYSICS_MANUAL, help_documents.PHYSICS_MANUAL_ZH):
        markdown = manual.read_text(encoding="utf-8")
        for image in re.findall(r"!\[[^\]]*\]\((manual_images/[^)]+)\)", markdown):
            assert (manual.parent / image).is_file(), image

    frame = work / "sample/galaxi_data.tif"
    analysis = AnalyzeFrame(DetectorIoFrameSource(), None)(AnalysisRequest(
        path=frame, mode="gisaxs", incidence_deg=0.463, with_map=False,
        gisaxs=GisaxsCutSettings(horizontal_row=607.5),
    ))
    files = ExportAnalysis(CsvCurveWriter()).curves(analysis, ["horizontal"], work / "export")
    record = next(path for path in files if path.suffix == ".json")
    entry = Path(sys.executable).with_name("gimap.exe" if os.name == "nt" else "gimap")
    with subprocess.Popen([entry, "reproduce", record, "--out", work / "replay"]) as child:
        process = psutil.Process(child.pid)
        peak_rss = 0
        while child.poll() is None:
            try:
                peak_rss = max(peak_rss, sum(p.memory_info().rss for p in [process, *process.children(recursive=True)]))
            except psutil.NoSuchProcess:
                pass
            try:
                child.wait(timeout=0.02)
            except subprocess.TimeoutExpired:
                pass
        assert child.returncode == 0
    comparison = json.loads((work / "replay/comparison.json").read_text(encoding="utf-8"))
    assert comparison["source_record_sha256"] == hashlib.sha256(record.read_bytes()).hexdigest()
    assert comparison["status"] == "identical"
    columns = comparison["tables"]["horizontal"]["columns"]
    assert {"x", "intensity", "sigma", "pixels"} <= columns.keys()
    assert all(item["status"] == "identical" for item in columns.values())
    print(f"Installed gimap reproduced GALAXI, peak sampled process-tree RSS {peak_rss / 1024**2:.1f} MiB")


if __name__ == "__main__":
    main()
