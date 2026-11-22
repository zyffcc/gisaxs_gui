"""The flat field in the Analyze window and on the command line.

- Mask & corrections ▸ Corrections takes a flat field (Choose… / ×), refuses one of another size with the reason,
  says what it did to the frame shown (its mean, the pixels it leaves out), in English and in Chinese; the Batch
  Export summary names it; a settings file whose flat field is missing says so and leaves it out;
- ``tools/gimap_agent.py auto --flat-field PATH`` divides every frame by it: a missing file is an argument error
  (exit code 1, nothing analysed), one of another size fails that frame, and the report names it among the
  Decisions (file, pixels left out) and in its inputs (SHA-256), GIWAXS and GISAXS alike.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np

from src.gimap.app.presentation.i18n import ZH, apply_language
from src.gimap.features.analyze.application import AnalyzeSettings, Corrections
from src.gimap.features.analyze.infrastructure.adapters import DetectorIoFrameSource
from src.gimap.features.assistant.application import PipelineOptions
from src.gimap.app.presentation.i18n.zh_flat_field import FLAT_FIELD_ZH
from tests.test_analyze_workspace import _context, _done, _page
from tests.test_assistant_calibration import SHAPE
from tests.test_review_analyze_run_and_clear import _frames, _profiles

DATA = Path(__file__).parent / "data" / "external"
XEUSS = DATA / "giwaxs_xeuss_pbte"
GALAXI = DATA / "gisaxs_galaxi"
TIPS = ("Choose a uniformly exposed frame D_ff. Each sample is divided by its normalised response F_ff.",
        "Stop dividing by the flat field")


def _close(page) -> None:
    page.tasks.wait(30)
    page.dispose()
    page.close()


def _flat_with_dead_block(shape, rows: int = 10, columns: int = 10) -> np.ndarray:
    flat = np.full(shape, 5000.0)
    flat[:rows, :columns] = 0.0
    return flat


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# -- the window ----------------------------------------------------------------------------------------------------


def test_the_corrections_take_a_flat_field_and_say_what_it_did(tmp_path: Path) -> None:
    apply_language("en")
    paths = _frames(tmp_path, 1)
    page = _page(_context(_profiles()))
    try:
        page.add_paths([str(path) for path in paths])
        _done(page)
        assert page.flat_field_label.text() == "No flat field" and page.flat_field_status.isHidden()
        assert (page.flat_field_button.toolTip(), page.flat_field_clear_button.toolTip()) == TIPS
        small = tmp_path / "small.npy"
        np.save(small, np.ones((10, 12)))
        assert page.load_flat_field(small) is False
        assert page.status_text() == (
            f"Flat field not loaded: The flat field small.npy is 10×12; the frame is {SHAPE[0]}×{SHAPE[1]}.")
        assert page.view_model.state.corrections.flat_field_path is None
        flat = tmp_path / "flat.npy"
        np.save(flat, _flat_with_dead_block(SHAPE, 2, 3))
        said = []
        status = page._status
        page._status = lambda text, level="info", *args, **kwargs: said.append(text) or status(text, level, *args, **kwargs)
        assert page.load_flat_field(flat) is True
        assert "Flat field flat.npy: 6 pixels left out" in said  # then the analysis it starts says its own
        del page._status
        _done(page)
        analysis = page.view_model.state.analysis
        assert analysis.flat_field is not None and analysis.flat_field.masked_pixels == 6
        assert not analysis.valid[0, 0] and page.flat_field_label.text() == "flat.npy"
        assert not page.flat_field_status.isHidden()
        assert page.flat_field_status.text() == "Normalised by its mean 5000; 6 pixels at or below 0.05 left out"
        assert "flat field flat.npy" in page.settings_summary()
        apply_language("zh", [page])
        page.refresh_language()
        assert page.flat_field_status.text() == "已用其平均值 5000 归一化；排除了 6 个不大于 0.05 的像素"
        assert (page.flat_field_button.toolTip(), page.flat_field_clear_button.toolTip()) == tuple(
            ZH[tip] for tip in TIPS)
        page._clear_flat_field()
        assert page.flat_field_label.text() == "无平场" and page.flat_field_status.isHidden()
        assert page.view_model.state.corrections.flat_field_path is None
        apply_language("en", [page])
        assert page.flat_field_button.toolTip() == TIPS[0]
        # A set-up whose flat field is not there any more: said, and left out.
        notes = page.view_model.apply_settings(AnalyzeSettings(corrections=Corrections(
            flat_field_path=str(tmp_path / "gone.npy"))))
        assert notes == [f"Flat-field file not found, left out: {tmp_path / 'gone.npy'}"]
        assert page.view_model.state.corrections.flat_field_path is None
    finally:
        apply_language("en")
        _close(page)


def test_every_text_of_the_flat_field_has_its_chinese() -> None:
    sources = [Path(__file__).resolve().parents[1] / "src/gimap/features/analyze" / name for name in (
        "presentation/views/options_panel_view.py", "presentation/bindings/options.py",
        "presentation/bindings/batch_export.py", "presentation/batch_model.py", "presentation/texts.py",
        "application/flat_field.py", "domain/flat_field.py")]
    text = "\n".join(path.read_text(encoding="utf-8") for path in sources)
    for english, chinese in FLAT_FIELD_ZH.items():
        assert ZH[english] == chinese, english
        if english not in ("Flat-field file not found, left out: {path}",):  # (the note is composed from its label)
            assert english.split("{")[0][:30] in text, english


# -- the command line ----------------------------------------------------------------------------------------------


def test_the_command_takes_a_flat_field_and_refuses_one_that_is_not_there(monkeypatch, tmp_path: Path, capsys) -> None:
    from src.gimap.app import headless_assistant
    from tools import gimap_agent

    frame = tmp_path / "film_1.tif"
    frame.write_bytes(b"")
    flat = tmp_path / "flat.npy"
    np.save(flat, np.ones((4, 4)))
    seen: dict = {}

    def analyse_frames(frames, options, out, **kwargs):
        seen.update(kwargs, frames=list(frames))
        return [{"ok": True, "frame": item, "procedure": "giwaxs", "needs_attention": [], "decisions": [], "steps": []}
                for item in frames]

    monkeypatch.setattr(headless_assistant, "analyse_frames", analyse_frames)
    out = tmp_path / "out"
    assert gimap_agent.main(["auto", str(tmp_path), "--flat-field", str(flat), "--quiet", "--out", str(out)]) == 0
    assert seen["flat_field"] == str(flat) and [Path(item).name for item in seen["frames"]] == ["film_1.tif"]
    assert seen["record"]["flat_field"] == str(flat)  # the options as given, for the record
    seen.clear()
    assert gimap_agent.main(["auto", str(frame), "--flat-field", str(tmp_path / "no_flat.npy"), "--out", str(out)]) == 1
    assert "No such flat-field file" in capsys.readouterr().err and not seen
    text = " ".join(gimap_agent.parser()._subparsers._group_actions[0].choices["auto"].format_help().split())
    assert "--flat-field PATH" in text and "normalised to mean 1" in text


def test_a_run_with_a_flat_field_names_it_in_the_decisions_and_the_inputs(tmp_path: Path) -> None:
    from src.gimap.app.headless_assistant import analyse_frames

    frame = XEUSS / "PbTe_3010_r1_PT_1_10_0_00028.edf"
    shape = DetectorIoFrameSource().load(frame).data.shape
    flat = tmp_path / "flat.npy"
    np.save(flat, _flat_with_dead_block(shape))
    options = PipelineOptions(calibration=str(XEUSS / "AgBe_250403_XENOCS.poni"), incidence_deg=0.2)
    before = sorted(path.name for path in XEUSS.iterdir())
    [report] = analyse_frames([str(frame)], options, tmp_path / "giwaxs", saved_profiles=False, flat_field=str(flat))
    assert report["ok"]
    [decision] = [item for item in report["decisions"] if item["what"] == "flat field"]
    assert decision["decision"] == "divided by flat.npy (100 pixels left out)"
    assert report["inputs"]["flat_field"] == {"file": "flat.npy", "path": str(flat), "bytes": flat.stat().st_size,
                                              "sha256": _sha256(flat)}
    record = report["analysis_record"]["corrections"]["flat_field"]
    assert record["masked_pixels"] == 100 and record["sha256"] == _sha256(flat)
    assert record["normalisation_mean"] == 5000.0 and report["analysis_record"]["error_model"] in (
        "scaled_poisson", "pixel_scatter")
    text = (tmp_path / "giwaxs" / frame.stem / "report.md").read_text(encoding="utf-8")
    assert "- **flat field**: divided by flat.npy (100 pixels left out)" in text
    assert f"- flat-field file: `{flat}` (SHA-256 {_sha256(flat)}" in text
    assert "divided by the flat field `flat.npy` (100 pixels left out)" in text
    assert sorted(path.name for path in XEUSS.iterdir()) == before  # nothing written next to the data
    # A flat field of another detector: that frame fails with the reason, nothing is analysed with it.
    [refused] = analyse_frames([str(GALAXI / "galaxi_data.tif")], PipelineOptions(), tmp_path / "other",
                               saved_profiles=False, flat_field=str(flat))
    assert not refused["ok"] and "The flat field flat.npy is" in refused["error"] and "the frame is" in refused["error"]
    # GISAXS: the same decision.
    galaxi = GALAXI / "galaxi_data.tif"
    np.save(flat, _flat_with_dead_block(DetectorIoFrameSource().load(galaxi).data.shape, 4, 5))
    options = PipelineOptions(calibration=str(GALAXI / "galaxi_bornagain.poni"), incidence_deg=0.2, fit=False)
    [gisaxs] = analyse_frames([str(galaxi)], options, tmp_path / "gisaxs", saved_profiles=False, flat_field=str(flat))
    assert gisaxs["procedure"] == "gisaxs"
    assert {"what": "flat field", "decision": "divided by flat.npy (20 pixels left out)"}.items() <= next(
        item for item in gisaxs["decisions"] if item["what"] == "flat field").items()
