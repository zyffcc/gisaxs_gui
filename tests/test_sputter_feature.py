"""Sputter workspace: result reading, progress parsing, the toolkit process and the page."""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from src.gimap.features.sputter.application import AnalyzeRequest, overall_fraction, parse_progress
from src.gimap.features.sputter.domain import reliability_report
from src.gimap.features.sputter.infrastructure import StructureFolderReader, SubprocessToolkit, ToolkitNotConfigured

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


def _result_folder(tmp_path: Path) -> Path:
    keys = list(dict.fromkeys(k for r in ROWS for k in r))
    with (tmp_path / "structure.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(ROWS)
    (tmp_path / "structure_summary.json").write_text(json.dumps(SUMMARY), encoding="utf-8")
    return tmp_path


def test_reader_parses_numbers_and_reasons(tmp_path):
    result = StructureFolderReader().read(_result_folder(tmp_path))
    assert [r["time_s"] for r in result.rows] == [0.05, 0.6, 1.1]
    assert result.rows[0]["D_nm"] is None and "no resolved peak" in result.rows[0]["D_reason"]
    assert result.counts("R") == {"A": 1, "B": 0, "C": 1, "S": 0, "T": 0, "L": 0, "missing": 1}
    assert result.series("D", "A").value == [9.0] and result.ts_length().value == [12.0]
    html = reliability_report(result)
    assert "±16%" in html and "±31%" in html and "flat-based hemisphere" in html


def test_reader_refuses_a_folder_without_table(tmp_path):
    with pytest.raises(FileNotFoundError):
        StructureFolderReader().read(tmp_path)


def test_progress_lines_are_parsed_and_weighted():
    p = parse_progress("SPUTTER_PROGRESS\tfeatures\t0.500\tdirect features")
    assert p is not None and p.stage == "features" and p.message == "direct features"
    assert 0.65 < overall_fraction(p) < 0.76
    assert parse_progress("some other output") is None
    assert overall_fraction(parse_progress("SPUTTER_PROGRESS\tdone\t1\tok")) == 1.0


def test_request_problems_are_explained(tmp_path):
    bad = AnalyzeRequest(scan_folder=tmp_path / "missing", out_folder=tmp_path, sdd_mm=3387, alpha_i_deg=.4, metal="Ag")
    assert any("energy" in p for p in bad.problems()) and any("scan folder" in p for p in bad.problems())
    ok = AnalyzeRequest(scan_folder=tmp_path, out_folder=tmp_path, sdd_mm=3387, alpha_i_deg=.4, metal="Ag", energy_ev=12000.)
    assert ok.problems() == []


def test_toolkit_process_streams_progress_and_reports_errors(tmp_path, monkeypatch):
    from src.gimap.features.sputter.infrastructure import toolkit as toolkit_module
    monkeypatch.setattr(toolkit_module, "default_python", lambda: "")
    monkeypatch.setattr(toolkit_module, "default_folder", lambda: "")
    (tmp_path / "scripts").mkdir()
    script = tmp_path / "scripts" / "sputter_struct_v1.py"
    script.write_text("import sys\nprint('SPUTTER_PROGRESS\\tintake\\t0.5\\thalf', flush=True)\n"
                      "print('plain line', flush=True)\nsys.exit(0 if '--metal' in sys.argv else 3)\n", encoding="utf-8")
    runner = SubprocessToolkit(lambda: (sys.executable, str(tmp_path)))
    lines = []
    request = AnalyzeRequest(scan_folder=tmp_path, out_folder=tmp_path / "out", sdd_mm=3387, alpha_i_deg=.4, metal="Ag",
                             energy_ev=12000.)
    runner.analyze(request, lines.append, lambda: False)
    assert lines[0].startswith("SPUTTER_PROGRESS") and "plain line" in lines
    script.write_text("import sys\nprint('boom', flush=True)\nsys.exit(2)\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="boom"):
        runner.analyze(request, lines.append, lambda: False)
    with pytest.raises(ToolkitNotConfigured):
        SubprocessToolkit(lambda: ("", "")).analyze(request, lines.append, lambda: False)


def test_page_shows_a_result(qt_application, tmp_path):
    from src.gimap.features.sputter.bootstrap import create_sputter_service
    from src.gimap.features.sputter.presentation.page import SputterPage

    holder = {}
    page = SputterPage(create_sputter_service(lambda: holder["page"].toolkit_settings()))
    holder["page"] = page
    page.open_results(_result_folder(tmp_path))
    assert page.result is not None and page.table.rowCount() == 3
    assert page.table.item(1, 3).text() == "2.4" and page.merge_button.isEnabled()
    assert "hemisphere" in page.report.toPlainText()
    page.close(); page.deleteLater()


def test_sketch_spacing_follows_the_reliability_rules():
    import math
    from src.gimap.features.sputter.domain import Sketcher, series_extent, storyboard_indices, window_sketch

    base = dict(time_s=5., module="polymer_embed", R_nm=2.0, R_grade="A", sigma_log=0.0)
    ok = window_sketch(dict(base, D_nm=20., D_grade="A", D_direct_check="consistent", D_corr_nm=21.))
    assert ok.spacing == 20. and ok.spacing_source.startswith("model D")
    assert math.isclose(ok.coverage, math.pi * 4 * 2 / (math.sqrt(3) * 400))
    bad = window_sketch(dict(base, D_nm=20., D_direct_check="outside simulated 90% range", D_corr_nm=40.))
    assert bad.spacing == 40. and bad.spacing_source == "direct 2π/q*" and bad.notes
    none = window_sketch(dict(base))
    assert none.spacing_source == "not measured" and none.coverage is None and none.spacing == 8.
    film = window_sketch(dict(time_s=50., module="TS_layer", L_TS_nm=70., TS_metal_fraction_conditional=.6))
    assert film.kind == "film" and film.spacing == 70.
    sk = Sketcher()
    assert sk.top_view(ok, 150.) and len(sk.top_view(window_sketch(dict(base, D_corr_nm=4.)), 150.)) > 1000
    assert sum(sum(r) for r in sk.film_field(film, 150., n=40)) > 0
    width, height = series_extent([ok, bad, none, film])
    assert 30 <= width <= 150 and height >= 3
    times = [window_sketch(dict(base, time_s=t, D_corr_nm=10. if t != 10. else None)) for t in (1., 2., 5., 10., 11., 100.)]
    picks = storyboard_indices(times, 3)
    assert picks[0] == 0 and picks[-1] == 5 and 3 not in picks  # 10 s has no measured spacing; 11 s is preferred


def test_sketch_tab_draws_and_saves_a_storyboard(qt_application, tmp_path):
    from src.gimap.features.sputter.bootstrap import create_sputter_service
    from src.gimap.features.sputter.presentation.page import SputterPage
    from src.gimap.features.sputter.presentation.sketch_view import save_storyboard

    holder = {}
    page = SputterPage(create_sputter_service(lambda: holder["page"].toolkit_settings()))
    holder["page"] = page
    page.open_results(_result_folder(tmp_path))
    view = page.sketch
    assert view.slider.isEnabled() and view.slider.maximum() == 2  # two island windows and one TS film window
    view.slider.setValue(2); view.burial.setChecked(True)
    assert "film" in view.info.text()
    out = tmp_path / "storyboard.png"
    picks = save_storyboard(page.result, out, count=3)
    assert out.is_file() and out.stat().st_size > 1000 and len(picks) == 3
    page.close(); page.deleteLater()


def test_median_smoothing_keeps_models_apart_and_flags_overlap():
    from src.gimap.features.sputter.domain import smoothed_rows, window_sketch

    rows = [dict(time_s=float(i), module="polymer_embed", R_nm=r, D_corr_nm=10.) for i, r in enumerate([2., 2., 9., 2., 2.])]
    rows.append(dict(time_s=5., module="TS_layer", L_TS_nm=60.))
    smooth = smoothed_rows(rows, half=2)
    assert smooth[2]["R_nm"] == 2.0 and smooth[2]["smoothed_windows"] == 5   # the single-window jump is gone
    assert smooth[5]["L_TS_nm"] == 60. and smooth[5]["smoothed_windows"] == 1  # TS window not mixed with islands
    tight = window_sketch(dict(time_s=1., module="polymer_embed", R_nm=4., D_corr_nm=6.))
    assert tight.coverage > .9 and any("overlap" in n for n in tight.notes)


def test_time_consistent_path_is_an_optional_variant(qt_application, tmp_path):
    import csv as _csv
    from src.gimap.features.sputter.bootstrap import create_sputter_service
    from src.gimap.features.sputter.presentation.page import SputterPage

    folder = _result_folder(tmp_path)
    rows = [dict(r, R_grade="S" if r.get("R_nm") else "", sequence_status="data-supported", sequence_data_cost_nats="0.4")
            for r in ROWS]
    rows[1]["R_grade"] = "T"; rows[1]["sequence_status"] = "constraint-driven"; rows[1]["sequence_data_cost_nats"] = "3.1"
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with (folder / "structure_sequence.csv").open("w", newline="", encoding="utf-8") as f:
        w = _csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    (folder / "structure_sequence_summary.json").write_text(json.dumps(dict(SUMMARY, sequence=dict(
        changed_from_unconstrained=1, constraint_driven=1, first_film_time_s=1.1))), encoding="utf-8")
    holder = {}
    page = SputterPage(create_sputter_service(lambda: holder["page"].toolkit_settings()))
    holder["page"] = page
    page.open_results(folder)
    assert page.choice_box.isEnabled() and page.result.variant == "window"
    page.choice_box.setCurrentIndex(page.choice_box.findData("sequence"))
    assert page.result.variant == "sequence" and page.result.counts("R")["T"] == 1
    assert "Time-consistent path" in page.report.toPlainText() and "not more accurate" in page.report.toPlainText()
    page.close(); page.deleteLater()


def test_learned_selector_variant_shows_its_test(qt_application, tmp_path):
    import csv as _csv
    from src.gimap.features.sputter.bootstrap import create_sputter_service
    from src.gimap.features.sputter.presentation.page import SputterPage

    folder = _result_folder(tmp_path)
    rows = [dict(r, R_grade="L" if r.get("R_nm") else "", selector_probability="0.8") for r in ROWS]
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with (folder / "structure_selected.csv").open("w", newline="", encoding="utf-8") as f:
        w = _csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    test = dict(passed=False, R=dict(top=dict(wrong_branch_frac=.28, median_rel_error=.15),
                                    sel=dict(wrong_branch_frac=.15, median_rel_error=.09), oracle=dict(wrong_branch_frac=.08), pass_=0),
                D=dict(top=dict(wrong_branch_frac=.21, median_rel_error=.09), sel=dict(wrong_branch_frac=.13, median_rel_error=.08),
                       oracle=dict(wrong_branch_frac=.14), pass_=1))
    test["R"]["pass"] = test["R"].pop("pass_") == 1; test["D"]["pass"] = test["D"].pop("pass_") == 1
    (folder / "structure_selected_summary.json").write_text(json.dumps(dict(SUMMARY, selector=dict(test=test, low_confidence=0))),
                                                            encoding="utf-8")
    holder = {}
    page = SputterPage(create_sputter_service(lambda: holder["page"].toolkit_settings()))
    holder["page"] = page
    page.open_results(folder, "selected")
    assert page.result.variant == "selected" and page.result.counts("R")["L"] == 2
    text = page.report.toPlainText()
    assert "Learned selector" in text and "R: not passed" in text and "D: passed" in text and "28%" in text
    page.close(); page.deleteLater()


def test_the_toolkit_comes_from_its_local_folder_or_its_absence_is_explained(tmp_path, monkeypatch):
    import importlib.util
    from src.gimap.features.sputter.infrastructure import toolkit as toolkit_module

    # The research code is bundled with the program (tracked by git, not ignored) in this folder.
    root = Path(__file__).resolve().parents[1]
    assert toolkit_module.LOCAL_TOOLKIT == root / "utils" / "sputter_toolkit"
    assert "utils/sputter_toolkit/" not in (root / ".gitignore").read_text(encoding="utf-8").splitlines()
    for name in ("GIMAP_SPUTTER_PYTHON", "GIMAP_SPUTTER_TOOLKIT"):
        monkeypatch.delenv(name, raising=False)
    kit = tmp_path / "sputter_toolkit"
    monkeypatch.setattr(toolkit_module, "LOCAL_TOOLKIT", kit)
    runner = SubprocessToolkit(lambda: ("", ""))
    assert toolkit_module.default_folder() == ""
    with pytest.raises(ToolkitNotConfigured, match="not part of GIMaP") as missing:
        runner._resolve()
    assert str(kit) in str(missing.value)

    (kit / "scripts").mkdir(parents=True)
    (kit / toolkit_module.ENTRY).write_text("", encoding="utf-8")
    assert toolkit_module.default_folder() == str(kit)
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a, **k: None)
    assert toolkit_module.default_python() == ""
    with pytest.raises(ToolkitNotConfigured, match="Start_GIMaP.bat"):
        runner._resolve()
    # With PyTorch, GIMaP's own interpreter runs the model: the console python.exe, whose output can be read.
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a, **k: object())
    chosen = toolkit_module.default_python()
    assert Path(chosen).is_file() and Path(chosen).name.lower() != "pythonw.exe"
    assert runner._resolve() == (chosen, kit)


def test_start_script_sets_up_a_source_folder_without_python_knowledge():
    from pathlib import Path as _Path
    root = _Path(__file__).resolve().parents[1]
    bat = (root / "Start_GIMaP.bat").read_text(encoding="ascii")
    source = bat.index("setup_source_env.ps1"); portable_error = bat.index("Portable Python was not found")
    assert source < portable_error and 'if not exist "%PYTHON_EXE%" if exist "%~dp0main.py"' in bat
    setup = (root / "tools" / "setup_source_env.ps1").read_bytes()
    assert all(byte < 128 for byte in setup)
    text = setup.decode("ascii")
    assert '$UvVersion = "' in text and "Get-FileHash -Algorithm SHA256" in text and ".sha256" in text
    assert "requirements-sputter.txt" in text and (root / "requirements-sputter.txt").is_file()
    # PyTorch is added only when the external toolkit has been copied in.
    assert "utils\\sputter_toolkit\\scripts\\sputter_struct_v1.py" in text
    assert ".gimap/" in (root / ".gitignore").read_text(encoding="utf-8")
    update = (root / "Update_GIMaP.bat").read_text(encoding="ascii")
    assert "update_source.ps1" in update
