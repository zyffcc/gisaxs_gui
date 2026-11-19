"""After Deposition: samples found in folders and read by their names, and the report of their series."""

from __future__ import annotations

import csv
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from src.gimap.features.sputter.application import (DC, HIPIMS, NO_SAMPLE_FRAMES, NO_THICKNESS, OTHER_METAL,
                                                     STAGES, StaticSample, complete, find_samples, joined, merged,
                                                     name_parts, read_series, report_html, series_of, series_slug,
                                                     stage, write_csv)


def _frames(folder, *names) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for name in names:
        (folder / name).write_bytes(b"x")


@pytest.mark.parametrize("name, nominal, method, metal, prefix", [
    ("850_10Au", 10.0, "", "Au", "850"), ("850_H10Au", 10.0, HIPIMS, "Au", "850"), ("RT_H4Au", 4.0, HIPIMS, "Au", "RT"),
    ("x_DC_8Au", 8.0, DC, "Au", "x"), ("sample_6p5nm", 6.5, "", "", "sample"), ("zz_6,5 nm", 6.5, "", "", "zz"),
    ("PS_5nm_Ag", 5.0, "", "Ag", "PS"), ("Si_HiPIMS_12Au", 12.0, HIPIMS, "Au", "Si"),
    ("850", None, "", "", ""), ("850_samples", None, "", "", ""), ("zz_1_gisaxs_bs_00001", None, "", "", ""),
    ("run1", None, "", "", ""), ("Si(with NPs)", None, "", "", "")])
def test_a_folder_name_gives_the_thickness_the_method_the_metal_and_the_prefix(name, nominal, method, metal, prefix):
    parts = name_parts(name)
    assert (parts.nominal_nm, parts.method, parts.metal, parts.prefix) == (nominal, method, metal, prefix)


def test_a_folder_of_samples_is_found_with_its_bare_substrate_and_its_series(tmp_path):
    holder = tmp_path / "zz_set" / "zz_set"  # unpacked into a folder of the same name: looked into
    for name in ("850", "850_4Au", "850_H4Au", "zz NPs", "PS_5Ag"):
        _frames(holder / name / "zz_1_gisaxs_bs_00001" / "embl_2m", "a.cbf")
        _frames(holder / name / "zz_1_giwaxs_bs_00001" / "embl_2m", "zz_1_giwaxs_00001.cbf")  # not counted
    _frames(holder / "850_6Au" / "run1" / "zz_2_gisaxs_bs_00001", "a.cbf", "b.cbf")  # however deep the frames lie
    _frames(holder / "850_6Au" / "run2" / "zz_2_gisaxs_bs_00001", "c.cbf")
    (holder / "empty_8Au").mkdir()  # no frames: not a sample
    found = find_samples(tmp_path / "zz_set")
    assert [(sample.name, sample.nominal_nm, sample.method, sample.frames, sample.included, sample.note)
            for sample in found] == [
        ("850_4Au", 4.0, DC, 1, True, ""), ("850_6Au", 6.0, DC, 3, True, ""), ("850_H4Au", 4.0, HIPIMS, 1, True, ""),
        ("PS_5Ag", 5.0, "", 1, False, OTHER_METAL), ("zz NPs", None, "", 1, False, NO_THICKNESS)]
    assert {sample.reference for sample in found if sample.prefix == "850"} == {str(holder / "850")}
    assert [(name, [sample.name for sample in samples]) for name, samples in series_of(found)] == [
        ("850 · DC", ["850_4Au", "850_6Au"]), ("850 · HiPIMS", ["850_H4Au"]), ("PS", ["PS_5Ag"]),
        ("zz NPs", ["zz NPs"])]
    assert series_slug("850 · HiPIMS") == "850_HiPIMS" and series_slug("zz NPs") == "zz_NPs"


def test_one_sample_folder_or_one_frame_is_one_sample(tmp_path):
    _frames(tmp_path / "set" / "850" / "scan", "a.cbf")
    _frames(tmp_path / "set" / "850_10Au" / "scan_1", "a.cbf")
    _frames(tmp_path / "set" / "850_10Au" / "scan_2", "b.cbf")
    _frames(tmp_path / "set" / "Si_H4Au" / "run1", "a.cbf")  # one measurement folder: still this sample
    (sample,) = find_samples(tmp_path / "set" / "850_10Au")
    assert (sample.name, sample.nominal_nm, sample.frames, sample.reference) == (
        "850_10Au", 10.0, 2, str(tmp_path / "set" / "850"))  # the bare substrate beside it
    (sample,) = find_samples(tmp_path / "set" / "Si_H4Au")
    assert (sample.name, sample.nominal_nm, sample.method, sample.reference) == ("Si_H4Au", 4.0, HIPIMS, "")
    (frame,) = find_samples(tmp_path / "set" / "850_10Au" / "scan_1" / "a.cbf")
    assert (frame.name, frame.frames, frame.note) == ("a", 1, NO_THICKNESS)
    _frames(tmp_path / "plain", "notes.txt")
    (nothing,) = find_samples(tmp_path / "plain")
    assert nothing.note == NO_SAMPLE_FRAMES and not nothing.included
    assert find_samples(tmp_path / "missing") == [] and find_samples(tmp_path / "plain" / "notes.txt") == []


def test_the_list_takes_frames_once_and_names_dc_beside_hipims():
    listed = [StaticSample("/a/850_4Au", "850_4Au", 4.0, prefix="850")]
    added = [StaticSample("/a/850_4Au", "850_4Au", 4.0, prefix="850"),
             StaticSample("/a/850_H4Au", "850_H4Au", 4.0, HIPIMS, "850")]
    together, known = merged(listed, added)
    assert [sample.name for sample in together] == ["850_4Au", "850_H4Au"] and known == 1
    completed = complete(together, {"850": "/a/850"})
    assert [(sample.series, sample.reference) for sample in completed] == [("850 · DC", "/a/850"),
                                                                          ("850 · HiPIMS", "/a/850")]
    sample = completed[0]
    assert StaticSample.from_dict(sample.to_dict()) == sample and StaticSample.from_dict({}) is None


@pytest.mark.parametrize("radius, spacing, height, base, relief, density, named", [
    (3.0, 12.0, 4.0, 0.0, None, None, "separate islands"),  # far apart
    (5.0, 10.4, 6.0, 0.0, None, None, "grains in contact"),  # touching, not across the picture
    (6.0, 10.0, 6.0, 0.0, None, None, "grains joined into a network"),  # overlapping: a cluster across the picture
    (5.5, 16.0, 7.7, 2.4, 5.3, 0.99, "grains on a thin layer"), (8.5, 21.0, 10.1, 7.0, 3.1, 1.0, "continuous film"),
    (6.8, 15.0, 9.3, 6.3, 3.0, 0.87, "film with a density below 1"), (None, 12.0, 4.0, 0.0, None, None, "")])
def test_the_stage_is_named_as_the_real_space_picture_draws_the_solution(radius, spacing, height, base, relief,
                                                                          density, named):
    assert stage(radius, spacing, height, base, relief, density) == named and (named in STAGES or not named)
    row = {"R": radius, "d_hex": spacing, "height": height, "t_b": base, "T": relief, "rb": density}
    assert joined(row) == (named in ("grains joined into a network", "grains on a thin layer", "continuous film",
                                     "film with a density below 1"))


def _series(folder, rows) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    columns = ["label", "nominal_nm", "frames", "fit", "h_m_model", "height", "t_b", "rb", "T", "R", "d_hex", "eta"]
    with (folder / "report.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)
    (folder / "patterns.png").write_bytes(b"\x89PNG zz fits")


def test_the_report_holds_each_series_in_two_parts_with_its_stages_and_figures(tmp_path):
    _series(tmp_path / "zz_DC", [["zz_10Au", 10, 8, 1.83, 8.8, 10.1, 7.0, 1.0, 3.1, 8.5, 21, 0.7],
                                 ["zz_4Au", 4, 7, 12.5, 3.6, 4.0, 0.0, "", "", 3.0, 12.0, 0.31]])
    assert read_series("zz · DC", tmp_path / "missing") is None
    series = [read_series("zz · DC", tmp_path / "zz_DC", "zz")]
    assert [row["label"] for row in series[0].rows] == ["zz_4Au", "zz_10Au"]  # by nominal thickness
    assert [row["label"] for row in series[0].particles()] == ["zz_4Au"]
    assert [row["label"] for row in series[0].film()] == ["zz_10Au"] and list(series[0].figures) == ["patterns"]
    said = []
    page = report_html(series, lambda template, **values: said.append(template) or template.format(**values),
                       {"date": "2026-11-19", "version": "0.2", "package": "abc", "geometry": "12 keV", "mode": "Quick"})
    particles, film = page.index("<h3>Separate particles</h3>"), page.index("<h3>Connected film</h3>")
    assert particles < page.index("separate islands", particles) < film < page.index("continuous film", film)
    assert "Gap D − 2R (nm)" in page[particles:film] and "Film thickness (nm)" in page[film:]
    assert "Radius R (nm)" not in page[film:] and "Relief radius (nm)" in page[film:]  # no particles in a film
    assert "The film is connected from 10.0 nm on." in page and "Bare substrate: zz" in page
    assert 'class="poor">12.50<' in page and 'class="good">1.83<' in page
    assert page.count("data:image/png;base64,") == 1 and "Au films after their deposition: structure from GISAXS" in said
    rows = list(csv.DictReader(write_csv(series, tmp_path / "samples.csv").open(encoding="utf-8")))
    assert [(row["sample"], row["part"], row["stage"], row["gap_nm"], row["film_thickness_nm"]) for row in rows] == [
        ("zz_4Au", "separate particles", "separate islands", "6.0", ""),
        ("zz_10Au", "connected film", "continuous film", "", "7.0")]


def test_the_figure_has_a_panel_for_each_part_on_one_axis(tmp_path):
    from src.gimap.features.sputter.presentation.static_figures import parts_figure

    _series(tmp_path / "zz_DC", [["zz_10Au", 10, 8, 1.83, 8.8, 10.1, 7.0, 1.0, 3.1, 8.5, 21, 0.7],
                                 ["zz_4Au", 4, 7, 2.5, 3.6, 4.0, 0.0, "", "", 3.0, 12.0, 0.31]])
    series = read_series("zz · DC", tmp_path / "zz_DC")
    said = []
    drawn = parts_figure(series, lambda template, **values: said.append(template) or template)
    assert drawn.startswith(b"\x89PNG") and {"Separate particles", "Connected film", "Gap", "Film density"} <= set(said)
    series.rows = []
    assert parts_figure(series, lambda template, **values: template) == b""


def test_export_writes_the_report_of_the_list_and_offers_it(tmp_path, monkeypatch):
    from src.gimap.app.presentation.components.toast import visible_toasts
    from tests.test_sputter_main_screen import _close, _samples, _settle, _static_page

    page, _runner = _static_page(tmp_path)
    page.set_analysis_kind("static")
    _samples(tmp_path / "set", "zz_4Au", "zz_8Au")
    page.add_static_samples(tmp_path / "set")
    page.export_static_report(tmp_path / "out")  # nothing analysed yet: said
    assert any("No series of the list has analysed samples" in toast.text() for toast in visible_toasts(page))
    page.analyze_static()
    _settle(lambda: not page.static_analyzing())
    assert page.export_button.isEnabled() and "report of every series" in page.export_button.toolTip()
    _series(page.series_folder("zz"), [["zz_4Au", 4, 1, 3.3, 3.5, 4.0, 0.0, "", "", 3.0, 9, 0.4]])
    monkeypatch.setattr("PyQt5.QtWidgets.QFileDialog.getExistingDirectory", lambda *args: str(tmp_path / "out"))
    page.export_button.click()
    _settle(lambda: not page._exporting)
    report = tmp_path / "out" / "samples_report" / "report.html"
    assert report.is_file() and (tmp_path / "out" / "samples_report" / "samples.csv").is_file()
    assert "Series zz" in report.read_text(encoding="utf-8") and "Separate particles" in report.read_text(encoding="utf-8")
    assert any("Report of 1 series written" in toast.text() for toast in visible_toasts(page))
    _close(page)
