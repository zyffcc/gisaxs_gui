"""Hand-overs between the workspaces: what one hands to another, and what the receiving one says about it.

Analyze ▸ Send to Fitting and Send Series to Fitting (a GIWAXS I(q) is said to be one, and no particle-model
fit is set up for it), the Start page's series entry (it ends on the Series tab), a project opened over
series results that were not saved, the comparison of the start of a series with its end, and the texts that
name what is opened.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PyQt5.QtWidgets import QApplication, QFileDialog, QMessageBox

from src.gimap.app.presentation import i18n
from src.gimap.app.presentation.i18n import DEFAULT_LANGUAGE, apply_language

Q_A = np.linspace(0.01, 0.2, 120)


def _settle(rounds: int = 5) -> None:
    for _ in range(rounds):
        QApplication.processEvents()


def _fit_input(path: Path, *, giwaxs: bool = False, signed: bool = True, halves: str | None = None,
               scale: float = 1.0) -> Path:
    """A fit input as Analyze writes it: a signed GISAXS cut (native detector columns) or a GIWAXS I(q)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if giwaxs:
        q = np.linspace(0.3, 3.0, 300)
        intensity = scale * (20.0 * q ** -1.5 + 400.0 * np.exp(-0.5 * ((q - 1.7) / 0.015) ** 2))
        observation = {"source": "radial_bins"}
        label = "q (1/A)"
    else:
        half = Q_A
        q = np.concatenate([-half[::-1], half]) if signed else half
        radius = 60.0  # Å
        x = np.abs(q) * radius
        intensity = scale * (1e4 * (3 * (np.sin(x) - x * np.cos(x)) / x ** 3) ** 2 + 2.0)
        observation = {"source": "native_detector_columns"}
        if halves:
            observation["halves"] = halves
        label = "qy (1/A)"
    record = {"counting_model_valid": True, "file_format": "cbf", "gap_guard_px": 0,
              "intensity_unit": "counts_per_pixel", "summed_frames": 1, "threshold_enabled": False, **observation}
    lines = ["# GIMaP Analyze fit input (q in 1/A)", f"# source: {path.with_suffix('.cbf')}",
             "# curve: I(q)" if giwaxs else "# curve: Horizontal cut", f"# observation: {json.dumps(record, sort_keys=True)}",
             f"# columns: {label}  I  sigma  pixels"]
    lines += [f"{a:.8g} {b:.8g} {np.sqrt(b / 20.0):.6g} 20" for a, b in zip(q, intensity)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# -- the automatic analysis: the start of a series against its end -------------------------------------


def test_compare_with_the_start_takes_as_many_frames_as_the_end_shows(monkeypatch) -> None:
    """A series of 12 ends on frames 11–12 (the last fifth); its start is frames 1–2, not 1–10."""
    from tests.test_assistant_gui import _app

    _app()
    from src.gimap.features.assistant.presentation import GuidedAnalysis

    guided = GuidedAnalysis(lambda: None)
    launched = []
    monkeypatch.setattr(guided, "_launch", lambda options, on_finished, message, *, restore=None:
                        launched.append((options.frame, options.sum_frames, restore)))
    guided.report = {"ok": True, "frames": {"total": 12, "first": 11, "summed": 2}}
    guided.compare_start()
    guided.report = {"ok": True, "frames": {"total": 9, "first": 9, "summed": 1}}  # a short series: one frame each
    guided.compare_start()
    guided.report = {"ok": True, "frames": {"total": 257, "first": 248, "summed": 10}}
    guided.compare_start()
    assert launched == [(1, 2, {"frame": -1, "sum": 2}), (1, 1, {"frame": -1, "sum": 1}),
                        (1, 10, {"frame": -1, "sum": 10})]


def test_the_tools_text_teaches_the_default_sum_and_what_both_halves_are_for() -> None:
    from src.gimap.features.assistant.application.tool_specs import tool_specs

    specs = {spec.name: spec.description for spec in tool_specs(allow_images=False)}
    frame = specs["set_frame"]
    assert "sum 10" not in frame and "the last ten" not in frame
    assert "the last fifth of the series, at most 10" in frame and "start: frame 1 with the same sum" in frame
    halves = specs["set_halves"]
    assert "never both together" in halves  # both halves on |qy| are for looking: a fit takes one curve


# -- the texts that name what is opened ------------------------------------------------------------------


def test_every_place_that_opens_detector_frames_names_the_formats_the_reader_takes() -> None:
    """NeXus is not only ``.nxs``: the filters list every suffix, the sentences say NeXus/HDF5, in both languages."""
    from tests.test_assistant_gui import _app

    _app()
    from src.gimap.features.analyze.presentation import source_texts
    from src.gimap.features.analyze.presentation.views import analyze_page_view
    from src.gimap.features.analyze.presentation.views.analyze_page_view import AnalyzePageView
    from src.gimap.shared.detector_io import DETECTOR_SUFFIXES
    from PyQt5.QtWidgets import QWidget

    assert analyze_page_view.FILE_FILTER == source_texts.FILE_FILTER  # the background frame's dialog and Open
    assert all(f"*{suffix}" in analyze_page_view.FILE_FILTER for suffix in DETECTOR_SUFFIXES)
    host = QWidget()
    view = AnalyzePageView()
    view.setup_ui(host)
    said = [view.open_files_button.toolTip(), view.summary_label.text()]
    assert said == ["Open detector frames (CBF, NeXus/HDF5, TIFF, EDF) — Ctrl+O. You can also drop files or folders here.",
                    "Open or drop detector files (CBF, NeXus/HDF5, TIFF, EDF) to start"]
    for text in said:
        assert i18n.ZH.get(text) and "NeXus/HDF5" in i18n.ZH[text]
    assert not [key for key in i18n.ZH if "(CBF, NXS, TIFF, EDF)" in key]  # the sentences of before are gone
    host.deleteLater()


# -- Compare ▸ Add Folder on a Batch Export folder of Analyze --------------------------------------------

Q_W = np.linspace(0.3, 3.0, 180)
CHI = np.linspace(-89.5, 89.5, 90)


def _giwaxs_curves(frame: int):
    """The curves Analyze writes of one GIWAXS frame (two I(q), an I(χ), a ring's I(|χ|)), growing with ``frame``."""
    from src.gimap.features.analyze.domain import Curve

    grow = 1.0 + 0.2 * frame
    radial = 20.0 * Q_W ** -1.5 + 300.0 * grow * np.exp(-0.5 * ((Q_W - 1.7) / 0.02) ** 2)
    chi = 40.0 + 30.0 * grow * np.exp(-0.5 * (CHI / 12.0) ** 2)
    ring = 40.0 + 25.0 * grow * np.exp(-0.5 * ((np.abs(CHI[45:]) - 30.0) / 8.0) ** 2)

    def curve(key, title, x, y, label, region):
        return Curve(key, title, x, y, np.sqrt(y / 50.0), np.full(x.shape, 50, dtype=np.int64), label, region=region)

    return [
        curve("radial", "I(q)", Q_W, radial, "q (Å⁻¹)", {"sector": "full"}),
        curve("in_plane", "In-plane sector I(q)", Q_W, 0.6 * radial, "q (Å⁻¹)", {"sector": "in_plane"}),
        curve("azimuthal", "I(χ), q = 1.660–1.740 Å⁻¹", CHI, chi, "χ (°)", {"q": [1.66, 1.74]}),
        curve("region1_chi", "Ring q 1.700: I(χ), q 1.660–1.740 Å⁻¹", np.abs(CHI[45:]), ring, "|χ| (°)",
              # as Analyze writes a region: the word “columns” in a comment line is no column header
              {"cut_region": "Ring q 1.700", "columns": [10, 20], "both_sides": True}),
    ]


def _gisaxs_curves(frame: int):
    from src.gimap.features.analyze.domain import Curve

    qy = np.concatenate([-Q_A[::-1], Q_A])
    qz = np.linspace(0.02, 0.25, 80)
    horizontal = 1e4 * np.exp(-(np.abs(qy) * (30.0 + 2.0 * frame)) ** 2 / 3.0) + 2.0
    vertical = 500.0 * np.exp(-0.5 * ((qz - 0.06) / 0.01) ** 2) + 3.0 + frame

    def curve(key, title, x, y, label, region):
        return Curve(key, title, x, y, np.sqrt(y / 20.0), np.full(x.shape, 20, dtype=np.int64), label, region=region)

    return [curve("horizontal", "Horizontal cut", qy, horizontal, "qy (Å⁻¹)", {"rows": [100, 106], "columns": [0, 487]}),
            curve("vertical", "Vertical cut", qz, vertical, "qz (Å⁻¹)", {"rows": [0, 619], "columns": [240, 246]})]


def _batch_export(root: Path, *, frames: int = 6, gisaxs: bool = False, text: str = "csv", tables: bool = True,
                  per_frame: bool = True, fit_input: bool = True) -> Path:
    """A Batch Export folder written with Analyze's own writer and README: the curve tables, ``curves/``,
    ``fit_input/``, ``README.txt`` and ``<batch>_batch.json`` — as a batch of ``frames`` frames leaves it."""
    from src.gimap.features.analyze.application import BatchChoices, CurveTables, readme_text
    from src.gimap.features.analyze.infrastructure.adapters.csv_export import CsvCurveWriter

    writer, batch = CsvCurveWriter(), root.name
    gathered, titles, labels = CurveTables(), {}, []
    for frame in range(frames):
        curves = _gisaxs_curves(frame) if gisaxs else _giwaxs_curves(frame)
        stem = f"{batch}_frame{frame:04d}"
        metadata = {"source_file": str(root.parent / f"{batch}.nxs"), "frame_index": frame,
                    "measurement": "gisaxs" if gisaxs else "giwaxs"}
        labels.append(f"{batch}.nxs #{frame + 1}")
        if per_frame:
            writer.write(curves, root / "curves", stem, metadata, text)
        if tables:
            gathered.add(labels[-1], curves)
        if fit_input:
            source = "native_detector_columns" if gisaxs else "radial_bins"
            writer.write_xy(curves[0], root / "fit_input" / f"{stem}_fit_input.dat",
                            {**metadata, "observation": {"source": source, **({"halves": "mean"} if gisaxs else {})}})
        for curve in curves:
            titles.setdefault(curve.key, curve.title)
    choices = BatchChoices(tables=tables, per_frame=per_frame, fit_input=fit_input, text_format=text)
    written = gathered.write(writer, root, batch, text) if tables else []
    writer.write_record(root / f"{batch}_batch.json", {
        "title": "GIMaP Analyze batch export", "choices": choices.to_dict(), "frames": labels, "failed": [],
        "files": [path.name for path in written], "settings_of_first_frame": {"measurement": "gisaxs" if gisaxs else "giwaxs"}})
    root.mkdir(parents=True, exist_ok=True)
    (root / "README.txt").write_text(readme_text(
        choices, batch=batch, frame=f"{batch}_frame0000", frames=frames, curves=list(titles.items()),
        halves="mean" if gisaxs else None), encoding="utf-8")
    return root


def test_a_batch_export_folder_is_read_one_kind_of_curve_at_a_time(tmp_path: Path) -> None:
    """critic[04]: the folder gave 4 × frames “frames” of mixed curves, with χ labelled q."""
    from src.gimap.features.compare.application import CompareSettings, SeveralKinds
    from src.gimap.features.compare.bootstrap import create_compare_service

    service = create_compare_service()
    root = _batch_export(tmp_path / "mapi_scan0042", frames=6)
    kinds = service.folder_kinds(root)
    assert [kind.key for kind in kinds] == ["radial", "in_plane", "azimuthal", "region1_chi"]  # as the README lists them
    assert [kind.frames for kind in kinds] == [6, 6, 6, 6]
    assert kinds[0].title == "I(q)" and kinds[2].title == "I(χ), q = 1.660–1.740 Å⁻¹"  # as the README says them
    with pytest.raises(SeveralKinds) as refused:
        service.series_from_folder(root)
    assert [kind.key for kind in refused.value.kinds] == ["radial", "in_plane", "azimuthal", "region1_chi"]
    assert str(refused.value) == ("mapi_scan0042 holds 4 kinds of curves (radial, in_plane, azimuthal, region1_chi): "
                                  "choose one of them.")
    radial = service.series_from_folder(root, kind="radial")
    assert radial.rows == 6 and radial.x_label == "q (Å⁻¹)" and radial.name == "mapi_scan0042 · radial"
    assert radial.labels[0] == "mapi_scan0042_frame0000_radial.csv" and len(radial.paths) == 6
    assert radial.x == pytest.approx(Q_W, rel=1e-7)
    assert radial.image[3] == pytest.approx(_giwaxs_curves(3)[0].intensity, rel=1e-6)  # the frame's own values
    chi = service.series_from_folder(root, kind="azimuthal")
    assert chi.x_label == "χ (°)" and chi.rows == 6 and (chi.x.min(), chi.x.max()) == (-89.5, 89.5)
    ring = service.series_from_folder(root, kind="region1_chi")
    assert ring.x_label == "|χ| (°)" and ring.name == "mapi_scan0042 · region1_chi"
    with pytest.raises(ValueError, match="different x axes"):  # χ against q is refused, not compared
        service.compare([radial, chi], CompareSettings())
    with pytest.raises(ValueError) as unknown:
        service.series_from_folder(root, kind="vertical")
    assert str(unknown.value) == "mapi_scan0042 holds no curves “vertical” (radial, in_plane, azimuthal, region1_chi)."


def test_the_subfolders_of_a_batch_export_and_its_tables_are_read_too(tmp_path: Path) -> None:
    from src.gimap.features.compare.application import SeveralKinds
    from src.gimap.features.compare.bootstrap import create_compare_service

    service = create_compare_service()
    root = _batch_export(tmp_path / "mapi_scan0042", frames=5)
    # curves/: the same kinds, told apart by the record of every frame; named after the batch, not “curves”
    with pytest.raises(SeveralKinds):
        service.series_from_folder(root / "curves")
    plane = service.series_from_folder(root / "curves", kind="in_plane")
    assert plane.rows == 5 and plane.name == "mapi_scan0042 · in_plane" and plane.x_label == "q (Å⁻¹)"
    # fit_input/: one kind — the curve Fitting reads —, named after the batch
    sent = service.series_from_folder(root / "fit_input")
    assert sent.rows == 5 and sent.name == "mapi_scan0042" and sent.x_label == "q (Å⁻¹)"
    assert [kind.key for kind in service.folder_kinds(root / "fit_input")] == ["fit_input"]
    # Only the tables of the batch (no file per frame): a table is one series, its columns the frames
    tables = _batch_export(tmp_path / "tables_only", frames=7, per_frame=False, fit_input=False)
    kinds = service.folder_kinds(tables)
    assert [(kind.key, kind.frames) for kind in kinds] == [("radial", 7), ("in_plane", 7), ("azimuthal", 7), ("region1_chi", 7)]
    table = service.series_from_folder(tables, kind="azimuthal")
    assert table.rows == 7 and table.x_label == "χ (°)" and table.labels[6] == "tables_only.nxs #7"
    assert table.image[2] == pytest.approx(_giwaxs_curves(2)[2].intensity, rel=1e-5)  # (six digits in a table)
    assert table.paths == () and table.name == "tables_only · azimuthal"
    # Send Series to Fitting leaves fit_input/ only: its root is that series
    only = _batch_export(tmp_path / "sent", frames=4, gisaxs=True, tables=False, per_frame=False)
    cut = service.series_from_folder(only)
    assert cut.rows == 4 and cut.x_label == "qy (Å⁻¹)" and cut.name == "sent" and cut.x.min() < 0 < cut.x.max()
    # A GISAXS batch: the horizontal and the vertical cut are two kinds (they were interleaved, or refused)
    gisaxs = _batch_export(tmp_path / "gisaxs_run", frames=4, gisaxs=True)
    assert [kind.key for kind in service.folder_kinds(gisaxs)] == ["horizontal", "vertical"]
    vertical = service.series_from_folder(gisaxs, kind="vertical")
    assert vertical.x_label == "qz (Å⁻¹)" and vertical.rows == 4  # (its “# region:” line holds the word columns)
    # Space- and tab-separated exports read the same
    for text in ("dat", "txt"):
        other = _batch_export(tmp_path / f"as_{text}", frames=3, text=text)
        assert [kind.key for kind in service.folder_kinds(other)] == ["radial", "in_plane", "azimuthal", "region1_chi"]
        assert service.series_from_folder(other, kind="region1_chi").x_label == "|χ| (°)"
        from_table = _batch_export(tmp_path / f"table_{text}", frames=3, text=text, per_frame=False, fit_input=False)
        read = service.series_from_folder(from_table, kind="radial")
        assert read.rows == 3 and read.x_label == "q (Å⁻¹)" and read.x == pytest.approx(Q_W, rel=1e-5)


def test_a_folder_of_one_frames_exports_and_a_plain_folder_keep_working(tmp_path: Path) -> None:
    """Analyze ▸ Export of single frames (gimap_analysis/) has no README: the record of each frame names its
    curves. A folder of curve files from elsewhere is one series, as before."""
    import shutil

    from src.gimap.features.analyze.infrastructure.adapters.csv_export import CsvCurveWriter
    from src.gimap.features.compare.application import SeveralKinds
    from src.gimap.features.compare.bootstrap import create_compare_service
    from tests.test_compare import _run, _write_curves

    service, writer = create_compare_service(), CsvCurveWriter()
    exports = tmp_path / "sample_B" / "gimap_analysis"
    for frame in range(4):
        writer.write(_giwaxs_curves(frame), exports, f"film_{frame + 1:05d}", {"source_file": f"film_{frame + 1:05d}.cbf"})
    assert [kind.key for kind in service.folder_kinds(exports)] == ["radial", "in_plane", "azimuthal", "region1_chi"]
    assert service.folder_kinds(exports)[2].title == "I(chi), q = 1.660-1.740 1/A"  # as the files say it
    with pytest.raises(SeveralKinds, match="gimap_analysis holds 4 kinds"):
        service.series_from_folder(exports)
    series = service.series_from_folder(exports, kind="azimuthal")
    assert series.name == "sample_B · azimuthal" and series.x_label == "χ (°)" and series.rows == 4
    # … and the curves sent to Fitting from there are one more kind of that folder
    for frame in range(4):
        writer.write_xy(_giwaxs_curves(frame)[0], exports / f"film_{frame + 1:05d}_fit_input.dat",
                        {"source_file": f"film_{frame + 1:05d}.cbf", "observation": {"source": "radial_bins"}})
    assert [kind.key for kind in service.folder_kinds(exports)] == ["radial", "in_plane", "azimuthal", "region1_chi", "fit_input"]
    assert service.folder_kinds(exports)[-1].title == "I(q)"  # which curve it is: data (the page says what it is for)
    sent = tmp_path / "sample_C" / "gimap_analysis" / "fit_input"
    _write_curves(sent, _run(8, seed=3))  # run_1_fit_input.dat …: one kind, named after the sample
    kinds = service.folder_kinds(sent)
    assert [(kind.key, kind.frames) for kind in kinds] == [("fit_input", 8)]
    assert service.series_from_folder(sent).name == "sample_C"
    plain = tmp_path / "from_elsewhere"
    plain.mkdir()
    for index, row in enumerate(_run(5, seed=4)):
        (plain / f"curve_{index + 1}.txt").write_text("\n".join(f"{a:.6g}\t{b:.6g}" for a, b in zip(np.linspace(0.2, 3.0, 300), row)))
    kinds = service.folder_kinds(plain)
    assert [(kind.key, kind.frames) for kind in kinds] == [("", 5)]
    assert service.series_from_folder(plain).name == "from_elsewhere" and service.series_from_folder(plain).rows == 5
    # No file is left out: one curve for Fitting among other curves does not make the others vanish …
    shutil.copy2(sent / "run_1_fit_input.dat", plain / "extra_fit_input.dat")
    assert [(kind.key, kind.frames) for kind in service.folder_kinds(plain)] == [("", 6)]
    # … and curve files beside Analyze's exports are a kind of their own
    shutil.copy2(plain / "curve_1.txt", exports / "from_another_program_1.txt")
    shutil.copy2(plain / "curve_2.txt", exports / "from_another_program_2.txt")
    kinds = service.folder_kinds(exports)
    assert [(kind.key, kind.title, kind.frames) for kind in kinds][-2:] == [("fit_input", "I(q)", 4), ("other", "", 2)]
    assert service.series_from_folder(exports, kind="other").name == "sample_B · other"


def test_the_question_about_a_folder_says_what_the_applications_own_kinds_are_in_the_interface_language(tmp_path: Path) -> None:
    """“the curve for Fitting” and “other files” are the application's words, not names from a file: they were
    English in the Chinese question. So were the two sentences about a batch's table that cannot be read."""
    from src.gimap.features.analyze.infrastructure.adapters.csv_export import CsvCurveWriter
    from src.gimap.features.compare.application import ERRORS, sources
    from src.gimap.features.compare.bootstrap import create_compare_service
    from src.gimap.features.compare.presentation import series_input
    from src.gimap.features.compare.presentation.errors import error_text
    from tests.test_assistant_gui import _app

    _app()
    service, writer = create_compare_service(), CsvCurveWriter()
    exports = tmp_path / "sample_B" / "gimap_analysis"
    for frame in range(3):
        stem = f"film_{frame + 1:05d}"
        writer.write(_gisaxs_curves(frame), exports, stem, {"source_file": f"{stem}.cbf"})
        writer.write_xy(_gisaxs_curves(frame)[0], exports / f"{stem}_fit_input.dat",
                        {"source_file": f"{stem}.cbf", "observation": {"source": "native_detector_columns"}})
    (exports / "my_reference_curve.dat").write_text("# q I\n" + "\n".join(f"{0.01 * i:.3f} {100 / i:.3f}" for i in range(1, 40)))
    kinds = service.folder_kinds(exports)
    assert [kind.key for kind in kinds] == ["horizontal", "vertical", sources.FIT_INPUT_KIND, sources.OTHER_KIND]
    assert [series_input.kind_text(kind) for kind in kinds][-2:] == [
        "fit_input — the curve for Fitting (Horizontal cut) · 3 frames", "other — other files · 1 frames"]
    assert series_input.kind_title(sources.CurveKind(sources.FIT_INPUT_KIND, "")) == "the curve for Fitting"
    # A table of a batch that is none: the reader's two sentences are templates of Compare's errors
    table = tmp_path / "run_radial_frames.csv"
    table.write_text("q (1/A),a.nxs #1,a.nxs #2\n0.1,1,2\n0.2,4,oops\n0.3,7,8\n0.4,1,1\n", encoding="utf-8")
    with pytest.raises(ValueError) as not_numbers:
        service.reader.read_table(table)
    assert str(not_numbers.value) == "not a table of numbers (could not convert string to float: 'oops')"
    assert sources.NO_TABLE in ERRORS and sources.NOT_NUMBERS in ERRORS
    apply_language("zh")
    try:
        said = [series_input.kind_text(kind) for kind in kinds]
        assert said[0] == "horizontal — Horizontal cut · 3 帧"  # the curve's own title is data: as its files say it
        assert said[-2:] == ["fit_input — " + i18n.ZH[series_input.FIT_INPUT_TITLE].format(title="Horizontal cut") + " · 3 帧",
                             "other — " + i18n.ZH[series_input.OTHER_TITLE] + " · 1 帧"]
        assert "the curve for Fitting" not in said[-2] and "other files" not in said[-1]
        assert error_text(sources.NO_TABLE) == i18n.ZH[sources.NO_TABLE] != sources.NO_TABLE
        assert error_text(str(not_numbers.value)) == i18n.ZH[sources.NOT_NUMBERS].format(
            error="could not convert string to float: 'oops'")
    finally:
        apply_language(DEFAULT_LANGUAGE)


def test_add_folder_on_a_batch_export_asks_which_curve_and_names_the_series_after_it(tmp_path: Path, monkeypatch) -> None:
    from PyQt5.QtWidgets import QInputDialog

    from src.gimap.features.compare.presentation import series_input
    from tests.test_compare import _close, _idle, _shown_page, _wait

    asked = []

    def answer(key):
        def get_item(_parent, title, label, items, current, _editable):
            asked.append((title, label, list(items), current))
            chosen = next((item for item in items if item.startswith(f"{key} ")), None)
            return (chosen or "", chosen is not None)
        monkeypatch.setattr(QInputDialog, "getItem", staticmethod(get_item))

    page = _shown_page()
    try:
        first = _batch_export(tmp_path / "sample_A", frames=6)
        second = _batch_export(tmp_path / "sample_B", frames=6)
        answer("azimuthal")
        added = page.add_folder(first)
        title, label, items, current = asked[-1]
        assert title == "Curves of the Folder"
        assert label == "sample_A holds 4 kinds of curves. One kind is one series — which one?"
        assert items[0] == "radial — I(q) · 6 frames" and current == 0  # I(q) first while nothing was chosen before
        assert items[3].startswith("region1_chi — Ring q 1.700: I(χ)") and items[3].endswith(" · 6 frames")
        assert added is page.series[0] and added.name == "sample_A · azimuthal" and added.x_label == "χ (°)"
        assert added.rows == 6 and page.status_label.text() == "Added sample_A · azimuthal: 6 frames."
        assert page.series_table.item(0, 2).text() == "sample_A"  # From: the folder that was added
        page.add_folder(second)
        assert asked[-1][3] == 2  # the curve chosen last is offered first
        assert [item.name for item in page.series] == ["sample_A · azimuthal", "sample_B · azimuthal"]
        _wait(page, lambda: _idle(page) and page.comparison is not None)
        assert page.comparison.x_label == "χ (°)" and page.range_label.text() == "χ range"
        # Cancelled: nothing is added, and the page says so
        monkeypatch.setattr(QInputDialog, "getItem", staticmethod(lambda *args: ("", False)))
        assert page.add_folder(first) is None and len(page.series) == 2
        assert page.status_label.text() == "Nothing added: no curve of sample_A was chosen."
        # A kind given by the caller, and a folder with one kind: no question
        monkeypatch.setattr(QInputDialog, "getItem", staticmethod(lambda *args: pytest.fail("asked")))
        page.clear_all()
        assert page.add_folder(first, kind="radial").name == "sample_A · radial"
        assert page.add_folder(first / "fit_input").name == "sample_A"
        assert page.add_folder(second, kind="vertical") is None  # not a curve of this folder
        assert page.status_label.text() == "sample_B holds no curves “vertical” (radial, in_plane, azimuthal, region1_chi)."
        # Two folders dropped together are asked about once
        page.clear_all()
        asked.clear()
        answer("in_plane")
        dropped = page.add_dropped([first, second])
        assert len(asked) == 1 and [item.name for item in dropped] == ["sample_A · in_plane", "sample_B · in_plane"]
        assert series_input.KIND_QUESTION in i18n.ZH and series_input.KIND_ITEM in i18n.ZH
        assert page.add_folder_action.toolTip().endswith("holds several kinds of curves: you choose one")
    finally:
        _close(page)


def test_the_reader_names_the_axis_from_the_header_row_not_from_a_line_that_holds_the_word_columns(tmp_path: Path) -> None:
    from src.gimap.features.compare.bootstrap import create_compare_service
    from src.gimap.features.compare.infrastructure.files import axis_label, first_cell

    reader = create_compare_service().reader
    path = tmp_path / "frame_region1_chi.csv"
    path.write_text("\n".join([
        "# GIMaP Analyze export (q in 1/A, angles in deg)", "# curve: Ring q 1.700: I(chi), q 1.660-1.740 1/A",
        '# region: {"rows": [0, 9], "columns": [3, 8]}', "|chi| (deg),I (counts/pixel),sigma,pixels",
        "5.5,43.5,0.1,2837", "6.5,43.2,0.2,1719", "7.5,42.3,0.2,1407", "8.5,43.4,0.2,1270"]) + "\n", encoding="utf-8")
    x, intensity, label = reader.read(path)
    assert label == "|χ| (°)" and x.tolist() == [5.5, 6.5, 7.5, 8.5] and intensity[0] == 43.5
    assert reader.title(path) == "Ring q 1.700: I(chi), q 1.660-1.740 1/A"
    assert [axis_label(name) for name in ("q (1/A)", "q (Å⁻¹)", "qy (1/A)", "qz_(1/A)", "chi (deg)", "χ (°)", "|chi|_(deg)",
                                           "2theta (deg)", "2θ (°)", "q_par (1/A)", "q∥ (Å⁻¹)", "x", "")] == \
        ["q (Å⁻¹)", "q (Å⁻¹)", "qy (Å⁻¹)", "qz (Å⁻¹)", "χ (°)", "χ (°)", "|χ| (°)", "2θ (°)", "2θ (°)", "q∥ (Å⁻¹)",
         "q∥ (Å⁻¹)", "q (Å⁻¹)", "q (Å⁻¹)"]
    assert [first_cell(text) for text in ("q (1/A),I,sigma", "q (1/A)  I  sigma  pixels", "q_(1/A) I sigma", "q I", "q (nm^-1) I",
                                          "chi (deg)\tI")] == ["q (1/A)", "q (1/A)", "q_(1/A)", "q", "q (nm^-1)", "chi (deg)"]
    # nm⁻¹ said in the header row or in the columns line is converted; degrees never are
    nm = tmp_path / "nm.dat"
    nm.write_text("q (nm^-1)  I\n1.0 5\n2.0 6\n3.0 7\n", encoding="utf-8")
    assert reader.read(nm)[0].tolist() == pytest.approx([0.1, 0.2, 0.3]) and reader.read(nm)[2] == "q (Å⁻¹)"
    degrees = tmp_path / "chi_nm.dat"
    degrees.write_text("# columns: chi (deg)  I  (ring at 17 nm^-1)\n1.0 5\n2.0 6\n3.0 7\n", encoding="utf-8")
    assert reader.read(degrees)[0].tolist() == [1.0, 2.0, 3.0] and reader.read(degrees)[2] == "χ (°)"
    # a table: a header row, an empty cell where a frame has no data
    table = tmp_path / "run_radial_frames.csv"
    table.write_text("\n".join([
        "# GIMaP Analyze batch export (GIMaP Analyze 1): I(q)", "# one column per frame (3 frames, in list order)",
        'q (1/A),"a, b.nxs #1",a.nxs #2,a.nxs #3', "0.1,1,2,3", "0.2,4,,6", "0.3,7,8,9", "0.4,1,1,1"]) + "\n", encoding="utf-8")
    x, image, labels, label = reader.read_table(table)
    assert label == "q (Å⁻¹)" and labels == ("a, b.nxs #1", "a.nxs #2", "a.nxs #3") and image.shape == (3, 4)
    assert np.isnan(image[1, 1]) and image[2].tolist() == [3.0, 6.0, 9.0, 1.0] and reader.table_frames(table) == 3
    assert reader.title(table) == "I(q)"
    empty = tmp_path / "empty_frames.csv"
    empty.write_text("# GIMaP Analyze batch export (GIMaP Analyze 1): I(q)\nq (1/A)\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no table of frames"):
        reader.read_table(empty)


# -- Analyze ▸ Send to Fitting and Send Series to Fitting: what Fitting does with what it is handed --------


def _toasts(window) -> list:
    from src.gimap.app.presentation.components import visible_toasts

    return [(toast.property("level"), toast.text(), toast.action_button.text() if toast.action_button else "")
            for toast in visible_toasts(window)]


def test_a_giwaxs_curve_or_series_in_fitting_is_said_to_be_one_and_no_particle_fit_is_set_up(tmp_path: Path, monkeypatch) -> None:
    """critic[02]: an I(q) of a GIWAXS frame opened with “Find the particle shape” chosen and the Fit step shown,
    and a GIWAXS series was fitted with particle models without a word."""
    from src.gimap.features.fitting.presentation.workspace import NOTE_TEXT
    from tests.test_fit_series import _answer
    from tests.test_fitting_presentation import _fitting_window

    app, window = _fitting_window()
    try:
        components = window.components
        workspace, page = components.fitting_workspace, components.fitting_workspace.fit_page
        note = workspace.handover_note
        assert note.isHidden()
        # A GISAXS cut: Predict parameters (fit1d) is chosen for it, no note
        cut = _fit_input(tmp_path / "sample" / "gimap_analysis" / "cut_00001_fit_input.dat")
        components._send_curve_to_fitting(cut, "mean")
        _settle()
        assert page.session.curve is not None and page.method() == "fit1d" and page.step_rail.current() == "fit"
        assert note.isHidden()
        assert ("ok", "Sent to Fitting — the curve file is in " + str(Path("…", "sample", "gimap_analysis")), "Open Folder") in _toasts(window)
        # A GIWAXS I(q): the note says what Fitting's models are for; nothing is preselected
        ring = _fit_input(tmp_path / "film" / "gimap_analysis" / "film_00001_fit_input.dat", giwaxs=True)
        components._send_curve_to_fitting(ring, "both_abs")
        _settle()
        assert page.session.curve.name == "film_00001_fit_input.dat"
        assert page.method() not in ("shapes", "fit1d") and page.step_rail.current() == "curve"
        assert [toast for toast in _toasts(window) if toast[2] == "Open Folder"] == [
            ("ok", "Sent to Fitting — the curve file is in " + str(Path("…", "film", "gimap_analysis")), "Open Folder")]  # one at a time
        assert note.isVisibleTo(window) and note.title_label.text() == "A GIWAXS I(q), not a GISAXS cut"
        assert "particle form factors for the horizontal cut of a GISAXS frame" in note.message_label.text()
        assert "Analyze ▸ Batch Export ▸ Peaks of the ring regions" in note.message_label.text()
        assert page.status_label.text() == "film_00001_fit_input.dat is a GIWAXS I(q): no particle-shape search was set up."
        # … also when it is opened by hand, and gone again with a GISAXS cut
        page.open_curve(cut)
        _settle()
        assert note.isHidden() and page.method() == "fit1d"
        page.open_curve(ring)
        _settle()
        assert note.isVisibleTo(window) and page.method() != "shapes"
        # In-situ series: the same for a folder of GIWAXS curves; Single keeps its own
        series = tmp_path / "film" / "gimap_analysis" / "run" / "fit_input"
        for frame in range(5):
            _fit_input(series / f"film_frame{frame:04d}_fit_input.dat", giwaxs=True, scale=1.0 + 0.1 * frame)
        _answer(monkeypatch, QMessageBox.Discard)
        components._send_series_to_fitting(series)
        _settle()
        assert len(workspace.series_page.paths) == 5 and workspace.context_stack.currentWidget() is workspace.series_page
        assert note.isVisibleTo(window) and note.title_label.text() == "GIWAXS I(q) curves, not GISAXS cuts"
        assert "Start would fit one to every frame" in note.message_label.text()
        page.open_curve(cut)  # (Single holds a GISAXS cut now)
        workspace.show_context("single")
        _settle()
        assert note.isHidden()
        workspace.show_context("insitu")
        _settle()
        assert note.isVisibleTo(window)
        # In Chinese, and closed for good for these curves once it is closed
        apply_language("zh", [window])
        components.refresh_language("zh")
        assert note.title_label.text() == i18n.ZH["GIWAXS I(q) curves, not GISAXS cuts"]
        assert note.message_label.text() == i18n.ZH[NOTE_TEXT["insitu"]]
        apply_language(DEFAULT_LANGUAGE, [window])
        components.refresh_language(DEFAULT_LANGUAGE)
        assert note.title_label.text() == "GIWAXS I(q) curves, not GISAXS cuts"
        note.dismiss_button.click()
        workspace.show_context("single")
        workspace.show_context("insitu")
        _settle()
        assert note.isHidden()
        # A series of GISAXS cuts: no note
        cuts = tmp_path / "sample" / "gimap_analysis" / "run" / "fit_input"
        for frame in range(4):
            _fit_input(cuts / f"cut_frame{frame:04d}_fit_input.dat", halves="positive", scale=1.0 + 0.1 * frame)
        components._send_series_to_fitting(cuts)
        _settle()
        assert len(workspace.series_page.paths) == 4 and note.isHidden()
        assert workspace.series_page.series_side() == "positive"  # the halves Analyze sent the series with
        components._send_series_to_fitting(series)  # the GIWAXS folder again: said again
        _settle()
        assert note.isVisibleTo(window)
    finally:
        apply_language(DEFAULT_LANGUAGE)
        window.close()
        _settle()


def test_a_run_on_a_series_of_giwaxs_curves_ends_with_the_note_saying_what_came_of_it(tmp_path: Path, monkeypatch) -> None:
    """A GIWAXS series could still be started, and its end read “Series done: 4 frames fitted, 0 failed.” —
    the sentence of a success — although every frame was a poor fit. The note comes back with what the run
    came to, also when it was closed before the run."""
    from src.gimap.features.fitting.application.series_fit import POOR_CHI2, poor_fit, series_counts
    from src.gimap.features.fitting.presentation.workspace import NOTE_RUN, NOTE_RUN_POOR
    from tests.test_fit_series import _answer, _wait

    window = _window()
    try:
        components = window.components
        workspace = components.fitting_workspace
        series, note = workspace.series_page, workspace.handover_note
        folder = tmp_path / "film" / "gimap_analysis" / "run" / "fit_input"
        for frame in range(4):
            _fit_input(folder / f"film_frame{frame:04d}_fit_input.dat", giwaxs=True, scale=1.0 + 0.1 * frame)
        _answer(monkeypatch, QMessageBox.Discard)
        components._send_series_to_fitting(folder)
        _settle()
        assert note.isVisibleTo(window)
        note.dismiss_button.click()  # closed for this series …
        _settle()
        assert note.isHidden()
        series.start()
        _wait(series)
        _settle()
        assert not series.running and len(series.fits) == 4
        counts = series_counts(list(series.fits.values()))
        fitted, poor = counts["ok"] + counts["not_converged"], sum(poor_fit(fit) for fit in series.fits.values())
        assert fitted == 4 and poor >= 3  # a sphere does not describe a diffraction peak
        # … and said again once the run has ended: what came of it, at the level of a warning
        assert note.isVisibleTo(window) and note.property("gimapBannerLevel") == "warning"
        assert note.message_label.text() == NOTE_RUN_POOR.format(poor=poor, fitted=fitted, limit=f"{POOR_CHI2:g}")
        apply_language("zh", [window])
        components.refresh_language("zh")
        assert note.message_label.text() == i18n.ZH[NOTE_RUN_POOR].format(poor=poor, fitted=fitted, limit=f"{POOR_CHI2:g}")
        apply_language(DEFAULT_LANGUAGE, [window])
        components.refresh_language(DEFAULT_LANGUAGE)
        note.dismiss_button.click()
        workspace.show_context("single")
        workspace.show_context("insitu")
        _settle()
        assert note.isHidden()  # closed again: it stays closed for this run
        for template in (NOTE_RUN, NOTE_RUN_POOR):
            assert i18n.ZH.get(template) and sorted(set(re_fields(template))) == sorted(set(re_fields(i18n.ZH[template])))
    finally:
        apply_language(DEFAULT_LANGUAGE)
        window.close()
        _settle()


def re_fields(text: str) -> list:
    import string

    return [name for _text, name, _spec, _conversion in string.Formatter().parse(text) if name]


# -- a project opened over results that were not saved; the Start page's series entry --------------------


def _window():
    from tests.test_fitting_presentation import _fitting_window

    _app, window = _fitting_window()
    return window


def test_opening_a_project_asks_before_unsaved_series_results_go(tmp_path: Path, monkeypatch) -> None:
    """fitting-insitu-ui[19]: a project replaces the In-situ series and keeps no fits; they went without a word."""
    from types import SimpleNamespace

    from src.gimap.app import project

    window = _window()
    try:
        components, menus = window.components, window.menus
        series = components.fitting_workspace.series_page
        saved = project.save(components, tmp_path / "sample.gimap")
        folder = tmp_path / "run" / "fit_input"
        for frame in range(3):
            _fit_input(folder / f"cut_{frame:04d}_fit_input.dat")
        assert series.open_series(folder) and len(series.paths) == 3
        series.fits = {0: SimpleNamespace(fitted=True, ok=True)}  # a run that ended, its table not saved
        series._unsaved = True
        asked = []

        def answer(reply):
            monkeypatch.setattr(QMessageBox, "question", staticmethod(
                lambda _parent, title, text, *_args: (asked.append((title, text)), reply)[1]))

        answer(QMessageBox.Cancel)
        assert menus.open_project(str(saved)) is False
        assert asked == [("In-situ series", "The results of the last run are not saved.\nSave the table of every frame first?")]
        assert len(series.paths) == 3 and series.fits and menus.project_path == ""  # nothing of the project applied
        answer(QMessageBox.Discard)
        assert menus.open_project(str(saved)) is True and len(asked) == 2
        assert series.paths == [] and not series.fits and menus.project_path == str(saved)
        assert menus.open_project(str(saved)) is True and len(asked) == 2  # nothing unsaved: no question
    finally:
        window.close()
        _settle()


def _frames(folder: Path, count: int = 4) -> Path:
    from PIL import Image

    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)
    for index in range(count):
        Image.fromarray(rng.poisson(20, (48, 64)).astype(np.int32)).save(folder / f"film_{index + 1:05d}.tif")
    return folder


def _stack(path: Path, frames: int = 6) -> Path:
    import h5py

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("entry/instrument/detector/data",
                              data=np.random.default_rng(2).poisson(20, (frames, 48, 64)).astype(np.int32))
    return path


def test_the_start_pages_series_entry_takes_a_folder_or_files_and_ends_on_the_series_tab(tmp_path: Path, monkeypatch) -> None:
    """critic[17]: the entry took a folder only and left the person on the Data step with the Curves tab."""
    from PyQt5.QtWidgets import QDialog
    from src.gimap.features.analyze.presentation.frame_import_dialog import FrameImportDialog

    monkeypatch.setattr(FrameImportDialog, "exec_", lambda _dialog: QDialog.Accepted)
    window = _window()
    try:
        components = window.components
        page, home = components.analyze_page, components.home_page
        folder, stack = _frames(tmp_path / "run_7"), _stack(tmp_path / "scans" / "insitu_0042.nxs")
        asked = []
        monkeypatch.setattr(QFileDialog, "getExistingDirectory", staticmethod(
            lambda *_args, **_kw: (asked.append("folder"), str(folder))[1]))
        monkeypatch.setattr(QFileDialog, "getOpenFileNames", staticmethod(
            lambda *_args, **_kw: (asked.append("files"), ([str(stack)], ""))[1]))
        card = home.cards["series"]
        assert [(button.text(), bool(button.toolTip())) for button in card.link_buttons] == [("Open Folder…", True), ("Open Files…", True)]
        assert home.cards["giwaxs"].link_buttons == []
        # The card (and its first link): a folder of frames
        components.show_page("home")
        card.click()
        assert asked == ["folder"] and components.current_page_key() == "analyze"
        assert page.current_right() == "series"  # the task is visible while metadata is inspected
        assert page.tasks.wait(60)
        _settle()
        assert len(page.view_model.state.files) == 4 and page.current_right() == "series"
        assert page.right_tabs.currentIndex() == page.right_tabs.findData("series")
        assert page.frame_listing_model.rowCount() == 4
        assert page.current_right() == "series"  # the first frame shown does not take the tab away
        # Frames already listed: the card leads to the tab, without a dialog …
        page.show_right("curves")
        components.show_page("home")
        card.click()
        assert asked == ["folder"] and components.current_page_key() == "analyze" and page.current_right() == "series"
        # … but a link says “…”: it always opens its dialog, here for the next series (one file may hold it whole)
        page.show_right("curves")
        components.show_page("home")
        card.link_buttons[1].click()
        assert asked == ["folder", "files"] and components.current_page_key() == "analyze"
        assert page.current_right() == "series"
        assert page.tasks.wait(60)
        _settle()
        assert [path.name for path in page.view_model.state.files][-1] == "insitu_0042.nxs" and len(page.view_model.state.files) == 5
        assert page.view_model.current_path.name == "insitu_0042.nxs" and page.current_right() == "series"
        assert page.view_model.listed_frames() == 4 + 6
        assert page.frame_listing_model.rowCount() == 4 + 6
        other = _frames(tmp_path / "run_8", 3)
        monkeypatch.setattr(QFileDialog, "getExistingDirectory", staticmethod(
            lambda *_args, **_kw: (asked.append("folder"), str(other))[1]))
        page.show_right("curves")
        components.show_page("home")
        card.link_buttons[0].click()
        assert asked == ["folder", "files", "folder"]
        assert page.tasks.wait(60)
        _settle()
        assert len(page.view_model.state.files) == 8 and page.frame_listing_model.rowCount() == 13
        assert components.current_page_key() == "analyze" and page.current_right() == "series"
        # A link's dialog cancelled with frames listed: Analyze and the Series tab, the list as it was
        monkeypatch.setattr(QFileDialog, "getOpenFileNames", staticmethod(lambda *_args, **_kw: (asked.append("files"), ([], ""))[1]))
        page.show_right("curves")
        components.show_page("home")
        card.link_buttons[1].click()
        assert asked[-1] == "files" and len(asked) == 4 and len(page.view_model.state.files) == 8
        assert components.current_page_key() == "analyze" and page.current_right() == "series"
        monkeypatch.setattr(QFileDialog, "getOpenFileNames", staticmethod(
            lambda *_args, **_kw: (asked.append("files"), ([str(stack)], ""))[1]))
        # From an empty Analyze the second link lists the one file that holds the series
        page.clear_files()
        page.show_right("curves")
        components.show_page("home")
        card.link_buttons[1].click()
        assert page.current_right() == "series"
        assert page.tasks.wait(60)
        _settle()
        assert [path.name for path in page.view_model.state.files] == ["insitu_0042.nxs"] and page.current_right() == "series"
        assert page.view_model.listed_frames() == 6
        assert page.frame_listing_model.rowCount() == 6
        # Nothing chosen: Analyze is shown as it is
        page.clear_files()
        page.show_right("curves")
        monkeypatch.setattr(QFileDialog, "getExistingDirectory", staticmethod(lambda *_args, **_kw: ""))
        components.show_page("home")
        card.click()
        assert components.current_page_key() == "analyze" and page.current_right() == "curves"
        # What the page says about the formats, in both languages
        said = "CBF, TIFF, EDF, NeXus / HDF5 — one file per frame, or a whole series in one file; the module files of a scan are stitched"
        from PyQt5.QtWidgets import QLabel

        assert said in [label.text() for label in home.drop_zone.findChildren(QLabel)]
        for text in (said, *(button.toolTip() for button in card.link_buttons)):
            assert i18n.ZH.get(text), text
    finally:
        window.close()
        _settle()


# -- Send to Fitting from a data folder that takes no file (raw data of a beamline) ---------------------


@pytest.mark.skipif(__import__("sys").platform.startswith("win"), reason="a folder is made read-only with its mode")
def test_send_to_fitting_from_a_read_only_data_folder_asks_once_where_to_write(tmp_path: Path, monkeypatch) -> None:
    """critic[03]: “Could not open the curve in Fitting: [Errno 13] Permission denied …”, and nothing was sent."""
    import os
    import shutil

    from src.gimap.features.analyze.bootstrap import create_analyze_view_model
    from src.gimap.features.analyze.presentation.page import AnalyzePage
    from src.gimap.features.fitting.infrastructure.adapters.local_files import read_fit_input_extras
    from src.gimap.integrations.state import InMemoryInstrumentProfileRepository
    from tests.test_analyze_workspace import CBF, _app, _context, _pilatus_profile
    from tests.test_giwaxs_workspace import _settle as _shown

    if not CBF.is_file():
        pytest.skip("the P03 frames of the repository are not here")
    _app()
    raw, out = tmp_path / "raw", tmp_path / "mine"
    raw.mkdir()
    out.mkdir()
    shutil.copy2(CBF, raw / CBF.name)
    sent, asked = [], []
    page = AnalyzePage(create_analyze_view_model(_context(InMemoryInstrumentProfileRepository([_pilatus_profile()]))),
                       send_to_fitting=lambda path, side: sent.append((Path(path), side)))
    page.show()
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", staticmethod(
        lambda _parent, title, *_args: (asked.append(title), str(out))[1]))
    try:
        page.add_paths([str(raw)])
        _shown(page, lambda: page.view_model.state.analysis is not None and page.view_model.state.analysis.reduction is not None)
        os.chmod(raw, 0o555)
        page.view_model.set_fit_side("positive")
        page.send_to_fitting()
        assert len(asked) == 1 and asked[0].startswith("Cannot write next to the data (Permission denied)")
        assert [(path.parent, side) for path, side in sent] == [(out, "positive")] and sent[0][0].is_file()
        assert read_fit_input_extras(sent[0][0])[1].get("halves") == "positive"  # the halves go with the file
        assert page.status_text() == (f"Opened {sent[0][0].name} in Fitting. The folder of the data cannot be written: "
                                      f"the file is in {out}") and page.status_level() == "ok"
        page.send_to_fitting()  # the next one: no question, the same folder
        assert len(asked) == 1 and len(sent) == 2 and sent[1][0].parent == out
        assert page.session_export_dir() == out
        # A folder that can be written: next to the data, as before, without a question
        os.chmod(raw, 0o755)
        page.send_to_fitting()
        assert len(asked) == 1 and sent[2][0].parent == raw / "gimap_analysis"
        assert page.status_text() == f"Opened {sent[2][0].name} in Fitting."
        # No folder chosen: nothing is sent, and the status line says why
        shutil.rmtree(raw / "gimap_analysis")
        os.chmod(raw, 0o555)
        page._session_export_dir = None  # (a new session)
        monkeypatch.setattr(QFileDialog, "getExistingDirectory", staticmethod(lambda *_args: ""))
        page.send_to_fitting()
        assert len(sent) == 3 and page.status_level() == "error"
        assert page.status_text().startswith("Could not open the curve in Fitting:")
    finally:
        os.chmod(raw, 0o755)
        page.tasks.wait(30)
        page.dispose()
        page.close()


def _read_only_page(tmp_path: Path, sent: list):
    """Analyze on one P03 frame in ``raw/`` (still writable: the caller takes the right away once it is shown)."""
    import shutil

    from src.gimap.features.analyze.bootstrap import create_analyze_view_model
    from src.gimap.features.analyze.presentation.page import AnalyzePage
    from src.gimap.integrations.state import InMemoryInstrumentProfileRepository
    from tests.test_analyze_workspace import CBF, _app, _context, _pilatus_profile
    from tests.test_giwaxs_workspace import _settle as _shown

    if not CBF.is_file():
        pytest.skip("the P03 frames of the repository are not here")
    _app()
    raw = tmp_path / "raw"
    raw.mkdir()
    shutil.copy2(CBF, raw / CBF.name)
    page = AnalyzePage(create_analyze_view_model(_context(InMemoryInstrumentProfileRepository([_pilatus_profile()]))),
                       send_to_fitting=lambda path, side: sent.append((Path(path), side)))
    page.show()
    page.add_paths([str(raw)])
    _shown(page, lambda: page.view_model.state.analysis is not None and page.view_model.state.analysis.reduction is not None)
    return page, raw


@pytest.mark.skipif(__import__("sys").platform.startswith("win"), reason="a folder is made read-only with its mode")
def test_the_fitting_buttons_on_a_read_only_data_folder_ask_where_to_write(tmp_path: Path, monkeypatch) -> None:
    """critic[03] where the person clicks: Fitting in the command bar and in the Export step were connected to
    another Send to Fitting, which said “Permission denied” and stayed on Analyze."""
    import os

    from src.gimap.features.analyze.presentation.bindings.fit_handover import FitHandoverMixin
    from src.gimap.features.analyze.presentation.page import AnalyzePage

    assert AnalyzePage.fit_current is FitHandoverMixin.fit_current  # the page's one Send to Fitting, by any name
    sent, asked = [], []
    page, raw = _read_only_page(tmp_path, sent)
    out = tmp_path / "mine"
    out.mkdir()
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", staticmethod(
        lambda _parent, title, *_args: (asked.append(title), str(out))[1]))
    try:
        os.chmod(raw, 0o555)
        page.fit_button.click()
        assert len(asked) == 1 and [path.parent for path, _side in sent] == [out] and page.status_level() == "ok"
        page.export_fit_button.click()  # the Export step's button: the same folder, no question
        assert len(asked) == 1 and [path.parent for path, _side in sent] == [out, out]
    finally:
        os.chmod(raw, 0o755)
        page.tasks.wait(30)
        page.dispose()
        page.close()


@pytest.mark.skipif(__import__("sys").platform.startswith("win"), reason="a folder is made read-only with its mode")
def test_a_folder_that_takes_no_file_either_is_not_kept_and_the_question_comes_again(tmp_path: Path, monkeypatch) -> None:
    """The folder answered was kept before anything was written into it: one that cannot be written (another
    folder of the beamline's file system) made every later Send to Fitting fail, with no question any more."""
    import os

    from src.gimap.features.analyze.presentation.bindings import fit_handover

    sent, asked = [], []
    page, raw = _read_only_page(tmp_path, sent)
    locked, good, later = tmp_path / "also_read_only", tmp_path / "mine", tmp_path / "second"
    for folder in (locked, good, later):
        folder.mkdir()
    answers: list = []
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", staticmethod(
        lambda _parent, title, *_args: (asked.append(title), answers.pop(0) if answers else "")[1]))
    try:
        os.chmod(raw, 0o555)
        os.chmod(locked, 0o555)
        # The folder answered cannot be written: asked once more, with that folder named; nothing of it is kept
        answers[:] = [str(locked), str(good)]
        page.send_to_fitting()
        assert len(asked) == 2 and asked[0].startswith("Cannot write next to the data (Permission denied)")
        assert asked[1] == f"Cannot write in {locked} (Permission denied); export to…"
        assert [path.parent for path, _side in sent] == [good] and page.status_level() == "ok"
        assert page.session_export_dir() == good
        page.send_to_fitting()  # written there once: kept, no question
        assert len(asked) == 2 and sent[1][0].parent == good
        # The folder of the session takes no file any more: it is forgotten, and the question names it
        os.chmod(good, 0o555)
        answers[:] = [str(later)]
        page.send_to_fitting()
        assert len(asked) == 3 and asked[2] == f"Cannot write in {good} (Permission denied); export to…"
        assert sent[2][0].parent == later and page.session_export_dir() == later
        # Two folders that take no file, one after the other: the send fails and says so — and the next one asks
        os.chmod(later, 0o555)
        answers[:] = [str(locked), str(locked)]
        page.send_to_fitting()
        assert len(asked) == 5 and len(sent) == 3 and page.status_level() == "error"
        assert page.status_text().startswith("Could not open the curve in Fitting:") and str(locked) in page.status_text()
        assert page.session_export_dir() is None
        os.chmod(good, 0o755)
        answers[:] = [str(good)]
        page.send_to_fitting()
        assert len(asked) == 6 and asked[5].startswith("Cannot write next to the data") and sent[3][0].parent == good
        assert i18n.ZH.get(fit_handover.FOLDER_REFUSED) and "{folder}" in i18n.ZH[fit_handover.FOLDER_REFUSED]
    finally:
        for folder in (raw, locked, good, later):
            os.chmod(folder, 0o755)
        page.tasks.wait(30)
        page.dispose()
        page.close()


def test_compares_error_notice_goes_once_the_series_are_compared_after_all() -> None:
    """An error notice stays until it is closed: “Could not compare …” stood beside “2 series compared.”"""
    from types import SimpleNamespace

    from src.gimap.app.presentation.components import visible_toasts
    from tests.test_compare import Q, _close, _idle, _run, _shown_page, _wait

    def a_map(seed, label="q (Å⁻¹)"):
        return SimpleNamespace(x=Q, image=_run(30, seed=seed), labels=tuple(f"f{i}" for i in range(30)), x_label=label)

    page = _shown_page()
    try:
        page.add_map(a_map(1), "run A")
        page.add_map(a_map(2, "χ (°)"), "ring of run A")  # another axis: not to be compared with I(q)
        _wait(page, lambda: _idle(page) and page.failure_reason() != "")
        errors = [toast for toast in visible_toasts(page) if toast.property("level") == "error"]
        assert [toast.text() for toast in errors] == ["Could not compare: The series have different x axes: q (Å⁻¹), χ (°)"]
        page.series_table.selectRow(1)
        page.remove_selected()
        page.add_map(a_map(3), "run B")
        _wait(page, lambda: _idle(page) and page.comparison is not None)
        assert page.status_label.text() == "2 series compared."
        assert not [toast for toast in visible_toasts(page) if toast.property("level") == "error"]
    finally:
        _close(page)


# -- the composition root, and the words ---------------------------------------------------------------


def test_notices_in_the_main_window_keep_off_analyzes_progress_bar_cancel_and_status_line() -> None:
    """During a batch the notices lay exactly over Analyze's progress bar and Cancel, and over “Batch 0/8: …”."""
    from PyQt5.QtCore import QPoint, QRect
    from PyQt5.QtTest import QTest

    from src.gimap.app.presentation.components.toast import LOOK_MS, MARGIN, show_toast, visible_toasts

    window = _window()
    try:
        components = window.components
        page = components.analyze_page
        window.resize(1280, 800)
        components.show_page("analyze")
        _settle()

        def drawn(widget) -> QRect:
            return QRect(widget.mapTo(window, QPoint(0, 0)), widget.size())

        error = show_toast(window, "Could not compare: The series have different x axes: qy (Å⁻¹), qz (Å⁻¹)", level="error")
        undo = show_toast(window, "Removed 2 series.", level="info", action=("Undo", lambda: None))
        page.status_label.setText("Batch 0/8: mapi_insitu_scan0042.nxs frame 1 of 8 …")
        page.progress_bar.show()
        page.cancel_button.show()  # a batch starts
        QTest.qWait(2 * LOOK_MS)
        _settle()
        assert visible_toasts(window) == [error, undo]
        for control in (page.progress_bar, page.cancel_button):
            assert not any(toast.geometry().intersects(drawn(control)) for toast in (error, undo)), control
        status = drawn(page.status_label)
        said_until = status.left() + page.status_label.fontMetrics().horizontalAdvance(page.status_label.text())
        assert min(toast.x() for toast in (error, undo)) > said_until  # the batch's own line is read whole
        assert error.x() > components.sidebar.width()
        page.progress_bar.hide()
        page.cancel_button.hide()  # … and ends
        page.status_label.setText("Exported 8 frames.")
        QTest.qWait(2 * LOOK_MS)
        _settle()
        assert undo.geometry().right() == window.width() - MARGIN - 1 and error.label.text() == error.text()
    finally:
        window.close()
        _settle()


def test_refine_and_show_in_fitting_go_through_the_pages_send_to_fitting() -> None:
    """The main window's own entries (Results ▸ Refine in Fitting, Fit details ▸ Show in Fitting) use the page's
    Send to Fitting that knows what to do with a read-only folder; an Analyze page without it, the old one."""
    from src.gimap.app.handover import HandoverMixin, short_folder

    calls = []
    new = type("Page", (), {"send_to_fitting": lambda self: calls.append("send_to_fitting"),
                            "fit_current": lambda self: calls.append("fit_current")})()
    old = type("Page", (), {"fit_current": lambda self: calls.append("fit_current")})()
    window = HandoverMixin()
    window.current_page_key = lambda: "analyze"
    for page in (new, old):
        window.analyze_page = page
        window._send_to_fitting()
        window._show_fit_solution({})
    assert calls == ["send_to_fitting", "send_to_fitting", "fit_current", "fit_current"]
    assert short_folder(Path("/data/2026/p03/sample_A/gimap_analysis")) == str(Path("…", "sample_A", "gimap_analysis"))
    assert short_folder(Path("run/out")) == str(Path("run/out"))


def test_every_sentence_of_the_hand_overs_has_its_chinese_with_the_same_values() -> None:
    import string

    from src.gimap.app import handover
    from src.gimap.app.presentation.home_page import TASK_LINKS
    from src.gimap.features.analyze.presentation.bindings import fit_handover
    from src.gimap.features.compare.application import sources, use_cases
    from src.gimap.features.compare.presentation import series_input
    from src.gimap.features.compare.presentation.views import compare_page_view
    from src.gimap.features.fitting.presentation import workspace

    said = [handover.SENT_TO_FITTING, fit_handover.SENT_ELSEWHERE, sources.SEVERAL_KINDS, sources.NO_SUCH_KIND,
            use_cases.TOO_FEW_IN_TABLE, series_input.KIND_TITLE, series_input.KIND_QUESTION, series_input.KIND_ITEM,
            series_input.KIND_ITEM_PLAIN, "Nothing added: no curve of {folder} was chosen.",
            series_input.FIT_INPUT_TITLE, series_input.FIT_INPUT_TITLE_PLAIN, series_input.OTHER_TITLE,
            sources.NO_TABLE, sources.NOT_NUMBERS, fit_handover.FOLDER_REFUSED, fit_handover.DATA_REFUSED,
            compare_page_view.FOLDER_TIP, workspace.NOTE_RUN, workspace.NOTE_RUN_POOR,
            *workspace.NOTE_TITLE.values(), *workspace.NOTE_TEXT.values(), workspace.GIWAXS_OPENED,
            workspace.LEGACY_CURVE_FAILED, workspace.LEGACY_MODEL_FAILED, "Close this note",
            *(tip for links in TASK_LINKS.values() for _request, _text, tip in links),
            *(text for links in TASK_LINKS.values() for _request, text, _tip in links)]

    def fields(text: str) -> list:
        return sorted(name for _text, name, _spec, _conversion in string.Formatter().parse(text) if name)

    for english in said:
        chinese = i18n.ZH.get(english)
        assert chinese and chinese != english, english
        assert fields(chinese) == fields(english), english
    assert sources.KIND_IN_NAME == "{name} · {kind}"  # a name: data, said as it is in either language
