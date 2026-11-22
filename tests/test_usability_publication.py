"""The figure and summary follow the selections made in Analyze."""

import numpy as np
from PIL import Image
from PyQt5.QtCore import Qt
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QFileDialog

from src.gimap.app.presentation.components import visible_toasts
from src.gimap.app.presentation.i18n import apply_language, trf
from src.gimap.features.analyze.presentation import qmap_figure
from src.gimap.features.assistant.application.pipeline_report import pipeline_markdown
from src.gimap.features.assistant.presentation.guided_report import report_html
from src.gimap.features.assistant.presentation.guided_text import saved_toast
from tests.test_analyze_workspace import _context, _done, _page
from tests.test_crystal_indexing import MAPI_CIF
from tests.test_open_frames_question import _answer
from tests.test_review_analyze_run_and_clear import _frames, _profiles


def test_publication_qmap_uses_visible_artists_dimensions_and_dpi(tmp_path, monkeypatch):
    page = _page(_context(_profiles()))
    page.resize(1280, 800)
    page.show()
    try:
        page.set_mode_choice("giwaxs")
        page.add_paths([str(path) for path in _frames(tmp_path, 1)])
        _done(page)
        page.set_view(1)
        _done(page)
        assert page.view_model.state.analysis.kind == "giwaxs" and page.view_combo.currentIndex() == 1
        cif = tmp_path / "candidate.cif"
        cif.write_text(MAPI_CIF, encoding="utf-8")
        page.indexing.add_cif(cif)
        page.indexing_changed()
        assert len(qmap_figure.displayed_layers(page)["arcs"]) > 0
        monkeypatch.setattr(qmap_figure, "figure_options", lambda *_: dict(
            width_cm=12.7, height_cm=10.16, font_pt=10, dpi=120, font_family="Arial"))
        path = tmp_path / "qmap.png"
        monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *_: (str(path), ""))
        page.canvas_save_figure_action.trigger()
        with Image.open(path) as fig:
            marked = np.asarray(fig).copy()
            assert fig.size == (600, 480) and abs(fig.info["dpi"][0] - 120) < 0.1
        path = tmp_path / "qmap.svg"
        page.save_image_action.trigger()
        svg = path.read_text(encoding="utf-8")
        assert "MAPbI3" in svg and "In-plane" in svg and "stroke-dasharray" in svg
        path = tmp_path / "qmap.pdf"
        page.save_image()
        assert path.read_bytes().startswith(b"%PDF")
        page.indexing.replace(0, visible=False)
        page.indexing_changed()
        page.detector_view.marks.set_visible("shapes", False)
        path = tmp_path / "plain.svg"
        page.save_image()
        plain = path.read_text(encoding="utf-8")
        assert "MAPbI3" not in plain and "In-plane" not in plain
        path = tmp_path / "plain.png"
        page.save_image()
        with Image.open(path) as fig:
            assert np.count_nonzero(np.any(marked != np.asarray(fig), axis=-1)) > 100
        page.indexing.replace(0, visible=True, orientation="plane")
        page.indexing_changed()
        assert qmap_figure.displayed_layers(page)["spots"]
        path = tmp_path / "fibre.svg"
        page.save_image()
        fibre = path.read_text(encoding="utf-8")
        assert "MAPbI3" in fibre and "PathCollection" in fibre
    finally:
        page.dispose()
        page.close()


def test_replace_can_reset_sample_without_discarding_phase_definitions(tmp_path):
    first, second = _frames(tmp_path, 2)
    page = _page(_context(_profiles()))
    try:
        page.show()
        page.import_paths([first])
        _done(page)
        cif = tmp_path / "other.cif"
        cif.write_text(MAPI_CIF, encoding="utf-8")
        page.indexing.add_cif(cif)
        page._use_incidence(0.37)
        _done(page)
        page._hidden_regions.add("in_plane")
        page.detector_view.marks.set_visible("shapes", False)
        assert "0.37°" in page._sample_choices() and "MAPbI3" in page._sample_choices()
        assert "Masks and cut regions" in page._sample_choices()
        _answer("Replace and reset", [])
        page.import_paths([second])
        _done(page)
        assert page.view_model.current_path == second and page.view_model.state.incidence_deg is None
        assert not page.indexing.active and len(page.indexing.phases) == 1
        assert not page._hidden_regions and page.detector_view.marks.is_visible("shapes")
    finally:
        page.dispose()
        page.close()


def test_summary_precedes_checks_and_preserves_uncertain_assignments():
    report = dict(ok=True, procedure="giwaxs", peaks=[
        {"phase": "Alpha", "hkl": "(100)", "q": 1.25},
        {"phase": "Beta", "hkl": "(010)", "q": 2.5, "ambiguous": True}, {"q": 3.2}],
        rings=[{"q": 1.25, "herman": 0.21, "texture": "tilted"}, {"q": 2.5, "herman": 0.55, "texture": "tilted"}],
        needs_attention=[{"item": "distance", "why": "Check the calibration."}])
    text = pipeline_markdown(report)
    summary, _ = text.split("## Needs attention", 1)
    assert "2 of 3 peaks" in summary and "1 ambiguous" in summary and "Check distance" in summary
    assert "Alpha — (100) · 1.25 Å⁻¹ · tilted (f = 0.21)" in summary
    assert "Beta — (010) · 2.5 Å⁻¹ · phase assignment uncertain" in summary
    assert "0.55" not in summary
    page = report_html(report, text, q_map=b"example")
    assert page.index("2 of 3 peaks") < page.index("<figure>") < page.index("Check the calibration.")


def test_open_cif_message_and_save_toast_switch_in_place(tmp_path, monkeypatch):
    page = _page(_context(_profiles()))
    try:
        page.show()
        cif = tmp_path / "candidate.cif"
        cif.write_text(MAPI_CIF, encoding="utf-8")
        monkeypatch.setattr(QFileDialog, "getOpenFileNames", lambda *_: ([str(cif)], ""))
        dialog = page.open_indexing()
        QTest.mouseClick(dialog.add_cif_button, Qt.LeftButton)
        msg = dialog.status_label.text()
        saved_toast(page, cif)
        notice = next(t for t in visible_toasts(page) if cif.name in t.text())
        apply_language("zh", [page, dialog])
        assert cif.name in dialog.status_label.text() and msg != dialog.status_label.text()
        assert notice.label.text() == trf("Saved {name}", name=cif.name) and notice.action_button.text() != "Open Folder"
        apply_language("en", [page, dialog])
        assert dialog.status_label.text() == msg and notice.text() == f"Saved {cif.name}"
    finally:
        apply_language("en")
        page.dispose()
        page.close()
